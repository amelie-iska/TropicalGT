from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GFlowNetPolicy(nn.Module):
    def __init__(self, dim: int, num_actions: int = 8) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.policy = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_actions))
        self.log_z = nn.Parameter(torch.zeros(()))

    def forward(self, state: Tensor) -> Tensor:
        return self.policy(state)

    def trajectory_balance_loss(self, states: Tensor, actions: Tensor, rewards: Tensor, return_metrics: bool = False):
        logits = self(states)
        log_probs = F.log_softmax(logits, dim=-1)
        logp = log_probs.gather(-1, actions[..., None]).squeeze(-1).sum(dim=-1)
        log_reward = rewards.clamp_min(1e-8).log()
        residual = self.log_z + logp - log_reward
        loss = (residual ** 2).mean()
        if not return_metrics:
            return loss
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        first_action = actions[..., 0]
        top_actions = logits[..., 0, :].argmax(dim=-1) if logits.ndim == 3 else logits.argmax(dim=-1)
        metrics = {
            "gflownet_logp_mean": logp.detach().mean(),
            "gflownet_log_reward_mean": log_reward.detach().mean(),
            "gflownet_tb_residual_abs_mean": residual.detach().abs().mean(),
            "gflownet_log_z": self.log_z.detach(),
            "gflownet_action_entropy_mean": entropy.detach().mean(),
            "gflownet_reward_mean": rewards.detach().mean(),
            "gflownet_reward_std": rewards.detach().std(unbiased=False),
            "gflownet_top_action_match_rate": (top_actions.detach() == first_action.detach()).float().mean(),
        }
        return loss, metrics


class GraphCGLoss(nn.Module):
    def __init__(
        self,
        dim: int,
        num_directions: int | None = None,
        full_rank_margin: float = 0.05,
        active_directions: int | None = None,
        exact_rank_max_directions: int = 512,
    ) -> None:
        super().__init__()
        num_directions = dim if num_directions is None else int(num_directions)
        directions = torch.empty(num_directions, dim)
        if num_directions > 0 and dim > 0:
            nn.init.orthogonal_(directions)
        else:
            directions.zero_()
        self.directions = nn.Parameter(directions)
        self.full_rank_margin = full_rank_margin
        self.active_directions = int(active_directions or min(num_directions, 64))
        self.exact_rank_max_directions = int(exact_rank_max_directions)

    def effective_directions(self, detach: bool = False) -> Tensor:
        """Return the full-rank steering basis used by GraphCG projections.

        When the requested number of directions is at most the ambient embedding
        dimension, the effective basis is an orthonormal Stiefel projection of
        the learnable directions. The raw directions still receive a rank-collapse
        penalty, but downstream steering cannot silently become rank-deficient.
        If more directions than dimensions are requested, full row rank is
        impossible; in that edge case we return normalized directions and report
        the attainable rank target separately.
        """

        raw = self.directions.detach() if detach else self.directions
        if raw.numel() == 0:
            return raw
        if raw.shape[0] <= raw.shape[1] and raw.shape[0] <= self.exact_rank_max_directions:
            q, _ = torch.linalg.qr(raw.transpose(0, 1), mode="reduced")
            dirs = q.transpose(0, 1)
            signs = torch.sign((dirs * raw).sum(dim=-1, keepdim=True))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            return dirs * signs
        return F.normalize(raw, dim=-1)

    def forward(self, z: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if z.shape[0] < 2:
            zero = z.sum() * 0.0
            return zero, {
                "graphcg_contrastive": zero,
                "graphcg_orthogonality": zero,
                "graphcg_sparsity": zero,
                "graphcg_full_rank": zero,
                "graphcg_direction_effective_rank": zero,
                "graphcg_effective_rank": zero,
                "graphcg_direction_numerical_rank": zero,
                "graphcg_numerical_rank": zero,
                "graphcg_direction_singular_min": zero,
                "graphcg_min_singular_value": zero,
                "graphcg_direction_singular_max": zero,
                "graphcg_max_singular_value": zero,
                "graphcg_full_rank_penalty": zero,
                "graphcg_raw_full_rank_penalty": zero,
                "graphcg_rank_target": zero,
                "graphcg_num_directions": zero,
                "graphcg_embedding_dim": zero,
                "graphcg_active_directions": zero,
                "graphcg_active_rank_target": zero,
                "graphcg_raw_effective_rank": zero,
                "graphcg_raw_numerical_rank": zero,
                "graphcg_full_rank_possible": zero,
                "graphcg_raw_singular_min": zero,
                "graphcg_raw_singular_max": zero,
                "graphcg_singular_value_floor": zero,
            }
        raw_dirs = F.normalize(self.directions, dim=-1)
        dirs = self.effective_directions()
        active_idx = self._active_indices(z.device)
        active_raw_dirs = raw_dirs.index_select(0, active_idx)
        active_dirs = dirs.index_select(0, active_idx)
        z_norm = F.normalize(z, dim=-1)
        shifted = F.normalize(z_norm[:, None, :] + 0.1 * active_dirs[None, :, :], dim=-1)
        sim = torch.einsum("brd,crd->bc", shifted, shifted) / active_dirs.shape[0]
        labels = torch.arange(z.shape[0], device=z.device)
        contrastive = F.cross_entropy(sim, labels)
        gram = active_dirs @ active_dirs.t()
        eye = torch.eye(gram.shape[0], device=z.device)
        orth = ((gram - eye) ** 2).mean()
        sparse = self.directions.abs().mean()
        offdiag = gram - eye
        norms = self.directions.norm(dim=-1)
        centered = active_dirs - active_dirs.mean(dim=0, keepdim=True)
        covariance = centered @ centered.t() / max(active_dirs.shape[-1] - 1, 1)
        singular_values = torch.linalg.svdvals(active_dirs)
        raw_singular_values = torch.linalg.svdvals(active_raw_dirs)
        rank_target = min(dirs.shape[0], dirs.shape[1])
        active_rank_target = min(active_dirs.shape[0], active_dirs.shape[1])
        active_singular_values = singular_values[:active_rank_target]
        raw_active_singular_values = raw_singular_values[:active_rank_target]
        raw_full_rank = F.relu(self.full_rank_margin - raw_active_singular_values).pow(2).mean()
        full_rank = F.relu(self.full_rank_margin - active_singular_values).pow(2).mean()
        spectral_mass = active_singular_values.clamp_min(1e-8)
        spectral_probs = spectral_mass / spectral_mass.sum().clamp_min(1e-8)
        effective_rank = torch.exp(-(spectral_probs * spectral_probs.log()).sum())
        raw_spectral_mass = raw_active_singular_values.clamp_min(1e-8)
        raw_spectral_probs = raw_spectral_mass / raw_spectral_mass.sum().clamp_min(1e-8)
        raw_effective_rank = torch.exp(-(raw_spectral_probs * raw_spectral_probs.log()).sum())
        numerical_rank = active_singular_values.gt(self.full_rank_margin).float().sum()
        raw_numerical_rank = raw_active_singular_values.gt(self.full_rank_margin).float().sum()
        condition_proxy = active_singular_values.max() / active_singular_values.abs().clamp_min(1e-8).min()
        eigvals = torch.linalg.eigvalsh((gram + gram.t()) * 0.5).real
        loss = contrastive + 0.05 * orth + 0.05 * raw_full_rank + 0.001 * sparse
        return loss, {
            "graphcg_contrastive": contrastive.detach(),
            "graphcg_orthogonality": orth.detach(),
            "graphcg_sparsity": sparse.detach(),
            "graphcg_full_rank": full_rank.detach(),
            "graphcg_full_rank_penalty": raw_full_rank.detach(),
            "graphcg_raw_full_rank_penalty": raw_full_rank.detach(),
            "graphcg_direction_rank_target": torch.tensor(float(rank_target), device=z.device),
            "graphcg_rank_target": torch.tensor(float(rank_target), device=z.device),
            "graphcg_num_directions": torch.tensor(float(dirs.shape[0]), device=z.device),
            "graphcg_embedding_dim": torch.tensor(float(dirs.shape[1]), device=z.device),
            "graphcg_active_directions": torch.tensor(float(active_dirs.shape[0]), device=z.device),
            "graphcg_active_rank_target": torch.tensor(float(active_rank_target), device=z.device),
            "graphcg_direction_effective_rank": effective_rank.detach(),
            "graphcg_effective_rank": effective_rank.detach(),
            "graphcg_direction_numerical_rank": numerical_rank.detach(),
            "graphcg_numerical_rank": numerical_rank.detach(),
            "graphcg_raw_effective_rank": raw_effective_rank.detach(),
            "graphcg_raw_numerical_rank": raw_numerical_rank.detach(),
            "graphcg_full_rank_possible": torch.tensor(float(dirs.shape[0] <= dirs.shape[1]), device=z.device),
            "graphcg_direction_singular_min": active_singular_values.detach().min(),
            "graphcg_min_singular_value": active_singular_values.detach().min(),
            "graphcg_direction_singular_max": active_singular_values.detach().max(),
            "graphcg_max_singular_value": active_singular_values.detach().max(),
            "graphcg_raw_singular_min": raw_active_singular_values.detach().min(),
            "graphcg_raw_singular_max": raw_active_singular_values.detach().max(),
            "graphcg_singular_value_floor": torch.tensor(float(self.full_rank_margin), device=z.device),
            "graphcg_direction_norm_mean": norms.detach().mean(),
            "graphcg_direction_norm_min": norms.detach().min(),
            "graphcg_direction_norm_max": norms.detach().max(),
            "graphcg_direction_norm_std": norms.detach().std(unbiased=False),
            "graphcg_direction_gram_offdiag_mean_abs": offdiag.detach().abs().mean(),
            "graphcg_direction_gram_offdiag_max_abs": offdiag.detach().abs().max(),
            "graphcg_direction_covariance_mean_abs": covariance.detach().abs().mean(),
            "graphcg_direction_gram_condition_proxy": (eigvals.detach().max() / eigvals.detach().abs().clamp_min(1e-8).min()),
            "graphcg_direction_svd_condition_proxy": condition_proxy.detach(),
        }

    def _active_indices(self, device: torch.device) -> Tensor:
        num_directions = self.directions.shape[0]
        active = max(1, min(int(self.active_directions), num_directions))
        if active >= num_directions:
            return torch.arange(num_directions, device=device)
        stride = max(num_directions // active, 1)
        return (torch.arange(active, device=device) * stride).remainder(num_directions)
