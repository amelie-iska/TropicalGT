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
    def __init__(self, dim: int, num_directions: int = 8) -> None:
        super().__init__()
        self.directions = nn.Parameter(torch.randn(num_directions, dim) * 0.02)

    def forward(self, z: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if z.shape[0] < 2:
            zero = z.sum() * 0.0
            return zero, {"graphcg_contrastive": zero, "graphcg_orthogonality": zero, "graphcg_sparsity": zero}
        dirs = F.normalize(self.directions, dim=-1)
        z_norm = F.normalize(z, dim=-1)
        shifted = F.normalize(z_norm[:, None, :] + 0.1 * dirs[None, :, :], dim=-1)
        sim = torch.einsum("brd,crd->bc", shifted, shifted) / dirs.shape[0]
        labels = torch.arange(z.shape[0], device=z.device)
        contrastive = F.cross_entropy(sim, labels)
        gram = dirs @ dirs.t()
        eye = torch.eye(gram.shape[0], device=z.device)
        orth = ((gram - eye) ** 2).mean()
        sparse = self.directions.abs().mean()
        offdiag = gram - eye
        norms = self.directions.norm(dim=-1)
        centered = dirs - dirs.mean(dim=0, keepdim=True)
        covariance = centered @ centered.t() / max(dirs.shape[-1] - 1, 1)
        eigvals = torch.linalg.eigvalsh((gram + gram.t()) * 0.5).real
        loss = contrastive + 0.05 * orth + 0.001 * sparse
        return loss, {
            "graphcg_contrastive": contrastive.detach(),
            "graphcg_orthogonality": orth.detach(),
            "graphcg_sparsity": sparse.detach(),
            "graphcg_direction_norm_mean": norms.detach().mean(),
            "graphcg_direction_norm_min": norms.detach().min(),
            "graphcg_direction_norm_max": norms.detach().max(),
            "graphcg_direction_norm_std": norms.detach().std(unbiased=False),
            "graphcg_direction_gram_offdiag_mean_abs": offdiag.detach().abs().mean(),
            "graphcg_direction_gram_offdiag_max_abs": offdiag.detach().abs().max(),
            "graphcg_direction_covariance_mean_abs": covariance.detach().abs().mean(),
            "graphcg_direction_gram_condition_proxy": (eigvals.detach().max() / eigvals.detach().abs().clamp_min(1e-8).min()),
        }
