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

    def trajectory_balance_loss(self, states: Tensor, actions: Tensor, rewards: Tensor) -> Tensor:
        logits = self(states)
        logp = F.log_softmax(logits, dim=-1).gather(-1, actions[..., None]).squeeze(-1).sum(dim=-1)
        log_reward = rewards.clamp_min(1e-8).log()
        return ((self.log_z + logp - log_reward) ** 2).mean()


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
        loss = contrastive + 0.05 * orth + 0.001 * sparse
        return loss, {"graphcg_contrastive": contrastive.detach(), "graphcg_orthogonality": orth.detach(), "graphcg_sparsity": sparse.detach()}
