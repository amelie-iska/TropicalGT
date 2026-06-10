from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class TropicalAttentionOutput:
    context: Tensor
    support: Tensor
    margin: Tensor
    scores: Tensor


class TropicalRingAttention(nn.Module):
    """Max-plus/min-plus graph-token attention with support and margin diagnostics."""

    def __init__(self, dim: int, mode: str = "maxplus") -> None:
        super().__init__()
        if mode not in {"maxplus", "minplus"}:
            raise ValueError("mode must be 'maxplus' or 'minplus'")
        self.mode = mode
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> TropicalAttentionOutput:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = tropical_hilbert_scores(q, k)
        if self.mode == "minplus":
            scores = -scores
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
        support = scores.argmax(dim=-1)
        top2 = torch.topk(scores, k=min(2, scores.shape[-1]), dim=-1).values
        margin = top2[..., 0] - (top2[..., 1] if top2.shape[-1] > 1 else top2[..., 0])
        context = torch.max(scores.unsqueeze(-1) + v[:, None, :, :], dim=2).values
        context = torch.nan_to_num(context, neginf=0.0, posinf=0.0)
        return TropicalAttentionOutput(self.out_proj(context), support, margin, scores)

    def blockwise(self, x: Tensor, mask: Tensor | None = None, block_size: int = 128) -> TropicalAttentionOutput:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        best_context = None
        all_scores = []
        for start in range(0, x.shape[1], block_size):
            stop = min(start + block_size, x.shape[1])
            scores = tropical_hilbert_scores(q, k[:, start:stop])
            if self.mode == "minplus":
                scores = -scores
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None, start:stop], -torch.inf)
            ctx = torch.max(scores.unsqueeze(-1) + v[:, None, start:stop, :], dim=2).values
            best_context = ctx if best_context is None else torch.maximum(best_context, ctx)
            all_scores.append(scores)
        scores_full = torch.cat(all_scores, dim=-1)
        support = scores_full.argmax(dim=-1)
        top2 = torch.topk(scores_full, k=min(2, scores_full.shape[-1]), dim=-1).values
        margin = top2[..., 0] - (top2[..., 1] if top2.shape[-1] > 1 else top2[..., 0])
        return TropicalAttentionOutput(self.out_proj(torch.nan_to_num(best_context, neginf=0.0, posinf=0.0)), support, margin, scores_full)


def tropical_hilbert_scores(q: Tensor, k: Tensor) -> Tensor:
    delta = q[:, :, None, :] - k[:, None, :, :]
    spread = delta.amax(dim=-1) - delta.amin(dim=-1)
    return -spread / math.sqrt(q.shape[-1])


def tropical_support_entropy(support: Tensor, mask: Tensor | None = None) -> Tensor:
    flat = support.reshape(-1) if mask is None else support[mask]
    if flat.numel() == 0:
        return torch.zeros((), device=support.device)
    counts = torch.bincount(flat.clamp_min(0), minlength=int(flat.max().item()) + 1).float()
    p = counts[counts > 0] / counts.sum()
    return -(p * p.log()).sum()
