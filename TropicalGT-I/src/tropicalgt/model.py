from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .attention import TropicalRingAttention, tropical_support_entropy
from .data import VOCAB_SIZE
from .records import GraphTokenBatch
from .losses import GFlowNetPolicy, GraphCGLoss


@dataclass
class TropicalGTConfig:
    vocab_size: int = VOCAB_SIZE
    graph_feature_dim: int = 48
    dim: int = 128
    hidden_dim: int = 128
    num_actions: int = 8
    gflownet_weight: float = 0.02
    graphcg_weight: float = 0.02
    margin_weight: float = 0.002
    entropy_weight: float = 0.001


class TropicalGTModel(nn.Module):
    def __init__(self, config: TropicalGTConfig) -> None:
        super().__init__()
        self.config = config
        self.byte_emb = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.graph_proj = nn.Sequential(nn.Linear(config.graph_feature_dim, config.dim), nn.GELU(), nn.LayerNorm(config.dim))
        self.graph_type_emb = nn.Embedding(4, config.dim)
        self.tropical = TropicalRingAttention(config.dim)
        self.gru = nn.GRU(config.dim, config.hidden_dim, batch_first=True)
        self.out = nn.Linear(config.hidden_dim, config.vocab_size)
        self.gfn = GFlowNetPolicy(config.dim, config.num_actions)
        self.graphcg = GraphCGLoss(config.dim, config.num_actions)

    def forward(self, input_ids: Tensor, graph_batch: GraphTokenBatch, target_ids: Tensor | None = None) -> dict[str, Tensor]:
        graph_batch = graph_batch.to(input_ids.device)
        g = self.graph_proj(graph_batch.token_features)
        type_ids = graph_batch.token_type_ids.clamp_min(0).clamp_max(3)
        g = g + self.graph_type_emb(type_ids)
        trop = self.tropical(g, graph_batch.attention_mask)
        masked = trop.context * graph_batch.attention_mask[..., None]
        denom = graph_batch.attention_mask.sum(dim=1).clamp_min(1).to(masked.dtype)[:, None]
        graph_state = masked.sum(dim=1) / denom
        x = self.byte_emb(input_ids) + graph_state[:, None, :]
        h, _ = self.gru(x)
        logits = self.out(h)
        valid_margin = trop.margin.masked_select(graph_batch.attention_mask)
        metrics: dict[str, Tensor] = {
            "support_entropy": tropical_support_entropy(trop.support).detach(),
            "margin_mean": valid_margin.mean().detach(),
            "graph_tokens_mean": graph_batch.graph_token_counts.float().mean().detach(),
            "edge_tokens_mean": graph_batch.edge_counts.float().mean().detach(),
        }
        loss = None
        if target_ids is not None:
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), ignore_index=0)
            reward = torch.exp(-nll.detach()).repeat(input_ids.shape[0]).clamp_min(1e-6)
            states = graph_state[:, None, :].repeat(1, 2, 1)
            actions = torch.zeros(input_ids.shape[0], 2, dtype=torch.long, device=input_ids.device)
            gfn_loss = self.gfn.trajectory_balance_loss(states, actions, reward)
            graphcg_loss, graphcg_metrics = self.graphcg(graph_state)
            margin_loss = -valid_margin.mean()
            entropy_loss = tropical_support_entropy(trop.support)
            loss = nll + self.config.gflownet_weight * gfn_loss + self.config.graphcg_weight * graphcg_loss + self.config.margin_weight * margin_loss + self.config.entropy_weight * entropy_loss
            metrics.update({"nll": nll.detach(), "gflownet_tb": gfn_loss.detach(), "graphcg_loss": graphcg_loss.detach(), "tropical_margin_loss": margin_loss.detach(), "tropical_entropy_loss": entropy_loss.detach(), **graphcg_metrics})
        return {"logits": logits, "loss": loss if loss is not None else torch.zeros((), device=input_ids.device), "graph_state": graph_state, "support": trop.support, "margin": trop.margin, **metrics}
