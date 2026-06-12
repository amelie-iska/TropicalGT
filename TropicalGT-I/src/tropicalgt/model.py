from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .attention import TropicalRingAttention, soft_tropical_support_entropy, tropical_support_entropy
from .data import VOCAB_SIZE
from .records import GraphTokenBatch
from .losses import GFlowNetPolicy, GraphCGLoss
from .memory import AnalogicalMemoryHead


@dataclass
class TropicalGTConfig:
    vocab_size: int = VOCAB_SIZE
    graph_feature_dim: int = 48
    dim: int = 128
    hidden_dim: int = 128
    num_actions: int = 8
    graphcg_num_directions: int | None = None
    graphcg_active_directions: int = 64
    gflownet_weight: float = 0.02
    graphcg_weight: float = 0.02
    margin_weight: float = 0.002
    entropy_weight: float = 0.001
    certificate_weight: float = 0.001
    wall_margin_threshold: float = 1.0e-3
    memory_dim: int = 32
    graph_tropical_block_size: int = 32
    use_sequence_tropical: bool = True
    sequence_tropical_weight: float = 0.125
    sequence_tropical_max_tokens: int = 32
    sequence_tropical_block_size: int = 16


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
        graphcg_num_directions = config.graphcg_num_directions or config.dim
        self.graphcg = GraphCGLoss(
            config.dim,
            num_directions=graphcg_num_directions,
            active_directions=config.graphcg_active_directions,
        )
        self.memory = AnalogicalMemoryHead(config.dim, config.memory_dim)

    def forward(self, input_ids: Tensor, graph_batch: GraphTokenBatch, target_ids: Tensor | None = None) -> dict[str, Tensor]:
        graph_batch = graph_batch.to(input_ids.device)
        g = self.graph_proj(graph_batch.token_features)
        type_ids = graph_batch.token_type_ids.clamp_min(0).clamp_max(3)
        g = g + self.graph_type_emb(type_ids)
        graph_token_embeddings = g
        if self.config.graph_tropical_block_size > 0 and g.shape[1] > self.config.graph_tropical_block_size:
            trop = self.tropical.blockwise(g, graph_batch.attention_mask, block_size=self.config.graph_tropical_block_size)
        else:
            trop = self.tropical(g, graph_batch.attention_mask)
        masked_support_scores = trop.scores.float().masked_fill(~graph_batch.attention_mask[:, None, :], -torch.inf)
        graph_token_support_probabilities = torch.softmax(masked_support_scores, dim=-1)
        graph_token_support_probabilities = torch.nan_to_num(graph_token_support_probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        masked = trop.context * graph_batch.attention_mask[..., None]
        denom = graph_batch.attention_mask.sum(dim=1).clamp_min(1).to(masked.dtype)[:, None]
        graph_state = masked.sum(dim=1) / denom
        x = self.byte_emb(input_ids) + graph_state[:, None, :]
        sequence_tropical_metrics: dict[str, Tensor] = {}
        if self.config.use_sequence_tropical and input_ids.shape[1] > 1:
            seq_mask = input_ids.ne(0)
            pooled_x, pooled_mask, stride = _pool_sequence_tokens(x, seq_mask, self.config.sequence_tropical_max_tokens)
            seq_trop = self.tropical.blockwise(pooled_x, pooled_mask, block_size=self.config.sequence_tropical_block_size)
            expanded_context = _expand_sequence_context(seq_trop.context, input_ids.shape[1], stride)
            x = x + float(self.config.sequence_tropical_weight) * expanded_context
            valid_seq_margin = seq_trop.margin.masked_select(pooled_mask)
            sequence_tropical_metrics = {
                "sequence_tropical_tokens_mean": pooled_mask.float().sum(dim=1).mean().detach(),
                "sequence_tropical_stride": torch.tensor(float(stride), device=input_ids.device),
                "sequence_tropical_margin_mean": _safe_mean(valid_seq_margin).detach(),
                "sequence_tropical_margin_min": _safe_min(valid_seq_margin).detach(),
                "sequence_tropical_support_entropy": tropical_support_entropy(seq_trop.support, pooled_mask).detach(),
                "sequence_tropical_weight": torch.tensor(float(self.config.sequence_tropical_weight), device=input_ids.device),
            }
        h, _ = self.gru(x)
        logits = self.out(h)
        valid_margin = trop.margin.masked_select(graph_batch.attention_mask)
        support_entropy = tropical_support_entropy(trop.support, graph_batch.attention_mask)
        soft_entropy = soft_tropical_support_entropy(trop.scores, graph_batch.attention_mask, graph_batch.attention_mask)
        certificate_loss, certificate_metrics = tropical_certificate_objective(trop.scores, trop.support, graph_batch)
        wall_hit_rate = tropical_wall_hit_rate(trop.margin, graph_batch.attention_mask, self.config.wall_margin_threshold)
        boundary_hit_rate = tropical_support_boundary_hit_rate(trop.support, graph_batch.attention_mask, graph_batch.graph_token_counts)
        node_edge_ratio = graph_batch.node_counts.float().sum() / graph_batch.edge_counts.float().sum().clamp_min(1.0)
        self_support_rate = tropical_self_support_rate(trop.support, graph_batch.attention_mask)
        invalid_support_rate = tropical_invalid_support_rate(trop.support, graph_batch.attention_mask, graph_batch.graph_token_counts)
        metrics: dict[str, Tensor] = {
            "support_entropy": support_entropy.detach(),
            "support_soft_entropy": soft_entropy.detach(),
            "support_unique_frac": tropical_support_unique_fraction(trop.support, graph_batch.attention_mask).detach(),
            "support_transition_rate": tropical_support_transition_rate(trop.support, graph_batch.attention_mask).detach(),
            "self_support_rate": self_support_rate.detach(),
            "invalid_support_rate": invalid_support_rate.detach(),
            "margin_mean": _safe_mean(valid_margin).detach(),
            "margin_min": _safe_min(valid_margin).detach(),
            "margin_p05": _safe_quantile(valid_margin, 0.05).detach(),
            "positive_margin_rate": (valid_margin.gt(0).float().mean() if valid_margin.numel() else valid_margin.sum() * 0.0).detach(),
            "graph_tokens_mean": graph_batch.graph_token_counts.float().mean().detach(),
            "node_tokens_mean": graph_batch.node_counts.float().mean().detach(),
            "edge_tokens_mean": graph_batch.edge_counts.float().mean().detach(),
            "node_edge_ratio": node_edge_ratio.detach(),
            "wall_hit_rate": wall_hit_rate.detach(),
            "wall_margin_threshold": torch.tensor(self.config.wall_margin_threshold, device=input_ids.device),
            "support_boundary_hit_rate": boundary_hit_rate.detach(),
            "analogical_memory_query_norm": self.memory(graph_state).detach().norm(dim=-1).mean(),
            **sequence_tropical_metrics,
            **certificate_metrics,
        }
        loss = None
        if target_ids is not None:
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), ignore_index=0)
            reward = torch.exp(-nll.detach()).repeat(input_ids.shape[0]).clamp_min(1e-6)
            states = graph_state[:, None, :].repeat(1, 2, 1)
            actions = torch.zeros(input_ids.shape[0], 2, dtype=torch.long, device=input_ids.device)
            gfn_loss, gfn_metrics = self.gfn.trajectory_balance_loss(states, actions, reward, return_metrics=True)
            graphcg_loss, graphcg_metrics = self.graphcg(graph_state)
            margin_loss = -_safe_mean(valid_margin)
            entropy_loss = soft_entropy
            gfn_weighted = self.config.gflownet_weight * gfn_loss
            graphcg_weighted = self.config.graphcg_weight * graphcg_loss
            margin_weighted = self.config.margin_weight * margin_loss
            entropy_weighted = self.config.entropy_weight * entropy_loss
            certificate_weighted = self.config.certificate_weight * certificate_loss
            regularizer_total = gfn_weighted + graphcg_weighted + margin_weighted + entropy_weighted + certificate_weighted
            loss = (
                nll
                + regularizer_total
            )
            metrics.update(
                {
                    "nll": nll.detach(),
                    "gflownet_tb": gfn_loss.detach(),
                    "graphcg_loss": graphcg_loss.detach(),
                    "certificate_loss": certificate_loss.detach(),
                    "tropical_margin_loss": margin_loss.detach(),
                    "tropical_entropy_loss": entropy_loss.detach(),
                    "loss_gflownet_weighted": gfn_weighted.detach(),
                    "loss_graphcg_weighted": graphcg_weighted.detach(),
                    "loss_margin_weighted": margin_weighted.detach(),
                    "loss_entropy_weighted": entropy_weighted.detach(),
                    "loss_certificate_weighted": certificate_weighted.detach(),
                    "loss_regularizer_total": regularizer_total.detach(),
                    "loss_regularizer_ratio": (regularizer_total.detach().abs() / nll.detach().abs().clamp_min(1e-8)),
                    **gfn_metrics,
                    **graphcg_metrics,
                }
            )
        return {
            "logits": logits,
            "loss": loss if loss is not None else torch.zeros((), device=input_ids.device),
            "graph_state": graph_state,
            "graph_token_embeddings": graph_token_embeddings,
            "graph_token_context_embeddings": trop.context,
            "graph_token_support_probabilities": graph_token_support_probabilities,
            "support": trop.support,
            "margin": trop.margin,
            **metrics,
        }


def tropical_certificate_objective(scores: Tensor, support: Tensor, graph_batch: GraphTokenBatch) -> tuple[Tensor, dict[str, Tensor]]:
    targets = tropical_certificate_targets(graph_batch)
    valid = graph_batch.attention_mask
    if valid.sum() == 0:
        zero = scores.sum() * 0.0
        return zero, {
            "certificate_agreement": zero.detach(),
            "certificate_coverage": zero.detach(),
            "certificate_allowed_mass_mean": zero.detach(),
            "certificate_allowed_mass_min": zero.detach(),
            "certificate_edge_agreement": zero.detach(),
            "certificate_node_agreement": zero.detach(),
            "certificate_graph_agreement": zero.detach(),
            "certificate_edge_loss": zero.detach(),
            "certificate_node_loss": zero.detach(),
            "certificate_graph_loss": zero.detach(),
            "certificate_disallowed_support_rate": zero.detach(),
            "certificate_graph_support_rate": zero.detach(),
            "certificate_node_graph_support_rate": zero.detach(),
            "certificate_edge_graph_support_rate": zero.detach(),
        }
    log_probs = F.log_softmax(scores, dim=-1)
    target_log_probs = torch.logsumexp(log_probs.masked_fill(~targets, -torch.inf), dim=-1)
    target_log_probs = torch.nan_to_num(target_log_probs, neginf=-80.0, posinf=0.0)
    per_token_loss = -target_log_probs
    allowed_mass = torch.exp(target_log_probs).clamp_min(0.0).clamp_max(1.0)
    loss = per_token_loss.masked_select(valid).mean()
    support_allowed = targets.gather(-1, support.unsqueeze(-1)).squeeze(-1) & valid
    type_ids = graph_batch.token_type_ids
    edge_mask = valid & type_ids.eq(1)
    node_mask = valid & type_ids.eq(0)
    graph_mask = valid & type_ids.eq(2)
    support_clamped = support.clamp_min(0).clamp_max(max(type_ids.shape[1] - 1, 0))
    support_type_ids = type_ids.gather(1, support_clamped)
    graph_support = valid & support_type_ids.eq(2)
    disallowed_support = valid & ~support_allowed
    valid_allowed_mass = allowed_mass.masked_select(valid)
    metrics = {
        "certificate_agreement": _rate(support_allowed, valid).detach(),
        "certificate_coverage": _rate(targets.any(dim=-1) & valid, valid).detach(),
        "certificate_allowed_mass_mean": _safe_mean(valid_allowed_mass).detach(),
        "certificate_allowed_mass_min": _safe_min(valid_allowed_mass).detach(),
        "certificate_edge_agreement": _rate(support_allowed, edge_mask).detach(),
        "certificate_node_agreement": _rate(support_allowed, node_mask).detach(),
        "certificate_graph_agreement": _rate(support_allowed, graph_mask).detach(),
        "certificate_edge_loss": _safe_mean(per_token_loss.masked_select(edge_mask)).detach(),
        "certificate_node_loss": _safe_mean(per_token_loss.masked_select(node_mask)).detach(),
        "certificate_graph_loss": _safe_mean(per_token_loss.masked_select(graph_mask)).detach(),
        "certificate_disallowed_support_rate": _rate(disallowed_support, valid).detach(),
        "certificate_graph_support_rate": _rate(graph_support, valid).detach(),
        "certificate_node_graph_support_rate": _rate(graph_support & node_mask, node_mask).detach(),
        "certificate_edge_graph_support_rate": _rate(graph_support & edge_mask, edge_mask).detach(),
    }
    return loss, metrics


def tropical_certificate_targets(graph_batch: GraphTokenBatch) -> Tensor:
    type_ids = graph_batch.token_type_ids
    endpoints = graph_batch.endpoint_ids
    valid = graph_batch.attention_mask
    batch, tokens = type_ids.shape
    targets = torch.zeros(batch, tokens, tokens, dtype=torch.bool, device=type_ids.device)
    for b in range(batch):
        offset = 1 if bool(valid[b, 0]) and int(type_ids[b, 0].item()) == 2 else 0
        for i in range(tokens):
            if not bool(valid[b, i]):
                continue
            targets[b, i, i] = True
            if int(type_ids[b, i].item()) == 1:
                for endpoint in endpoints[b, i].tolist():
                    target = int(endpoint) + offset
                    if 0 <= target < tokens and bool(valid[b, target]):
                        targets[b, i, target] = True
    return targets


def tropical_wall_hit_rate(margin: Tensor, mask: Tensor, threshold: float) -> Tensor:
    if mask.sum() == 0:
        return margin.sum() * 0.0
    hits = margin.le(float(threshold)) & mask
    return hits.float().sum() / mask.float().sum().clamp_min(1.0)


def tropical_support_boundary_hit_rate(support: Tensor, mask: Tensor, graph_token_counts: Tensor) -> Tensor:
    if mask.sum() == 0:
        return support.sum() * 0.0
    last = (graph_token_counts - 1).clamp_min(0)[:, None].expand_as(support)
    wall = (support.eq(0) | support.eq(last)) & mask
    return wall.float().sum() / mask.float().sum().clamp_min(1.0)


def tropical_support_transition_rate(support: Tensor, mask: Tensor) -> Tensor:
    if support.shape[1] < 2:
        return support.sum() * 0.0
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    if valid_pairs.sum() == 0:
        return support.sum() * 0.0
    switches = support[:, 1:].ne(support[:, :-1]) & valid_pairs
    return switches.float().sum() / valid_pairs.float().sum().clamp_min(1.0)


def tropical_self_support_rate(support: Tensor, mask: Tensor) -> Tensor:
    indices = torch.arange(support.shape[1], device=support.device)[None, :].expand_as(support)
    hits = support.eq(indices) & mask
    return hits.float().sum() / mask.float().sum().clamp_min(1.0)


def tropical_invalid_support_rate(support: Tensor, mask: Tensor, graph_token_counts: Tensor) -> Tensor:
    invalid = support.ge(graph_token_counts[:, None]) & mask
    return invalid.float().sum() / mask.float().sum().clamp_min(1.0)


def tropical_support_unique_fraction(support: Tensor, mask: Tensor) -> Tensor:
    fractions = []
    for row, row_mask in zip(support, mask):
        valid = row.masked_select(row_mask)
        if valid.numel():
            fractions.append(valid.unique().numel() / float(valid.numel()))
    if not fractions:
        return support.sum() * 0.0
    return torch.tensor(fractions, dtype=torch.float32, device=support.device).mean()


def _pool_sequence_tokens(x: Tensor, mask: Tensor, max_tokens: int) -> tuple[Tensor, Tensor, int]:
    max_tokens = max(int(max_tokens), 1)
    batch, seq_len, dim = x.shape
    stride = max(1, (seq_len + max_tokens - 1) // max_tokens)
    chunks = (seq_len + stride - 1) // stride
    pad = chunks * stride - seq_len
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad), value=False)
    x_chunks = x.reshape(batch, chunks, stride, dim)
    mask_chunks = mask.reshape(batch, chunks, stride)
    denom = mask_chunks.float().sum(dim=2).clamp_min(1.0)
    pooled = (x_chunks * mask_chunks[..., None]).sum(dim=2) / denom[..., None]
    pooled_mask = mask_chunks.any(dim=2)
    return pooled, pooled_mask, stride


def _expand_sequence_context(context: Tensor, seq_len: int, stride: int) -> Tensor:
    batch, chunks, dim = context.shape
    expanded = context[:, :, None, :].expand(batch, chunks, stride, dim).reshape(batch, chunks * stride, dim)
    return expanded[:, :seq_len, :]


def _rate(mask: Tensor, denom_mask: Tensor) -> Tensor:
    denom = denom_mask.float().sum()
    numerator = (mask & denom_mask).float().sum()
    return numerator / denom.clamp_min(1.0)


def _safe_mean(values: Tensor) -> Tensor:
    return values.mean() if values.numel() else values.sum() * 0.0


def _safe_min(values: Tensor) -> Tensor:
    return values.min() if values.numel() else values.sum() * 0.0


def _safe_quantile(values: Tensor, q: float) -> Tensor:
    return torch.quantile(values.float(), q) if values.numel() else values.sum() * 0.0
