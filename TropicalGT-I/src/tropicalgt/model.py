from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    enable_chart_bundle_auxiliary: bool = False
    bundle_num_charts: int = 4
    toric_num_active_rows: int = 16
    bundle_transport_weight: float = 0.0
    bundle_cocycle_weight: float = 0.0
    bundle_flat_rank_weight: float = 0.0
    toric_normal_fan_weight: float = 0.0
    graphcg_toric_cell_agreement_weight: float = 0.0
    chart_bpb_consistency_weight: float = 0.0
    bundle_atom_stability_weight: float = 0.0


class ChartBundleToricHead(nn.Module):
    """Telemetry/loss head for zero-default tropical chart-bundle diagnostics.

    The head reads model graph states and tropical support probabilities. It never
    feeds back into token logits; the main model only adds weighted auxiliary
    losses when the corresponding config coefficients are nonzero.
    """

    def __init__(self, dim: int, num_charts: int = 4, toric_rows: int = 16) -> None:
        super().__init__()
        self.num_charts = max(1, int(num_charts))
        self.toric_rows = max(1, int(toric_rows))
        self.chart_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.num_charts))
        self.transport_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.num_charts * self.num_charts))
        self.toric_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.toric_rows))

    def transport_metadata(self) -> dict[str, Any]:
        chart_ids = [f"chart_{idx:02d}" for idx in range(self.num_charts)]
        pairs = [
            {
                "id": f"{chart_ids[i]}__to__{chart_ids[j]}",
                "source_chart": chart_ids[i],
                "target_chart": chart_ids[j],
                "transport_logit_index": [i, j],
            }
            for i in range(self.num_charts)
            for j in range(self.num_charts)
            if i != j
        ]
        triples = [
            {
                "id": f"{chart_ids[i]}__to__{chart_ids[j]}__to__{chart_ids[k]}",
                "source_chart": chart_ids[i],
                "middle_chart": chart_ids[j],
                "target_chart": chart_ids[k],
                "pair_ids": [f"{chart_ids[i]}__to__{chart_ids[j]}", f"{chart_ids[j]}__to__{chart_ids[k]}", f"{chart_ids[i]}__to__{chart_ids[k]}"],
            }
            for i in range(self.num_charts)
            for j in range(self.num_charts)
            for k in range(self.num_charts)
            if i != j and j != k and i != k
        ]
        return {
            "available": True,
            "source": "ChartBundleToricHead.transport_metadata",
            "chart_ids": chart_ids,
            "overlap_pairs": pairs,
            "overlap_triples": triples,
            "overlap_pair_count": len(pairs),
            "overlap_triple_count": len(triples),
            "directed_overlap_policy": "ordered chart pairs/triples matching transport T_ab and cocycle T_bc T_ab = T_ac",
        }

    def forward(
        self,
        graph_state: Tensor,
        graph_token_support_probabilities: Tensor | None = None,
        graph_token_mask: Tensor | None = None,
        graphcg_projection: Tensor | None = None,
        per_record_bpb: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        zero = graph_state.sum() * 0.0
        batch = graph_state.shape[0]
        chart_logits = self.chart_head(graph_state)
        chart_probs = torch.softmax(chart_logits, dim=-1)
        chart_ids = chart_probs.argmax(dim=-1)
        monomial_projection = F.one_hot(chart_ids, num_classes=self.num_charts).to(graph_state.dtype)
        monomial_one_hotness = monomial_projection.max(dim=-1).values.mean() if batch else zero
        chart_confidence = chart_probs.max(dim=-1).values

        transport = self.transport_head(graph_state).reshape(batch, self.num_charts, self.num_charts)
        eye = torch.eye(self.num_charts, dtype=torch.bool, device=graph_state.device)
        offdiag = transport[:, ~eye] if self.num_charts > 1 else transport.reshape(batch, -1)
        transport_l1 = offdiag.abs().mean() if offdiag.numel() else zero
        if self.num_charts >= 3:
            cocycles = []
            for i in range(self.num_charts):
                for j in range(self.num_charts):
                    for k in range(self.num_charts):
                        if i == j or j == k or i == k:
                            continue
                        cocycles.append((transport[:, i, j] + transport[:, j, k] - transport[:, i, k]).abs())
            cocycle_defect = torch.stack(cocycles, dim=0).mean() if cocycles else zero
        else:
            cocycle_defect = zero

        centered = chart_probs - chart_probs.mean(dim=0, keepdim=True)
        rank_target = min(max(batch - 1, 0), self.num_charts)
        if batch >= 2 and rank_target > 0:
            singular_values = torch.linalg.svdvals(centered)[:rank_target]
            flat_rank_defect = F.relu(0.05 - singular_values).pow(2).mean()
        else:
            flat_rank_defect = zero

        toric_logits = self.toric_head(graph_state)
        toric_probs = torch.softmax(toric_logits, dim=-1)
        toric_top = toric_probs.topk(k=min(2, self.toric_rows), dim=-1).values
        if toric_top.shape[-1] >= 2:
            toric_margin = toric_top[:, 0] - toric_top[:, 1]
        else:
            toric_margin = torch.ones(batch, dtype=graph_state.dtype, device=graph_state.device)
        toric_normal_fan_loss = F.relu(0.05 - toric_margin).mean() if toric_margin.numel() else zero
        toric_active_rows = toric_probs.argmax(dim=-1)

        if graphcg_projection is not None and graphcg_projection.numel() and graphcg_projection.shape[0] == batch:
            graphcg_cells = graphcg_projection.detach().abs().argmax(dim=-1).remainder(self.toric_rows)
            graphcg_toric_cell_agreement = toric_active_rows.detach().eq(graphcg_cells).float().mean()
            graphcg_toric_cell_agreement_loss = F.cross_entropy(toric_logits, graphcg_cells)
            graphcg_toric_cell_available = torch.ones((), dtype=graph_state.dtype, device=graph_state.device)
        else:
            graphcg_toric_cell_agreement = zero
            graphcg_toric_cell_agreement_loss = zero
            graphcg_toric_cell_available = zero

        if graph_token_support_probabilities is not None and graph_token_support_probabilities.numel():
            support_conf = graph_token_support_probabilities.max(dim=-1).values
            if graph_token_mask is not None:
                mask = graph_token_mask.to(dtype=support_conf.dtype)
                per_record_support_conf = (support_conf * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
            else:
                per_record_support_conf = support_conf.mean(dim=-1)
            atom_stability_gap = (chart_confidence - per_record_support_conf).abs().mean()
            atom_stability_available = torch.ones((), dtype=graph_state.dtype, device=graph_state.device)
        else:
            atom_stability_gap = zero
            atom_stability_available = zero

        if per_record_bpb is not None and per_record_bpb.numel() == batch and batch > 0:
            bpb = per_record_bpb.detach().to(dtype=graph_state.dtype, device=graph_state.device)
            chart_mass = chart_probs.sum(dim=0)
            active = chart_mass.gt(1.0e-6)
            chart_bpb_values = (chart_probs * bpb[:, None]).sum(dim=0) / chart_mass.clamp_min(1.0e-6)
            global_bpb = bpb.mean()
            chart_mass_weights = chart_mass / chart_mass.sum().clamp_min(1.0e-6)
            chart_bpb_consistency = (chart_mass_weights * (chart_bpb_values - global_bpb).pow(2)).sum()
            if active.any():
                active_values = chart_bpb_values.masked_select(active)
                chart_bpb_min = active_values.min()
                chart_bpb_max = active_values.max()
                chart_bpb_active_count = active.to(dtype=graph_state.dtype).sum()
            else:
                chart_bpb_min = zero
                chart_bpb_max = zero
                chart_bpb_active_count = zero
            chart_bpb_spread = chart_bpb_max - chart_bpb_min
            chart_bpb_consistency_available = torch.ones((), dtype=graph_state.dtype, device=graph_state.device)
        else:
            chart_bpb_consistency = zero
            chart_bpb_consistency_available = zero
            global_bpb = zero
            chart_bpb_min = zero
            chart_bpb_max = zero
            chart_bpb_spread = zero
            chart_bpb_active_count = zero

        metrics = {
            "bundle_transport_l1": transport_l1,
            "bundle_cocycle_defect": cocycle_defect,
            "bundle_flat_rank_defect": flat_rank_defect,
            "bundle_monomial_projection_one_hotness": monomial_one_hotness,
            "bundle_chart_confidence_mean": chart_confidence.mean() if chart_confidence.numel() else zero,
            "bundle_chart_count": torch.tensor(float(self.num_charts), dtype=graph_state.dtype, device=graph_state.device),
            "bundle_overlap_pair_count": torch.tensor(float(self.num_charts * max(self.num_charts - 1, 0)), dtype=graph_state.dtype, device=graph_state.device),
            "bundle_overlap_triple_count": torch.tensor(float(self.num_charts * max(self.num_charts - 1, 0) * max(self.num_charts - 2, 0)), dtype=graph_state.dtype, device=graph_state.device),
            "toric_normal_fan_loss": toric_normal_fan_loss,
            "toric_active_row_count": torch.tensor(float(self.toric_rows), dtype=graph_state.dtype, device=graph_state.device),
            "graphcg_toric_cell_agreement": graphcg_toric_cell_agreement,
            "graphcg_toric_cell_agreement_available": graphcg_toric_cell_available,
            "chart_bpb_consistency": chart_bpb_consistency,
            "chart_bpb_consistency_available": chart_bpb_consistency_available,
            "chart_bpb_global": global_bpb,
            "chart_bpb_min": chart_bpb_min,
            "chart_bpb_max": chart_bpb_max,
            "chart_bpb_spread": chart_bpb_spread,
            "chart_bpb_active_count": chart_bpb_active_count,
            "bundle_atom_stability_gap": atom_stability_gap,
            "bundle_atom_stability_available": atom_stability_available,
        }
        losses = {
            "bundle_transport_l1": transport_l1,
            "bundle_cocycle_defect": cocycle_defect,
            "bundle_flat_rank_defect": flat_rank_defect,
            "toric_normal_fan_loss": toric_normal_fan_loss,
            "graphcg_toric_cell_agreement": graphcg_toric_cell_agreement_loss,
            "chart_bpb_consistency": chart_bpb_consistency,
            "bundle_atom_stability_gap": atom_stability_gap,
        }
        return metrics, losses


def _zero_chart_bundle_outputs(reference: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    zero = reference.sum() * 0.0
    metrics = {
        "bundle_transport_l1": zero.detach(),
        "bundle_cocycle_defect": zero.detach(),
        "bundle_flat_rank_defect": zero.detach(),
        "bundle_monomial_projection_one_hotness": zero.detach(),
        "bundle_chart_confidence_mean": zero.detach(),
        "bundle_chart_count": zero.detach(),
        "bundle_overlap_pair_count": zero.detach(),
        "bundle_overlap_triple_count": zero.detach(),
        "toric_normal_fan_loss": zero.detach(),
        "toric_active_row_count": zero.detach(),
        "graphcg_toric_cell_agreement": zero.detach(),
        "graphcg_toric_cell_agreement_available": zero.detach(),
        "chart_bpb_consistency": zero.detach(),
        "chart_bpb_consistency_available": zero.detach(),
        "chart_bpb_global": zero.detach(),
        "chart_bpb_min": zero.detach(),
        "chart_bpb_max": zero.detach(),
        "chart_bpb_spread": zero.detach(),
        "chart_bpb_active_count": zero.detach(),
        "bundle_atom_stability_gap": zero.detach(),
        "bundle_atom_stability_available": zero.detach(),
    }
    losses = {
        "bundle_transport_l1": zero,
        "bundle_cocycle_defect": zero,
        "bundle_flat_rank_defect": zero,
        "toric_normal_fan_loss": zero,
        "graphcg_toric_cell_agreement": zero,
        "chart_bpb_consistency": zero,
        "bundle_atom_stability_gap": zero,
    }
    return metrics, losses


def _nll_and_per_record_bpb(logits: Tensor, target_ids: Tensor) -> tuple[Tensor, Tensor]:
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape_as(target_ids)
    valid = target_ids.ne(0)
    if valid.any():
        valid_loss = flat_loss.masked_select(valid)
        nll = valid_loss.mean()
        per_record_counts = valid.sum(dim=1).clamp_min(1).to(dtype=flat_loss.dtype)
        per_record_nll = (flat_loss * valid.to(dtype=flat_loss.dtype)).sum(dim=1) / per_record_counts
    else:
        nll = flat_loss.sum() * 0.0
        per_record_nll = flat_loss.sum(dim=1) * 0.0
    log2 = torch.log(torch.tensor(2.0, dtype=per_record_nll.dtype, device=per_record_nll.device))
    return nll, per_record_nll / log2


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
        requested_graphcg_num_directions = int(config.graphcg_num_directions or config.dim)
        graphcg_num_directions = int(config.dim)
        self.graphcg = GraphCGLoss(
            config.dim,
            num_directions=graphcg_num_directions,
            active_directions=config.graphcg_active_directions,
        )
        self.graphcg.requested_num_directions = requested_graphcg_num_directions
        self.graphcg.effective_num_directions = graphcg_num_directions
        self.graphcg.direction_bank_clamped_to_embedding_dim = requested_graphcg_num_directions != graphcg_num_directions
        self.memory = AnalogicalMemoryHead(config.dim, config.memory_dim)
        self.chart_bundle = (
            ChartBundleToricHead(config.dim, config.bundle_num_charts, config.toric_num_active_rows)
            if config.enable_chart_bundle_auxiliary
            else None
        )

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
        nll: Tensor | None = None
        per_record_bpb: Tensor | None = None
        if target_ids is not None:
            nll, per_record_bpb = _nll_and_per_record_bpb(logits, target_ids)
            per_record_bpb = per_record_bpb.detach()
        valid_margin = trop.margin.masked_select(graph_batch.attention_mask)
        support_entropy = tropical_support_entropy(trop.support, graph_batch.attention_mask)
        soft_entropy = soft_tropical_support_entropy(trop.scores, graph_batch.attention_mask, graph_batch.attention_mask)
        certificate_loss, certificate_metrics = tropical_certificate_objective(trop.scores, trop.support, graph_batch)
        wall_hit_rate = tropical_wall_hit_rate(trop.margin, graph_batch.attention_mask, self.config.wall_margin_threshold)
        near_wall_margin_threshold = max(float(self.config.wall_margin_threshold) * 10.0, float(self.config.wall_margin_threshold))
        near_wall_hit_rate = tropical_wall_hit_rate(trop.margin, graph_batch.attention_mask, near_wall_margin_threshold)
        boundary_hit_rate = tropical_support_boundary_hit_rate(trop.support, graph_batch.attention_mask, graph_batch.graph_token_counts)
        node_edge_ratio = graph_batch.node_counts.float().sum() / graph_batch.edge_counts.float().sum().clamp_min(1.0)
        self_support_rate = tropical_self_support_rate(trop.support, graph_batch.attention_mask)
        invalid_support_rate = tropical_invalid_support_rate(trop.support, graph_batch.attention_mask, graph_batch.graph_token_counts)
        if self.chart_bundle is not None:
            graphcg_projection = graph_state @ self.graphcg.effective_directions(detach=True).to(graph_state.device).t()
            chart_bundle_metrics_raw, chart_bundle_loss_terms = self.chart_bundle(
                graph_state,
                graph_token_support_probabilities=graph_token_support_probabilities,
                graph_token_mask=graph_batch.attention_mask,
                graphcg_projection=graphcg_projection,
                per_record_bpb=per_record_bpb,
            )
            chart_bundle_metrics = {key: value.detach() for key, value in chart_bundle_metrics_raw.items()}
            chart_bundle_transport_metadata = self.chart_bundle.transport_metadata()
        else:
            chart_bundle_metrics, chart_bundle_loss_terms = _zero_chart_bundle_outputs(graph_state)
            chart_bundle_transport_metadata = {
                "available": False,
                "source": "chart_bundle_disabled",
                "chart_ids": [],
                "overlap_pairs": [],
                "overlap_triples": [],
                "overlap_pair_count": 0,
                "overlap_triple_count": 0,
            }
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
            "strict_wall_hit_rate": wall_hit_rate.detach(),
            "near_wall_hit_rate": near_wall_hit_rate.detach(),
            "wall_margin_threshold": torch.tensor(self.config.wall_margin_threshold, device=input_ids.device),
            "near_wall_margin_threshold": torch.tensor(near_wall_margin_threshold, device=input_ids.device),
            "support_boundary_hit_rate": boundary_hit_rate.detach(),
            "analogical_memory_query_norm": self.memory(graph_state).detach().norm(dim=-1).mean(),
            **sequence_tropical_metrics,
            **certificate_metrics,
            **chart_bundle_metrics,
        }
        loss = None
        if target_ids is not None:
            assert nll is not None
            reward = torch.exp(-nll.detach()).repeat(input_ids.shape[0]).clamp_min(1e-6)
            states = graph_state[:, None, :].repeat(1, 2, 1)
            actions = torch.zeros(input_ids.shape[0], 2, dtype=torch.long, device=input_ids.device)
            gfn_loss, gfn_metrics = self.gfn.trajectory_balance_loss(states, actions, reward, return_metrics=True)
            graphcg_loss, graphcg_metrics = self.graphcg(graph_state)
            margin_reward = _safe_mean(valid_margin)
            margin_signed_objective = -margin_reward
            margin_threshold = torch.as_tensor(float(self.config.wall_margin_threshold), device=valid_margin.device)
            margin_shortfall = _safe_mean(F.relu(margin_threshold - valid_margin.float()))
            margin_shortfall_rate = (
                valid_margin.lt(float(self.config.wall_margin_threshold)).float().mean()
                if valid_margin.numel()
                else valid_margin.sum() * 0.0
            )
            entropy_loss = soft_entropy
            gfn_weighted = self.config.gflownet_weight * gfn_loss
            graphcg_weighted = self.config.graphcg_weight * graphcg_loss
            margin_weighted = self.config.margin_weight * margin_signed_objective
            margin_shortfall_weighted = self.config.margin_weight * margin_shortfall
            entropy_weighted = self.config.entropy_weight * entropy_loss
            certificate_weighted = self.config.certificate_weight * certificate_loss
            certificate_diagnostic_penalty_weighted = self.config.certificate_weight * certificate_metrics["certificate_diagnostic_penalty"].to(certificate_loss.device)
            bundle_transport_weighted = self.config.bundle_transport_weight * chart_bundle_loss_terms["bundle_transport_l1"]
            bundle_cocycle_weighted = self.config.bundle_cocycle_weight * chart_bundle_loss_terms["bundle_cocycle_defect"]
            bundle_flat_rank_weighted = self.config.bundle_flat_rank_weight * chart_bundle_loss_terms["bundle_flat_rank_defect"]
            toric_normal_fan_weighted = self.config.toric_normal_fan_weight * chart_bundle_loss_terms["toric_normal_fan_loss"]
            graphcg_toric_cell_agreement_weighted = self.config.graphcg_toric_cell_agreement_weight * chart_bundle_loss_terms["graphcg_toric_cell_agreement"]
            chart_bpb_consistency_weighted = self.config.chart_bpb_consistency_weight * chart_bundle_loss_terms["chart_bpb_consistency"]
            bundle_atom_stability_weighted = self.config.bundle_atom_stability_weight * chart_bundle_loss_terms["bundle_atom_stability_gap"]
            bundle_toric_regularizer_total = (
                bundle_transport_weighted
                + bundle_cocycle_weighted
                + bundle_flat_rank_weighted
                + toric_normal_fan_weighted
                + graphcg_toric_cell_agreement_weighted
                + chart_bpb_consistency_weighted
                + bundle_atom_stability_weighted
            )
            regularizer_total = (
                gfn_weighted
                + graphcg_weighted
                + margin_weighted
                + entropy_weighted
                + certificate_weighted
                + bundle_toric_regularizer_total
            )
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
                    "tropical_margin_loss": margin_shortfall.detach(),
                    "tropical_margin_signed_loss": margin_signed_objective.detach(),
                    "tropical_margin_signed_objective": margin_signed_objective.detach(),
                    "tropical_margin_reward": margin_reward.detach(),
                    "tropical_margin_shortfall_loss": margin_shortfall.detach(),
                    "tropical_margin_shortfall_rate": margin_shortfall_rate.detach(),
                    "tropical_entropy_loss": entropy_loss.detach(),
                    "loss_gflownet_weighted": gfn_weighted.detach(),
                    "loss_graphcg_weighted": graphcg_weighted.detach(),
                    "loss_margin_weighted": margin_weighted.detach(),
                    "loss_tropical_margin_signed_weighted": margin_weighted.detach(),
                    "loss_tropical_margin_shortfall_weighted": margin_shortfall_weighted.detach(),
                    "loss_entropy_weighted": entropy_weighted.detach(),
                    "loss_certificate_weighted": certificate_weighted.detach(),
                    "loss_certificate_objective_weighted": certificate_weighted.detach(),
                    "loss_certificate_diagnostic_penalty_weighted": certificate_diagnostic_penalty_weighted.detach(),
                    "loss_bundle_transport_weighted": bundle_transport_weighted.detach(),
                    "loss_bundle_cocycle_weighted": bundle_cocycle_weighted.detach(),
                    "loss_bundle_flat_rank_weighted": bundle_flat_rank_weighted.detach(),
                    "loss_toric_normal_fan_weighted": toric_normal_fan_weighted.detach(),
                    "loss_graphcg_toric_cell_agreement_weighted": graphcg_toric_cell_agreement_weighted.detach(),
                    "loss_chart_bpb_consistency_weighted": chart_bpb_consistency_weighted.detach(),
                    "loss_bundle_atom_stability_weighted": bundle_atom_stability_weighted.detach(),
                    "loss_bundle_toric_regularizer_total": bundle_toric_regularizer_total.detach(),
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
            "chart_bundle_transport_metadata": chart_bundle_transport_metadata,
            **metrics,
        }


def tropical_certificate_objective(scores: Tensor, support: Tensor, graph_batch: GraphTokenBatch) -> tuple[Tensor, dict[str, Tensor]]:
    targets = tropical_certificate_targets(graph_batch)
    valid = graph_batch.attention_mask
    if valid.sum() == 0:
        zero = scores.sum() * 0.0
        return zero, {
            "certificate_objective_loss": zero.detach(),
            "certificate_diagnostic_penalty": zero.detach(),
            "certificate_loss_reconstruction_error": zero.detach(),
            "certificate_valid_token_count": zero.detach(),
            "certificate_allowed_target_count_mean": zero.detach(),
            "certificate_allowed_target_count_min": zero.detach(),
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
    diagnostic_penalty = loss * 0.0
    target_counts = targets.float().sum(dim=-1)
    valid_target_counts = target_counts.masked_select(valid)
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
        "certificate_objective_loss": loss.detach(),
        "certificate_diagnostic_penalty": diagnostic_penalty.detach(),
        "certificate_loss_reconstruction_error": (loss - loss - diagnostic_penalty).abs().detach(),
        "certificate_valid_token_count": valid.float().sum().detach(),
        "certificate_allowed_target_count_mean": _safe_mean(valid_target_counts).detach(),
        "certificate_allowed_target_count_min": _safe_min(valid_target_counts).detach(),
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
