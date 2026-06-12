from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .attention import tropical_support_entropy
from .algebra import compute_topological_algebra_report
from .records import GraphRecord, GraphTokenBatch
from .simplicial import build_embedding_radius_simplicial_object
from .tokenizer import TokenGTTokenizer

ACTION_NAMES = ["expand", "merge", "refine", "stop", "retrieve", "verify", "compress", "reject"]


def per_record_nll(logits: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target_ids.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape_as(target_ids)
    mask = target_ids.ne(0)
    token_counts = mask.sum(dim=1).clamp_min(1)
    nll = (losses * mask).sum(dim=1) / token_counts
    return nll.detach(), token_counts.detach()


def describe_graph_tokens(record: GraphRecord, tokenizer: TokenGTTokenizer) -> list[dict[str, Any]]:
    graph = record.graph_json or {"nodes": [], "edges": []}
    raw_nodes = list(graph.get("nodes", []))[: tokenizer.max_nodes]
    node_ids = [str(node.get("id", idx)) for idx, node in enumerate(raw_nodes)]
    node_lookup = {node_id: idx for idx, node_id in enumerate(node_ids)}
    tokens: list[dict[str, Any]] = []
    if tokenizer.graph_token:
        tokens.append(
            {
                "index": len(tokens),
                "kind": "graph",
                "label": "graph",
                "endpoint_ids": [-1, -1],
                "text": record.text[:256],
            }
        )
    for idx, node in enumerate(raw_nodes):
        tokens.append(
            {
                "index": len(tokens),
                "kind": "node",
                "node_id": node_ids[idx],
                "node_type": str(node.get("type", "node")),
                "label": str(node.get("type", "node")),
                "endpoint_ids": [idx, idx],
                "text": str(node.get("text", ""))[:512],
            }
        )
    edge_count = 0
    for edge in list(graph.get("edges", []))[: tokenizer.max_edges]:
        src = str(edge.get("source", edge.get("src", "")))
        dst = str(edge.get("target", edge.get("dst", "")))
        s = node_lookup.get(src, -1)
        t = node_lookup.get(dst, -1)
        if s < 0 or t < 0:
            continue
        tokens.append(
            {
                "index": len(tokens),
                "kind": "edge",
                "edge_type": str(edge.get("type", "edge")),
                "label": f"{src}->{dst}",
                "source": src,
                "target": dst,
                "endpoint_ids": [s, t],
                "text": str(edge.get("text", ""))[:512],
            }
        )
        edge_count += 1
    return tokens


def graph_token_trace(
    records: Sequence[GraphRecord],
    graph_batch: GraphTokenBatch,
    support: torch.Tensor,
    margin: torch.Tensor,
    tokenizer: TokenGTTokenizer,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    support_cpu = support.detach().cpu()
    margin_cpu = margin.detach().cpu()
    mask_cpu = graph_batch.attention_mask.detach().cpu()
    endpoint_cpu = graph_batch.endpoint_ids.detach().cpu()
    traces = []
    for batch_idx, record in enumerate(records):
        descriptors = describe_graph_tokens(record, tokenizer)
        token_count = int(mask_cpu[batch_idx].sum().item())
        rows = []
        for token_idx in range(token_count if max_tokens is None else min(token_count, max_tokens)):
            desc = dict(descriptors[token_idx]) if token_idx < len(descriptors) else {"index": token_idx, "kind": "unknown"}
            active = int(support_cpu[batch_idx, token_idx].item())
            active_desc = descriptors[active] if 0 <= active < len(descriptors) else {"index": active, "kind": "padded"}
            desc.update(
                {
                    "endpoint_ids": [int(v) for v in endpoint_cpu[batch_idx, token_idx].tolist()],
                    "active_support_index": active,
                    "active_support_kind": active_desc.get("kind"),
                    "active_support_label": active_desc.get("label"),
                    "margin": float(margin_cpu[batch_idx, token_idx].item()),
                }
            )
            rows.append(desc)
        traces.append(
            {
                "record_id": record.record_id,
                "graph_token_count": token_count,
                "tokens": rows,
                "truncated": max_tokens is not None and token_count > max_tokens,
            }
        )
    return traces


def tropical_record_summary(support: torch.Tensor, margin: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    valid_margin = margin.detach().masked_select(mask)
    support_valid = support.detach()[mask]
    return {
        "support_entropy": float(tropical_support_entropy(support.detach(), mask.detach()).cpu()),
        "margin_mean": float(valid_margin.mean().cpu()) if valid_margin.numel() else 0.0,
        "margin_min": float(valid_margin.min().cpu()) if valid_margin.numel() else 0.0,
        "active_support_histogram": _histogram(support_valid.cpu().tolist()),
    }


def gflownet_diagnostics(model, graph_state: torch.Tensor, top_k: int = 3) -> dict[str, Any]:
    logits = model.gfn(graph_state)
    probs = torch.softmax(logits, dim=-1).detach().cpu()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    records = []
    for row in probs:
        k = min(top_k, row.numel())
        values, indices = torch.topk(row, k=k)
        records.append(
            [
                {
                    "action": ACTION_NAMES[int(idx)] if int(idx) < len(ACTION_NAMES) else f"action_{int(idx)}",
                    "probability": float(value),
                }
                for value, idx in zip(values, indices)
            ]
        )
    return {"entropy_mean": float(entropy.mean()), "top_actions": records}


def graphcg_diagnostics(model, graph_state: torch.Tensor, top_k: int = 3) -> dict[str, Any]:
    raw_dirs = model.graphcg.directions.detach()
    dirs = model.graphcg.effective_directions(detach=True) if hasattr(model.graphcg, "effective_directions") else raw_dirs
    norms = raw_dirs.norm(dim=-1).cpu()
    dirs_norm = F.normalize(dirs, dim=-1)
    state_norm = F.normalize(graph_state.detach(), dim=-1)
    scores = (state_norm @ dirs_norm.t()).cpu()
    gram = (dirs_norm @ dirs_norm.t()).cpu()
    offdiag = gram - torch.eye(gram.shape[0])
    singular_values = torch.linalg.svdvals(dirs_norm).cpu()
    rank_margin = float(getattr(model.graphcg, "full_rank_margin", 0.05))
    rank_target = min(dirs_norm.shape[0], dirs_norm.shape[1])
    active = singular_values[:rank_target].clamp_min(1e-8)
    probs = active / active.sum().clamp_min(1e-8)
    effective_rank = torch.exp(-(probs * probs.log()).sum())
    top = []
    for row in scores:
        k = min(top_k, row.numel())
        values, indices = torch.topk(row, k=k)
        top.append([{"direction": int(idx), "cosine": float(value)} for value, idx in zip(values, indices)])
    return {
        "direction_norms": [float(v) for v in norms.tolist()],
        "basis": "effective_full_rank_qr" if hasattr(model.graphcg, "effective_directions") else "raw_normalized",
        "mean_abs_offdiag_cosine": float(offdiag.abs().mean()),
        "full_rank_margin": rank_margin,
        "rank_target": int(rank_target),
        "numerical_rank": int(singular_values[:rank_target].gt(rank_margin).sum().item()),
        "effective_rank": float(effective_rank),
        "singular_values": [float(v) for v in singular_values.tolist()],
        "singular_min": float(active.min()),
        "singular_max": float(active.max()),
        "svd_condition_proxy": float(active.max() / active.min().clamp_min(1e-8)),
        "top_directions": top,
        "projection_scores": [
            [{"direction": int(idx), "cosine": float(value)} for idx, value in enumerate(row.tolist())]
            for row in scores
        ],
    }


def record_diagnostics(
    records: Sequence[GraphRecord],
    graph_batch: GraphTokenBatch,
    out: dict[str, torch.Tensor],
    tokenizer: TokenGTTokenizer,
    target_ids: torch.Tensor | None = None,
    max_records: int | None = None,
    max_trace_tokens: int | None = 16,
    audit_level: str = "none",
    ph_backend: str = "auto",
    audit_max_simplices: int = 1024,
) -> list[dict[str, Any]]:
    support = out["support"].detach().cpu()
    margin = out["margin"].detach().cpu()
    nll = token_counts = None
    if target_ids is not None:
        nll, token_counts = per_record_nll(out["logits"].detach().cpu(), target_ids.detach().cpu())
    traces = graph_token_trace(records, graph_batch, support, margin, tokenizer, max_tokens=max_trace_tokens)
    graph_token_embeddings = out.get("graph_token_embeddings")
    graph_token_support_probabilities = out.get("graph_token_support_probabilities")
    if torch.is_tensor(graph_token_embeddings):
        graph_token_embeddings = graph_token_embeddings.detach().cpu()
    if torch.is_tensor(graph_token_support_probabilities):
        graph_token_support_probabilities = graph_token_support_probabilities.detach().cpu()
    rows = []
    limit = len(records) if max_records is None else min(max_records, len(records))
    for idx in range(limit):
        mask = graph_batch.attention_mask[idx].detach().cpu()
        graph_token_count = int(graph_batch.graph_token_counts[idx].item())
        descriptors = describe_graph_tokens(records[idx], tokenizer)[:graph_token_count]
        embeddings = (
            graph_token_embeddings[idx, :graph_token_count]
            if torch.is_tensor(graph_token_embeddings) and graph_token_embeddings.ndim >= 3
            else []
        )
        probabilities = (
            graph_token_support_probabilities[idx, :graph_token_count, :graph_token_count]
            if torch.is_tensor(graph_token_support_probabilities) and graph_token_support_probabilities.ndim >= 3
            else None
        )
        filtered_object = build_embedding_radius_simplicial_object(
            records[idx],
            descriptors,
            embeddings,
            token_probabilities=probabilities,
            metric="jensen_shannon",
        )
        embedding_filtered_object = build_embedding_radius_simplicial_object(
            records[idx],
            descriptors,
            embeddings,
            metric="euclidean",
        )
        row = {
            "record_id": records[idx].record_id,
            "nll": float(nll[idx]) if nll is not None else None,
            "token_count": int(token_counts[idx]) if token_counts is not None else None,
            "graph_tokens": graph_token_count,
            "node_tokens": int(graph_batch.node_counts[idx].item()),
            "edge_tokens": int(graph_batch.edge_counts[idx].item()),
            "graph_json_fallback": bool((records[idx].metadata or {}).get("graph_json_fallback", False)),
            "graph_json_sequentialized": bool((records[idx].metadata or {}).get("graph_json_sequentialized", False)),
            "tropical": tropical_record_summary(support[idx : idx + 1], margin[idx : idx + 1], mask[None, :]),
            "graph_token_trace": traces[idx],
            "filtered_simplicial_object": filtered_object,
            "probability_filtered_simplicial_object": filtered_object,
            "embedding_filtered_simplicial_object": embedding_filtered_object,
        }
        if (audit_level or "none").lower() != "none":
            row["topological_algebra"] = compute_topological_algebra_report(
                filtered_object,
                audit_level=audit_level,
                ph_backend=ph_backend,
                max_simplices=audit_max_simplices,
            )
        rows.append(row)
    return rows


def _histogram(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return out
