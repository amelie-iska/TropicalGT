from __future__ import annotations

import json
import math
from typing import Any, Iterable

import torch
from torch import Tensor

from .data import PAD_ID
from .records import GraphRecord, GraphTokenBatch


def batch_bpb_metrics(
    nll: float | Tensor,
    target_ids: Tensor,
    graph_batch: GraphTokenBatch,
    records: Iterable[GraphRecord],
    graph_side_weight: float = 1.0,
) -> dict[str, float]:
    """Compute ordinary and graph-aware bits-per-byte metrics.

    ``nll`` is the mean natural-log cross entropy over non-padding target bytes.
    The ordinary BPB is therefore total negative log-likelihood in bits divided
    by the number of predicted target bytes.  The graph-aware variants make the
    graph accounting explicit rather than silently treating graph structure as
    free side information.
    """

    if torch.is_tensor(nll):
        nll_value = float(nll.detach().cpu())
    else:
        nll_value = float(nll)
    target_bytes = int(target_ids.detach().ne(PAD_ID).sum().cpu().item())
    graph_token_bytes = int(graph_token_structural_bytes(graph_batch))
    explicit_graph_bytes = int(sum(explicit_graph_json_bytes(record) for record in records))
    nll_bits = nll_value * max(target_bytes, 1) / math.log(2.0)
    side_bits = float(graph_side_weight) * 8.0 * explicit_graph_bytes
    denom_text = max(target_bytes, 1)
    denom_graph = max(target_bytes + graph_token_bytes, 1)
    return {
        "target_bytes": float(target_bytes),
        "nll_bits": float(nll_bits),
        "bpb": float(nll_bits / denom_text),
        "text_bpb": float(nll_bits / denom_text),
        "graph_bpb": float((nll_bits + side_bits) / denom_graph),
        "graph_sideinfo_bpb": float((nll_bits + side_bits) / denom_text),
        "graph_conditioned_bpb_no_side_cost": float(nll_bits / denom_graph),
        "graph_token_structural_bytes": float(graph_token_bytes),
        "explicit_graph_json_bytes": float(explicit_graph_bytes),
        "graph_sideinfo_bits": float(side_bits),
    }


def aggregate_bpb_metrics(
    nll_bits: float,
    target_bytes: int,
    graph_token_bytes: int,
    explicit_graph_bytes: int,
    graph_side_weight: float = 1.0,
) -> dict[str, float]:
    side_bits = float(graph_side_weight) * 8.0 * int(explicit_graph_bytes)
    denom_text = max(int(target_bytes), 1)
    denom_graph = max(int(target_bytes) + int(graph_token_bytes), 1)
    return {
        "bpb": float(nll_bits / denom_text),
        "text_bpb": float(nll_bits / denom_text),
        "graph_bpb": float((nll_bits + side_bits) / denom_graph),
        "graph_sideinfo_bpb": float((nll_bits + side_bits) / denom_text),
        "graph_conditioned_bpb_no_side_cost": float(nll_bits / denom_graph),
        "target_bytes": float(target_bytes),
        "nll_bits": float(nll_bits),
        "graph_token_structural_bytes": float(graph_token_bytes),
        "explicit_graph_json_bytes": float(explicit_graph_bytes),
        "graph_sideinfo_bits": float(side_bits),
    }


def graph_token_structural_bytes(graph_batch: GraphTokenBatch) -> int:
    """Estimate a deterministic byte budget for TokenGT structural tokens."""

    mask = graph_batch.attention_mask.detach().cpu()
    counts = graph_batch.node_counts.detach().cpu()
    endpoints = graph_batch.endpoint_ids.detach().cpu()
    type_ids = graph_batch.token_type_ids.detach().cpu()
    total = 0
    for row in range(mask.shape[0]):
        node_count = max(int(counts[row].item()), 1)
        endpoint_width = max(1, math.ceil(math.log2(node_count + 2) / 8.0))
        for col in range(mask.shape[1]):
            if not bool(mask[row, col].item()):
                continue
            total += 1
            if int(type_ids[row, col].item()) == 1:
                valid_endpoints = sum(1 for value in endpoints[row, col].tolist() if int(value) >= 0)
                total += endpoint_width * max(valid_endpoints, 1)
    return total


def explicit_graph_json_bytes(record: GraphRecord) -> int:
    """Bytes of non-derived graph side information.

    Sequential text graphs and conservative fallback graphs are deterministic
    from the byte stream, so they should not be charged as external side
    information.  Parsed graph JSON is charged after stripping the derived
    sequence path that ``GraphRecord.from_mapping`` appends for text training.
    """

    metadata = record.metadata or {}
    if metadata.get("graph_json_fallback", False):
        return 0
    graph = strip_derived_sequence_graph(record.graph_json or {})
    if not graph.get("nodes") and not graph.get("edges"):
        return 0
    payload = json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return len(payload.encode("utf-8", "ignore"))


def strip_derived_sequence_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    removed_ids: set[str] = set()
    for idx, node in enumerate(graph.get("nodes", [])):
        node_id = str(node.get("id", idx))
        if str(node.get("type", "")) in {"sequence_document", "sequence_chunk"}:
            removed_ids.add(node_id)
            continue
        nodes.append(node)
    edges = []
    for edge in graph.get("edges", []):
        if str(edge.get("type", "")) in {"starts_text", "next_text_chunk"}:
            continue
        if str(edge.get("source", "")) in removed_ids or str(edge.get("target", "")) in removed_ids:
            continue
        edges.append(edge)
    return {**graph, "nodes": nodes, "edges": edges}
