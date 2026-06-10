from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from .data import encode_bytes
from .algebra import compute_topological_algebra_report
from .diagnostics import ACTION_NAMES, graph_token_trace, per_record_nll, record_diagnostics
from .records import GraphRecord, GraphTokenBatch
from .simplicial import build_filtered_simplicial_object, build_reasoning_trajectory_complex
from .tokenizer import TokenGTTokenizer


def apply_reasoning_action(record: GraphRecord, action: str, rank: int = 0) -> GraphRecord:
    graph = deepcopy(record.graph_json or {"nodes": [], "edges": []})
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    graph["nodes"] = nodes
    graph["edges"] = edges
    source = _last_node_id(nodes)
    safe_action = action if action in ACTION_NAMES else "expand"
    if safe_action == "stop":
        return GraphRecord(
            record_id=f"{record.record_id}|stop{rank}",
            text=record.text,
            question=record.question,
            answer=record.answer,
            reasoning=record.reasoning,
            metadata={**(record.metadata or {}), "scaling_action": safe_action, "scaling_rank": rank},
            graph_json=graph,
        )
    node_id = _fresh_node_id(nodes, safe_action)
    node_type = {
        "expand": "reasoning_step",
        "merge": "merged_state",
        "refine": "refinement",
        "retrieve": "retrieved_evidence",
        "verify": "verification",
        "compress": "compressed_state",
        "reject": "rejected_branch",
    }.get(safe_action, "reasoning_step")
    text = {
        "expand": f"Expand candidate {rank}: decompose the prompt into a next reasoning state.",
        "merge": f"Merge candidate {rank}: combine active predecessors selected by tropical support.",
        "refine": f"Refine candidate {rank}: sharpen the current graph state before answering.",
        "retrieve": f"Retrieve candidate {rank}: attach an evidence placeholder for the prompt.",
        "verify": f"Verify candidate {rank}: check consistency of the current reasoning path.",
        "compress": f"Compress candidate {rank}: summarize the active proof state.",
        "reject": f"Reject candidate {rank}: mark a low-reward branch for pruning.",
    }[safe_action]
    nodes.append({"id": node_id, "type": node_type, "text": text})
    if source is not None:
        edges.append({"source": source, "target": node_id, "type": f"{safe_action}_transition"})
    return GraphRecord(
        record_id=f"{record.record_id}|{safe_action}{rank}",
        text=record.text,
        question=record.question,
        answer=record.answer,
        reasoning=record.reasoning,
        metadata={**(record.metadata or {}), "scaling_action": safe_action, "scaling_rank": rank},
        graph_json=graph,
    )


def run_inference_scaling(
    model,
    seed_record: GraphRecord,
    tokenizer: TokenGTTokenizer,
    seq_len: int,
    device: torch.device,
    depth: int = 2,
    width: int = 3,
    branch_factor: int = 2,
    trace_limit: int = 16,
    audit_level: str = "none",
    ph_backend: str = "auto",
    audit_max_simplices: int = 1024,
) -> dict[str, Any]:
    depth = max(int(depth), 0)
    width = max(int(width), 1)
    branch_factor = max(int(branch_factor), 1)
    frontier = [_candidate_shell(seed_record, path=[], parent=None, level=0)]
    levels: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    for level in range(depth + 1):
        records = [item["record"] for item in frontier]
        scored = score_records(
            model,
            records,
            tokenizer,
            seq_len,
            device,
            trace_limit=trace_limit,
            audit_level=audit_level,
            ph_backend=ph_backend,
            audit_max_simplices=audit_max_simplices,
        )
        for item, score in zip(frontier, scored, strict=True):
            score.update({"path": item["path"], "parent": item["parent"], "level": level})
        scored = sorted(scored, key=lambda row: row["score"], reverse=True)[:width]
        evaluated.extend(scored)
        levels.append(
            {
                "level": level,
                "candidate_count": len(scored),
                "best_score": scored[0]["score"] if scored else None,
                "candidates": _compact_candidates(scored),
            }
        )
        if level >= depth:
            break
        next_frontier = []
        for rank, row in enumerate(scored):
            probs = row["gflownet_action_probs"]
            actions = sorted(probs, key=lambda item: item["probability"], reverse=True)[:branch_factor]
            for branch_rank, action_row in enumerate(actions):
                child = apply_reasoning_action(row["record"], action_row["action"], rank=branch_rank)
                next_frontier.append(
                    _candidate_shell(
                        child,
                        path=row["path"] + [action_row["action"]],
                        parent=row["record_id"],
                        level=level + 1,
                    )
                )
        frontier = next_frontier or frontier
    best = max(evaluated, key=lambda row: row["score"])
    public_candidates = [_public_candidate(row) for row in evaluated]
    trajectory_complex = build_reasoning_trajectory_complex(public_candidates)
    trajectory_algebra = (
        compute_topological_algebra_report(
            trajectory_complex,
            audit_level=audit_level,
            ph_backend=ph_backend,
            max_simplices=audit_max_simplices,
        )
        if (audit_level or "none").lower() != "none"
        else None
    )
    trajectory_growth = []
    if trajectory_algebra is not None:
        max_level = max((int(row.get("level", 0) or 0) for row in public_candidates), default=0)
        for level in range(max_level + 1):
            level_complex = build_reasoning_trajectory_complex(public_candidates, up_to_level=level)
            trajectory_growth.append(
                {
                    "level": level,
                    "filtered_simplicial_object": level_complex,
                    "topological_algebra": compute_topological_algebra_report(
                        level_complex,
                        audit_level=audit_level,
                        ph_backend=ph_backend,
                        max_simplices=audit_max_simplices,
                    ),
                }
            )
    return {
        "enabled": depth > 0,
        "depth": depth,
        "width": width,
        "branch_factor": branch_factor,
        "evaluated_candidates": len(evaluated),
        "levels": levels,
        "best": _public_candidate(best),
        "candidates": public_candidates,
        "trajectory_filtered_simplicial_object": trajectory_complex,
        "trajectory_topological_algebra": trajectory_algebra,
        "trajectory_growth": trajectory_growth,
    }


def score_records(
    model,
    records: list[GraphRecord],
    tokenizer: TokenGTTokenizer,
    seq_len: int,
    device: torch.device,
    trace_limit: int = 16,
    audit_level: str = "none",
    ph_backend: str = "auto",
    audit_max_simplices: int = 1024,
) -> list[dict[str, Any]]:
    xs, ys = zip(*(encode_bytes(record.text, seq_len) for record in records))
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    graph_batch = tokenizer.batch_encode(records)
    with torch.no_grad():
        out = model(x, graph_batch, y)
    graph_batch_cpu = graph_batch.to("cpu")
    out_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in out.items()}
    nll, token_counts = per_record_nll(out_cpu["logits"], torch.stack(ys))
    gfn_logits = model.gfn(out["graph_state"]).detach().cpu()
    gfn_probs = torch.softmax(gfn_logits, dim=-1)
    graphcg_projection = _graphcg_projection(model, out["graph_state"])
    traces = graph_token_trace(records, graph_batch_cpu, out_cpu["support"], out_cpu["margin"], tokenizer, max_tokens=trace_limit)
    rows = []
    for idx, record in enumerate(records):
        mask = graph_batch_cpu.attention_mask[idx]
        margins = out_cpu["margin"][idx].masked_select(mask)
        margin_mean = float(margins.mean()) if margins.numel() else 0.0
        action_probs = _action_probs(gfn_probs[idx])
        action_bonus = action_probs[0]["probability"] if action_probs else 0.0
        graph_tokens = int(graph_batch_cpu.graph_token_counts[idx].item())
        score = -float(nll[idx]) + 0.05 * margin_mean + 0.02 * action_bonus - 0.0005 * graph_tokens
        rows.append(
            {
                "record": record,
                "record_id": record.record_id,
                "score": score,
                "nll": float(nll[idx]),
                "token_count": int(token_counts[idx]),
                "graph_tokens": graph_tokens,
                "node_tokens": int(graph_batch_cpu.node_counts[idx].item()),
                "edge_tokens": int(graph_batch_cpu.edge_counts[idx].item()),
                "margin_mean": margin_mean,
                "gflownet_action_probs": action_probs,
                "embedding": [float(v) for v in out_cpu["graph_state"][idx].tolist()],
                "graphcg_projection": graphcg_projection[idx],
                "graph_token_trace": traces[idx],
                "filtered_simplicial_object": build_filtered_simplicial_object(record),
                "record_diagnostics": record_diagnostics(
                    [record],
                    _slice_graph_batch(graph_batch_cpu, idx),
                    {k: v[idx : idx + 1] if torch.is_tensor(v) and v.ndim > 0 else v for k, v in out_cpu.items()},
                    tokenizer,
                    target_ids=torch.stack([ys[idx]]),
                    max_records=1,
                    max_trace_tokens=trace_limit,
                    audit_level=audit_level,
                    ph_backend=ph_backend,
                    audit_max_simplices=audit_max_simplices,
                )[0],
            }
        )
    return rows


def _candidate_shell(record: GraphRecord, path: list[str], parent: str | None, level: int) -> dict[str, Any]:
    return {"record": record, "path": path, "parent": parent, "level": level}


def _public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "path": row.get("path", []),
        "parent": row.get("parent"),
        "level": row.get("level"),
        "score": row["score"],
        "nll": row["nll"],
        "graph_tokens": row["graph_tokens"],
        "node_tokens": row["node_tokens"],
        "edge_tokens": row["edge_tokens"],
        "margin_mean": row["margin_mean"],
        "gflownet_action_probs": row["gflownet_action_probs"],
        "embedding": row.get("embedding"),
        "graphcg_projection": row.get("graphcg_projection"),
        "graph_token_trace": row["graph_token_trace"],
        "filtered_simplicial_object": row["filtered_simplicial_object"],
        "topological_algebra": row.get("record_diagnostics", {}).get("topological_algebra"),
    }


def _compact_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "path": row.get("path", []),
            "score": row["score"],
            "nll": row["nll"],
            "graph_tokens": row["graph_tokens"],
            "margin_mean": row["margin_mean"],
            "top_action": row["gflownet_action_probs"][0] if row["gflownet_action_probs"] else None,
        }
        for row in rows
    ]


def _action_probs(row: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {
            "action": ACTION_NAMES[idx] if idx < len(ACTION_NAMES) else f"action_{idx}",
            "probability": float(value),
        }
        for idx, value in sorted(enumerate(row.tolist()), key=lambda item: item[1], reverse=True)
    ]


def _graphcg_projection(model, graph_state: torch.Tensor, top_k: int = 4) -> list[dict[str, Any]]:
    dirs = F.normalize(model.graphcg.directions.detach(), dim=-1)
    states = F.normalize(graph_state.detach(), dim=-1)
    scores = (states @ dirs.t()).detach().cpu()
    norms = model.graphcg.directions.detach().norm(dim=-1).cpu()
    gram = (dirs @ dirs.t()).detach().cpu()
    offdiag = gram - torch.eye(gram.shape[0])
    rows = []
    for row in scores:
        k = min(top_k, row.numel())
        values, indices = torch.topk(row, k=k)
        rows.append(
            {
                "direction_norms": [float(v) for v in norms.tolist()],
                "mean_abs_offdiag_cosine": float(offdiag.abs().mean()),
                "max_abs_offdiag_cosine": float(offdiag.abs().max()),
                "top_directions": [
                    {"direction": int(idx), "cosine": float(value)}
                    for value, idx in zip(values, indices)
                ],
                "all_direction_cosines": [float(v) for v in row.tolist()],
            }
        )
    return rows


def _last_node_id(nodes: list[dict[str, Any]]) -> str | None:
    if not nodes:
        return None
    return str(nodes[-1].get("id", len(nodes) - 1))


def _fresh_node_id(nodes: list[dict[str, Any]], prefix: str) -> str:
    existing = {str(node.get("id", "")) for node in nodes}
    idx = len(nodes)
    while f"{prefix}_{idx:03d}" in existing:
        idx += 1
    return f"{prefix}_{idx:03d}"


def _slice_graph_batch(batch: GraphTokenBatch, idx: int) -> GraphTokenBatch:
    return GraphTokenBatch(
        token_features=batch.token_features[idx : idx + 1],
        token_type_ids=batch.token_type_ids[idx : idx + 1],
        endpoint_ids=batch.endpoint_ids[idx : idx + 1],
        attention_mask=batch.attention_mask[idx : idx + 1],
        graph_token_counts=batch.graph_token_counts[idx : idx + 1],
        node_counts=batch.node_counts[idx : idx + 1],
        edge_counts=batch.edge_counts[idx : idx + 1],
        hover_payloads=batch.hover_payloads[idx : idx + 1] if batch.hover_payloads else None,
    )
