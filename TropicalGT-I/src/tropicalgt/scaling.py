from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F

from .data import encode_bytes
from .algebra import compute_level_radius_bifiltration_report, compute_topological_algebra_report
from .diagnostics import ACTION_NAMES, graph_token_trace, per_record_nll, record_diagnostics
from .records import GraphRecord, GraphTokenBatch, graph_decoding_order
from .simplicial import build_reasoning_trajectory_complex
from .tokenizer import TokenGTTokenizer

def _has_real_probability_complex(obj: Any) -> bool:
    if not isinstance(obj, dict) or obj.get("available") is False:
        return False
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    model = str(summary.get("filtration_model", summary.get("metric", ""))).lower()
    if "probability" not in model and "jensen" not in model and "js" not in model:
        return False
    if int(summary.get("num_edges", summary.get("edges", 0)) or 0) <= 0:
        return False
    simplices = obj.get("simplices", [])
    if not isinstance(simplices, list):
        return False
    for simplex in simplices:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1) or -1) != 0:
            continue
        if any(key in simplex for key in ("probability", "probability_vector", "model_probability_vector", "token_probability")):
            return True
    return False



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
    microsteps = _reasoning_microsteps(record, safe_action, rank)
    headline = f"{safe_action.title()} candidate {rank}: execute {len(microsteps)} graph-of-thought micro-steps."
    nodes.append({"id": node_id, "type": node_type, "text": headline, "action": safe_action, "microstep_count": len(microsteps)})
    if source is not None:
        edges.append({"source": source, "target": node_id, "type": f"{safe_action}_transition"})
    previous = node_id
    micro_payloads = []
    for micro_idx, micro in enumerate(microsteps):
        micro_id = _fresh_node_id(nodes, f"{safe_action}_{micro['kind']}")
        payload = {
            "id": micro_id,
            "type": f"{node_type}_{micro['kind']}",
            "text": micro["text"],
            "action": safe_action,
            "reasoning_step_id": node_id,
            "microstep_index": micro_idx,
            "filtration_hint": round((micro_idx + 1) / max(len(microsteps), 1), 6),
        }
        nodes.append(payload)
        micro_payloads.append(payload)
        edges.append({"source": previous, "target": micro_id, "type": "starts_microstep" if micro_idx == 0 else "next_microstep"})
        if micro_idx > 0:
            edges.append({"source": node_id, "target": micro_id, "type": "contains_microstep"})
        previous = micro_id
    updated_text = _append_reasoning_trace(record.text, safe_action, micro_payloads)
    updated_reasoning = _append_reasoning_trace(record.reasoning, safe_action, micro_payloads) if record.reasoning else _append_reasoning_trace("", safe_action, micro_payloads)
    metadata = {
        **(record.metadata or {}),
        "scaling_action": safe_action,
        "scaling_rank": rank,
        "reasoning_microstep_count": len(micro_payloads),
        "reasoning_microsteps": [
            {"id": item["id"], "type": item["type"], "text": item["text"], "microstep_index": item["microstep_index"]}
            for item in micro_payloads
        ],
    }
    metadata.update(graph_decoding_order(graph, seed=rank, record_id=f"{record.record_id}|{safe_action}{rank}"))
    return GraphRecord(
        record_id=f"{record.record_id}|{safe_action}{rank}",
        text=updated_text,
        question=record.question,
        answer=record.answer,
        reasoning=updated_reasoning,
        metadata=metadata,
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
    allow_stop: bool = False,
    diverse_actions: bool = True,
    stochastic_actions: bool = False,
    sampling_temperature: float = 1.0,
    sampling_exploration: float = 0.0,
    sampling_seed: int | None = None,
    require_complete_reasoning_steps: bool = False,
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
            actions = _select_branch_actions(
                probs,
                branch_factor=branch_factor,
                allow_stop=allow_stop,
                diverse_actions=diverse_actions,
                path=row.get("path", []),
                stochastic=stochastic_actions,
                temperature=sampling_temperature,
                exploration=sampling_exploration,
                seed=_branch_sampling_seed(sampling_seed, row.get("record_id", ""), level, rank, row.get("path", [])),
            )
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
    reasoning_step_audit = _reasoning_step_audit(public_candidates, require_complete=bool(require_complete_reasoning_steps))
    if require_complete_reasoning_steps and reasoning_step_audit["incomplete_candidate_count"] > 0:
        preview = reasoning_step_audit["incomplete_candidates"][:6]
        raise ValueError(
            "Full reasoning audit requires complete model-derived reasoning steps; "
            f"incomplete={reasoning_step_audit['incomplete_candidate_count']} preview={preview}"
        )
    trajectory_complex = build_reasoning_trajectory_complex(public_candidates)
    trajectory_probability_complex = build_reasoning_trajectory_complex(public_candidates, metric="jensen_shannon")
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
    trajectory_probability_algebra = (
        compute_topological_algebra_report(
            trajectory_probability_complex,
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
            level_probability_complex = build_reasoning_trajectory_complex(public_candidates, up_to_level=level, metric="jensen_shannon")
            trajectory_growth.append(
                {
                    "level": level,
                    "filtered_simplicial_object": level_complex,
                    "probability_filtered_simplicial_object": level_probability_complex,
                    "topological_algebra": compute_topological_algebra_report(
                        level_complex,
                        audit_level=audit_level,
                        ph_backend=ph_backend,
                        max_simplices=audit_max_simplices,
                    ),
                    "probability_topological_algebra": compute_topological_algebra_report(
                        level_probability_complex,
                        audit_level=audit_level,
                        ph_backend=ph_backend,
                        max_simplices=audit_max_simplices,
                    ),
                }
            )
    probability_growth_available = bool(
        trajectory_growth
        and _has_real_probability_complex(trajectory_probability_complex)
        and all(_has_real_probability_complex(row.get("probability_filtered_simplicial_object")) for row in trajectory_growth if isinstance(row, dict))
    )
    trajectory_level_radius_bifiltration = compute_level_radius_bifiltration_report(
        trajectory_growth,
        object_key="probability_filtered_simplicial_object" if probability_growth_available else "filtered_simplicial_object",
        max_simplices=audit_max_simplices,
    ) if trajectory_growth else {"available": False, "reason": "trajectory growth unavailable"}
    if isinstance(trajectory_level_radius_bifiltration, dict):
        trajectory_level_radius_bifiltration["object_key_policy"] = (
            "probability Jensen-Shannon complexes only when every growth row has real model probability vertices/edges; otherwise embedding radius complexes"
        )
        trajectory_level_radius_bifiltration["object_key_selected"] = "probability_filtered_simplicial_object" if probability_growth_available else "filtered_simplicial_object"

    return {
        "enabled": depth > 0,
        "depth": depth,
        "width": width,
        "branch_factor": branch_factor,
        "allow_stop": bool(allow_stop),
        "diverse_actions": bool(diverse_actions),
        "stochastic_actions": bool(stochastic_actions),
        "sampling_temperature": float(sampling_temperature),
        "sampling_exploration": float(sampling_exploration),
        "sampling_seed": sampling_seed,
        "require_complete_reasoning_steps": bool(require_complete_reasoning_steps),
        "reasoning_step_audit": reasoning_step_audit,
        "evaluated_candidates": len(evaluated),
        "levels": levels,
        "best": _public_candidate(best),
        "candidates": public_candidates,
        "trajectory_filtered_simplicial_object": trajectory_complex,
        "trajectory_probability_filtered_simplicial_object": trajectory_probability_complex,
        "trajectory_topological_algebra": trajectory_algebra,
        "trajectory_probability_topological_algebra": trajectory_probability_algebra,
        "trajectory_growth": trajectory_growth,
        "trajectory_level_radius_bifiltration": trajectory_level_radius_bifiltration,
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
    decoded_argmax = [_decode_shifted_bytes(row) for row in out_cpu["logits"].argmax(dim=-1)]
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
        diagnostics = record_diagnostics(
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
        )[0]
        probability_complex = diagnostics.get("probability_filtered_simplicial_object") or diagnostics.get("filtered_simplicial_object", {})
        embedding_complex = diagnostics.get("embedding_filtered_simplicial_object", {})
        action_probability_vector = [float(row.get("probability", 0.0)) for row in action_probs]
        rows.append(
            {
                "record": record,
                "record_id": record.record_id,
                "score": score,
                "nll": float(nll[idx]),
                "token_count": int(token_counts[idx]),
                "input_text": record.text,
                "target_text": _decode_shifted_bytes(ys[idx]),
                "decoded_argmax": decoded_argmax[idx],
                "graph_json_summary": _graph_json_summary(record.graph_json),
                "decoding_order_report": _decoding_order_report(record),
                "graph_tokens": graph_tokens,
                "node_tokens": int(graph_batch_cpu.node_counts[idx].item()),
                "edge_tokens": int(graph_batch_cpu.edge_counts[idx].item()),
                "margin_mean": margin_mean,
                "gflownet_action_probs": action_probs,
                "action_probability_vector": action_probability_vector,
                "action_probability_source": "TropicalGTModel.gfn(graph_state).softmax",
                "embedding": [float(v) for v in out_cpu["graph_state"][idx].tolist()],
                "embedding_source": "TropicalGTModel.graph_state",
                "graphcg_projection": graphcg_projection[idx],
                "graph_token_trace": traces[idx],
                "filtered_simplicial_object": probability_complex,
                "filtered_simplicial_object_source": "probability_filtered_simplicial_object",
                "probability_filtered_simplicial_object": probability_complex,
                "probability_filtered_simplicial_object_source": "TropicalGTModel.graph_token_support_probabilities+jensen_shannon",
                "embedding_filtered_simplicial_object": embedding_complex,
                "embedding_filtered_simplicial_object_source": "TropicalGTModel.graph_token_embeddings+euclidean",
                "record_diagnostics": diagnostics,
            }
        )
    return rows


def _candidate_shell(record: GraphRecord, path: list[str], parent: str | None, level: int) -> dict[str, Any]:
    return {"record": record, "path": path, "parent": parent, "level": level}


def _public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    completeness = _reasoning_step_completeness(row)
    return {
        "record_id": row["record_id"],
        "path": row.get("path", []),
        "parent": row.get("parent"),
        "level": row.get("level"),
        "score": row["score"],
        "nll": row["nll"],
        "token_count": row.get("token_count"),
        "graph_tokens": row["graph_tokens"],
        "node_tokens": row["node_tokens"],
        "edge_tokens": row["edge_tokens"],
        "margin_mean": row["margin_mean"],
        "gflownet_action_probs": row["gflownet_action_probs"],
        "action_probability_vector": row.get("action_probability_vector"),
        "action_probability_source": row.get("action_probability_source"),
        "embedding": row.get("embedding"),
        "embedding_source": row.get("embedding_source"),
        "input_text": row.get("input_text", ""),
        "target_text": row.get("target_text", ""),
        "decoded_argmax": row.get("decoded_argmax", ""),
        "graph_json_summary": row.get("graph_json_summary", {}),
        "decoding_order_report": row.get("decoding_order_report", {}),
        "graphcg_projection": row.get("graphcg_projection"),
        "graph_token_trace": row["graph_token_trace"],
        "filtered_simplicial_object": row["filtered_simplicial_object"],
        "filtered_simplicial_object_source": row.get("filtered_simplicial_object_source"),
        "probability_filtered_simplicial_object": row.get("probability_filtered_simplicial_object"),
        "probability_filtered_simplicial_object_source": row.get("probability_filtered_simplicial_object_source"),
        "embedding_filtered_simplicial_object": row.get("embedding_filtered_simplicial_object"),
        "embedding_filtered_simplicial_object_source": row.get("embedding_filtered_simplicial_object_source"),
        "topological_algebra": row.get("record_diagnostics", {}).get("topological_algebra"),
        "reasoning_step_structure": _reasoning_step_structure(row),
        "reasoning_step_complete": bool(completeness["complete"]),
        "reasoning_step_completeness": completeness,
        "reasoning_step_model_data": _reasoning_step_model_data(row),
    }


def _reasoning_step_model_data(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("graph_token_trace") or {}
    graph_summary = row.get("graph_json_summary") or {}
    return {
        "source": "TropicalGTModel.forward+record_diagnostics",
        "directionality": {
            "parent": row.get("parent"),
            "level": row.get("level"),
            "path": list(row.get("path") or []),
        },
        "sequence_budget": {
            "target_tokens": int(row.get("token_count") or 0),
            "graph_tokens": int(row.get("graph_tokens") or 0),
            "node_tokens": int(row.get("node_tokens") or 0),
            "edge_tokens": int(row.get("edge_tokens") or 0),
        },
        "trace_budget": {
            "graph_token_count": int(trace.get("graph_token_count") or 0) if isinstance(trace, dict) else 0,
            "emitted_trace_tokens": _trace_token_count(trace),
            "truncated": bool(trace.get("truncated")) if isinstance(trace, dict) else True,
        },
        "graph_json": {
            "node_count": int(graph_summary.get("node_count") or 0),
            "edge_count": int(graph_summary.get("edge_count") or 0),
        },
        "embedding_source": row.get("embedding_source"),
        "action_probability_source": row.get("action_probability_source"),
        "probability_complex_source": row.get("probability_filtered_simplicial_object_source"),
        "embedding_complex_source": row.get("embedding_filtered_simplicial_object_source"),
        "probability_distance": _complex_metric(row.get("probability_filtered_simplicial_object")),
        "embedding_distance": _complex_metric(row.get("embedding_filtered_simplicial_object")),
        "graphcg_basis": (row.get("graphcg_projection") or {}).get("basis") if isinstance(row.get("graphcg_projection"), dict) else None,
    }


def _reasoning_step_completeness(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("graph_token_trace") or {}
    trace_tokens = _trace_token_count(trace)
    trace_total = int(trace.get("graph_token_count") or 0) if isinstance(trace, dict) else 0
    trace_truncated = bool(trace.get("truncated")) if isinstance(trace, dict) else True
    checks = {
        "sequence_model_output": _is_finite_number(row.get("nll")) and int(row.get("token_count") or 0) > 0 and bool(str(row.get("decoded_argmax", ""))),
        "graph_state_embedding": isinstance(row.get("embedding"), list) and len(row.get("embedding") or []) > 0 and row.get("embedding_source") == "TropicalGTModel.graph_state",
        "action_distribution": _probability_vector_is_valid(row.get("action_probability_vector")),
        "graph_token_trace_complete": trace_total > 0 and trace_tokens >= trace_total and not trace_truncated,
        "probability_complex_js": _complex_is_available(
            row.get("probability_filtered_simplicial_object"),
            required_metric="jensen_shannon",
            required_source="model_tropical_support_probabilities",
        ),
        "embedding_complex_euclidean": _complex_is_available(
            row.get("embedding_filtered_simplicial_object"),
            required_metric="euclidean",
            required_source="TropicalGTModel.graph_token_embeddings",
        ),
        "graphcg_all_directions": _graphcg_projection_complete(row.get("graphcg_projection")),
        "directed_graph_context": int(row.get("graph_tokens") or 0) > 0 and int(row.get("node_tokens") or 0) + int(row.get("edge_tokens") or 0) > 0,
        "directed_trajectory_edge": _has_directed_trajectory_edge(row),
        "complete_reasoning_step_structure": _has_complete_reasoning_step_structure(row),
    }
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "complete": not missing,
        "checks": checks,
        "missing": missing,
        "policy": "complete_reasoning_step_requires_model_sequence_output_graph_state_action_probs_graph_token_trace_probability_js_complex_embedding_complex_graphcg_projection_directed_parent_edge_and_complete_microstep_chain",
    }


def _reasoning_step_audit(candidates: list[dict[str, Any]], require_complete: bool = False) -> dict[str, Any]:
    rows = [row for row in candidates if isinstance(row, dict)]
    incomplete = []
    missing_counts: dict[str, int] = {}
    for row in rows:
        completeness = row.get("reasoning_step_completeness") if isinstance(row, dict) else None
        if not isinstance(completeness, dict):
            missing = ["reasoning_step_completeness"]
        else:
            missing = [str(item) for item in completeness.get("missing", [])]
        if missing:
            for item in missing:
                missing_counts[item] = missing_counts.get(item, 0) + 1
            incomplete.append(
                {
                    "record_id": row.get("record_id"),
                    "level": row.get("level"),
                    "parent": row.get("parent"),
                    "path": row.get("path", []),
                    "missing": missing,
                }
            )
    return {
        "policy": "browserable_full_audit_requires_each_reasoning_step_to_have_real_model_nll_graph_state_action_distribution_graph_token_trace_probability_js_complex_embedding_complex_graphcg_directed_parent_edge_and_complete_microstep_chain",
        "require_complete": bool(require_complete),
        "candidate_count": len(rows),
        "complete_candidate_count": len(rows) - len(incomplete),
        "incomplete_candidate_count": len(incomplete),
        "missing_counts": dict(sorted(missing_counts.items())),
        "incomplete_candidates": incomplete,
    }


def _trace_token_count(trace: Any) -> int:
    if not isinstance(trace, dict):
        return 0
    tokens = trace.get("tokens")
    return len(tokens) if isinstance(tokens, list) else 0


def _reasoning_step_structure(row: dict[str, Any]) -> dict[str, Any]:
    level = int(row.get("level") or 0)
    path = row.get("path") if isinstance(row.get("path"), list) else []
    record = row.get("record")
    metadata = getattr(record, "metadata", {}) if record is not None else {}
    if not isinstance(metadata, dict):
        metadata = {}
    graph_json = getattr(record, "graph_json", None) if record is not None else None
    graph = graph_json if isinstance(graph_json, dict) else {}
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []
    microsteps = metadata.get("reasoning_microsteps")
    microstep_rows = microsteps if isinstance(microsteps, list) else []
    microstep_count = int(metadata.get("reasoning_microstep_count") or len(microstep_rows))
    action = str(metadata.get("scaling_action") or (path[-1] if path else "seed"))
    graph_microstep_nodes = [
        node for node in nodes if isinstance(node, dict) and str(node.get("reasoning_step_id", "")).strip()
    ]
    edge_types = {str(edge.get("type", "")) for edge in edges if isinstance(edge, dict)}
    return {
        "kind": "seed" if level <= 0 else "model_evaluated_action_step",
        "level": level,
        "parent": row.get("parent"),
        "path": path,
        "action": action,
        "microstep_count": microstep_count,
        "metadata_microsteps": len(microstep_rows),
        "graph_microstep_nodes": len(graph_microstep_nodes),
        "has_starts_microstep_edge": "starts_microstep" in edge_types,
        "has_next_microstep_edge": "next_microstep" in edge_types,
        "source": "GraphRecord.metadata.reasoning_microsteps+GraphRecord.graph_json",
    }


def _has_complete_reasoning_step_structure(row: dict[str, Any]) -> bool:
    structure = _reasoning_step_structure(row)
    level = int(structure.get("level") or 0)
    if level <= 0:
        return row.get("parent") in (None, "") and not (row.get("path") or [])
    if not (isinstance(row.get("parent"), str) and row.get("parent")):
        return False
    path = row.get("path")
    if not isinstance(path, list) or len(path) < level:
        return False
    action = str(structure.get("action") or "")
    if action not in ACTION_NAMES or action == "seed":
        return False
    microstep_count = int(structure.get("microstep_count") or 0)
    return (
        microstep_count >= 3
        and int(structure.get("metadata_microsteps") or 0) >= microstep_count
        and int(structure.get("graph_microstep_nodes") or 0) >= microstep_count
        and bool(structure.get("has_starts_microstep_edge"))
        and bool(structure.get("has_next_microstep_edge"))
    )


def _complex_metric(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    summary = obj.get("summary")
    if not isinstance(summary, dict):
        return None
    metric = summary.get("embedding_metric")
    return str(metric) if metric is not None else None


def _complex_is_available(obj: Any, required_metric: str, required_source: str) -> bool:
    if not isinstance(obj, dict) or not obj.get("available"):
        return False
    summary = obj.get("summary")
    if not isinstance(summary, dict):
        return False
    if str(summary.get("embedding_metric")) != required_metric:
        return False
    source_blob = " ".join(
        str(value)
        for value in (
            summary.get("embedding_source"),
            summary.get("filtration_model"),
            summary.get("probability_transform"),
        )
    )
    if required_source not in source_blob:
        return False
    return int(summary.get("num_vertices") or 0) > 0 and isinstance(obj.get("simplices"), list) and len(obj.get("simplices") or []) > 0


def _probability_vector_is_valid(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    total = 0.0
    for value in values:
        if not _is_finite_number(value) or float(value) < -1e-8:
            return False
        total += float(value)
    return abs(total - 1.0) <= 1e-3


def _graphcg_projection_complete(projection: Any) -> bool:
    if not isinstance(projection, dict):
        return False
    cosines = projection.get("all_direction_cosines")
    return isinstance(cosines, list) and len(cosines) > 0 and all(_is_finite_number(value) for value in cosines)


def _has_directed_trajectory_edge(row: dict[str, Any]) -> bool:
    level = int(row.get("level") or 0)
    if level <= 0:
        return row.get("parent") in (None, "")
    parent = row.get("parent")
    path = row.get("path")
    return isinstance(parent, str) and bool(parent) and isinstance(path, list) and len(path) >= level


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _compact_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "path": row.get("path", []),
            "score": row["score"],
            "nll": row["nll"],
            "graph_tokens": row["graph_tokens"],
            "margin_mean": row["margin_mean"],
            "input_preview": str(row.get("input_text", ""))[:240],
            "decoded_preview": str(row.get("decoded_argmax", ""))[:240],
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


def _select_branch_actions(
    action_probs: list[dict[str, Any]],
    branch_factor: int,
    allow_stop: bool = False,
    diverse_actions: bool = True,
    path: list[str] | None = None,
    stochastic: bool = False,
    temperature: float = 1.0,
    exploration: float = 0.0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    branch_factor = max(int(branch_factor), 1)
    path_counts = {action: list(path or []).count(action) for action in ACTION_NAMES}
    candidates = []
    for row in action_probs:
        action = str(row.get("action", ""))
        if action == "stop" and not allow_stop:
            continue
        prob = float(row.get("probability", 0.0))
        repeat_penalty = 0.035 * float(path_counts.get(action, 0))
        candidates.append({**row, "audit_selection_score": prob - repeat_penalty})
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda item: float(item.get("audit_selection_score", 0.0)), reverse=True)
    if stochastic:
        return _sample_branch_actions(
            ranked,
            branch_factor=branch_factor,
            temperature=temperature,
            exploration=exploration,
            seed=seed,
        )
    if not diverse_actions:
        return ranked[:branch_factor]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_action(action_name: str) -> None:
        if len(selected) >= branch_factor or action_name in seen:
            return
        match = next((row for row in ranked if str(row.get("action", "")) == action_name), None)
        if match is not None:
            selected.append(match)
            seen.add(action_name)

    if ranked:
        add_action(str(ranked[0].get("action", "")))
    for action_name in ("expand", "verify", "retrieve", "refine", "merge", "compress", "reject"):
        add_action(action_name)
    for row in ranked:
        add_action(str(row.get("action", "")))
    return selected[:branch_factor] or ranked[:branch_factor]


def _sample_branch_actions(
    ranked: list[dict[str, Any]],
    branch_factor: int,
    temperature: float = 1.0,
    exploration: float = 0.0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if not ranked:
        return []
    branch_factor = max(1, min(int(branch_factor), len(ranked)))
    temperature = max(float(temperature), 1e-4)
    exploration = max(0.0, min(float(exploration), 0.95))
    probs = torch.tensor([max(float(row.get("probability", 0.0)), 1e-9) for row in ranked], dtype=torch.float64)
    logits = torch.log(probs) / temperature
    weights = torch.softmax(logits, dim=0)
    if exploration > 0.0:
        weights = (1.0 - exploration) * weights + exploration / float(weights.numel())
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed) % (2**63 - 1))
    indices = torch.multinomial(weights, num_samples=branch_factor, replacement=False, generator=generator)
    selected = []
    for sample_rank, idx in enumerate(indices.tolist()):
        row = dict(ranked[int(idx)])
        row["sampling_weight"] = float(weights[int(idx)])
        row["sampling_rank"] = sample_rank
        row["audit_selection_score"] = float(row.get("audit_selection_score", row.get("probability", 0.0)))
        selected.append(row)
    return selected


def _branch_sampling_seed(base_seed: int | None, record_id: object, level: int, rank: int, path: object) -> int | None:
    if base_seed is None:
        return None
    digest = hashlib.sha1(f"{base_seed}|{record_id}|{level}|{rank}|{path}".encode("utf-8", "ignore")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _graphcg_projection(model, graph_state: torch.Tensor, top_k: int = 4) -> list[dict[str, Any]]:
    if hasattr(model.graphcg, "effective_directions"):
        dirs = F.normalize(model.graphcg.effective_directions(detach=True), dim=-1)
    else:
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
                "basis": "effective_full_rank_qr" if hasattr(model.graphcg, "effective_directions") else "raw_normalized",
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


def _decode_shifted_bytes(ids: object) -> str:
    if torch.is_tensor(ids):
        values = ids.detach().cpu().reshape(-1).tolist()
    else:
        values = list(ids) if isinstance(ids, (list, tuple)) else []
    raw = bytearray()
    for value in values:
        try:
            token = int(value)
        except Exception:
            continue
        if token <= 0:
            continue
        raw.append(max(0, min(255, token - 1)))
    return bytes(raw).decode("utf-8", "ignore")


def _graph_json_summary(graph_json: dict[str, Any] | None) -> dict[str, Any]:
    graph = graph_json if isinstance(graph_json, dict) else {}
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": sorted({str(node.get("type", "")) for node in nodes if isinstance(node, dict)})[:12],
        "edge_types": sorted({str(edge.get("type", "")) for edge in edges if isinstance(edge, dict)})[:12],
        "nodes_preview": nodes[:6],
        "edges_preview": edges[:8],
    }


def _decoding_order_report(record: GraphRecord) -> dict[str, Any]:
    graph = record.graph_json if isinstance(record.graph_json, dict) else {"nodes": [], "edges": []}
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    report = graph_decoding_order(graph, seed=int(metadata.get("decoding_random_seed", 0) or 0), record_id=record.record_id)
    for key in (
        "decoding_order_kind",
        "decoding_reverse_order_kind",
        "decoding_is_dag",
        "decoding_node_order",
        "decoding_reverse_node_order",
        "decoding_random_seed",
    ):
        if key in metadata:
            report[key] = metadata[key]
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
    node_set = {str(node.get("id", idx)) for idx, node in enumerate(nodes) if isinstance(node, dict)}
    causal_edges: list[dict[str, Any]] = []
    noncausal_edges = 0
    for idx, raw_edge in enumerate(graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []):
        if not isinstance(raw_edge, dict):
            continue
        source = str(raw_edge.get("source", raw_edge.get("src", "")))
        target = str(raw_edge.get("target", raw_edge.get("dst", "")))
        if source not in node_set or target not in node_set:
            continue
        causal = bool(raw_edge.get("causal") is True or raw_edge.get("directed") is True)
        if causal:
            causal_edges.append(
                {
                    "source": source,
                    "target": target,
                    "edge_index": idx,
                    "edge_type": str(raw_edge.get("type", "graph_edge")),
                    "causal": True,
                    "directed": True,
                }
            )
        else:
            noncausal_edges += 1
    def order_edges(order: object, role: str) -> list[dict[str, Any]]:
        if not isinstance(order, list):
            return []
        clean = [str(item) for item in order if str(item) in node_set]
        return [
            {
                "source": clean[idx],
                "target": clean[idx + 1],
                "decoding_step": idx + 1,
                "edge_type": role,
                "causal": bool(report.get("decoding_is_dag")),
                "directed": True,
            }
            for idx in range(max(0, len(clean) - 1))
        ]
    forward = order_edges(report.get("decoding_node_order"), "forward_decoding_order")
    reverse = order_edges(report.get("decoding_reverse_node_order"), "reverse_decoding_order")
    return {
        "source": "GraphRecord.metadata+graph_decoding_order",
        "record_id": record.record_id,
        "decoding_order_kind": str(report.get("decoding_order_kind", "unknown")),
        "decoding_reverse_order_kind": str(report.get("decoding_reverse_order_kind", "unknown")),
        "decoding_is_dag": bool(report.get("decoding_is_dag")),
        "decoding_node_order": list(report.get("decoding_node_order", []) or [])[:256],
        "decoding_reverse_node_order": list(report.get("decoding_reverse_node_order", []) or [])[:256],
        "causal_edges": causal_edges[:512],
        "forward_decoding_edges": forward[:512],
        "reverse_decoding_edges": reverse[:512],
        "noncausal_edge_count": int(noncausal_edges),
        "overlay_policy": "causal DAGs render forward+reverse causal order; cyclic/noncausal graphs render ROAR/random-order forward+reverse decoding edges",
    }


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


def _reasoning_microsteps(record: GraphRecord, action: str, rank: int) -> list[dict[str, str]]:
    prompt = _clip_plain(record.question or record.text, 180)
    answer = _clip_plain(record.answer, 120) or "unknown target"
    templates = {
        "expand": [
            ("parse", "Parse the active prompt and isolate givens, unknowns, and graph constraints."),
            ("subgoal", "Create a local subgoal node whose answer can reduce the current uncertainty."),
            ("proposal", "Propose the next embedding-space state and attach it to the active reasoning frontier."),
        ],
        "merge": [
            ("support", "Read the tropical active-support set and choose compatible predecessor states."),
            ("align", "Align overlapping symbols, graph nodes, and certificate fragments across predecessors."),
            ("commit", "Commit the merged proof state only where the aligned constraints agree."),
        ],
        "refine": [
            ("localize", "Locate the weakest margin, ambiguous token support, or high-NLL subclaim."),
            ("rewrite", "Rewrite that subclaim as a smaller graph operation with explicit dependencies."),
            ("stabilize", "Update the state so its tropical margin and certificate agreement are easier to inspect."),
        ],
        "retrieve": [
            ("query", "Form an analogical memory query from the current filtered simplicial object."),
            ("match", "Retrieve candidate memories with similar persistence, chain-presentation diagnostics, and derived signatures."),
            ("attach", "Attach the retrieved evidence as a separate graph branch for downstream verification."),
        ],
        "verify": [
            ("claim", "Select the current claim and expose the assumptions that support it."),
            ("check", "Check causal, topological, and algebraic consistency against the graph state."),
            ("verdict", "Record a pass, repair, or rejection verdict before the next decoding action."),
        ],
        "compress": [
            ("select", "Select the minimal active subcomplex carrying the useful reasoning state."),
            ("summarize", "Summarize redundant branches while preserving persistent-homology witnesses."),
            ("emit", "Emit a compact state for long-context continuation and BPB-aware decoding."),
        ],
        "reject": [
            ("diagnose", "Identify the contradiction, low reward, or unstable tropical wall crossing."),
            ("mark", "Mark the branch as inactive without deleting its audit trail."),
            ("recover", "Return control to the surviving frontier states."),
        ],
    }
    steps = templates.get(action, templates["expand"])
    return [
        {
            "kind": kind,
            "text": f"{action}:{kind} candidate {rank}. {body} Prompt: {prompt}. Target: {answer}.",
        }
        for kind, body in steps
    ]


def _append_reasoning_trace(text: str, action: str, microsteps: list[dict[str, Any]]) -> str:
    block = "\n".join(
        f"[got:{action}:{int(step.get('microstep_index', idx)) + 1}] {str(step.get('text', ''))}"
        for idx, step in enumerate(microsteps)
    )
    if not block:
        return text
    return f"{text.rstrip()}\n{block}" if text else block


def _clip_plain(value: str, limit: int) -> str:
    clean = " ".join(str(value or "").split())
    return clean if len(clean) <= limit else clean[: max(limit - 1, 0)] + "..."


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
