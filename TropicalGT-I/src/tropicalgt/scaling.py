from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

import torch
import torch.nn.functional as F

from .data import encode_bytes
from .algebra import compute_topological_algebra_report
from .diagnostics import ACTION_NAMES, graph_token_trace, per_record_nll, record_diagnostics
from .records import GraphRecord, GraphTokenBatch, graph_decoding_order
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
        "allow_stop": bool(allow_stop),
        "diverse_actions": bool(diverse_actions),
        "stochastic_actions": bool(stochastic_actions),
        "sampling_temperature": float(sampling_temperature),
        "sampling_exploration": float(sampling_exploration),
        "sampling_seed": sampling_seed,
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
        "input_text": row.get("input_text", ""),
        "target_text": row.get("target_text", ""),
        "decoded_argmax": row.get("decoded_argmax", ""),
        "graph_json_summary": row.get("graph_json_summary", {}),
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
        fallback = next((row for row in action_probs if str(row.get("action", "")) != "stop"), None)
        return [fallback or {"action": "expand", "probability": 1.0, "audit_selection_score": 1.0}]
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
            ("match", "Retrieve candidate memories with similar persistence, free-resolution, and derived signatures."),
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
