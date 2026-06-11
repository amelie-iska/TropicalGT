from __future__ import annotations

from typing import Any

from .records import GraphRecord


def build_filtered_simplicial_object(record: GraphRecord) -> dict[str, Any]:
    """Build a small filtered simplicial object from a reasoning graph.

    The v1 object is intentionally finite and JSON-serializable.  Vertices are
    graph nodes, 1-simplices are graph edges, and 2-simplices are directed
    length-two reasoning motifs.  Filtration values are deterministic proxies
    derived from node order and local text/provenance, so the object can be
    hovered, stored, and compared without requiring a persistent homology
    backend during smoke runs.
    """
    graph = record.graph_json or {"nodes": [], "edges": []}
    raw_nodes = list(graph.get("nodes", []))
    raw_edges = list(graph.get("edges", []))
    node_index = {str(node.get("id", idx)): idx for idx, node in enumerate(raw_nodes)}

    vertices = []
    for idx, node in enumerate(raw_nodes):
        text = str(node.get("text", ""))
        vertices.append(
            {
                "simplex": [str(node.get("id", idx))],
                "dimension": 0,
                "filtration": round(idx / max(len(raw_nodes), 1), 6),
                "type": str(node.get("type", "node")),
                "text": text,
                "weight": round(min(len(text), 512) / 512.0, 6),
            }
        )

    one_simplices = []
    adjacency: dict[str, list[str]] = {}
    for edge in raw_edges:
        src = str(edge.get("source", edge.get("src", "")))
        dst = str(edge.get("target", edge.get("dst", "")))
        if src not in node_index or dst not in node_index:
            continue
        filt = max(node_index[src], node_index[dst]) / max(len(raw_nodes), 1)
        one_simplices.append(
            {
                "simplex": [src, dst],
                "dimension": 1,
                "filtration": round(filt, 6),
                "type": str(edge.get("type", "edge")),
                "weight": 1.0,
            }
        )
        adjacency.setdefault(src, []).append(dst)

    two_simplices = []
    seen = set()
    for src, mids in adjacency.items():
        for mid in mids:
            for dst in adjacency.get(mid, []):
                simplex = (src, mid, dst)
                if simplex in seen:
                    continue
                seen.add(simplex)
                filt = max(node_index[src], node_index[mid], node_index[dst]) / max(len(raw_nodes), 1)
                two_simplices.append(
                    {
                        "simplex": list(simplex),
                        "dimension": 2,
                        "filtration": round(min(filt + 0.05, 1.0), 6),
                        "type": "directed_path_2",
                        "weight": 1.0,
                    }
                )

    simplices = vertices + one_simplices + two_simplices
    thresholds = sorted({s["filtration"] for s in simplices})
    return {
        "record_id": record.record_id,
        "summary": {
            "num_vertices": len(vertices),
            "num_edges": len(one_simplices),
            "num_two_simplices": len(two_simplices),
            "num_thresholds": len(thresholds),
        },
        "thresholds": thresholds,
        "simplices": simplices,
    }


def build_reasoning_trajectory_complex(candidates: list[dict[str, Any]], up_to_level: int | None = None) -> dict[str, Any]:
    """Build a filtered complex whose points are graph-of-thought states."""

    usable = []
    for idx, row in enumerate(candidates):
        level = int(row.get("level", 0) or 0)
        if up_to_level is not None and level > up_to_level:
            continue
        record_id = str(row.get("record_id", f"candidate-{idx}"))
        usable.append((record_id, row))

    ids = [rid for rid, _ in usable]
    id_set = set(ids)
    vertices = []
    for order, (record_id, row) in enumerate(usable):
        level = int(row.get("level", 0) or 0)
        score = float(row.get("score", 0.0) or 0.0)
        nll = float(row.get("nll", 0.0) or 0.0)
        vertices.append(
            {
                "simplex": [record_id],
                "dimension": 0,
                "filtration": round(_trajectory_filtration(level, score, order), 6),
                "type": "got_state",
                "level": level,
                "score": score,
                "nll": nll,
                "path": row.get("path", []),
                "embedding": row.get("embedding", []),
                "input_text": row.get("input_text", ""),
                "target_text": row.get("target_text", ""),
                "decoded_argmax": row.get("decoded_argmax", ""),
                "graph_json_summary": row.get("graph_json_summary", {}),
                "weight": round(1.0 / (1.0 + max(nll, 0.0)), 6),
            }
        )

    one_simplices = []
    parent_lookup: dict[str, str] = {}
    row_lookup = {rid: row for rid, row in usable}
    for record_id, row in usable:
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_set:
            parent_lookup[record_id] = parent
            child_level = int(row.get("level", 0) or 0)
            parent_level = int(row_lookup[parent].get("level", 0) or 0)
            action_path = row.get("path", [])
            action = action_path[-1] if isinstance(action_path, list) and action_path else "transition"
            one_simplices.append(
                {
                    "simplex": [parent, record_id],
                    "dimension": 1,
                    "filtration": round(max(_trajectory_filtration(parent_level, float(row_lookup[parent].get("score", 0.0) or 0.0), 0), _trajectory_filtration(child_level, float(row.get("score", 0.0) or 0.0), 0)), 6),
                    "type": f"got_{action}",
                    "weight": 1.0,
                }
            )

    two_simplices = []
    for child, parent in parent_lookup.items():
        grandparent = parent_lookup.get(parent)
        if grandparent is None:
            continue
        rows = [row_lookup[grandparent], row_lookup[parent], row_lookup[child]]
        filt = max(
            _trajectory_filtration(int(row.get("level", 0) or 0), float(row.get("score", 0.0) or 0.0), 0)
            for row in rows
        )
        two_simplices.append(
            {
                "simplex": [grandparent, parent, child],
                "dimension": 2,
                "filtration": round(min(filt + 0.05, 1.0), 6),
                "type": "got_length_two_path",
                "weight": 1.0,
            }
        )

    simplices = vertices + one_simplices + two_simplices
    thresholds = sorted({simplex["filtration"] for simplex in simplices})
    return {
        "record_id": "graph_of_thought_trajectory",
        "summary": {
            "num_vertices": len(vertices),
            "num_edges": len(one_simplices),
            "num_two_simplices": len(two_simplices),
            "num_thresholds": len(thresholds),
            "max_level": max((int(row.get("level", 0) or 0) for _, row in usable), default=0),
        },
        "thresholds": thresholds,
        "simplices": simplices,
    }


def _trajectory_filtration(level: int, score: float, order: int) -> float:
    level_term = max(level, 0) / 8.0
    score_term = 1.0 / (1.0 + np_exp_safe(score))
    order_term = min(order, 128) / 1024.0
    return max(0.0, min(1.0, 0.72 * level_term + 0.25 * score_term + 0.03 * order_term))


def np_exp_safe(value: float) -> float:
    import math

    return math.exp(max(min(value, 30.0), -30.0))
