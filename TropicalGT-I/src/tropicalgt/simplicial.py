from __future__ import annotations

from itertools import combinations
import math
from typing import Any

from .records import GraphRecord


def build_filtered_simplicial_object(record: GraphRecord) -> dict[str, Any]:
    """Build a small filtered simplicial object from a reasoning graph.

    The object is finite and JSON-serializable.  Vertices are graph nodes,
    1-simplices are graph edges, and 2-simplices are directed length-two
    reasoning motifs.  This is a legacy graph-combinatorial helper for records
    that are not being evaluated by the model.  It is not a radius filtration.
    Radius figures and topology/retrieval metrics produced during training,
    evaluation, and inference must use
    :func:`build_embedding_radius_simplicial_object`, which receives actual
    model graph-token embedding vectors/probabilities.
    """
    graph = record.graph_json or {"nodes": [], "edges": []}
    raw_nodes = list(graph.get("nodes", []))
    raw_edges = list(graph.get("edges", []))
    node_index = {str(node.get("id", idx)): idx for idx, node in enumerate(raw_nodes)}

    vertices = []
    for idx, node in enumerate(raw_nodes):
        text = str(node.get("text", ""))
        embedding = _finite_vector(node.get("embedding"))
        vertices.append(
            {
                "simplex": [str(node.get("id", idx))],
                "dimension": 0,
                "filtration": 0.0,
                "type": str(node.get("type", "node")),
                "text": text,
                "embedding": embedding,
                "weight": round(min(len(text), 512) / 512.0, 6),
            }
        )

    valid_edges = []
    edge_distances = []
    embedding_by_node = {
        str(node.get("id", idx)): _finite_vector(node.get("embedding"))
        for idx, node in enumerate(raw_nodes)
    }
    for edge in raw_edges:
        src = str(edge.get("source", edge.get("src", "")))
        dst = str(edge.get("target", edge.get("dst", "")))
        if src not in node_index or dst not in node_index:
            continue
        distance = _embedding_distance(embedding_by_node.get(src), embedding_by_node.get(dst))
        valid_edges.append((edge, src, dst, distance))
        if distance is not None:
            edge_distances.append(distance)
    max_distance = max(edge_distances, default=0.0)
    positive_floor = 1.0 / max(len(raw_nodes), 1)

    one_simplices = []
    adjacency: dict[str, list[str]] = {}
    zero_distance_edges = 0
    for edge, src, dst, distance in valid_edges:
        filt = _edge_filtration(src, dst, node_index, len(raw_nodes), distance, max_distance, positive_floor)
        if distance is not None and distance <= 1e-12:
            zero_distance_edges += 1
        one_simplices.append(
            {
                "simplex": [src, dst],
                "dimension": 1,
                "filtration": round(filt, 6),
                "type": str(edge.get("type", "edge")),
                "filtration_source": (
                    "explicit_edge_distance_radius"
                    if distance is not None
                    else "graph_combinatorial_observed_edge_not_radius"
                ),
                "embedding_distance": None if distance is None else round(distance, 8),
                "weight": 1.0,
            }
        )
        adjacency.setdefault(src, []).append(dst)
    edge_filtration = {tuple(row["simplex"]): float(row["filtration"]) for row in one_simplices}

    two_simplices = []
    seen = set()
    for src, mids in adjacency.items():
        for mid in mids:
            for dst in adjacency.get(mid, []):
                simplex = (src, mid, dst)
                if simplex in seen:
                    continue
                seen.add(simplex)
                filt = max(
                    edge_filtration.get((src, mid), positive_floor),
                    edge_filtration.get((mid, dst), positive_floor),
                    positive_floor,
                )
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
            "filtration_model": "graph_combinatorial_non_radius_legacy",
            "radius_filtration": False,
            "zero_distance_edges": zero_distance_edges,
        },
        "thresholds": thresholds,
        "simplices": simplices,
    }


def build_embedding_radius_simplicial_object(
    record: GraphRecord,
    token_descriptors: list[dict[str, Any]],
    token_embeddings: object,
    token_probabilities: object | None = None,
    max_vertices: int = 160,
    max_two_simplices: int = 12000,
    metric: str = "euclidean",
    probability_temperature: float = 1.0,
) -> dict[str, Any]:
    """Build the actual radius-filtered complex on model graph-token embeddings.

    Vertices are TokenGT-style graph tokens (graph token, node tokens, and edge
    tokens) with their post-projection model embeddings.  The displayed complex
    is a bounded Vietoris-Rips 2-skeleton: all vertices enter at zero, pairwise
    edges enter at normalized radius, and 2-simplices enter at the maximum
    radius of their faces.  Supported metrics are Euclidean distance on the
    embedding vectors and Jensen-Shannon distance on model probability vectors.
    For graph-token complexes the probabilities must be model-predicted
    tropical attention support distributions, i.e. softmax rows of the
    tropical score matrix.  JS mode fails closed when these probabilities are
    absent; embedding-softmax or graph-order substitutes are intentionally not
    used.
    """

    vectors = _coerce_matrix(token_embeddings)
    usable_count = min(len(token_descriptors), len(vectors), int(max_vertices))
    if usable_count <= 0:
        return {
            "record_id": record.record_id,
            "available": False,
            "reason": "model graph-token embeddings unavailable; radius filtration was not computed",
            "summary": {
                "num_vertices": 0,
                "num_edges": 0,
                "num_two_simplices": 0,
                "num_thresholds": 0,
                "filtration_model": "unavailable_no_model_embeddings",
                "radius_filtration": False,
            },
            "thresholds": [],
            "simplices": [],
        }

    descriptors = token_descriptors[:usable_count]
    vectors = vectors[:usable_count]
    metric_name = str(metric or "euclidean").lower().replace("-", "_")
    probability_source = None
    if metric_name in {"js", "jensen", "jensen_shannon", "jensen_shannon_distance"}:
        metric_name = "jensen_shannon"
        supplied_probabilities = _coerce_matrix(token_probabilities)
        if len(supplied_probabilities) < usable_count:
            return {
                "record_id": record.record_id,
                "available": False,
                "reason": "model probability vectors unavailable; Jensen-Shannon radius filtration was not computed",
                "summary": {
                    "num_vertices": 0,
                    "num_edges": 0,
                    "num_two_simplices": 0,
                    "num_thresholds": 0,
                    "filtration_model": "unavailable_no_model_probabilities",
                    "radius_filtration": False,
                    "embedding_metric": "jensen_shannon",
                },
                "thresholds": [],
                "simplices": [],
            }
        probability_vectors = [
            _normalize_probability_vector(supplied_probabilities[idx])
            for idx in range(usable_count)
        ]
        if any(vector is None for vector in probability_vectors):
            return {
                "record_id": record.record_id,
                "available": False,
                "reason": "invalid model probability vectors; Jensen-Shannon radius filtration was not computed",
                "summary": {
                    "num_vertices": 0,
                    "num_edges": 0,
                    "num_two_simplices": 0,
                    "num_thresholds": 0,
                    "filtration_model": "unavailable_invalid_model_probabilities",
                    "radius_filtration": False,
                    "embedding_metric": "jensen_shannon",
                },
                "thresholds": [],
                "simplices": [],
            }
        probability_source = "model_tropical_support_probabilities"
        distance_vectors = probability_vectors
    elif metric_name == "euclidean":
        probability_vectors = [None for _ in vectors]
        distance_vectors = vectors
    else:
        raise ValueError(f"Unsupported embedding radius metric: {metric!r}")

    vertices: list[dict[str, Any]] = []
    labels: list[str] = []
    for idx, (desc, vector) in enumerate(zip(descriptors, vectors, strict=False)):
        label = _token_vertex_label(desc, idx)
        labels.append(label)
        probability = probability_vectors[idx]
        vertices.append(
            {
                "simplex": [label],
                "dimension": 0,
                "filtration": 0.0,
                "type": f"tokengt_{desc.get('kind', 'token')}",
                "token_index": int(desc.get("index", idx) or idx),
                "token_kind": str(desc.get("kind", "token")),
                "node_id": desc.get("node_id"),
                "edge_type": desc.get("edge_type"),
                "source": desc.get("source"),
                "target": desc.get("target"),
                "endpoint_ids": desc.get("endpoint_ids", []),
                "text": str(desc.get("text", "")),
                "embedding": [float(v) for v in vector],
                "probability": None if probability is None else [float(v) for v in probability],
                "probability_source": probability_source,
                "filtration_source": f"model_graph_token_embedding_{metric_name}",
                "weight": round(1.0 / (1.0 + float(_vector_norm(vector))), 6),
            }
        )

    pair_distances: dict[tuple[int, int], float] = {}
    distances: list[float] = []
    for i, j in combinations(range(usable_count), 2):
        dist = (
            _jensen_shannon_distance(distance_vectors[i], distance_vectors[j])
            if metric_name == "jensen_shannon"
            else _embedding_distance(distance_vectors[i], distance_vectors[j])
        )
        if dist is None:
            continue
        pair_distances[(i, j)] = dist
        distances.append(dist)
    max_distance = max(distances, default=0.0)
    one_simplices: list[dict[str, Any]] = []
    edge_filtration: dict[tuple[int, int], float] = {}
    zero_distance_edges = 0
    for (i, j), dist in sorted(pair_distances.items(), key=lambda item: (item[1], item[0])):
        filt = _normalized_radius(dist, max_distance, positive_floor=1.0 / max(usable_count, 1))
        if dist <= 1e-12:
            zero_distance_edges += 1
        edge_filtration[(i, j)] = filt
        one_simplices.append(
            {
                "simplex": [labels[i], labels[j]],
                "dimension": 1,
                "filtration": round(filt, 8),
                "type": f"{metric_name}_embedding_vietoris_rips_edge",
                "filtration_source": f"model_graph_token_embedding_{metric_name}_radius",
                "embedding_distance": round(dist, 8),
                "embedding_metric": metric_name,
                "weight": 1.0,
            }
        )

    two_simplices: list[dict[str, Any]] = []
    skipped_two_simplices = 0
    for i, j, k in combinations(range(usable_count), 3):
        keys = [(min(a, b), max(a, b)) for a, b in ((i, j), (i, k), (j, k))]
        if not all(key in edge_filtration for key in keys):
            continue
        filt = max(edge_filtration[key] for key in keys)
        if len(two_simplices) >= int(max_two_simplices):
            skipped_two_simplices += 1
            continue
        two_simplices.append(
            {
                "simplex": [labels[i], labels[j], labels[k]],
                "dimension": 2,
                "filtration": round(filt, 8),
                "type": f"{metric_name}_embedding_vietoris_rips_2simplex",
                "filtration_source": f"model_graph_token_embedding_{metric_name}_radius_clique",
                "embedding_metric": metric_name,
                "weight": 1.0,
            }
        )

    simplices = vertices + one_simplices + two_simplices
    thresholds = sorted({float(s["filtration"]) for s in simplices})
    return {
        "record_id": record.record_id,
        "available": True,
        "summary": {
            "num_vertices": len(vertices),
            "num_edges": len(one_simplices),
            "num_two_simplices": len(two_simplices),
            "num_thresholds": len(thresholds),
            "filtration_model": (
                "model_tropical_support_probability_jensen_shannon_vietoris_rips_2_skeleton"
                if metric_name == "jensen_shannon"
                else "model_graph_token_embedding_vietoris_rips_2_skeleton"
            ),
            "radius_filtration": True,
            "embedding_source": "TropicalGTModel.graph_token_embeddings",
            "embedding_metric": metric_name,
            "probability_transform": (
                {
                    "kind": "model_supplied",
                    "source": probability_source,
                    "temperature": None,
                }
                if metric_name == "jensen_shannon"
                else None
            ),
            "embedding_dim": len(vectors[0]) if vectors else 0,
            "truncated_vertices": max(0, len(token_descriptors) - usable_count),
            "truncated_two_simplices": int(skipped_two_simplices),
            "zero_distance_edges": zero_distance_edges,
        },
        "thresholds": thresholds,
        "simplices": simplices,
    }


def build_reasoning_trajectory_complex(
    candidates: list[dict[str, Any]],
    up_to_level: int | None = None,
    metric: str = "euclidean",
    probability_temperature: float = 1.0,
) -> dict[str, Any]:
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
    metric_name = str(metric or "euclidean").lower().replace("-", "_")
    if metric_name in {"js", "jensen", "jensen_shannon", "jensen_shannon_distance"}:
        metric_name = "jensen_shannon"
    elif metric_name != "euclidean":
        raise ValueError(f"Unsupported trajectory complex metric: {metric!r}")
    raw_embedding_by_id = {record_id: _finite_vector(row.get("embedding")) for record_id, row in usable}
    raw_probability_by_id = {
        record_id: _candidate_probability_vector(row)
        for record_id, row in usable
    }
    if metric_name == "jensen_shannon":
        probability_source_by_id = {}
        embedding_by_id = {}
        for record_id, vector in raw_embedding_by_id.items():
            supplied = raw_probability_by_id.get(record_id)
            embedding_by_id[record_id] = supplied
            probability_source_by_id[record_id] = "TropicalGTModel.gfn(graph_state).softmax_action_probability_vector" if supplied is not None else None
    else:
        embedding_by_id = raw_embedding_by_id
        probability_source_by_id = {record_id: None for record_id in raw_embedding_by_id}
    for order, (record_id, row) in enumerate(usable):
        level = int(row.get("level", 0) or 0)
        score = float(row.get("score", 0.0) or 0.0)
        nll = float(row.get("nll", 0.0) or 0.0)
        vertices.append(
            {
                "simplex": [record_id],
                "dimension": 0,
                "filtration": 0.0,
                "type": "got_state",
                "level": level,
                "score": score,
                "nll": nll,
                "path": row.get("path", []),
                "embedding": row.get("embedding", []),
                "probability": (
                    [float(v) for v in embedding_by_id.get(record_id)]
                    if metric_name == "jensen_shannon" and embedding_by_id.get(record_id) is not None
                    else None
                ),
                "probability_source": probability_source_by_id.get(record_id),
                "input_text": row.get("input_text", ""),
                "target_text": row.get("target_text", ""),
                "decoded_argmax": row.get("decoded_argmax", ""),
                "graph_json_summary": row.get("graph_json_summary", {}),
                "filtered_simplicial_object": row.get("filtered_simplicial_object", {}),
                "topological_algebra": row.get("topological_algebra", {}),
                "weight": round(1.0 / (1.0 + max(nll, 0.0)), 6),
            }
        )

    one_simplices = []
    parent_lookup: dict[str, str] = {}
    row_lookup = {rid: row for rid, row in usable}
    edge_rows: list[dict[str, Any]] = []
    edge_distance_values: list[float] = []

    if any(embedding_by_id.get(record_id) is not None for record_id in ids):
        for a, b in combinations(ids, 2):
            distance = (
                _jensen_shannon_distance(embedding_by_id.get(a), embedding_by_id.get(b))
                if metric_name == "jensen_shannon"
                else _embedding_distance(embedding_by_id.get(a), embedding_by_id.get(b))
            )
            if distance is None:
                continue
            edge_distance_values.append(distance)
            edge_rows.append(
                {
                    "source": a,
                    "target": b,
                    "distance": distance,
                    "type": f"{metric_name}_embedding_vietoris_rips_edge",
                    "reasoning_transition": False,
                }
            )

    for record_id, row in usable:
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_set:
            parent_lookup[record_id] = parent
            action_path = row.get("path", [])
            action = action_path[-1] if isinstance(action_path, list) and action_path else "transition"
            if edge_rows:
                for edge in edge_rows:
                    if {edge["source"], edge["target"]} == {parent, record_id}:
                        edge["type"] = f"got_transition_{action}_rips_edge"
                        edge["reasoning_transition"] = True
                        break

    max_edge_distance = max(edge_distance_values, default=0.0)
    positive_floor = 1.0 / max(len(ids), 1)
    edge_filtration: dict[tuple[str, str], float] = {}
    zero_distance_edges = 0
    for edge in edge_rows:
        src = str(edge["source"])
        dst = str(edge["target"])
        distance = edge.get("distance")
        if isinstance(distance, (int, float)) and math.isfinite(float(distance)):
            filt = _normalized_radius(float(distance), max_edge_distance, positive_floor)
            if float(distance) <= 1e-12:
                zero_distance_edges += 1
        else:
            continue
        key = tuple(sorted((src, dst)))
        if key in edge_filtration and edge_filtration[key] <= filt:
            continue
        edge_filtration[key] = filt

    for edge in edge_rows:
        src = str(edge["source"])
        dst = str(edge["target"])
        key = tuple(sorted((src, dst)))
        filt = edge_filtration.get(key)
        if filt is None:
            continue
        one_simplices.append(
                {
                    "simplex": [src, dst],
                    "dimension": 1,
                    "filtration": round(filt, 6),
                    "type": str(edge.get("type", f"{metric_name}_embedding_vietoris_rips_edge")),
                    "filtration_source": (
                        f"{metric_name}_probability_radius"
                        if isinstance(edge.get("distance"), (int, float)) and metric_name == "jensen_shannon"
                        else (
                            "embedding_radius"
                            if isinstance(edge.get("distance"), (int, float))
                            else "unavailable_no_radius_distance"
                        )
                    ),
                    "embedding_distance": None if edge.get("distance") is None else round(float(edge["distance"]), 8),
                    "embedding_metric": metric_name if isinstance(edge.get("distance"), (int, float)) else None,
                    "reasoning_transition": bool(edge.get("reasoning_transition", False)),
                    "weight": 1.0,
                }
            )

    two_simplices = []
    if edge_filtration and len(ids) <= 96:
        for a, b, c in combinations(ids, 3):
            keys = [tuple(sorted(pair)) for pair in ((a, b), (a, c), (b, c))]
            if not all(key in edge_filtration for key in keys):
                continue
            two_simplices.append(
                {
                    "simplex": [a, b, c],
                    "dimension": 2,
                    "filtration": round(max(edge_filtration[key] for key in keys), 6),
                    "type": f"{metric_name}_embedding_vietoris_rips_2simplex",
                    "filtration_source": f"{metric_name}_embedding_radius_clique",
                    "embedding_metric": metric_name,
                    "weight": 1.0,
                }
            )

    simplices = vertices + one_simplices + two_simplices
    thresholds = sorted({simplex["filtration"] for simplex in simplices})
    unavailable_reason = None if edge_distance_values else "unavailable_no_embedding_or_probability_radius_edges"
    return {
        "record_id": "graph_of_thought_trajectory",
        "available": unavailable_reason is None,
        **({"reason": unavailable_reason} if unavailable_reason else {}),
        "summary": {
            "num_vertices": len(vertices),
            "num_edges": len(one_simplices),
            "num_two_simplices": len(two_simplices),
            "num_thresholds": len(thresholds),
            "max_level": max((int(row.get("level", 0) or 0) for _, row in usable), default=0),
            "filtration_model": (
                "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton"
                if edge_distance_values and metric_name == "jensen_shannon"
                else (
                    "embedding_vietoris_rips_2_skeleton"
                    if edge_distance_values
                    else "unavailable_no_embedding_or_probability_radius_edges"
                )
            ),
            "zero_distance_edges": zero_distance_edges,
            "embedding_edge_count": len(edge_distance_values),
            "embedding_metric": metric_name if edge_distance_values else None,
            "probability_transform": (
                {
                    "kind": "model_candidate_probability_vector",
                    "temperature": None,
                }
                if edge_distance_values and metric_name == "jensen_shannon"
                else None
            ),
            "reasoning_transition_edge_count": sum(1 for row in one_simplices if row.get("reasoning_transition")),
        },
        "thresholds": thresholds,
        "simplices": simplices,
    }


def _trajectory_filtration(level: int, score: float, order: int) -> float:
    level_term = max(level, 0) / 8.0
    score_term = 1.0 / (1.0 + np_exp_safe(score))
    order_term = min(order, 128) / 1024.0
    return max(0.0, min(1.0, 0.72 * level_term + 0.25 * score_term + 0.03 * order_term))


def _candidate_probability_vector(row: dict[str, Any]) -> list[float] | None:
    raw = row.get("probability")
    normalized = _normalize_probability_vector(_finite_vector(raw))
    if normalized is not None:
        return normalized
    raw = row.get("action_probability_vector")
    normalized = _normalize_probability_vector(_finite_vector(raw))
    if normalized is not None:
        return normalized
    action_probs = row.get("gflownet_action_probs")
    if isinstance(action_probs, list) and action_probs:
        values = []
        for item in action_probs:
            if isinstance(item, dict):
                value = item.get("probability")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
        normalized = _normalize_probability_vector(values)
        if normalized is not None:
            return normalized
    return None


def np_exp_safe(value: float) -> float:
    import math

    return math.exp(max(min(value, 30.0), -30.0))


def _finite_vector(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[float] = []
    for item in value:
        try:
            val = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val):
            return None
        out.append(val)
    return out or None


def _coerce_matrix(value: object) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    for row in value:
        vector = _finite_vector(row)
        if vector is not None:
            rows.append(vector)
    return rows


def _embedding_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _softmax_probability_vector(value: list[float] | None, temperature: float = 1.0) -> list[float] | None:
    if not value:
        return None
    temp = max(float(temperature), 1e-6)
    scaled = [float(v) / temp for v in value]
    shift = max(scaled)
    exps = [math.exp(max(min(v - shift, 60.0), -60.0)) for v in scaled]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        return None
    return [v / total for v in exps]


def _normalize_probability_vector(value: list[float] | None) -> list[float] | None:
    if not value:
        return None
    vals = [max(float(v), 0.0) for v in value if math.isfinite(float(v))]
    if not vals:
        return None
    total = sum(vals)
    if total <= 0.0 or not math.isfinite(total):
        return None
    return [float(v / total) for v in vals]


def _jensen_shannon_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    eps = 1e-12
    pa = [max(float(v), eps) for v in a]
    pb = [max(float(v), eps) for v in b]
    sa = sum(pa)
    sb = sum(pb)
    if sa <= 0.0 or sb <= 0.0:
        return None
    pa = [v / sa for v in pa]
    pb = [v / sb for v in pb]
    m = [0.5 * (x + y) for x, y in zip(pa, pb)]

    def kl(p: list[float], q: list[float]) -> float:
        return sum(pi * math.log(pi / max(qi, eps)) for pi, qi in zip(p, q))

    js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    return math.sqrt(max(js, 0.0))


def _vector_norm(value: list[float]) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in value))


def _token_vertex_label(desc: dict[str, Any], idx: int) -> str:
    kind = str(desc.get("kind", "token"))
    if kind == "graph":
        base = "graph"
    elif kind == "node":
        base = str(desc.get("node_id") or desc.get("label") or f"node_{idx}")
    elif kind == "edge":
        src = str(desc.get("source", "src"))
        dst = str(desc.get("target", "dst"))
        etype = str(desc.get("edge_type", "edge"))
        base = f"{src}->{dst}:{etype}"
    else:
        base = str(desc.get("label") or f"token_{idx}")
    safe = base.replace(" ", "_")
    return f"{idx}:{kind}:{safe}"


def _normalized_radius(distance: float, max_distance: float, positive_floor: float) -> float:
    if max_distance <= 1e-12:
        return 0.0 if distance <= 1e-12 else positive_floor
    return max(0.0, min(1.0, distance / max_distance))


def _edge_filtration(
    src: str,
    dst: str,
    node_index: dict[str, int],
    node_count: int,
    distance: float | None,
    max_distance: float,
    positive_floor: float,
) -> float:
    if distance is not None:
        return _normalized_radius(distance, max_distance, positive_floor)
    # Observed graph edges without endpoint embeddings are not a radius
    # filtration.  They enter only at the final diagnostic threshold and are
    # labeled as graph-combinatorial, so training/inference topology cannot
    # silently treat graph order as geometry.
    return 1.0
