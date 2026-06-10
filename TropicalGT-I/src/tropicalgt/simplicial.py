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

