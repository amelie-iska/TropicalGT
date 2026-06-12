from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Iterable

import torch


@dataclass
class GraphRecord:
    record_id: str
    text: str
    question: str = ""
    answer: str = ""
    reasoning: str = ""
    metadata: dict[str, Any] | None = None
    graph_json: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any], index: int = 0) -> "GraphRecord":
        rid = str(row.get("record_id") or row.get("id") or f"record-{index}")
        question = _string(row.get("question"))
        answer = _string(row.get("answer"))
        reasoning = _string(row.get("reasoning") or row.get("solution"))
        text = _string(row.get("text")) or _join_nonempty([question, reasoning, answer])
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        graph_obj = _parse_graph(row.get("graph_json"))
        if graph_obj is None:
            graph_obj = conservative_graph(question=question, reasoning=reasoning, answer=answer, text=text)
            metadata["graph_json_fallback"] = True
        else:
            metadata["graph_json_fallback"] = False
        graph_obj, sequence_added = attach_sequential_text_graph(graph_obj, text=text, question=question, answer=answer)
        graph_obj, causal_report = normalize_graph_causal_structure(graph_obj)
        metadata["graph_json_sequentialized"] = sequence_added
        metadata.update(causal_report)
        metadata.update(graph_decoding_order(graph_obj, seed=0, record_id=rid))
        if "dataset" in row:
            metadata["dataset"] = _string(row.get("dataset"))
        return cls(rid, text, question, answer, reasoning, metadata, graph_obj)

    def to_hover_html(self) -> str:
        graph = self.graph_json or {}
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        return (
            f"<b>{self.record_id}</b><br>nodes={len(nodes)} edges={len(edges)}"
            f"<br><b>question</b>: {_clip(self.question or self.text, 360)}"
            f"<br><b>answer</b>: {_clip(self.answer, 240)}"
            f"<br><b>graph</b>: {_clip(json.dumps(graph, ensure_ascii=False), 900)}"
        )

    def autoregressive_text(self, seed: int = 0, max_node_chars: int = 4096, direction: str = "forward") -> str:
        """Flatten node payloads in the graph-aware autoregressive order."""

        graph = self.graph_json or {}
        nodes = list(graph.get("nodes", []))
        if not nodes:
            return self.text
        node_by_id = {str(node.get("id", idx)): node for idx, node in enumerate(nodes)}
        metadata = self.metadata if isinstance(self.metadata, dict) else {}
        reverse = str(direction).lower() in {"reverse", "backward", "right_to_left", "rtl"}
        order_key = "decoding_reverse_node_order" if reverse else "decoding_node_order"
        order = metadata.get(order_key)
        if not isinstance(order, list):
            order_report = graph_decoding_order(graph, seed=seed, record_id=self.record_id)
            order = order_report["decoding_reverse_node_order" if reverse else "decoding_node_order"]
        parts: list[str] = []
        for node_id in order:
            node = node_by_id.get(str(node_id))
            if not node:
                continue
            text = _string(node.get("text") or node.get("label") or node.get("value")).strip()
            if text:
                kind = _string(node.get("type") or "node").strip()
                parts.append(f"[{kind}] {text[:max_node_chars]}")
        return "\n".join(parts) if parts else self.text


@dataclass
class GraphTokenBatch:
    token_features: torch.Tensor
    token_type_ids: torch.Tensor
    endpoint_ids: torch.Tensor
    attention_mask: torch.Tensor
    graph_token_counts: torch.Tensor
    node_counts: torch.Tensor
    edge_counts: torch.Tensor
    hover_payloads: list[str] | None = None

    def to(self, device: torch.device | str) -> "GraphTokenBatch":
        return GraphTokenBatch(
            token_features=self.token_features.to(device),
            token_type_ids=self.token_type_ids.to(device),
            endpoint_ids=self.endpoint_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            graph_token_counts=self.graph_token_counts.to(device),
            node_counts=self.node_counts.to(device),
            edge_counts=self.edge_counts.to(device),
            hover_payloads=self.hover_payloads,
        )


def conservative_graph(question: str = "", reasoning: str = "", answer: str = "", text: str = "") -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    def add(kind: str, label: str) -> str:
        node_id = f"{kind}-{len(nodes)}"
        nodes.append({"id": node_id, "type": kind, "text": label})
        return node_id
    problem = add("problem", question or text[:512] or "empty problem")
    prev = problem
    steps = [s.strip() for s in reasoning.replace("\r", "\n").split("\n") if s.strip()]
    for step in steps[:24]:
        sid = add("reasoning_step", step)
        edges.append({"source": prev, "target": sid, "type": "depends_on", "directed": True, "causal": True})
        prev = sid
    ans = add("answer", answer or "")
    edges.append({"source": prev, "target": ans, "type": "supports_answer", "directed": True, "causal": True})
    return {"nodes": nodes, "edges": edges}


def attach_sequential_text_graph(
    graph: dict[str, Any] | None,
    text: str = "",
    question: str = "",
    answer: str = "",
    max_chunks: int = 24,
    chunk_chars: int = 220,
) -> tuple[dict[str, Any], bool]:
    """Attach a bounded path graph for the underlying text sequence."""

    graph_obj = graph if isinstance(graph, dict) else {"nodes": [], "edges": []}
    nodes = list(graph_obj.get("nodes", []))
    edges = list(graph_obj.get("edges", []))
    if any(str(node.get("type", "")) == "sequence_chunk" for node in nodes):
        return {**graph_obj, "nodes": nodes, "edges": edges}, False

    source_text = text or question or answer
    chunks = _text_chunks(source_text, chunk_chars=chunk_chars, max_chunks=max_chunks)
    if not chunks:
        return {**graph_obj, "nodes": nodes, "edges": edges}, False

    existing_ids = {str(node.get("id", idx)) for idx, node in enumerate(nodes)}
    prefix = "seq"
    while f"{prefix}_root" in existing_ids:
        prefix = f"{prefix}_x"

    root_id = f"{prefix}_root"
    nodes.append({"id": root_id, "type": "sequence_document", "text": _clip(source_text, 512)})
    previous = root_id
    for idx, chunk in enumerate(chunks):
        node_id = f"{prefix}_{idx:03d}"
        nodes.append({"id": node_id, "type": "sequence_chunk", "text": chunk, "position": idx})
        edges.append({
            "source": previous,
            "target": node_id,
            "type": "next_text_chunk" if previous != root_id else "starts_text",
            "directed": True,
            "causal": True,
        })
        previous = node_id

    return {**graph_obj, "nodes": nodes, "edges": edges}, True


CAUSAL_EDGE_TYPES = {
    "depends_on",
    "supports_answer",
    "starts_text",
    "next_text_chunk",
    "precedes",
    "causes",
    "causal",
    "entails",
    "derives",
    "implies",
    "leads_to",
    "parent_of",
    "step_to",
    "then",
    "before",
    "after",
    "control_flow",
    "data_flow",
}

NONCAUSAL_EDGE_TYPES = {
    "undirected",
    "noncausal",
    "cooccurs",
    "co_occurs",
    "similar",
    "related",
    "adjacent",
    "sibling",
    "contrastive_pair",
    "correlates",
    "analogy",
    "retrieves",
    "memory_match",
    "nearest_neighbor",
    "same_as",
}


def normalize_graph_causal_structure(graph: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Annotate dataset graph edges with explicit causal/noncausal semantics."""

    graph_obj = graph if isinstance(graph, dict) else {"nodes": [], "edges": []}
    nodes = list(graph_obj.get("nodes", []))
    edges: list[dict[str, Any]] = []
    inferred_causal = 0
    explicit_causal = 0
    explicit_noncausal = 0
    unknown = 0
    for raw_edge in list(graph_obj.get("edges", [])):
        if not isinstance(raw_edge, dict):
            continue
        edge = dict(raw_edge)
        kind = _edge_kind(edge)
        if edge.get("causal") is True or edge.get("directed") is True:
            edge["causal"] = bool(edge.get("causal", True))
            edge["directed"] = bool(edge.get("directed", True))
            explicit_causal += int(edge["causal"])
            explicit_noncausal += int(not edge["causal"])
        elif edge.get("causal") is False or edge.get("directed") is False or kind in NONCAUSAL_EDGE_TYPES:
            edge["causal"] = False
            edge["directed"] = False
            explicit_noncausal += 1
        elif kind in CAUSAL_EDGE_TYPES or _edge_has_temporal_positions(edge):
            edge["causal"] = True
            edge["directed"] = True
            edge["causal_inferred_from"] = kind or "temporal_position"
            inferred_causal += 1
        else:
            edge.setdefault("causal", False)
            edge.setdefault("directed", False)
            unknown += 1
        edges.append(edge)
    causal_edges = sum(1 for edge in edges if edge.get("causal") is True)
    report = {
        "causal_edge_count": int(causal_edges),
        "noncausal_edge_count": int(sum(1 for edge in edges if edge.get("causal") is False)),
        "causal_edge_inferred_count": int(inferred_causal),
        "causal_edge_explicit_count": int(explicit_causal),
        "causal_edge_unknown_count": int(unknown),
        "graph_has_explicit_causal_structure": bool(causal_edges > 0),
    }
    return {**graph_obj, "nodes": nodes, "edges": edges}, report


def graph_decoding_order(graph: dict[str, Any], seed: int = 0, record_id: str = "") -> dict[str, Any]:
    """Select causal-DAG or random-order autoregressive decoding semantics."""

    nodes = list(graph.get("nodes", [])) if isinstance(graph, dict) else []
    edges = list(graph.get("edges", [])) if isinstance(graph, dict) else []
    node_ids = [str(node.get("id", idx)) for idx, node in enumerate(nodes)]
    node_set = set(node_ids)
    if not node_ids:
        return {
            "decoding_order_kind": "empty_graph",
            "decoding_reverse_order_kind": "empty_graph",
            "decoding_is_dag": True,
            "decoding_node_order": [],
            "decoding_reverse_node_order": [],
            "decoding_random_seed": int(seed),
        }

    causal_edges: list[tuple[str, str]] = []
    has_noncausal_edge = False
    for edge in edges:
        src = str(edge.get("source", edge.get("src", "")))
        dst = str(edge.get("target", edge.get("dst", "")))
        if src not in node_set or dst not in node_set:
            continue
        if _edge_is_noncausal(edge):
            has_noncausal_edge = True
            continue
        causal_edges.append((src, dst))

    if causal_edges and not has_noncausal_edge:
        order, is_dag = _topological_order(node_ids, causal_edges)
        if is_dag:
            return {
                "decoding_order_kind": "causal_dag",
                "decoding_reverse_order_kind": "reverse_causal_dag",
                "decoding_is_dag": True,
                "decoding_node_order": order,
                "decoding_reverse_node_order": list(reversed(order)),
                "decoding_random_seed": int(seed),
            }

    order = list(node_ids)
    rng = random.Random(_stable_seed(seed, record_id, node_ids, edges))
    rng.shuffle(order)
    return {
        "decoding_order_kind": "random_autoregressive",
        "decoding_reverse_order_kind": "reverse_random_autoregressive",
        "decoding_is_dag": False,
        "decoding_node_order": order,
        "decoding_reverse_node_order": list(reversed(order)),
        "decoding_random_seed": int(seed),
    }


def _parse_graph(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _edge_is_noncausal(edge: dict[str, Any]) -> bool:
    if edge.get("directed") is False or edge.get("causal") is False:
        return True
    return _edge_kind(edge) in NONCAUSAL_EDGE_TYPES


def _edge_kind(edge: dict[str, Any]) -> str:
    return _string(edge.get("type") or edge.get("relation") or edge.get("label") or "").strip().lower()


def _edge_has_temporal_positions(edge: dict[str, Any]) -> bool:
    temporal_keys = ("position", "step", "time", "source_position", "target_position", "src_position", "dst_position")
    return any(key in edge for key in temporal_keys)


def _topological_order(node_ids: list[str], edges: list[tuple[str, str]]) -> tuple[list[str], bool]:
    adjacency = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for src, dst in edges:
        adjacency[src].append(dst)
        indegree[dst] += 1
    ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for dst in adjacency[node_id]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
    return order, len(order) == len(node_ids)


def _stable_seed(seed: int, record_id: str, node_ids: list[str], edges: list[dict[str, Any]]) -> int:
    payload = json.dumps(
        {"seed": int(seed), "record_id": record_id, "nodes": node_ids, "edges": edges},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.blake2b(payload.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _text_chunks(text: str, chunk_chars: int, max_chunks: int) -> list[str]:
    clean = " ".join((text or "").replace("\r", "\n").split())
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean) and len(chunks) < max_chunks:
        stop = min(len(clean), start + chunk_chars)
        if stop < len(clean):
            boundary = clean.rfind(" ", start, stop)
            if boundary > start + chunk_chars // 2:
                stop = boundary
        chunks.append(clean[start:stop].strip())
        start = stop
        while start < len(clean) and clean[start].isspace():
            start += 1
    return [chunk for chunk in chunks if chunk]


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _join_nonempty(parts: Iterable[str]) -> str:
    return "\n".join(p for p in parts if p)


def _clip(text: str, limit: int) -> str:
    text = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    return text if len(text) <= limit else text[: limit - 3] + "..."
