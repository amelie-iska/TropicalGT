from __future__ import annotations

from dataclasses import dataclass
import json
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
        edges.append({"source": prev, "target": sid, "type": "depends_on"})
        prev = sid
    ans = add("answer", answer or "")
    edges.append({"source": prev, "target": ans, "type": "supports_answer"})
    return {"nodes": nodes, "edges": edges}


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
