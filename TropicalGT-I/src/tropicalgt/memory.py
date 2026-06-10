from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AnalogicalMemoryRecord:
    memory_id: str
    record_id: str
    score: float
    nll: float
    embedding: list[float]
    signature_vector: list[float]
    trajectory_embeddings: list[list[float]]
    trajectory_edges: list[dict[str, Any]]
    trajectory_paths: list[list[str]]
    filtered_simplicial_object: dict[str, Any]
    topological_algebra: dict[str, Any]
    derived_signature: dict[str, Any]
    metadata: dict[str, Any]


class AnalogicalMemoryHead(nn.Module):
    """Small trainable projection head for analogical memory queries."""

    def __init__(self, dim: int, memory_dim: int = 32) -> None:
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, memory_dim))

    def forward(self, graph_state: Tensor) -> Tensor:
        return F.normalize(self.query(graph_state), dim=-1)

    def contrastive_loss(self, graph_state: Tensor, memory_vectors: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        queries = self(graph_state)
        memory_vectors = F.normalize(memory_vectors.to(graph_state.device), dim=-1)
        if queries.numel() == 0 or memory_vectors.numel() == 0:
            zero = graph_state.sum() * 0.0
            return zero, {"memory_retrieval_loss": zero, "memory_retrieval_top1": zero}
        logits = queries @ memory_vectors.t()
        labels = torch.arange(queries.shape[0], device=graph_state.device).clamp_max(memory_vectors.shape[0] - 1)
        loss = F.cross_entropy(logits, labels)
        top1 = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, {"memory_retrieval_loss": loss.detach(), "memory_retrieval_top1": top1.detach()}


class AnalogicalMemoryBank:
    def __init__(self, path: str | Path, max_records: int = 2048) -> None:
        self.path = Path(path)
        self.max_records = int(max_records)
        self.records: list[AnalogicalMemoryRecord] = []
        if self.path.exists():
            self.load()

    def add(self, record: AnalogicalMemoryRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

    def extend(self, records: Iterable[AnalogicalMemoryRecord]) -> None:
        for record in records:
            self.add(record)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load(self) -> None:
        loaded = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loaded.append(AnalogicalMemoryRecord(**json.loads(line)))
        self.records = loaded[-self.max_records :]

    def retrieve(
        self,
        embedding: list[float] | np.ndarray,
        signature_vector: list[float] | np.ndarray,
        top_k: int = 5,
        embedding_weight: float = 0.55,
        signature_weight: float = 0.35,
        score_weight: float = 0.10,
    ) -> list[dict[str, Any]]:
        if not self.records:
            return []
        query_embedding = _normalize(np.asarray(embedding, dtype=float))
        query_signature = _normalize(np.asarray(signature_vector, dtype=float))
        rows = []
        for record in self.records:
            emb_sim = _cosine(query_embedding, _normalize(np.asarray(record.embedding, dtype=float)))
            sig_sim = _cosine(query_signature, _normalize(np.asarray(record.signature_vector, dtype=float)))
            quality = float(record.score) - 0.05 * float(record.nll)
            retrieval_score = embedding_weight * emb_sim + signature_weight * sig_sim + score_weight * quality
            rows.append(
                {
                    "memory_id": record.memory_id,
                    "record_id": record.record_id,
                    "retrieval_score": float(retrieval_score),
                    "embedding_similarity": float(emb_sim),
                    "signature_similarity": float(sig_sim),
                    "quality_score": quality,
                    "score": record.score,
                    "nll": record.nll,
                    "trajectory_edges": record.trajectory_edges,
                    "trajectory_paths": record.trajectory_paths,
                    "filtered_summary": record.filtered_simplicial_object.get("summary", {}),
                    "derived_signature": record.derived_signature,
                    "metadata": record.metadata,
                }
            )
        return sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)[:top_k]


def memory_records_from_scaling_report(
    scaling_report: dict[str, Any],
    source: str = "inference",
    min_score: float | None = None,
    max_records: int = 8,
) -> list[AnalogicalMemoryRecord]:
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict)]
    if not candidates:
        best = scaling_report.get("best")
        candidates = [best] if isinstance(best, dict) else []
    records = []
    trajectory_embeddings = [
        [float(v) for v in row.get("embedding", [])]
        for row in candidates
        if row.get("embedding") is not None
    ]
    trajectory_edges = [
        {"source": row.get("parent"), "target": row.get("record_id"), "path": row.get("path", [])}
        for row in candidates
        if row.get("parent") is not None
    ]
    trajectory_paths = [list(row.get("path", [])) for row in candidates]
    trajectory_complex = scaling_report.get("trajectory_filtered_simplicial_object", {})
    trajectory_algebra = scaling_report.get("trajectory_topological_algebra", {})
    for row in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_records]:
        score = float(row.get("score", 0.0))
        if min_score is not None and score < min_score:
            continue
        topology = row.get("topological_algebra") or trajectory_algebra or {}
        embedding = [float(v) for v in row.get("embedding", [])]
        signature = signature_vector(topology)
        memory_id = _memory_id(row.get("record_id", ""), embedding, signature)
        records.append(
            AnalogicalMemoryRecord(
                memory_id=memory_id,
                record_id=str(row.get("record_id", "")),
                score=score,
                nll=float(row.get("nll", 0.0)),
                embedding=embedding,
                signature_vector=signature,
                trajectory_embeddings=trajectory_embeddings,
                trajectory_edges=trajectory_edges,
                trajectory_paths=trajectory_paths,
                filtered_simplicial_object=trajectory_complex or row.get("filtered_simplicial_object", {}),
                topological_algebra=trajectory_algebra or topology,
                derived_signature=(trajectory_algebra or topology).get("derived_equivalence_signature", {}),
                metadata={
                    "source": source,
                    "level": row.get("level"),
                    "path": row.get("path", []),
                    "graphcg_projection": row.get("graphcg_projection"),
                },
            )
        )
    return records


def query_signature_from_report(result: dict[str, Any]) -> tuple[list[float], list[float]]:
    scaling = result.get("inference_scaling", {})
    best = scaling.get("best", {}) if isinstance(scaling, dict) else {}
    embedding = best.get("embedding") or []
    topology = best.get("topological_algebra") or result.get("topological_algebra") or {}
    return [float(v) for v in embedding], signature_vector(topology)


def signature_vector(topology: dict[str, Any], length: int = 32) -> list[float]:
    sig = topology.get("derived_equivalence_signature", {}) if isinstance(topology, dict) else {}
    chain = topology.get("chain_complex", {}) if isinstance(topology, dict) else {}
    mp = topology.get("multiparameter_persistence", {}) if isinstance(topology, dict) else {}
    graph = topology.get("graph_metrics", {}) if isinstance(topology, dict) else {}
    values: list[float] = []
    values.extend(float(v) for v in sig.get("betti_vector", [])[:4])
    values.extend(
        [
            float(sig.get("persistence_finite_interval_count", 0.0)),
            float(sig.get("persistence_infinite_interval_count", 0.0)),
            float(sig.get("persistence_total_finite_length", 0.0)),
            float(sig.get("multiparameter_grid_points", 0.0)),
            float(chain.get("euler_characteristic", 0.0)),
            float(graph.get("nodes", 0.0)),
            float(graph.get("edges", 0.0)),
            float(graph.get("undirected_cycle_rank", 0.0)),
            float(graph.get("dag_longest_path_length") or 0.0),
            float(graph.get("density", 0.0)),
        ]
    )
    rank_samples = mp.get("rank_invariant_samples", []) if isinstance(mp, dict) else []
    values.extend(float(row.get("h0_rank", 0.0)) for row in rank_samples[: length])
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return values[:length]


def _memory_id(record_id: object, embedding: list[float], signature: list[float]) -> str:
    h = hashlib.sha1()
    h.update(str(record_id).encode("utf-8", "ignore"))
    h.update(np.asarray(embedding[:16], dtype=np.float32).tobytes())
    h.update(np.asarray(signature[:16], dtype=np.float32).tobytes())
    return h.hexdigest()[:20]


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    norm = np.linalg.norm(values)
    return values / max(norm, 1e-12)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    size = min(left.size, right.size)
    if size == 0:
        return 0.0
    return float(np.dot(left[:size], right[:size]))
