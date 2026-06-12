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
    probability_filtered_simplicial_object: dict[str, Any]
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
            row = json.loads(line)
            row.setdefault("probability_filtered_simplicial_object", {})
            loaded.append(AnalogicalMemoryRecord(**row))
        self.records = loaded[-self.max_records :]

    def retrieve(
        self,
        embedding: list[float] | np.ndarray,
        signature_vector: list[float] | np.ndarray,
        top_k: int = 5,
        embedding_weight: float = 0.55,
        signature_weight: float = 0.35,
        score_weight: float = 0.10,
        diversity_weight: float = 0.18,
        exclude_record_ids: set[str] | None = None,
        exclude_memory_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.records:
            return []
        exclude_record_ids = {str(value) for value in (exclude_record_ids or set())}
        exclude_memory_ids = {str(value) for value in (exclude_memory_ids or set())}
        query_embedding = _normalize(np.asarray(embedding, dtype=float))
        query_signature = _normalize(np.asarray(signature_vector, dtype=float))
        rows = []
        for record in self.records:
            if record.record_id in exclude_record_ids or record.memory_id in exclude_memory_ids:
                continue
            emb_sim = _cosine(query_embedding, _normalize(np.asarray(record.embedding, dtype=float)))
            sig_sim = _cosine(query_signature, _normalize(np.asarray(record.signature_vector, dtype=float)))
            quality = float(record.score) - 0.05 * float(record.nll)
            retrieval_score = embedding_weight * emb_sim + signature_weight * sig_sim + score_weight * quality
            family = _record_family(record.record_id)
            signature_hash = _signature_hash(record.signature_vector)
            trajectory_source = str(record.metadata.get("source", record.record_id)) if isinstance(record.metadata, dict) else record.record_id
            rows.append(
                {
                    "memory_id": record.memory_id,
                    "record_id": record.record_id,
                    "trajectory_source": trajectory_source,
                    "record_family": family,
                    "signature_hash": signature_hash,
                    "retrieval_score": float(retrieval_score),
                    "base_retrieval_score": float(retrieval_score),
                    "embedding_similarity": float(emb_sim),
                    "signature_similarity": float(sig_sim),
                    "quality_score": quality,
                    "score": record.score,
                    "nll": record.nll,
                    "trajectory_edges": record.trajectory_edges,
                    "trajectory_paths": record.trajectory_paths,
                    "trajectory_embeddings": record.trajectory_embeddings,
                    "filtered_summary": record.filtered_simplicial_object.get("summary", {}),
                    "filtered_simplicial_object": record.filtered_simplicial_object,
                    "probability_filtered_simplicial_object": record.probability_filtered_simplicial_object,
                    "probability_filtered_summary": record.probability_filtered_simplicial_object.get("summary", {}),
                    "topological_algebra": record.topological_algebra,
                    "signature_vector": record.signature_vector,
                    "derived_signature": record.derived_signature,
                    "metadata": record.metadata,
                }
            )
        rows = _best_per_trajectory_source(rows)
        return _diverse_top_k(rows, top_k=max(int(top_k), 0), diversity_weight=float(diversity_weight))


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
    trajectory_probability_complex = scaling_report.get("trajectory_probability_filtered_simplicial_object", {})
    trajectory_algebra = scaling_report.get("trajectory_topological_algebra", {})
    trajectory_probability_algebra = scaling_report.get("trajectory_probability_topological_algebra", {})
    for row in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_records]:
        score = float(row.get("score", 0.0))
        if min_score is not None and score < min_score:
            continue
        topology = row.get("topological_algebra") or trajectory_algebra or {}
        row_complex = row.get("filtered_simplicial_object", {})
        row_probability_complex = row.get("probability_filtered_simplicial_object", {})
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
                filtered_simplicial_object=row_complex or trajectory_complex,
                probability_filtered_simplicial_object=row_probability_complex or trajectory_probability_complex,
                topological_algebra=topology or trajectory_algebra,
                derived_signature=(topology or trajectory_algebra).get("derived_equivalence_signature", {}),
                metadata={
                    "source": source,
                    "level": row.get("level"),
                    "path": row.get("path", []),
                    "graphcg_projection": row.get("graphcg_projection"),
                    "trajectory_filtered_simplicial_object": trajectory_complex,
                    "trajectory_probability_filtered_simplicial_object": trajectory_probability_complex,
                    "trajectory_probability_topological_algebra": trajectory_probability_algebra,
                    "trajectory_summary": trajectory_complex.get("summary", {}) if isinstance(trajectory_complex, dict) else {},
                    "trajectory_probability_summary": trajectory_probability_complex.get("summary", {}) if isinstance(trajectory_probability_complex, dict) else {},
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


def _diverse_top_k(rows: list[dict[str, Any]], top_k: int, diversity_weight: float) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    ranked = sorted(rows, key=lambda row: float(row.get("retrieval_score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_families: set[str] = set()
    selected_signatures: set[str] = set()
    remaining = ranked
    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")
        for idx, row in enumerate(remaining):
            penalty = 0.0
            if row.get("record_family") in selected_families:
                penalty += diversity_weight
            if row.get("signature_hash") in selected_signatures:
                penalty += diversity_weight * 0.5
            adjusted = float(row.get("base_retrieval_score", row.get("retrieval_score", 0.0))) - penalty
            if adjusted > best_score:
                best_idx = idx
                best_score = adjusted
        row = dict(remaining.pop(best_idx))
        row["retrieval_score"] = float(best_score)
        row["diversity_adjusted"] = True
        row["diversity_penalty_applied"] = float(row.get("base_retrieval_score", 0.0)) - float(best_score)
        selected.append(row)
        selected_families.add(str(row.get("record_family", "")))
        selected_signatures.add(str(row.get("signature_hash", "")))
    return selected


def _best_per_trajectory_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse candidate-level memories that carry the same trajectory complex.

    Analogical maps are trajectory-complex maps. Storing one memory record per
    candidate is useful for retrieval scores, but rendering top-k maps from the
    same trajectory produces duplicate complexes. Keep the best-scoring record
    per trajectory source before the top-k/diversity pass.
    """

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("trajectory_source") or row.get("record_id") or row.get("memory_id"))
        current = best.get(key)
        if current is None or float(row.get("retrieval_score", 0.0)) > float(current.get("retrieval_score", 0.0)):
            best[key] = row
    return list(best.values())


def _record_family(record_id: object) -> str:
    value = str(record_id or "")
    if "|" in value:
        return value.split("|", 1)[0]
    return value


def _signature_hash(signature: list[float]) -> str:
    rounded = [round(float(v), 4) for v in signature[:16]]
    return hashlib.sha1(json.dumps(rounded, sort_keys=True).encode("utf-8")).hexdigest()[:12]


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
