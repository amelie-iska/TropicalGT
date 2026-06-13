from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
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


@dataclass(frozen=True)
class AnalogicalMemoryQualityGate:
    """Conservative insertion gate for analogical trajectory memories.

    Retrieval may ask for many analogies, but storage is curated: a trajectory is
    admitted only when model-derived probability topology and configured quality
    thresholds are present. If no early trajectory clears the gate, the memory
    bank should legitimately remain empty.
    """

    min_score: float | None = None
    min_quality_score: float | None = None
    max_nll: float | None = None
    max_bpb: float | None = None
    min_nll_improvement: float | None = None
    min_margin_mean: float | None = None
    require_probability_complex: bool = True
    min_probability_vertices: int = 2
    min_probability_simplices: int = 1
    require_topological_algebra: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None, *, min_score: float | None = None) -> "AnalogicalMemoryQualityGate":
        cfg = cfg or {}
        return cls(
            min_score=_optional_float(cfg.get("memory_quality_min_score", cfg.get("memory_min_score", min_score))),
            min_quality_score=_optional_float(cfg.get("memory_quality_min_quality_score")),
            max_nll=_optional_float(cfg.get("memory_quality_max_nll")),
            max_bpb=_optional_float(cfg.get("memory_quality_max_bpb")),
            min_nll_improvement=_optional_float(cfg.get("memory_quality_min_nll_improvement")),
            min_margin_mean=_optional_float(cfg.get("memory_quality_min_margin_mean")),
            require_probability_complex=bool(cfg.get("memory_quality_require_probability_complex", True)),
            min_probability_vertices=max(0, int(cfg.get("memory_quality_min_probability_vertices", 2))),
            min_probability_simplices=max(0, int(cfg.get("memory_quality_min_probability_simplices", 1))),
            require_topological_algebra=bool(cfg.get("memory_quality_require_topological_algebra", True)),
        )

    def evaluate(
        self,
        row: dict[str, Any],
        *,
        scaling_report: dict[str, Any] | None = None,
        probability_complex: dict[str, Any] | None = None,
        topology: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        score = _safe_float(row.get("score"), 0.0)
        nll = _safe_float(row.get("nll"), 0.0)
        bpb = nll / math.log(2.0) if math.isfinite(nll) else math.inf
        margin_mean = _safe_float(row.get("margin_mean"), 0.0)
        improvement = _nll_improvement(row, scaling_report or {})
        quality_score = score - 0.05 * nll + max(0.0, improvement if improvement is not None else 0.0)
        probability_report = _probability_complex_quality(probability_complex or {})
        topology_report = _topology_quality(topology or {})
        reasons: list[str] = []
        if self.min_score is not None and score < self.min_score:
            reasons.append("score_below_threshold")
        if self.min_quality_score is not None and quality_score < self.min_quality_score:
            reasons.append("quality_score_below_threshold")
        if self.max_nll is not None and nll > self.max_nll:
            reasons.append("nll_above_threshold")
        if self.max_bpb is not None and bpb > self.max_bpb:
            reasons.append("bpb_above_threshold")
        if self.min_nll_improvement is not None:
            if improvement is None or improvement < self.min_nll_improvement:
                reasons.append("nll_improvement_below_threshold")
        if self.min_margin_mean is not None and margin_mean < self.min_margin_mean:
            reasons.append("margin_mean_below_threshold")
        if self.require_probability_complex:
            if not probability_report["available"]:
                reasons.append("probability_complex_unavailable")
            if probability_report["probability_vertices"] < self.min_probability_vertices:
                reasons.append("insufficient_probability_vertices")
            if probability_report["simplices"] < self.min_probability_simplices:
                reasons.append("insufficient_probability_simplices")
        if self.require_topological_algebra and not topology_report["available"]:
            reasons.append("topological_algebra_unavailable")
        return {
            "passed": not reasons,
            "reasons": reasons,
            "score": score,
            "quality_score": quality_score,
            "nll": nll,
            "bpb": bpb,
            "nll_improvement": improvement,
            "margin_mean": margin_mean,
            "probability_complex": probability_report,
            "topological_algebra": topology_report,
            "thresholds": asdict(self),
        }


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
        for line in _tail_jsonl_lines(self.path, self.max_records):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
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
        landscape_weight: float = 0.18,
        diversity_weight: float = 0.18,
        query_topology: dict[str, Any] | None = None,
        exclude_record_ids: set[str] | None = None,
        exclude_memory_ids: set[str] | None = None,
        exclude_sources: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.records:
            return []
        exclude_record_ids = {str(value) for value in (exclude_record_ids or set())}
        exclude_memory_ids = {str(value) for value in (exclude_memory_ids or set())}
        exclude_sources = {str(value) for value in (exclude_sources or set())}
        query_embedding = _normalize(np.asarray(embedding, dtype=float))
        query_signature = _normalize(np.asarray(signature_vector, dtype=float))
        query_topology = query_topology if isinstance(query_topology, dict) else {}
        rows = []
        for record in self.records:
            trajectory_source = str(record.metadata.get("source", record.record_id)) if isinstance(record.metadata, dict) else record.record_id
            if record.record_id in exclude_record_ids or record.memory_id in exclude_memory_ids or trajectory_source in exclude_sources:
                continue
            emb_sim = _cosine(query_embedding, _normalize(np.asarray(record.embedding, dtype=float)))
            sig_sim = _cosine(query_signature, _normalize(np.asarray(record.signature_vector, dtype=float)))
            record_metadata = record.metadata if isinstance(record.metadata, dict) else {}
            memory_topology = record_metadata.get("trajectory_probability_topological_algebra")
            if not isinstance(memory_topology, dict):
                memory_topology = record.topological_algebra if isinstance(record.topological_algebra, dict) else {}
            landscape_report = persistence_landscape_vector_similarity(query_topology, memory_topology)
            vector_report = persistence_vector_representation_similarity(query_topology, memory_topology)
            landscape_sim = float(landscape_report.get("l2_similarity", 0.0)) if landscape_report.get("available") else 0.0
            vector_sim = float(vector_report.get("aggregate_similarity", 0.0)) if vector_report.get("available") else landscape_sim
            quality = float(record.score) - 0.05 * float(record.nll)
            retrieval_score = embedding_weight * emb_sim + signature_weight * sig_sim + score_weight * quality
            if vector_report.get("available") or landscape_report.get("available"):
                retrieval_score += float(landscape_weight) * vector_sim
            family = _record_family(record.record_id)
            signature_hash = _signature_hash(record.signature_vector)
            trajectory_probability_complex = record_metadata.get("trajectory_probability_filtered_simplicial_object", {})
            if not isinstance(trajectory_probability_complex, dict):
                trajectory_probability_complex = {}
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
                    "persistence_landscape_vector_similarity": landscape_report,
                    "persistence_vector_representation_similarity": vector_report,
                    "persistence_vector_aggregate_similarity": float(vector_report.get("aggregate_similarity", 0.0)) if vector_report.get("available") else 0.0,
                    "persistence_vector_component_count": int(vector_report.get("component_count", 0) or 0),
                    "persistence_vector_available_methods": list(vector_report.get("available_methods", [])) if vector_report.get("available") else [],
                    "persistence_landscape_l2_similarity": float(landscape_report.get("l2_similarity", 0.0)) if landscape_report.get("available") else 0.0,
                    "persistence_landscape_cosine": float(landscape_report.get("cosine", 0.0)) if landscape_report.get("available") else 0.0,
                    "persistence_landscape_correlation": float(landscape_report.get("correlation", 0.0)) if landscape_report.get("available") else 0.0,
                    "persistence_landscape_l2_distance": float(landscape_report.get("l2_distance", 0.0)) if landscape_report.get("available") else 0.0,
                    "persistence_landscape_overlap_dim": int(landscape_report.get("overlap_dim", 0) or 0),
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
                    "row_probability_filtered_simplicial_object": record.probability_filtered_simplicial_object,
                    "trajectory_probability_filtered_simplicial_object": trajectory_probability_complex,
                    "trajectory_probability_filtered_summary": trajectory_probability_complex.get("summary", {}),
                    "topological_algebra": record.topological_algebra,
                    "signature_vector": record.signature_vector,
                    "derived_signature": record.derived_signature,
                    "metadata": record.metadata,
                }
            )
        rows = _best_per_trajectory_source(rows)
        return _diverse_top_k(rows, top_k=max(int(top_k), 0), diversity_weight=float(diversity_weight))


def _tail_jsonl_lines(
    path: Path,
    max_lines: int,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    max_record_bytes: int = 16 * 1024 * 1024,
) -> list[str]:
    """Return bounded tail JSONL rows without materializing huge memory banks."""

    max_lines = max(int(max_lines), 0)
    if max_lines == 0:
        return []
    lines: deque[str] = deque(maxlen=max_lines)
    with path.open("rb") as fh:
        fh.seek(0, 2)
        position = fh.tell()
        pending = b""
        while position > 0 and len(lines) < max_lines:
            read_size = min(int(chunk_size), position)
            position -= read_size
            fh.seek(position)
            chunk = fh.read(read_size)
            block = chunk + pending
            parts = block.splitlines()
            if position > 0:
                pending = parts[0] if parts else block
                if len(pending) > max_record_bytes:
                    pending = b""
                parts = parts[1:]
            else:
                pending = b""
            for raw in parts:
                if raw.strip() and len(raw) <= max_record_bytes:
                    lines.append(raw.decode("utf-8", "replace"))
    return list(lines)[-max_lines:]

def memory_records_from_scaling_report(
    scaling_report: dict[str, Any],
    source: str = "inference",
    min_score: float | None = None,
    max_records: int = 8,
    quality_gate: AnalogicalMemoryQualityGate | dict[str, Any] | None = None,
) -> list[AnalogicalMemoryRecord]:
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict)]
    if not candidates:
        best = scaling_report.get("best")
        candidates = [best] if isinstance(best, dict) else []
    records = []
    gate = _coerce_quality_gate(quality_gate, min_score=min_score)
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
    trajectory_complex = _compact_filtered_object(scaling_report.get("trajectory_filtered_simplicial_object", {}))
    trajectory_probability_complex = _compact_filtered_object(scaling_report.get("trajectory_probability_filtered_simplicial_object", {}))
    trajectory_algebra = _compact_topological_algebra(scaling_report.get("trajectory_topological_algebra", {}))
    trajectory_probability_algebra = _compact_topological_algebra(scaling_report.get("trajectory_probability_topological_algebra", {}))
    for row in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_records]:
        score = _safe_float(row.get("score"), 0.0)
        topology = _compact_topological_algebra(row.get("topological_algebra") or trajectory_algebra or {})
        row_complex = _compact_filtered_object(row.get("filtered_simplicial_object", {}))
        row_probability_complex = _compact_filtered_object(row.get("probability_filtered_simplicial_object", {}))
        probability_complex_for_gate = trajectory_probability_complex if trajectory_probability_complex else row_probability_complex
        quality_report = gate.evaluate(
            row,
            scaling_report=scaling_report,
            probability_complex=probability_complex_for_gate,
            topology=topology or trajectory_algebra,
        )
        if not quality_report["passed"]:
            continue
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
                probability_filtered_simplicial_object=row_probability_complex,
                topological_algebra=topology or trajectory_algebra,
                derived_signature=(topology or trajectory_algebra).get("derived_equivalence_signature", {}),
                metadata={
                    "source": source,
                    "level": row.get("level"),
                    "path": row.get("path", []),
                    "graphcg_projection": row.get("graphcg_projection"),
                    "trajectory_summary": trajectory_complex.get("summary", {}) if isinstance(trajectory_complex, dict) else {},
                    "trajectory_filtered_simplicial_object": trajectory_complex,
                    "trajectory_probability_summary": trajectory_probability_complex.get("summary", {}) if isinstance(trajectory_probability_complex, dict) else {},
                    "trajectory_probability_filtered_simplicial_object": trajectory_probability_complex,
                    "trajectory_probability_topological_algebra": trajectory_probability_algebra,
                    "trajectory_probability_topological_algebra_summary": trajectory_probability_algebra.get("summary", {}) if isinstance(trajectory_probability_algebra, dict) else {},
                    "row_probability_filtered_simplicial_object": row_probability_complex,
                    "quality_gate": quality_report,
                    "memory_payload_policy": "compact_real_trajectory_payload_no_duplicate_full_complexes",
                },
            )
        )
    return records



def memory_quality_gate_summary(
    scaling_report: dict[str, Any],
    quality_gate: AnalogicalMemoryQualityGate | dict[str, Any] | None = None,
    *,
    min_score: float | None = None,
    max_records: int = 8,
) -> dict[str, Any]:
    gate = _coerce_quality_gate(quality_gate, min_score=min_score)
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict)]
    if not candidates and isinstance(scaling_report.get("best"), dict):
        candidates = [scaling_report["best"]]
    trajectory_algebra = _compact_topological_algebra(scaling_report.get("trajectory_topological_algebra", {}))
    trajectory_probability_complex = _compact_filtered_object(scaling_report.get("trajectory_probability_filtered_simplicial_object", {}))
    reason_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True)[:max_records]:
        topology = _compact_topological_algebra(row.get("topological_algebra") or trajectory_algebra or {})
        probability_complex = _compact_filtered_object(row.get("probability_filtered_simplicial_object", {})) or trajectory_probability_complex
        report = gate.evaluate(row, scaling_report=scaling_report, probability_complex=probability_complex, topology=topology)
        for reason in report["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rows.append(
            {
                "record_id": str(row.get("record_id", "")),
                "level": row.get("level"),
                "path": row.get("path", []),
                "passed": bool(report["passed"]),
                "reasons": report["reasons"],
                "score": report["score"],
                "quality_score": report["quality_score"],
                "nll": report["nll"],
                "bpb": report["bpb"],
                "nll_improvement": report["nll_improvement"],
                "probability_vertices": report["probability_complex"]["probability_vertices"],
                "probability_simplices": report["probability_complex"]["simplices"],
                "topological_algebra_available": report["topological_algebra"]["available"],
            }
        )
    eligible = sum(1 for row in rows if row["passed"])
    return {
        "candidate_count": len(rows),
        "eligible_count": eligible,
        "rejected_count": len(rows) - eligible,
        "reason_counts": reason_counts,
        "thresholds": asdict(gate),
        "rows": rows,
        "policy": "store_only_quality_gated_model_probability_trajectory_memories",
    }


def _coerce_quality_gate(quality_gate: AnalogicalMemoryQualityGate | dict[str, Any] | None, *, min_score: float | None = None) -> AnalogicalMemoryQualityGate:
    if isinstance(quality_gate, AnalogicalMemoryQualityGate):
        if min_score is not None and quality_gate.min_score is None:
            values = asdict(quality_gate)
            values["min_score"] = min_score
            return AnalogicalMemoryQualityGate(**values)
        return quality_gate
    if isinstance(quality_gate, dict):
        return AnalogicalMemoryQualityGate.from_config(quality_gate, min_score=min_score)
    return AnalogicalMemoryQualityGate(min_score=min_score, require_probability_complex=False, require_topological_algebra=False)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _nll_improvement(row: dict[str, Any], scaling_report: dict[str, Any]) -> float | None:
    current = _safe_float(row.get("nll"), math.inf)
    candidates = [cand for cand in scaling_report.get("candidates", []) if isinstance(cand, dict)]
    roots = [_safe_float(cand.get("nll"), math.inf) for cand in candidates if _candidate_level(cand) == 0]
    roots = [value for value in roots if math.isfinite(value)]
    if not roots or not math.isfinite(current):
        return None
    return min(roots) - current


def _candidate_level(row: dict[str, Any]) -> int:
    try:
        return int(row.get("level", -1))
    except (TypeError, ValueError):
        return -1


def _probability_complex_quality(obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, dict) or obj.get("available") is False:
        return {"available": False, "vertices": 0, "probability_vertices": 0, "simplices": 0, "filtration_model": ""}
    simplices = obj.get("simplices") if isinstance(obj.get("simplices"), list) else []
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    vertices = sum(1 for simplex in simplices if isinstance(simplex, dict) and _simplex_dimension(simplex) == 0)
    probability_vertices = sum(
        1
        for simplex in simplices
        if isinstance(simplex, dict)
        and _simplex_dimension(simplex) == 0
        and any(isinstance(simplex.get(key), list) and len(simplex.get(key, [])) > 0 for key in ("probability", "model_probability_vector", "probability_vector"))
    )
    simplex_count = len(simplices) or int(summary.get("simplices", summary.get("simplex_count", 0)) or 0)
    filtration_model = str(summary.get("filtration_model", obj.get("filtration_model", "")) or "")
    available = bool(probability_vertices or ("jensen_shannon" in filtration_model and simplex_count > 0))
    return {
        "available": available,
        "vertices": vertices,
        "probability_vertices": probability_vertices,
        "simplices": simplex_count,
        "filtration_model": filtration_model,
    }


def _simplex_dimension(simplex: dict[str, Any]) -> int:
    try:
        return int(simplex.get("dimension", -1))
    except (TypeError, ValueError):
        return -1


def _topology_quality(topology: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(topology, dict) or topology.get("available") is False:
        return {
            "available": False,
            "has_persistence": False,
            "has_free_resolution": False,
            "has_real_free_resolution": False,
            "has_chain_presentation_diagnostics": False,
            "has_commutative_algebra": False,
        }
    has_persistence = bool(topology.get("persistence") or topology.get("persistence_summary") or topology.get("betti_numbers"))
    ca = topology.get("commutative_algebra") if isinstance(topology.get("commutative_algebra"), dict) else {}
    chain_keys = (
        "two_parameter_chain_presentation_diagnostics",
        "multiparameter_chain_presentation_diagnostics",
        "two_parameter_free_resolution",
        "multiparameter_free_resolution_proxy",
    )
    has_chain = any(isinstance(ca.get(key), dict) and bool(ca.get(key)) for key in chain_keys)
    has_real = False
    for key in chain_keys:
        value = ca.get(key)
        if isinstance(value, dict):
            real = value.get("real_free_resolution") if isinstance(value.get("real_free_resolution"), dict) else {}
            if bool(real.get("available")) and bool(real.get("certificate_attached")):
                has_real = True
                break
    return {
        "available": bool(has_persistence or has_chain or topology.get("derived_equivalence_signature")),
        "has_persistence": has_persistence,
        "has_free_resolution": has_real,
        "has_real_free_resolution": has_real,
        "has_chain_presentation_diagnostics": has_chain,
        "has_commutative_algebra": bool(ca),
    }


def _compact_filtered_object(obj: Any, *, max_simplices: int = 512, max_vector_len: int = 4096) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("available", "reason", "summary", "filtration_model", "source", "simplex_tree", "persistence", "persistent_homology"):
        if key in obj:
            out[key] = obj[key]
    simplices = obj.get("simplices")
    if isinstance(simplices, list):
        compact = []
        for simplex in simplices[:max_simplices]:
            if isinstance(simplex, dict):
                compact.append(_compact_simplex(simplex, max_vector_len=max_vector_len))
        out["simplices"] = compact
        out["simplices_truncated"] = len(simplices) > len(compact)
        out["original_simplex_count"] = len(simplices)
    return out


def _compact_simplex(simplex: dict[str, Any], *, max_vector_len: int) -> dict[str, Any]:
    keep = {
        "simplex",
        "dimension",
        "filtration",
        "label",
        "record_id",
        "source",
        "target",
        "path",
        "level",
        "score",
        "nll",
        "action",
        "vertex",
        "vertices",
        "edge",
    }
    out = {key: value for key, value in simplex.items() if key in keep}
    for vector_key in ("embedding", "probability", "model_probability_vector", "probability_vector"):
        vector = simplex.get(vector_key)
        if isinstance(vector, list):
            if len(vector) <= max_vector_len:
                out[vector_key] = vector
            else:
                out[vector_key + "_omitted"] = True
                out[vector_key + "_length"] = len(vector)
    return out


def _compact_topological_algebra(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    keep = {
        "summary",
        "betti_numbers",
        "persistence_summary",
        "multiparameter_summary",
        "free_resolution_summary",
        "derived_equivalence_signature",
        "derived_signature",
        "chain_complex",
        "persistence",
        "persistence_representations",
        "multiparameter_persistence",
        "status",
        "available",
        "reason",
    }
    out = {key: value for key, value in obj.items() if key in keep}
    if isinstance(obj.get("commutative_algebra"), dict):
        out["commutative_algebra"] = _compact_commutative_algebra(obj["commutative_algebra"])
    return out


def _compact_commutative_algebra(ca: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "two_parameter_chain_presentation_diagnostics",
        "multiparameter_chain_presentation_diagnostics",
        "two_parameter_free_resolution",
        "multiparameter_free_resolution_proxy",
        "taylor_resolution_upper_bound",
    ):
        value = ca.get(key)
        if isinstance(value, dict):
            out[key] = _compact_free_resolution(value)
    for key in ("hochster_betti", "stanley_reisner"):
        value = ca.get(key)
        if isinstance(value, dict):
            out[key] = {k: value.get(k) for k in ("available", "truncated", "total_betti", "num_generators", "minimal_nonfaces") if k in value}
    return out


def _compact_free_resolution(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ring": value.get("ring"),
        "method": value.get("method"),
        "resolution_status": value.get("resolution_status"),
        "not_a_free_resolution": value.get("not_a_free_resolution", True),
        "certificate_attached": value.get("certificate_attached", False),
        "deprecated_alias_for": value.get("deprecated_alias_for"),
        "field": value.get("field"),
        "free_chain_modules": value.get("free_chain_modules", []),
        "monomial_labeled_boundary_entry_counts": value.get("monomial_labeled_boundary_entry_counts", {}),
        "real_free_resolution": value.get("real_free_resolution", {}),
        "minimal_free_resolution": value.get("minimal_free_resolution", {}),
    }
    det = value.get("determinantal_ideals") if isinstance(value.get("determinantal_ideals"), dict) else {}
    fit = value.get("fitting_ideals") if isinstance(value.get("fitting_ideals"), dict) else {}
    be = value.get("buchsbaum_eisenbud") if isinstance(value.get("buchsbaum_eisenbud"), dict) else {}
    out["determinantal_ideal_summary"] = {
        "available": det.get("available", False),
        "map_count": len(det.get("maps", {})) if isinstance(det.get("maps"), dict) else 0,
        "nonzero_sampled_generators": sum(
            int(size.get("nonzero_generator_count_in_checked_minors", 0) or 0)
            for mp in (det.get("maps", {}) or {}).values()
            if isinstance(mp, dict)
            for size in mp.get("determinantal_ideals_by_minor_size", [])
            if isinstance(size, dict)
        ) if isinstance(det.get("maps"), dict) else 0,
    }
    out["fitting_ideal_summary"] = {
        "available": fit.get("available", False),
        "map_count": len(fit.get("maps", {})) if isinstance(fit.get("maps"), dict) else 0,
        "bounded_invariant_count": sum(
            len(mp.get("fitting_invariants", []))
            for mp in (fit.get("maps", {}) or {}).values()
            if isinstance(mp, dict)
        ) if isinstance(fit.get("maps"), dict) else 0,
    }
    out["buchsbaum_eisenbud_summary"] = {
        "available": be.get("available", False),
        "composition_zero_checks": len(be.get("composition_zero_checks", [])) if isinstance(be.get("composition_zero_checks"), list) else 0,
        "exact_chain_modules": sum(1 for row in be.get("rank_exactness_checks", []) if isinstance(row, dict) and row.get("exact_at_chain_module_over_F2_incidence")) if isinstance(be.get("rank_exactness_checks"), list) else 0,
        "multiplier_certificate_available": bool((be.get("buchsbaum_eisenbud_multipliers") or {}).get("available")) if isinstance(be.get("buchsbaum_eisenbud_multipliers"), dict) else False,
    }
    return out


def query_signature_from_report(result: dict[str, Any]) -> tuple[list[float], list[float]]:
    scaling = result.get("inference_scaling", {})
    best = scaling.get("best", {}) if isinstance(scaling, dict) else {}
    embedding = best.get("embedding") or []
    topology = best.get("topological_algebra") or result.get("topological_algebra") or {}
    return [float(v) for v in embedding], signature_vector(topology)


def query_topology_from_report(result: dict[str, Any]) -> dict[str, Any]:
    scaling = result.get("inference_scaling", {}) if isinstance(result, dict) else {}
    best = scaling.get("best", {}) if isinstance(scaling, dict) else {}
    for candidate in (
        scaling.get("trajectory_probability_topological_algebra") if isinstance(scaling, dict) else None,
        scaling.get("trajectory_topological_algebra") if isinstance(scaling, dict) else None,
        best.get("topological_algebra") if isinstance(best, dict) else None,
        result.get("topological_algebra") if isinstance(result, dict) else None,
    ):
        if isinstance(candidate, dict):
            return candidate
    return {}


def persistence_landscape_vector(topology: dict[str, Any], max_dimension: int | None = None) -> dict[int, np.ndarray]:
    if not isinstance(topology, dict):
        return {}
    reps = topology.get("persistence_representations")
    if not isinstance(reps, dict) or not reps.get("available"):
        return {}
    methods = reps.get("methods")
    if not isinstance(methods, dict):
        return {}
    dims: list[int] = []
    for key in methods.keys():
        try:
            dim = int(key)
        except (TypeError, ValueError):
            continue
        if max_dimension is None or dim <= max_dimension:
            dims.append(dim)
    vectors: dict[int, np.ndarray] = {}
    for dim in sorted(set(dims)):
        row = methods.get(str(dim), {})
        if not isinstance(row, dict) or not row.get("available"):
            continue
        landscape = row.get("landscape")
        if not isinstance(landscape, dict):
            continue
        raw = landscape.get("vector")
        if not isinstance(raw, list) or not raw:
            continue
        vals: list[float] = []
        for value in raw:
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append(v)
        if vals:
            vectors[dim] = np.asarray(vals, dtype=float)
    return vectors


def _concatenate_landscape_vectors(left: dict[int, np.ndarray], right: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    dims = sorted(set(left) | set(right))
    if not dims:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for dim in dims:
        lv = left.get(dim, np.zeros(0, dtype=float))
        rv = right.get(dim, np.zeros(0, dtype=float))
        n = max(int(lv.size), int(rv.size))
        if n <= 0:
            continue
        lp = np.zeros(n, dtype=float)
        rp = np.zeros(n, dtype=float)
        lp[: int(lv.size)] = lv[:n]
        rp[: int(rv.size)] = rv[:n]
        left_parts.append(lp)
        right_parts.append(rp)
    if not left_parts:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), []
    return np.concatenate(left_parts), np.concatenate(right_parts), dims


_VECTOR_REPRESENTATION_SPECS: dict[str, tuple[str, tuple[str, ...], float]] = {
    "landscape": ("gudhi.representations.Landscape.vector", ("landscape", "vector"), 1.00),
    "betti_curve": ("gudhi.representations.BettiCurve.values", ("betti_curve", "values"), 0.85),
    "silhouette": ("gudhi.representations.Silhouette.values", ("silhouette", "values"), 0.65),
    "entropy_vector": ("gudhi.representations.Entropy.vector", ("entropy", "vector"), 0.35),
    "persistence_lengths": ("gudhi.representations.PersistenceLengths.values", ("persistence_lengths", "values"), 0.55),
    "topological_vector": ("gudhi.representations.TopologicalVector.values", ("topological_vector", "values"), 0.80),
    "persistence_image": ("gudhi.representations.PersistenceImage.values", ("persistence_image", "values"), 0.55),
}


def _flatten_numeric_values(raw: Any) -> list[float]:
    values: list[float] = []
    if isinstance(raw, dict):
        for key in sorted(raw):
            values.extend(_flatten_numeric_values(raw[key]))
        return values
    if isinstance(raw, (list, tuple)):
        for item in raw:
            values.extend(_flatten_numeric_values(item))
        return values
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return values
    if math.isfinite(value):
        values.append(value)
    return values


def _nested_method_value(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def persistence_vector_representations(topology: dict[str, Any], max_dimension: int | None = None) -> dict[str, dict[int, np.ndarray]]:
    """Return real GUDHI vector-representation features grouped by method and homology dimension."""

    if not isinstance(topology, dict):
        return {}
    reps = topology.get("persistence_representations")
    if not isinstance(reps, dict) or not reps.get("available"):
        return {}
    methods = reps.get("methods")
    if not isinstance(methods, dict):
        return {}
    out: dict[str, dict[int, np.ndarray]] = {name: {} for name in _VECTOR_REPRESENTATION_SPECS}
    for key, row in methods.items():
        try:
            dim = int(key)
        except (TypeError, ValueError):
            continue
        if max_dimension is not None and dim > max_dimension:
            continue
        if not isinstance(row, dict) or not row.get("available"):
            continue
        for name, (_source, value_path, _weight) in _VECTOR_REPRESENTATION_SPECS.items():
            values = _flatten_numeric_values(_nested_method_value(row, value_path))
            if values:
                out[name][dim] = np.asarray(values, dtype=float)
    return {name: dims for name, dims in out.items() if dims}


def _vector_similarity_from_arrays(q_vec: np.ndarray, m_vec: np.ndarray) -> dict[str, float]:
    q_norm = float(np.linalg.norm(q_vec))
    m_norm = float(np.linalg.norm(m_vec))
    denom = q_norm * m_norm
    if denom > 1e-12:
        cosine = float(np.dot(q_vec, m_vec) / denom)
    else:
        cosine = 1.0 if float(np.linalg.norm(q_vec - m_vec)) <= 1e-12 else 0.0
    l2_distance = float(np.linalg.norm(q_vec - m_vec))
    l2_similarity = float(1.0 / (1.0 + l2_distance))
    if q_vec.size > 1 and m_vec.size > 1 and float(np.std(q_vec)) > 1e-12 and float(np.std(m_vec)) > 1e-12:
        correlation = float(np.corrcoef(q_vec, m_vec)[0, 1])
    else:
        correlation = cosine
    cosine_01 = float(max(0.0, min(1.0, 0.5 * (cosine + 1.0))))
    return {
        "query_norm": q_norm,
        "memory_norm": m_norm,
        "cosine": cosine,
        "cosine_01": cosine_01,
        "l2_distance": l2_distance,
        "l2_similarity": l2_similarity,
        "correlation": correlation,
        "vector_similarity": float(0.55 * l2_similarity + 0.45 * cosine_01),
    }


def persistence_vector_representation_similarity(query_topology: dict[str, Any], memory_topology: dict[str, Any]) -> dict[str, Any]:
    """Compare all available vectorized GUDHI persistence representations.

    These are real cached vectors produced from persistence diagrams by GUDHI.
    The comparison operations are vector-space operations suitable for retrieval
    losses or rewards; the upstream GUDHI transforms in this project remain
    NumPy/scikit-learn transforms, so this is not claiming end-to-end autograd
    through persistent homology.
    """

    q_methods = persistence_vector_representations(query_topology)
    m_methods = persistence_vector_representations(memory_topology)
    components: dict[str, Any] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    available_methods: list[str] = []
    for name, (source, _value_path, default_weight) in _VECTOR_REPRESENTATION_SPECS.items():
        q_by_dim = q_methods.get(name, {})
        m_by_dim = m_methods.get(name, {})
        if not q_by_dim or not m_by_dim:
            components[name] = {
                "available": False,
                "source": source,
                "reason": "missing_query_or_memory_vector",
                "query_dims": sorted(q_by_dim.keys()),
                "memory_dims": sorted(m_by_dim.keys()),
            }
            continue
        q_vec, m_vec, dims = _concatenate_landscape_vectors(q_by_dim, m_by_dim)
        if q_vec.size == 0 or m_vec.size == 0:
            components[name] = {
                "available": False,
                "source": source,
                "reason": "empty_concatenated_vector",
                "query_dims": sorted(q_by_dim.keys()),
                "memory_dims": sorted(m_by_dim.keys()),
            }
            continue
        metrics = _vector_similarity_from_arrays(q_vec, m_vec)
        component = {
            "available": True,
            "source": source,
            "dims": dims,
            "query_dims": sorted(q_by_dim.keys()),
            "memory_dims": sorted(m_by_dim.keys()),
            "overlap_dim": int(q_vec.size),
            "weight": float(default_weight),
            **metrics,
        }
        components[name] = component
        weighted_sum += float(default_weight) * float(component["vector_similarity"])
        weight_total += float(default_weight)
        available_methods.append(name)
    if weight_total <= 0.0:
        return {
            "available": False,
            "source": "gudhi.representations.vector_methods",
            "reason": "no_shared_vectorized_persistence_representations",
            "query_methods": sorted(q_methods.keys()),
            "memory_methods": sorted(m_methods.keys()),
            "components": components,
        }
    aggregate_similarity = float(weighted_sum / weight_total)
    return {
        "available": True,
        "source": "gudhi.representations.vector_methods",
        "comparison_space": "weighted vector-space comparison of GUDHI Landscape, BettiCurve, Silhouette, Entropy, PersistenceLengths, TopologicalVector, and PersistenceImage features",
        "differentiable_comparison_note": "Cosine/L2/correlation over cached vectors are differentiable with respect to the vectors; this code does not backpropagate through GUDHI diagram vectorization.",
        "aggregate_similarity": aggregate_similarity,
        "component_count": len(available_methods),
        "available_methods": available_methods,
        "weight_total": float(weight_total),
        "components": components,
    }


def persistence_landscape_vector_similarity(query_topology: dict[str, Any], memory_topology: dict[str, Any]) -> dict[str, Any]:
    q_by_dim = persistence_landscape_vector(query_topology)
    m_by_dim = persistence_landscape_vector(memory_topology)
    if not q_by_dim or not m_by_dim:
        return {
            "available": False,
            "source": "gudhi.representations.Landscape.vector",
            "reason": "missing_gudhi_landscape_vector",
            "query_dims": sorted(q_by_dim.keys()),
            "memory_dims": sorted(m_by_dim.keys()),
        }
    q_vec, m_vec, dims = _concatenate_landscape_vectors(q_by_dim, m_by_dim)
    if q_vec.size == 0 or m_vec.size == 0:
        return {
            "available": False,
            "source": "gudhi.representations.Landscape.vector",
            "reason": "empty_concatenated_landscape_vector",
            "query_dims": sorted(q_by_dim.keys()),
            "memory_dims": sorted(m_by_dim.keys()),
        }
    q_norm = float(np.linalg.norm(q_vec))
    m_norm = float(np.linalg.norm(m_vec))
    denom = q_norm * m_norm
    cosine = float(np.dot(q_vec, m_vec) / denom) if denom > 1e-12 else (1.0 if float(np.linalg.norm(q_vec - m_vec)) <= 1e-12 else 0.0)
    l2_distance = float(np.linalg.norm(q_vec - m_vec))
    l2_similarity = float(1.0 / (1.0 + l2_distance))
    if q_vec.size > 1 and m_vec.size > 1 and float(np.std(q_vec)) > 1e-12 and float(np.std(m_vec)) > 1e-12:
        correlation = float(np.corrcoef(q_vec, m_vec)[0, 1])
    else:
        correlation = cosine
    return {
        "available": True,
        "source": "gudhi.representations.Landscape.vector",
        "comparison_space": "concatenated sampled persistence landscape lambda_k(t) vectors by homology dimension",
        "differentiable_comparison_note": "cosine, L2, and correlation are differentiable vector comparisons; the current GUDHI vectorizer is a cached NumPy transform, not a torch-native differentiable layer",
        "dims": dims,
        "query_dims": sorted(q_by_dim.keys()),
        "memory_dims": sorted(m_by_dim.keys()),
        "overlap_dim": int(q_vec.size),
        "query_norm": q_norm,
        "memory_norm": m_norm,
        "cosine": cosine,
        "l2_distance": l2_distance,
        "l2_similarity": l2_similarity,
        "correlation": correlation,
    }


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
    values.extend(_commutative_algebra_signature_values(topology)[: length])
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return values[:length]



def _commutative_algebra_signature_values(topology: dict[str, Any]) -> list[float]:
    ca = topology.get("commutative_algebra", {}) if isinstance(topology, dict) else {}
    values: list[float] = []
    chain_keys = (
        "two_parameter_chain_presentation_diagnostics",
        "multiparameter_chain_presentation_diagnostics",
    )
    legacy_keys = ("two_parameter_free_resolution", "multiparameter_free_resolution_proxy")
    keys = chain_keys if any(isinstance(ca.get(key), dict) for key in chain_keys) else legacy_keys
    for key in keys:
        fr = ca.get(key, {}) if isinstance(ca.get(key), dict) else {}
        for row in fr.get("free_chain_modules", []) if isinstance(fr.get("free_chain_modules"), list) else []:
            if isinstance(row, dict):
                values.append(float(row.get("rank", row.get("rank_upper_bound", 0.0)) or 0.0))
        det_summary = fr.get("determinantal_ideal_summary", {}) if isinstance(fr.get("determinantal_ideal_summary"), dict) else {}
        fit_summary = fr.get("fitting_ideal_summary", {}) if isinstance(fr.get("fitting_ideal_summary"), dict) else {}
        be_summary = fr.get("buchsbaum_eisenbud_summary", {}) if isinstance(fr.get("buchsbaum_eisenbud_summary"), dict) else {}
        if not det_summary and isinstance(fr.get("determinantal_ideals"), dict):
            maps = fr["determinantal_ideals"].get("maps", {}) if isinstance(fr["determinantal_ideals"].get("maps"), dict) else {}
            det_summary = {"map_count": len(maps)}
        if not fit_summary and isinstance(fr.get("fitting_ideals"), dict):
            maps = fr["fitting_ideals"].get("maps", {}) if isinstance(fr["fitting_ideals"].get("maps"), dict) else {}
            fit_summary = {"map_count": len(maps)}
        if not be_summary and isinstance(fr.get("buchsbaum_eisenbud"), dict):
            be = fr["buchsbaum_eisenbud"]
            be_summary = {
                "composition_zero_checks": len(be.get("composition_zero_checks", [])) if isinstance(be.get("composition_zero_checks"), list) else 0,
                "exact_chain_modules": sum(1 for row in be.get("rank_exactness_checks", []) if isinstance(row, dict) and row.get("exact_at_chain_module_over_F2_incidence")) if isinstance(be.get("rank_exactness_checks"), list) else 0,
            }
        values.extend([
            float(det_summary.get("map_count", 0.0) or 0.0),
            float(det_summary.get("nonzero_sampled_generators", 0.0) or 0.0),
            float(fit_summary.get("map_count", 0.0) or 0.0),
            float(fit_summary.get("bounded_invariant_count", 0.0) or 0.0),
            float(be_summary.get("composition_zero_checks", 0.0) or 0.0),
            float(be_summary.get("exact_chain_modules", 0.0) or 0.0),
        ])
    return values


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
