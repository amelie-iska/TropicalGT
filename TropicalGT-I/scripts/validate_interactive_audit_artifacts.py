#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_HTML = {
    "embedding_map": ("got_embedding_map_3d.html", ("Graph-of-thought embedding-space trajectory map", "actual graph_state PCA")),
    "trajectory_nll": ("got_trajectory_pca_3d.html", ("Graph-of-thought branching trajectory", "raw NLL")),
    "trajectory_nll_density": ("got_nll_density_cloud_pca_3d.html", ("3D PCA NLL density cloud", "Gaussian", "actual model GoT state anchors", "not a model state")),
    "full_complex": ("got_full_trajectory_complex.html", ("Full graph-of-thought trajectory filtered simplicial complex", "play filtration", "filtration backend=")),
    "full_simplex_tree": ("got_full_trajectory_simplex_tree_3d.html", ("Full graph-of-thought trajectory GUDHI SimplexTree face-coface poset", "face-coface poset view", "GUDHI SimplexTree")),
    "probability_complex": ("got_full_trajectory_complex_jensen_shannon.html", ("probability filtered simplicial complex", "Jensen-Shannon")),
    "probability_simplex_tree": ("got_full_trajectory_simplex_tree_3d_jensen_shannon.html", ("probability", "SimplexTree", "Jensen-Shannon")),
    "step_complex_index": ("reasoning_step_complex_maps/index.html", ("Reasoning step filtered simplicial complex maps",)),
    "tropical_support": ("tropical_support_heatmap.html", ("Tropical", "support")),
    "tropical_fan": ("tropical_fan_diagnostics.html", ("Tropical fan diagnostics", "one dimensional cones", "Macaulay2", "not a multigraded free-resolution")),
    "graphcg": ("graphcg_direction_cosines.html", ("GraphCG", "full-rank direction audit")),
    "analogical_index": ("analogical_memory_topk_index.html", ("Analogical top-k probability correspondences",)),
    "analogical_map": ("analogical_memory_map_02.html", ("Analogical", "probability-matched correspondence", "filtered-complex certificate")),
    "analogical_simplex_tree": ("analogical_simplex_tree_analogy.html", ("Analogical simplex-tree analogy", "finite simplex-tree rows", "preserved face-to-coface chains")),
    "trajectory_barcode": ("trajectory_persistence/persistence_barcode.html", ("Trajectory", "barcode")),
    "trajectory_bifiltration": ("trajectory_persistence/two_parameter_bifiltration.html", ("Trajectory 2-parameter persistence over F2[x_level,x_radius]", "2-parameter module fibers", "Miller-Sturmfels staircase", "H0 fiber rank")),
    "trajectory_betti": ("trajectory_persistence/persistence_module_betti.html", ("Trajectory", "Betti", "2D matrix", "decorative 3D")),
    "trajectory_representations": ("trajectory_persistence/persistence_representations.html", ("Trajectory", "GUDHI persistence vectorization", "Fast train", "eval features")),
    "trajectory_landscapes": ("trajectory_persistence/persistence_landscapes.html", ("Trajectory", "Actual GUDHI persistence landscape functions", "lambda_1(t)", "not norm-only summaries")),
}

REQUIRED_JSON = {
    "scaling_tree": "inference_scaling_tree.json",
    "trajectory_payload": "got_trajectory_payloads.json",
    "nll_density_payload": "got_nll_density_cloud_payload.json",
    "embedding_payload": "got_embedding_map_payloads.json",
    "full_complex_payload": "got_full_trajectory_complex_payload.json",
    "step_manifest": "reasoning_step_complex_maps/manifest.json",
    "inference_audit": "inference_audit.json",
    "tropical_support_payload": "tropical_support_payload.json",
    "tropical_fan_diagnostics": "tropical_fan_diagnostics.json",
    "graphcg_payload": "graphcg_direction_cosines_payload.json",
    "analogical_simplex_tree_analogy": "analogical_simplex_tree_analogy.json",
    "trajectory_bifiltration_payload": "trajectory_level_radius_bifiltration.json",
    "trajectory_landscapes_payload": "trajectory_persistence/persistence_landscapes.json",
}


EVIDENCE_GAP_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("row_coverage", ("rows available", "expected at least")),
    ("persistence_landscapes", ("persistence landscapes", "persistence_landscapes", "lambda_k", "landscape")),
    (
        "cas_resolution_certificate",
        (
            "free-resolution",
            "cas certificate",
            "be/fitting",
            "cas execution manifest",
            "unavailable cas guard",
            "chain-presentation",
            "finite-chain",
        ),
    ),
    ("nll_density_surface", ("nll density", "nll surface", "local embedding-neighborhood surface", "pc3 z-axis")),
    ("tropical_support", ("tropical support", "wall margin", "support assignment")),
    ("graphcg_direction_audit", ("graphcg",)),
    ("embedding_map_identity", ("embedding map", "trajectory identity", "graph_state", "parent-child")),
    (
        "trajectory_overlay_radius",
        (
            "trajectory complex overlay",
            "radius slider",
            "slider contract",
            "full trajectory complex",
            "probability trajectory complex",
        ),
    ),
    ("analogical_memory", ("analogical", "jensen-shannon", "top-k")),
    ("reasoning_step_contracts", ("reasoning-step", "reasoning step", "complex fingerprint", "fingerprint basis")),
    ("simplex_tree_poset", ("simplex tree", "simplex-tree", "face-coface")),
    ("bifiltration_module", ("bifiltration", "miller-sturmfels", "structure map")),
    ("browser_index", ("browser index", "codex browser")),
    ("missing_artifact", ("missing json", "missing html", "references missing", "missing")),
)


EVIDENCE_GAP_ACTIONS = {
    "row_coverage": "Regenerate or collect the required number of real audit rows; do not duplicate rows to satisfy min-row gates.",
    "missing_artifact": "Regenerate the missing artifact from the original model/audit payload or keep the bundle failed; do not create placeholder evidence.",
    "cas_resolution_certificate": "Attach a real CAS certificate or an explicit safe-unavailable CAS guard; chain diagnostics cannot stand in for a resolution.",
    "persistence_landscapes": "Compute real GUDHI/persim lambda_k(t) landscape rows from persistence intervals, or render an explicit unavailable landscape state.",
    "nll_density_surface": "Regenerate NLL density/surface payloads with the current visual-layer contracts from model GoT anchors.",
    "tropical_support": "Regenerate tropical support payloads with trace-backed support, margin, and no-normal-fan-certificate scope.",
    "graphcg_direction_audit": "Regenerate GraphCG direction evidence with per-direction rows, contiguous ids, and bounded top-active summaries.",
    "embedding_map_identity": "Regenerate embedding-map payloads with graph_state provenance, branch/depth metadata, and parent-child transitions.",
    "trajectory_overlay_radius": "Regenerate radius filtration overlays and slider sidecars from canonical filtered complexes.",
    "analogical_memory": "Regenerate probability-vector analogical reports from model probabilities and certified simplex-tree correspondence checks.",
    "reasoning_step_contracts": "Regenerate per-step complex pages and manifest contracts from each candidate filtered simplicial object.",
    "simplex_tree_poset": "Regenerate GUDHI SimplexTree face-to-coface poset sidecars from canonical complexes.",
    "bifiltration_module": "Regenerate the two-parameter bifiltration sidecar from actual level/radius fibers and structure maps.",
    "browser_index": "Rebuild browser indexes only after the referenced real artifacts exist.",
    "other": "Inspect the exact strict-validator error and repair the underlying source artifact; this inventory does not downgrade failures.",
}


def _evidence_gap_category(error: str) -> str:
    text = error.lower()
    for category, needles in EVIDENCE_GAP_CATEGORY_RULES:
        if any(needle in text for needle in needles):
            return category
    return "other"


def _build_evidence_gap_inventory(errors: list[str]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for error in errors:
        category = _evidence_gap_category(error)
        counts[category] += 1
        examples.setdefault(category, [])
        if len(examples[category]) < 5:
            examples[category].append(error)
    categories = [
        {
            "category": category,
            "count": count,
            "examples": examples.get(category, []),
            "required_action": EVIDENCE_GAP_ACTIONS.get(category, EVIDENCE_GAP_ACTIONS["other"]),
        }
        for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "schema_version": "tropicalgt.interactive_audit_evidence_gap_inventory.v1",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "strict_validation_still_required": True,
        "gap_count": len(errors),
        "category_counts": {category: count for category, count in sorted(counts.items())},
        "categories": categories,
        "policy": (
            "This inventory classifies strict-validator failures so stale or incomplete bundles can be repaired from real payloads. "
            "It does not make an artifact valid, synthesize evidence, or replace missing CAS/topology/geometry/algebra objects."
        ),
    }


class ArtifactValidationError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser message is platform-dependent
        raise ArtifactValidationError(f"invalid json: {path}: {exc}") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        raise ArtifactValidationError(f"invalid text/html: {path}: {exc}") from exc


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _trajectory_finite_persistence_interval_count(row_dir: Path) -> int:
    path = row_dir / "trajectory_topological_algebra.json"
    if not path.exists():
        return -1
    try:
        topology = _read_json(path)
    except Exception:
        return -1
    persistence = topology.get("persistence", {}) if isinstance(topology, dict) and isinstance(topology.get("persistence"), dict) else {}
    intervals = persistence.get("intervals", []) if isinstance(persistence.get("intervals"), list) else []
    count = 0
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        try:
            birth = float(interval.get("birth", 0.0) or 0.0)
            death = interval.get("death")
            death_value = float(death) if death is not None else None
        except (TypeError, ValueError):
            continue
        if death_value is not None and math.isfinite(death_value) and death_value > birth:
            count += 1
    return count


def _pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0, 0), dtype=float)
    diffs = values[:, None, :] - values[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diffs * diffs, axis=-1), 0.0))
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def _distance_diagnostics(embeddings: np.ndarray, coords: np.ndarray) -> dict[str, float]:
    original = _pairwise_euclidean(embeddings)
    projected = _pairwise_euclidean(coords)
    mask = np.triu(np.ones_like(original, dtype=bool), k=1)
    target = original[mask]
    realized = projected[mask]
    if target.size >= 2 and float(np.std(target)) > 1e-12 and float(np.std(realized)) > 1e-12:
        corr = float(np.corrcoef(target, realized)[0, 1])
    else:
        corr = 1.0
    denom = max(float(np.dot(target, target)), 1e-12)
    stress = math.sqrt(float(np.dot(target - realized, target - realized)) / denom)
    return {"pairwise_distance_correlation": corr, "normalized_stress": stress}


def _assert(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _validate_real_free_resolution_guard(real: Any, errors: list[str], label: str) -> None:
    _assert(isinstance(real, dict), errors, f"{label} missing nested real free-resolution guard")
    if not isinstance(real, dict):
        return
    _assert(real.get("schema_version") == "tropicalgt.real_free_resolution.v1", errors, f"{label} real free-resolution guard has wrong schema")
    contract = real.get("certificate_contract", {}) if isinstance(real.get("certificate_contract"), dict) else {}
    _assert(contract.get("schema_version") == "tropicalgt.cas_free_resolution_contract.v1", errors, f"{label} real free-resolution guard lacks CAS certificate contract")
    _assert(contract.get("no_proxy_or_fallback") is True, errors, f"{label} CAS certificate contract allows proxy/fallback data")
    _assert("finite chain diagnostics" in str(contract.get("no_proxy_policy", "")), errors, f"{label} CAS certificate contract lacks finite-chain no-proxy policy")
    paper = real.get("paper_method_contract") if isinstance(real.get("paper_method_contract"), dict) else contract.get("paper_method_contract", {})
    paper = paper if isinstance(paper, dict) else {}
    _assert(paper.get("schema_version") == "tropicalgt.be_fitting_method_contract.v1", errors, f"{label} lacks BE/Fitting paper method contract")
    _assert(paper.get("arxiv_id") == "2210.11433v1", errors, f"{label} BE/Fitting paper method contract has wrong arXiv id")
    _assert(paper.get("no_proxy_or_fallback") is True, errors, f"{label} BE/Fitting method contract allows proxy/fallback data")
    manifest = real.get("cas_execution_manifest", {}) if isinstance(real.get("cas_execution_manifest"), dict) else {}
    _assert(manifest.get("schema_version") == "tropicalgt.cas_execution_manifest.v1", errors, f"{label} lacks CAS execution manifest")
    _assert(manifest.get("no_proxy_or_fallback") is True, errors, f"{label} CAS execution manifest allows proxy/fallback data")
    entries = manifest.get("backend_entries", []) if isinstance(manifest.get("backend_entries"), list) else []
    _assert(bool(entries), errors, f"{label} CAS execution manifest lacks backend entries")
    _assert(all(isinstance(row, dict) and row.get("certificate_required_before_rendering") is True for row in entries), errors, f"{label} CAS execution manifest does not require certificates before rendering")
    _assert(isinstance(real.get("command_templates"), dict), errors, f"{label} lacks CAS command templates")
    available = real.get("available") is True
    cert_flags = [
        "certificate_attached",
        "real_free_resolution_certified",
        "total_graded_resolution_certified",
        "ungraded_resolution_certified",
        "multigraded_free_resolution_certified",
        "exactness_certified",
        "minimality_certified",
        "safe_to_render_as_real_free_resolution",
        "safe_to_render_as_total_graded_resolution",
        "safe_to_render_as_multigraded_free_resolution",
    ]
    for key in cert_flags:
        _assert(key in real, errors, f"{label} real free-resolution guard lacks {key}")
    if available:
        _assert(real.get("status") == "certified", errors, f"{label} available CAS guard is not marked certified")
        _assert(real.get("certificate_attached") is True, errors, f"{label} available CAS guard lacks attached certificate")
        _assert(real.get("exactness_certified") is True, errors, f"{label} available CAS guard lacks exactness certificate")
        _assert(real.get("real_free_resolution_certified") is True, errors, f"{label} available CAS guard does not certify a real free resolution")
        artifacts = real.get("cas_artifacts", {}) if isinstance(real.get("cas_artifacts"), dict) else {}
        _assert(bool(artifacts), errors, f"{label} certified CAS guard lacks artifacts")
        _assert(isinstance(real.get("free_resolution_summary"), dict) and real.get("free_resolution_summary", {}).get("available") is True, errors, f"{label} certified CAS guard lacks renderable resolution summary")
        if real.get("safe_to_render_as_multigraded_free_resolution") is True:
            _assert(real.get("multigraded_free_resolution_certified") is True, errors, f"{label} multigraded render flag is not backed by multigraded certification")
        if real.get("total_graded_resolution_certified") is True or real.get("ungraded_resolution_certified") is True:
            _assert(real.get("safe_to_render_as_multigraded_free_resolution") is not True, errors, f"{label} non-multigraded CAS output is incorrectly renderable as multigraded")
        cert_summary = artifacts.get("certificate_summary", {}) if isinstance(artifacts.get("certificate_summary"), dict) else {}
        _assert(cert_summary.get("no_proxy_policy"), errors, f"{label} certified CAS artifact lacks certificate no-proxy summary")
    else:
        _assert(str(real.get("status", "")) in {"unavailable_no_certificate", "backend_not_installed", "backend_error", "timeout", "unsupported_ring", "invalid_grading", "parse_error", "certificate_failed", "disabled_by_environment", "complexity_guard"}, errors, f"{label} unavailable CAS guard has invalid status")
        _assert(str(real.get("reason", "")).strip() != "", errors, f"{label} unavailable CAS guard lacks exact reason")
        _assert(real.get("safe_unavailable_render") is True, errors, f"{label} unavailable CAS guard is not safe to render as unavailable")
        _assert(real.get("cas_artifacts") == {}, errors, f"{label} unavailable CAS guard contains CAS artifacts")
        _assert(all(real.get(key) is False for key in cert_flags), errors, f"{label} unavailable CAS guard sets certified/render flags")
        unavailable = real.get("unavailable_diagnostic", {}) if isinstance(real.get("unavailable_diagnostic"), dict) else {}
        _assert(unavailable.get("safe_to_render_only_as_unavailable") is True, errors, f"{label} unavailable diagnostic is not restricted to unavailable rendering")
        _assert("Do not substitute chain diagnostics" in str(unavailable.get("no_proxy_policy", "")), errors, f"{label} unavailable diagnostic lacks chain-diagnostic no-proxy policy")


def _slider_contract_path(html_path: Path) -> Path:
    return html_path.with_name(f"{html_path.stem}_slider_contract.json")


def _simplex_tree_poset_contract_path(html_path: Path) -> Path:
    return html_path.with_name(f"{html_path.stem}_simplex_tree_poset_contract.json")


def _validate_simplex_tree_poset_contract(
    contract_path: Path,
    errors: list[str],
    label: str,
    *,
    required_available: bool = True,
) -> dict[str, Any]:
    _assert(contract_path.exists(), errors, f"{label} missing simplex-tree poset contract sidecar {contract_path.name}")
    if not contract_path.exists():
        return {}
    payload = _read_json(contract_path)
    _assert(isinstance(payload, dict), errors, f"{label} simplex-tree poset contract is not an object")
    if not isinstance(payload, dict):
        return {}
    _assert(payload.get("schema_version") == "tropicalgt.simplex_tree_poset.v1", errors, f"{label} simplex-tree poset contract has wrong schema")
    _assert(payload.get("source") == "gudhi_canonical_complex(filtered_simplicial_object).simplex_tree", errors, f"{label} simplex-tree poset contract has wrong source")
    _assert(payload.get("actual_data_only") is True, errors, f"{label} simplex-tree poset contract is not actual-data-only")
    _assert(payload.get("no_proxy_or_fallback") is True, errors, f"{label} simplex-tree poset contract allows proxy/fallback data")
    _assert(str(payload.get("html_file", "")).strip() != "", errors, f"{label} simplex-tree poset contract lacks html file")
    if not required_available:
        _assert(payload.get("available") is False, errors, f"{label} simplex-tree poset contract should be unavailable")
        _assert(payload.get("safe_to_render_simplex_tree") is False, errors, f"{label} unavailable simplex-tree poset contract is marked safe")
        _assert(payload.get("safe_unavailable_render") is True, errors, f"{label} unavailable simplex-tree poset contract is not safe unavailable render")
        _assert(str(payload.get("reason", "")).strip() != "", errors, f"{label} unavailable simplex-tree poset contract lacks reason")
        _assert(int(_finite_float(payload.get("displayed_simplex_count"), -1.0)) == 0, errors, f"{label} unavailable simplex-tree poset contract displays simplices")
        return payload
    _assert(payload.get("available") is True, errors, f"{label} simplex-tree poset contract is unavailable")
    _assert(payload.get("backend") == "gudhi.SimplexTree", errors, f"{label} simplex-tree poset contract is not backed by GUDHI")
    _assert(payload.get("safe_to_render_simplex_tree") is True, errors, f"{label} simplex-tree poset contract is unsafe")
    _assert(payload.get("layout") == "model_embedding_barycentric_face_coface_poset", errors, f"{label} simplex-tree poset layout is wrong")
    _assert(payload.get("not_disconnected_simplex_columns") is True, errors, f"{label} simplex-tree poset permits disconnected simplex columns")
    _assert(payload.get("empty_simplex_root_present") is True, errors, f"{label} simplex-tree poset lacks empty-simplex root")
    _assert(int(_finite_float(payload.get("displayed_simplex_count"), 0.0)) > 0, errors, f"{label} simplex-tree poset displays no simplices")
    _assert(int(_finite_float(payload.get("source_simplex_count"), 0.0)) >= int(_finite_float(payload.get("displayed_simplex_count"), 0.0)), errors, f"{label} simplex-tree poset source/displayed count mismatch")
    _assert(int(_finite_float(payload.get("actual_face_to_coface_cover_edges"), 0.0)) >= int(_finite_float(payload.get("empty_root_vertex_cover_edges"), 0.0)), errors, f"{label} simplex-tree poset cover-edge count is inconsistent")
    _assert(payload.get("primary_edges") == "actual_face_to_coface_covers", errors, f"{label} simplex-tree poset primary edges are not face-to-coface covers")
    _assert(payload.get("optional_prefix_links_visible") == "legendonly", errors, f"{label} simplex-tree trie prefix links are not legend-only")
    _assert(payload.get("position_source") == "model_embedding_barycenters_with_dimension_and_filtration_lift", errors, f"{label} simplex-tree poset position source is wrong")
    readability = payload.get("readability_contract", {}) if isinstance(payload.get("readability_contract"), dict) else {}
    _assert(readability.get("schema_version") == "tropicalgt.simplex_tree_readability.v1", errors, f"{label} simplex-tree readability contract has wrong schema")
    _assert(readability.get("summary_first_default") is True, errors, f"{label} simplex-tree readability contract is not summary-first")
    _assert(readability.get("representative_inclusions_visible_by_default") is True, errors, f"{label} simplex-tree readability contract does not show representative inclusions")
    _assert(readability.get("dense_face_to_coface_visibility_policy") == "legendonly_when_actual_cover_edges_exceed_120", errors, f"{label} simplex-tree readability contract has wrong dense-edge policy")
    _assert(isinstance(readability.get("dimension_count_rows"), list) and bool(readability.get("dimension_count_rows")), errors, f"{label} simplex-tree readability contract lacks dimension counts")
    _assert(isinstance(readability.get("filtration_histogram"), list), errors, f"{label} simplex-tree readability contract lacks filtration histogram")
    _assert(readability.get("no_proxy_or_fallback") is True, errors, f"{label} simplex-tree readability contract lacks no-proxy flag")
    _assert(payload.get("all_non_vertex_simplices_have_face_cover_edges") is True, errors, f"{label} simplex-tree poset has non-vertex simplices without face-cover edges")
    return payload


def _validate_radius_slider_contract(contract_path: Path, errors: list[str], label: str) -> dict[str, Any]:
    _assert(contract_path.exists(), errors, f"{label} missing radius slider contract sidecar {contract_path.name}")
    if not contract_path.exists():
        return {}
    payload = _read_json(contract_path)
    _assert(isinstance(payload, dict), errors, f"{label} radius slider contract is not an object")
    if not isinstance(payload, dict):
        return {}
    _assert(payload.get("schema_version") == "tropicalgt.radius_filtration_slider_contract.v1", errors, f"{label} radius slider contract has wrong schema")
    _assert(payload.get("source") == "canonical_gudhi_filtered_complex_simplices", errors, f"{label} radius slider contract has wrong source")
    _assert(payload.get("actual_data_only") is True, errors, f"{label} radius slider contract is not actual-data-only")
    _assert(payload.get("no_proxy_or_fallback") is True, errors, f"{label} radius slider contract allows proxy/fallback data")
    _assert(payload.get("radius_filtration") is True, errors, f"{label} slider contract is not a radius filtration")
    _assert(payload.get("threshold_order") == "ascending_min_to_max", errors, f"{label} radius slider contract does not declare ascending min-to-max order")
    _assert(payload.get("thresholds_ascending") is True, errors, f"{label} radius slider thresholds are not ascending")
    _assert(int(_finite_float(payload.get("frame_count"), 0.0)) == int(_finite_float(payload.get("threshold_count"), -1.0)), errors, f"{label} radius slider frame/threshold count mismatch")
    _assert(isinstance(payload.get("frames"), list) and bool(payload.get("frames")), errors, f"{label} radius slider contract lacks frame rows")
    _assert(payload.get("monotone_visible_counts") is True, errors, f"{label} radius slider visible counts are not monotone")
    _assert(payload.get("first_frame_disjoint_vertices_only") is True, errors, f"{label} initial radius frame is not vertex-only")
    _assert(int(_finite_float(payload.get("first_frame_vertex_count"), 0.0)) > 0, errors, f"{label} initial radius frame has no vertices")
    _assert(int(_finite_float(payload.get("first_frame_solid_edge_count"), -1.0)) == 0, errors, f"{label} initial radius frame shows solid edges")
    _assert(int(_finite_float(payload.get("first_frame_filled_face_count"), -1.0)) == 0, errors, f"{label} initial radius frame shows filled faces")
    _assert(int(_finite_float(payload.get("first_frame_dotted_overlay_count"), -1.0)) == 0, errors, f"{label} initial radius frame shows dotted overlays")
    _assert(payload.get("solid_lines_semantics") == "radius-filtered 1-simplices only", errors, f"{label} solid-line semantics are not radius simplices")
    _assert(payload.get("filled_faces_semantics") == "radius-gated 2-simplices only", errors, f"{label} filled-face semantics are not radius gated")
    _assert(payload.get("dotted_lines_semantics") == "causal_decoding_or_direction_overlay_only_and_radius_gated", errors, f"{label} dotted-line semantics are not causal/decoding/direction overlays")
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else []
    thresholds = [_finite_float(row.get("threshold")) for row in frames if isinstance(row, dict)]
    _assert(all(math.isfinite(value) for value in thresholds), errors, f"{label} radius slider frame thresholds are not finite")
    _assert(all(a <= b + 1e-12 for a, b in zip(thresholds, thresholds[1:])), errors, f"{label} radius slider frame thresholds decrease")
    return payload


def _validate_trajectory_overlay_view_contract(
    contract: Any,
    errors: list[str],
    label: str,
    *,
    expected_metric: str,
    required_available: bool,
) -> None:
    _assert(isinstance(contract, dict), errors, f"{label} overlay contract is missing")
    if not isinstance(contract, dict):
        return
    _assert(contract.get("schema_version") == "tropicalgt.trajectory_complex_overlay_view_contract.v1", errors, f"{label} overlay contract has wrong schema")
    _assert(contract.get("actual_data_only") is True, errors, f"{label} overlay contract is not actual-data-only")
    _assert(contract.get("no_proxy_or_fallback") is True, errors, f"{label} overlay contract allows proxy/fallback data")
    if not required_available:
        _assert(contract.get("available") is False, errors, f"{label} overlay contract should be unavailable")
        _assert(bool(contract.get("reason")), errors, f"{label} unavailable overlay contract lacks reason")
        return
    _assert(contract.get("available") is True, errors, f"{label} overlay contract is unavailable")
    _assert(contract.get("distance_metric") == expected_metric, errors, f"{label} overlay contract has wrong distance metric")
    _assert(contract.get("expected_distance_metric") == expected_metric, errors, f"{label} overlay contract has wrong expected metric")
    _assert(contract.get("radius_filtration") is True, errors, f"{label} overlay contract is not a radius filtration")
    _assert(contract.get("solid_lines_semantics") == "radius-filtered 1-simplices only", errors, f"{label} solid-line semantics are wrong")
    _assert(contract.get("filled_faces_semantics") == "radius-gated 2-simplices only", errors, f"{label} filled-face semantics are wrong")
    _assert(contract.get("dotted_lines_semantics") == "trajectory and decoding/order overlays only", errors, f"{label} dotted-line semantics are wrong")
    _assert(contract.get("solid_edges_from_radius_simplices") is True, errors, f"{label} does not reserve solid edges for radius simplices")
    _assert(contract.get("dotted_edges_reserved_for_overlays") is True, errors, f"{label} does not reserve dotted edges for overlays")
    _assert(_finite_float(contract.get("vertex_count"), 0.0) > 0, errors, f"{label} has no vertices")
    _assert(contract.get("source_counts_match_summary") is True, errors, f"{label} source counts do not match summary")
    _assert(contract.get("simplex_tree_backend") == "gudhi.SimplexTree", errors, f"{label} missing GUDHI SimplexTree backend")
    _assert(contract.get("trajectory_overlay_source") == "graph_of_thought_parent_edges", errors, f"{label} trajectory overlay source is wrong")
    _assert(contract.get("trajectory_overlay_distance_metric") == expected_metric, errors, f"{label} trajectory overlay metric is wrong")
    _assert(contract.get("decoding_overlay_source") == "graph_of_thought_parent_decoding_order", errors, f"{label} decoding overlay source is wrong")
    _assert(contract.get("decoding_overlay_distance_metric") == expected_metric, errors, f"{label} decoding overlay metric is wrong")
    if _finite_float(contract.get("decoding_overlay_edge_count"), 0.0) > 0:
        _assert(contract.get("decoding_overlay_edges_are_dotted") is True, errors, f"{label} decoding overlay edges are not dotted")
        _assert(contract.get("decoding_overlay_edges_are_directed") is True, errors, f"{label} decoding overlay edges are not directed")
    _assert(contract.get("safe_to_render_overlay_semantics") is True, errors, f"{label} overlay contract is unsafe")


def _validate_reasoning_step_slider_summary(
    summary: Any,
    sidecar: dict[str, Any],
    errors: list[str],
    label: str,
    *,
    expected_contract_file: str,
    expected_html_file: str,
) -> None:
    _assert(isinstance(summary, dict), errors, f"{label} missing reasoning-step radius slider summary")
    if not isinstance(summary, dict):
        return
    _assert(summary.get("schema_version") == "tropicalgt.reasoning_step_radius_slider_summary.v1", errors, f"{label} reasoning-step radius slider summary has wrong schema")
    _assert(summary.get("source") == f"reasoning_step_complex_maps/{expected_contract_file}", errors, f"{label} reasoning-step radius slider summary has wrong source")
    _assert(summary.get("contract_file") == expected_contract_file, errors, f"{label} reasoning-step radius slider summary contract file mismatch")
    _assert(summary.get("html_file") == expected_html_file, errors, f"{label} reasoning-step radius slider summary html file mismatch")
    _assert(summary.get("contract_schema_version") == sidecar.get("schema_version"), errors, f"{label} reasoning-step radius slider summary schema/sidecar mismatch")
    _assert(summary.get("contract_source") == sidecar.get("source"), errors, f"{label} reasoning-step radius slider summary source/sidecar mismatch")
    for key in [
        "actual_data_only",
        "no_proxy_or_fallback",
        "radius_filtration",
        "thresholds_ascending",
        "first_frame_disjoint_vertices_only",
        "monotone_visible_counts",
        "monotone_solid_radius_edges",
        "monotone_filled_radius_faces",
        "initial_radius_frame_hides_dotted_overlays",
        "initial_radius_frame_hides_solid_edges_and_faces",
    ]:
        _assert(summary.get(key) == (sidecar.get(key) is True), errors, f"{label} reasoning-step radius slider summary {key} mismatch")
    _assert(summary.get("threshold_order") == sidecar.get("threshold_order"), errors, f"{label} reasoning-step radius slider summary threshold order mismatch")
    for key in [
        "threshold_count",
        "frame_count",
        "first_frame_vertex_count",
        "first_frame_solid_edge_count",
        "first_frame_filled_face_count",
        "first_frame_dotted_overlay_count",
    ]:
        _assert(int(_finite_float(summary.get(key), -999.0)) == int(_finite_float(sidecar.get(key), -998.0)), errors, f"{label} reasoning-step radius slider summary {key} mismatch")
    _assert(summary.get("solid_lines_semantics_ok") is True, errors, f"{label} reasoning-step radius slider summary rejects solid-line semantics")
    _assert(summary.get("filled_faces_semantics_ok") is True, errors, f"{label} reasoning-step radius slider summary rejects filled-face semantics")
    _assert(summary.get("dotted_lines_semantics_ok") is True, errors, f"{label} reasoning-step radius slider summary rejects dotted-line semantics")
    _assert(summary.get("safe_to_render_radius_filtration") is True, errors, f"{label} reasoning-step radius slider summary is not safe to render")


def _validate_reasoning_step_source_contract(
    contract: Any,
    step: dict[str, Any],
    errors: list[str],
    label: str,
) -> None:
    _assert(isinstance(contract, dict), errors, f"{label} missing reasoning-step complex source contract")
    if not isinstance(contract, dict):
        return
    _assert(contract.get("schema_version") == "tropicalgt.reasoning_step_complex_source_contract.v1", errors, f"{label} source contract has wrong schema")
    _assert(contract.get("source") == "candidate.filtered_simplicial_object", errors, f"{label} source contract does not cite candidate filtered object")
    _assert(contract.get("actual_data_only") is True, errors, f"{label} source contract is not actual-data-only")
    _assert(contract.get("no_proxy_or_fallback") is True, errors, f"{label} source contract allows proxy/fallback data")
    _assert(contract.get("candidate_record_id") == step.get("record_id"), errors, f"{label} source contract candidate record mismatch")
    _assert(contract.get("candidate_level") == step.get("level"), errors, f"{label} source contract level mismatch")
    _assert(contract.get("candidate_path") == step.get("path"), errors, f"{label} source contract path mismatch")
    _assert(contract.get("step_complex_fingerprint") == step.get("step_complex_fingerprint"), errors, f"{label} source contract fingerprint mismatch")
    _assert(contract.get("uses_global_trajectory_complex_as_proxy") is False, errors, f"{label} source contract allows global trajectory proxy")
    _assert(contract.get("uses_embedding_trajectory_map_as_proxy") is False, errors, f"{label} source contract allows embedding trajectory proxy")
    _assert(contract.get("uses_static_probability_complex_as_proxy") is False, errors, f"{label} source contract allows static probability proxy")
    summary = step.get("summary", {}) if isinstance(step.get("summary"), dict) else {}
    _assert(int(_finite_float(contract.get("displayed_vertex_count"), -1.0)) == int(_finite_float(summary.get("num_vertices"), -2.0)), errors, f"{label} source contract vertex count mismatch")
    _assert(int(_finite_float(contract.get("displayed_edge_count"), -1.0)) == int(_finite_float(summary.get("num_edges"), -2.0)), errors, f"{label} source contract edge count mismatch")
    _assert(int(_finite_float(contract.get("displayed_face_count"), -1.0)) == int(_finite_float(summary.get("num_two_simplices"), -2.0)), errors, f"{label} source contract face count mismatch")
    _assert(int(_finite_float(contract.get("displayed_vertex_count"), 0.0)) > 0, errors, f"{label} source contract has no displayed vertices")
    _assert(contract.get("source_counts_match_canonical_summary") is True, errors, f"{label} source contract counts do not match canonical summary")
    _assert(contract.get("safe_to_render_as_step_complex") is True, errors, f"{label} source contract is unsafe")
    _assert(isinstance(contract.get("displayed_vertex_labels_sample"), list) and bool(contract.get("displayed_vertex_labels_sample")), errors, f"{label} source contract lacks vertex labels")
    _assert(contract.get("simplex_tree_backend") == step.get("simplex_tree_backend"), errors, f"{label} source contract simplex-tree backend mismatch")
    _assert(contract.get("simplex_tree_available") == step.get("simplex_tree_available"), errors, f"{label} source contract simplex-tree availability mismatch")


def _html_has_plotly(html: str) -> bool:
    return "Plotly.newPlot" in html or "plotly" in html.lower()


def _ignored_ref(ref: str) -> bool:
    ref = html.unescape(str(ref)).strip()
    return not ref or ref.startswith(("#", "http://", "https://", "file://", "data:", "mailto:", "javascript:"))


def _relative_target(row_dir: Path, ref: str) -> Path | None:
    ref = html.unescape(str(ref)).strip()
    if _ignored_ref(ref):
        return None
    clean = ref.split("#", 1)[0].split("?", 1)[0]
    if not clean:
        return None
    return (row_dir / clean).resolve()


def _attr_values(markup: str, attr: str) -> list[str]:
    pattern = rf"""\b{re.escape(attr)}\s*=\s*([\"'])(.*?)\1"""
    return [html.unescape(value) for _, value in re.findall(pattern, markup, flags=re.IGNORECASE | re.DOTALL)]


def _tag_attrs(tag: str) -> dict[str, str]:
    return {
        name.lower(): html.unescape(value)
        for name, _, value in re.findall(r"""([:\w-]+)\s*=\s*([\"'])(.*?)\2""", tag, flags=re.IGNORECASE | re.DOTALL)
    }


def _tags(markup: str, name: str) -> list[str]:
    return re.findall(rf"""<{re.escape(name)}\b[^>]*>""", markup, flags=re.IGNORECASE | re.DOTALL)


def _has_class(attrs: dict[str, str], class_name: str) -> bool:
    return class_name in str(attrs.get("class", "")).split()


def _artifact_button_refs(row_dir: Path, markup: str, file_name: str, errors: list[str]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    broken: list[str] = []
    for tag in _tags(markup, "button"):
        attrs = _tag_attrs(tag)
        if not _has_class(attrs, "artifact"):
            continue
        sample = str(attrs.get("data-sample", "")).strip()
        src = str(attrs.get("data-src", "")).strip()
        _assert(sample != "", errors, f"{file_name} artifact button missing data-sample")
        _assert(bool(src), errors, f"{file_name} artifact button missing data-src")
        target = _relative_target(row_dir, src) if src else None
        if target is not None and not target.exists():
            broken.append(src)
        if sample != "" and src:
            refs.append((sample, src))
    _assert(not broken, errors, f"{file_name} has broken artifact button targets: {broken[:8]}")
    return refs


def _sample_payloads(markup: str, file_name: str, errors: list[str]) -> list[dict[str, Any]]:
    values = _attr_values(markup, "data-samples")
    if not values:
        return []
    try:
        payload = json.loads(values[0])
    except json.JSONDecodeError as exc:
        errors.append(f"{file_name} has invalid data-samples payload: {exc}")
        return []
    if not isinstance(payload, list):
        errors.append(f"{file_name} data-samples payload is not a list")
        return []
    return [row for row in payload if isinstance(row, dict)]


def _validate_file_set(row_dir: Path, errors: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for key, rel in REQUIRED_JSON.items():
        path = row_dir / rel
        _assert(path.exists(), errors, f"missing json {rel}")
        files[key] = str(path)
    for key, (rel, needles) in REQUIRED_HTML.items():
        path = row_dir / rel
        if key in {"analogical_index", "analogical_map"}:
            maps_path = row_dir / "analogical_simplicial_maps.json"
            if maps_path.exists():
                try:
                    map_payload = _read_json(maps_path)
                except ArtifactValidationError:
                    map_payload = {}
                if map_payload.get("available") is False and map_payload.get("reason") in {"missing_model_probability_query_complex", "missing_model_probability_codomain_complex", "no_non_self_model_memory"}:
                    files[key] = str(path)
                    continue
        if key == "analogical_map" and not path.exists() and (row_dir / "analogical_memory_retrieval.html").exists():
            path = row_dir / "analogical_memory_retrieval.html"
        _assert(path.exists(), errors, f"missing html {rel}")
        files[key] = str(path)
        if path.exists():
            html = _read_text(path)
            for needle in needles:
                _assert(needle in html, errors, f"{rel} missing marker {needle!r}")
            _assert("synthetic fallback" not in html, errors, f"{rel} contains retired synthetic persistence fallback text")
            _assert("module_beta0" not in html, errors, f"{rel} contains retired module_beta0 synthetic interval text")
            if key not in {"step_complex_index", "analogical_index"}:
                _assert(_html_has_plotly(html), errors, f"{rel} does not look like a Plotly/interactive page")
                _assert("plotly.min.js" in html, errors, f"{rel} does not use the local Plotly asset")
            if key in {"embedding_map", "full_complex", "analogical_map", "trajectory_barcode", "trajectory_betti"}:
                _assert("simplicial-object-plot" in html and "selected-complex-graph" in html, errors, f"{rel} does not render the selected complex as a live Plotly panel")
                _assert("plotly_click" in html, errors, f"{rel} does not support click-to-select for the simplicial detail panel")
            if key == "full_complex":
                _assert("model input" in html and "model output" in html, errors, f"{rel} does not expose model input/output in hover payload")
                _assert("play filtration min-to-max" in html, errors, f"{rel} does not use the flipped min-to-max filtration control")
            if key == "analogical_map":
                _assert("query trajectory complex" in html and "retrieved memory complex" in html, errors, f"{rel} is not comparing trajectory complexes")
                _assert("slider filters domain" in html and "sliders" in html, errors, f"{rel} is missing Plotly filtration slider for the correspondence certificate")
            if key == "tropical_support":
                support_audit = "observed supports only" in html and "top-support collapse rate" in html
                collapse_audit = "active-support collapse diagnostic" in html and "Collapse metrics" in html
                _assert(support_audit or collapse_audit, errors, f"{rel} is not the interpretable support audit view")
            if key == "graphcg":
                _assert(("Readable top-direction heatmap" in html) or ("GraphCG full-rank direction audit" in html and "heatmap shows all" in html), errors, f"{rel} does not use the readable GraphCG full-rank audit layout")
            if key == "trajectory_nll":
                _assert("selected-complex-graph" in html and "plotly_click" in html, errors, f"{rel} does not click-select an interactive reasoning-step complex")
                _assert("open interactive reasoning-step complex page" in html, errors, f"{rel} does not expose per-step complex page links")
    return files


def _validate_html_index(row_dir: Path, file_name: str, errors: list[str]) -> dict[str, Any]:
    index = row_dir / file_name
    if not index.exists():
        return {"available": False}
    html = _read_text(index)
    refs = _attr_values(html, "href") + _attr_values(html, "src")
    broken = []
    for ref in refs:
        target = _relative_target(row_dir, ref)
        if target is None:
            continue
        if not target.exists():
            broken.append(ref)
    _assert(not broken, errors, f"{file_name} has broken relative refs: {broken[:8]}")
    return {"available": True, "relative_refs": len([r for r in refs if not _ignored_ref(r)]), "broken_refs": broken}



def _validate_browser_index(row_dir: Path, errors: list[str]) -> dict[str, Any]:
    return _validate_html_index(row_dir, "browser_index.html", errors)


def _validate_codex_browser_index(row_dir: Path, errors: list[str]) -> dict[str, Any]:
    result = _validate_html_index(row_dir, "codex_browser_index.html", errors)
    if result.get("available"):
        markup = _read_text(row_dir / "codex_browser_index.html")
        _assert("Sample-first audit" in markup, errors, "codex_browser_index.html is not sample-first")
        sample_sections = [_tag_attrs(tag) for tag in _tags(markup, "section") if _has_class(_tag_attrs(tag), "sample")]
        _assert(bool(sample_sections), errors, "codex_browser_index.html has no sample cards")
        button_refs = _artifact_button_refs(row_dir, markup, "codex_browser_index.html", errors)
        _assert(bool(button_refs), errors, "codex_browser_index.html has no per-sample artifact buttons")
        samples = _sample_payloads(markup, "codex_browser_index.html", errors)
        if samples:
            sample_ids = {str(attrs.get("data-sample", "")).strip() for attrs in sample_sections}
            open_links = [_tag_attrs(tag).get("href", "") for tag in _tags(markup, "a") if _has_class(_tag_attrs(tag), "open-sample")]
            _assert(len(sample_sections) >= len(samples), errors, "codex_browser_index.html does not render one sample card per data-samples row")
            _assert(len(open_links) >= len(samples), errors, "codex_browser_index.html does not expose one open-sample link per data-samples row")
            button_set = set(button_refs)
            for sample in samples:
                sample_index = str(sample.get("index", "")).strip()
                _assert(sample_index in sample_ids, errors, f"codex_browser_index.html missing sample card for data-samples row {sample_index}")
                artifacts = [row for row in sample.get("artifacts", []) if isinstance(row, dict)]
                _assert(bool(artifacts), errors, f"codex_browser_index.html data-samples row {sample_index} has no artifact entries")
                for artifact in artifacts:
                    src = str(artifact.get("src", "")).strip()
                    _assert(bool(src), errors, f"codex_browser_index.html data-samples row {sample_index} has artifact without src")
                    if src:
                        _assert((sample_index, src) in button_set, errors, f"codex_browser_index.html missing artifact button for sample {sample_index} src {src}")
    return result



def validate_row(row_dir: Path, *, min_candidates: int = 8, min_depth: int = 2, nll_residual_tol: float = 1e-6) -> dict[str, Any]:
    row_dir = row_dir.resolve()
    errors: list[str] = []
    files = _validate_file_set(row_dir, errors)

    scaling = _read_json(row_dir / REQUIRED_JSON["scaling_tree"]) if (row_dir / REQUIRED_JSON["scaling_tree"]).exists() else {}
    payload = _read_json(row_dir / REQUIRED_JSON["trajectory_payload"]) if (row_dir / REQUIRED_JSON["trajectory_payload"]).exists() else {}
    nll_density_payload = _read_json(row_dir / REQUIRED_JSON["nll_density_payload"]) if (row_dir / REQUIRED_JSON["nll_density_payload"]).exists() else {}
    embedding_payload = _read_json(row_dir / REQUIRED_JSON["embedding_payload"]) if (row_dir / REQUIRED_JSON["embedding_payload"]).exists() else {}
    full_complex_payload = _read_json(row_dir / REQUIRED_JSON["full_complex_payload"]) if (row_dir / REQUIRED_JSON["full_complex_payload"]).exists() else {}
    manifest = _read_json(row_dir / REQUIRED_JSON["step_manifest"]) if (row_dir / REQUIRED_JSON["step_manifest"]).exists() else {}
    tropical_fan_payload = _read_json(row_dir / REQUIRED_JSON["tropical_fan_diagnostics"]) if (row_dir / REQUIRED_JSON["tropical_fan_diagnostics"]).exists() else {}
    if tropical_fan_payload:
        fan_diag = tropical_fan_payload.get("diagnostics", {}) if isinstance(tropical_fan_payload.get("diagnostics"), dict) else {}
        _assert(tropical_fan_payload.get("schema_version") == "tropicalgt.tropical_fan_visual_audit.v1", errors, "tropical fan diagnostics payload has wrong schema")
        _assert(fan_diag.get("schema_version") == "tropicalgt.cas_tropical_fan.v1", errors, "tropical fan diagnostics payload is missing CAS schema")
        _assert("support-token proxies" in str(tropical_fan_payload.get("render_contract", "")), errors, "tropical fan diagnostics missing no-proxy render contract")
        if tropical_fan_payload.get("available") is True:
            _assert(fan_diag.get("safe_to_render_as_tropical_fan") is True, errors, "available tropical fan diagnostics are not marked safe to render")
            _assert(fan_diag.get("certificate_attached") is True, errors, "available tropical fan diagnostics missing certificate")
            _assert(fan_diag.get("fan_diagnostics_certified") is True, errors, "available tropical fan diagnostics are not certified")
            summary = fan_diag.get("fan_summary", {}) if isinstance(fan_diag.get("fan_summary"), dict) else {}
            _assert(_finite_float(summary.get("ray_count"), 0.0) > 0, errors, "available tropical fan diagnostics have no rays")
        else:
            _assert(tropical_fan_payload.get("safe_to_render_as_tropical_fan") is False, errors, "unavailable tropical fan diagnostics marked safe to render")
            _assert(fan_diag.get("safe_to_render_as_tropical_fan") is False, errors, "unavailable CAS tropical fan report marked safe to render")

    toric_sidecar_path = row_dir / "toric_embedding_sidecar.json"
    toric_payload = _read_json(toric_sidecar_path) if toric_sidecar_path.exists() else {}
    if toric_payload:
        toric_diag = toric_payload.get("diagnostics", {}) if isinstance(toric_payload.get("diagnostics"), dict) else {}
        _assert(toric_payload.get("schema_version") == "tropicalgt.toric_embedding_sidecar_visual_audit.v1", errors, "toric embedding sidecar payload has wrong schema")
        _assert(toric_diag.get("schema_version") == "tropicalgt.cas_toric_embedding.v1", errors, "toric embedding sidecar payload is missing CAS schema")
        _assert("chart-bundle" in str(toric_payload.get("render_contract", "")) and "proxies" in str(toric_payload.get("render_contract", "")), errors, "toric embedding sidecar missing no-proxy render contract")
        _assert(toric_payload.get("safe_to_render_as_tropical_variety_embedding") is False, errors, "toric sidecar incorrectly claims safe tropical-variety embedding")
        _assert(toric_payload.get("safe_to_render_as_global_toric_variety_embedding") is False, errors, "toric sidecar incorrectly claims global toric-variety embedding")
        _assert(toric_payload.get("safe_to_use_as_normal_fan_certificate") is False, errors, "toric sidecar incorrectly claims normal-fan certificate")
        if toric_payload.get("available") is True:
            _assert(toric_payload.get("safe_to_render_as_finite_toric_ideal_sidecar") is True, errors, "available toric sidecar is not marked safe as finite toric-ideal sidecar")
            _assert(toric_diag.get("certificate_attached") is True, errors, "available toric sidecar missing certificate")
            _assert(toric_diag.get("toric_ideal_certified") is True, errors, "available toric sidecar is not toric-ideal certified")
            _assert(toric_diag.get("safe_to_render_as_toric_embedding") is True, errors, "available toric sidecar CAS report is not safe to render as finite toric sidecar")
            summary = toric_diag.get("monomial_map_summary", {}) if isinstance(toric_diag.get("monomial_map_summary"), dict) else {}
            matrix = summary.get("exponent_matrix") if isinstance(summary.get("exponent_matrix"), list) else []
            _assert(bool(matrix), errors, "available toric sidecar is missing exponent matrix evidence")
        else:
            _assert(toric_payload.get("safe_to_render_as_finite_toric_ideal_sidecar") is False, errors, "unavailable toric sidecar marked safe to render")
            _assert(toric_diag.get("safe_to_render_as_toric_embedding") is False, errors, "unavailable CAS toric report marked safe to render")

    chart_bundle_path = row_dir / "chart_bundle_transport_sidecar.json"
    chart_bundle_payload = _read_json(chart_bundle_path) if chart_bundle_path.exists() else {}
    if chart_bundle_payload:
        metadata = chart_bundle_payload.get("metadata", {}) if isinstance(chart_bundle_payload.get("metadata"), dict) else {}
        transport_contract = chart_bundle_payload.get("monomial_transport_contract", {}) if isinstance(chart_bundle_payload.get("monomial_transport_contract"), dict) else {}
        matroid_contract = chart_bundle_payload.get("bundle_matroid_contract", {}) if isinstance(chart_bundle_payload.get("bundle_matroid_contract"), dict) else {}
        paper_sidecar = chart_bundle_payload.get("vector_bundle_paper_sidecar", {}) if isinstance(chart_bundle_payload.get("vector_bundle_paper_sidecar"), dict) else {}
        _assert(chart_bundle_payload.get("schema_version") == "tropicalgt.chart_bundle_transport_sidecar.v1", errors, "chart-bundle transport sidecar payload has wrong schema")
        _assert(chart_bundle_payload.get("actual_data_only") is True, errors, "chart-bundle transport sidecar missing actual-data-only flag")
        _assert(chart_bundle_payload.get("no_proxy_or_fallback") is True, errors, "chart-bundle transport sidecar missing no-proxy flag")
        render_contract = str(chart_bundle_payload.get("render_contract", ""))
        _assert("not toric embedding" in render_contract or "not a toric embedding" in render_contract, errors, "chart-bundle transport sidecar missing non-toric render contract")
        _assert("no proxies" in render_contract or "no proxy" in render_contract, errors, "chart-bundle transport sidecar missing no-proxy render contract")
        _assert(chart_bundle_payload.get("safe_to_render_as_toric_embedding_certificate") is False, errors, "chart-bundle sidecar incorrectly claims toric embedding certificate safety")
        _assert(chart_bundle_payload.get("safe_to_render_as_tropical_variety_embedding") is False, errors, "chart-bundle sidecar incorrectly claims tropical-variety embedding")
        _assert(chart_bundle_payload.get("safe_to_render_as_global_toric_variety_embedding") is False, errors, "chart-bundle sidecar incorrectly claims global toric-variety embedding")
        _assert(chart_bundle_payload.get("safe_to_use_as_normal_fan_certificate") is False, errors, "chart-bundle sidecar incorrectly claims normal-fan certificate")
        _assert(paper_sidecar.get("schema_version") == "tropicalgt.vector_bundle_paper_sidecar.v1", errors, "chart-bundle sidecar missing vector-bundle paper sidecar schema")
        _assert(paper_sidecar.get("actual_data_only") is True, errors, "vector-bundle paper sidecar missing actual-data-only flag")
        _assert(paper_sidecar.get("no_proxy_or_fallback") is True, errors, "vector-bundle paper sidecar missing no-proxy flag")
        _assert(paper_sidecar.get("safe_to_use_as_vector_bundle_theorem_certificate") is False, errors, "vector-bundle paper sidecar incorrectly claims theorem-certificate safety")
        _assert(paper_sidecar.get("safe_to_use_as_toric_or_tropical_embedding_certificate") is False, errors, "vector-bundle paper sidecar incorrectly claims toric/tropical certificate safety")
        paper_claim_scope = paper_sidecar.get("paper_claim_scope", {}) if isinstance(paper_sidecar.get("paper_claim_scope"), dict) else {}
        _assert(paper_claim_scope.get("actual_tropical_toric_variety_constructed") is False, errors, "vector-bundle paper sidecar incorrectly claims a constructed tropical toric variety")
        _assert(paper_claim_scope.get("actual_tropical_scheme_constructed") is False, errors, "vector-bundle paper sidecar incorrectly claims a constructed tropical scheme")
        _assert(paper_claim_scope.get("no_proxy_or_fallback") is True, errors, "vector-bundle paper claim scope missing no-proxy flag")
        _assert("regularizer" in str(paper_claim_scope.get("monomial_transports", "")), errors, "vector-bundle paper claim scope does not keep monomial transports as regularizers/telemetry")
        paper_contract = str(paper_sidecar.get("render_contract", ""))
        _assert("no proxies" in paper_contract or "no proxy" in paper_contract, errors, "vector-bundle paper sidecar missing no-proxy render contract")
        required_paper_keys = (
            "chart_ids",
            "monomial_transport_ids",
            "toric_active_rows",
            "toric_active_row_count",
            "one_dimensional_cone_filtration_flat_defects",
            "graphcg_toric_agreement",
            "transported_persistence_landscape_metrics",
        )
        for key in required_paper_keys:
            _assert(key in paper_sidecar, errors, f"vector-bundle paper sidecar missing {key}")
        flat_defects = paper_sidecar.get("one_dimensional_cone_filtration_flat_defects", {}) if isinstance(paper_sidecar.get("one_dimensional_cone_filtration_flat_defects"), dict) else {}
        graphcg_agreement = paper_sidecar.get("graphcg_toric_agreement", {}) if isinstance(paper_sidecar.get("graphcg_toric_agreement"), dict) else {}
        landscape_metrics = paper_sidecar.get("transported_persistence_landscape_metrics", {}) if isinstance(paper_sidecar.get("transported_persistence_landscape_metrics"), dict) else {}
        _assert(flat_defects.get("actual_data_only") is True and flat_defects.get("no_proxy_or_fallback") is True, errors, "vector-bundle flat-defect sidecar is not no-proxy")
        _assert(bool(flat_defects.get("coordinate_one_dimensional_cones")), errors, "vector-bundle flat-defect sidecar missing one-dimensional cone labels")
        _assert(graphcg_agreement.get("actual_data_only") is True and graphcg_agreement.get("no_proxy_or_fallback") is True, errors, "GraphCG-toric agreement sidecar is not no-proxy")
        _assert(landscape_metrics.get("actual_data_only") is True and landscape_metrics.get("no_proxy_or_fallback") is True, errors, "transported persistence-landscape sidecar is not no-proxy")
        if chart_bundle_payload.get("available") is True:
            chart_ids = chart_bundle_payload.get("chart_ids", []) if isinstance(chart_bundle_payload.get("chart_ids"), list) else []
            overlap_pairs = metadata.get("overlap_pairs", []) if isinstance(metadata.get("overlap_pairs"), list) else []
            overlap_triples = metadata.get("overlap_triples", []) if isinstance(metadata.get("overlap_triples"), list) else []
            _assert(metadata.get("schema_version") == "tropicalgt.chart_bundle_transport_metadata.v1", errors, "available chart-bundle sidecar missing metadata schema")
            _assert(metadata.get("available") is True, errors, "available chart-bundle sidecar metadata is not available")
            _assert(bool(chart_ids), errors, "available chart-bundle sidecar missing chart ids")
            _assert(paper_sidecar.get("chart_ids") == chart_ids, errors, "vector-bundle paper sidecar chart ids do not match chart-bundle metadata")
            _assert(bool(paper_sidecar.get("monomial_transport_ids")), errors, "available vector-bundle paper sidecar missing monomial transport ids")
            toric_rows = paper_sidecar.get("toric_active_rows", {}) if isinstance(paper_sidecar.get("toric_active_rows"), dict) else {}
            _assert(toric_rows.get("actual_data_only") is True and toric_rows.get("no_proxy_or_fallback") is True, errors, "vector-bundle toric active rows are not no-proxy")
            _assert(toric_rows.get("available") is True, errors, "available vector-bundle paper sidecar missing configured toric active rows")
            _assert(int(chart_bundle_payload.get("overlap_pair_count", -1)) == len(overlap_pairs), errors, "chart-bundle sidecar overlap pair count mismatch")
            _assert(int(chart_bundle_payload.get("overlap_triple_count", -1)) == len(overlap_triples), errors, "chart-bundle sidecar overlap triple count mismatch")
            _assert(transport_contract.get("actual_data_only") is True and transport_contract.get("no_proxy_or_fallback") is True, errors, "available chart-bundle transport contract is not no-proxy")
            _assert(matroid_contract.get("actual_data_only") is True and matroid_contract.get("no_proxy_or_fallback") is True, errors, "available chart-bundle matroid contract is not no-proxy")

    support_payload = _read_json(row_dir / REQUIRED_JSON["tropical_support_payload"]) if (row_dir / REQUIRED_JSON["tropical_support_payload"]).exists() else {}
    graphcg_payload = _read_json(row_dir / REQUIRED_JSON["graphcg_payload"]) if (row_dir / REQUIRED_JSON["graphcg_payload"]).exists() else {}
    analogical_simplex_tree_payload = _read_json(row_dir / REQUIRED_JSON["analogical_simplex_tree_analogy"]) if (row_dir / REQUIRED_JSON["analogical_simplex_tree_analogy"]).exists() else {}
    bifiltration_payload = _read_json(row_dir / REQUIRED_JSON["trajectory_bifiltration_payload"]) if (row_dir / REQUIRED_JSON["trajectory_bifiltration_payload"]).exists() else {}
    bifiltration_visual_path = row_dir / "trajectory_persistence" / "two_parameter_bifiltration.json"
    bifiltration_visual_payload = _read_json(bifiltration_visual_path) if bifiltration_visual_path.exists() else {}
    landscapes_payload_path = row_dir / REQUIRED_JSON["trajectory_landscapes_payload"]
    landscapes_payload = _read_json(landscapes_payload_path) if landscapes_payload_path.exists() else {}

    candidates = [row for row in scaling.get("candidates", []) if isinstance(row, dict)]
    nodes = [row for row in payload.get("nodes", []) if isinstance(row, dict)]
    edges = [row for row in payload.get("edges", []) if isinstance(row, dict)]
    paths = [tuple(row.get("path", [])) for row in candidates]
    levels = sorted({_finite_float(row.get("level"), 0.0) for row in candidates})
    max_level = int(max(levels, default=0))
    branch_counts: dict[str, int] = {}
    for edge in edges:
        source = str(edge.get("source", ""))
        branch_counts[source] = branch_counts.get(source, 0) + 1


    _assert(bifiltration_payload.get("available") is True, errors, "trajectory bifiltration payload is unavailable")
    _assert(bifiltration_payload.get("coefficient_ring") == "F2[x_level,x_radius]", errors, "trajectory bifiltration is not over F2[x_level,x_radius]")
    _assert(int(_finite_float(bifiltration_payload.get("num_parameters"), 0.0)) == 2, errors, "trajectory bifiltration is not 2-parameter")
    parameters = bifiltration_payload.get("parameters", [])
    parameter_names = [str(row.get("name")) for row in parameters if isinstance(row, dict)] if isinstance(parameters, list) else []
    _assert(parameter_names == ["trajectory_level", "radius"], errors, "trajectory bifiltration parameters are not trajectory_level and radius")
    grid_axes = bifiltration_payload.get("grid_axes", [])
    levels_axis = grid_axes[0] if isinstance(grid_axes, list) and len(grid_axes) >= 1 and isinstance(grid_axes[0], list) else []
    radius_axis = grid_axes[1] if isinstance(grid_axes, list) and len(grid_axes) >= 2 and isinstance(grid_axes[1], list) else []
    bif_levels = bifiltration_payload.get("levels", [])
    bif_radii = bifiltration_payload.get("radii", [])
    _assert(isinstance(bif_levels, list) and len(bif_levels) >= 1, errors, "trajectory bifiltration has no level grades")
    _assert(isinstance(bif_radii, list) and len(bif_radii) >= 1, errors, "trajectory bifiltration has no radius grades")
    _assert(levels_axis == list(range(len(bif_levels))), errors, "trajectory bifiltration level grid axis does not match level grades")
    _assert(radius_axis == list(range(len(bif_radii))), errors, "trajectory bifiltration radius grid axis does not match radius grades")
    _assert(bifiltration_payload.get("radius_grade_policy") == "exact_sorted_radius_grid_index_no_bucket_collision", errors, "trajectory bifiltration radius grades are not exact sorted radius-grid indices")
    structure_maps = bifiltration_payload.get("structure_maps", [])
    structure_direction_counts: Counter[str] = Counter()
    _assert(isinstance(structure_maps, list) and bool(structure_maps), errors, "trajectory bifiltration has no adjacent F2 structure maps")
    if isinstance(structure_maps, list):
        for index, row in enumerate(structure_maps):
            if not isinstance(row, dict):
                errors.append(f"trajectory bifiltration structure map {index} is not an object")
                continue
            direction = row.get("direction")
            if isinstance(direction, str):
                structure_direction_counts[direction] += 1
            _assert(direction in {"x_level", "x_radius"}, errors, f"trajectory bifiltration structure map {index} has invalid direction")
            _assert(row.get("field") == "F2", errors, f"trajectory bifiltration structure map {index} is not over F2")
            source_grade = row.get("source_grade")
            target_grade = row.get("target_grade")
            _assert(isinstance(source_grade, list) and len(source_grade) >= 2, errors, f"trajectory bifiltration structure map {index} lacks a source bidegree")
            _assert(isinstance(target_grade, list) and len(target_grade) >= 2, errors, f"trajectory bifiltration structure map {index} lacks a target bidegree")
            ranks = row.get("homology_rank")
            _assert(isinstance(ranks, dict) and {"0", "1"}.issubset(set(ranks)), errors, f"trajectory bifiltration structure map {index} lacks H0/H1 rank evidence")
    _assert(structure_direction_counts.get("x_level", 0) > 0 and structure_direction_counts.get("x_radius", 0) > 0, errors, "trajectory bifiltration lacks both x_level and x_radius adjacent structure maps")
    if bifiltration_visual_payload:
        _assert(bifiltration_visual_payload.get("schema_version") == "tropicalgt.two_parameter_bifiltration_visual.v1", errors, "trajectory bifiltration visual payload has wrong schema")
        _assert(bifiltration_visual_payload.get("primary_view") == "miller_sturmfels_bivariate_staircase", errors, "trajectory bifiltration primary view is not the Miller-Sturmfels staircase")
        _assert(bifiltration_visual_payload.get("rank_surface_primary") is False, errors, "trajectory bifiltration marks rank surfaces as primary")
        axes = bifiltration_visual_payload.get("axes", {}) if isinstance(bifiltration_visual_payload.get("axes"), dict) else {}
        _assert(axes.get("horizontal") == "x_radius" and axes.get("vertical") == "x_level", errors, "trajectory bifiltration visual axes are not x_radius horizontal / x_level vertical")
        _assert("rho_x_radius" in axes.get("coordinate_one_dimensional_cones", []) and "rho_x_level" in axes.get("coordinate_one_dimensional_cones", []), errors, "trajectory bifiltration visual payload lacks coordinate one dimensional cone records")
        _assert(bifiltration_visual_payload.get("actual_data_only") is True, errors, "trajectory bifiltration visual payload does not assert actual-data-only rendering")
        _assert(bifiltration_visual_payload.get("no_proxy_resolution_claim") is True, errors, "trajectory bifiltration visual payload allows proxy resolution claims")
        structure_summary = bifiltration_visual_payload.get("structure_map_summary", {}) if isinstance(bifiltration_visual_payload.get("structure_map_summary"), dict) else {}
        _assert(structure_summary.get("schema_version") == "tropicalgt.two_parameter_structure_maps.v1", errors, "trajectory bifiltration visual payload lacks the structure-map summary schema")
        _assert(structure_summary.get("source") == "bifiltration.structure_maps", errors, "trajectory bifiltration structure-map summary does not cite raw structure maps")
        _assert(int(_finite_float(structure_summary.get("actual_adjacent_map_count"), -1.0)) == (len(structure_maps) if isinstance(structure_maps, list) else 0), errors, "trajectory bifiltration structure-map summary count does not match raw structure maps")
        _assert(structure_summary.get("field") == "F2", errors, "trajectory bifiltration structure-map summary is not over F2")
        _assert(structure_summary.get("east_north_structure_maps_present") is True, errors, "trajectory bifiltration structure-map summary lacks east/north map evidence")
        _assert(structure_summary.get("no_proxy_or_fallback") is True, errors, "trajectory bifiltration structure-map summary allows proxy/fallback evidence")
        summary_direction_counts = structure_summary.get("direction_counts", {}) if isinstance(structure_summary.get("direction_counts"), dict) else {}
        _assert(int(_finite_float(summary_direction_counts.get("x_level"), -1.0)) == structure_direction_counts.get("x_level", 0), errors, "trajectory bifiltration x_level structure-map count does not match raw data")
        _assert(int(_finite_float(summary_direction_counts.get("x_radius"), -1.0)) == structure_direction_counts.get("x_radius", 0), errors, "trajectory bifiltration x_radius structure-map count does not match raw data")
        rank_rows = structure_summary.get("rank_rows", [])
        _assert(isinstance(rank_rows, list) and len(rank_rows) == (len(structure_maps) if isinstance(structure_maps, list) else 0), errors, "trajectory bifiltration structure-map summary lacks per-map rank rows")
        _assert(structure_summary.get("module_lattice_overlay_available") is True, errors, "trajectory bifiltration structure-map summary does not expose the module-lattice overlay")
        overlay_trace_names = structure_summary.get("module_lattice_overlay_trace_names", [])
        _assert(isinstance(overlay_trace_names, list) and "actual x_level structure maps over F2" in overlay_trace_names and "actual x_radius structure maps over F2" in overlay_trace_names, errors, "trajectory bifiltration module-lattice overlay lacks x_level/x_radius trace names")
        primary_structure_evidence = bifiltration_visual_payload.get("primary_structure_map_evidence", {}) if isinstance(bifiltration_visual_payload.get("primary_structure_map_evidence"), dict) else {}
        _assert(primary_structure_evidence.get("schema_version") == "tropicalgt.primary_structure_map_evidence.v1", errors, "trajectory bifiltration visual payload lacks primary structure-map evidence schema")
        _assert(primary_structure_evidence.get("available") is True, errors, "trajectory bifiltration primary structure-map evidence is unavailable")
        _assert(primary_structure_evidence.get("source") == "bifiltration.structure_maps", errors, "trajectory bifiltration primary structure-map evidence does not cite raw structure maps")
        directions_rendered = primary_structure_evidence.get("directions_rendered", []) if isinstance(primary_structure_evidence.get("directions_rendered"), list) else []
        _assert("x_level" in directions_rendered and "x_radius" in directions_rendered, errors, "trajectory bifiltration primary structure-map evidence lacks both directions")
        _assert(int(_finite_float(primary_structure_evidence.get("primary_table_rows"), -1.0)) == (len(structure_maps) if isinstance(structure_maps, list) else 0), errors, "trajectory bifiltration primary structure-map evidence table count does not match raw maps")
        _assert(primary_structure_evidence.get("no_proxy_or_fallback") is True, errors, "trajectory bifiltration primary structure-map evidence allows proxy/fallback evidence")
        cas_indexed = bifiltration_visual_payload.get("certificate_indexed_cas_evidence") if isinstance(bifiltration_visual_payload.get("certificate_indexed_cas_evidence"), dict) else {}
        _assert(cas_indexed.get("schema_version") == "tropicalgt.cas_certificate_indexed_evidence.v1", errors, "trajectory bifiltration visual payload lacks certificate-indexed CAS evidence schema")
        _assert(cas_indexed.get("no_proxy_or_fallback") is True, errors, "trajectory bifiltration certificate-indexed CAS evidence allows proxy/fallback evidence")
        if cas_indexed.get("available") is True:
            _assert(cas_indexed.get("exactness_certified") is True, errors, "available certificate-indexed CAS evidence is not exactness-certified")
            _assert(isinstance(cas_indexed.get("evidence_blocks"), dict), errors, "available certificate-indexed CAS evidence lacks evidence blocks")
            _assert(cas_indexed.get("derived_category_claim_requires_chain_map_or_resolution_comparison") is True, errors, "available certificate-indexed CAS evidence lacks derived-category guard")
        else:
            _assert(cas_indexed.get("safe_unavailable_render") is True, errors, "unavailable certificate-indexed CAS evidence is not marked safe-unavailable")
            _assert(bool(cas_indexed.get("reason")), errors, "unavailable certificate-indexed CAS evidence lacks reason")
        staircase_cards = bifiltration_visual_payload.get("staircase_cards", [])
        _assert(isinstance(staircase_cards, list) and bool(staircase_cards), errors, "trajectory bifiltration visual payload lacks staircase card contracts")
        if isinstance(staircase_cards, list):
            for index, card in enumerate(staircase_cards[:6]):
                if not isinstance(card, dict):
                    errors.append(f"trajectory bifiltration staircase card {index} is not an object")
                    continue
                _assert(isinstance(card.get("generator_labels"), list), errors, f"trajectory bifiltration staircase card {index} lacks generator labels")
                _assert(isinstance(card.get("upward_closed_regions"), list), errors, f"trajectory bifiltration staircase card {index} lacks upward-closed region contracts")
                _assert(isinstance(card.get("quotient_basis_lattice_points"), list), errors, f"trajectory bifiltration staircase card {index} lacks quotient-basis lattice points")
                _assert(int(_finite_float(card.get("quotient_basis_lattice_count"), -1.0)) == len(card.get("quotient_basis_lattice_points", [])), errors, f"trajectory bifiltration staircase card {index} quotient-basis count mismatch")
                _assert(isinstance(card.get("hilbert_numerator_terms"), list), errors, f"trajectory bifiltration staircase card {index} lacks Hilbert numerator terms")
                _assert(isinstance(card.get("adjacent_lcm_syzygies"), list), errors, f"trajectory bifiltration staircase card {index} lacks adjacent LCM syzygy terms")
                _assert("not a full persistence-module free resolution" in str(card.get("theorem_scope", "")), errors, f"trajectory bifiltration staircase card {index} theorem scope lacks no-proxy resolution boundary")
        staircase_evidence = bifiltration_visual_payload.get("miller_sturmfels_staircase_evidence", {}) if isinstance(bifiltration_visual_payload.get("miller_sturmfels_staircase_evidence"), dict) else {}
        _assert(staircase_evidence.get("schema_version") == "tropicalgt.miller_sturmfels_staircase_evidence.v1", errors, "trajectory bifiltration visual payload lacks Miller-Sturmfels staircase evidence schema")
        _assert(staircase_evidence.get("source") == "staircase_cards_from_bifiltration.chain_module_generators[*].multidegree", errors, "Miller-Sturmfels staircase evidence does not cite actual bifiltration chain-generator bidegrees")
        _assert(staircase_evidence.get("coefficient_ring") == "F2[x_level,x_radius]", errors, "Miller-Sturmfels staircase evidence has wrong coefficient ring")
        _assert(staircase_evidence.get("actual_data_only") is True, errors, "Miller-Sturmfels staircase evidence is not actual-data-only")
        _assert(staircase_evidence.get("no_proxy_or_fallback") is True, errors, "Miller-Sturmfels staircase evidence allows proxy/fallback data")
        _assert(staircase_evidence.get("primary_view") == "miller_sturmfels_bivariate_staircase", errors, "Miller-Sturmfels staircase evidence points at the wrong primary view")
        evidence_axes = staircase_evidence.get("axes", {}) if isinstance(staircase_evidence.get("axes"), dict) else {}
        _assert(evidence_axes.get("horizontal") == "x_radius" and evidence_axes.get("vertical") == "x_level", errors, "Miller-Sturmfels staircase evidence axes are not x_radius horizontal / x_level vertical")
        _assert("rho_x_radius" in evidence_axes.get("coordinate_one_dimensional_cones", []) and "rho_x_level" in evidence_axes.get("coordinate_one_dimensional_cones", []), errors, "Miller-Sturmfels staircase evidence lacks coordinate one dimensional cones")
        valid_cards = [card for card in staircase_cards if isinstance(card, dict)] if isinstance(staircase_cards, list) else []
        primary_card_count = sum(1 for card in valid_cards if card.get("primary_card") is True)
        primary_indices = [index for index, card in enumerate(valid_cards) if card.get("primary_card") is True]
        total_actual_generators = sum(int(_finite_float(card.get("actual_generator_bidegree_count"), 0.0)) for card in valid_cards)
        total_minimal = sum(len(card.get("minimal_antichain", [])) for card in valid_cards if isinstance(card.get("minimal_antichain"), list))
        total_labels = sum(len(card.get("generator_labels", [])) for card in valid_cards if isinstance(card.get("generator_labels"), list))
        total_regions = sum(len(card.get("upward_closed_regions", [])) for card in valid_cards if isinstance(card.get("upward_closed_regions"), list))
        total_quotient = sum(int(_finite_float(card.get("quotient_basis_lattice_count"), 0.0)) for card in valid_cards)
        total_hilbert = sum(len(card.get("hilbert_numerator_terms", [])) for card in valid_cards if isinstance(card.get("hilbert_numerator_terms"), list))
        total_syzygies = sum(len(card.get("adjacent_lcm_syzygies", [])) for card in valid_cards if isinstance(card.get("adjacent_lcm_syzygies"), list))
        _assert(int(_finite_float(staircase_evidence.get("card_count"), -1.0)) == len(valid_cards), errors, "Miller-Sturmfels staircase evidence card count does not match staircase cards")
        _assert(int(_finite_float(staircase_evidence.get("primary_card_count"), -1.0)) == primary_card_count and primary_card_count == 1, errors, "Miller-Sturmfels staircase evidence does not identify exactly one primary card")
        _assert(int(_finite_float(staircase_evidence.get("primary_card_index"), -1.0)) == (primary_indices[0] if primary_indices else -1), errors, "Miller-Sturmfels staircase evidence primary card index mismatch")
        _assert(int(_finite_float(staircase_evidence.get("total_actual_generator_bidegree_count"), -1.0)) == total_actual_generators, errors, "Miller-Sturmfels staircase evidence actual-generator aggregate mismatch")
        _assert(int(_finite_float(staircase_evidence.get("total_minimal_antichain_count"), -1.0)) == total_minimal, errors, "Miller-Sturmfels staircase evidence minimal-antichain aggregate mismatch")
        _assert(int(_finite_float(staircase_evidence.get("total_generator_label_count"), -1.0)) == total_labels, errors, "Miller-Sturmfels staircase evidence generator-label aggregate mismatch")
        _assert(int(_finite_float(staircase_evidence.get("total_upward_closed_region_count"), -1.0)) == total_regions, errors, "Miller-Sturmfels staircase evidence upward-closed region aggregate mismatch")
        _assert(int(_finite_float(staircase_evidence.get("total_quotient_basis_lattice_count"), -1.0)) == total_quotient, errors, "Miller-Sturmfels staircase evidence quotient-basis aggregate mismatch")
        _assert(int(_finite_float(staircase_evidence.get("total_hilbert_numerator_term_count"), -1.0)) == total_hilbert, errors, "Miller-Sturmfels staircase evidence Hilbert numerator aggregate mismatch")
        _assert(int(_finite_float(staircase_evidence.get("total_adjacent_lcm_syzygy_count"), -1.0)) == total_syzygies, errors, "Miller-Sturmfels staircase evidence adjacent-LCM syzygy aggregate mismatch")
        per_card_counts = staircase_evidence.get("per_card_counts", []) if isinstance(staircase_evidence.get("per_card_counts"), list) else []
        _assert(len(per_card_counts) == len(valid_cards), errors, "Miller-Sturmfels staircase evidence lacks per-card aggregate rows")
        _assert(staircase_evidence.get("all_cards_have_generator_labels") is True, errors, "Miller-Sturmfels staircase evidence does not certify generator labels")
        _assert(staircase_evidence.get("all_cards_have_upward_closed_regions") is True, errors, "Miller-Sturmfels staircase evidence does not certify upward-closed regions")
        _assert(staircase_evidence.get("all_cards_have_quotient_basis_lattice_points") is True, errors, "Miller-Sturmfels staircase evidence does not certify quotient-basis lattice points")
        _assert(staircase_evidence.get("quotient_basis_counts_match_lattice_points") is True, errors, "Miller-Sturmfels staircase evidence quotient-basis counts do not match lattice points")
        _assert(staircase_evidence.get("theorem_scope_boundary_all_cards") is True, errors, "Miller-Sturmfels staircase evidence lacks theorem-scope no-proxy boundaries")
        _assert(staircase_evidence.get("coordinate_axes_are_one_dimensional_cones") is True, errors, "Miller-Sturmfels staircase evidence does not certify coordinate axes as one dimensional cones")
        _assert(staircase_evidence.get("safe_to_render_miller_sturmfels_staircase") is True, errors, "Miller-Sturmfels staircase evidence is unsafe to render")
    _assert(all(_finite_float(radius, float("nan")) >= 0.0 for radius in bif_radii), errors, "trajectory bifiltration contains negative radius grades")
    _assert(all(_finite_float(bif_radii[i], float("nan")) <= _finite_float(bif_radii[i + 1], float("nan")) for i in range(max(0, len(bif_radii) - 1))), errors, "trajectory bifiltration radius grades are not sorted min-to-max")
    rank_samples = bifiltration_payload.get("rank_invariant_samples", [])
    _assert(isinstance(rank_samples, list) and len(rank_samples) > 0, errors, "trajectory bifiltration has no rank invariant samples")
    for rank_sample in rank_samples[: min(16, len(rank_samples))] if isinstance(rank_samples, list) else []:
        if not isinstance(rank_sample, dict):
            errors.append("trajectory bifiltration rank invariant sample is not an object")
            continue
        _assert(isinstance(rank_sample.get("source_grade"), list) and len(rank_sample.get("source_grade", [])) == 2, errors, "rank invariant sample missing 2D source grade")
        _assert(isinstance(rank_sample.get("target_grade"), list) and len(rank_sample.get("target_grade", [])) == 2, errors, "rank invariant sample missing 2D target grade")
        _assert(_finite_float(rank_sample.get("h0_rank"), -1.0) >= 0.0, errors, "rank invariant sample has negative H0 rank")
    boundary = bifiltration_payload.get("boundary_monomials", {})
    diagnostics = bifiltration_payload.get("chain_presentation_diagnostics", {})
    _assert(isinstance(boundary, dict) and bool(boundary), errors, "trajectory bifiltration is missing boundary monomial data")
    _assert(isinstance(diagnostics, dict), errors, "trajectory bifiltration is missing chain-presentation diagnostics")
    if isinstance(diagnostics, dict):
        _assert(diagnostics.get("ring") == "F2[x_level,x_radius]", errors, "chain-presentation diagnostics have the wrong ring")
        _assert(diagnostics.get("real_free_resolution_certified") is not True, errors, "validator expected uncertified chain presentation, but payload claims a real free resolution without CAS validation")
        _validate_real_free_resolution_guard(diagnostics.get("real_free_resolution"), errors, "trajectory bifiltration chain-presentation")

    _assert(landscapes_payload_path.exists(), errors, "trajectory persistence landscapes payload is missing")
    _assert(isinstance(landscapes_payload, dict), errors, "trajectory persistence landscapes payload is not an object")
    if isinstance(landscapes_payload, dict):
        _assert(landscapes_payload.get("schema_version") == "tropicalgt.persistence_landscape_visual_contract.v1", errors, "trajectory persistence landscapes payload has wrong schema")
        backend_provenance = landscapes_payload.get("backend_provenance", {}) if isinstance(landscapes_payload.get("backend_provenance"), dict) else {}
        unavailable_reasons = landscapes_payload.get("unavailable_reasons", []) if isinstance(landscapes_payload.get("unavailable_reasons"), list) else []
        finite_interval_count = _trajectory_finite_persistence_interval_count(row_dir)
        no_finite_interval_unavailable = (
            landscapes_payload.get("available") is False
            and finite_interval_count == 0
            and landscapes_payload.get("unavailable_state_verified_by_intervals") is True
            and "no_finite_persistence_intervals_for_gudhi_landscape" in unavailable_reasons
        )
        if no_finite_interval_unavailable:
            _assert(landscapes_payload.get("actual_data_only") is True, errors, "trajectory persistence landscapes payload is not actual-data-only")
            _assert(landscapes_payload.get("no_proxy_or_fallback") is True, errors, "trajectory persistence landscapes payload allows proxy/fallback data")
            _assert(landscapes_payload.get("not_nll_fitness_landscape") is True, errors, "trajectory persistence landscapes payload confuses GUDHI landscapes with NLL/fitness landscape")
            _assert(landscapes_payload.get("source") == "topology.persistence_representations.methods[*].landscape", errors, "trajectory persistence landscapes payload has wrong source")
            _assert(int(_finite_float(landscapes_payload.get("finite_persistence_interval_count"), -1.0)) == 0, errors, "trajectory persistence landscapes unavailable state does not report zero finite intervals")
        else:
            _assert(landscapes_payload.get("available") is True, errors, "trajectory persistence landscapes payload is unavailable")
            _assert(str(landscapes_payload.get("landscape_backend", "")).strip() != "", errors, "trajectory persistence landscapes payload lacks landscape backend provenance")
            _assert(backend_provenance.get("available") is True, errors, "trajectory persistence landscapes backend provenance is unavailable")
            _assert(backend_provenance.get("source_field") == "topology.persistence_representations.backend", errors, "trajectory persistence landscapes backend provenance has wrong source field")
            _assert(isinstance(backend_provenance.get("backends"), list) and bool(backend_provenance.get("backends")), errors, "trajectory persistence landscapes backend provenance lists no backends")
            _assert(landscapes_payload.get("actual_data_only") is True, errors, "trajectory persistence landscapes payload is not actual-data-only")
            _assert(landscapes_payload.get("no_proxy_or_fallback") is True, errors, "trajectory persistence landscapes payload allows proxy/fallback data")
            _assert(landscapes_payload.get("not_nll_fitness_landscape") is True, errors, "trajectory persistence landscapes payload confuses GUDHI landscapes with NLL/fitness landscape")
            _assert(landscapes_payload.get("not_norm_only_summary") is True, errors, "trajectory persistence landscapes payload is norm-only rather than lambda_k curves")
            _assert(landscapes_payload.get("safe_to_render_actual_landscape_functions") is True, errors, "trajectory persistence landscapes payload is unsafe to render")
            _assert(_finite_float(landscapes_payload.get("curve_trace_count"), 0.0) > 0, errors, "trajectory persistence landscapes payload has no curve traces")
            _assert(_finite_float(landscapes_payload.get("rendered_growth_level_count"), 0.0) > 0, errors, "trajectory persistence landscapes payload has no growth levels")
            rows = landscapes_payload.get("landscape_rows") if isinstance(landscapes_payload.get("landscape_rows"), list) else []
            _assert(bool(rows), errors, "trajectory persistence landscapes payload has no landscape rows")
            _assert(isinstance(landscapes_payload.get("homology_dimensions"), list) and bool(landscapes_payload.get("homology_dimensions")), errors, "trajectory persistence landscapes payload has no homology dimensions")
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    errors.append(f"trajectory persistence landscape row {idx} is not an object")
                    continue
                _assert(row.get("source") == "topology.persistence_representations.methods[*].landscape", errors, f"trajectory persistence landscape row {idx} has wrong source")
                _assert(row.get("actual_gudhi_landscape_values") is True, errors, f"trajectory persistence landscape row {idx} is not actual GUDHI values")
                _assert(row.get("not_norm_only_summary") is True, errors, f"trajectory persistence landscape row {idx} is norm-only")
                _assert(row.get("not_nll_fitness_landscape") is True, errors, f"trajectory persistence landscape row {idx} confuses NLL/fitness landscape")
                _assert(_finite_float(row.get("layer_count"), 0.0) > 0, errors, f"trajectory persistence landscape row {idx} has no lambda layers")
                _assert(_finite_float(row.get("grid_count"), 0.0) > 0, errors, f"trajectory persistence landscape row {idx} has no grid/sample coordinates")
                _assert(_finite_float(row.get("finite_value_count"), 0.0) > 0, errors, f"trajectory persistence landscape row {idx} has no finite values")
                _assert(str(row.get("values_source", "")).startswith("gudhi.representations.Landscape") or row.get("values_source") == "reported_landscape_values", errors, f"trajectory persistence landscape row {idx} has unaudited values source")

    _assert(len(candidates) >= min_candidates, errors, f"candidate count {len(candidates)} < {min_candidates}")
    _assert(len(edges) >= max(0, len(candidates) - 1), errors, f"edge count {len(edges)} is smaller than candidate tree count {len(candidates) - 1}")
    _assert(max_level >= min_depth, errors, f"max level {max_level} < {min_depth}")
    _assert(len(set(paths)) == len(paths), errors, "reasoning paths are not unique")
    _assert(sum(1 for count in branch_counts.values() if count > 1) > 0, errors, "no branching node found")
    _assert(bool(scaling.get("stochastic_actions")), errors, "stochastic_actions is not true")

    pca_diag = payload.get("embedding_pca_diagnostics", {})
    _assert(pca_diag.get("coordinate_source") == "model graph_state embeddings", errors, "PCA source is not model graph_state embeddings")
    _assert(_finite_float(pca_diag.get("n_samples")) == len(nodes), errors, "PCA sample count does not match payload node count")

    embedded_nodes = [row for row in embedding_payload.get("nodes", []) if isinstance(row, dict)]
    embedding_objects = [row for row in embedding_payload.get("filtered_simplicial_objects", []) if isinstance(row, dict)]
    _assert(sum(1 for row in nodes if row.get("embedding") is not None) == len(nodes), errors, "trajectory payload nodes do not all carry raw model embeddings")
    embedding_by_id = {str(row.get("record_id")): row for row in embedded_nodes}
    node_basis = []
    for row in nodes:
        if row.get("embedding") is not None and row.get("embedding_pca") is not None:
            node_basis.append(row)
        else:
            node_basis.append(embedding_by_id.get(str(row.get("record_id")), row))
    embeddings = np.asarray([row.get("embedding", []) for row in node_basis], dtype=float) if node_basis else np.zeros((0, 0))
    coords = (
        np.asarray(
            [
                [
                    row.get("embedding_pca", row.get("pca", {})).get("pc1"),
                    row.get("embedding_pca", row.get("pca", {})).get("pc2"),
                    row.get("embedding_pca", row.get("pca", {})).get("pc3"),
                ]
                for row in node_basis
            ],
            dtype=float,
        )
        if node_basis
        else np.zeros((0, 3))
    )
    _assert(embeddings.ndim == 2 and coords.ndim == 2 and coords.shape[1] == 3, errors, "embedding/PCA coordinate arrays have invalid shape")
    _assert(np.isfinite(embeddings).all() and np.isfinite(coords).all(), errors, "embedding/PCA coordinate arrays contain non-finite values")
    recomputed = _distance_diagnostics(embeddings, coords) if len(nodes) >= 2 else {"pairwise_distance_correlation": 1.0, "normalized_stress": 0.0}
    diag_corr = _finite_float(pca_diag.get("pairwise_distance_correlation"))
    diag_stress = _finite_float(pca_diag.get("normalized_stress"))
    _assert(abs(recomputed["pairwise_distance_correlation"] - diag_corr) <= 1e-6, errors, "PCA distance correlation does not match recomputation")
    _assert(abs(recomputed["normalized_stress"] - diag_stress) <= 1e-6, errors, "PCA normalized stress does not match recomputation")
    _assert(diag_corr >= 0.75, errors, f"PCA pairwise-distance correlation too low: {diag_corr}")
    _assert(diag_stress <= 0.75, errors, f"PCA normalized stress too high: {diag_stress}")

    surface = payload.get("nll_surface", {})
    _assert(surface.get("available") is True, errors, "NLL surface unavailable")
    _assert(surface.get("touches_points") is True, errors, "NLL surface is not point-anchored")
    _assert(_finite_float(surface.get("max_point_residual"), 999.0) <= nll_residual_tol, errors, "NLL surface residual exceeds tolerance")
    _assert(
        str(surface.get("surface_contact_contract", "")).startswith("disabled for the main trajectory page"),
        errors,
        "NLL surface contract does not document PC3 marker geometry with NLL as metadata",
    )
    _assert(
        _finite_float(surface.get("trajectory_point_surface_residual_max"), 999.0) <= nll_residual_tol,
        errors,
        "NLL surface projection residual exceeds tolerance",
    )
    projected_by_record = surface.get("surface_projected_z_by_record_id", {})
    _assert(isinstance(projected_by_record, dict) and len(projected_by_record) >= len(nodes), errors, "NLL surface is missing per-record projected z values")
    local_surface = surface.get("local_embedding_neighborhood_surface")
    _assert(isinstance(local_surface, dict), errors, "NLL surface is missing local embedding-neighborhood surface contract")
    if isinstance(local_surface, dict):
        _assert(
            local_surface.get("schema_version") == "tropicalgt.local_embedding_neighborhood_surface.v1",
            errors,
            "local embedding-neighborhood surface has the wrong schema version",
        )
        _assert(local_surface.get("no_proxy_or_fallback") is True, errors, "local embedding-neighborhood surface does not declare no-proxy/no-fallback")
        if local_surface.get("available") is True:
            _assert(local_surface.get("invented_nll_values") is False, errors, "local embedding-neighborhood surface invented NLL values")
            _assert(local_surface.get("support_samples_are_model_states") is False, errors, "local embedding-neighborhood surface treats support samples as model states")
            _assert(_finite_float(local_surface.get("model_evaluated_anchor_count"), 0.0) >= len(nodes), errors, "local embedding-neighborhood surface has too few model-evaluated anchors")
            _assert(
                _finite_float(local_surface.get("trajectory_point_surface_residual_max"), 999.0) <= nll_residual_tol,
                errors,
                "local embedding-neighborhood surface residual exceeds tolerance",
            )
            local_projected = local_surface.get("surface_projected_z_by_record_id", {})
            _assert(isinstance(local_projected, dict) and len(local_projected) >= len(nodes), errors, "local embedding-neighborhood surface lacks per-record projected z values")
            for node in nodes:
                rid = str(node.get("record_id", ""))
                _assert(isinstance(local_projected, dict) and rid in local_projected, errors, f"local embedding-neighborhood surface lacks projected z for {rid}")
                _assert(isinstance(projected_by_record, dict) and rid in projected_by_record, errors, f"NLL surface missing projected z for {rid}")
                if isinstance(local_projected, dict) and isinstance(projected_by_record, dict) and rid in local_projected and rid in projected_by_record:
                    _assert(
                        abs(_finite_float(local_projected.get(rid)) - _finite_float(projected_by_record.get(rid))) <= nll_residual_tol,
                        errors,
                        f"local embedding-neighborhood surface projection for {rid} does not match main NLL surface projection",
                    )
        else:
            _assert(local_surface.get("safe_unavailable_render") is True, errors, "unavailable local embedding-neighborhood surface lacks a safe unavailable render flag")
            _assert(bool(local_surface.get("reason")), errors, "unavailable local embedding-neighborhood surface lacks an exact reason")
    for node in nodes:
        rid = str(node.get("record_id", ""))
        plot = node.get("plot", {}) if isinstance(node.get("plot"), dict) else {}
        pca = node.get("embedding_pca", node.get("pca", {})) if isinstance(node, dict) else {}
        _assert(plot.get("touches_nll_surface") is False, errors, f"trajectory node {rid} should keep PC3 geometry rather than touch the NLL surface")
        _assert(math.isfinite(_finite_float(plot.get("z"))), errors, f"trajectory node {rid} is missing finite plotted z")
        _assert(plot.get("z_surface") is None, errors, f"trajectory node {rid} should leave z_surface null under the PC3 geometry contract")
        _assert(math.isfinite(_finite_float(plot.get("raw_centered_scaled_nll"))), errors, f"trajectory node {rid} is missing raw centered/scaled NLL z")
        _assert(math.isfinite(_finite_float(plot.get("z_centered_scaled_nll"))), errors, f"trajectory node {rid} is missing centered/scaled NLL metadata z")
        if isinstance(pca, dict) and "pc3" in pca:
            _assert(
                abs(_finite_float(plot.get("z")) - _finite_float(pca.get("pc3"))) <= nll_residual_tol,
                errors,
                f"trajectory node {rid} plotted z does not match PCA pc3 geometry",
            )
        if isinstance(projected_by_record, dict) and rid in projected_by_record:
            _assert(
                abs(_finite_float(plot.get("z_centered_scaled_nll")) - _finite_float(projected_by_record.get(rid))) <= nll_residual_tol
                and abs(_finite_float(plot.get("raw_centered_scaled_nll")) - _finite_float(projected_by_record.get(rid))) <= nll_residual_tol,
                errors,
                f"trajectory node {rid} centered/scaled NLL metadata does not match the displayed NLL surface projection",
            )
    z_axis = surface.get("z_axis")
    _assert(
        z_axis in {"centered_scaled_nll", "projected_nll_fitness_energy"},
        errors,
        "NLL trajectory surface is not rendered with an audited NLL/fitness z-axis",
    )
    _assert(_finite_float(surface.get("raw_nll_range"), -1.0) >= 0.0, errors, "NLL surface payload is missing raw_nll_range")
    _assert(surface.get("exact_anchor_layer") is True, errors, "NLL surface is missing exact anchor layer metadata")
    dense_field = surface.get("dense_model_evaluated_field") is True
    sparse_anchor = surface.get("sparse_observed_anchor_layer") is True and surface.get("actual_landscape_layer") is False
    _assert(dense_field or sparse_anchor, errors, "NLL surface is neither a genuine dense model field nor a truthful sparse observed-anchor mesh")
    surface_kind = surface.get("surface_kind")
    _assert(
        surface_kind in {"sample_supported_local_idw_surface", "sparse_observed_state_nll_anchor_mesh", "sparse_exact_triangular_nll_mesh"},
        errors,
        "NLL surface is neither a sample-supported local field nor a sparse point-anchored mesh",
    )
    local_sheet = surface.get("local_interpolating_sheet", {})
    surrogate = surface.get("surrogate_landscape_layer", {})
    if isinstance(local_sheet, dict) and local_sheet.get("available") is True:
        _assert(False, errors, "local NLL interpolating sheet is enabled despite the exact surface-contact contract")
    if isinstance(surrogate, dict):
        _assert(surrogate.get("available") is not True, errors, "retired global NLL surrogate layer is enabled")
    support_radius = surface.get("support_radius")
    if support_radius is not None:
        _assert(_finite_float(support_radius, -1.0) > 0.0, errors, "NLL surface payload has non-positive support_radius")

    nll_progress = payload.get("nll_progress", {})
    _assert(isinstance(nll_progress, dict), errors, "trajectory payload is missing NLL progress diagnostics")
    _assert(_finite_float(nll_progress.get("edge_count"), 0.0) > 0, errors, "NLL progress diagnostics have no edges")
    _assert("improving_edge_fraction" in nll_progress, errors, "NLL progress diagnostics missing improving edge fraction")
    _assert("by_level" in nll_progress, errors, "NLL progress diagnostics missing level summary")

    density_cloud = nll_density_payload.get("density_cloud", {}) if isinstance(nll_density_payload, dict) else {}
    density_contract = nll_density_payload.get("density_contract", {}) if isinstance(nll_density_payload, dict) else {}
    density_support = nll_density_payload.get("support_samples", {}) if isinstance(nll_density_payload, dict) else {}
    density_render_contract = str(nll_density_payload.get("render_contract") or density_cloud.get("render_contract") or "")
    density_visual_contract = nll_density_payload.get("visual_layer_contract") if isinstance(nll_density_payload, dict) else {}
    density_anchor_count = nll_density_payload.get("anchor_count", density_cloud.get("anchor_count"))
    density_actual_anchor_count = nll_density_payload.get("actual_model_anchor_count", density_cloud.get("anchor_count"))
    density_sample_count = nll_density_payload.get("support_sample_count", density_cloud.get("sample_count"))
    density_kernel_bandwidth = nll_density_payload.get("kernel_bandwidth", density_cloud.get("sigma"))
    density_volume = nll_density_payload.get("density_volume") or density_cloud.get("density_volume", {})
    density_samples_hidden = nll_density_payload.get(
        "support_samples_hidden_as_model_states",
        density_cloud.get("support_samples_are_not_model_states"),
    )
    density_samples_are_states = nll_density_payload.get("sample_points_are_model_states")
    density_support_trace_visibility = nll_density_payload.get(
        "support_sample_trace_visibility",
        density_cloud.get("support_sample_trace_visibility", density_contract.get("support_sample_trace_visibility")),
    )
    if isinstance(density_visual_contract, dict):
        density_support_trace_visibility = density_visual_contract.get("support_sample_trace_visibility", density_support_trace_visibility)
    density_z_axis_policy = nll_density_payload.get("z_axis_policy", density_cloud.get("z_axis_policy"))
    if isinstance(density_visual_contract, dict):
        density_z_axis_policy = density_visual_contract.get("z_axis_policy", density_z_axis_policy)
    density_visible_layers = nll_density_payload.get("visible_density_layers", density_cloud.get("visible_density_layers"))
    if isinstance(density_visual_contract, dict):
        density_visible_layers = density_visual_contract.get("visible_density_layers", density_visible_layers)
    nll_range = nll_density_payload.get("nll_range", {}) if isinstance(nll_density_payload, dict) else {}
    if not isinstance(nll_range, dict) or "span" not in nll_range:
        nll_min = density_cloud.get("nll_min")
        nll_max = density_cloud.get("nll_max")
        if math.isfinite(_finite_float(nll_min)) and math.isfinite(_finite_float(nll_max)):
            nll_range = {"min": nll_min, "max": nll_max, "span": _finite_float(nll_max) - _finite_float(nll_min)}
    _assert(nll_density_payload.get("available") is True, errors, "NLL density payload is unavailable")
    _assert(isinstance(density_cloud, dict) and density_cloud.get("available") is True, errors, "NLL density cloud metadata is unavailable")
    _assert("not model states" in density_render_contract or "not model state" in density_render_contract, errors, "NLL density payload render contract does not distinguish density samples from model states")
    _assert(isinstance(density_visual_contract, dict) and density_visual_contract.get("schema_version") == "tropicalgt.nll_density_render.v1", errors, "NLL density payload is missing visual-layer render contract")
    if isinstance(density_visual_contract, dict):
        _assert(density_visual_contract.get("no_proxy_or_fallback") is True, errors, "NLL density visual-layer contract is missing no-proxy flag")
        _assert(density_visual_contract.get("actual_anchor_layer_visible_by_default") is True, errors, "NLL density visual-layer contract is missing visible actual-anchor layer")
        _assert(density_visual_contract.get("support_samples_are_model_states") is False, errors, "NLL density visual-layer contract incorrectly treats support samples as model states")
        _assert(density_visual_contract.get("support_samples_hidden_as_model_states") is True, errors, "NLL density visual-layer contract does not hide support samples as model states")
        _assert(int(_finite_float(density_visual_contract.get("actual_model_anchor_count"), 0.0)) == len(nodes), errors, "NLL density visual-layer contract anchor count does not match trajectory node count")
        _assert(int(_finite_float(density_visual_contract.get("support_sample_count"), 0.0)) >= len(nodes), errors, "NLL density visual-layer contract has too few support samples")
        _assert(_finite_float(density_visual_contract.get("kernel_bandwidth"), -1.0) > 0.0, errors, "NLL density visual-layer contract has non-positive kernel bandwidth")
    _assert(density_support_trace_visibility == "legendonly", errors, "NLL density support samples must render legend-only")
    _assert(isinstance(density_z_axis_policy, str) and density_z_axis_policy.startswith("z is PC3"), errors, "NLL density visual-layer contract is missing PC3 z-axis policy")
    _assert(isinstance(density_visible_layers, list) and {"density_volume", "actual_model_anchor_markers"}.issubset(set(density_visible_layers)), errors, "NLL density visual-layer contract is missing visible density layers")
    if density_contract:
        _assert(density_contract.get("actual_model_anchor_layer") is True, errors, "NLL density contract is missing actual-anchor layer provenance")
        _assert(density_contract.get("support_samples_hidden_as_model_states") is True, errors, "NLL density contract does not hide support samples as model states")
        _assert(density_contract.get("sample_points_are_model_states") is False, errors, "NLL density contract incorrectly treats sample points as model states")
        if "support_sample_trace_visibility" in density_contract:
            _assert(density_contract.get("support_sample_trace_visibility") == "legendonly", errors, "NLL density contract support samples must render legend-only")
    _assert(density_samples_hidden is True, errors, "NLL density payload does not hide support samples as model states")
    if density_samples_are_states is not None:
        _assert(density_samples_are_states is False, errors, "NLL density payload incorrectly treats support samples as model states")
    _assert(int(_finite_float(density_anchor_count, 0.0)) == len(nodes), errors, "NLL density anchor count does not match trajectory node count")
    _assert(int(_finite_float(density_actual_anchor_count, 0.0)) == len(nodes), errors, "NLL density actual anchor count does not match trajectory node count")
    _assert(int(_finite_float(density_sample_count, 0.0)) >= len(nodes), errors, "NLL density payload has too few support samples")
    _assert(_finite_float(density_kernel_bandwidth, -1.0) > 0.0, errors, "NLL density payload has non-positive kernel bandwidth")
    _assert(isinstance(nll_range, dict) and _finite_float(nll_range.get("span"), -1.0) >= 0.0, errors, "NLL density payload is missing NLL range")
    local_nll_summary = nll_density_payload.get("local_nll_summary", {})
    if isinstance(local_nll_summary, dict) and local_nll_summary:
        _assert(int(_finite_float(local_nll_summary.get("count"), 0.0)) == int(_finite_float(density_sample_count, 0.0)), errors, "NLL density local-NLL summary does not cover support samples")
    edge_delta_summary = nll_density_payload.get("edge_nll_delta_summary", {})
    if isinstance(edge_delta_summary, dict) and edge_delta_summary:
        _assert(int(_finite_float(edge_delta_summary.get("count"), 0.0)) == len(edges), errors, "NLL density edge-delta summary does not match trajectory edges")
    if "terminal_nll_progress" in nll_density_payload:
        _assert(isinstance(nll_density_payload.get("terminal_nll_progress"), dict), errors, "NLL density payload is missing terminal NLL progress")
    if "anchors" in nll_density_payload:
        _assert(isinstance(nll_density_payload.get("anchors"), list) and len(nll_density_payload.get("anchors", [])) == len(nodes), errors, "NLL density payload does not list the actual anchors")
    if density_support:
        _assert(isinstance(density_support, dict) and density_support.get("visible_as_model_states") is False and density_support.get("visible_by_default") is False, errors, "NLL density support samples are not hidden legend-only visualization support")
    _assert(isinstance(density_volume, dict) and density_volume.get("support_samples_are_not_model_states") is True, errors, "NLL density volume is missing non-model-state provenance")

    support_metrics = support_payload.get("metrics", {}) if isinstance(support_payload, dict) else {}
    support_render_contract = support_payload.get("tropical_support_render_contract", {}) if isinstance(support_payload, dict) else {}
    support_readability_contract = support_payload.get("tropical_support_readability_contract", {}) if isinstance(support_payload, dict) else {}
    _assert(support_metrics.get("available") is True, errors, "tropical support payload is unavailable")
    _assert(_finite_float(support_metrics.get("token_count"), 0.0) > 0, errors, "tropical support payload has no tokens")
    _assert(_finite_float(support_metrics.get("unique_support_count"), 0.0) >= 1, errors, "tropical support payload has no observed supports")
    _assert("interpretation" in support_metrics, errors, "tropical support payload is missing collapse interpretation")
    _assert(isinstance(support_metrics.get("margin_summary"), dict), errors, "tropical support payload is missing margin summary")
    _assert(isinstance(support_render_contract, dict) and support_render_contract.get("schema_version") == "tropicalgt.tropical_support_render.v1", errors, "tropical support payload is missing no-proxy render contract")
    if isinstance(support_render_contract, dict):
        _assert(support_render_contract.get("no_proxy_or_fallback") is True, errors, "tropical support render contract is missing no-proxy flag")
        _assert(support_render_contract.get("support_columns_policy") == "observed_valid_active_support_indices_only", errors, "tropical support render contract has wrong support-column policy")
        _assert(support_render_contract.get("assignment_matrix_binary") is True, errors, "tropical support render contract does not mark assignment matrix as binary")
        _assert(str(support_render_contract.get("assignment_matrix_semantics", "")).startswith("binary argmax support-selection mask"), errors, "tropical support render contract is missing assignment-mask semantics")
        _assert(support_render_contract.get("normal_fan_wall_crossing_certified") is False, errors, "tropical support render contract incorrectly certifies normal-fan wall crossings")
        _assert(support_render_contract.get("wall_margin_metric_scope") == "margin_threshold_audit_not_certified_normal_fan_wall_crossing", errors, "tropical support render contract is missing wall-margin metric scope")
    _assert(
        isinstance(support_readability_contract, dict)
        and support_readability_contract.get("schema_version") == "tropicalgt.tropical_support_readability.v1",
        errors,
        "tropical support payload is missing readability contract",
    )
    if isinstance(support_readability_contract, dict):
        panel_roles = set(support_readability_contract.get("panel_roles", [])) if isinstance(support_readability_contract.get("panel_roles"), list) else set()
        required_roles = set(support_readability_contract.get("required_panel_roles", [])) if isinstance(support_readability_contract.get("required_panel_roles"), list) else set()
        layout_mode = str(support_readability_contract.get("layout_mode", support_metrics.get("layout_mode", "")))
        _assert(support_readability_contract.get("no_proxy_or_fallback") is True, errors, "tropical support readability contract is missing no-proxy flag")
        _assert(support_readability_contract.get("panels_are_separate") is True, errors, "tropical support readability contract does not separate panels")
        _assert(bool(required_roles) and required_roles.issubset(panel_roles), errors, "tropical support readability contract lacks required panel roles")
        _assert(support_readability_contract.get("assignment_and_margin_panels_separated") is True, errors, "tropical support readability contract merges assignment and margin evidence")
        _assert(support_readability_contract.get("support_strip_split_from_margin_profile") is True, errors, "tropical support readability contract does not split support strip from margin profile")
        _assert(support_readability_contract.get("model_probability_summaries_separate_from_assignment_matrix") is True, errors, "tropical support readability contract mixes support probabilities into assignment matrix")
        _assert(support_readability_contract.get("compact_tick_labels") is True, errors, "tropical support readability contract is missing compact tick label guarantee")
        _assert(support_readability_contract.get("full_token_text_preserved_in_hover_and_payload") is True, errors, "tropical support readability contract does not preserve full token text")
        _assert(support_readability_contract.get("exact_token_indices_preserved_in_payload") is True, errors, "tropical support readability contract does not preserve exact token ids")
        _assert(support_readability_contract.get("group_summaries_from_trace_fields") is True, errors, "tropical support readability contract does not use trace-backed group summaries")
        _assert(support_readability_contract.get("invalid_active_support_indices_not_fabricated") is True, errors, "tropical support readability contract allows fabricated invalid support cells")
        _assert(support_readability_contract.get("normal_fan_wall_crossing_certified") is False, errors, "tropical support readability contract incorrectly certifies normal-fan walls")
        if layout_mode == "observed_support_matrix":
            observed_roles = {"support_frequency", "mean_selected_margin_by_support", "query_token_category_strip"}
            _assert(observed_roles.issubset(panel_roles), errors, "tropical support observed-support layout is missing split support/category panels")
            _assert(support_readability_contract.get("support_frequency_and_mean_margin_split") is True, errors, "tropical support observed-support layout merges frequency and margin panels")
            _assert(support_readability_contract.get("query_token_category_strip_visible") is True, errors, "tropical support observed-support layout hides token categories")
        if layout_mode == "collapse_diagnostic":
            _assert({"margin_distribution", "collapse_metrics_table"}.issubset(panel_roles), errors, "tropical support collapse layout is missing diagnostic panels")
            _assert(support_readability_contract.get("collapse_diagnostic_visible") is True, errors, "tropical support collapse layout hides collapse diagnostics")
    support_probability_source = support_metrics.get("support_probability_source")
    active_prob_summary = support_metrics.get("active_support_probability_summary")
    probability_trace_available = (
        support_probability_source == "model_tropical_support_probabilities"
        and isinstance(active_prob_summary, dict)
        and active_prob_summary.get("available") is not False
    )
    legacy_probability_unavailable = support_probability_source in {"unavailable_in_trace", "missing_model_tropical_support_probabilities", None}
    wall_audit = support_metrics.get("wall_margin_audit")
    if isinstance(wall_audit, dict):
        _assert(_finite_float(wall_audit.get("near_wall_hit_rate"), -1.0) >= _finite_float(wall_audit.get("strict_wall_hit_rate"), 0.0), errors, "near-wall hit rate is below strict wall-hit rate")
        _assert(_finite_float(wall_audit.get("near_wall_margin_threshold"), -1.0) >= _finite_float(wall_audit.get("wall_margin_threshold"), 0.0), errors, "near-wall threshold is below strict wall threshold")
        _assert(wall_audit.get("metric_scope") == "margin_threshold_audit_not_certified_normal_fan_wall_crossing", errors, "wall margin audit is missing no-proxy metric scope")
        _assert(isinstance(wall_audit.get("low_strict_wall_interpretation_status"), str) and bool(wall_audit.get("low_strict_wall_interpretation_status")), errors, "wall margin audit is missing low-strict interpretation status")
        _assert(isinstance(wall_audit.get("low_strict_wall_interpretation"), str) and bool(wall_audit.get("low_strict_wall_interpretation")), errors, "wall margin audit is missing low-strict interpretation text")
        _assert(isinstance(wall_audit.get("metric_issue"), bool), errors, "wall margin audit is missing metric_issue boolean")
    else:
        _assert(legacy_probability_unavailable, errors, "tropical support payload is missing wall margin audit")
    _assert(str(support_metrics.get("render_contract", "")).startswith("assignment_matrix is binary model argmax support"), errors, "tropical support payload is missing binary assignment render contract")
    if probability_trace_available:
        _assert(support_probability_source == "model_tropical_support_probabilities", errors, "tropical support payload is missing model support-probability provenance")
    else:
        _assert(legacy_probability_unavailable, errors, "tropical support payload has unavailable support probabilities without recognized provenance")
    _assert(isinstance(support_metrics.get("active_support_probability_summary"), dict), errors, "tropical support payload is missing active-support probability summary")
    _assert(isinstance(support_metrics.get("support_probability_entropy_bits_summary"), dict), errors, "tropical support payload is missing support-probability entropy summary")
    flow_edges = support_payload.get("support_flow_edges", []) if isinstance(support_payload, dict) else []
    status_by_token = support_payload.get("support_assignment_status_by_token", []) if isinstance(support_payload, dict) else []
    token_count = int(_finite_float(support_metrics.get("token_count"), 0.0))
    invalid_support_count = int(_finite_float(support_metrics.get("invalid_support_count"), 0.0))
    _assert(isinstance(flow_edges, list) and len(flow_edges) >= token_count, errors, "tropical support payload is missing query-to-support flow edges")
    _assert(isinstance(status_by_token, list) and len(status_by_token) >= token_count, errors, "tropical support payload is missing support assignment status rows")
    if isinstance(support_render_contract, dict):
        _assert(int(_finite_float(support_render_contract.get("token_count"), 0.0)) == token_count, errors, "tropical support render contract token count does not match metrics")
        _assert(int(_finite_float(support_render_contract.get("invalid_support_count"), -1.0)) == invalid_support_count, errors, "tropical support render contract invalid-support count does not match metrics")
    for edge in flow_edges if isinstance(flow_edges, list) else []:
        if not isinstance(edge, dict):
            errors.append("tropical support payload contains non-object support-flow edge")
            continue
        query_idx = int(_finite_float(edge.get("query_index"), -1.0))
        support_idx = int(_finite_float(edge.get("support_index"), -1.0))
        assignment_status = edge.get("support_assignment_status")
        _assert(0 <= query_idx < token_count, errors, "tropical support flow has out-of-range query_index")
        if 0 <= support_idx < token_count:
            _assert(assignment_status in {"selected_observed_support", None}, errors, "tropical support flow has valid support index but wrong assignment status")
            _assert(edge.get("rendered_as_assignment_cell") is not False, errors, "tropical support flow has valid support index but is not rendered as assignment cell")
        else:
            _assert(assignment_status == "invalid_active_support_index", errors, "tropical support flow with out-of-range support index is not marked invalid")
            _assert(edge.get("rendered_as_assignment_cell") is False, errors, "tropical support flow fabricates an assignment cell for invalid support index")
        if probability_trace_available:
            _assert(_finite_float(edge.get("active_support_probability"), -1.0) >= 0.0, errors, "tropical support flow is missing active support probability")
            _assert(edge.get("support_probability_source") == "model_tropical_support_probabilities", errors, "tropical support flow is missing probability provenance")
        else:
            _assert(edge.get("support_probability_source") in {None, "unavailable_in_trace", "missing_model_tropical_support_probabilities"}, errors, "tropical support flow has unavailable probability values without recognized provenance")
        if isinstance(wall_audit, dict):
            _assert(edge.get("wall_margin_bucket") in {"strict_wall", "near_wall", "interior", "unavailable"}, errors, "tropical support flow is missing wall margin bucket")
            _assert("strict_wall_hit" in edge and "near_wall_hit" in edge, errors, "tropical support flow is missing strict/near wall flags")
        _assert(isinstance(edge.get("top_model_support_probabilities"), list), errors, "tropical support flow is missing top model support probabilities")

    graphcg_available = graphcg_payload.get("available") is True
    _assert(graphcg_available, errors, "GraphCG payload is unavailable")
    matrix_shape = graphcg_payload.get("matrix_shape", [])
    matrix_width = int(_finite_float(matrix_shape[1], 0.0)) if isinstance(matrix_shape, list) and len(matrix_shape) == 2 else 0
    _assert(isinstance(matrix_shape, list) and len(matrix_shape) == 2 and int(_finite_float(matrix_shape[0], 0.0)) >= len(candidates), errors, "GraphCG payload has invalid matrix shape")
    _assert(matrix_width > 0, errors, "GraphCG payload has invalid matrix width")
    if graphcg_available and matrix_width > 0:
        _assert(_finite_float(graphcg_payload.get("active_rank_nonzero_mean_abs"), 0.0) > 0, errors, "GraphCG payload reports no active directions")
        _assert(int(_finite_float(graphcg_payload.get("full_rank_direction_count"), 0.0)) == matrix_width, errors, "GraphCG full-rank direction count does not match matrix width")
        _assert(len(graphcg_payload.get("candidate_effective_direction_count", [])) >= len(candidates), errors, "GraphCG payload is missing candidate effective-direction counts")
        _assert(len(graphcg_payload.get("direction_activity_sorted", [])) == matrix_width, errors, "GraphCG payload is missing full direction activity spectrum")
        _assert(graphcg_payload.get("interpretation"), errors, "GraphCG payload is missing heatmap interpretation")
        readability = graphcg_payload.get("graphcg_readability_contract")
        _assert(isinstance(readability, dict) and bool(readability), errors, "GraphCG payload is missing structured readability contract")
        if isinstance(readability, dict) and readability:
            _assert(readability.get("schema_version") == "tropicalgt.graphcg_direction_readability.v1", errors, "GraphCG readability contract has wrong schema")
            _assert(readability.get("all_model_directions_rendered") is True, errors, "GraphCG readability contract does not render all directions")
            _assert(readability.get("directions_sampled_for_heatmap") is False, errors, "GraphCG heatmap must not sample directions")
            _assert(readability.get("panels_are_separate") is True, errors, "GraphCG readability contract must separate panels")
            required = {"all_direction_heatmap", "top_active_direction_panel", "full_rank_activity_spectrum", "candidate_activity_by_observed_got_state", "direction_signed_bias"}
            required_panels = set(readability.get("required_panels", []))
            panel_names = set(graphcg_payload.get("panel_names", [])) if isinstance(graphcg_payload.get("panel_names"), list) else set()
            _assert(required.issubset(required_panels), errors, "GraphCG readability contract is missing required panels")
            _assert(required.issubset(panel_names), errors, "GraphCG payload panel_names are missing required panels")
            _assert(graphcg_payload.get("top_active_direction_panel_available") is True, errors, "GraphCG payload is missing top-active direction panel")
            _assert(readability.get("exact_direction_ids_preserved_in_hover_and_payload") is True, errors, "GraphCG readability contract does not preserve exact direction ids")
            _assert(readability.get("candidate_path_action_text_preserved_in_hover_and_payload") is True, errors, "GraphCG readability contract does not preserve candidate path/action text")
        direction_rows = graphcg_payload.get("direction_rows") if isinstance(graphcg_payload.get("direction_rows"), list) else []
        direction_evidence = graphcg_payload.get("graphcg_direction_evidence_contract", {})
        _assert(isinstance(direction_evidence, dict) and bool(direction_evidence), errors, "GraphCG payload is missing per-direction evidence contract")
        if isinstance(direction_evidence, dict) and direction_evidence:
            _assert(direction_evidence.get("schema_version") == "tropicalgt.graphcg_direction_evidence.v1", errors, "GraphCG direction evidence contract has wrong schema")
            _assert(direction_evidence.get("source") == "candidate.graphcg_projection.all_direction_cosines", errors, "GraphCG direction evidence contract has wrong source")
            _assert(direction_evidence.get("no_proxy_or_fallback") is True, errors, "GraphCG direction evidence contract allows proxy/fallback data")
            _assert(direction_evidence.get("all_model_directions_have_rows") is True, errors, "GraphCG direction evidence contract does not cover all directions")
            _assert(int(_finite_float(direction_evidence.get("direction_count"), 0.0)) == matrix_width, errors, "GraphCG direction evidence direction count mismatch")
            _assert(int(_finite_float(direction_evidence.get("direction_row_count"), 0.0)) == matrix_width, errors, "GraphCG direction evidence row count mismatch")
            _assert(direction_evidence.get("exact_direction_ids_preserved") is True, errors, "GraphCG direction evidence does not preserve exact ids")
            _assert(direction_evidence.get("all_directions_rendered_in_heatmap") is True, errors, "GraphCG direction evidence says heatmap omits directions")
            _assert(direction_evidence.get("all_directions_rendered_in_activity_spectrum") is True, errors, "GraphCG direction evidence says activity spectrum omits directions")
            _assert(direction_evidence.get("all_directions_rendered_in_signed_bias_panel") is True, errors, "GraphCG direction evidence says signed-bias panel omits directions")
            _assert(direction_evidence.get("safe_to_render_full_rank_direction_evidence") is True, errors, "GraphCG direction evidence contract is unsafe")
        top_active_rows = graphcg_payload.get("top_active_direction_rows") if isinstance(graphcg_payload.get("top_active_direction_rows"), list) else []
        _assert(top_active_rows and len(top_active_rows) <= matrix_width, errors, "GraphCG payload is missing bounded top-active direction rows")
        for idx, row in enumerate(top_active_rows):
            if not isinstance(row, dict):
                errors.append(f"GraphCG top-active direction row {idx} is not an object")
                continue
            _assert(row.get("source") == "candidate.graphcg_projection.all_direction_cosines", errors, f"GraphCG top-active direction row {idx} has wrong source")
            _assert(row.get("no_proxy_or_fallback") is True, errors, f"GraphCG top-active direction row {idx} allows proxy/fallback data")
            _assert(row.get("exact_direction_id_preserved") is True, errors, f"GraphCG top-active direction row {idx} does not preserve exact id")
            _assert(row.get("rendered_in_top_active_direction_panel") is True, errors, f"GraphCG top-active direction row {idx} missing top-panel flag")
            _assert(math.isfinite(_finite_float(row.get("mean_abs_cosine"))), errors, f"GraphCG top-active direction row {idx} missing finite mean_abs_cosine")
            _assert(math.isfinite(_finite_float(row.get("signed_mean_cosine"))), errors, f"GraphCG top-active direction row {idx} missing finite signed_mean_cosine")
        _assert(len(direction_rows) == matrix_width, errors, "GraphCG direction rows do not cover every model direction")
        seen_direction_ids = set()
        for idx, row in enumerate(direction_rows):
            if not isinstance(row, dict):
                errors.append(f"GraphCG direction row {idx} is not an object")
                continue
            direction_id = int(_finite_float(row.get("direction_id"), -1.0))
            seen_direction_ids.add(direction_id)
            _assert(row.get("source") == "candidate.graphcg_projection.all_direction_cosines", errors, f"GraphCG direction row {idx} has wrong source")
            _assert(row.get("no_proxy_or_fallback") is True, errors, f"GraphCG direction row {idx} allows proxy/fallback data")
            _assert(row.get("exact_direction_id_preserved") is True, errors, f"GraphCG direction row {idx} does not preserve exact id")
            _assert(row.get("rendered_in_all_direction_heatmap") is True, errors, f"GraphCG direction row {idx} missing heatmap flag")
            _assert(row.get("rendered_in_full_rank_activity_spectrum") is True, errors, f"GraphCG direction row {idx} missing activity-spectrum flag")
            _assert(row.get("rendered_in_signed_bias_panel") is True, errors, f"GraphCG direction row {idx} missing signed-bias flag")
            _assert(math.isfinite(_finite_float(row.get("mean_abs_cosine"))), errors, f"GraphCG direction row {idx} missing finite mean_abs_cosine")
            _assert(math.isfinite(_finite_float(row.get("signed_mean_cosine"))), errors, f"GraphCG direction row {idx} missing finite signed_mean_cosine")
            _assert(int(_finite_float(row.get("activity_rank_desc"), 0.0)) >= 1, errors, f"GraphCG direction row {idx} missing activity rank")
        _assert(seen_direction_ids == set(range(matrix_width)), errors, "GraphCG direction rows do not preserve exact contiguous direction ids")
        _assert(len(graphcg_payload.get("candidate_hover_rows", [])) >= len(candidates), errors, "GraphCG payload is missing candidate hover rows")
        basis_certificate = graphcg_payload.get("projection_basis_certificate", {})
        _assert(isinstance(basis_certificate, dict), errors, "GraphCG payload is missing projection-basis certificate")
        if isinstance(basis_certificate, dict):
            _assert(basis_certificate.get("source") == "candidate.graphcg_projection", errors, "GraphCG projection-basis certificate has wrong source")
            _assert(basis_certificate.get("available") is True, errors, "GraphCG projection-basis certificate is unavailable")
            _assert(bool(basis_certificate.get("basis_sources")), errors, "GraphCG projection-basis certificate has no basis sources")
            _assert(int(_finite_float(basis_certificate.get("candidate_count"), 0.0)) >= len(candidates), errors, "GraphCG projection-basis certificate does not cover candidates")
            _assert(int(_finite_float(basis_certificate.get("direction_count"), 0.0)) == matrix_width, errors, "GraphCG projection-basis direction count does not match matrix width")
            _assert(basis_certificate.get("all_candidates_have_all_direction_cosines") is True, errors, "GraphCG projection-basis certificate is missing full direction cosines")

    _assert(len(embedded_nodes) == len(nodes), errors, "embedding map payload node count differs from trajectory payload")
    _assert(embedding_payload.get("coordinate_source", "").startswith("PCA of model graph_state embeddings"), errors, "embedding map payload has wrong coordinate source")
    embedding_layout = embedding_payload.get("layout_contract", {}) if isinstance(embedding_payload.get("layout_contract"), dict) else {}
    embedded_edges = embedding_payload.get("edges", []) if isinstance(embedding_payload.get("edges"), list) else []
    _assert(embedding_layout.get("schema_version") == "tropicalgt.embedding_trajectory_identity.v1", errors, "embedding map payload lacks trajectory identity contract")
    _assert(embedding_layout.get("coordinate_source") == "model graph_state embeddings", errors, "embedding map trajectory identity has wrong coordinate source")
    _assert(embedding_layout.get("branch_depth_metadata_present") is True, errors, "embedding map trajectory identity lacks branch/depth metadata")
    _assert(embedding_layout.get("parent_child_transitions_present") is True, errors, "embedding map trajectory identity lacks parent-child transitions")
    _assert(embedding_layout.get("edge_source") == "graph_of_thought_parent_edges", errors, "embedding map trajectory identity has wrong edge source")
    _assert(embedding_layout.get("node_embedding_source") == "model graph_state", errors, "embedding map trajectory identity has wrong node embedding source")
    _assert(embedding_layout.get("geometric_separation_overclaim_allowed") is False, errors, "embedding map trajectory identity permits geometric separation overclaims")
    _assert(embedding_layout.get("no_proxy_or_fallback") is True, errors, "embedding map trajectory identity lacks no-proxy flag")
    _assert(int(_finite_float(embedding_layout.get("parent_child_transition_count"), -1.0)) == len(embedded_edges), errors, "embedding map parent-child transition count mismatches edges")
    for index, node in enumerate(embedded_nodes):
        if not isinstance(node, dict):
            continue
        _assert(node.get("embedding_source") == "model graph_state", errors, f"embedding map node {index} lacks model graph_state source")
        _assert("branch_id" in node and "depth" in node, errors, f"embedding map node {index} lacks branch/depth metadata")
    for index, edge in enumerate(embedded_edges):
        if not isinstance(edge, dict):
            continue
        _assert(edge.get("edge_source") == "graph_of_thought_parent_edges", errors, f"embedding map edge {index} has wrong source")
        _assert(edge.get("transition_kind") == "GoT parent-child trajectory", errors, f"embedding map edge {index} has wrong transition kind")
        _assert(edge.get("source") and edge.get("target"), errors, f"embedding map edge {index} lacks source/target")
    _assert(len(embedding_objects) == len(embedded_nodes), errors, "embedding map payload does not include one filtered simplicial object per node")
    _assert(
        any(isinstance(obj.get("simplices"), list) and obj.get("simplices") for obj in embedding_objects),
        errors,
        "embedding map filtered simplicial objects are empty",
    )
    embedding_html = _read_text(row_dir / REQUIRED_HTML["embedding_map"][0]) if (row_dir / REQUIRED_HTML["embedding_map"][0]).exists() else ""
    _assert("simplicial-object-panel" in embedding_html and "hover-simplicial-card" in embedding_html, errors, "embedding map does not expose filtered-complex hover panel")

    overlay_contract = full_complex_payload.get("trajectory_complex_overlay_contract", {}) if isinstance(full_complex_payload, dict) else {}
    _assert(isinstance(overlay_contract, dict), errors, "trajectory complex overlay contract is missing")
    if isinstance(overlay_contract, dict):
        _assert(overlay_contract.get("schema_version") == "tropicalgt.trajectory_complex_overlay_contract.v1", errors, "trajectory complex overlay contract has wrong schema")
        _assert(overlay_contract.get("actual_data_only") is True, errors, "trajectory complex overlay contract is not actual-data-only")
        _assert(overlay_contract.get("no_proxy_or_fallback") is True, errors, "trajectory complex overlay contract allows proxy/fallback data")
        _assert(overlay_contract.get("solid_lines_reserved_for_radius_simplices") is True, errors, "trajectory complex overlay contract does not reserve solid lines for radius simplices")
        _assert(overlay_contract.get("dotted_lines_reserved_for_trajectory_decoding_order_overlays") is True, errors, "trajectory complex overlay contract does not reserve dotted lines for overlays")
        _assert(overlay_contract.get("safe_to_render_available_views") is True, errors, "trajectory complex overlay contract marks available views unsafe")
    full_obj = full_complex_payload.get("filtered_simplicial_object", {})
    summary = full_obj.get("summary", {}) if isinstance(full_obj, dict) else {}
    simplex_tree = full_obj.get("simplex_tree", {}) if isinstance(full_obj, dict) and isinstance(full_obj.get("simplex_tree"), dict) else {}
    full_vertices = [row for row in full_obj.get("simplices", []) if isinstance(row, dict) and int(row.get("dimension", -1)) == 0] if isinstance(full_obj, dict) else []
    _assert(simplex_tree.get("backend") == "gudhi.SimplexTree", errors, "full trajectory complex payload is missing GUDHI SimplexTree provenance")
    prob_obj = full_complex_payload.get("probability_filtered_simplicial_object", {}) if isinstance(full_complex_payload, dict) else {}
    if isinstance(overlay_contract, dict):
        _validate_trajectory_overlay_view_contract(
            overlay_contract.get("embedding_view"),
            errors,
            "embedding trajectory complex",
            expected_metric="euclidean",
            required_available=True,
        )
    if isinstance(prob_obj, dict):
        if prob_obj.get("available") is False:
            _assert(prob_obj.get("reason") in {"missing_model_probability_vectors", "unavailable_no_embedding_or_probability_radius_edges"}, errors, "probability complex unavailable for an unrecognized reason")
            if isinstance(overlay_contract, dict):
                _validate_trajectory_overlay_view_contract(
                    overlay_contract.get("probability_view"),
                    errors,
                    "probability trajectory complex",
                    expected_metric="jensen_shannon",
                    required_available=False,
                )
        else:
            prob_summary = prob_obj.get("summary", {}) if isinstance(prob_obj.get("summary"), dict) else {}
            _assert(prob_summary.get("filtration_model") == "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton", errors, "probability complex is not a model-probability Jensen-Shannon filtration")
            _assert(int(prob_summary.get("num_edges", 0) or 0) > 0, errors, "probability complex has no Jensen-Shannon radius edges")
            prob_tree = prob_obj.get("simplex_tree", {}) if isinstance(prob_obj.get("simplex_tree"), dict) else {}
            _assert(prob_tree.get("backend") == "gudhi.SimplexTree", errors, "probability complex is missing GUDHI SimplexTree provenance")
            if isinstance(overlay_contract, dict):
                _validate_trajectory_overlay_view_contract(
                    overlay_contract.get("probability_view"),
                    errors,
                    "probability trajectory complex",
                    expected_metric="jensen_shannon",
                    required_available=True,
                )
                _assert(overlay_contract.get("probability_view_available") is True, errors, "trajectory overlay contract says probability view unavailable")
                _assert(overlay_contract.get("safe_to_render_probability_view") is True, errors, "trajectory overlay contract says probability view is unsafe")
    _assert(int(summary.get("num_vertices", 0) or 0) >= len(candidates), errors, "full trajectory complex has fewer vertices than candidates")
    _assert(int(summary.get("num_edges", 0) or 0) >= len(edges), errors, "full trajectory complex has fewer edges than trajectory")
    _assert(sum(1 for row in full_vertices if row.get("embedding")) == len(full_vertices), errors, "full trajectory complex vertices do not all carry embeddings")
    _assert(sum(1 for row in full_vertices if row.get("input_text") or row.get("decoded_argmax")) == len(full_vertices), errors, "full trajectory complex vertices do not all carry model I/O")
    _validate_radius_slider_contract(
        _slider_contract_path(row_dir / REQUIRED_HTML["full_complex"][0]),
        errors,
        "full trajectory complex",
    )
    _validate_simplex_tree_poset_contract(
        _simplex_tree_poset_contract_path(row_dir / REQUIRED_HTML["full_simplex_tree"][0]),
        errors,
        "full trajectory simplex tree",
        required_available=True,
    )
    if isinstance(prob_obj, dict) and prob_obj.get("available") is not False:
        _validate_radius_slider_contract(
            _slider_contract_path(row_dir / REQUIRED_HTML["probability_complex"][0]),
            errors,
            "probability trajectory complex",
        )
        _validate_simplex_tree_poset_contract(
            _simplex_tree_poset_contract_path(row_dir / REQUIRED_HTML["probability_simplex_tree"][0]),
            errors,
            "probability trajectory simplex tree",
            required_available=True,
        )
    elif isinstance(prob_obj, dict):
        _validate_simplex_tree_poset_contract(
            _simplex_tree_poset_contract_path(row_dir / REQUIRED_HTML["probability_simplex_tree"][0]),
            errors,
            "probability trajectory simplex tree",
            required_available=False,
        )

    maps_path = row_dir / "analogical_simplicial_maps.json"
    if maps_path.exists():
        map_payload = _read_json(maps_path)
        maps = map_payload.get("maps", [])
        analogy_contract = analogical_simplex_tree_payload.get("contract", {}) if isinstance(analogical_simplex_tree_payload, dict) else {}
        analogy_pairs = analogical_simplex_tree_payload.get("pairs", []) if isinstance(analogical_simplex_tree_payload, dict) else []
        _assert(
            isinstance(analogy_contract, dict) and analogy_contract.get("schema_version") == "tropicalgt.analogical_simplex_tree_analogy.v1",
            errors,
            "analogical simplex-tree analogy contract is missing or has wrong schema",
        )
        if isinstance(analogy_contract, dict):
            _assert(analogy_contract.get("no_proxy_or_fallback") is True, errors, "analogical simplex-tree analogy contract is missing no-proxy flag")
            _assert(analogy_contract.get("compares_query_and_memory_simplex_trees") is True, errors, "analogical simplex-tree analogy does not compare query and memory trees")
            _assert(analogy_contract.get("renders_hasse_face_to_coface_rows") is True, errors, "analogical simplex-tree analogy omits Hasse face-to-coface rows")
            _assert(analogy_contract.get("preserved_face_coface_chains_highlighted") is True, errors, "analogical simplex-tree analogy does not highlight preserved face/coface chains")
            _assert(analogy_contract.get("failed_or_distorted_chains_labeled_not_maps") is True, errors, "analogical simplex-tree analogy overclaims failed chains as maps")
            _assert(analogy_contract.get("chain_map_claim_requires_certified_filtered_simplicial_map") is True, errors, "analogical simplex-tree analogy permits uncertified chain-map claims")
            _assert(analogy_contract.get("persistence_module_morphism_claim_requires_certified_filtered_simplicial_map") is True, errors, "analogical simplex-tree analogy permits uncertified persistence-module morphisms")
            _assert(str(analogy_contract.get("source", "")) == "probability_simplicial_map.simplex_tree_map.rows", errors, "analogical simplex-tree analogy has wrong source rows")
        topk_contract = map_payload.get("topk_contract", {}) if isinstance(map_payload, dict) else {}
        topk_readability = topk_contract.get("readability_contract", {}) if isinstance(topk_contract, dict) else {}
        _assert(isinstance(topk_contract, dict) and topk_contract.get("schema_version") == "tropicalgt.analogical_topk.v1", errors, "analogical top-k contract is missing or has wrong schema")
        if isinstance(topk_contract, dict):
            _assert(topk_contract.get("no_proxy_or_fallback") is True, errors, "analogical top-k contract is missing no-proxy flag")
            _assert(topk_contract.get("retrieval_requires_model_probability_vectors") is True, errors, "analogical top-k contract does not require model probability vectors")
            _assert(topk_contract.get("embedding_only_assignment_allowed") is False, errors, "analogical top-k contract allows embedding-only assignment")
            _assert(topk_contract.get("assignment_metric") == "jensen_shannon_distance_on_model_probability_vectors", errors, "analogical top-k contract has wrong assignment metric")
            _assert(topk_contract.get("query_complex_required") == "trajectory_probability_filtered_simplicial_object", errors, "analogical top-k contract has wrong query complex requirement")
            _assert(topk_contract.get("codomain_complex_required") == "trajectory_probability_filtered_simplicial_object", errors, "analogical top-k contract has wrong codomain complex requirement")
            _assert(
                isinstance(topk_readability, dict) and topk_readability.get("schema_version") == "tropicalgt.analogical_topk_readability.v1",
                errors,
                "analogical top-k readability contract is missing or has wrong schema",
            )
            if isinstance(topk_readability, dict):
                required_columns = set(topk_readability.get("required_table_columns", [])) if isinstance(topk_readability.get("required_table_columns"), list) else set()
                expected_columns = {"correspondence", "retrieval", "prob-map source", "map claim", "derived/algebraic", "coarse signature", "simplex-tree map", "edge certificate"}
                _assert(topk_readability.get("no_proxy_or_fallback") is True, errors, "analogical top-k readability contract is missing no-proxy flag")
                _assert(topk_readability.get("topk_index_has_readable_table") is True, errors, "analogical top-k readability contract does not require a readable table")
                _assert(topk_readability.get("one_selected_map_view_per_rendered_rank") is True, errors, "analogical top-k readability contract does not require one map per rendered rank")
                _assert(topk_readability.get("table_rows_link_to_pair_pages") is True, errors, "analogical top-k readability contract does not require linked pair pages")
                _assert(topk_readability.get("insufficient_memory_state_explicit") is True, errors, "analogical top-k readability contract hides insufficient-memory state")
                _assert(topk_readability.get("displays_quality_gate_and_filtered_counts") is True, errors, "analogical top-k readability contract omits quality-gate/filter counts")
                _assert(topk_readability.get("separates_retrieval_probability_topology_algebra_columns") is True, errors, "analogical top-k readability contract merges evidence columns")
                _assert(topk_readability.get("probability_js_assignment_column_required") is True, errors, "analogical top-k readability contract omits probability-JS assignment column")
                _assert(topk_readability.get("map_claim_column_required") is True, errors, "analogical top-k readability contract omits map-claim column")
                _assert(topk_readability.get("simplex_tree_preservation_column_required") is True, errors, "analogical top-k readability contract omits simplex-tree preservation column")
                _assert(topk_readability.get("edge_face_filtration_preservation_not_overclaimed") is True, errors, "analogical top-k readability contract allows overclaimed preservation")
                _assert(expected_columns.issubset(required_columns), errors, "analogical top-k readability contract lacks required table columns")
                _assert("coarse signature" in str(topk_readability.get("signature_cosine_column_policy", "")), errors, "analogical top-k readability contract does not separate coarse signature cosine")
                _assert("cannot substitute" in str(topk_readability.get("derived_algebraic_column_policy", "")), errors, "analogical top-k readability contract does not clamp derived/algebraic overclaims")
        if map_payload.get("available") is False:
            _assert(map_payload.get("reason") in {"missing_model_probability_query_complex", "missing_model_probability_codomain_complex", "no_non_self_model_memory"}, errors, "analogical maps are unavailable for an unrecognized reason")
            if isinstance(topk_contract, dict):
                _assert(int(_finite_float(topk_contract.get("top_k_rendered"), -1.0)) == 0, errors, "unavailable analogical top-k contract renders map rows")
            if isinstance(analogy_contract, dict):
                _assert(analogy_contract.get("available") is False and int(_finite_float(analogy_contract.get("pair_count"), -1.0)) == 0, errors, "unavailable analogical simplex-tree analogy renders pair rows")
        else:
            allowed_sources = {"trajectory_probability_filtered_simplicial_object"}
            _assert(bool(maps), errors, "analogical_simplicial_maps.json contains no maps")
            _assert(isinstance(analogy_pairs, list) and len(analogy_pairs) == len(maps), errors, "analogical simplex-tree analogy pair count does not match maps")
            if isinstance(analogy_contract, dict):
                _assert(analogy_contract.get("available") is True, errors, "available analogical simplex-tree analogy is not marked available")
                _assert(int(_finite_float(analogy_contract.get("pair_count"), -1.0)) == len(maps), errors, "analogical simplex-tree analogy contract pair_count does not match maps")
                _assert(int(_finite_float(analogy_contract.get("total_checked_simplices"), -1.0)) >= len(maps), errors, "analogical simplex-tree analogy checked no simplices")
            for pair in analogy_pairs if isinstance(analogy_pairs, list) else []:
                if not isinstance(pair, dict):
                    errors.append("analogical simplex-tree analogy pair is not an object")
                    continue
                _assert(isinstance(pair.get("simplex_rows"), list) and bool(pair.get("simplex_rows")), errors, "analogical simplex-tree analogy pair lacks simplex rows")
                _assert(isinstance(pair.get("preserved_face_coface_chains"), list), errors, "analogical simplex-tree analogy pair lacks face/coface chain rows")
                _assert(int(_finite_float(pair.get("checked_simplices"), -1.0)) >= len(pair.get("simplex_rows", [])) or pair.get("simplex_rows_truncated") is True, errors, "analogical simplex-tree analogy checked count is inconsistent with rows")
                for simplex_row in pair.get("simplex_rows", [])[:12] if isinstance(pair.get("simplex_rows"), list) else []:
                    if not isinstance(simplex_row, dict):
                        errors.append("analogical simplex-tree analogy simplex row is not an object")
                        continue
                    _assert(isinstance(simplex_row.get("domain_simplex"), list), errors, "analogical simplex-tree analogy row lacks domain simplex")
                    _assert(isinstance(simplex_row.get("image_simplex"), list), errors, "analogical simplex-tree analogy row lacks image simplex")
                    _assert("preserved_in_simplex_tree" in simplex_row, errors, "analogical simplex-tree analogy row lacks preservation flag")
            if isinstance(topk_contract, dict):
                _assert(int(_finite_float(topk_contract.get("top_k_rendered"), -1.0)) == len(maps), errors, "analogical top-k rendered count does not match maps")
                _assert(int(_finite_float(topk_contract.get("qualified_model_probability_memory_count"), -1.0)) >= len(maps), errors, "analogical top-k qualified count is smaller than rendered maps")
                _assert(int(_finite_float(topk_contract.get("raw_retrieved_count"), -1.0)) >= int(_finite_float(topk_contract.get("qualified_model_probability_memory_count"), 0.0)), errors, "analogical top-k raw retrieved count is smaller than qualified count")
                _assert(topk_contract.get("query_complex_source") in allowed_sources, errors, "analogical top-k contract query source is not trajectory model-probability complex")
            _assert(all(row.get("query_complex_source") in allowed_sources for row in maps if isinstance(row, dict)), errors, "analogical maps are not using query trajectory-level model-probability complexes")
            _assert(all(row.get("codomain_complex_source") in allowed_sources for row in maps if isinstance(row, dict)), errors, "analogical maps are not using codomain trajectory-level model-probability complexes")
            _assert(all(not bool(row.get("is_identity_self_map")) for row in maps if isinstance(row, dict)), errors, "analogical maps include identity self-maps")
            _assert(all(row.get("map_source") == "model_probability_jensen_shannon_assignment" for row in maps if isinstance(row, dict)), errors, "analogical maps are not derived from model probability vectors")
            for index, row in enumerate(maps):
                if not isinstance(row, dict):
                    continue
                evidence = row.get("probability_vector_evidence", {}) if isinstance(row.get("probability_vector_evidence"), dict) else {}
                _assert(evidence.get("schema_version") == "tropicalgt.probability_vector_assignment_evidence.v1", errors, f"analogical map {index} lacks probability-vector assignment evidence")
                _assert(evidence.get("source") == "probability_filtered_complex_vertices", errors, f"analogical map {index} probability-vector evidence has wrong source")
                _assert(evidence.get("probability_vector_source") == "model_probability_vectors_on_vertices", errors, f"analogical map {index} probability-vector evidence is not model-derived")
                _assert(evidence.get("assignment_metric") == "jensen_shannon_distance_on_model_probability_vectors", errors, f"analogical map {index} probability-vector evidence has wrong assignment metric")
                _assert(str(evidence.get("assignment_solver", "")).strip() != "", errors, f"analogical map {index} probability-vector evidence lacks assignment solver")
                _assert(evidence.get("embedding_only_assignment_used") is False, errors, f"analogical map {index} probability-vector evidence uses embedding-only assignment")
                _assert(evidence.get("no_proxy_or_fallback") is True, errors, f"analogical map {index} probability-vector evidence allows proxy/fallback data")
                _assert(evidence.get("all_displayed_query_vertices_have_probability_vectors") is True, errors, f"analogical map {index} does not certify probability vectors on all displayed query vertices")
                _assert(evidence.get("all_displayed_memory_vertices_have_probability_vectors") is True, errors, f"analogical map {index} does not certify probability vectors on all displayed memory vertices")
                _assert(int(_finite_float(evidence.get("displayed_query_vertices"), -1.0)) == int(_finite_float(row.get("displayed_domain_vertices"), -2.0)), errors, f"analogical map {index} probability query vertex count mismatches displayed domain")
                _assert(int(_finite_float(evidence.get("displayed_memory_vertices"), -1.0)) == int(_finite_float(row.get("displayed_codomain_vertices"), -2.0)), errors, f"analogical map {index} probability memory vertex count mismatches displayed codomain")
                layout = row.get("layout_contract", {}) if isinstance(row.get("layout_contract"), dict) else {}
                table_rows = row.get("correspondence_table_rows", []) if isinstance(row.get("correspondence_table_rows"), list) else []
                _assert(layout.get("schema_version") == "tropicalgt.analogical_map_layout.v1", errors, f"analogical map {index} lacks analogical map layout contract")
                _assert(layout.get("default_view") == "side_by_side_query_codomain_small_multiples_plus_correspondence_table", errors, f"analogical map {index} layout contract has wrong default view")
                _assert(layout.get("query_panel") == "query trajectory probability complex", errors, f"analogical map {index} layout contract lacks query panel")
                _assert(layout.get("codomain_panel") == "retrieved memory probability complex", errors, f"analogical map {index} layout contract lacks codomain panel")
                _assert(layout.get("assignment_metric") == "jensen_shannon_distance_on_model_probability_vectors", errors, f"analogical map {index} layout contract has wrong assignment metric")
                _assert(layout.get("embedding_only_assignment_allowed") is False, errors, f"analogical map {index} layout contract allows embedding-only assignment")
                _assert(layout.get("fail_closed_when_probability_vectors_missing") is True, errors, f"analogical map {index} layout contract does not fail closed")
                _assert(layout.get("draws_pseudo_map_when_unavailable") is False, errors, f"analogical map {index} layout contract permits pseudo maps")
                _assert(layout.get("no_proxy_or_fallback") is True, errors, f"analogical map {index} layout contract lacks no-proxy flag")
                _assert(isinstance(table_rows, list) and int(_finite_float(layout.get("correspondence_table_rows"), -1.0)) == len(table_rows), errors, f"analogical map {index} correspondence-table row count mismatches layout contract")
                for table_row in table_rows[:12]:
                    if not isinstance(table_row, dict):
                        errors.append(f"analogical map {index} correspondence table contains a non-object row")
                        continue
                    _assert(table_row.get("assignment_metric") == "jensen_shannon_distance_on_model_probability_vectors", errors, f"analogical map {index} correspondence row has wrong assignment metric")
                    _assert(table_row.get("embedding_only_assignment_used") is False, errors, f"analogical map {index} correspondence row uses embedding-only assignment")
                    _assert(table_row.get("no_proxy_or_fallback") is True, errors, f"analogical map {index} correspondence row allows proxy/fallback data")
            _assert(all(isinstance(row.get("jensen_shannon_distance_summary"), dict) and _finite_float(row["jensen_shannon_distance_summary"].get("count"), 0.0) > 0 for row in maps if isinstance(row, dict)), errors, "analogical maps are missing Jensen-Shannon distance summaries")
            _assert(all(isinstance(row.get("assignment_cost_summary"), dict) and _finite_float(row["assignment_cost_summary"].get("count"), 0.0) > 0 for row in maps if isinstance(row, dict)), errors, "analogical maps are missing assignment-cost summaries")
            _assert(all(isinstance(row.get("filtration_distortion_summary"), dict) for row in maps if isinstance(row, dict)), errors, "analogical maps are missing filtration-distortion summaries")
            _assert(all(isinstance(row.get("domain_simplex_tree"), dict) and row["domain_simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in maps if isinstance(row, dict)), errors, "analogical maps are missing domain GUDHI SimplexTree provenance")
            _assert(all(isinstance(row.get("codomain_simplex_tree"), dict) and row["codomain_simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in maps if isinstance(row, dict)), errors, "analogical maps are missing codomain GUDHI SimplexTree provenance")
            pair_pages = [str(row.get("pair_page", "")).strip() for row in maps if isinstance(row, dict)]
            _assert(len(pair_pages) == len(maps) and all(pair_pages), errors, "analogical maps are missing per-rank pair_page links")
            absolute_pair_pages = [Path(pair_page) for pair_page in pair_pages if Path(pair_page).is_absolute()]
            _assert(not absolute_pair_pages, errors, f"analogical maps must use relative pair_page links, found absolute paths: {[path.name for path in absolute_pair_pages[:8]]}")
            missing_pair_files = []
            for pair_page in pair_pages:
                pair_path = Path(pair_page)
                candidates_for_page = [row_dir / pair_path, row_dir / pair_path.name]
                if not any(path.exists() for path in candidates_for_page):
                    missing_pair_files.append(pair_path.name)
            _assert(not missing_pair_files, errors, f"analogical maps reference missing pair pages: {missing_pair_files[:8]}")
            analogical_index_path = row_dir / REQUIRED_HTML["analogical_index"][0]
            if analogical_index_path.exists() and pair_pages:
                index_html = _read_text(analogical_index_path)
                link_names = {Path(ref.split("#", 1)[0].split("?", 1)[0]).name for ref in _attr_values(index_html, "href") if not _ignored_ref(ref)}
                missing_links = sorted({Path(pair_page).name for pair_page in pair_pages} - link_names)
                _assert(not missing_links, errors, f"analogical_memory_topk_index.html missing map links: {missing_links[:8]}")
            _assert(all(_finite_float(row.get("displayed_domain_vertices"), 0.0) > 0 and _finite_float(row.get("displayed_codomain_vertices"), 0.0) > 0 for row in maps if isinstance(row, dict)), errors, "analogical maps are missing displayed vertex counts")
            _assert(all("is_simplicial_on_displayed_skeleton" in row for row in maps if isinstance(row, dict)), errors, "analogical maps are missing displayed-skeleton simplicial status")
            _assert(all(isinstance(row.get("preserved_edge_pairs"), list) and isinstance(row.get("failed_edge_pairs"), list) for row in maps if isinstance(row, dict)), errors, "analogical maps are missing preserved/failed edge evidence")
            _assert(all(isinstance(row.get("preserved_edge_query_vertices"), list) for row in maps if isinstance(row, dict)), errors, "analogical maps are missing preserved-edge vertex sets")
            js_means = {_finite_float(row.get("jensen_shannon_distance_mean"), -1.0) for row in maps if isinstance(row, dict)}
            assignment_means = {_finite_float(row.get("assignment_cost_mean"), -1.0) for row in maps if isinstance(row, dict)}
            edge_rates = {_finite_float(row.get("edge_preservation_rate"), -1.0) for row in maps if isinstance(row, dict)}
            _assert(len(edge_rates) > 1 or len(js_means) > 1 or len(assignment_means) > 1 or len(maps) <= 1, errors, "analogical maps have identical preservation and probability-assignment diagnostics")
            analogical_path = row_dir / REQUIRED_HTML["analogical_map"][0]
            if not analogical_path.exists() and (row_dir / "analogical_memory_retrieval.html").exists():
                analogical_path = row_dir / "analogical_memory_retrieval.html"
            analogical_html = _read_text(analogical_path) if analogical_path.exists() else ""
            preserved_marker = "preserved 1-simplex" in analogical_html or "preserve displayed 1-simplices" in analogical_html
            _assert("vertex-only correspondences" in analogical_html and preserved_marker, errors, "analogical map HTML does not distinguish preserved simplices from vertex-only correspondences")

    steps = [row for row in manifest.get("steps", []) if isinstance(row, dict)]
    step_contract = manifest.get("contract", {}) if isinstance(manifest.get("contract"), dict) else {}
    _assert(len(steps) == len(candidates), errors, "reasoning-step complex map count does not match candidates")
    _assert(isinstance(step_contract, dict) and step_contract.get("schema_version") == "tropicalgt.reasoning_step_complex_maps.v1", errors, "reasoning-step manifest is missing contract schema")
    if isinstance(step_contract, dict):
        _assert(step_contract.get("no_proxy_or_fallback") is True, errors, "reasoning-step manifest contract missing no-proxy flag")
        _assert(step_contract.get("actual_data_only") is True, errors, "reasoning-step manifest contract is not actual-data-only")
        _assert(step_contract.get("one_page_per_model_evaluated_reasoning_step") is True, errors, "reasoning-step manifest contract does not require one page per model state")
        _assert(step_contract.get("embedding_trajectory_map_is_not_a_step_complex") is True, errors, "reasoning-step manifest allows trajectory map as step complex")
        _assert(step_contract.get("all_step_complex_fingerprints_present") is True, errors, "reasoning-step manifest contract says fingerprints are missing")
        _assert(step_contract.get("source_contract_schema_version") == "tropicalgt.reasoning_step_complex_source_contract.v1", errors, "reasoning-step manifest missing source contract schema")
        _assert(int(_finite_float(step_contract.get("rendered_source_contracts"), -1.0)) == len(steps), errors, "reasoning-step manifest source contract count mismatch")
        _assert(step_contract.get("all_steps_have_source_contracts") is True, errors, "reasoning-step manifest says source contracts are missing")
        _assert(step_contract.get("all_step_complexes_use_candidate_filtered_object_source") is True, errors, "reasoning-step manifest does not require candidate filtered object source")
        _assert(step_contract.get("all_step_complex_source_contracts_no_proxy") is True, errors, "reasoning-step manifest source contracts allow proxy data")
        _assert(step_contract.get("all_step_complex_source_contracts_safe") is True, errors, "reasoning-step manifest source contracts are unsafe")
        _assert(step_contract.get("all_step_complex_source_counts_match_summary") is True, errors, "reasoning-step manifest source counts do not match summaries")
        _assert(step_contract.get("all_step_complexes_have_vertices") is True, errors, "reasoning-step manifest source contracts lack vertices")
        _assert(int(_finite_float(step_contract.get("source_contract_unavailable_count"), -1.0)) == 0, errors, "reasoning-step manifest lists unavailable source contracts")
        _assert(step_contract.get("simplex_tree_poset_contract_schema_version") == "tropicalgt.simplex_tree_poset.v1", errors, "reasoning-step manifest missing simplex-tree poset contract schema")
        _assert(int(_finite_float(step_contract.get("rendered_simplex_tree_poset_contracts"), -1.0)) == len(steps), errors, "reasoning-step manifest rendered simplex-tree poset contract count mismatch")
        _assert(step_contract.get("all_steps_have_simplex_tree_poset_contracts") is True, errors, "reasoning-step manifest says simplex-tree poset contracts are missing")
        _assert(step_contract.get("all_step_simplex_tree_posets_no_proxy") is True, errors, "reasoning-step manifest says simplex-tree poset contracts allow proxy data")
        _assert(step_contract.get("all_step_simplex_tree_posets_use_gudhi") is True, errors, "reasoning-step manifest says simplex-tree posets are not GUDHI-backed")
        _assert(step_contract.get("all_step_simplex_tree_posets_face_coface_primary") is True, errors, "reasoning-step manifest says simplex-tree posets are not face-to-coface primary")
        _assert(step_contract.get("all_step_simplex_tree_posets_safe_to_render") is True, errors, "reasoning-step manifest says simplex-tree posets are unsafe")
        _assert(int(_finite_float(step_contract.get("simplex_tree_poset_unavailable_count"), -1.0)) == 0, errors, "reasoning-step manifest lists unavailable simplex-tree poset contracts")
        _assert(step_contract.get("slider_contract_schema_version") == "tropicalgt.reasoning_step_radius_slider_summary.v1", errors, "reasoning-step manifest missing radius slider summary schema")
        _assert(int(_finite_float(step_contract.get("rendered_slider_contracts"), -1.0)) == len(steps), errors, "reasoning-step manifest rendered slider contract count mismatch")
        _assert(step_contract.get("all_steps_have_radius_slider_contracts") is True, errors, "reasoning-step manifest says radius slider summaries are missing")
        _assert(step_contract.get("all_step_radius_sliders_start_disjoint_vertices") is True, errors, "reasoning-step manifest says radius sliders do not start as disjoint vertices")
        _assert(step_contract.get("all_step_radius_sliders_monotone") is True, errors, "reasoning-step manifest says radius sliders are not monotone")
        _assert(step_contract.get("all_step_radius_sliders_no_proxy") is True, errors, "reasoning-step manifest says radius sliders allow proxy data")
        _assert(step_contract.get("all_step_radius_sliders_safe_to_render") is True, errors, "reasoning-step manifest says radius sliders are unsafe")
        _assert(int(_finite_float(step_contract.get("radius_slider_unavailable_count"), -1.0)) == 0, errors, "reasoning-step manifest lists unavailable radius slider summaries")
        _assert(int(_finite_float(step_contract.get("unique_step_complex_fingerprint_count"), -1.0)) >= 1 if steps else True, errors, "reasoning-step manifest has no unique complex fingerprints")
        duplicate_groups = step_contract.get("duplicate_step_complex_fingerprint_groups", [])
        _assert(isinstance(duplicate_groups, list), errors, "reasoning-step manifest duplicate fingerprint groups are not a list")
        if isinstance(duplicate_groups, list):
            for group in duplicate_groups:
                if not isinstance(group, dict):
                    errors.append("reasoning-step manifest duplicate fingerprint group is not an object")
                    continue
                _assert(bool(group.get("fingerprint")), errors, "reasoning-step duplicate group lacks fingerprint")
                _assert(len(group.get("step_indices", [])) > 1 if isinstance(group.get("step_indices"), list) else False, errors, "reasoning-step duplicate group lacks repeated step indices")
                _assert(group.get("allowed_only_if_canonical_filtered_complex_payload_identical") is True, errors, "reasoning-step duplicate group lacks identical-payload caveat")
    _assert(all(isinstance(row.get("simplex_tree"), dict) and row["simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in steps), errors, "reasoning-step manifest is missing GUDHI SimplexTree provenance")
    fingerprints = [str(row.get("step_complex_fingerprint", "")) for row in steps]
    _assert(all(fingerprints), errors, "reasoning-step manifest steps are missing complex fingerprints")
    _assert(all(isinstance(row.get("step_complex_fingerprint_basis"), dict) for row in steps), errors, "reasoning-step manifest steps are missing fingerprint basis")
    if fingerprints and isinstance(step_contract, dict):
        _assert(int(_finite_float(step_contract.get("unique_step_complex_fingerprint_count"), -1.0)) == len(set(fingerprints)), errors, "reasoning-step manifest fingerprint unique count mismatch")
        expected_duplicate_groups = sum(1 for count in Counter(fingerprints).values() if count > 1)
        duplicate_groups = step_contract.get("duplicate_step_complex_fingerprint_groups", []) if isinstance(step_contract.get("duplicate_step_complex_fingerprint_groups"), list) else []
        _assert(len(duplicate_groups) == expected_duplicate_groups, errors, "reasoning-step manifest duplicate fingerprint groups do not match step fingerprints")
        if len(set(fingerprints)) == len(fingerprints):
            _assert(step_contract.get("all_step_complex_fingerprints_unique") is True, errors, "reasoning-step manifest unique-fingerprint flag is false despite unique fingerprints")
    for idx, node in enumerate(nodes):
        rid = str(node.get("record_id", ""))
        step = steps[idx] if idx < len(steps) else {}
        expected_step_href = f"reasoning_step_complex_maps/{step.get('file', '')}"
        expected_tree_href = f"reasoning_step_complex_maps/{step.get('simplex_tree_file', '')}"
        _assert(str(step.get("record_id", rid)) == rid, errors, f"reasoning-step manifest record_id mismatch at index {idx}")
        basis = step.get("step_complex_fingerprint_basis", {}) if isinstance(step.get("step_complex_fingerprint_basis"), dict) else {}
        _assert(basis.get("schema_version") == "tropicalgt.reasoning_step_complex_fingerprint_basis.v1", errors, f"reasoning-step manifest fingerprint basis schema mismatch at index {idx}")
        _assert(basis.get("source") == "gudhi_canonical_complex(filtered_simplicial_object)", errors, f"reasoning-step manifest fingerprint source mismatch at index {idx}")
        _assert(basis.get("no_record_id_or_path_in_hash") is True, errors, f"reasoning-step manifest fingerprint includes record id/path at index {idx}")
        _assert(isinstance(basis.get("simplices"), list) and bool(basis.get("simplices")), errors, f"reasoning-step manifest fingerprint basis lacks simplices at index {idx}")
        _validate_reasoning_step_source_contract(
            step.get("step_complex_source_contract"),
            step,
            errors,
            f"reasoning-step complex {idx}",
        )
        _assert(node.get("reasoning_step_index") == idx, errors, f"trajectory node {rid} has wrong reasoning_step_index")
        _assert(node.get("step_complex_href") == expected_step_href, errors, f"trajectory node {rid} has wrong step_complex_href")
        _assert(node.get("step_simplex_tree_href") == expected_tree_href, errors, f"trajectory node {rid} has wrong step_simplex_tree_href")
        _assert((row_dir / expected_step_href).exists(), errors, f"trajectory node {rid} references missing step complex page")
        _assert((row_dir / expected_tree_href).exists(), errors, f"trajectory node {rid} references missing step simplex tree page")
        expected_slider_file = str(step.get("slider_contract_file") or Path(str(step.get("file", ""))).with_suffix("").name + "_slider_contract.json")
        slider_path = _slider_contract_path(row_dir / expected_step_href)
        _assert(slider_path.name == expected_slider_file, errors, f"reasoning-step complex {idx} manifest slider contract file mismatch")
        slider_payload = _validate_radius_slider_contract(
            slider_path,
            errors,
            f"reasoning-step complex {idx}",
        )
        expected_poset_file = str(step.get("simplex_tree_poset_contract_file") or Path(str(step.get("simplex_tree_file", ""))).with_suffix("").name + "_simplex_tree_poset_contract.json")
        poset_path = _simplex_tree_poset_contract_path(row_dir / expected_tree_href)
        _assert(poset_path.name == expected_poset_file, errors, f"reasoning-step simplex tree {idx} manifest poset contract file mismatch")
        poset_payload = _validate_simplex_tree_poset_contract(
            poset_path,
            errors,
            f"reasoning-step simplex tree {idx}",
            required_available=True,
        )
        embedded_poset = step.get("simplex_tree_poset_contract") if isinstance(step.get("simplex_tree_poset_contract"), dict) else {}
        _assert(embedded_poset.get("schema_version") == poset_payload.get("schema_version"), errors, f"reasoning-step simplex tree {idx} embedded poset schema mismatch")
        _assert(embedded_poset.get("safe_to_render_simplex_tree") == poset_payload.get("safe_to_render_simplex_tree"), errors, f"reasoning-step simplex tree {idx} embedded poset safety mismatch")
        _assert(int(_finite_float(embedded_poset.get("displayed_simplex_count"), -1.0)) == int(_finite_float(poset_payload.get("displayed_simplex_count"), -2.0)), errors, f"reasoning-step simplex tree {idx} embedded poset count mismatch")
        _validate_reasoning_step_slider_summary(
            step.get("radius_slider_contract"),
            slider_payload,
            errors,
            f"reasoning-step complex {idx}",
            expected_contract_file=expected_slider_file,
            expected_html_file=str(step.get("file", "")),
        )
    for idx in [0, len(steps) // 2, len(steps) - 1] if steps else []:
        step = steps[idx]
        step_file = row_dir / "reasoning_step_complex_maps" / str(step.get("file", ""))
        _assert(step_file.exists(), errors, f"missing reasoning step page {step_file.name}")
        if step_file.exists():
            step_html = _read_text(step_file)
            _assert("Filtration radius" in step_html and "play filtration" in step_html, errors, f"{step_file.name} missing filtration controls")
        tree_file = row_dir / "reasoning_step_complex_maps" / str(step.get("simplex_tree_file", ""))
        _assert(tree_file.exists(), errors, f"missing reasoning step simplex tree page {tree_file.name}")
        if tree_file.exists():
            tree_html = _read_text(tree_file)
            _assert("simplex-tree inclusion" in tree_html or "GUDHI simplex tree" in tree_html or "GUDHI SimplexTree" in tree_html or "face-coface poset" in tree_html, errors, f"{tree_file.name} missing simplex-tree inclusion view")

    browser_index = _validate_browser_index(row_dir, errors)
    codex_browser_index = _validate_codex_browser_index(row_dir, errors)
    nll_values = np.asarray([_finite_float(row.get("nll")) for row in nodes], dtype=float) if nodes else np.asarray([])
    embedding_dist = _pairwise_euclidean(embeddings) if embeddings.ndim == 2 else np.zeros((0, 0))
    tri = embedding_dist[np.triu_indices(len(embedding_dist), 1)] if len(embedding_dist) > 1 else np.asarray([])

    return {
        "row_dir": str(row_dir),
        "ok": not errors,
        "errors": errors,
        "files": files,
        "browser_index": browser_index,
        "codex_browser_index": codex_browser_index,
        "candidates": len(candidates),
        "edges": len(edges),
        "levels": [int(v) for v in levels],
        "max_level": max_level,
        "unique_paths": len(set(paths)),
        "branching_nodes": sum(1 for count in branch_counts.values() if count > 1),
        "max_branch": max(branch_counts.values() or [0]),
        "stochastic_actions": bool(scaling.get("stochastic_actions")),
        "sampling_temperature": scaling.get("sampling_temperature"),
        "sampling_exploration": scaling.get("sampling_exploration"),
        "pca_distance_correlation": diag_corr,
        "pca_normalized_stress": diag_stress,
        "pca_explained_variance_sum3": pca_diag.get("explained_variance_ratio_sum3"),
        "nll_surface_kind": surface.get("surface_kind"),
        "nll_surface_residual": surface.get("max_point_residual"),
        "nll_min": float(np.nanmin(nll_values)) if nll_values.size else None,
        "nll_mean": float(np.nanmean(nll_values)) if nll_values.size else None,
        "nll_max": float(np.nanmax(nll_values)) if nll_values.size else None,
        "nll_std": float(np.nanstd(nll_values)) if nll_values.size else None,
        "embedding_distance_mean": float(np.nanmean(tri)) if tri.size else None,
        "embedding_distance_max": float(np.nanmax(tri)) if tri.size else None,
        "step_complex_maps": len(steps),
    }


def _looks_like_audit_row(path: Path) -> bool:
    return (path / REQUIRED_JSON["scaling_tree"]).exists() and (path / REQUIRED_JSON["trajectory_payload"]).exists()


def _candidate_row_dirs(audit_root: Path) -> list[Path]:
    audit_root = audit_root.resolve()
    if audit_root.name != "got_audit" and (audit_root / "got_audit").is_dir():
        audit_root = audit_root / "got_audit"
    if not _looks_like_audit_row(audit_root):
        sample_rows = sorted(
            path
            for path in audit_root.iterdir()
            if path.is_dir() and path.name.startswith("sample_") and _looks_like_audit_row(path)
        )
        if sample_rows:
            return sample_rows
    rows = [audit_root]
    rows.extend(sorted(path for path in audit_root.glob("example_*") if path.is_dir()))
    return rows


def _validation_metrics(audit_root: Path) -> dict[str, Any]:
    root = audit_root.resolve()
    if root.name == "got_audit":
        step_dir = root.parent
    elif (root / "got_audit").is_dir():
        step_dir = root
    else:
        step_dir = root.parent
    report = step_dir / "validation_report.json"
    if not report.exists():
        return {"available": False, "path": str(report)}
    data = _read_json(report)
    keys = [
        "bpb",
        "text_bpb",
        "graph_bpb",
        "graph_sideinfo_bpb",
        "graph_conditioned_bpb_no_side_cost",
        "nll",
        "ppl",
        "invalid_graph_rate",
        "causal_dag_ar_rate",
        "random_graph_ar_rate",
        "parameter_golf_source_rate",
        "graph_autoregressive_decoding_enabled",
    ]
    return {"available": True, "path": str(report), **{key: data.get(key) for key in keys if key in data}}


def validate_audit_root(audit_root: str | Path, *, min_rows: int = 3, min_candidates: int = 8, min_depth: int = 2) -> dict[str, Any]:
    root = Path(audit_root).resolve()
    rows = _candidate_row_dirs(root)
    row_reports = [validate_row(row, min_candidates=min_candidates, min_depth=min_depth) for row in rows[: max(min_rows, len(rows))]]
    errors: list[str] = []
    if len(row_reports) < min_rows:
        errors.append(f"only {len(row_reports)} rows available, expected at least {min_rows}")
    for idx, report in enumerate(row_reports):
        if not report["ok"]:
            errors.extend([f"row {idx} {err}" for err in report["errors"]])
    return {
        "audit_root": str(root),
        "ok": not errors,
        "errors": errors,
        "rows_checked": len(row_reports),
        "row_reports": row_reports,
        "validation_metrics": _validation_metrics(root),
        "evidence_gap_inventory": _build_evidence_gap_inventory(errors),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Interactive Audit Artifact Validation",
        "",
        f"- Audit root: `{report['audit_root']}`",
        f"- Overall status: {'PASS' if report['ok'] else 'FAIL'}",
        f"- Rows checked: `{report['rows_checked']}`",
    ]
    metrics = report.get("validation_metrics", {})
    if metrics.get("available"):
        lines.append(f"- Validation report: `{metrics.get('path')}`")
        for key, value in metrics.items():
            if key not in {"available", "path"}:
                lines.append(f"- `{key}`: `{value}`")
    inventory = report.get("evidence_gap_inventory", {}) if isinstance(report.get("evidence_gap_inventory"), dict) else {}
    categories = inventory.get("categories", []) if isinstance(inventory.get("categories"), list) else []
    if inventory.get("gap_count"):
        lines.extend(["", "## Evidence Gap Inventory"])
        lines.append(f"- Gap count: `{inventory.get('gap_count')}`")
        lines.append("- Strict validation still required: `true`")
        lines.append(f"- Policy: {inventory.get('policy')}")
        for row in categories:
            if not isinstance(row, dict):
                continue
            lines.append(f"- `{row.get('category')}`: `{row.get('count')}` - {row.get('required_action')}")
            for example in row.get("examples", [])[:2] if isinstance(row.get("examples"), list) else []:
                lines.append(f"  - example: `{example}`")
    if report.get("errors"):
        lines.extend(["", "## Errors"])
        lines.extend(f"- {err}" for err in report["errors"])
    lines.extend(["", "## Rows"])
    for idx, row in enumerate(report.get("row_reports", [])):
        lines.append(
            "- "
            + f"row `{idx}` status={'PASS' if row['ok'] else 'FAIL'} "
            + f"candidates={row['candidates']} edges={row['edges']} depth={row['max_level']} "
            + f"branches={row['branching_nodes']} max_branch={row['max_branch']} "
            + f"pca_corr={row['pca_distance_correlation']:.6f} stress={row['pca_normalized_stress']:.6f} "
            + f"nll_residual={row['nll_surface_residual']} step_maps={row['step_complex_maps']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate TropicalGT-I interactive audit artifacts.")
    parser.add_argument("--audit-root", required=True, help="Path to a got_audit directory or its step directory.")
    parser.add_argument("--min-rows", type=int, default=3)
    parser.add_argument("--min-candidates", type=int, default=8)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--json-output", default="")
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args(argv)
    report = validate_audit_root(
        args.audit_root,
        min_rows=args.min_rows,
        min_candidates=args.min_candidates,
        min_depth=args.min_depth,
    )
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown_report(report), encoding="utf-8")
    print(_markdown_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
