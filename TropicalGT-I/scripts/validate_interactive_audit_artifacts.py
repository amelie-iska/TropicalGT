#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    "trajectory_nll": ("got_trajectory_pca_3d.html", ("Graph-of-thought branching trajectory", "model-evaluated NLL landscape")),
    "full_complex": ("got_full_trajectory_complex.html", ("Full graph-of-thought trajectory filtered simplicial complex", "play filtration", "filtration backend=")),
    "full_simplex_tree": ("got_full_trajectory_simplex_tree_3d.html", ("Full graph-of-thought trajectory GUDHI simplex tree", "simplex-tree inclusion")),
    "probability_complex": ("got_full_trajectory_complex_jensen_shannon.html", ("probability filtered simplicial complex", "Jensen-Shannon")),
    "probability_simplex_tree": ("got_full_trajectory_simplex_tree_3d_jensen_shannon.html", ("probability", "SimplexTree", "Jensen-Shannon")),
    "step_complex_index": ("reasoning_step_complex_maps/index.html", ("Reasoning step filtered simplicial complex maps",)),
    "tropical_support": ("tropical_support_heatmap.html", ("Tropical", "support")),
    "graphcg": ("graphcg_direction_cosines.html", ("GraphCG", "full-rank direction audit")),
    "analogical_index": ("analogical_memory_topk_index.html", ("Analogical top-k retrieval",)),
    "analogical_map": ("analogical_memory_map_02.html", ("Analogical", "simplicial", "binary filtered-complex map")),
    "trajectory_barcode": ("trajectory_persistence/persistence_barcode.html", ("Trajectory", "barcode")),
    "trajectory_betti": ("trajectory_persistence/persistence_module_betti.html", ("Trajectory", "Betti", "2D matrix", "decorative 3D")),
    "trajectory_representations": ("trajectory_persistence/persistence_representations.html", ("Trajectory", "GUDHI persistence vectorization", "Fast train", "eval features")),
    "trajectory_landscapes": ("trajectory_persistence/persistence_landscapes.html", ("Trajectory", "Actual GUDHI persistence landscape functions", "lambda_1(t)", "not norm-only summaries")),
}

REQUIRED_JSON = {
    "scaling_tree": "inference_scaling_tree.json",
    "trajectory_payload": "got_trajectory_payloads.json",
    "embedding_payload": "got_embedding_map_payloads.json",
    "full_complex_payload": "got_full_trajectory_complex_payload.json",
    "step_manifest": "reasoning_step_complex_maps/manifest.json",
    "inference_audit": "inference_audit.json",
    "tropical_support_payload": "tropical_support_payload.json",
    "graphcg_payload": "graphcg_direction_cosines_payload.json",
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
                _assert("trajectory-complex map" in html, errors, f"{rel} is not mapping trajectory complexes")
                _assert("slider filters domain" in html and "sliders" in html, errors, f"{rel} is missing Plotly filtration slider for the simplicial map")
            if key == "tropical_support":
                support_audit = "observed supports only" in html and "top-support collapse rate" in html
                collapse_audit = "active-support collapse diagnostic" in html and "Collapse metrics" in html
                _assert(support_audit or collapse_audit, errors, f"{rel} is not the interpretable support audit view")
            if key == "graphcg":
                _assert("Readable top-direction heatmap" in html, errors, f"{rel} does not use the readable GraphCG top-direction audit layout")
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
    embedding_payload = _read_json(row_dir / REQUIRED_JSON["embedding_payload"]) if (row_dir / REQUIRED_JSON["embedding_payload"]).exists() else {}
    full_complex_payload = _read_json(row_dir / REQUIRED_JSON["full_complex_payload"]) if (row_dir / REQUIRED_JSON["full_complex_payload"]).exists() else {}
    manifest = _read_json(row_dir / REQUIRED_JSON["step_manifest"]) if (row_dir / REQUIRED_JSON["step_manifest"]).exists() else {}
    support_payload = _read_json(row_dir / REQUIRED_JSON["tropical_support_payload"]) if (row_dir / REQUIRED_JSON["tropical_support_payload"]).exists() else {}
    graphcg_payload = _read_json(row_dir / REQUIRED_JSON["graphcg_payload"]) if (row_dir / REQUIRED_JSON["graphcg_payload"]).exists() else {}

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
        str(surface.get("surface_contact_contract", "")).startswith("every rendered GoT state marker"),
        errors,
        "NLL surface is missing the trajectory point surface-contact contract",
    )
    _assert(
        _finite_float(surface.get("trajectory_point_surface_residual_max"), 999.0) <= nll_residual_tol,
        errors,
        "NLL trajectory points are not guaranteed to touch the displayed surface",
    )
    projected_by_record = surface.get("surface_projected_z_by_record_id", {})
    _assert(isinstance(projected_by_record, dict) and len(projected_by_record) >= len(nodes), errors, "NLL surface is missing per-record projected z values")
    for node in nodes:
        rid = str(node.get("record_id", ""))
        plot = node.get("plot", {}) if isinstance(node.get("plot"), dict) else {}
        _assert(plot.get("touches_nll_surface") is True, errors, f"trajectory node {rid} is not marked as touching the NLL surface")
        _assert(math.isfinite(_finite_float(plot.get("z_surface"))), errors, f"trajectory node {rid} is missing finite z_surface")
        _assert(math.isfinite(_finite_float(plot.get("raw_centered_scaled_nll"))), errors, f"trajectory node {rid} is missing raw centered/scaled NLL z")
        if isinstance(projected_by_record, dict) and rid in projected_by_record:
            _assert(
                abs(_finite_float(plot.get("z_surface")) - _finite_float(projected_by_record.get(rid))) <= nll_residual_tol,
                errors,
                f"trajectory node {rid} z_surface does not match the displayed NLL surface projection",
            )
    z_axis = surface.get("z_axis")
    _assert(
        z_axis in {"centered_scaled_nll", "projected_nll_fitness_energy"},
        errors,
        "NLL trajectory surface is not rendered with an audited NLL/fitness z-axis",
    )
    _assert(_finite_float(surface.get("raw_nll_range"), -1.0) >= 0.0, errors, "NLL surface payload is missing raw_nll_range")
    _assert(surface.get("exact_anchor_layer") is True, errors, "NLL surface is missing exact anchor layer metadata")
    _assert(surface.get("actual_landscape_layer") is True, errors, "NLL surface is missing actual sampled landscape metadata")
    surface_kind = surface.get("surface_kind")
    _assert(
        surface_kind in {"sample_supported_local_idw_surface", "exact_delaunay_nll_mesh", "sparse_exact_triangular_nll_mesh"},
        errors,
        "NLL surface is neither a sample-supported local field nor an exact point-anchored mesh",
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

    support_metrics = support_payload.get("metrics", {}) if isinstance(support_payload, dict) else {}
    _assert(support_metrics.get("available") is True, errors, "tropical support payload is unavailable")
    _assert(_finite_float(support_metrics.get("token_count"), 0.0) > 0, errors, "tropical support payload has no tokens")
    _assert(_finite_float(support_metrics.get("unique_support_count"), 0.0) >= 1, errors, "tropical support payload has no observed supports")
    _assert("interpretation" in support_metrics, errors, "tropical support payload is missing collapse interpretation")
    _assert(isinstance(support_metrics.get("margin_summary"), dict), errors, "tropical support payload is missing margin summary")
    flow_edges = support_payload.get("support_flow_edges", []) if isinstance(support_payload, dict) else []
    token_count = int(_finite_float(support_metrics.get("token_count"), 0.0))
    _assert(isinstance(flow_edges, list) and len(flow_edges) >= token_count, errors, "tropical support payload is missing query-to-support flow edges")
    for edge in flow_edges if isinstance(flow_edges, list) else []:
        if not isinstance(edge, dict):
            errors.append("tropical support payload contains non-object support-flow edge")
            continue
        query_idx = int(_finite_float(edge.get("query_index"), -1.0))
        support_idx = int(_finite_float(edge.get("support_index"), -1.0))
        _assert(0 <= query_idx < token_count, errors, "tropical support flow has out-of-range query_index")
        _assert(0 <= support_idx < token_count, errors, "tropical support flow has out-of-range support_index")

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

    _assert(len(embedded_nodes) == len(nodes), errors, "embedding map payload node count differs from trajectory payload")
    _assert(embedding_payload.get("coordinate_source", "").startswith("PCA of model graph_state embeddings"), errors, "embedding map payload has wrong coordinate source")
    _assert(len(embedding_objects) == len(embedded_nodes), errors, "embedding map payload does not include one filtered simplicial object per node")
    _assert(
        any(isinstance(obj.get("simplices"), list) and obj.get("simplices") for obj in embedding_objects),
        errors,
        "embedding map filtered simplicial objects are empty",
    )
    embedding_html = _read_text(row_dir / REQUIRED_HTML["embedding_map"][0]) if (row_dir / REQUIRED_HTML["embedding_map"][0]).exists() else ""
    _assert("simplicial-object-panel" in embedding_html and "hover-simplicial-card" in embedding_html, errors, "embedding map does not expose filtered-complex hover panel")

    full_obj = full_complex_payload.get("filtered_simplicial_object", {})
    summary = full_obj.get("summary", {}) if isinstance(full_obj, dict) else {}
    simplex_tree = full_obj.get("simplex_tree", {}) if isinstance(full_obj, dict) and isinstance(full_obj.get("simplex_tree"), dict) else {}
    full_vertices = [row for row in full_obj.get("simplices", []) if isinstance(row, dict) and int(row.get("dimension", -1)) == 0] if isinstance(full_obj, dict) else []
    _assert(simplex_tree.get("backend") == "gudhi.SimplexTree", errors, "full trajectory complex payload is missing GUDHI SimplexTree provenance")
    prob_obj = full_complex_payload.get("probability_filtered_simplicial_object", {}) if isinstance(full_complex_payload, dict) else {}
    if isinstance(prob_obj, dict):
        if prob_obj.get("available") is False:
            _assert(prob_obj.get("reason") in {"missing_model_probability_vectors", "unavailable_no_embedding_or_probability_radius_edges"}, errors, "probability complex unavailable for an unrecognized reason")
        else:
            prob_summary = prob_obj.get("summary", {}) if isinstance(prob_obj.get("summary"), dict) else {}
            _assert(prob_summary.get("filtration_model") == "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton", errors, "probability complex is not a model-probability Jensen-Shannon filtration")
            _assert(int(prob_summary.get("num_edges", 0) or 0) > 0, errors, "probability complex has no Jensen-Shannon radius edges")
            prob_tree = prob_obj.get("simplex_tree", {}) if isinstance(prob_obj.get("simplex_tree"), dict) else {}
            _assert(prob_tree.get("backend") == "gudhi.SimplexTree", errors, "probability complex is missing GUDHI SimplexTree provenance")
    _assert(int(summary.get("num_vertices", 0) or 0) >= len(candidates), errors, "full trajectory complex has fewer vertices than candidates")
    _assert(int(summary.get("num_edges", 0) or 0) >= len(edges), errors, "full trajectory complex has fewer edges than trajectory")
    _assert(sum(1 for row in full_vertices if row.get("embedding")) == len(full_vertices), errors, "full trajectory complex vertices do not all carry embeddings")
    _assert(sum(1 for row in full_vertices if row.get("input_text") or row.get("decoded_argmax")) == len(full_vertices), errors, "full trajectory complex vertices do not all carry model I/O")

    maps_path = row_dir / "analogical_simplicial_maps.json"
    if maps_path.exists():
        map_payload = _read_json(maps_path)
        maps = map_payload.get("maps", [])
        if map_payload.get("available") is False:
            _assert(map_payload.get("reason") in {"missing_model_probability_query_complex", "missing_model_probability_codomain_complex", "no_non_self_model_memory"}, errors, "analogical maps are unavailable for an unrecognized reason")
        else:
            allowed_sources = {"trajectory_probability_filtered_simplicial_object"}
            _assert(bool(maps), errors, "analogical_simplicial_maps.json contains no maps")
            _assert(all(row.get("query_complex_source") in allowed_sources for row in maps if isinstance(row, dict)), errors, "analogical maps are not using query trajectory-level model-probability complexes")
            _assert(all(row.get("codomain_complex_source") in allowed_sources for row in maps if isinstance(row, dict)), errors, "analogical maps are not using codomain trajectory-level model-probability complexes")
            _assert(all(not bool(row.get("is_identity_self_map")) for row in maps if isinstance(row, dict)), errors, "analogical maps include identity self-maps")
            _assert(all(row.get("map_source") == "model_probability_jensen_shannon_assignment" for row in maps if isinstance(row, dict)), errors, "analogical maps are not derived from model probability vectors")
            _assert(all(isinstance(row.get("jensen_shannon_distance_summary"), dict) and _finite_float(row["jensen_shannon_distance_summary"].get("count"), 0.0) > 0 for row in maps if isinstance(row, dict)), errors, "analogical maps are missing Jensen-Shannon distance summaries")
            _assert(all(isinstance(row.get("assignment_cost_summary"), dict) and _finite_float(row["assignment_cost_summary"].get("count"), 0.0) > 0 for row in maps if isinstance(row, dict)), errors, "analogical maps are missing assignment-cost summaries")
            _assert(all(isinstance(row.get("filtration_distortion_summary"), dict) for row in maps if isinstance(row, dict)), errors, "analogical maps are missing filtration-distortion summaries")
            _assert(all(isinstance(row.get("domain_simplex_tree"), dict) and row["domain_simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in maps if isinstance(row, dict)), errors, "analogical maps are missing domain GUDHI SimplexTree provenance")
            _assert(all(isinstance(row.get("codomain_simplex_tree"), dict) and row["codomain_simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in maps if isinstance(row, dict)), errors, "analogical maps are missing codomain GUDHI SimplexTree provenance")
            pair_pages = [str(row.get("pair_page", "")).strip() for row in maps if isinstance(row, dict)]
            _assert(len(pair_pages) == len(maps) and all(pair_pages), errors, "analogical maps are missing per-rank pair_page links")
            missing_pair_files = []
            for pair_page in pair_pages:
                pair_path = Path(pair_page)
                candidates_for_page = [pair_path if pair_path.is_absolute() else row_dir / pair_path, row_dir / pair_path.name]
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
            preserved_marker = "preserved 1-simplex map" in analogical_html or "preserve displayed 1-simplices" in analogical_html
            _assert("vertex-only correspondences" in analogical_html and preserved_marker, errors, "analogical map HTML does not distinguish preserved simplices from vertex-only correspondences")

    steps = [row for row in manifest.get("steps", []) if isinstance(row, dict)]
    _assert(len(steps) == len(candidates), errors, "reasoning-step complex map count does not match candidates")
    _assert(all(isinstance(row.get("simplex_tree"), dict) and row["simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in steps), errors, "reasoning-step manifest is missing GUDHI SimplexTree provenance")
    for idx, node in enumerate(nodes):
        rid = str(node.get("record_id", ""))
        step = steps[idx] if idx < len(steps) else {}
        expected_step_href = f"reasoning_step_complex_maps/{step.get('file', '')}"
        expected_tree_href = f"reasoning_step_complex_maps/{step.get('simplex_tree_file', '')}"
        _assert(str(step.get("record_id", rid)) == rid, errors, f"reasoning-step manifest record_id mismatch at index {idx}")
        _assert(node.get("reasoning_step_index") == idx, errors, f"trajectory node {rid} has wrong reasoning_step_index")
        _assert(node.get("step_complex_href") == expected_step_href, errors, f"trajectory node {rid} has wrong step_complex_href")
        _assert(node.get("step_simplex_tree_href") == expected_tree_href, errors, f"trajectory node {rid} has wrong step_simplex_tree_href")
        _assert((row_dir / expected_step_href).exists(), errors, f"trajectory node {rid} references missing step complex page")
        _assert((row_dir / expected_tree_href).exists(), errors, f"trajectory node {rid} references missing step simplex tree page")
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
            _assert("simplex-tree inclusion" in tree_html or "GUDHI simplex tree" in tree_html, errors, f"{tree_file.name} missing simplex-tree inclusion view")

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
