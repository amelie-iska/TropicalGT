#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_HTML = {
    "embedding_map": ("got_embedding_map_3d.html", ("Graph-of-thought embedding-space trajectory map", "actual graph_state PCA")),
    "trajectory_nll": ("got_trajectory_pca_3d.html", ("Graph-of-thought branching trajectory", "centered NLL")),
    "full_complex": ("got_full_trajectory_complex.html", ("Full graph-of-thought trajectory filtered simplicial complex", "play filtration", "filtration backend=")),
    "step_complex_index": ("reasoning_step_complex_maps/index.html", ("Reasoning step filtered simplicial complex maps",)),
    "tropical_support": ("tropical_support_heatmap.html", ("Tropical", "support")),
    "graphcg": ("graphcg_direction_cosines.html", ("GraphCG", "full-rank direction audit")),
    "analogical_index": ("analogical_memory_topk_index.html", ("Analogical top-k retrieval",)),
    "analogical_map": ("analogical_memory_map_02.html", ("Analogical", "simplicial", "binary filtered-complex map")),
    "trajectory_barcode": ("trajectory_persistence/persistence_barcode.html", ("Trajectory", "barcode")),
    "trajectory_betti": ("trajectory_persistence/persistence_module_betti.html", ("Trajectory", "Betti", "2D matrix", "decorative 3D")),
    "trajectory_representations": ("trajectory_persistence/persistence_representations.html", ("Trajectory", "GUDHI persistence vectorization", "Fast train", "eval features")),
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


def _validate_file_set(row_dir: Path, errors: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for key, rel in REQUIRED_JSON.items():
        path = row_dir / rel
        _assert(path.exists(), errors, f"missing json {rel}")
        files[key] = str(path)
    for key, (rel, needles) in REQUIRED_HTML.items():
        path = row_dir / rel
        _assert(path.exists(), errors, f"missing html {rel}")
        files[key] = str(path)
        if path.exists():
            html = _read_text(path)
            for needle in needles:
                _assert(needle in html, errors, f"{rel} missing marker {needle!r}")
            if key not in {"step_complex_index", "analogical_index"}:
                _assert(_html_has_plotly(html), errors, f"{rel} does not look like a Plotly/interactive page")
            if key == "full_complex":
                _assert("model input" in html and "model output" in html, errors, f"{rel} does not expose model input/output in hover payload")
            if key == "analogical_map":
                _assert("trajectory-complex map" in html, errors, f"{rel} is not mapping trajectory complexes")
                _assert("slider filters domain" in html and "sliders" in html, errors, f"{rel} is missing Plotly filtration slider for the simplicial map")
            if key == "tropical_support":
                support_audit = "observed supports only" in html and "top-support collapse rate" in html
                collapse_audit = "active-support collapse diagnostic" in html and "Collapse metrics" in html
                _assert(support_audit or collapse_audit, errors, f"{rel} is not the interpretable support audit view")
    return files


def _validate_browser_index(row_dir: Path, errors: list[str]) -> dict[str, Any]:
    index = row_dir / "browser_index.html"
    if not index.exists():
        return {"available": False}
    html = _read_text(index)
    refs = re.findall(r"""(?:href|src)=["']([^"']+)["']""", html)
    broken = []
    for ref in refs:
        if ref.startswith(("#", "http://", "https://", "file://", "data:")):
            continue
        target = (row_dir / ref).resolve()
        if not target.exists():
            broken.append(ref)
    _assert(not broken, errors, f"browser_index.html has broken relative refs: {broken[:8]}")
    return {"available": True, "relative_refs": len([r for r in refs if not r.startswith(("#", "http://", "https://", "file://", "data:"))]), "broken_refs": broken}


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
    z_axis = surface.get("z_axis")
    _assert(
        z_axis in {"centered_scaled_nll", "projected_nll_fitness_energy"},
        errors,
        "NLL trajectory surface is not rendered with an audited NLL/fitness z-axis",
    )
    _assert(_finite_float(surface.get("raw_nll_range"), -1.0) >= 0.0, errors, "NLL surface payload is missing raw_nll_range")
    _assert(surface.get("exact_anchor_layer") is True, errors, "NLL surface is missing exact anchor layer metadata")
    surface_kind = surface.get("surface_kind")
    _assert(
        surface_kind in {"sample_supported_local_idw_surface", "exact_delaunay_nll_mesh"},
        errors,
        "NLL surface is neither a sample-supported local field nor an exact point-anchored mesh",
    )
    if surface_kind == "exact_delaunay_nll_mesh":
        footprint = surface.get("support_footprint_layer", {})
        surrogate = surface.get("surrogate_landscape_layer", {})
        has_footprint = isinstance(footprint, dict) and footprint.get("available") is True
        has_surrogate = isinstance(surrogate, dict) and surrogate.get("available") is True
        _assert(has_footprint or has_surrogate, errors, "Exact NLL mesh is missing an audited support/landscape layer")
        if has_footprint:
            _assert(
                footprint.get("surface_kind") == "projected_local_support_footprint",
                errors,
                "Exact NLL mesh footprint is not the projected local support layer",
            )
        if has_surrogate:
            _assert(
                surrogate.get("surface_kind") == "smooth_projected_nll_fitness_landscape",
                errors,
                "NLL surrogate layer is not labeled as the smooth projected NLL/fitness landscape",
            )
            _assert(surrogate.get("touches_points") is True, errors, "NLL surrogate layer is not point-anchored")
            _assert(
                _finite_float(surrogate.get("max_point_residual"), 999.0) <= nll_residual_tol,
                errors,
                "NLL surrogate anchor residual exceeds tolerance",
            )
            _assert("provenance" in surrogate, errors, "NLL surrogate layer is missing provenance")
    _assert(_finite_float(surface.get("support_radius"), -1.0) > 0.0, errors, "NLL surface payload is missing a positive support_radius")

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

    _assert(graphcg_payload.get("available") is True, errors, "GraphCG payload is unavailable")
    matrix_shape = graphcg_payload.get("matrix_shape", [])
    _assert(isinstance(matrix_shape, list) and len(matrix_shape) == 2 and int(matrix_shape[0]) >= len(candidates), errors, "GraphCG payload has invalid matrix shape")
    _assert(_finite_float(graphcg_payload.get("active_rank_nonzero_mean_abs"), 0.0) > 0, errors, "GraphCG payload reports no active directions")
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
    _assert(int(summary.get("num_vertices", 0) or 0) >= len(candidates), errors, "full trajectory complex has fewer vertices than candidates")
    _assert(int(summary.get("num_edges", 0) or 0) >= len(edges), errors, "full trajectory complex has fewer edges than trajectory")
    _assert(sum(1 for row in full_vertices if row.get("embedding")) == len(full_vertices), errors, "full trajectory complex vertices do not all carry embeddings")
    _assert(sum(1 for row in full_vertices if row.get("input_text") or row.get("decoded_argmax")) == len(full_vertices), errors, "full trajectory complex vertices do not all carry model I/O")

    maps_path = row_dir / "analogical_simplicial_maps.json"
    if maps_path.exists():
        maps = _read_json(maps_path).get("maps", [])
        _assert(bool(maps), errors, "analogical_simplicial_maps.json contains no maps")
        _assert(all(row.get("codomain_complex_source") == "trajectory_filtered_simplicial_object" for row in maps if isinstance(row, dict)), errors, "analogical maps are not using trajectory-level memory complexes")
        _assert(all(isinstance(row.get("domain_simplex_tree"), dict) and row["domain_simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in maps if isinstance(row, dict)), errors, "analogical maps are missing domain GUDHI SimplexTree provenance")
        _assert(all(isinstance(row.get("codomain_simplex_tree"), dict) and row["codomain_simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in maps if isinstance(row, dict)), errors, "analogical maps are missing codomain GUDHI SimplexTree provenance")
        _assert(all(_finite_float(row.get("displayed_domain_vertices"), 0.0) > 0 and _finite_float(row.get("displayed_codomain_vertices"), 0.0) > 0 for row in maps if isinstance(row, dict)), errors, "analogical maps are missing displayed vertex counts")
        edge_rates = {_finite_float(row.get("edge_preservation_rate"), -1.0) for row in maps if isinstance(row, dict)}
        _assert(len(edge_rates) > 1 or len(maps) <= 1, errors, "analogical map edge-preservation rates are all identical")

    steps = [row for row in manifest.get("steps", []) if isinstance(row, dict)]
    _assert(len(steps) == len(candidates), errors, "reasoning-step complex map count does not match candidates")
    _assert(all(isinstance(row.get("simplex_tree"), dict) and row["simplex_tree"].get("backend") == "gudhi.SimplexTree" for row in steps), errors, "reasoning-step manifest is missing GUDHI SimplexTree provenance")
    for idx in [0, len(steps) // 2, len(steps) - 1] if steps else []:
        step = steps[idx]
        step_file = row_dir / "reasoning_step_complex_maps" / str(step.get("file", ""))
        _assert(step_file.exists(), errors, f"missing reasoning step page {step_file.name}")
        if step_file.exists():
            step_html = _read_text(step_file)
            _assert("Filtration radius" in step_html and "play filtration" in step_html, errors, f"{step_file.name} missing filtration controls")

    browser_index = _validate_browser_index(row_dir, errors)
    nll_values = np.asarray([_finite_float(row.get("nll")) for row in nodes], dtype=float) if nodes else np.asarray([])
    embedding_dist = _pairwise_euclidean(embeddings) if embeddings.ndim == 2 else np.zeros((0, 0))
    tri = embedding_dist[np.triu_indices(len(embedding_dist), 1)] if len(embedding_dist) > 1 else np.asarray([])

    return {
        "row_dir": str(row_dir),
        "ok": not errors,
        "errors": errors,
        "files": files,
        "browser_index": browser_index,
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


def _candidate_row_dirs(audit_root: Path) -> list[Path]:
    audit_root = audit_root.resolve()
    if audit_root.name != "got_audit" and (audit_root / "got_audit").is_dir():
        audit_root = audit_root / "got_audit"
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
