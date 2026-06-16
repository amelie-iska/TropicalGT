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


def _slider_contract_path(html_path: Path) -> Path:
    return html_path.with_name(f"{html_path.stem}_slider_contract.json")


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

    support_payload = _read_json(row_dir / REQUIRED_JSON["tropical_support_payload"]) if (row_dir / REQUIRED_JSON["tropical_support_payload"]).exists() else {}
    graphcg_payload = _read_json(row_dir / REQUIRED_JSON["graphcg_payload"]) if (row_dir / REQUIRED_JSON["graphcg_payload"]).exists() else {}
    analogical_simplex_tree_payload = _read_json(row_dir / REQUIRED_JSON["analogical_simplex_tree_analogy"]) if (row_dir / REQUIRED_JSON["analogical_simplex_tree_analogy"]).exists() else {}
    bifiltration_payload = _read_json(row_dir / REQUIRED_JSON["trajectory_bifiltration_payload"]) if (row_dir / REQUIRED_JSON["trajectory_bifiltration_payload"]).exists() else {}
    bifiltration_visual_path = row_dir / "trajectory_persistence" / "two_parameter_bifiltration.json"
    bifiltration_visual_payload = _read_json(bifiltration_visual_path) if bifiltration_visual_path.exists() else {}

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
    if bifiltration_visual_payload:
        _assert(bifiltration_visual_payload.get("schema_version") == "tropicalgt.two_parameter_bifiltration_visual.v1", errors, "trajectory bifiltration visual payload has wrong schema")
        _assert(bifiltration_visual_payload.get("primary_view") == "miller_sturmfels_bivariate_staircase", errors, "trajectory bifiltration primary view is not the Miller-Sturmfels staircase")
        _assert(bifiltration_visual_payload.get("rank_surface_primary") is False, errors, "trajectory bifiltration marks rank surfaces as primary")
        axes = bifiltration_visual_payload.get("axes", {}) if isinstance(bifiltration_visual_payload.get("axes"), dict) else {}
        _assert(axes.get("horizontal") == "x_radius" and axes.get("vertical") == "x_level", errors, "trajectory bifiltration visual axes are not x_radius horizontal / x_level vertical")
        _assert("rho_x_radius" in axes.get("coordinate_one_dimensional_cones", []) and "rho_x_level" in axes.get("coordinate_one_dimensional_cones", []), errors, "trajectory bifiltration visual payload lacks coordinate one dimensional cone records")
        _assert(bifiltration_visual_payload.get("actual_data_only") is True, errors, "trajectory bifiltration visual payload does not assert actual-data-only rendering")
        _assert(bifiltration_visual_payload.get("no_proxy_resolution_claim") is True, errors, "trajectory bifiltration visual payload allows proxy resolution claims")
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
            _assert("support_frequency_mean_margin" in panel_roles, errors, "tropical support observed-support layout is missing support-frequency panel")
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
            required = {"all_direction_heatmap", "full_rank_activity_spectrum", "candidate_activity_by_observed_got_state", "direction_signed_bias"}
            _assert(required.issubset(set(readability.get("required_panels", []))), errors, "GraphCG readability contract is missing required panels")
            _assert(readability.get("exact_direction_ids_preserved_in_hover_and_payload") is True, errors, "GraphCG readability contract does not preserve exact direction ids")
            _assert(readability.get("candidate_path_action_text_preserved_in_hover_and_payload") is True, errors, "GraphCG readability contract does not preserve candidate path/action text")
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
    _validate_radius_slider_contract(
        _slider_contract_path(row_dir / REQUIRED_HTML["full_complex"][0]),
        errors,
        "full trajectory complex",
    )
    if isinstance(prob_obj, dict) and prob_obj.get("available") is not False:
        _validate_radius_slider_contract(
            _slider_contract_path(row_dir / REQUIRED_HTML["probability_complex"][0]),
            errors,
            "probability trajectory complex",
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
        _assert(node.get("reasoning_step_index") == idx, errors, f"trajectory node {rid} has wrong reasoning_step_index")
        _assert(node.get("step_complex_href") == expected_step_href, errors, f"trajectory node {rid} has wrong step_complex_href")
        _assert(node.get("step_simplex_tree_href") == expected_tree_href, errors, f"trajectory node {rid} has wrong step_simplex_tree_href")
        _assert((row_dir / expected_step_href).exists(), errors, f"trajectory node {rid} references missing step complex page")
        _assert((row_dir / expected_tree_href).exists(), errors, f"trajectory node {rid} references missing step simplex tree page")
        _validate_radius_slider_contract(
            _slider_contract_path(row_dir / expected_step_href),
            errors,
            f"reasoning-step complex {idx}",
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
