#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tropicalgt.cas_free_resolution import _hydrate_cached_result_contracts, _unavailable_resolution_diagnostic  # noqa: E402
from tropicalgt.visualization import (  # noqa: E402
    _write_inference_dashboard,
    _write_reasoning_step_complex_maps,
    write_analogical_memory_visualization,
    write_got_trajectory_visualization,
    write_graphcg_trajectory_visualization,
    write_persistence_visualizations,
    write_toric_embedding_sidecar,
    write_tropical_fan_diagnostics,
    write_tropical_support_heatmap,
    write_two_parameter_bifiltration_visualization,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _dashboard_artifact_paths(root: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "inference_audit.html":
            continue
        if path.suffix.lower() not in {".html", ".json"}:
            continue
        key = path.relative_to(root).as_posix().replace("/", "_").replace(".", "_")
        paths[key] = str(path)
    return paths





def _tropical_support_contracts_need_backfill(root: Path) -> bool:
    payload_path = root / "tropical_support_payload.json"
    if not payload_path.exists():
        return False
    payload = _read_json(payload_path)
    if not payload:
        return False
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    render_contract = payload.get("tropical_support_render_contract", {}) if isinstance(payload.get("tropical_support_render_contract"), dict) else {}
    readability = payload.get("tropical_support_readability_contract", {}) if isinstance(payload.get("tropical_support_readability_contract"), dict) else {}
    if render_contract.get("schema_version") != "tropicalgt.tropical_support_render.v1":
        return True
    if render_contract.get("no_proxy_or_fallback") is not True:
        return True
    if render_contract.get("support_columns_policy") != "observed_valid_active_support_indices_only":
        return True
    if render_contract.get("assignment_matrix_binary") is not True:
        return True
    if not str(render_contract.get("assignment_matrix_semantics", "")).startswith("binary argmax support-selection mask"):
        return True
    if render_contract.get("normal_fan_wall_crossing_certified") is not False:
        return True
    if render_contract.get("wall_margin_metric_scope") != "margin_threshold_audit_not_certified_normal_fan_wall_crossing":
        return True
    if int(render_contract.get("token_count", -1)) != int(metrics.get("token_count", -2)):
        return True
    if int(render_contract.get("invalid_support_count", -1)) != int(metrics.get("invalid_support_count", -2)):
        return True
    if readability.get("schema_version") != "tropicalgt.tropical_support_readability.v1":
        return True
    panel_roles = set(readability.get("panel_roles", [])) if isinstance(readability.get("panel_roles"), list) else set()
    required_roles = set(readability.get("required_panel_roles", [])) if isinstance(readability.get("required_panel_roles"), list) else set()
    if readability.get("no_proxy_or_fallback") is not True:
        return True
    if readability.get("panels_are_separate") is not True:
        return True
    if not required_roles or not required_roles.issubset(panel_roles):
        return True
    for key in (
        "assignment_and_margin_panels_separated",
        "support_strip_split_from_margin_profile",
        "model_probability_summaries_separate_from_assignment_matrix",
        "compact_tick_labels",
        "full_token_text_preserved_in_hover_and_payload",
        "exact_token_indices_preserved_in_payload",
        "group_summaries_from_trace_fields",
        "invalid_active_support_indices_not_fabricated",
    ):
        if readability.get(key) is not True:
            return True
    if readability.get("normal_fan_wall_crossing_certified") is not False:
        return True
    html_path = root / "tropical_support_heatmap.html"
    if not html_path.exists():
        return True
    try:
        html = html_path.read_text(encoding="utf-8")
    except Exception:
        return True
    required_html_markers = (
        "Plotly.newPlot",
        "plotly.min.js",
        "tropical_support_render_contract",
        "tropical_support_readability_contract",
        "No support-token proxies",
    )
    return any(marker not in html for marker in required_html_markers)



def _graphcg_contracts_need_backfill(root: Path) -> bool:
    payload_path = root / "graphcg_direction_cosines_payload.json"
    html_path = root / "graphcg_direction_cosines.html"
    if not (payload_path.exists() and html_path.exists()):
        return True
    payload = _read_json(payload_path)
    if payload.get("available") is not True:
        return False
    matrix_shape = payload.get("matrix_shape") if isinstance(payload.get("matrix_shape"), list) else []
    if len(matrix_shape) != 2:
        return True
    try:
        direction_count = int(matrix_shape[1])
    except (TypeError, ValueError):
        return True
    if direction_count <= 0:
        return True
    readability = payload.get("graphcg_readability_contract") if isinstance(payload.get("graphcg_readability_contract"), dict) else {}
    if readability.get("schema_version") != "tropicalgt.graphcg_direction_readability.v1":
        return True
    if readability.get("all_model_directions_rendered") is not True:
        return True
    if readability.get("directions_sampled_for_heatmap") is not False:
        return True
    if readability.get("panels_are_separate") is not True:
        return True
    required_panels = {
        "all_direction_heatmap",
        "top_active_direction_panel",
        "full_rank_activity_spectrum",
        "candidate_activity_by_observed_got_state",
        "direction_signed_bias",
    }
    panel_names = set(payload.get("panel_names", [])) if isinstance(payload.get("panel_names"), list) else set()
    if not required_panels.issubset(set(readability.get("required_panels", [])) if isinstance(readability.get("required_panels"), list) else set()):
        return True
    if not required_panels.issubset(panel_names):
        return True
    evidence = payload.get("graphcg_direction_evidence_contract") if isinstance(payload.get("graphcg_direction_evidence_contract"), dict) else {}
    if evidence.get("schema_version") != "tropicalgt.graphcg_direction_evidence.v1":
        return True
    if evidence.get("source") != "candidate.graphcg_projection.all_direction_cosines":
        return True
    if evidence.get("no_proxy_or_fallback") is not True:
        return True
    if evidence.get("safe_to_render_full_rank_direction_evidence") is not True:
        return True
    if int(evidence.get("direction_count", -1) or -1) != direction_count:
        return True
    direction_rows = payload.get("direction_rows") if isinstance(payload.get("direction_rows"), list) else []
    if len(direction_rows) != direction_count:
        return True
    seen = set()
    for row in direction_rows:
        if not isinstance(row, dict):
            return True
        try:
            direction_id = int(row.get("direction_id"))
        except (TypeError, ValueError):
            return True
        seen.add(direction_id)
        for key in (
            "rendered_in_all_direction_heatmap",
            "rendered_in_full_rank_activity_spectrum",
            "rendered_in_signed_bias_panel",
            "exact_direction_id_preserved",
            "no_proxy_or_fallback",
        ):
            if row.get(key) is not True:
                return True
    if seen != set(range(direction_count)):
        return True
    top_rows = payload.get("top_active_direction_rows") if isinstance(payload.get("top_active_direction_rows"), list) else []
    if not top_rows or len(top_rows) > direction_count:
        return True
    hover_rows = payload.get("candidate_hover_rows") if isinstance(payload.get("candidate_hover_rows"), list) else []
    candidate_labels = payload.get("candidate_labels") if isinstance(payload.get("candidate_labels"), list) else []
    if len(hover_rows) < len(candidate_labels):
        return True
    try:
        html = html_path.read_text(encoding="utf-8")
    except Exception:
        return True
    required_markers = (
        "Plotly.newPlot",
        "plotly.min.js",
        "GraphCG full-rank direction audit",
        "Readable full-rank heatmap",
    )
    return any(marker not in html for marker in required_markers)


def _analogical_contracts_need_backfill(root: Path) -> bool:
    maps_payload = _read_json(root / "analogical_simplicial_maps.json")
    topk_contract = maps_payload.get("topk_contract", {}) if isinstance(maps_payload.get("topk_contract"), dict) else {}
    if topk_contract.get("schema_version") != "tropicalgt.analogical_topk.v1":
        return True
    readability = topk_contract.get("readability_contract", {}) if isinstance(topk_contract.get("readability_contract"), dict) else {}
    if readability.get("schema_version") != "tropicalgt.analogical_topk_readability.v1":
        return True
    analogy_path = root / "analogical_simplex_tree_analogy.json"
    analogy_payload = _read_json(analogy_path)
    analogy_contract = analogy_payload.get("contract", {}) if isinstance(analogy_payload.get("contract"), dict) else {}
    if analogy_contract.get("schema_version") != "tropicalgt.analogical_simplex_tree_analogy.v1":
        return True
    if analogy_contract.get("renders_interactive_plotly_table") is not True:
        return True
    if analogy_contract.get("local_plotly_asset_required") is not True:
        return True
    required = (
        root / "analogical_memory_retrieval.html",
        root / "analogical_memory_topk_index.html",
        root / "analogical_memory_map_02.html",
        root / "analogical_simplex_tree_analogy.html",
        analogy_path,
    )
    if any(not path.exists() for path in required):
        return True
    analogy_html = (root / "analogical_simplex_tree_analogy.html").read_text(encoding="utf-8")
    required_markers = (
        "script src",
        "plotly.min.js",
        "Plotly.newPlot",
        "finite simplex-tree rows",
        "preserved face-to-coface chains",
    )
    return any(marker not in analogy_html for marker in required_markers)


def _got_trajectory_contracts_need_backfill(root: Path) -> bool:
    embedding_payload = _read_json(root / "got_embedding_map_payloads.json")
    layout_contract = embedding_payload.get("layout_contract", {}) if isinstance(embedding_payload.get("layout_contract"), dict) else {}
    if layout_contract.get("schema_version") != "tropicalgt.embedding_trajectory_identity.v1":
        return True
    if layout_contract.get("no_proxy_or_fallback") is not True:
        return True
    density_payload = _read_json(root / "got_nll_density_cloud_payload.json")
    visual_contract = density_payload.get("visual_layer_contract", {}) if isinstance(density_payload.get("visual_layer_contract"), dict) else {}
    if visual_contract.get("schema_version") != "tropicalgt.nll_density_render.v1":
        return True
    if visual_contract.get("support_samples_are_model_states") is not False:
        return True
    if density_payload.get("sample_points_are_model_states") is True:
        return True
    required_sidecars = (
        root / "got_full_trajectory_complex_slider_contract.json",
        root / "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json",
        root / "got_full_trajectory_complex_jensen_shannon_slider_contract.json",
        root / "got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json",
    )
    return any(not path.exists() for path in required_sidecars)


def _reasoning_step_contracts_need_backfill(root: Path) -> bool:
    manifest_path = root / "reasoning_step_complex_maps" / "manifest.json"
    if not manifest_path.exists():
        return True
    manifest = _read_json(manifest_path)
    contract = manifest.get("contract", {}) if isinstance(manifest.get("contract"), dict) else {}
    if contract.get("schema_version") != "tropicalgt.reasoning_step_complex_maps.v1":
        return True
    steps = manifest.get("steps", []) if isinstance(manifest.get("steps"), list) else []
    if not steps:
        return True
    directory = root / "reasoning_step_complex_maps"
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return True
        if step.get("step_complex_source_contract", {}).get("schema_version") != "tropicalgt.reasoning_step_complex_source_contract.v1":
            return True
        if step.get("radius_slider_contract", {}).get("schema_version") != "tropicalgt.reasoning_step_radius_slider_summary.v1":
            return True
        if step.get("simplex_tree_poset_contract", {}).get("schema_version") != "tropicalgt.simplex_tree_poset.v1":
            return True
        step_file = directory / str(step.get("file", f"reasoning_step_{index:03d}.html"))
        slider_file = directory / str(step.get("slider_contract_file") or f"{step_file.stem}_slider_contract.json")
        tree_file = directory / str(step.get("simplex_tree_file", f"reasoning_step_{index:03d}_simplex_tree.html"))
        poset_file = directory / str(step.get("simplex_tree_poset_contract_file") or f"{tree_file.stem}_simplex_tree_poset_contract.json")
        if not (step_file.exists() and slider_file.exists() and tree_file.exists() and poset_file.exists()):
            return True
    return False


def _persistence_landscapes_need_backfill(root: Path) -> bool:
    payload_path = root / "trajectory_persistence" / "persistence_landscapes.json"
    if not payload_path.exists():
        return True
    payload = _read_json(payload_path)
    if payload.get("schema_version") != "tropicalgt.persistence_landscape_visual_contract.v1":
        return True
    if payload.get("actual_data_only") is not True or payload.get("no_proxy_or_fallback") is not True:
        return True
    if payload.get("not_nll_fitness_landscape") is not True:
        return True
    if payload.get("available") is True:
        return not (int(payload.get("curve_trace_count", 0) or 0) > 0 and bool(payload.get("landscape_rows")))
    reasons = payload.get("unavailable_reasons", []) if isinstance(payload.get("unavailable_reasons"), list) else []
    return "no_finite_persistence_intervals_for_gudhi_landscape" not in reasons


def _real_resolution_guard_needs_backfill(real: Any) -> bool:
    if not isinstance(real, dict):
        return False
    if real.get("schema_version") != "tropicalgt.real_free_resolution.v1":
        return True
    contract = real.get("certificate_contract") if isinstance(real.get("certificate_contract"), dict) else {}
    if contract.get("schema_version") != "tropicalgt.cas_free_resolution_contract.v1":
        return True
    if contract.get("no_proxy_or_fallback") is not True:
        return True
    paper = real.get("paper_method_contract") if isinstance(real.get("paper_method_contract"), dict) else contract.get("paper_method_contract", {})
    if not isinstance(paper, dict) or paper.get("schema_version") != "tropicalgt.be_fitting_method_contract.v1" or paper.get("arxiv_id") != "2210.11433v1":
        return True
    manifest = real.get("cas_execution_manifest") if isinstance(real.get("cas_execution_manifest"), dict) else {}
    entries = manifest.get("backend_entries", []) if isinstance(manifest.get("backend_entries"), list) else []
    if manifest.get("schema_version") != "tropicalgt.cas_execution_manifest.v1" or manifest.get("no_proxy_or_fallback") is not True or not entries:
        return True
    unavailable = real.get("unavailable_diagnostic") if isinstance(real.get("unavailable_diagnostic"), dict) else {}
    if real.get("available") is not True:
        if real.get("safe_unavailable_render") is not True:
            return True
        if unavailable.get("safe_to_render_only_as_unavailable") is not True:
            return True
        if "Do not substitute chain diagnostics" not in str(unavailable.get("no_proxy_policy", "")):
            return True
    return False


def _dashboard_missing_links(root: Path, required_names: tuple[str, ...]) -> bool:
    dashboard = root / "inference_audit.html"
    existing_required = [name for name in required_names if (root / name).exists()]
    if not existing_required:
        return False
    if not dashboard.exists():
        return True
    try:
        markup = dashboard.read_text(encoding="utf-8")
    except Exception:
        return True
    return any(name not in markup for name in existing_required)


def backfill_audit_root(audit_root: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    root = Path(audit_root)
    actions: list[dict[str, Any]] = []
    if root.name == "got_audit":
        step_dir = root.parent
    elif (root / "got_audit").is_dir():
        step_dir = root
        root = root / "got_audit"
    else:
        step_dir = root.parent

    fan_json = root / "tropical_fan_diagnostics.json"
    fan_html = root / "tropical_fan_diagnostics.html"
    if overwrite or not (fan_json.exists() and fan_html.exists()):
        paths = write_tropical_fan_diagnostics({}, root)
        actions.append(
            {
                "kind": "tropical_fan_unavailable_backfill",
                "reason": "No explicit model-derived tropical ideal was available in this legacy audit bundle; wrote explicit unavailable diagnostics rather than a proxy fan.",
                "paths": paths,
            }
        )

    toric_json = root / "toric_embedding_sidecar.json"
    toric_html = root / "toric_embedding_sidecar.html"
    if overwrite or not (toric_json.exists() and toric_html.exists()):
        paths = write_toric_embedding_sidecar({}, root)
        actions.append(
            {
                "kind": "toric_embedding_sidecar_unavailable_backfill",
                "reason": "No explicit model-derived toric exponent matrix was available in this legacy audit bundle; wrote explicit unavailable finite toric-ideal sidecar diagnostics rather than a chart-bundle, support-token, GraphCG, embedding, or visualization proxy.",
                "paths": paths,
            }
        )


    support_payload_path = root / "tropical_support_payload.json"
    tropical_support_needed = overwrite or _tropical_support_contracts_need_backfill(root)
    if tropical_support_needed and support_payload_path.exists():
        payload = _read_json(support_payload_path)
        tokens = payload.get("tokens", []) if isinstance(payload.get("tokens"), list) else []
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        if tokens:
            trace: dict[str, Any] = {"tokens": tokens}
            if metrics:
                trace["metrics"] = metrics
            paths = write_tropical_support_heatmap({"graph_token_trace": trace, "metrics": metrics}, root)
            refreshed = _read_json(root / "tropical_support_payload.json")
            refreshed_metrics = refreshed.get("metrics", {}) if isinstance(refreshed.get("metrics"), dict) else {}
            actions.append(
                {
                    "kind": "tropical_support_contract_backfill",
                    "reason": "Regenerated tropical support heatmap, no-proxy render contract, readability contract, support assignment status rows, and flow edges from stored tropical_support_payload.json token rows only.",
                    "token_count": int(refreshed_metrics.get("token_count", len(tokens)) or len(tokens)),
                    "unique_support_count": int(refreshed_metrics.get("unique_support_count", 0) or 0),
                    "invalid_support_count": int(refreshed_metrics.get("invalid_support_count", 0) or 0),
                    "paths": paths,
                }
            )
        else:
            actions.append(
                {
                    "kind": "tropical_support_contract_unavailable",
                    "reason": "Tropical support contracts were stale, but tropical_support_payload.json had no stored token rows; no support assignment matrix or margin evidence was fabricated.",
                    "paths": {"tropical_support_payload": str(support_payload_path)},
                }
            )


    scaling_path = root / "inference_scaling_tree.json"
    graphcg_needed = overwrite or _graphcg_contracts_need_backfill(root)
    if graphcg_needed and scaling_path.exists():
        scaling = _read_json(scaling_path)
        candidates = scaling.get("candidates", []) if isinstance(scaling.get("candidates"), list) else []
        candidates_with_graphcg = [
            row
            for row in candidates
            if isinstance(row, dict)
            and isinstance(row.get("graphcg_projection"), dict)
            and isinstance(row["graphcg_projection"].get("all_direction_cosines"), list)
            and row["graphcg_projection"].get("all_direction_cosines")
        ]
        if candidates_with_graphcg:
            paths = write_graphcg_trajectory_visualization(scaling, root)
            refreshed = _read_json(root / "graphcg_direction_cosines_payload.json")
            matrix_shape = refreshed.get("matrix_shape") if isinstance(refreshed.get("matrix_shape"), list) else [0, 0]
            actions.append(
                {
                    "kind": "graphcg_direction_contract_backfill",
                    "reason": "Regenerated GraphCG full-rank direction HTML and payload contracts from stored inference_scaling_tree.json candidate.graphcg_projection.all_direction_cosines only; exact direction ids, top-active rows, candidate hover rows, and no-proxy evidence rows were preserved.",
                    "candidate_count": len(candidates_with_graphcg),
                    "direction_count": int(matrix_shape[1]) if len(matrix_shape) == 2 else 0,
                    "paths": paths,
                }
            )
        else:
            actions.append(
                {
                    "kind": "graphcg_direction_contract_unavailable",
                    "reason": "GraphCG direction contracts were stale, but inference_scaling_tree.json had no stored candidate.graphcg_projection.all_direction_cosines arrays; no direction rows or heatmap were fabricated.",
                    "paths": {"inference_scaling_tree": str(scaling_path)},
                }
            )


    analogical_needed = overwrite or _analogical_contracts_need_backfill(root)
    analogical_memory_path = root / "analogical_memory_retrieval.json"
    if analogical_needed and analogical_memory_path.exists():
        memory = _read_json(analogical_memory_path)
        scaling_for_query = _read_json(root / "inference_scaling_tree.json")
        query_context: dict[str, Any] = {}
        if isinstance(scaling_for_query.get("trajectory_probability_filtered_simplicial_object"), dict):
            query_context["trajectory_probability_filtered_simplicial_object"] = scaling_for_query["trajectory_probability_filtered_simplicial_object"]
        if isinstance(scaling_for_query.get("trajectory_probability_topological_algebra"), dict):
            query_context["topological_algebra"] = scaling_for_query["trajectory_probability_topological_algebra"]
        paths = write_analogical_memory_visualization(memory, root, query_context=query_context)
        actions.append(
            {
                "kind": "analogical_memory_contract_backfill",
                "reason": "Regenerated analogical top-k, unavailable/map, and simplex-tree analogy contracts from stored analogical_memory_retrieval.json plus stored trajectory probability complex evidence when available.",
                "raw_retrieved_count": len(memory.get("retrieved", [])) if isinstance(memory.get("retrieved"), list) else 0,
                "query_probability_complex_available": isinstance(query_context.get("trajectory_probability_filtered_simplicial_object"), dict),
                "paths": paths,
            }
        )

    got_needed = overwrite or _got_trajectory_contracts_need_backfill(root)
    if got_needed and scaling_path.exists():
        scaling = _read_json(scaling_path)
        candidates = scaling.get("candidates", []) if isinstance(scaling.get("candidates"), list) else []
        if candidates:
            paths = write_got_trajectory_visualization(scaling, root)
            actions.append(
                {
                    "kind": "got_trajectory_contract_backfill",
                    "reason": "Regenerated GoT embedding/NLL/full-complex/reasoning-step visualization contracts from stored inference_scaling_tree.json candidates and their model outputs.",
                    "candidate_count": len(candidates),
                    "paths": paths,
                }
            )
        else:
            actions.append(
                {
                    "kind": "got_trajectory_contract_unavailable",
                    "reason": "GoT trajectory contracts were missing or stale, but inference_scaling_tree.json had no stored candidates; no embedding/NLL/full-complex contracts were fabricated.",
                    "paths": {"inference_scaling_tree": str(scaling_path)},
                }
            )

    reasoning_needed = overwrite or _reasoning_step_contracts_need_backfill(root)
    if reasoning_needed and scaling_path.exists():
        scaling = _read_json(scaling_path)
        candidates = scaling.get("candidates", []) if isinstance(scaling.get("candidates"), list) else []
        candidates_with_complex = [row for row in candidates if isinstance(row, dict) and isinstance(row.get("filtered_simplicial_object"), dict)]
        if candidates_with_complex:
            paths = _write_reasoning_step_complex_maps(candidates_with_complex, root)
            actions.append(
                {
                    "kind": "reasoning_step_complex_contract_backfill",
                    "reason": "Regenerated per-step complex pages, radius-slider contracts, SimplexTree poset contracts, source contracts, fingerprints, and manifest from stored inference_scaling_tree.json candidate filtered_simplicial_object payloads.",
                    "candidate_count": len(candidates_with_complex),
                    "paths": paths,
                }
            )
        else:
            actions.append(
                {
                    "kind": "reasoning_step_complex_contract_unavailable",
                    "reason": "Reasoning-step contracts were missing or stale, but inference_scaling_tree.json had no candidates with stored filtered_simplicial_object payloads; no per-step contracts were fabricated.",
                    "paths": {"inference_scaling_tree": str(scaling_path)},
                }
            )

    bif_raw_for_cas = root / "trajectory_level_radius_bifiltration.json"
    if bif_raw_for_cas.exists():
        bif_payload_for_cas = _read_json(bif_raw_for_cas)
        diagnostics = bif_payload_for_cas.get("chain_presentation_diagnostics") if isinstance(bif_payload_for_cas.get("chain_presentation_diagnostics"), dict) else {}
        real = diagnostics.get("real_free_resolution") if isinstance(diagnostics.get("real_free_resolution"), dict) else None
        if overwrite or _real_resolution_guard_needs_backfill(real):
            if isinstance(real, dict):
                hydrated_real = _hydrate_cached_result_contracts(real)
                if hydrated_real.get("available") is not True:
                    backend_probe = hydrated_real.get("backend_probe") if isinstance(hydrated_real.get("backend_probe"), dict) else {}
                    bemultipliers_probe = hydrated_real.get("bemultipliers_probe") if isinstance(hydrated_real.get("bemultipliers_probe"), dict) else {}
                    attempts = hydrated_real.get("backend_attempts") if isinstance(hydrated_real.get("backend_attempts"), list) else []
                    unavailable = hydrated_real.get("unavailable_diagnostic") if isinstance(hydrated_real.get("unavailable_diagnostic"), dict) else {}
                    if unavailable.get("safe_to_render_only_as_unavailable") is not True:
                        hydrated_real["unavailable_diagnostic"] = _unavailable_resolution_diagnostic(
                            str(hydrated_real.get("status", "certificate_failed")),
                            str(hydrated_real.get("reason", "No certified CAS backend output is available for this module.")),
                            attempts,
                            backend_probe,
                            bemultipliers_probe,
                        )
                    hydrated_real["safe_unavailable_render"] = True
                    hydrated_real["unavailable_dependency_action"] = hydrated_real["unavailable_diagnostic"].get("action")
                diagnostics["real_free_resolution"] = hydrated_real
                bif_payload_for_cas["chain_presentation_diagnostics"] = diagnostics
                bif_raw_for_cas.write_text(json.dumps(bif_payload_for_cas, indent=2), encoding="utf-8")
                bif_html_for_cas = root / "trajectory_persistence" / "two_parameter_bifiltration.html"
                rendered = write_two_parameter_bifiltration_visualization(
                    bif_html_for_cas,
                    bif_payload_for_cas,
                    title="Trajectory 2-parameter persistence over F2[x_level,x_radius]",
                )
                refreshed = diagnostics["real_free_resolution"]
                actions.append(
                    {
                        "kind": "cas_resolution_guard_contract_backfill",
                        "reason": "Hydrated stale unavailable CAS free-resolution guard in trajectory_level_radius_bifiltration.json with current certificate, BE/Fitting paper-method, execution-manifest, and unavailable no-proxy diagnostics; no resolution certificate was fabricated.",
                        "available": bool(refreshed.get("available")),
                        "status": refreshed.get("status"),
                        "safe_unavailable_render": bool(refreshed.get("safe_unavailable_render")),
                        "paths": {"raw_bifiltration": str(bif_raw_for_cas), "visual_bifiltration": rendered},
                    }
                )
            else:
                actions.append(
                    {
                        "kind": "cas_resolution_guard_contract_unavailable",
                        "reason": "trajectory_level_radius_bifiltration.json has no real_free_resolution guard to hydrate; no CAS certificate was fabricated.",
                        "paths": {"raw_bifiltration": str(bif_raw_for_cas)},
                    }
                )


    landscapes_needed = overwrite or _persistence_landscapes_need_backfill(root)
    topology_path = root / "trajectory_topological_algebra.json"
    if landscapes_needed and topology_path.exists():
        topology = _read_json(topology_path)
        if topology:
            paths = write_persistence_visualizations(
                topology,
                root / "trajectory_persistence",
                growth=[{"level": 0, "topological_algebra": topology}],
                title_prefix="Trajectory ",
            )
            payload = _read_json(root / "trajectory_persistence" / "persistence_landscapes.json")
            actions.append(
                {
                    "kind": "persistence_landscape_contract_backfill",
                    "reason": "Regenerated trajectory persistence landscape HTML/JSON contract from stored trajectory_topological_algebra.json intervals and GUDHI representation logic only; unavailable state remains explicit when no finite intervals exist.",
                    "available": bool(payload.get("available")),
                    "curve_trace_count": int(payload.get("curve_trace_count", 0) or 0),
                    "finite_persistence_interval_count": int(payload.get("finite_persistence_interval_count", 0) or 0),
                    "unavailable_reasons": payload.get("unavailable_reasons", []),
                    "paths": paths,
                }
            )
        else:
            actions.append(
                {
                    "kind": "persistence_landscape_contract_unavailable",
                    "reason": "Trajectory persistence landscapes were stale or missing, but trajectory_topological_algebra.json could not be parsed; no landscape curves were fabricated.",
                    "paths": {"trajectory_topological_algebra": str(topology_path)},
                }
            )


    bif_raw = root / "trajectory_level_radius_bifiltration.json"
    bif_html = root / "trajectory_persistence" / "two_parameter_bifiltration.html"
    bif_sidecar = bif_html.with_suffix(".json")
    if bif_raw.exists() and (overwrite or not bif_sidecar.exists()):
        payload = _read_json(bif_raw)
        if payload:
            rendered = write_two_parameter_bifiltration_visualization(
                bif_html,
                payload,
                title="Trajectory 2-parameter persistence over F2[x_level,x_radius]",
            )
            actions.append(
                {
                    "kind": "two_parameter_bifiltration_visual_contract_backfill",
                    "reason": "Regenerated the bivariate staircase HTML/JSON contract from the raw trajectory_level_radius_bifiltration.json payload.",
                    "paths": {
                        "html": rendered,
                        "json": str(bif_sidecar),
                        "raw_bifiltration": str(bif_raw),
                    },
                }
            )
        else:
            actions.append(
                {
                    "kind": "two_parameter_bifiltration_visual_contract_unavailable",
                    "reason": "Raw trajectory_level_radius_bifiltration.json was present but could not be parsed as an object; no sidecar was fabricated.",
                    "paths": {"raw_bifiltration": str(bif_raw)},
                }
            )

    dashboard_needs_refresh = _dashboard_missing_links(
        root,
        (
            "tropical_fan_diagnostics.html",
            "toric_embedding_sidecar.html",
            "trajectory_persistence/two_parameter_bifiltration.html",
            "reasoning_step_complex_maps/manifest.json",
            "analogical_memory_topk_index.html",
            "analogical_simplex_tree_analogy.html",
        ),
    )
    if actions or overwrite or dashboard_needs_refresh:
        dashboard_paths = _dashboard_artifact_paths(root)
        if dashboard_paths:
            dashboard_path = _write_inference_dashboard(dashboard_paths, root)
            actions.append(
                {
                    "kind": "inference_audit_dashboard_rebuilt",
                    "reason": "Rebuilt the local audit dashboard so explicit no-proxy sidecars are reachable from inference_audit.html.",
                    "paths": {"inference_audit": str(dashboard_path)},
                }
            )

    report = {
        "schema_version": "tropicalgt.interactive_audit_backfill.v1",
        "audit_root": str(root),
        "step_dir": str(step_dir),
        "overwrite": bool(overwrite),
        "actions": actions,
        "policy": "Backfills only explicit unavailable diagnostics or rerenders visual contracts from existing raw payloads; it does not fabricate CAS certificates, tropical fans, toric ideals, toric embeddings, normal fans, tropical-variety embeddings, or persistence modules.",
    }
    report_path = root / "backfill_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill legacy TropicalGT-I interactive audit bundles with explicit unavailable/visual-contract artifacts.")
    parser.add_argument("--audit-root", required=True, help="Path to a got_audit directory or its step directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args(argv)
    report = backfill_audit_root(args.audit_root, overwrite=args.overwrite)
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
