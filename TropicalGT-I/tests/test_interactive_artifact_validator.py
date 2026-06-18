from __future__ import annotations

import html
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_interactive_audit_artifacts.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_interactive_audit_artifacts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")



def test_evidence_gap_inventory_groups_strict_validator_failures_without_relaxing_gate():
    module = _load_validator()
    errors = [
        "only 1 rows available, expected at least 3",
        "row 0 missing json analogical_simplex_tree_analogy.json",
        "row 0 trajectory persistence landscapes payload is missing",
        "row 0 GraphCG payload is missing structured readability contract",
        "row 0 reasoning-step manifest is missing contract schema",
        "row 0 trajectory bifiltration chain-presentation lacks CAS execution manifest",
    ]

    inventory = module._build_evidence_gap_inventory(errors)

    assert inventory["schema_version"] == "tropicalgt.interactive_audit_evidence_gap_inventory.v1"
    assert inventory["actual_data_only"] is True
    assert inventory["no_proxy_or_fallback"] is True
    assert inventory["strict_validation_still_required"] is True
    assert inventory["gap_count"] == len(errors)
    assert inventory["category_counts"]["row_coverage"] == 1
    assert inventory["category_counts"]["analogical_memory"] == 1
    assert inventory["category_counts"]["persistence_landscapes"] == 1
    assert inventory["category_counts"]["graphcg_direction_audit"] == 1
    assert inventory["category_counts"]["reasoning_step_contracts"] == 1
    assert inventory["category_counts"]["cas_resolution_certificate"] == 1
    assert "does not make an artifact valid" in inventory["policy"]

    markdown = module._markdown_report(
        {
            "audit_root": "/tmp/got_audit",
            "ok": False,
            "rows_checked": 0,
            "errors": errors,
            "row_reports": [],
            "validation_metrics": {"available": False},
            "evidence_gap_inventory": inventory,
        }
    )
    assert "## Evidence Gap Inventory" in markdown
    assert "Strict validation still required: `true`" in markdown
    assert "`analogical_memory`: `1`" in markdown
    assert "## Errors" in markdown


def _html(title: str, extra: str = "Plotly.newPlot play filtration Filtration radius") -> str:
    return f"<!doctype html><title>{title}</title><script src='plotly.min.js'></script><body>{title} {extra}</body>"


def _slider_contract(html_file: str, *, vertices: int = 4, solid_edges: int = 3, filled_faces: int = 0) -> dict[str, object]:
    frames = [
        {
            "threshold": 0.0,
            "initial_radius_frame": True,
            "vertices": vertices,
            "solid_edges": 0,
            "filled_faces": 0,
            "dotted_trajectory_overlays": 0,
            "dotted_direction_overlays": 0,
            "dotted_decoding_overlays": 0,
            "dotted_overlays": 0,
        },
        {
            "threshold": 1.0,
            "initial_radius_frame": False,
            "vertices": vertices,
            "solid_edges": solid_edges,
            "filled_faces": filled_faces,
            "dotted_trajectory_overlays": 0,
            "dotted_direction_overlays": 0,
            "dotted_decoding_overlays": 0,
            "dotted_overlays": 0,
        },
    ]
    return {
        "schema_version": "tropicalgt.radius_filtration_slider_contract.v1",
        "source": "canonical_gudhi_filtered_complex_simplices",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "radius_filtration": True,
        "threshold_order": "ascending_min_to_max",
        "thresholds_ascending": True,
        "threshold_count": len(frames),
        "first_threshold": 0.0,
        "last_threshold": 1.0,
        "frame_count": len(frames),
        "first_frame_vertex_count": vertices,
        "first_frame_solid_edge_count": 0,
        "first_frame_filled_face_count": 0,
        "first_frame_dotted_overlay_count": 0,
        "last_frame_solid_edge_count": solid_edges,
        "last_frame_filled_face_count": filled_faces,
        "solid_lines_semantics": "radius-filtered 1-simplices only",
        "filled_faces_semantics": "radius-gated 2-simplices only",
        "dotted_lines_semantics": "causal_decoding_or_direction_overlay_only_and_radius_gated",
        "initial_radius_frame_hides_dotted_overlays": True,
        "initial_radius_frame_hides_solid_edges_and_faces": True,
        "first_frame_disjoint_vertices_only": True,
        "monotone_visible_counts": True,
        "monotone_solid_radius_edges": True,
        "monotone_filled_radius_faces": True,
        "frames": frames,
        "html_file": html_file,
        "title": html_file,
    }


def _simplex_tree_poset_contract(
    html_file: str,
    *,
    displayed: int = 1,
    cover_edges: int | None = None,
    root_edges: int | None = None,
) -> dict[str, object]:
    cover_edges = displayed if cover_edges is None else cover_edges
    root_edges = min(displayed, cover_edges) if root_edges is None else root_edges
    readability_contract = {
        "schema_version": "tropicalgt.simplex_tree_readability.v1",
        "summary_first_default": True,
        "dimension_counts_panel": "annotation_and_contract_payload",
        "filtration_histogram_panel": "annotation_and_contract_payload",
        "representative_inclusions_visible_by_default": True,
        "representative_cover_edge_count": min(cover_edges, 120),
        "dense_face_to_coface_links_hidden_by_default": bool(cover_edges > 120),
        "dense_face_to_coface_visibility_policy": "legendonly_when_actual_cover_edges_exceed_120",
        "dense_face_to_coface_edge_count": cover_edges,
        "dimension_count_rows": [{"dimension": "0", "count": displayed}],
        "filtration_histogram": [{"lo": 0.0, "hi": 1.0, "count": displayed}],
        "representative_inclusions": [{"face": "{root}", "coface": "{root,a}"}] if cover_edges else [],
        "no_proxy_or_fallback": True,
    }
    return {
        "schema_version": "tropicalgt.simplex_tree_poset.v1",
        "available": True,
        "html_file": html_file,
        "title": html_file,
        "source": "gudhi_canonical_complex(filtered_simplicial_object).simplex_tree",
        "backend": "gudhi.SimplexTree",
        "safe_to_render_simplex_tree": True,
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "layout": "model_embedding_barycentric_face_coface_poset",
        "not_disconnected_simplex_columns": True,
        "empty_simplex_root_present": True,
        "displayed_simplex_count": displayed,
        "source_simplex_count": displayed,
        "truncated": False,
        "max_nodes": 1600,
        "dimension_counts": {"dim_0": displayed},
        "actual_face_to_coface_cover_edges": cover_edges,
        "empty_root_vertex_cover_edges": root_edges,
        "optional_sorted_label_trie_prefix_edges": displayed,
        "primary_edges": "actual_face_to_coface_covers",
        "optional_prefix_links_visible": "legendonly",
        "position_source": "model_embedding_barycenters_with_dimension_and_filtration_lift",
        "readability_contract": readability_contract,
        "summary_first_default": True,
        "representative_cover_edges_visible_by_default": True,
        "dense_face_to_coface_links_hidden_by_default": bool(cover_edges > 120),
        "all_non_vertex_simplices_have_face_cover_edges": True,
    }


def _step_source_contract(candidate: dict[str, object], step_file: str, simplex_tree_file: str, fingerprint: str) -> dict[str, object]:
    return {
        "schema_version": "tropicalgt.reasoning_step_complex_source_contract.v1",
        "source": "candidate.filtered_simplicial_object",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "candidate_record_id": candidate["record_id"],
        "candidate_level": candidate["level"],
        "candidate_path": candidate["path"],
        "step_complex_fingerprint": fingerprint,
        "uses_global_trajectory_complex_as_proxy": False,
        "uses_embedding_trajectory_map_as_proxy": False,
        "uses_static_probability_complex_as_proxy": False,
        "displayed_vertex_count": 1,
        "displayed_edge_count": 0,
        "displayed_face_count": 0,
        "summary_vertex_count": 1,
        "summary_edge_count": 0,
        "summary_two_simplex_count": 0,
        "displayed_probability_vector_vertex_count": 0,
        "displayed_embedding_vertex_count": 1,
        "displayed_vertex_labels_sample": [candidate["record_id"]],
        "simplex_tree_backend": "gudhi.SimplexTree",
        "simplex_tree_available": True,
        "simplex_tree_num_vertices": 1,
        "simplex_tree_num_simplices": 1,
        "graph_token_direction_overlay_source": "candidate_trace_overlay_when_present",
        "decoding_causal_overlay_source": "candidate_decoding_or_causal_metadata_when_present",
        "source_counts_match_canonical_summary": True,
        "safe_to_render_as_step_complex": True,
    }


def _slider_summary(html_file: str, contract_file: str, *, vertices: int = 4, solid_edges: int = 3, filled_faces: int = 1) -> dict[str, object]:
    contract = _slider_contract(html_file, vertices=vertices, solid_edges=solid_edges, filled_faces=filled_faces)
    return {
        "schema_version": "tropicalgt.reasoning_step_radius_slider_summary.v1",
        "source": f"reasoning_step_complex_maps/{contract_file}",
        "contract_file": contract_file,
        "html_file": html_file,
        "contract_schema_version": contract["schema_version"],
        "contract_source": contract["source"],
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "radius_filtration": True,
        "threshold_order": contract["threshold_order"],
        "thresholds_ascending": True,
        "threshold_count": contract["threshold_count"],
        "frame_count": contract["frame_count"],
        "first_frame_vertex_count": vertices,
        "first_frame_solid_edge_count": 0,
        "first_frame_filled_face_count": 0,
        "first_frame_dotted_overlay_count": 0,
        "initial_radius_frame_hides_dotted_overlays": True,
        "initial_radius_frame_hides_solid_edges_and_faces": True,
        "first_frame_disjoint_vertices_only": True,
        "monotone_visible_counts": True,
        "monotone_solid_radius_edges": True,
        "monotone_filled_radius_faces": True,
        "solid_lines_semantics_ok": True,
        "filled_faces_semantics_ok": True,
        "dotted_lines_semantics_ok": True,
        "safe_to_render_radius_filtration": True,
    }


def _row(root: Path, name: str) -> Path:
    row = root if name == "." else root / name
    row.mkdir(parents=True, exist_ok=True)
    candidates = [
        {"record_id": "root", "path": [], "parent": None, "level": 0, "nll": 1.0, "embedding": [0.0, 0.0, 0.0]},
        {"record_id": "a", "path": ["expand"], "parent": "root", "level": 1, "nll": 0.9, "embedding": [1.0, 0.0, 0.0]},
        {"record_id": "b", "path": ["verify"], "parent": "root", "level": 1, "nll": 1.1, "embedding": [0.0, 1.0, 0.0]},
        {"record_id": "c", "path": ["expand", "refine"], "parent": "a", "level": 2, "nll": 0.8, "embedding": [1.0, 1.0, 0.0]},
    ]
    nodes = []
    for idx, candidate in enumerate(candidates):
        nodes.append(
            {
                "record_id": candidate["record_id"],
                "level": candidate["level"],
                "nll": candidate["nll"],
                "embedding": candidate["embedding"],
                "embedding_source": "model graph_state",
                "depth": candidate["level"],
                "branch_id": "/".join(candidate["path"]) if candidate["path"] else "root",
                "embedding_pca": {"pc1": candidate["embedding"][0], "pc2": candidate["embedding"][1], "pc3": candidate["embedding"][2]},
                "plot": {
                    "x": candidate["embedding"][0],
                    "y": candidate["embedding"][1],
                    "z": candidate["embedding"][2],
                    "z_surface": None,
                    "z_centered_scaled_nll": candidate["nll"],
                    "raw_centered_scaled_nll": candidate["nll"],
                    "raw_nll": candidate["nll"],
                    "touches_nll_surface": False,
                },
                "reasoning_step_index": idx,
                "step_complex_href": f"reasoning_step_complex_maps/reasoning_step_{idx:03d}.html",
                "step_simplex_tree_href": f"reasoning_step_complex_maps/reasoning_step_{idx:03d}_simplex_tree.html",
                "step_complex_contract": "this GoT state maps to a per-step interactive filtered simplicial complex page and simplex-tree page",
                "filtered_simplicial_object": {
                    "summary": {"num_vertices": 1, "num_edges": 0, "num_two_simplices": 0},
                    "simplices": [{"simplex": [candidate["record_id"]], "dimension": 0, "filtration": 0.0}],
                },
            }
        )
    payload = {
        "nodes": nodes,
        "edges": [
            {"source": "root", "target": "a"},
            {"source": "root", "target": "b"},
            {"source": "a", "target": "c"},
        ],
        "embedding_pca_diagnostics": {
            "coordinate_source": "model graph_state embeddings",
            "n_samples": 4,
            "embedding_dim": 3,
            "pairwise_distance_correlation": 1.0,
            "normalized_stress": 0.0,
            "explained_variance_ratio_sum3": 1.0,
        },
        "nll_surface": {
            "available": True,
            "touches_points": True,
            "surface_kind": "sample_supported_local_idw_surface",
            "max_point_residual": 0.0,
            "z_axis": "projected_nll_fitness_energy",
            "raw_nll_range": 0.3,
            "exact_anchor_layer": True,
            "sparse_observed_anchor_layer": True,
            "actual_landscape_layer": False,
            "support_radius": 0.5,
            "surface_contact_contract": "disabled for the main trajectory page: rendered GoT state marker z uses PC3 geometry, while raw NLL is preserved as color, hover, centered/scaled metadata, and projected surface diagnostics",
            "trajectory_point_surface_residual_max": 0.0,
            "surface_projected_z_by_record_id": {row["record_id"]: row["nll"] for row in candidates},
            "local_embedding_neighborhood_surface": {
                "schema_version": "tropicalgt.local_embedding_neighborhood_surface.v1",
                "available": True,
                "surface_kind": "local_interpolating_nll_sheet",
                "anchor_source": "scaling_report.candidates embeddings plus measured raw NLL values",
                "model_evaluated_anchor_count": len(candidates),
                "anchor_record_ids": [row["record_id"] for row in candidates],
                "surface_projected_z_by_record_id": {row["record_id"]: row["nll"] for row in candidates},
                "trajectory_point_surface_residual_max": 0.0,
                "invented_nll_values": False,
                "support_samples_are_model_states": False,
                "trace_visibility": "legendonly",
                "no_proxy_or_fallback": True,
            },
            "local_interpolating_sheet": {"available": False, "reason": "legacy_key_disabled; use local_embedding_neighborhood_surface for observed-anchor-only interpolation metadata"},
            "surrogate_landscape_layer": {"available": False, "reason": "disabled_by_default_not_model_evaluated"},
        },
        "nll_progress": {
            "edge_count": 3,
            "improving_edge_fraction": 2 / 3,
            "mean_edge_delta": -0.0666666667,
            "best_terminal_improvement_from_root": 0.2,
            "by_level": [
                {"level": 0, "count": 1, "mean_nll": 1.0},
                {"level": 1, "count": 2, "mean_nll": 1.0},
                {"level": 2, "count": 1, "mean_nll": 0.8},
            ],
        },
    }
    _write(row / "inference_scaling_tree.json", json.dumps({"stochastic_actions": True, "sampling_temperature": 2.0, "sampling_exploration": 0.4, "candidates": candidates}))
    _write(row / "got_trajectory_payloads.json", json.dumps(payload))
    density_edges = [
        {"source": "root", "target": "a", "action": "expand", "source_nll": 1.0, "target_nll": 0.9, "nll_delta": -0.1, "improves": True},
        {"source": "root", "target": "b", "action": "verify", "source_nll": 1.0, "target_nll": 1.1, "nll_delta": 0.1, "improves": False},
        {"source": "a", "target": "c", "action": "refine", "source_nll": 0.9, "target_nll": 0.8, "nll_delta": -0.1, "improves": True},
    ]
    density_visual_contract = {
        "schema_version": "tropicalgt.nll_density_render.v1",
        "page": "standalone_density_cloud",
        "coordinate_space": "actual 3D PCA coordinates of model graph-state embeddings",
        "z_axis_policy": "z is PC3(graph_state embedding); raw NLL is encoded by color, hover, and density metadata",
        "actual_model_anchor_count": 4,
        "support_sample_count": 440,
        "kernel": "isotropic Gaussian in 3D PCA coordinates",
        "kernel_bandwidth": 0.25,
        "actual_anchor_trace_name": "actual model GoT state anchors",
        "actual_anchor_layer_visible_by_default": True,
        "support_sample_trace_name": "audit samples from the Gaussian NLL field (hidden by default)",
        "support_sample_trace_visibility": "legendonly",
        "support_samples_are_model_states": False,
        "support_samples_hidden_as_model_states": True,
        "visible_density_layers": ["density_volume", "anchor_gaussian_neighborhoods", "actual_model_anchor_markers"],
        "audit_sample_layer_role": "legend-only Gaussian support samples for local NLL density inspection",
        "camera_eye": {"x": 1.8, "y": 1.8, "z": 1.2},
        "no_proxy_or_fallback": True,
    }
    _write(
        row / "got_nll_density_cloud_payload.json",
        json.dumps(
            {
                "available": True,
                "render_contract": "Gaussian cloud points are not model states; they visualize local NLL density around actual embedding vectors, while only large labeled markers are model states",
                "visual_layer_contract": density_visual_contract,
                "density_contract": {
                    "actual_model_anchor_layer": True,
                    "support_samples_hidden_as_model_states": True,
                    "sample_points_are_model_states": False,
                    "support_sample_trace_visibility": "legendonly",
                    "support_sample_count": 440,
                    "actual_model_anchor_count": 4,
                    "kernel_bandwidth": 0.25,
                    "visible_density_layers": ["density_volume", "anchor_gaussian_neighborhoods", "actual_model_anchor_markers"],
                    "z_axis_policy": "z is PC3(graph_state embedding); raw NLL is encoded by color, hover, and density metadata",
                    "edge_delta_rule": "target raw NLL minus source raw NLL over actual GoT tree edges",
                },
                "anchor_count": 4,
                "actual_model_anchor_count": 4,
                "support_sample_count": 440,
                "support_samples_hidden_as_model_states": True,
                "sample_points_are_model_states": False,
                "support_sample_trace_visibility": "legendonly",
                "visible_density_layers": ["density_volume", "anchor_gaussian_neighborhoods", "actual_model_anchor_markers"],
                "z_axis_policy": "z is PC3(graph_state embedding); raw NLL is encoded by color, hover, and density metadata",
                "kernel_bandwidth": 0.25,
                "nll_range": {"min": 0.8, "max": 1.1, "span": 0.3},
                "local_nll_summary": {"count": 440, "min": 0.8, "max": 1.1, "mean": 0.95, "std": 0.1},
                "edge_nll_delta_summary": {"count": 3, "min": -0.1, "max": 0.1, "mean": -0.033333333, "std": 0.094},
                "terminal_nll_progress": {"root_mean_nll": 1.0, "terminal_count": 2, "terminal_min_nll": 0.8, "best_terminal_improvement_from_root": 0.2, "improving_edge_fraction": 2 / 3},
                "density_volume": {"available": True, "support_samples_are_not_model_states": True, "grid_size": 28},
                "density_cloud": {
                    "available": True,
                    "source": "actual model-evaluated graph_state PCA anchors and measured raw NLL values",
                    "support_samples_are_not_model_states": True,
                    "exact_anchor_layer": True,
                    "anchor_count": 4,
                    "sample_count": 440,
                    "sigma": 0.25,
                    "nll_min": 0.8,
                    "nll_max": 1.1,
                    "render_contract": "Gaussian cloud points are not model states; they visualize local NLL density around actual embedding vectors, while only large labeled markers are model states",
                    "visual_layer_contract": density_visual_contract,
                    "support_sample_trace_visibility": "legendonly",
                    "visible_density_layers": ["density_volume", "anchor_gaussian_neighborhoods", "actual_model_anchor_markers"],
                    "z_axis_policy": "z is PC3(graph_state embedding); raw NLL is encoded by color, hover, and density metadata",
                    "density_volume": {"available": True, "support_samples_are_not_model_states": True},
                },
                "anchors": [
                    {"record_id": candidate["record_id"], "level": candidate["level"], "nll": candidate["nll"], "density_cloud_role": "actual_model_evaluated_graph_state_anchor"}
                    for candidate in candidates
                ],
                "support_samples": {"available": True, "count": 440, "visible_by_default": False, "visible_as_model_states": False, "rendered_trace_visibility": "legendonly"},
                "nodes": [
                    {"record_id": candidate["record_id"], "level": candidate["level"], "nll": candidate["nll"], "density_cloud_role": "actual_model_evaluated_graph_state_anchor"}
                    for candidate in candidates
                ],
                "edges": density_edges,
            }
        ),
    )
    _write(
        row / "got_embedding_map_payloads.json",
        json.dumps(
            {
                "coordinate_source": "PCA of model graph_state embeddings; no level/tree layout coordinates are used",
                "layout_contract": {
                    "schema_version": "tropicalgt.embedding_trajectory_identity.v1",
                    "coordinate_source": "model graph_state embeddings",
                    "branch_depth_metadata_present": True,
                    "parent_child_transitions_present": True,
                    "parent_child_transition_count": 3,
                    "edge_source": "graph_of_thought_parent_edges",
                    "node_embedding_source": "model graph_state",
                    "pca_quality_warning": False,
                    "geometric_separation_overclaim_allowed": False,
                    "no_proxy_or_fallback": True,
                },
                "nodes": nodes,
                "edges": [
                    {"source": "root", "target": "a", "edge_source": "graph_of_thought_parent_edges", "transition_kind": "GoT parent-child trajectory", "source_depth": 0, "target_depth": 1, "nll_delta": -0.1},
                    {"source": "root", "target": "b", "edge_source": "graph_of_thought_parent_edges", "transition_kind": "GoT parent-child trajectory", "source_depth": 0, "target_depth": 1, "nll_delta": 0.1},
                    {"source": "a", "target": "c", "edge_source": "graph_of_thought_parent_edges", "transition_kind": "GoT parent-child trajectory", "source_depth": 1, "target_depth": 2, "nll_delta": -0.1},
                ],
                "filtered_simplicial_objects": [node["filtered_simplicial_object"] for node in nodes],
            }
        ),
    )
    _write(
        row / "got_full_trajectory_complex_payload.json",
        json.dumps(
            {
                "filtered_simplicial_object": {
                    "summary": {"num_vertices": 4, "num_edges": 3, "num_two_simplices": 0, "radius_filtration": True, "embedding_metric": "euclidean", "filtration_model": "embedding_vietoris_rips_2_skeleton"},
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "dimension": 1, "num_simplices": 7},
                    "trajectory_overlay": {"source": "graph_of_thought_parent_edges", "distance_metric": "euclidean", "edge_count": 3, "edges": [{"source": "root", "target": "a"}, {"source": "root", "target": "b"}, {"source": "a", "target": "c"}]},
                    "decoding_causal_overlay": {"source": "graph_of_thought_parent_decoding_order", "distance_metric": "euclidean", "edge_count": 3, "edges": [{"source": "root", "target": "a", "style": "dotted", "directed": True}, {"source": "root", "target": "b", "style": "dotted", "directed": True}, {"source": "a", "target": "c", "style": "dotted", "directed": True}]},
                    "simplices": [
                        {"simplex": ["root"], "dimension": 0, "embedding": [0, 0, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["a"], "dimension": 0, "embedding": [1, 0, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["b"], "dimension": 0, "embedding": [0, 1, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["c"], "dimension": 0, "embedding": [1, 1, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["root", "a"], "dimension": 1},
                        {"simplex": ["root", "b"], "dimension": 1},
                        {"simplex": ["a", "c"], "dimension": 1},
                    ],
                },
                "probability_filtered_simplicial_object": {
                    "available": True,
                    "summary": {
                        "num_vertices": 4,
                        "num_edges": 3,
                        "num_two_simplices": 0,
                        "radius_filtration": True,
                        "embedding_metric": "jensen_shannon",
                        "filtration_model": "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton",
                    },
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "dimension": 1, "num_simplices": 7},
                    "trajectory_overlay": {"source": "graph_of_thought_parent_edges", "distance_metric": "jensen_shannon", "edge_count": 3, "edges": [{"source": "root", "target": "a"}, {"source": "root", "target": "b"}, {"source": "a", "target": "c"}]},
                    "decoding_causal_overlay": {"source": "graph_of_thought_parent_decoding_order", "distance_metric": "jensen_shannon", "edge_count": 3, "edges": [{"source": "root", "target": "a", "style": "dotted", "directed": True}, {"source": "root", "target": "b", "style": "dotted", "directed": True}, {"source": "a", "target": "c", "style": "dotted", "directed": True}]},
                    "simplices": [
                        {"simplex": ["root"], "dimension": 0, "model_probability_vector": [0.7, 0.2, 0.1]},
                        {"simplex": ["a"], "dimension": 0, "model_probability_vector": [0.2, 0.7, 0.1]},
                        {"simplex": ["b"], "dimension": 0, "model_probability_vector": [0.2, 0.1, 0.7]},
                        {"simplex": ["c"], "dimension": 0, "model_probability_vector": [0.4, 0.4, 0.2]},
                        {"simplex": ["root", "a"], "dimension": 1},
                        {"simplex": ["root", "b"], "dimension": 1},
                        {"simplex": ["a", "c"], "dimension": 1},
                    ],
                },
                "trajectory_complex_overlay_contract": {
                    "schema_version": "tropicalgt.trajectory_complex_overlay_contract.v1",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "solid_lines_reserved_for_radius_simplices": True,
                    "filled_faces_reserved_for_radius_simplices": True,
                    "dotted_lines_reserved_for_trajectory_decoding_order_overlays": True,
                    "probability_view_available": True,
                    "safe_to_render_embedding_view": True,
                    "safe_to_render_probability_view": True,
                    "safe_to_render_available_views": True,
                    "embedding_view": {
                        "schema_version": "tropicalgt.trajectory_complex_overlay_view_contract.v1",
                        "view": "embedding_radius_trajectory_complex",
                        "available": True,
                        "source": "trajectory_filtered_simplicial_object",
                        "actual_data_only": True,
                        "no_proxy_or_fallback": True,
                        "distance_metric": "euclidean",
                        "expected_distance_metric": "euclidean",
                        "filtration_model": "embedding_vietoris_rips_2_skeleton",
                        "radius_filtration": True,
                        "solid_lines_semantics": "radius-filtered 1-simplices only",
                        "filled_faces_semantics": "radius-gated 2-simplices only",
                        "dotted_lines_semantics": "trajectory and decoding/order overlays only",
                        "solid_edges_from_radius_simplices": True,
                        "filled_faces_from_radius_simplices": True,
                        "dotted_edges_reserved_for_overlays": True,
                        "vertex_count": 4,
                        "radius_edge_count": 3,
                        "radius_face_count": 0,
                        "summary_vertex_count": 4,
                        "summary_edge_count": 3,
                        "summary_two_simplex_count": 0,
                        "source_counts_match_summary": True,
                        "simplex_tree_backend": "gudhi.SimplexTree",
                        "trajectory_overlay_source": "graph_of_thought_parent_edges",
                        "trajectory_overlay_distance_metric": "euclidean",
                        "trajectory_overlay_edge_count": 3,
                        "decoding_overlay_source": "graph_of_thought_parent_decoding_order",
                        "decoding_overlay_distance_metric": "euclidean",
                        "decoding_overlay_edge_count": 3,
                        "decoding_overlay_edges_are_dotted": True,
                        "decoding_overlay_edges_are_directed": True,
                        "safe_to_render_overlay_semantics": True,
                    },
                    "probability_view": {
                        "schema_version": "tropicalgt.trajectory_complex_overlay_view_contract.v1",
                        "view": "probability_jensen_shannon_trajectory_complex",
                        "available": True,
                        "source": "trajectory_probability_filtered_simplicial_object",
                        "actual_data_only": True,
                        "no_proxy_or_fallback": True,
                        "distance_metric": "jensen_shannon",
                        "expected_distance_metric": "jensen_shannon",
                        "filtration_model": "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton",
                        "radius_filtration": True,
                        "solid_lines_semantics": "radius-filtered 1-simplices only",
                        "filled_faces_semantics": "radius-gated 2-simplices only",
                        "dotted_lines_semantics": "trajectory and decoding/order overlays only",
                        "solid_edges_from_radius_simplices": True,
                        "filled_faces_from_radius_simplices": True,
                        "dotted_edges_reserved_for_overlays": True,
                        "vertex_count": 4,
                        "radius_edge_count": 3,
                        "radius_face_count": 0,
                        "summary_vertex_count": 4,
                        "summary_edge_count": 3,
                        "summary_two_simplex_count": 0,
                        "source_counts_match_summary": True,
                        "simplex_tree_backend": "gudhi.SimplexTree",
                        "trajectory_overlay_source": "graph_of_thought_parent_edges",
                        "trajectory_overlay_distance_metric": "jensen_shannon",
                        "trajectory_overlay_edge_count": 3,
                        "decoding_overlay_source": "graph_of_thought_parent_decoding_order",
                        "decoding_overlay_distance_metric": "jensen_shannon",
                        "decoding_overlay_edge_count": 3,
                        "decoding_overlay_edges_are_dotted": True,
                        "decoding_overlay_edges_are_directed": True,
                        "safe_to_render_overlay_semantics": True,
                    },
                    "render_contract": "Full trajectory complex pages reserve solid lines/faces for radius-filtered simplices and dotted directed lines for GoT trajectory/decoding-order overlays. The Jensen-Shannon page is rendered only from model candidate probability vectors; unavailable probability views are explicit and are not substituted by embedding or static probability proxies.",
                }
            }
        ),
    )
    steps = []
    for idx in range(4):
        basis = {
            "schema_version": "tropicalgt.reasoning_step_complex_fingerprint_basis.v1",
            "hash_algorithm": "sha256_canonical_json",
            "source": "gudhi_canonical_complex(filtered_simplicial_object)",
            "summary": {"num_vertices": 1},
            "thresholds": [0.0],
            "simplices": [
                {
                    "simplex": [candidates[idx]["record_id"]],
                    "dimension": 0,
                    "filtration": 0.0,
                    "type": "reasoning_step_vertex",
                    "probability_source": "",
                    "has_probability_vector": False,
                    "has_embedding": True,
                }
            ],
            "simplex_tree": {"backend": "gudhi.SimplexTree", "available": True, "dimension": 0, "num_simplices": 1, "num_vertices": 1},
            "no_record_id_or_path_in_hash": True,
        }
        fingerprint = f"fixture-step-fingerprint-{idx}"
        step_file = f"reasoning_step_{idx:03d}.html"
        simplex_tree_file = f"reasoning_step_{idx:03d}_simplex_tree.html"
        simplex_tree_poset_contract_file = f"reasoning_step_{idx:03d}_simplex_tree_simplex_tree_poset_contract.json"
        simplex_tree_poset_contract = _simplex_tree_poset_contract(simplex_tree_file, displayed=1, cover_edges=1, root_edges=1)
        steps.append(
            {
                "index": idx,
                "record_id": candidates[idx]["record_id"],
                "level": candidates[idx]["level"],
                "path": candidates[idx]["path"],
                "file": step_file,
                "simplex_tree_file": simplex_tree_file,
                "simplex_tree_poset_contract_file": simplex_tree_poset_contract_file,
                "simplex_tree_poset_contract": simplex_tree_poset_contract,
                "slider_contract_file": f"reasoning_step_{idx:03d}_slider_contract.json",
                "radius_slider_contract": _slider_summary(
                    step_file,
                    f"reasoning_step_{idx:03d}_slider_contract.json",
                    vertices=1,
                    solid_edges=0,
                    filled_faces=0,
                ),
                "step_complex_source_contract": _step_source_contract(candidates[idx], step_file, simplex_tree_file, fingerprint),
                "summary": {"num_vertices": 1, "num_edges": 0, "num_two_simplices": 0},
                "step_complex_fingerprint": fingerprint,
                "step_complex_fingerprint_basis": basis,
                "simplex_tree": {"backend": "gudhi.SimplexTree"},
                "simplex_tree_backend": "gudhi.SimplexTree",
                "simplex_tree_available": True,
            }
        )
    step_manifest_contract = {
        "schema_version": "tropicalgt.reasoning_step_complex_maps.v1",
        "available": True,
        "no_proxy_or_fallback": True,
        "actual_data_only": True,
        "one_page_per_model_evaluated_reasoning_step": True,
        "embedding_trajectory_map_is_not_a_step_complex": True,
        "step_count": 4,
        "rendered_complex_pages": 4,
        "rendered_simplex_tree_pages": 4,
        "simplex_tree_poset_contract_schema_version": "tropicalgt.simplex_tree_poset.v1",
        "simplex_tree_poset_contract_source": "per-step reasoning_step_*_simplex_tree_poset_contract.json sidecars summarized into this manifest",
        "rendered_simplex_tree_poset_contracts": 4,
        "all_steps_have_simplex_tree_poset_contracts": True,
        "all_step_simplex_tree_posets_no_proxy": True,
        "all_step_simplex_tree_posets_use_gudhi": True,
        "all_step_simplex_tree_posets_face_coface_primary": True,
        "all_step_simplex_tree_posets_safe_to_render": True,
        "simplex_tree_poset_unavailable_count": 0,
        "simplex_tree_poset_unavailable_steps": [],
        "source_contract_schema_version": "tropicalgt.reasoning_step_complex_source_contract.v1",
        "source_contract_source": "candidate.filtered_simplicial_object on each manifest row",
        "rendered_source_contracts": 4,
        "all_steps_have_source_contracts": True,
        "all_step_complexes_use_candidate_filtered_object_source": True,
        "all_step_complex_source_contracts_no_proxy": True,
        "all_step_complex_source_contracts_safe": True,
        "all_step_complex_source_counts_match_summary": True,
        "all_step_complexes_have_vertices": True,
        "source_contract_unavailable_count": 0,
        "source_contract_unavailable_steps": [],
        "slider_contract_schema_version": "tropicalgt.reasoning_step_radius_slider_summary.v1",
        "radius_slider_contract_source": "per-step reasoning_step_*_slider_contract.json sidecars summarized into this manifest",
        "rendered_slider_contracts": 4,
        "all_steps_have_radius_slider_contracts": True,
        "all_step_radius_sliders_start_disjoint_vertices": True,
        "all_step_radius_sliders_monotone": True,
        "all_step_radius_sliders_no_proxy": True,
        "all_step_radius_sliders_safe_to_render": True,
        "radius_slider_unavailable_count": 0,
        "radius_slider_unavailable_steps": [],
        "fingerprint_source": "sha256 canonical JSON over per-step gudhi_canonical_complex(filtered_simplicial_object); record id/path excluded",
        "all_step_complex_fingerprints_present": True,
        "unique_step_complex_fingerprint_count": 4,
        "all_step_complex_fingerprints_unique": True,
        "duplicate_step_complex_fingerprint_groups": [],
        "gudhi_simplex_tree_step_count": 4,
        "simplex_tree_unavailable_count": 0,
        "simplex_tree_unavailable_steps": [],
        "claim": "Each listed step opens its own radius-filtered complex and SimplexTree/explicit-unavailable page; no global trajectory PCA surface is used as a substitute.",
    }
    _write(row / "reasoning_step_complex_maps/manifest.json", json.dumps({"contract": step_manifest_contract, "steps": steps}))
    analogical_topk_readability_contract = {
        "schema_version": "tropicalgt.analogical_topk_readability.v1",
        "status": "available",
        "top_k_rendered": 2,
        "no_proxy_or_fallback": True,
        "topk_index_has_readable_table": True,
        "one_selected_map_view_per_rendered_rank": True,
        "table_rows_link_to_pair_pages": True,
        "insufficient_memory_state_explicit": True,
        "displays_quality_gate_and_filtered_counts": True,
        "separates_retrieval_probability_topology_algebra_columns": True,
        "probability_js_assignment_column_required": True,
        "map_claim_column_required": True,
        "simplex_tree_preservation_column_required": True,
        "edge_face_filtration_preservation_not_overclaimed": True,
        "derived_algebraic_column_policy": "conservative derived/algebraic score; high coarse signatures cannot substitute for missing PH/free-resolution/rank/chain-map evidence",
        "signature_cosine_column_policy": "coarse signature cosine is displayed separately and is not a derived-equivalence claim",
        "required_table_columns": [
            "correspondence",
            "retrieval",
            "prob-map source",
            "map claim",
            "derived/algebraic",
            "coarse signature",
            "simplex-tree map",
            "edge certificate",
        ],
    }
    analogical_topk_contract = {
        "schema_version": "tropicalgt.analogical_topk.v1",
        "available": True,
        "status": "available",
        "reason_detail": "",
        "no_proxy_or_fallback": True,
        "retrieval_requires_model_probability_vectors": True,
        "query_complex_required": "trajectory_probability_filtered_simplicial_object",
        "codomain_complex_required": "trajectory_probability_filtered_simplicial_object",
        "embedding_only_assignment_allowed": False,
        "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors",
        "simplicial_map_claim_requires": "vertex_edge_face_simplex_tree_and_filtration_preservation",
        "chain_map_claim_requires": "certified_filtered_simplicial_map",
        "persistence_module_morphism_claim_requires": "certified_filtered_simplicial_map",
        "query_complex_source": "trajectory_probability_filtered_simplicial_object",
        "raw_retrieved_count": 2,
        "qualified_model_probability_memory_count": 2,
        "rejected_retrieved_count": 0,
        "top_k_requested": 2,
        "top_k_rendered": 2,
        "quality_gate": {"min_quality_score": 0.2},
        "bank_path": "",
        "readability_contract": analogical_topk_readability_contract,
    }
    _write(row / "analogical_simplicial_maps.json", json.dumps({"topk_contract": analogical_topk_contract, "maps": [
        {
            "query_complex_source": "trajectory_probability_filtered_simplicial_object",
            "codomain_complex_source": "trajectory_probability_filtered_simplicial_object",
            "map_source": "model_probability_jensen_shannon_assignment",
            "probability_vector_evidence": {"schema_version": "tropicalgt.probability_vector_assignment_evidence.v1", "source": "probability_filtered_complex_vertices", "probability_vector_source": "model_probability_vectors_on_vertices", "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors", "assignment_solver": "scipy_linear_sum_assignment", "probability_alignment": "zero_pad_to_common_token_index_feature_space_then_renormalize", "displayed_query_vertices": 2, "displayed_memory_vertices": 2, "query_probability_vertex_count": 2, "memory_probability_vertex_count": 2, "all_displayed_query_vertices_have_probability_vectors": True, "all_displayed_memory_vertices_have_probability_vectors": True, "embedding_only_assignment_used": False, "no_proxy_or_fallback": True},
            "correspondence_table_rows": [
                {"row_index": 0, "query_vertex": "a", "memory_vertex": "x", "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors", "assignment_solver": "scipy_linear_sum_assignment", "jensen_shannon_distance": 0.1, "assignment_cost": 0.15, "embedding_only_assignment_used": False, "rendered_as_map_edge": False, "no_proxy_or_fallback": True},
                {"row_index": 1, "query_vertex": "b", "memory_vertex": "y", "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors", "assignment_solver": "scipy_linear_sum_assignment", "jensen_shannon_distance": 0.14, "assignment_cost": 0.19, "embedding_only_assignment_used": False, "rendered_as_map_edge": False, "no_proxy_or_fallback": True},
            ],
            "layout_contract": {
                "schema_version": "tropicalgt.analogical_map_layout.v1",
                "default_view": "side_by_side_query_codomain_small_multiples_plus_correspondence_table",
                "query_panel": "query trajectory probability complex",
                "codomain_panel": "retrieved memory probability complex",
                "visual_layers_separated": ["query_vertices_edges_faces", "codomain_vertices_edges_faces", "probability_js_vertex_correspondence_edges", "preserved_simplex_evidence", "failed_simplex_evidence", "filtration_distortion_diagnostics", "certificate_quality_table"],
                "correspondence_table_rows": 2,
                "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors",
                "assignment_evidence_available": True,
                "embedding_only_assignment_allowed": False,
                "fail_closed_when_probability_vectors_missing": True,
                "draws_pseudo_map_when_unavailable": False,
                "safe_to_draw_simplicial_map_edges": False,
                "safe_to_draw_chain_map_or_module_morphism": False,
                "map_render_claim": "probability_correspondence_not_a_simplicial_map",
                "map_claim_failure_reason": "simplex_tree_map_not_fully_preserved",
                "checked_edges": 2,
                "preserved_edges": 1,
                "checked_two_simplices": 0,
                "preserved_two_simplices": 0,
                "no_proxy_or_fallback": True,
            },
            "is_identity_self_map": False,
            "pair_page": "analogical_memory_retrieval.html",
            "jensen_shannon_distance_mean": 0.12,
            "assignment_cost_mean": 0.17,
            "jensen_shannon_distance_summary": {"count": 2, "min": 0.1, "max": 0.14, "mean": 0.12, "std": 0.02},
            "assignment_cost_summary": {"count": 2, "min": 0.15, "max": 0.19, "mean": 0.17, "std": 0.02},
            "filtration_distortion_summary": {"count": 1, "min": 0.0, "max": 0.07, "mean": 0.07, "std": 0.0},
            "edge_preservation_rate": 0.25,
            "domain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "codomain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "displayed_domain_vertices": 2,
            "displayed_codomain_vertices": 2,
            "is_simplicial_on_displayed_skeleton": False,
            "preserved_edge_pairs": [{"query_edge": ["a", "b"], "memory_edge": ["x", "y"]}],
            "failed_edge_pairs": [{"query_edge": ["b", "c"], "memory_edge": ["y", "z"]}],
            "preserved_edge_query_vertices": ["a", "b"],
        },
        {
            "query_complex_source": "trajectory_probability_filtered_simplicial_object",
            "codomain_complex_source": "trajectory_probability_filtered_simplicial_object",
            "map_source": "model_probability_jensen_shannon_assignment",
            "probability_vector_evidence": {"schema_version": "tropicalgt.probability_vector_assignment_evidence.v1", "source": "probability_filtered_complex_vertices", "probability_vector_source": "model_probability_vectors_on_vertices", "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors", "assignment_solver": "scipy_linear_sum_assignment", "probability_alignment": "zero_pad_to_common_token_index_feature_space_then_renormalize", "displayed_query_vertices": 2, "displayed_memory_vertices": 2, "query_probability_vertex_count": 2, "memory_probability_vertex_count": 2, "all_displayed_query_vertices_have_probability_vectors": True, "all_displayed_memory_vertices_have_probability_vectors": True, "embedding_only_assignment_used": False, "no_proxy_or_fallback": True},
            "correspondence_table_rows": [
                {"row_index": 0, "query_vertex": "a", "memory_vertex": "x", "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors", "assignment_solver": "scipy_linear_sum_assignment", "jensen_shannon_distance": 0.1, "assignment_cost": 0.15, "embedding_only_assignment_used": False, "rendered_as_map_edge": False, "no_proxy_or_fallback": True},
                {"row_index": 1, "query_vertex": "b", "memory_vertex": "y", "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors", "assignment_solver": "scipy_linear_sum_assignment", "jensen_shannon_distance": 0.14, "assignment_cost": 0.19, "embedding_only_assignment_used": False, "rendered_as_map_edge": False, "no_proxy_or_fallback": True},
            ],
            "layout_contract": {
                "schema_version": "tropicalgt.analogical_map_layout.v1",
                "default_view": "side_by_side_query_codomain_small_multiples_plus_correspondence_table",
                "query_panel": "query trajectory probability complex",
                "codomain_panel": "retrieved memory probability complex",
                "visual_layers_separated": ["query_vertices_edges_faces", "codomain_vertices_edges_faces", "probability_js_vertex_correspondence_edges", "preserved_simplex_evidence", "failed_simplex_evidence", "filtration_distortion_diagnostics", "certificate_quality_table"],
                "correspondence_table_rows": 2,
                "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors",
                "assignment_evidence_available": True,
                "embedding_only_assignment_allowed": False,
                "fail_closed_when_probability_vectors_missing": True,
                "draws_pseudo_map_when_unavailable": False,
                "safe_to_draw_simplicial_map_edges": False,
                "safe_to_draw_chain_map_or_module_morphism": False,
                "map_render_claim": "probability_correspondence_not_a_simplicial_map",
                "map_claim_failure_reason": "simplex_tree_map_not_fully_preserved",
                "checked_edges": 2,
                "preserved_edges": 1,
                "checked_two_simplices": 0,
                "preserved_two_simplices": 0,
                "no_proxy_or_fallback": True,
            },
            "is_identity_self_map": False,
            "pair_page": "analogical_memory_map_02.html",
            "jensen_shannon_distance_mean": 0.23,
            "assignment_cost_mean": 0.27,
            "jensen_shannon_distance_summary": {"count": 2, "min": 0.2, "max": 0.26, "mean": 0.23, "std": 0.03},
            "assignment_cost_summary": {"count": 2, "min": 0.24, "max": 0.3, "mean": 0.27, "std": 0.03},
            "filtration_distortion_summary": {"count": 1, "min": 0.0, "max": 0.02, "mean": 0.02, "std": 0.0},
            "edge_preservation_rate": 0.5,
            "domain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "codomain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "displayed_domain_vertices": 2,
            "displayed_codomain_vertices": 2,
            "is_simplicial_on_displayed_skeleton": False,
            "preserved_edge_pairs": [{"query_edge": ["a", "b"], "memory_edge": ["x", "y"]}],
            "failed_edge_pairs": [],
            "preserved_edge_query_vertices": ["a", "b"],
        },
    ]}))
    analogical_simplex_tree_payload = {
        "contract": {
            "schema_version": "tropicalgt.analogical_simplex_tree_analogy.v1",
            "available": True,
            "status": "available",
            "reason_detail": "",
            "source": "probability_simplicial_map.simplex_tree_map.rows",
            "simplex_tree_source": "finite GUDHI SimplexTree enumeration from trajectory_probability_filtered_simplicial_object pairs",
            "no_proxy_or_fallback": True,
            "compares_query_and_memory_simplex_trees": True,
            "renders_hasse_face_to_coface_rows": True,
            "preserved_face_coface_chains_highlighted": True,
            "failed_or_distorted_chains_labeled_not_maps": True,
            "chain_map_claim_requires_certified_filtered_simplicial_map": True,
            "persistence_module_morphism_claim_requires_certified_filtered_simplicial_map": True,
            "pair_count": 2,
            "total_checked_simplices": 4,
            "total_preserved_simplices": 2,
            "topk_contract_schema": "tropicalgt.analogical_topk.v1",
        },
        "pairs": [
            {
                "rank": 1,
                "pair_page": "analogical_memory_retrieval.html",
                "memory_id": "mem-1",
                "map_render_claim": "probability_correspondence_not_a_simplicial_map",
                "checked_simplices": 2,
                "preserved_simplices": 1,
                "preservation_rate": 0.5,
                "dimension_counts": {"dim_0": {"checked": 1, "preserved": 1, "missing_codomain": 0}, "dim_1": {"checked": 1, "preserved": 0, "missing_codomain": 1}},
                "positive_filtration_distortion_summary": {"count": 1, "max": 0.07},
                "simplex_rows_truncated": False,
                "simplex_rows": [
                    {"domain_simplex": ["a"], "image_simplex": ["x"], "dimension": 0, "domain_filtration": 0.0, "codomain_filtration": 0.0, "signed_filtration_distortion": 0.0, "preserved_in_simplex_tree": True, "failure_reason": None},
                    {"domain_simplex": ["a", "b"], "image_simplex": ["x", "y"], "dimension": 1, "domain_filtration": 0.2, "codomain_filtration": None, "signed_filtration_distortion": None, "preserved_in_simplex_tree": False, "failure_reason": "missing_codomain_simplex"},
                ],
                "preserved_face_coface_chains": [
                    {"domain_coface": ["a", "b"], "image_coface": ["x", "y"], "dimension": 1, "coface_preserved_in_simplex_tree": False, "all_boundary_faces_present_and_preserved": False, "boundary_faces": []}
                ],
            },
            {
                "rank": 2,
                "pair_page": "analogical_memory_map_02.html",
                "memory_id": "mem-2",
                "map_render_claim": "probability_correspondence_not_a_simplicial_map",
                "checked_simplices": 2,
                "preserved_simplices": 1,
                "preservation_rate": 0.5,
                "dimension_counts": {"dim_0": {"checked": 1, "preserved": 1, "missing_codomain": 0}, "dim_1": {"checked": 1, "preserved": 0, "missing_codomain": 0}},
                "positive_filtration_distortion_summary": {"count": 1, "max": 0.02},
                "simplex_rows_truncated": False,
                "simplex_rows": [
                    {"domain_simplex": ["a"], "image_simplex": ["x"], "dimension": 0, "domain_filtration": 0.0, "codomain_filtration": 0.0, "signed_filtration_distortion": 0.0, "preserved_in_simplex_tree": True, "failure_reason": None},
                    {"domain_simplex": ["a", "b"], "image_simplex": ["x", "y"], "dimension": 1, "domain_filtration": 0.2, "codomain_filtration": 0.22, "signed_filtration_distortion": 0.02, "preserved_in_simplex_tree": False, "failure_reason": "filtration_not_preserved"},
                ],
                "preserved_face_coface_chains": [
                    {"domain_coface": ["a", "b"], "image_coface": ["x", "y"], "dimension": 1, "coface_preserved_in_simplex_tree": False, "all_boundary_faces_present_and_preserved": True, "boundary_faces": [{"domain_face": ["a"], "image_face": ["x"], "preserved_in_simplex_tree": True, "missing_from_certificate": False}]}
                ],
            },
        ],
    }
    _write(row / "analogical_simplex_tree_analogy.json", json.dumps(analogical_simplex_tree_payload))
    _write(row / "analogical_simplex_tree_analogy.html", _html("Analogical simplex-tree analogy", "finite simplex-tree rows preserved face-to-coface chains no proxy"))
    _write(
        row / "tropical_fan_diagnostics.json",
        json.dumps(
            {
                "schema_version": "tropicalgt.tropical_fan_visual_audit.v1",
                "available": False,
                "source_path": "unavailable",
                "ideal_spec": None,
                "safe_to_render_as_tropical_fan": False,
                "render_contract": "Tropical fan diagnostics render one dimensional cones only from explicit model-derived ideal specs and real Macaulay2 Tropical certificates; unavailable states are not substituted by support-token proxies.",
                "diagnostics": {
                    "schema_version": "tropicalgt.cas_tropical_fan.v1",
                    "available": False,
                    "status": "unavailable_no_model_derived_tropical_ideal",
                    "reason": "fixture has no explicit model-derived tropical ideal",
                    "backend": "Macaulay2",
                    "certificate_attached": False,
                    "fan_diagnostics_certified": False,
                    "safe_to_render_as_tropical_fan": False,
                    "cas_artifacts": {},
                },
            }
        ),
    )
    support_render_contract = {
        "schema_version": "tropicalgt.tropical_support_render.v1",
        "source_trace": "graph_token_trace.tokens",
        "support_index_source": "token.active_support_index emitted by the model tropical attention trace",
        "support_columns_policy": "observed_valid_active_support_indices_only",
        "assignment_matrix_semantics": "binary argmax support-selection mask; zeros are unselected cells, not zero margins",
        "assignment_matrix_binary": True,
        "selected_margin_matrix_policy": "finite model tropical margins are stored only on selected observed support cells; unselected cells are null",
        "legacy_margin_matrix_policy": "zero-filled display matrix retained for backward compatibility; use selected_margin_matrix for margin evidence",
        "support_flow_edge_policy": "one edge per query token; invalid active_support_index rows are marked invalid and no support column is fabricated",
        "normal_fan_wall_crossing_certified": False,
        "wall_margin_metric_scope": "margin_threshold_audit_not_certified_normal_fan_wall_crossing",
        "support_probability_source": "model_tropical_support_probabilities",
        "observed_support_count": 1,
        "token_count": 4,
        "valid_support_assignment_count": 4,
        "invalid_support_count": 0,
        "invalid_support_rows": [],
        "assignment_matrix_shape": [4, 1],
        "no_proxy_or_fallback": True,
    }
    support_readability_contract = {
        "schema_version": "tropicalgt.tropical_support_readability.v1",
        "source_trace": "graph_token_trace.tokens",
        "layout_mode": "collapse_diagnostic",
        "no_proxy_or_fallback": True,
        "panels_are_separate": True,
        "panel_roles": [
            "observed_support_assignment_matrix",
            "selected_margin_profile",
            "wall_margin_threshold_overlays",
            "token_group_summary",
            "top_support_collapse_diagnostic",
            "margin_distribution",
            "collapse_metrics_table",
        ],
        "required_panel_roles": [
            "observed_support_assignment_matrix",
            "selected_margin_profile",
            "wall_margin_threshold_overlays",
            "token_group_summary",
        ],
        "assignment_and_margin_panels_separated": True,
        "support_strip_split_from_margin_profile": True,
        "model_probability_summaries_separate_from_assignment_matrix": True,
        "compact_tick_labels": True,
        "full_token_text_preserved_in_hover_and_payload": True,
        "exact_token_indices_preserved_in_payload": True,
        "group_summaries_from_trace_fields": True,
        "collapse_diagnostic_visible": True,
        "collapse_diagnostic_available": True,
        "invalid_active_support_indices_not_fabricated": True,
        "normal_fan_wall_crossing_certified": False,
    }
    _write(
        row / "tropical_support_payload.json",
        json.dumps(
            {
                "tropical_support_render_contract": support_render_contract,
                "tropical_support_readability_contract": support_readability_contract,
                "metrics": {
                    "available": True,
                    "token_count": 4,
                    "unique_support_count": 1,
                    "effective_supports": 1.0,
                    "support_entropy_bits": 0.0,
                    "top_support_collapse_rate": 1.0,
                    "margin_summary": {"min": 0.0004, "max": 0.4, "mean": 0.13, "std": 0.1, "p05": 0.0004, "p50": 0.05, "p95": 0.4},
                    "wall_margin_audit": {"wall_margin_threshold": 0.001, "near_wall_margin_threshold": 0.01, "strict_wall_hit_count": 1, "near_wall_hit_count": 2, "near_wall_only_count": 1, "strict_wall_hit_rate": 0.25, "near_wall_hit_rate": 0.5, "near_wall_only_rate": 0.25, "metric_scope": "margin_threshold_audit_not_certified_normal_fan_wall_crossing", "strict_definition": "strict_wall_hit_rate counts finite selected-support margins with margin <= wall_margin_threshold.", "near_wall_definition": "near_wall_hit_rate counts finite selected-support margins with margin <= near_wall_margin_threshold; near_wall_only counts near-wall hits outside the strict band.", "low_strict_wall_interpretation_status": "strict_wall_margin_events_observed", "low_strict_wall_interpretation": "At least one selected-support margin is within the configured strict threshold; inspect the per-token wall buckets for the actual observed margin events.", "metric_issue": False, "definition": "strict_wall_hit_rate counts margin <= wall_margin_threshold; near_wall_hit_rate counts margin <= near_wall_margin_threshold. This is a model tropical-margin threshold audit, not a certified normal-fan wall-crossing count."},
                    "strict_wall_hit_rate": 0.25,
                    "near_wall_hit_rate": 0.5,
                    "near_wall_only_rate": 0.25,
                    "wall_margin_threshold": 0.001,
                    "near_wall_margin_threshold": 0.01,
                    "active_support_probability_summary": {"available": True, "count": 4, "min": 0.7, "max": 0.9, "mean": 0.8, "p05": 0.7, "p50": 0.8, "p95": 0.9},
                    "support_probability_entropy_bits_summary": {"available": True, "count": 4, "min": 0.2, "max": 0.6, "mean": 0.4, "p05": 0.2, "p50": 0.4, "p95": 0.6},
                    "support_probability_source": "model_tropical_support_probabilities",
                    "valid_support_assignment_count": 4,
                    "invalid_support_count": 0,
                    "invalid_support_rows": [],
                    "support_columns_policy": "observed_valid_active_support_indices_only",
                    "render_contract_schema_version": "tropicalgt.tropical_support_render.v1",
                    "readability_contract_schema_version": "tropicalgt.tropical_support_readability.v1",
                    "readability_panel_roles": support_readability_contract["panel_roles"],
                    "layout_mode": "collapse_diagnostic",
                    "normal_fan_wall_crossing_certified": False,
                    "no_proxy_or_fallback": True,
                    "render_contract": "assignment_matrix is binary model argmax support; selected_margin_matrix is model tropical margin only on selected cells; probability summaries come from model_tropical_support_probabilities and are not fabricated scores",
                    "interpretation": "Uniform blocks indicate true active-support collapse or nearly constant margins. No support-token proxies are introduced for invalid active_support_index rows.",
                },
                "support_assignment_status_by_token": [
                    {"query_index": idx, "query_label": f"q{idx}", "active_support_index": 0, "status": "selected_observed_support", "rendered_as_assignment_cell": True, "reason": None}
                    for idx in range(4)
                ],
                "support_flow_edges": [
                    {"query_index": idx, "query_label": f"q{idx}", "support_index": 0, "support_label": "q0", "margin": [0.0004, 0.004, 0.1, 0.4][idx], "active_support_probability": 0.8, "support_probability_entropy_bits": 0.4, "support_probability_source": "model_tropical_support_probabilities", "top_model_support_probabilities": [{"index": 0, "probability": 0.8}], "strict_wall_hit": idx == 0, "near_wall_hit": idx in (0, 1), "wall_margin_bucket": ["strict_wall", "near_wall", "interior", "interior"][idx], "wall_margin_threshold": 0.001, "near_wall_margin_threshold": 0.01, "support_assignment_status": "selected_observed_support", "rendered_as_assignment_cell": True, "no_proxy_or_fallback": True}
                    for idx in range(4)
                ],
            }
        ),
    )
    _write(
        row / "graphcg_direction_cosines_payload.json",
        json.dumps(
            {
                "available": True,
                "matrix_shape": [4, 8],
                "display_count": 4,
                "full_rank_direction_count": 8,
                "active_rank_nonzero_mean_abs": 8,
                "candidate_effective_direction_count": [4.0, 4.0, 4.0, 4.0],
                "direction_activity_sorted": [0.2 for _ in range(8)],
                "candidate_hover_rows": [f"candidate {idx} path action text" for idx in range(4)],
                "direction_rows": [
                    {
                        "direction_id": idx,
                        "display_column": idx,
                        "source": "candidate.graphcg_projection.all_direction_cosines",
                        "mean_abs_cosine": 0.2,
                        "signed_mean_cosine": 0.05,
                        "activity_rank_desc": idx + 1,
                        "rendered_in_all_direction_heatmap": True,
                        "rendered_in_full_rank_activity_spectrum": True,
                        "rendered_in_signed_bias_panel": True,
                        "rendered_in_top_active_direction_panel": idx < 4,
                        "exact_direction_id_preserved": True,
                        "no_proxy_or_fallback": True,
                    }
                    for idx in range(8)
                ],
                "top_active_direction_rows": [
                    {
                        "rank": idx + 1,
                        "direction_id": idx,
                        "label": f"d{idx}",
                        "mean_abs_cosine": 0.2,
                        "signed_mean_cosine": 0.05,
                        "source": "candidate.graphcg_projection.all_direction_cosines",
                        "rendered_in_top_active_direction_panel": True,
                        "exact_direction_id_preserved": True,
                        "no_proxy_or_fallback": True,
                    }
                    for idx in range(4)
                ],
                "panel_names": [
                    "all_direction_heatmap",
                    "top_active_direction_panel",
                    "full_rank_activity_spectrum",
                    "candidate_activity_by_observed_got_state",
                    "direction_signed_bias",
                ],
                "top_active_direction_panel_available": True,
                "graphcg_direction_evidence_contract": {
                    "schema_version": "tropicalgt.graphcg_direction_evidence.v1",
                    "source": "candidate.graphcg_projection.all_direction_cosines",
                    "no_proxy_or_fallback": True,
                    "all_model_directions_have_rows": True,
                    "direction_count": 8,
                    "direction_row_count": 8,
                    "exact_direction_ids_preserved": True,
                    "all_directions_rendered_in_heatmap": True,
                    "all_directions_rendered_in_activity_spectrum": True,
                    "all_directions_rendered_in_signed_bias_panel": True,
                    "top_active_direction_panel_count": 4,
                    "top_active_direction_panel_source": "top directions by mean_abs_cosine over observed candidate GraphCG projections",
                    "mean_abs_source": "mean absolute cosine over observed candidate GraphCG projections",
                    "signed_mean_source": "signed mean cosine over observed candidate GraphCG projections",
                    "activity_rank_source": "descending order of mean_abs_cosine across every model-derived direction",
                    "safe_to_render_full_rank_direction_evidence": True,
                },
                "graphcg_readability_contract": {
                    "schema_version": "tropicalgt.graphcg_direction_readability.v1",
                    "source": "candidate.graphcg_projection",
                    "no_proxy_or_fallback": True,
                    "all_model_directions_rendered": True,
                    "directions_sampled_for_heatmap": False,
                    "panels_are_separate": True,
                    "required_panels": [
                        "all_direction_heatmap",
                        "top_active_direction_panel",
                        "full_rank_activity_spectrum",
                        "candidate_activity_by_observed_got_state",
                        "direction_signed_bias",
                    ],
                    "exact_direction_ids_preserved_in_hover_and_payload": True,
                    "candidate_path_action_text_preserved_in_hover_and_payload": True,
                    "hover_fields": ["path", "direction", "signed cosine"],
                },
                "projection_basis_certificate": {
                    "source": "candidate.graphcg_projection",
                    "available": True,
                    "projection_basis": "effective_full_rank_qr",
                    "basis_sources": ["effective_full_rank_qr"],
                    "basis_source_counts": {"effective_full_rank_qr": 4},
                    "candidate_count": 4,
                    "direction_count": 8,
                    "all_candidates_have_all_direction_cosines": True,
                    "mean_abs_offdiag_cosine_values": [0.01, 0.01, 0.01, 0.01],
                    "max_abs_offdiag_cosine_values": [0.02, 0.02, 0.02, 0.02],
                    "mean_abs_offdiag_cosine_max": 0.01,
                    "max_abs_offdiag_cosine_max": 0.02,
                },
                "interpretation": "Heatmap colors encode absolute cosine activity; signed cosine values are preserved in hover.",
            }
        ),
    )
    _write(
        row / "trajectory_level_radius_bifiltration.json",
        json.dumps(
            {
                "available": True,
                "coefficient_ring": "F2[x_level,x_radius]",
                "num_parameters": 2,
                "parameters": [
                    {"name": "trajectory_level", "meaning": "reasoning growth level"},
                    {"name": "radius", "meaning": "scalar radius threshold"},
                ],
                "grid_axes": [[0, 1, 2], [0, 1, 2]],
                "levels": [0, 1, 2],
                "radii": [0.0, 0.5, 1.0],
                "radius_grade_policy": "exact_sorted_radius_grid_index_no_bucket_collision",
                "radius_grade_values": {"0": 0.0, "1": 0.5, "2": 1.0},
                "rank_invariant_samples": [
                    {"source_grade": [0, 0], "target_grade": [0, 1], "h0_rank": 1},
                    {"source_grade": [0, 0], "target_grade": [1, 1], "h0_rank": 1},
                ],
                "structure_maps": [
                    {"source_grade": [0, 0], "target_grade": [1, 0], "direction": "x_level", "field": "F2", "homology_rank": {"0": 1, "1": 0}, "method": "rank(B_target + image(Z_source)) over F2"},
                    {"source_grade": [0, 0], "target_grade": [0, 1], "direction": "x_radius", "field": "F2", "homology_rank": {"0": 1, "1": 0}, "method": "rank(B_target + image(Z_source)) over F2"},
                ],
                "boundary_monomials": {"d1": [{"monomial": "x_level", "monomial_exponent": [1, 0]}]},
                "chain_presentation_diagnostics": {
                    "ring": "F2[x_level,x_radius]",
                    "real_free_resolution_certified": False,
                    "resolution_status": "computed finite presentation; minimality not certified",
                    "real_free_resolution": {
                        "schema_version": "tropicalgt.real_free_resolution.v1",
                        "available": False,
                        "status": "backend_not_installed",
                        "reason": "Macaulay2, Sage, and Singular unavailable in validator fixture.",
                        "coefficient_ring": "F2[x_level,x_radius]",
                        "module_schema_version": "tropicalgt.level_radius_module.v1",
                        "input_sha256": "fixture-cas-input",
                        "module_summary": {"coefficient_ring": "F2[x_level,x_radius]", "variables": ["x_level", "x_radius"], "generators": 2, "boundary_monomials": 1, "presentation_shape": [1, 1]},
                        "backend_attempts": [],
                        "backend_probe": {"backends": [], "preferred_order": ["M2", "sage", "Singular"]},
                        "bemultipliers_probe": {"is_resolution_backend": False, "execution_policy": "never substitute BEMultipliers for a free-resolution certificate"},
                        "certificate_contract": {
                            "schema_version": "tropicalgt.cas_free_resolution_contract.v1",
                            "required_input_schema": "tropicalgt.level_radius_module.v1",
                            "no_proxy_or_fallback": True,
                            "no_proxy_policy": "Only backend-emitted exactness certificates may render as a free resolution; finite chain diagnostics cannot substitute.",
                            "paper_method_contract": {
                                "schema_version": "tropicalgt.be_fitting_method_contract.v1",
                                "paper_reference": "references/2210.11433v1.pdf",
                                "arxiv_id": "2210.11433v1",
                                "no_proxy_or_fallback": True,
                            },
                        },
                        "paper_method_contract": {
                            "schema_version": "tropicalgt.be_fitting_method_contract.v1",
                            "paper_reference": "references/2210.11433v1.pdf",
                            "arxiv_id": "2210.11433v1",
                            "no_proxy_or_fallback": True,
                        },
                        "cas_artifacts": {},
                        "command_templates": {"macaulay2": "-- fixture", "sage": "# fixture", "singular": "// fixture"},
                        "cas_execution_manifest": {
                            "schema_version": "tropicalgt.cas_execution_manifest.v1",
                            "coefficient_ring": "F2[x_level,x_radius]",
                            "backend_order": ["M2", "sage", "Singular"],
                            "backend_entries": [
                                {"name": "M2", "template_key": "macaulay2", "template_available": True, "certificate_required_before_rendering": True},
                                {"name": "sage", "template_key": "sage", "template_available": True, "certificate_required_before_rendering": True},
                                {"name": "Singular", "template_key": "singular", "template_available": True, "certificate_required_before_rendering": True},
                            ],
                            "no_proxy_or_fallback": True,
                            "render_rule": "Render a free resolution only after returned certificate flags are true.",
                        },
                        "certificate_attached": False,
                        "real_free_resolution_certified": False,
                        "total_graded_resolution_certified": False,
                        "ungraded_resolution_certified": False,
                        "multigraded_free_resolution_certified": False,
                        "exactness_certified": False,
                        "minimality_certified": False,
                        "safe_to_render_as_real_free_resolution": False,
                        "safe_to_render_as_total_graded_resolution": False,
                        "safe_to_render_as_multigraded_free_resolution": False,
                        "safe_unavailable_render": True,
                        "unavailable_diagnostic": {
                            "available": True,
                            "status": "backend_not_installed",
                            "reason": "fixture unavailable state",
                            "safe_to_render_only_as_unavailable": True,
                            "no_proxy_policy": "Do not substitute chain diagnostics, rank samples, Fitting ideals, minors, or BEMultipliers output for a certified free resolution.",
                        },
                        "unavailable_dependency_action": "Install a real CAS backend before rendering a free resolution.",
                        "render_warning": "No certified CAS free resolution is available for this module.",
                    },
                },
            }
        ),
    )
    _write(
        row / "trajectory_persistence" / "two_parameter_bifiltration.json",
        json.dumps(
            {
                "schema_version": "tropicalgt.two_parameter_bifiltration_visual.v1",
                "coefficient_ring": "F2[x_level,x_radius]",
                "primary_view": "miller_sturmfels_bivariate_staircase",
                "rank_surface_primary": False,
                "axes": {
                    "horizontal": "x_radius",
                    "vertical": "x_level",
                    "coordinate_one_dimensional_cones": ["rho_x_radius", "rho_x_level"],
                },
                "actual_data_only": True,
                "no_proxy_resolution_claim": True,
                "structure_map_summary": {
                    "schema_version": "tropicalgt.two_parameter_structure_maps.v1",
                    "source": "bifiltration.structure_maps",
                    "source_grade_convention": "[x_level_exponent, x_radius_exponent]",
                    "coefficient_ring": "F2[x_level,x_radius]",
                    "actual_adjacent_map_count": 2,
                    "valid_grade_edge_count": 2,
                    "direction_counts": {"x_level": 1, "x_radius": 1},
                    "field": "F2",
                    "field_counts": {"F2": 2},
                    "homology_dimensions_observed": [0, 1],
                    "east_north_structure_maps_present": True,
                    "module_lattice_overlay_trace_names": ["actual x_level structure maps over F2", "actual x_radius structure maps over F2"],
                    "module_lattice_overlay_available": True,
                    "rank_rows": [
                        {"source_bidegree_x_level_x_radius": [0, 0], "target_bidegree_x_level_x_radius": [1, 0], "source_monomial": "x_level^0 x_radius^0", "target_monomial": "x_level^1 x_radius^0", "direction": "x_level", "field": "F2", "homology_rank": {"0": 1, "1": 0}, "method": "rank(B_target + image(Z_source)) over F2"},
                        {"source_bidegree_x_level_x_radius": [0, 0], "target_bidegree_x_level_x_radius": [0, 1], "source_monomial": "x_level^0 x_radius^0", "target_monomial": "x_level^0 x_radius^1", "direction": "x_radius", "field": "F2", "homology_rank": {"0": 1, "1": 0}, "method": "rank(B_target + image(Z_source)) over F2"},
                    ],
                    "no_proxy_or_fallback": True,
                },
                "primary_structure_map_evidence": {
                    "schema_version": "tropicalgt.primary_structure_map_evidence.v1",
                    "available": True,
                    "source": "bifiltration.structure_maps",
                    "directions_rendered": ["x_level", "x_radius"],
                    "primary_table_rows": 2,
                    "module_lattice_overlay_trace_names": ["actual x_level structure maps over F2", "actual x_radius structure maps over F2"],
                    "no_proxy_or_fallback": True,
                },
                "certificate_indexed_cas_evidence": {
                    "schema_version": "tropicalgt.cas_certificate_indexed_evidence.v1",
                    "available": False,
                    "reason": "fixture_cas_backend_unavailable",
                    "safe_unavailable_render": True,
                    "no_proxy_or_fallback": True,
                    "render_rule": "No certificate-indexed CAS evidence is displayed unless a certified CAS result emitted the evidence block.",
                },
                "miller_sturmfels_staircase_evidence": {
                    "schema_version": "tropicalgt.miller_sturmfels_staircase_evidence.v1",
                    "coefficient_ring": "F2[x_level,x_radius]",
                    "source": "staircase_cards_from_bifiltration.chain_module_generators[*].multidegree",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "primary_view": "miller_sturmfels_bivariate_staircase",
                    "axes": {"horizontal": "x_radius", "vertical": "x_level", "coordinate_one_dimensional_cones": ["rho_x_radius", "rho_x_level"]},
                    "card_count": 1,
                    "primary_card_count": 1,
                    "primary_card_index": 0,
                    "primary_homological_degree": 1,
                    "total_actual_generator_bidegree_count": 2,
                    "total_minimal_antichain_count": 2,
                    "total_generator_label_count": 2,
                    "total_upward_closed_region_count": 2,
                    "total_quotient_basis_lattice_count": 3,
                    "total_hilbert_numerator_term_count": 1,
                    "total_adjacent_lcm_syzygy_count": 1,
                    "cards_with_quotient_basis_count": 1,
                    "cards_with_hilbert_numerator_terms_count": 1,
                    "cards_with_adjacent_lcm_syzygies_count": 1,
                    "per_card_counts": [
                        {
                            "homological_degree": 1,
                            "primary_card": True,
                            "actual_generator_bidegree_count": 2,
                            "minimal_antichain_count": 2,
                            "generator_label_count": 2,
                            "upward_closed_region_count": 2,
                            "quotient_basis_lattice_count": 3,
                            "hilbert_numerator_term_count": 1,
                            "adjacent_lcm_syzygy_count": 1,
                            "theorem_scope": "exact two-variable monomial-ideal staircase resolution when adjacent-LCM theorem applies; not a full persistence-module free resolution without CAS certification",
                        }
                    ],
                    "all_cards_have_generator_labels": True,
                    "all_cards_have_upward_closed_regions": True,
                    "all_cards_have_quotient_basis_lattice_points": True,
                    "all_cards_have_hilbert_numerator_terms": True,
                    "all_cards_have_adjacent_lcm_syzygy_lists": True,
                    "quotient_basis_counts_match_lattice_points": True,
                    "theorem_scope_boundary_all_cards": True,
                    "coordinate_axes_are_one_dimensional_cones": True,
                    "safe_to_render_miller_sturmfels_staircase": True,
                },
                "staircase_cards": [
                    {
                        "schema_version": "tropicalgt.two_parameter_staircase_card.v1",
                        "homological_degree": 1,
                        "actual_generator_bidegrees_source": "bifiltration.chain_module_generators[*].multidegree grouped by homological_degree",
                        "x_radius_horizontal": True,
                        "x_level_vertical": True,
                        "shaded_regions_are_upward_closed_generated_submodules": True,
                        "white_points_are_displayed_quotient_basis_lattice_points": True,
                        "primary_card": True,
                        "actual_generator_bidegree_count": 2,
                        "minimal_antichain": [[2, 1], [1, 2]],
                        "generator_labels": [
                            {"label": "g1", "bidegree": [2, 1], "monomial": "x_level^2 x_radius^1", "homological_degree": 1},
                            {"label": "g2", "bidegree": [1, 2], "monomial": "x_level^1 x_radius^2", "homological_degree": 1},
                        ],
                        "dominated_generator_bidegrees": [],
                        "upward_closed_regions": [
                            {"generator_label": "g1", "generator_bidegree": [2, 1], "x_radius_min": 1, "x_level_min": 2, "x_radius_max_displayed": 4, "x_level_max_displayed": 4},
                            {"generator_label": "g2", "generator_bidegree": [1, 2], "x_radius_min": 2, "x_level_min": 1, "x_radius_max_displayed": 4, "x_level_max_displayed": 4},
                        ],
                        "quotient_basis_lattice_points": [[0, 0], [0, 1], [1, 0]],
                        "quotient_basis_lattice_count": 3,
                        "display_grid_extent": {"x_radius_max": 4, "x_level_max": 4, "coordinate_axes": ["rho_x_radius", "rho_x_level"]},
                        "hilbert_numerator_terms": [{"sign": 1, "bidegree": [2, 1], "monomial": "x_level^2*x_radius"}],
                        "adjacent_lcm_syzygies": [{"index": 0, "lcm_bidegree": [2, 2], "relation": "x_radius*g1 + x_level*g2"}],
                        "resolution_available": True,
                        "resolution_scope": "displayed_two_variable_staircase_monomial_ideal",
                        "theorem_scope": "exact two-variable monomial-ideal staircase resolution when adjacent-LCM theorem applies; not a full persistence-module free resolution without CAS certification",
                    }
                ],
            }
        ),
    )
    _write(row / "inference_audit.json", "{}")
    html_files = {
        "got_embedding_map_3d.html": _html(
            "Graph-of-thought embedding-space trajectory map actual graph_state PCA",
            "Plotly.newPlot simplicial-object-panel simplicial-object-plot selected-complex-graph hover-simplicial-card plotly_click",
        ),
        "got_trajectory_pca_3d.html": _html(
            "Graph-of-thought branching trajectory with raw NLL metadata",
            "Plotly.newPlot selected-complex-graph plotly_click open interactive reasoning-step complex page raw NLL PC3 marker geometry",
        ),
        "got_nll_density_cloud_pca_3d.html": _html("3D PCA NLL density cloud", "Gaussian cloud actual model GoT state anchors not a model state Plotly.newPlot plotly.min.js"),
        "got_full_trajectory_complex.html": _html("Full graph-of-thought trajectory filtered simplicial complex", "Plotly.newPlot play filtration min-to-max Filtration radius model input model output filtration backend= simplicial-object-plot selected-complex-graph plotly_click"),
        "got_full_trajectory_simplex_tree_3d.html": _html("Full graph-of-thought trajectory GUDHI SimplexTree face-coface poset", "Plotly.newPlot face-coface poset view not a literal trie layout actual face-to-coface covers optional sorted-label trie prefix links not disconnected simplex columns"),
        "got_full_trajectory_complex_jensen_shannon.html": _html("Full graph-of-thought trajectory probability filtered simplicial complex", "Plotly.newPlot Jensen-Shannon probability filtered simplicial complex"),
        "got_full_trajectory_simplex_tree_3d_jensen_shannon.html": _html("Full graph-of-thought trajectory probability SimplexTree", "Plotly.newPlot Jensen-Shannon probability SimplexTree actual face-to-coface covers optional sorted-label trie prefix links not disconnected simplex columns"),
        "reasoning_step_complex_maps/index.html": _html("Reasoning step filtered simplicial complex maps", "table"),
        "tropical_support_heatmap.html": _html("Tropical active support", "Plotly.newPlot observed supports only top-support collapse rate No support-token proxies tropical_support_render_contract tropical_support_readability_contract Collapse metrics"),
        "tropical_fan_diagnostics.html": _html("Tropical fan diagnostics unavailable", "Plotly.newPlot Macaulay2 one dimensional cones not a multigraded free-resolution"),
        "graphcg_direction_cosines.html": _html("GraphCG full-rank direction audit", "Plotly.newPlot Readable top-direction heatmap"),
        "analogical_memory_topk_index.html": "<!doctype html><title>Analogical top-k probability correspondences</title><body>Analogical top-k probability correspondences Index readability contract <a href='analogical_memory_retrieval.html'>rank 1</a> <a href='analogical_memory_map_02.html'>rank 2</a></body>",
        "analogical_memory_retrieval.html": _html("Analogical probability-matched correspondence filtered-complex certificate", "Plotly.newPlot query trajectory complex retrieved memory complex slider filters domain and codomain sliders vertex-only correspondences preserved 1-simplex map simplicial-object-plot selected-complex-graph plotly_click"),
        "analogical_memory_map_02.html": _html("Analogical probability-matched correspondence filtered-complex certificate", "Plotly.newPlot query trajectory complex retrieved memory complex slider filters domain and codomain sliders vertex-only correspondences preserved 1-simplex map simplicial-object-plot selected-complex-graph plotly_click"),
        "analogical_simplex_tree_analogy.html": _html("Analogical simplex-tree analogy", "Plotly.newPlot finite simplex-tree rows preserved face-to-coface chains no proxy"),
        "trajectory_persistence/persistence_barcode.html": _html("Trajectory persistence barcode", "Plotly.newPlot simplicial-object-plot selected-complex-graph plotly_click"),
        "trajectory_persistence/two_parameter_bifiltration.html": _html("Trajectory 2-parameter persistence over F2[x_level,x_radius]", "Plotly.newPlot 2-parameter module fibers Miller-Sturmfels staircase H0 fiber rank"),
        "trajectory_persistence/persistence_module_betti.html": _html("Trajectory persistence Betti", "Plotly.newPlot 2D matrix decorative 3D simplicial-object-plot selected-complex-graph plotly_click"),
        "trajectory_persistence/persistence_representations.html": _html("Trajectory GUDHI persistence vectorization", "Plotly.newPlot Fast train/eval features"),
        "trajectory_persistence/persistence_landscapes.html": _html("Trajectory Actual GUDHI persistence landscape functions", "Plotly.newPlot lambda_1(t) not norm-only summaries"),
    }
    for rel, content in html_files.items():
        _write(row / rel, content)
    landscape_payload = {
        "schema_version": "tropicalgt.persistence_landscape_visual_contract.v1",
        "available": True,
        "source": "topology.persistence_representations.methods[*].landscape",
        "landscape_backend": "gudhi.representations",
        "backend_provenance": {
            "available": True,
            "source_field": "topology.persistence_representations.backend",
            "backends": ["gudhi.representations"],
            "row_backend_field": "landscape_rows[*].backend",
            "values_source_field": "landscape_rows[*].values_source",
        },
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "not_nll_fitness_landscape": True,
        "not_norm_only_summary": True,
        "safe_to_render_actual_landscape_functions": True,
        "curve_trace_count": 2,
        "growth_row_count": 2,
        "rendered_growth_level_count": 2,
        "rendered_growth_levels": [0, 1],
        "homology_dimensions": [0, 1],
        "heatmap_available": True,
        "heatmap_source": "first available lambda_1(t) rows from actual landscape values",
        "landscape_rows": [
            {
                "level": 0,
                "homology_dimension": 0,
                "source": "topology.persistence_representations.methods[*].landscape",
                "backend": "gudhi.representations",
                "grid_source": "normalized_index_from_gudhi_vector_resolution",
                "values_source": "gudhi.representations.Landscape.vector",
                "layer_count": 1,
                "grid_count": 4,
                "vector_length": 4,
                "finite_value_count": 4,
                "nonzero_value_count": 2,
                "min_value": 0.0,
                "max_value": 0.4,
                "actual_gudhi_landscape_values": True,
                "not_norm_only_summary": True,
                "not_nll_fitness_landscape": True,
            },
            {
                "level": 1,
                "homology_dimension": 1,
                "source": "topology.persistence_representations.methods[*].landscape",
                "backend": "gudhi.representations",
                "grid_source": "normalized_index_from_gudhi_vector_resolution",
                "values_source": "gudhi.representations.Landscape.vector",
                "layer_count": 1,
                "grid_count": 4,
                "vector_length": 4,
                "finite_value_count": 4,
                "nonzero_value_count": 2,
                "min_value": 0.0,
                "max_value": 0.3,
                "actual_gudhi_landscape_values": True,
                "not_norm_only_summary": True,
                "not_nll_fitness_landscape": True,
            },
        ],
        "unavailable_reasons": [],
        "render_contract": "Persistence landscape pages render only actual GUDHI Landscape vectors/lambda_k rows from persistence_representations; unavailable states are explicit and are not replaced by NLL/fitness landscapes, zero vectors, norms, or proxy summaries.",
    }
    _write(row / "trajectory_persistence/persistence_landscapes.json", json.dumps(landscape_payload))
    _write(row / "got_full_trajectory_complex_slider_contract.json", json.dumps(_slider_contract("got_full_trajectory_complex.html")))
    _write(row / "got_full_trajectory_complex_jensen_shannon_slider_contract.json", json.dumps(_slider_contract("got_full_trajectory_complex_jensen_shannon.html")))
    _write(row / "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json", json.dumps(_simplex_tree_poset_contract("got_full_trajectory_simplex_tree_3d.html", displayed=7, cover_edges=10, root_edges=4)))
    _write(row / "got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json", json.dumps(_simplex_tree_poset_contract("got_full_trajectory_simplex_tree_3d_jensen_shannon.html", displayed=7, cover_edges=10, root_edges=4)))
    for step in steps:
        _write(row / "reasoning_step_complex_maps" / step["file"], _html("Reasoning step filtered simplicial complex map"))
        _write(row / "reasoning_step_complex_maps" / step["simplex_tree_file"], _html("Reasoning step GUDHI simplex tree", "Plotly.newPlot simplex-tree inclusion"))
        _write(
            row / "reasoning_step_complex_maps" / step["file"].replace(".html", "_slider_contract.json"),
            json.dumps(_slider_contract(step["file"], vertices=1, solid_edges=0, filled_faces=0)),
        )
        _write(
            row / "reasoning_step_complex_maps" / step["simplex_tree_poset_contract_file"],
            json.dumps(step["simplex_tree_poset_contract"]),
        )
    return row

def _browser_samples(audit: Path, sample_names: list[str]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for idx, name in enumerate(sample_names):
        sample_dir = audit if name == "." else audit / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write(sample_dir / "browser_index.html", "<!doctype html><a href='got_trajectory_pca_3d.html'>trajectory</a>")
        prefix = "" if name == "." else f"{name}/"
        samples.append(
            {
                "index": idx,
                "label": "root sample" if name == "." else name,
                "dir": name,
                "artifacts": [
                    {"src": f"{prefix}got_trajectory_pca_3d.html", "label": "Trajectory", "tag": "plot"},
                    {"src": f"{prefix}analogical_memory_topk_index.html", "label": "Analogical top-k", "tag": "index"},
                ],
            }
        )
    return samples


def _codex_browser_html(samples: list[dict[str, object]], *, omit_button_src: str | None = None) -> str:
    sample_sections = []
    for sample in samples:
        idx = str(sample["index"])
        sample_dir = str(sample["dir"])
        open_href = "browser_index.html" if sample_dir == "." else f"{sample_dir}/browser_index.html"
        buttons = []
        for artifact in sample["artifacts"]:
            src = str(artifact["src"])
            if src == omit_button_src:
                continue
            label = str(sample["label"]) + " / " + str(artifact["label"])
            buttons.append(
                f'<button class="artifact" data-sample="{html.escape(idx, quote=True)}" '
                f'data-src="{html.escape(src, quote=True)}" '
                f'data-label="{html.escape(label, quote=True)}">'
                f'<span>{html.escape(str(artifact["label"]))}</span></button>'
            )
        sample_sections.append(
            f'<section class="sample" data-sample="{html.escape(idx, quote=True)}">'
            f'<a class="open-sample" href="{html.escape(open_href, quote=True)}">open sample</a>'
            f'{"".join(buttons)}</section>'
        )
    first_src = str(samples[0]["artifacts"][0]["src"])
    payload = html.escape(json.dumps(samples), quote=True)
    return f'<!doctype html><body data-samples="{payload}">Sample-first audit {"".join(sample_sections)}<a id="open" href="{html.escape(first_src, quote=True)}">open full page</a><iframe src="{html.escape(first_src, quote=True)}"></iframe></body>'


def test_validate_audit_root_accepts_three_interactive_rows(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    _row(audit, ".")
    _row(audit, "example_01")
    _row(audit, "example_02")
    samples = _browser_samples(audit, [".", "example_01", "example_02"])
    _write(audit / "codex_browser_index.html", _codex_browser_html(samples))
    _write(tmp_path / "step_00000001" / "validation_report.json", json.dumps({"bpb": 1.5, "graph_bpb": 2.5, "invalid_graph_rate": 0.0}))
    report = validator.validate_audit_root(audit, min_rows=3, min_candidates=4, min_depth=2)
    assert report["ok"], report["errors"]
    assert report["rows_checked"] == 3
    assert report["validation_metrics"]["bpb"] == 1.5
    assert all(row["step_complex_maps"] == 4 for row in report["row_reports"])


def test_validate_audit_root_uses_real_periodic_sibling_rows_without_duplication(tmp_path: Path):
    validator = _load_validator()
    periodic = tmp_path / "run" / "periodic"
    current = periodic / "step_00005000" / "got_audit"
    previous = periodic / "step_00004750" / "got_audit"
    older = periodic / "step_00002500" / "got_audit"
    _row(current, ".")
    _row(previous, ".")
    _row(older, ".")
    _write(periodic / "step_00005000" / "validation_report.json", json.dumps({"bpb": 1.4, "graph_bpb": 2.4, "invalid_graph_rate": 0.0}))

    report = validator.validate_audit_root(current, min_rows=3, min_candidates=4, min_depth=2)

    assert report["ok"], report["errors"]
    assert report["rows_checked"] == 3
    coverage = report["row_coverage"]
    assert coverage["schema_version"] == "tropicalgt.interactive_audit_row_coverage.v1"
    assert coverage["actual_data_only"] is True
    assert coverage["no_proxy_or_fallback"] is True
    assert coverage["unique_row_paths"] is True
    assert coverage["satisfies_min_rows"] is True
    assert coverage["row_paths"] == [str(current.resolve()), str(previous.resolve()), str(older.resolve())]
    assert "Never duplicate rows" in coverage["row_source_policy"]



def test_validate_audit_root_rejects_chart_bundle_sidecar_proxy_claim(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    _write(
        row / "chart_bundle_transport_sidecar.json",
        json.dumps(
            {
                "schema_version": "tropicalgt.chart_bundle_transport_sidecar.v1",
                "available": True,
                "source_path": "test",
                "metadata": {
                    "schema_version": "tropicalgt.chart_bundle_transport_metadata.v1",
                    "available": True,
                    "chart_ids": ["chart_00"],
                    "overlap_pairs": [],
                    "overlap_triples": [],
                },
                "chart_ids": ["chart_00"],
                "overlap_pair_count": 0,
                "overlap_triple_count": 0,
                "monomial_transport_contract": {"actual_data_only": False, "no_proxy_or_fallback": False},
                "bundle_matroid_contract": {"actual_data_only": True, "no_proxy_or_fallback": True},
                "actual_data_only": True,
                "no_proxy_or_fallback": True,
                "safe_to_render_as_toric_embedding_certificate": True,
                "safe_to_render_as_tropical_variety_embedding": False,
                "safe_to_render_as_global_toric_variety_embedding": False,
                "safe_to_use_as_normal_fan_certificate": False,
                "render_contract": "chart bundle proxy",
            }
        ),
    )
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("chart-bundle" in err for err in report["errors"])


def test_validate_audit_root_rejects_vector_bundle_paper_ready_with_missing_groups(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    completeness = {
        "schema_version": "tropicalgt.vector_bundle_paper_sidecar_completeness.v1",
        "tier": "paper_ready",
        "paper_ready": True,
        "basic_telemetry_available": True,
        "required_groups": {
            "chart_and_monomial_transport_ids": True,
            "configured_toric_active_rows": True,
            "flat_incidence_diagnostics": True,
            "graphcg_toric_agreement": False,
            "transported_persistence_landscapes": True,
        },
        "missing_required_groups": ["graphcg_toric_agreement"],
        "unavailable_fields": [],
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
    }
    paper_sidecar = {
        "schema_version": "tropicalgt.vector_bundle_paper_sidecar.v1",
        "available": True,
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "chart_ids": ["chart_00"],
        "monomial_transport_ids": ["chart_00__to__chart_00"],
        "toric_active_rows": {"available": True, "actual_data_only": True, "no_proxy_or_fallback": True, "value": [{"row_id": "toric_row_00"}]},
        "toric_active_row_count": {"available": True, "value": 1},
        "one_dimensional_cone_filtration_flat_defects": {
            "available": True,
            "actual_data_only": True,
            "no_proxy_or_fallback": True,
            "coordinate_one_dimensional_cones": [{"name": "rho_chart"}],
            "metrics": {},
        },
        "graphcg_toric_agreement": {"available": True, "actual_data_only": True, "no_proxy_or_fallback": True},
        "transported_persistence_landscape_metrics": {"available": True, "actual_data_only": True, "no_proxy_or_fallback": True},
        "unavailable_fields": [],
        "completeness_tier": "paper_ready",
        "completeness_contract": completeness,
        "safe_to_use_as_vector_bundle_paper_ready_evidence": True,
        "safe_to_use_as_vector_bundle_theorem_certificate": False,
        "safe_to_use_as_toric_or_tropical_embedding_certificate": False,
        "paper_claim_scope": {
            "actual_tropical_toric_variety_constructed": False,
            "actual_tropical_scheme_constructed": False,
            "no_proxy_or_fallback": True,
            "monomial_transports": "regularizer telemetry only",
        },
        "render_contract": "paper telemetry only; no proxies",
    }
    _write(
        row / "chart_bundle_transport_sidecar.json",
        json.dumps(
            {
                "schema_version": "tropicalgt.chart_bundle_transport_sidecar.v1",
                "available": True,
                "source_path": "test",
                "metadata": {"schema_version": "tropicalgt.chart_bundle_transport_metadata.v1", "available": True, "chart_ids": ["chart_00"], "overlap_pairs": [], "overlap_triples": []},
                "chart_ids": ["chart_00"],
                "overlap_pair_count": 0,
                "overlap_triple_count": 0,
                "monomial_transport_contract": {"actual_data_only": True, "no_proxy_or_fallback": True},
                "bundle_matroid_contract": {"actual_data_only": True, "no_proxy_or_fallback": True},
                "vector_bundle_paper_sidecar": paper_sidecar,
                "actual_data_only": True,
                "no_proxy_or_fallback": True,
                "safe_to_render_as_toric_embedding_certificate": False,
                "safe_to_render_as_tropical_variety_embedding": False,
                "safe_to_render_as_global_toric_variety_embedding": False,
                "safe_to_use_as_normal_fan_certificate": False,
                "render_contract": "chart bundle sidecar; not a toric embedding; no proxies",
            }
        ),
    )
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("vector-bundle paper-ready evidence still has missing required groups" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_trajectory_overlay_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "got_full_trajectory_complex_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("trajectory_complex_overlay_contract")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("trajectory complex overlay contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_probability_overlay_metric_mismatch(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "got_full_trajectory_complex_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["trajectory_complex_overlay_contract"]["probability_view"]["distance_metric"] = "euclidean"
    payload["trajectory_complex_overlay_contract"]["probability_view"]["safe_to_render_overlay_semantics"] = False
    payload["trajectory_complex_overlay_contract"]["safe_to_render_probability_view"] = False
    payload["trajectory_complex_overlay_contract"]["safe_to_render_available_views"] = False
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("probability trajectory complex" in err and "distance metric" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_persistence_landscape_payload(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    (row / "trajectory_persistence" / "persistence_landscapes.json").unlink()
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("persistence landscapes payload" in err for err in report["errors"])



def test_validate_audit_root_accepts_no_finite_interval_persistence_landscape_unavailable_state(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    (row / "trajectory_topological_algebra.json").write_text(
        json.dumps(
            {
                "persistence": {
                    "backend": "gudhi",
                    "available": True,
                    "intervals": [
                        {"dimension": 0, "birth": 0.0, "death": None, "infinite": True},
                        {"dimension": 1, "birth": 0.0, "death": None, "infinite": True},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    payload_path = row / "trajectory_persistence" / "persistence_landscapes.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "available": False,
            "landscape_backend": "unavailable",
            "backend_provenance": {"available": False, "reason": "no finite persistence intervals", "source_field": "topology.persistence_representations.backend"},
            "not_norm_only_summary": False,
            "safe_to_render_actual_landscape_functions": False,
            "curve_trace_count": 0,
            "rendered_growth_level_count": 0,
            "homology_dimensions": [],
            "landscape_rows": [],
            "finite_persistence_interval_count": 0,
            "unavailable_state_verified_by_intervals": True,
            "unavailable_reasons": ["no_finite_persistence_intervals_for_gudhi_landscape"],
        }
    )
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert report["ok"], report["errors"]


def test_validate_audit_root_rejects_missing_persistence_landscape_backend_provenance(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "trajectory_persistence" / "persistence_landscapes.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("backend_provenance")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("backend provenance" in err for err in report["errors"])


def test_validate_audit_root_rejects_norm_only_persistence_landscape_payload(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "trajectory_persistence" / "persistence_landscapes.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["not_norm_only_summary"] = False
    payload["safe_to_render_actual_landscape_functions"] = False
    payload["curve_trace_count"] = 0
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("norm-only" in err or "no curve traces" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_bifiltration_structure_map_summary(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    visual_path = row / "trajectory_persistence" / "two_parameter_bifiltration.json"
    payload = json.loads(visual_path.read_text(encoding="utf-8"))
    payload.pop("structure_map_summary")
    visual_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("structure-map summary" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_bifiltration_real_cas_guard(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "trajectory_level_radius_bifiltration.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["chain_presentation_diagnostics"].pop("real_free_resolution")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("real free-resolution guard" in err for err in report["errors"])


def test_validate_audit_root_rejects_bifiltration_real_cas_proxy_flags(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "trajectory_level_radius_bifiltration.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    guard = payload["chain_presentation_diagnostics"]["real_free_resolution"]
    guard["safe_to_render_as_multigraded_free_resolution"] = True
    guard["cas_artifacts"] = {"fake": "proxy"}
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("unavailable CAS guard" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_miller_sturmfels_staircase_evidence(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    visual_path = row / "trajectory_persistence" / "two_parameter_bifiltration.json"
    payload = json.loads(visual_path.read_text(encoding="utf-8"))
    payload.pop("miller_sturmfels_staircase_evidence")
    visual_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("Miller-Sturmfels staircase evidence" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_certificate_indexed_cas_evidence(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    visual_path = row / "trajectory_persistence" / "two_parameter_bifiltration.json"
    payload = json.loads(visual_path.read_text(encoding="utf-8"))
    payload.pop("certificate_indexed_cas_evidence")
    visual_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("certificate-indexed CAS evidence" in err for err in report["errors"])


def test_validate_audit_root_rejects_unsafe_certificate_indexed_cas_evidence(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    visual_path = row / "trajectory_persistence" / "two_parameter_bifiltration.json"
    payload = json.loads(visual_path.read_text(encoding="utf-8"))
    payload["certificate_indexed_cas_evidence"]["no_proxy_or_fallback"] = False
    payload["certificate_indexed_cas_evidence"]["safe_unavailable_render"] = False
    visual_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("certificate-indexed CAS evidence" in err and ("proxy/fallback" in err or "safe-unavailable" in err) for err in report["errors"])


def test_validate_audit_root_rejects_miller_sturmfels_staircase_aggregate_mismatch(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    visual_path = row / "trajectory_persistence" / "two_parameter_bifiltration.json"
    payload = json.loads(visual_path.read_text(encoding="utf-8"))
    payload["miller_sturmfels_staircase_evidence"]["total_quotient_basis_lattice_count"] = 999
    visual_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("quotient-basis aggregate" in err for err in report["errors"])


def test_validate_audit_root_rejects_absolute_analogical_pair_pages(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    maps_path = row / "analogical_simplicial_maps.json"
    payload = json.loads(maps_path.read_text(encoding="utf-8"))
    payload["maps"][0]["pair_page"] = str(row / "analogical_memory_retrieval.html")
    maps_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("relative pair_page links" in err for err in report["errors"])


def test_validate_audit_root_rejects_analogical_probability_js_provenance_gaps(tmp_path: Path):
    validator = _load_validator()
    cases = [
        (
            "wrong_query_source",
            lambda row: row.__setitem__("query_complex_source", "trajectory_embedding_filtered_simplicial_object"),
            "query trajectory-level model-probability complexes",
        ),
        (
            "wrong_codomain_source",
            lambda row: row.__setitem__("codomain_complex_source", "trajectory_embedding_filtered_simplicial_object"),
            "codomain trajectory-level model-probability complexes",
        ),
        (
            "wrong_map_source",
            lambda row: row.__setitem__("map_source", "embedding_nearest_neighbor_assignment"),
            "derived from model probability vectors",
        ),
        (
            "missing_js_summary",
            lambda row: row.pop("jensen_shannon_distance_summary"),
            "missing Jensen-Shannon distance summaries",
        ),
        (
            "missing_probability_vector_evidence",
            lambda row: row.pop("probability_vector_evidence"),
            "probability-vector assignment evidence",
        ),
        (
            "embedding_only_probability_assignment",
            lambda row: row["probability_vector_evidence"].__setitem__("embedding_only_assignment_used", True),
            "embedding-only assignment",
        ),
        (
            "missing_layout_contract",
            lambda row: row.pop("layout_contract"),
            "lacks analogical map layout contract",
        ),
        (
            "embedding_only_correspondence_row",
            lambda row: row["correspondence_table_rows"][0].__setitem__("embedding_only_assignment_used", True),
            "correspondence row uses embedding-only assignment",
        ),
        (
            "missing_domain_simplex_tree_provenance",
            lambda row: row.__setitem__("domain_simplex_tree", {"backend": "networkx"}),
            "missing domain GUDHI SimplexTree provenance",
        ),
    ]
    for case_name, mutate, expected in cases:
        audit = tmp_path / case_name / "got_audit"
        row = _row(audit, ".")
        maps_path = row / "analogical_simplicial_maps.json"
        payload = json.loads(maps_path.read_text(encoding="utf-8"))
        mutate(payload["maps"][0])
        maps_path.write_text(json.dumps(payload), encoding="utf-8")
        _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
        report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
        assert not report["ok"], case_name
        assert any(expected in err for err in report["errors"]), (case_name, report["errors"])


def test_validate_audit_root_rejects_missing_tropical_support_readability_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "tropical_support_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("tropical_support_readability_contract")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("tropical support payload is missing readability contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_reasoning_step_complex_fingerprints(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    manifest_path = row / "reasoning_step_complex_maps" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["steps"][0].pop("step_complex_fingerprint")
    payload["contract"]["all_step_complex_fingerprints_present"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("fingerprint" in err and "reasoning-step" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_reasoning_step_source_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    manifest_path = row / "reasoning_step_complex_maps" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["steps"][0].pop("step_complex_source_contract")
    payload["contract"]["all_steps_have_source_contracts"] = False
    payload["contract"]["all_step_complex_source_contracts_safe"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("reasoning-step complex source contract" in err or "source contracts" in err for err in report["errors"])


def test_validate_audit_root_rejects_reasoning_step_source_proxy_claim(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    manifest_path = row / "reasoning_step_complex_maps" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["steps"][0]["step_complex_source_contract"]["uses_embedding_trajectory_map_as_proxy"] = True
    payload["steps"][0]["step_complex_source_contract"]["safe_to_render_as_step_complex"] = False
    payload["contract"]["all_step_complex_source_contracts_no_proxy"] = False
    payload["contract"]["all_step_complex_source_contracts_safe"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("embedding trajectory proxy" in err or "source contracts allow proxy" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_reasoning_step_slider_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    (row / "reasoning_step_complex_maps" / "reasoning_step_000_slider_contract.json").unlink()
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("radius slider contract" in err and "reasoning-step complex 0" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_reasoning_step_simplex_tree_poset_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    (row / "reasoning_step_complex_maps" / "reasoning_step_000_simplex_tree_simplex_tree_poset_contract.json").unlink()
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("simplex-tree poset contract" in err and "reasoning-step simplex tree 0" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_simplex_tree_readability_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    contract_path = row / "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload.pop("readability_contract")
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("simplex-tree readability contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_reasoning_step_slider_summary(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    manifest_path = row / "reasoning_step_complex_maps" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["steps"][0].pop("radius_slider_contract")
    payload["contract"]["all_steps_have_radius_slider_contracts"] = False
    payload["contract"]["all_step_radius_sliders_safe_to_render"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("reasoning-step radius slider summary" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_analogical_simplex_tree_analogy_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "analogical_simplex_tree_analogy.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("contract")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("analogical simplex-tree analogy contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_analogical_topk_readability_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    maps_path = row / "analogical_simplicial_maps.json"
    payload = json.loads(maps_path.read_text(encoding="utf-8"))
    payload["topk_contract"].pop("readability_contract")
    maps_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("analogical top-k readability contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_graphcg_basis_certificate(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "graphcg_direction_cosines_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("projection_basis_certificate")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("projection-basis certificate" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_graphcg_direction_evidence_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "graphcg_direction_cosines_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("graphcg_direction_evidence_contract")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("per-direction evidence contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_graphcg_direction_row_gap(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "graphcg_direction_cosines_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["direction_rows"] = payload["direction_rows"][:-1]
    payload["graphcg_direction_evidence_contract"]["all_model_directions_have_rows"] = False
    payload["graphcg_direction_evidence_contract"]["direction_row_count"] = 7
    payload["graphcg_direction_evidence_contract"]["safe_to_render_full_rank_direction_evidence"] = False
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("direction evidence" in err or "direction rows" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_graphcg_readability_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "graphcg_direction_cosines_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("graphcg_readability_contract")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("structured readability contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_missing_embedding_trajectory_identity_contract(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "got_embedding_map_payloads.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("layout_contract")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("trajectory identity contract" in err for err in report["errors"])


def test_validate_audit_root_rejects_nll_density_state_provenance_gaps(tmp_path: Path):
    validator = _load_validator()
    cases = [
        ("support_samples_claimed_as_states", lambda payload: payload.__setitem__("sample_points_are_model_states", True), "incorrectly treats support samples as model states"),
        ("missing_anchor_layer", lambda payload: payload["density_contract"].pop("actual_model_anchor_layer"), "actual-anchor layer provenance"),
        ("wrong_anchor_count", lambda payload: payload.__setitem__("anchor_count", 3), "anchor count does not match"),
        ("missing_density_volume_provenance", lambda payload: payload.__setitem__("density_volume", {"available": True}), "density volume is missing non-model-state provenance"),
        ("missing_visual_layer_contract", lambda payload: payload.pop("visual_layer_contract"), "visual-layer render contract"),
        ("visible_support_samples", lambda payload: payload["visual_layer_contract"].__setitem__("support_sample_trace_visibility", True), "support samples must render legend-only"),
        ("missing_pc3_z_axis_policy", lambda payload: payload["visual_layer_contract"].__setitem__("z_axis_policy", "raw NLL z axis"), "PC3 z-axis policy"),
    ]
    for case_name, mutate, expected in cases:
        audit = tmp_path / case_name / "got_audit"
        row = _row(audit, ".")
        payload_path = row / "got_nll_density_cloud_payload.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        mutate(payload)
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
        report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
        assert not report["ok"], case_name
        assert any(expected in err for err in report["errors"]), (case_name, report["errors"])


def test_validate_audit_root_accepts_legacy_unavailable_tropical_support_probabilities(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "legacy_support" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "tropical_support_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    metrics.pop("wall_margin_audit")
    metrics.pop("strict_wall_hit_rate")
    metrics.pop("near_wall_hit_rate")
    metrics.pop("near_wall_only_rate")
    metrics.pop("wall_margin_threshold")
    metrics.pop("near_wall_margin_threshold")
    metrics["support_probability_source"] = "unavailable_in_trace"
    metrics["active_support_probability_summary"] = {"available": False, "count": 0, "min": None, "max": None, "mean": None, "p05": None, "p50": None, "p95": None}
    metrics["support_probability_entropy_bits_summary"] = {"available": False, "count": 0, "min": None, "max": None, "mean": None, "p05": None, "p50": None, "p95": None}
    for edge in payload["support_flow_edges"]:
        edge.pop("strict_wall_hit")
        edge.pop("near_wall_hit")
        edge.pop("wall_margin_bucket")
        edge.pop("wall_margin_threshold")
        edge.pop("near_wall_margin_threshold")
        edge["active_support_probability"] = None
        edge["support_probability_entropy_bits"] = None
        edge["support_probability_source"] = None
        edge["top_model_support_probabilities"] = []
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert report["ok"], report["errors"]


def test_codex_browser_index_requires_artifact_button_for_each_payload_item(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    _row(audit, ".")
    _row(audit, "example_01")
    _row(audit, "example_02")
    samples = _browser_samples(audit, [".", "example_01", "example_02"])
    missing_src = str(samples[1]["artifacts"][0]["src"])
    _write(audit / "codex_browser_index.html", _codex_browser_html(samples, omit_button_src=missing_src))
    report = validator.validate_audit_root(audit, min_rows=3, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("missing artifact button" in err and missing_src in err for err in report["errors"])


def test_validate_bundle_root_accepts_sample_directories(tmp_path: Path):
    validator = _load_validator()
    bundle = tmp_path / "multi_sample_browser" / "latest"
    _row(bundle, "sample_000")
    _row(bundle, "sample_001")
    _write(bundle / "browser_index.html", "<!doctype html><body>sample bundle</body>")
    report = validator.validate_audit_root(bundle, min_rows=2, min_candidates=4, min_depth=2)
    assert report["ok"], report["errors"]
    assert report["rows_checked"] == 2


def test_validate_audit_root_rejects_broken_nll_surface_contact_fields(tmp_path: Path):
    validator = _load_validator()
    cases = [
        ("missing_projected_map", lambda payload: payload["nll_surface"].pop("surface_projected_z_by_record_id"), "missing per-record projected z values"),
        ("missing_local_surface", lambda payload: payload["nll_surface"].pop("local_embedding_neighborhood_surface"), "missing local embedding-neighborhood surface contract"),
        ("local_surface_proxy", lambda payload: payload["nll_surface"]["local_embedding_neighborhood_surface"].__setitem__("invented_nll_values", True), "invented NLL values"),
        ("local_surface_residual", lambda payload: payload["nll_surface"]["local_embedding_neighborhood_surface"].__setitem__("trajectory_point_surface_residual_max", 42.0), "local embedding-neighborhood surface residual exceeds tolerance"),
        ("local_surface_projection_mismatch", lambda payload: payload["nll_surface"]["local_embedding_neighborhood_surface"]["surface_projected_z_by_record_id"].__setitem__("root", 42.0), "local embedding-neighborhood surface projection for root does not match"),
        ("touch_flag_true", lambda payload: payload["nodes"][0]["plot"].__setitem__("touches_nll_surface", True), "should keep PC3 geometry rather than touch"),
        ("missing_z", lambda payload: payload["nodes"][0]["plot"].pop("z"), "missing finite plotted z"),
        ("non_null_z_surface", lambda payload: payload["nodes"][0]["plot"].__setitem__("z_surface", 999.0), "should leave z_surface null"),
        ("missing_raw_centered", lambda payload: payload["nodes"][0]["plot"].pop("raw_centered_scaled_nll"), "missing raw centered/scaled NLL z"),
        ("missing_centered", lambda payload: payload["nodes"][0]["plot"].pop("z_centered_scaled_nll"), "missing centered/scaled NLL metadata z"),
        ("projected_z_mismatch", lambda payload: payload["nodes"][0]["plot"].__setitem__("z", 999.0), "plotted z does not match PCA pc3 geometry"),
        ("projected_centered_mismatch", lambda payload: payload["nodes"][0]["plot"].__setitem__("z_centered_scaled_nll", 999.0), "centered/scaled NLL metadata does not match"),
    ]
    for case_name, mutate, expected in cases:
        audit = tmp_path / case_name / "got_audit"
        _row(audit, ".")
        payload_path = audit / "got_trajectory_payloads.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        mutate(payload)
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
        assert not report["ok"], case_name
        assert any(expected in err for err in report["errors"]), (case_name, report["errors"])


def test_validator_cli_writes_json_and_markdown(tmp_path: Path):
    audit = tmp_path / "step_00000001" / "got_audit"
    _row(audit, ".")
    _row(audit, "example_01")
    _row(audit, "example_02")
    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--audit-root",
            str(audit),
            "--min-rows",
            "3",
            "--min-candidates",
            "4",
            "--min-depth",
            "2",
            "--json-output",
            str(json_out),
            "--markdown-output",
            str(md_out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(json_out.read_text(encoding="utf-8"))["ok"] is True
    assert "Overall status: PASS" in md_out.read_text(encoding="utf-8")
