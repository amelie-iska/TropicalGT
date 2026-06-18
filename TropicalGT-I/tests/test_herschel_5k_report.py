import argparse
import importlib.util
import json
from pathlib import Path


def _load_report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "write_herschel_5k_report.py"
    spec = importlib.util.spec_from_file_location("write_herschel_5k_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_bundle_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_5k_review_bundle.py"
    spec = importlib.util.spec_from_file_location("prepare_5k_review_bundle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_herschel_report_preserves_blockers_and_sidecar_groups(tmp_path: Path):
    module = _load_report_module()
    query_context_contract = {
        "schema_version": "tropicalgt.analogical_query_context_conversion.v1",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "observed_query_context_keys": ["trajectory_probability_filtered_simplicial_object", "topological_algebra"],
        "accepted_query_complex_keys": ["trajectory_probability_filtered_simplicial_object"],
        "selected_query_complex_source": "trajectory_probability_filtered_simplicial_object",
        "selected_query_complex_available": True,
        "selected_query_probability_vertex_count": 3,
        "query_topological_algebra_source": "topological_algebra",
        "query_topological_algebra_available": True,
        "rejected_query_context_keys": [
            {
                "key": "probability_filtered_simplicial_object",
                "reason": "non_trajectory_probability_complex_not_accepted_as_query_fallback",
                "has_real_probability_filtration": True,
                "probability_vertex_count": 3,
            }
        ],
        "rejects_probability_filtered_simplicial_object_alias_as_fallback": True,
        "rejects_filtered_simplicial_object_without_model_probabilities": True,
        "conversion_path": "query_context.trajectory_probability_filtered_simplicial_object",
        "conversion_status": "valid_query_probability_trajectory_complex",
        "fail_closed_reason": None,
        "embedding_only_assignment_allowed": False,
        "probability_assignment_metric_required": "jensen_shannon_distance_on_model_probability_vectors",
    }
    analogical_maps_path = tmp_path / "analogical_simplicial_maps.json"
    analogical_maps_path.write_text(
        json.dumps(
            {
                "available": True,
                "maps": [{"rank": 0, "probability_simplicial_map_available": True}],
                "query_context_contract": query_context_contract,
                "topk_contract": {
                    "schema_version": "tropicalgt.analogical_topk.v1",
                    "status": "topk_maps_available",
                    "no_proxy_or_fallback": True,
                    "retrieval_requires_model_probability_vectors": True,
                    "embedding_only_assignment_allowed": False,
                    "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors",
                    "query_complex_required": "trajectory_probability_filtered_simplicial_object",
                    "codomain_complex_required": "trajectory_probability_filtered_simplicial_object",
                    "query_context_contract_schema_version": "tropicalgt.analogical_query_context_conversion.v1",
                    "query_context_contract": query_context_contract,
                    "top_k_requested": 12,
                    "top_k_rendered": 1,
                    "raw_retrieved_count": 2,
                    "qualified_model_probability_memory_count": 1,
                    "rejected_retrieved_count": 1,
                    "readability_contract": {
                        "schema_version": "tropicalgt.analogical_topk_readability.v1",
                        "no_proxy_or_fallback": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    analogical_memory_path = tmp_path / "analogical_memory_retrieval.json"
    analogical_memory_path.write_text(
        json.dumps(
            {
                "bank_path": "outputs/unit/memory_bank/trajectory_memories.jsonl",
                "bank_size": 3,
                "records_added": 1,
                "top_k": 12,
                "retrieved": [
                    {
                        "retrieval_score": 0.82,
                        "probability_simplicial_map_available": True,
                        "probability_simplicial_map_preservation_rate": 1.0,
                        "probability_simplicial_map_source": "model_probability_jensen_shannon_assignment",
                        "probability_simplicial_map_chain_map_certified": True,
                        "probability_simplicial_map_persistence_morphism_certified": True,
                    }
                ],
                "quality_gate": {
                    "candidate_count": 2,
                    "eligible_count": 1,
                    "rejected_count": 1,
                    "policy": "store memories only above quality threshold",
                    "reason_counts": {"below_quality_threshold": 1},
                    "thresholds": {"min_quality": 0.72},
                    "rows": [],
                },
                "retrieval_weights": {
                    "probability_simplicial_map_weight": 1.0,
                    "persistence_landscape_weight": 0.2,
                    "certified_cas_weight": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    analogical_simplex_path = tmp_path / "analogical_simplex_tree_analogy.json"
    analogical_simplex_path.write_text(
        json.dumps(
            {
                "contract": {
                    "schema_version": "tropicalgt.analogical_simplex_tree_analogy.v1",
                    "available": True,
                    "status": "simplex_tree_analogy_available",
                    "no_proxy_or_fallback": True,
                    "compares_query_and_memory_simplex_trees": True,
                    "renders_hasse_face_to_coface_rows": True,
                    "preserved_face_coface_chains_highlighted": True,
                    "failed_or_distorted_chains_labeled_not_maps": True,
                    "chain_map_claim_requires_certified_filtered_simplicial_map": True,
                    "persistence_module_morphism_claim_requires_certified_filtered_simplicial_map": True,
                    "source": "probability_simplicial_map.simplex_tree_map.rows",
                    "pair_count": 1,
                    "total_checked_simplices": 2,
                    "total_preserved_simplices": 2,
                },
                "pairs": [{"rank": 0, "query_simplex": [0], "memory_simplex": [1], "preserved": True}],
            }
        ),
        encoding="utf-8",
    )
    full_complex_payload_path = tmp_path / "got_full_trajectory_complex_payload.json"
    full_complex_payload_path.write_text(
        json.dumps(
            {
                "trajectory_complex_overlay_contract": {
                    "schema_version": "tropicalgt.trajectory_complex_overlay_contract.v1",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "solid_lines_reserved_for_radius_simplices": True,
                    "filled_faces_reserved_for_radius_simplices": True,
                    "dotted_lines_reserved_for_trajectory_decoding_order_overlays": True,
                    "safe_to_render_available_views": True,
                    "embedding_view": {
                        "available": True,
                        "actual_data_only": True,
                        "no_proxy_or_fallback": True,
                        "safe_to_render_overlay_semantics": True,
                    },
                    "probability_view": {
                        "available": True,
                        "actual_data_only": True,
                        "no_proxy_or_fallback": True,
                        "safe_to_render_overlay_semantics": True,
                    },
                },
                "filtered_simplicial_object": {
                    "available": True,
                    "summary": {"num_vertices": 3, "num_edges": 3, "num_two_simplices": 1},
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "available": True, "num_simplices": 7},
                },
                "probability_filtered_simplicial_object": {
                    "available": True,
                    "summary": {"num_vertices": 3, "num_edges": 3, "num_two_simplices": 1},
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "available": True, "num_simplices": 7},
                },
            }
        ),
        encoding="utf-8",
    )
    slider_payload = {
        "schema_version": "tropicalgt.radius_filtration_slider_contract.v1",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "radius_filtration": True,
        "threshold_order": "ascending_min_to_max",
        "thresholds_ascending": True,
        "first_frame_disjoint_vertices_only": True,
        "initial_radius_frame_hides_solid_edges_and_faces": True,
        "monotone_visible_counts": True,
        "monotone_solid_radius_edges": True,
        "monotone_filled_radius_faces": True,
        "threshold_count": 4,
        "frame_count": 4,
        "first_frame_vertex_count": 3,
        "last_frame_solid_edge_count": 3,
        "last_frame_filled_face_count": 1,
    }
    full_slider_path = tmp_path / "got_full_trajectory_complex_slider_contract.json"
    full_slider_path.write_text(json.dumps(slider_payload), encoding="utf-8")
    probability_slider_path = tmp_path / "got_full_trajectory_complex_jensen_shannon_slider_contract.json"
    probability_slider_path.write_text(json.dumps(slider_payload), encoding="utf-8")
    simplex_poset_payload = {
        "schema_version": "tropicalgt.simplex_tree_poset.v1",
        "available": True,
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "backend": "gudhi.SimplexTree",
        "safe_to_render_simplex_tree": True,
        "not_disconnected_simplex_columns": True,
        "primary_edges": "actual_face_to_coface_covers",
        "all_non_vertex_simplices_have_face_cover_edges": True,
        "displayed_simplex_count": 5,
        "source_simplex_count": 7,
        "actual_face_to_coface_cover_edges": 6,
        "empty_simplex_root_present": True,
        "truncated": False,
    }
    full_simplex_poset_path = tmp_path / "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json"
    full_simplex_poset_path.write_text(json.dumps(simplex_poset_payload), encoding="utf-8")
    probability_simplex_poset_path = tmp_path / "got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json"
    probability_simplex_poset_path.write_text(json.dumps(simplex_poset_payload), encoding="utf-8")
    step_manifest_path = tmp_path / "reasoning_step_complex_maps" / "manifest.json"
    step_manifest_path.parent.mkdir()
    step_manifest_path.write_text(
        json.dumps(
            {
                "contract": {
                    "schema_version": "tropicalgt.reasoning_step_complex_maps.v1",
                    "available": True,
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "step_count": 2,
                    "rendered_complex_pages": 2,
                    "rendered_simplex_tree_pages": 2,
                    "rendered_slider_contracts": 2,
                    "rendered_simplex_tree_poset_contracts": 2,
                    "gudhi_simplex_tree_step_count": 2,
                    "all_steps_have_source_contracts": True,
                    "all_step_complex_source_contracts_safe": True,
                    "all_step_radius_sliders_start_disjoint_vertices": True,
                    "all_step_radius_sliders_monotone": True,
                    "all_step_radius_sliders_safe_to_render": True,
                    "all_step_simplex_tree_posets_use_gudhi": True,
                    "all_step_simplex_tree_posets_face_coface_primary": True,
                    "all_step_simplex_tree_posets_safe_to_render": True,
                    "radius_slider_unavailable_count": 0,
                    "simplex_tree_poset_unavailable_count": 0,
                    "source_contract_unavailable_count": 0,
                    "all_step_complex_fingerprints_present": True,
                    "all_step_complex_fingerprints_unique": True,
                },
                "steps": [{"index": 0}, {"index": 1}],
            }
        ),
        encoding="utf-8",
    )

    action_selection_contract = {
        "schema_version": "tropicalgt.gflownet_action_selection_contract.v1",
        "source": "gflownet_action_probs",
        "probability_source": "TropicalGTModel.gfn(graph_state).softmax",
        "audit_selection_score_source": "model_probability_minus_repeat_penalty",
        "selection_policy": "ranked_diverse_action_sweep",
        "branch_factor_requested": 2,
        "ranked_candidate_count": 4,
        "selected_action_count": 2,
        "allow_stop": False,
        "diverse_actions": True,
        "stochastic": False,
        "temperature": 1.0,
        "exploration": 0.0,
        "selected_from_real_model_action_probabilities": True,
        "not_a_policy_quality_certificate": True,
        "no_proxy_or_fallback": True,
    }
    inference_scaling_tree_path = tmp_path / "inference_scaling_tree.json"
    inference_scaling_tree_path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "level": 0,
                        "branch_selection": [
                            {
                                "schema_version": "tropicalgt.gflownet_branch_selection_audit.v1",
                                "source": "run_inference_scaling._select_branch_actions",
                                "parent_record_id": "unit",
                                "parent_path": [],
                                "level": 0,
                                "parent_rank": 0,
                                "selected_action_count": 2,
                                "selected_actions": [
                                    {"branch_rank": 0, "action": "expand", "probability": 0.7, "audit_selection_score": 0.7, "action_selection_contract": action_selection_contract},
                                    {"branch_rank": 1, "action": "verify", "probability": 0.2, "audit_selection_score": 0.2, "action_selection_contract": action_selection_contract},
                                ],
                                "action_selection_contract": action_selection_contract,
                                "no_proxy_or_fallback": True,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tropical_support_path = tmp_path / "tropical_support_payload.json"
    tropical_support_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "available": True,
                    "token_count": 5,
                    "unique_support_count": 2,
                    "effective_supports": 1.8,
                    "support_entropy_bits": 0.72,
                    "top_support_collapse_rate": 0.6,
                    "valid_support_assignment_count": 5,
                    "invalid_support_count": 0,
                    "support_probability_source": "model_tropical_support_probabilities",
                    "wall_margin_audit": {
                        "wall_margin_threshold": 0.001,
                        "near_wall_margin_threshold": 0.01,
                        "strict_wall_hit_count": 1,
                        "near_wall_hit_count": 2,
                        "near_wall_only_count": 1,
                        "strict_wall_hit_rate": 0.2,
                        "near_wall_hit_rate": 0.4,
                        "near_wall_only_rate": 0.2,
                        "metric_scope": "margin_threshold_audit_not_certified_normal_fan_wall_crossing",
                        "low_strict_wall_interpretation_status": "strict_wall_margin_events_observed",
                    },
                    "strict_wall_hit_rate": 0.2,
                    "near_wall_hit_rate": 0.4,
                    "near_wall_only_rate": 0.2,
                    "wall_margin_threshold": 0.001,
                    "near_wall_margin_threshold": 0.01,
                    "normal_fan_wall_crossing_certified": False,
                    "render_contract_schema_version": "tropicalgt.tropical_support_render.v1",
                    "readability_contract_schema_version": "tropicalgt.tropical_support_readability.v1",
                    "no_proxy_or_fallback": True,
                },
                "tropical_support_render_contract": {
                    "schema_version": "tropicalgt.tropical_support_render.v1",
                    "support_probability_source": "model_tropical_support_probabilities",
                    "observed_support_count": 2,
                    "token_count": 5,
                    "valid_support_assignment_count": 5,
                    "invalid_support_count": 0,
                    "normal_fan_wall_crossing_certified": False,
                    "wall_margin_metric_scope": "margin_threshold_audit_not_certified_normal_fan_wall_crossing",
                    "no_proxy_or_fallback": True,
                },
                "tropical_support_readability_contract": {
                    "schema_version": "tropicalgt.tropical_support_readability.v1",
                    "no_proxy_or_fallback": True,
                },
            }
        ),
        encoding="utf-8",
    )
    graphcg_direction_path = tmp_path / "graphcg_direction_cosines_payload.json"
    graphcg_direction_path.write_text(
        json.dumps(
            {
                "available": True,
                "matrix_shape": [3, 4],
                "full_rank_direction_count": 4,
                "active_rank_nonzero_mean_abs": 4,
                "panel_count": 5,
                "mean_abs_min": 0.1,
                "mean_abs_max": 0.9,
                "mean_abs_p90": 0.8,
                "graphcg_direction_evidence_contract": {
                    "schema_version": "tropicalgt.graphcg_direction_evidence.v1",
                    "source": "candidate.graphcg_projection.all_direction_cosines",
                    "no_proxy_or_fallback": True,
                    "all_model_directions_have_rows": True,
                    "direction_count": 4,
                    "direction_row_count": 4,
                    "exact_direction_ids_preserved": True,
                    "all_directions_rendered_in_heatmap": True,
                    "all_directions_rendered_in_activity_spectrum": True,
                    "all_directions_rendered_in_signed_bias_panel": True,
                    "top_active_direction_panel_count": 2,
                    "safe_to_render_full_rank_direction_evidence": True,
                },
                "graphcg_readability_contract": {
                    "schema_version": "tropicalgt.graphcg_direction_readability.v1",
                    "source": "candidate.graphcg_projection",
                    "no_proxy_or_fallback": True,
                    "all_model_directions_rendered": True,
                    "directions_sampled_for_heatmap": False,
                    "panels_are_separate": True,
                    "exact_direction_ids_preserved_in_hover_and_payload": True,
                },
                "projection_basis_certificate": {
                    "source": "candidate.graphcg_projection",
                    "available": True,
                    "projection_basis": "effective_full_rank_qr",
                    "basis_source_counts": {"effective_full_rank_qr": 3},
                    "candidate_count": 3,
                    "direction_count": 4,
                    "all_candidates_have_all_direction_cosines": True,
                },
                "direction_rows": [
                    {"direction_id": 0, "no_proxy_or_fallback": True},
                    {"direction_id": 1, "no_proxy_or_fallback": True},
                    {"direction_id": 2, "no_proxy_or_fallback": True},
                    {"direction_id": 3, "no_proxy_or_fallback": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    nll_density_path = tmp_path / "got_nll_density_cloud_payload.json"
    nll_density_path.write_text(
        json.dumps(
            {
                "available": True,
                "render_contract": "Gaussian cloud points are not model states; actual model anchors remain distinct.",
                "anchor_count": 3,
                "actual_model_anchor_count": 3,
                "support_sample_count": 9,
                "support_samples_hidden_as_model_states": True,
                "sample_points_are_model_states": False,
                "kernel_bandwidth": 0.25,
                "nll_range": {"min": 1.0, "max": 1.4, "span": 0.4},
                "visual_layer_contract": {
                    "schema_version": "tropicalgt.nll_density_render.v1",
                    "page": "standalone_density_cloud",
                    "z_axis_policy": "z is PC3(graph_state embedding); raw NLL is encoded by color",
                    "actual_model_anchor_count": 3,
                    "support_sample_count": 9,
                    "kernel_bandwidth": 0.25,
                    "actual_anchor_layer_visible_by_default": True,
                    "support_sample_trace_visibility": "legendonly",
                    "support_samples_are_model_states": False,
                    "support_samples_hidden_as_model_states": True,
                    "visible_density_layers": ["density_volume", "anchor_gaussian_neighborhoods", "actual_model_anchor_markers"],
                    "no_proxy_or_fallback": True,
                },
                "density_contract": {
                    "actual_model_anchor_layer": True,
                    "sample_points_are_model_states": False,
                    "support_samples_hidden_as_model_states": True,
                },
                "density_volume": {"available": True, "support_samples_are_not_model_states": True},
            }
        ),
        encoding="utf-8",
    )
    bivariate_module_path = tmp_path / "trajectory_level_radius_bifiltration.json"
    bivariate_module_path.write_text(
        json.dumps(
            {
                "available": True,
                "coefficient_ring": "F2[x_level,x_radius]",
                "num_parameters": 2,
                "object_key_selected": "filtered_simplicial_object",
                "fiber_rank_profile": [{"grade": [0, 0], "chain_group_ranks": {"0": 1}}],
                "structure_maps": [{"source_grade": [0, 0], "target_grade": [1, 0], "direction": "x_level", "homology_rank": {"0": 1}}],
                "rank_invariant_samples": [{"source_grade": [0, 0], "target_grade": [1, 0], "h0_rank": 1}],
                "chain_module_generators": [{"simplex": ["v0"], "homological_degree": 0, "multidegree": [0, 0]}],
                "boundary_monomials": {"d1": []},
                "chain_presentation_diagnostics": {
                    "method": "finite_multigraded_chain_presentation_diagnostics",
                    "ring": "F2[x_level,x_radius]",
                    "field": "F2",
                    "not_a_free_resolution": True,
                    "certificate_attached": False,
                    "resolution_status": "chain_presentation_only",
                    "real_free_resolution": {
                        "schema_version": "tropicalgt.real_free_resolution.v1",
                        "available": False,
                        "status": "certificate_failed",
                        "reason": "No CAS backend returned a certified real free resolution.",
                        "certificate_attached": False,
                        "real_free_resolution_certified": False,
                        "exactness_certified": False,
                        "multigraded_free_resolution_certified": False,
                        "safe_to_render_as_multigraded_free_resolution": False,
                        "safe_unavailable_render": True,
                        "input_sha256": "module-input-sha"
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    two_parameter_visual_path = tmp_path / "trajectory_persistence" / "two_parameter_bifiltration.json"
    two_parameter_visual_path.parent.mkdir()
    two_parameter_visual_path.write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.two_parameter_bifiltration_visual.v1",
                "actual_data_only": True,
                "coefficient_ring": "F2[x_level,x_radius]",
                "primary_view": "miller_sturmfels_bivariate_staircase",
                "rank_surface_primary": False,
                "rank_surface_policy": "3D fiber-rank displays are secondary diagnostics only.",
                "no_proxy_resolution_claim": True,
                "rank_invariant_sample_count": 2,
                "axes": {"horizontal": "x_radius", "vertical": "x_level", "coordinate_one_dimensional_cones": ["rho_x_radius", "rho_x_level"]},
                "grid": {"fiber_row_count": 4, "x_level_grades": [0, 1], "x_radius_grades": [0, 1]},
                "chain_generator_summary": {"total_generator_count": 6, "homological_dimensions": [0, 1], "minimal_antichain": [[0, 0]]},
                "module_visual_contract": {
                    "schema_version": "tropicalgt.two_parameter_module_staircase_contract.v1",
                    "no_proxy_or_fallback": True,
                    "primary_view": "miller_sturmfels_bivariate_staircase",
                    "rank_slabs_removed_as_primary_view": True,
                    "secondary_rank_diagnostics_labeled": True,
                    "x_radius_horizontal": True,
                    "x_level_vertical": True,
                    "shaded_regions_are_upward_closed_generated_submodules": True,
                    "white_points_are_displayed_quotient_basis_lattice_points": True,
                    "structure_map_summary_required": True,
                    "primary_structure_map_evidence_required": True,
                    "safe_resolution_policy": "Only scoped two-variable monomial staircase resolutions or certified CAS free resolutions may be rendered; chain-presentation diagnostics are never substituted as a free resolution.",
                },
                "structure_map_summary": {
                    "schema_version": "tropicalgt.two_parameter_structure_maps.v1",
                    "coefficient_ring": "F2[x_level,x_radius]",
                    "actual_adjacent_map_count": 2,
                    "valid_grade_edge_count": 2,
                    "direction_counts": {"x_level": 1, "x_radius": 1},
                    "east_north_structure_maps_present": True,
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
                "miller_sturmfels_staircase_evidence": {
                    "schema_version": "tropicalgt.miller_sturmfels_staircase_evidence.v1",
                    "coefficient_ring": "F2[x_level,x_radius]",
                    "source": "staircase_cards_from_bifiltration.chain_module_generators[*].multidegree",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "primary_view": "miller_sturmfels_bivariate_staircase",
                    "axes": {"horizontal": "x_radius", "vertical": "x_level", "coordinate_one_dimensional_cones": ["rho_x_radius", "rho_x_level"]},
                    "card_count": 2,
                    "primary_card_count": 1,
                    "primary_card_index": 0,
                    "primary_homological_degree": 1,
                    "total_actual_generator_bidegree_count": 3,
                    "total_minimal_antichain_count": 2,
                    "total_generator_label_count": 2,
                    "total_upward_closed_region_count": 2,
                    "total_quotient_basis_lattice_count": 5,
                    "total_hilbert_numerator_term_count": 4,
                    "total_adjacent_lcm_syzygy_count": 1,
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
                "certificate_indexed_cas_evidence": {
                    "schema_version": "tropicalgt.cas_certificate_indexed_evidence.v1",
                    "available": False,
                    "safe_unavailable_render": True,
                    "no_proxy_or_fallback": True,
                },
                "staircase_cards": [
                    {
                        "schema_version": "tropicalgt.two_parameter_staircase_card.v1",
                        "homological_degree": 1,
                        "primary_card": True,
                        "x_radius_horizontal": True,
                        "x_level_vertical": True,
                        "shaded_regions_are_upward_closed_generated_submodules": True,
                        "white_points_are_displayed_quotient_basis_lattice_points": True,
                        "generator_labels": [{"label": "g1", "bidegree": [1, 0]}, {"label": "g2", "bidegree": [0, 2]}],
                        "upward_closed_regions": [{"generator_label": "g1"}, {"generator_label": "g2"}],
                        "quotient_basis_lattice_points": [[0, 0], [0, 1]],
                        "quotient_basis_lattice_count": 2,
                        "hilbert_numerator_terms": ["1", "-x_level"],
                        "adjacent_lcm_syzygies": [{"lcm_bidegree": [1, 2]}],
                        "theorem_scope": "exact two-variable monomial-ideal staircase resolution when adjacent-LCM theorem applies; not a full persistence-module free resolution without CAS certification",
                    },
                    {
                        "schema_version": "tropicalgt.two_parameter_staircase_card.v1",
                        "homological_degree": 0,
                        "primary_card": False,
                        "x_radius_horizontal": True,
                        "x_level_vertical": True,
                        "shaded_regions_are_upward_closed_generated_submodules": True,
                        "white_points_are_displayed_quotient_basis_lattice_points": True,
                        "generator_labels": [],
                        "upward_closed_regions": [],
                        "quotient_basis_lattice_points": [],
                        "quotient_basis_lattice_count": 0,
                        "hilbert_numerator_terms": ["1", "-x_radius"],
                        "adjacent_lcm_syzygies": [],
                        "theorem_scope": "exact two-variable monomial-ideal staircase resolution when adjacent-LCM theorem applies; not a full persistence-module free resolution without CAS certification",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    pca_diag = {
        "coordinate_source": "model graph_state embeddings",
        "method": "sklearn PCA",
        "n_samples": 3,
        "embedding_dim": 2,
        "pairwise_distance_correlation": 0.98,
        "normalized_stress": 0.02,
        "duplicate_pca_coordinates_rounded8": 1,
        "unique_embedding_ratio_rounded8": 0.67,
    }
    trajectory_nodes = [
        {"record_id": "root", "parent": None, "embedding": [0.0, 1.0], "embedding_source": "model graph_state", "embedding_pca": {"pc1": 0.0, "pc2": 0.0, "pc3": 0.0}},
        {"record_id": "a", "parent": "root", "embedding": [1.0, 0.0], "embedding_source": "model graph_state", "embedding_pca": {"pc1": 1.0, "pc2": 0.0, "pc3": 0.1}},
        {"record_id": "b", "parent": "a", "embedding": [0.5, 0.5], "embedding_source": "model graph_state", "embedding_pca": {"pc1": 1.5, "pc2": 0.2, "pc3": 0.2}},
    ]
    trajectory_edges = [
        {"source": "root", "target": "a", "edge_source": "graph_of_thought_parent_edges"},
        {"source": "a", "target": "b", "edge_source": "graph_of_thought_parent_edges"},
    ]
    trajectory_payload_path = tmp_path / "got_trajectory_payloads.json"
    trajectory_payload_path.write_text(
        json.dumps(
            {
                "embedding_pca_diagnostics": pca_diag,
                "nodes": trajectory_nodes,
                "edges": trajectory_edges,
                "filtered_simplicial_objects": [{"record_id": row["record_id"]} for row in trajectory_nodes],
                "nll_surface": {"available": True, "touches_points": True, "max_point_residual": 0.0},
                "nll_progress": {"available": True, "node_count": 3},
            }
        ),
        encoding="utf-8",
    )
    embedding_payload_path = tmp_path / "got_embedding_map_payloads.json"
    embedding_payload_path.write_text(
        json.dumps(
            {
                "coordinate_source": "PCA of model graph_state embeddings; no level/tree layout coordinates are used",
                "layout_contract": {
                    "schema_version": "tropicalgt.embedding_trajectory_identity.v1",
                    "coordinate_source": "model graph_state embeddings",
                    "branch_depth_metadata_present": True,
                    "parent_child_transitions_present": True,
                    "parent_child_transition_count": 2,
                    "edge_source": "graph_of_thought_parent_edges",
                    "node_embedding_source": "model graph_state",
                    "geometric_separation_overclaim_allowed": False,
                    "no_proxy_or_fallback": True,
                    "pca_quality_warning": True,
                },
                "sampling": {"stochastic_actions": True, "temperature": 1.1, "exploration": 0.2, "seed": 17},
                "embedding_pca_diagnostics": pca_diag,
                "nodes": trajectory_nodes,
                "filtered_simplicial_objects": [{"record_id": row["record_id"]} for row in trajectory_nodes],
                "edges": trajectory_edges,
            }
        ),
        encoding="utf-8",
    )
    persistence_landscape_path = tmp_path / "persistence_landscapes.json"
    persistence_landscape_path.write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.persistence_landscape_visual_contract.v1",
                "available": True,
                "source": "topology.persistence_representations.methods[*].landscape",
                "landscape_backend": "gudhi.representations.Landscape",
                "backend_provenance": {"available": True, "backends": ["gudhi.representations.Landscape"]},
                "actual_data_only": True,
                "no_proxy_or_fallback": True,
                "not_nll_fitness_landscape": True,
                "not_norm_only_summary": True,
                "safe_to_render_actual_landscape_functions": True,
                "curve_trace_count": 2,
                "finite_persistence_interval_count": 3,
                "growth_row_count": 2,
                "homology_dimensions": [0, 1],
                "small_multiples_available": True,
                "heatmap_available": True,
                "landscape_rows": [
                    {"level": 0, "homology_dimension": 0, "backend": "gudhi.representations.Landscape", "values_source": "reported_landscape_grid_values"},
                    {"level": 1, "homology_dimension": 1, "backend": "gudhi.representations.Landscape", "values_source": "reported_landscape_grid_values"},
                ],
                "unavailable_reasons": [],
                "unavailable_state_verified_by_intervals": False,
            }
        ),
        encoding="utf-8",
    )
    chart_bundle_path = tmp_path / "chart_bundle_transport_sidecar.json"
    chart_bundle_payload = {
        "schema_version": "tropicalgt.chart_bundle_transport_sidecar.v1",
        "available": True,
        "source_path": "unit.chart_bundle_transport_metadata",
        "reason": "exported_chart_bundle_transport_metadata_available",
        "metadata": {
            "schema_version": "tropicalgt.chart_bundle_transport_metadata.v1",
            "available": True,
            "source": "unit.fixture",
            "chart_ids": ["chart_00", "chart_01"],
            "overlap_pairs": [{"id": "chart_00__to__chart_01"}],
            "overlap_triples": [],
        },
        "chart_ids": ["chart_00", "chart_01"],
        "overlap_pair_count": 1,
        "overlap_triple_count": 0,
        "monomial_transport_contract": {
            "schema_version": "tropicalgt.monomial_transport_head.v1",
            "transport_ids": ["chart_00__to__chart_01"],
            "actual_data_only": True,
            "no_proxy_or_fallback": True,
        },
        "bundle_matroid_contract": {
            "schema_version": "tropicalgt.bundle_matroid_flat_incidence.v1",
            "flat_incidence_shape": [2, 3],
            "actual_data_only": True,
            "no_proxy_or_fallback": True,
        },
        "vector_bundle_paper_sidecar": {
            "schema_version": "tropicalgt.vector_bundle_paper_sidecar.v1",
            "available": True,
            "actual_data_only": True,
            "no_proxy_or_fallback": True,
            "chart_ids": ["chart_00", "chart_01"],
            "monomial_transport_ids": ["chart_00__to__chart_01"],
            "completeness_tier": "paper_ready",
            "safe_to_use_as_vector_bundle_paper_ready_evidence": True,
            "safe_to_use_as_vector_bundle_theorem_certificate": False,
            "safe_to_use_as_toric_or_tropical_embedding_certificate": False,
            "completeness_contract": {
                "schema_version": "tropicalgt.vector_bundle_paper_sidecar_completeness.v1",
                "tier": "paper_ready",
                "paper_ready": True,
                "basic_telemetry_available": True,
                "required_groups": {
                    "chart_and_monomial_transport_ids": True,
                    "configured_toric_active_rows": True,
                    "flat_incidence_diagnostics": True,
                    "graphcg_toric_agreement": True,
                    "transported_persistence_landscapes": True,
                },
                "missing_required_groups": [],
                "actual_data_only": True,
                "no_proxy_or_fallback": True,
            },
        },
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "safe_to_render_as_toric_embedding_certificate": False,
        "safe_to_render_as_tropical_variety_embedding": False,
        "safe_to_render_as_global_toric_variety_embedding": False,
        "safe_to_use_as_normal_fan_certificate": False,
    }
    chart_bundle_path.write_text(json.dumps(chart_bundle_payload), encoding="utf-8")
    toric_sidecar_path = tmp_path / "toric_embedding_sidecar.json"
    toric_sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.toric_embedding_sidecar_visual_audit.v1",
                "available": True,
                "source_path": "unit.model_derived_toric_exponent_matrix",
                "exponent_matrix_spec": {"variables": ["x", "y"], "exponent_matrix": [[1, 0], [0, 1]]},
                "cas_input_contract": {
                    "schema_version": "tropicalgt.toric_embedding_input_contract.v1",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "proxy_substitution_allowed": False,
                    "explicit_cas_input_present": True,
                    "safe_to_render_certificate": True,
                    "input_sha256": "toric-input-sha",
                },
                "diagnostics": {
                    "schema_version": "tropicalgt.cas_toric_embedding.v1",
                    "available": True,
                    "status": "certified",
                    "backend": "Macaulay2",
                    "certificate_attached": True,
                    "toric_ideal_certified": True,
                    "safe_to_render_as_toric_embedding": True,
                    "safe_to_render_as_tropical_variety_embedding": False,
                    "safe_to_render_as_global_toric_variety_embedding": False,
                    "safe_to_use_as_normal_fan_certificate": False,
                },
                "safe_to_render_as_finite_toric_ideal_sidecar": True,
                "safe_to_render_as_tropical_variety_embedding": False,
                "safe_to_render_as_global_toric_variety_embedding": False,
                "safe_to_use_as_normal_fan_certificate": False,
            }
        ),
        encoding="utf-8",
    )
    tropical_fan_path = tmp_path / "tropical_fan_diagnostics.json"
    tropical_fan_path.write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.tropical_fan_visual_audit.v1",
                "available": True,
                "source_path": "unit.model_derived_tropical_ideal",
                "ideal_spec": {"variables": ["x", "y"], "generators": ["x+y+1"]},
                "cas_input_contract": {
                    "schema_version": "tropicalgt.tropical_fan_input_contract.v1",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "proxy_substitution_allowed": False,
                    "explicit_cas_input_present": True,
                    "safe_to_render_certificate": True,
                    "input_sha256": "fan-input-sha",
                },
                "diagnostics": {
                    "schema_version": "tropicalgt.cas_tropical_fan.v1",
                    "available": True,
                    "status": "certified",
                    "backend": "Macaulay2",
                    "certificate_attached": True,
                    "fan_diagnostics_certified": True,
                    "tropical_cycle_certified": True,
                    "safe_to_render_as_tropical_fan": True,
                    "fan_summary": {"ray_count": 2, "ambient_dimension": 2},
                },
                "safe_to_render_as_tropical_fan": True,
            }
        ),
        encoding="utf-8",
    )
    topological_report = {
        "enabled": True,
        "audit_level": "full",
        "chain_complex": {"chain_group_ranks": {"0": 3, "1": 2}, "boundary_maps": {"d1": [[1, 0], [0, 1]]}},
        "graph_metrics": {"backend": "networkx"},
        "persistence": {"available": True, "backend": "gudhi", "intervals": [[0.0, 1.0], [0.2, "inf"]]},
        "multiparameter_persistence": {
            "num_parameters": 2,
            "fiber_rank_profile": [{"grade": [0, 0], "chain_group_ranks": {"0": 3}}],
            "rank_invariant_samples": [{"source_grade": [0, 0], "target_grade": [1, 0], "rank": 1}],
            "chain_module_generators": [{"simplex": ["v0"], "homological_degree": 0, "multidegree": [0, 0]}],
        },
        "persistence_representations": {"available": True},
    }
    trajectory_topological_algebra_path = tmp_path / "trajectory_topological_algebra.json"
    trajectory_topological_algebra_path.write_text(json.dumps(topological_report), encoding="utf-8")
    inference_topology_path = tmp_path / "inference_topology.json"
    inference_topology_path.write_text(json.dumps(topological_report), encoding="utf-8")
    inference_algebra_path = tmp_path / "inference_algebra.json"
    inference_algebra_path.write_text(json.dumps({}), encoding="utf-8")
    trajectory_growth_topology_path = tmp_path / "trajectory_growth_topology.json"
    trajectory_growth_topology_path.write_text(
        json.dumps([{"topological_algebra": topological_report, "probability_topological_algebra": topological_report} for _ in range(2)]),
        encoding="utf-8",
    )
    validator_json = tmp_path / "interactive_validator.json"
    validator_json.write_text(
        json.dumps(
            {
                "ok": False,
                "errors": [
                    "row 0 missing json analogical_simplex_tree_analogy.json",
                    "row 0 analogical top-k contract is missing or has wrong schema",
                    "row 0 trajectory persistence landscapes payload is missing",
                ],
                "evidence_gap_inventory": {
                    "schema_version": "tropicalgt.interactive_audit_evidence_gap_inventory.v1",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "strict_validation_still_required": True,
                    "gap_count": 3,
                    "category_counts": {"analogical_memory": 2, "persistence_landscapes": 1},
                    "categories": [
                        {
                            "category": "analogical_memory",
                            "count": 2,
                            "examples": [
                                "row 0 missing json analogical_simplex_tree_analogy.json",
                                "row 0 analogical top-k contract is missing or has wrong schema",
                            ],
                            "required_action": "regenerate analogical memory sidecars from real probability-vector maps",
                        },
                        {
                            "category": "persistence_landscapes",
                            "count": 1,
                            "examples": ["row 0 trajectory persistence landscapes payload is missing"],
                            "required_action": "rerun persistence landscape backfill from real GUDHI payloads",
                        },
                    ],
                    "policy": "does not make an artifact valid",
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = {
        "boundary_step": 5000,
        "target_bpb": 1.12,
        "config": "configs/unit.json",
        "report": "outputs/unit/periodic/step_00005000/periodic_validation_artifacts.json",
        "checkpoint": "checkpoints/unit.latest.pt",
        "stop_record": "outputs/training_stop_records/unit.json",
        "decision": {
            "bpb": 1.31,
            "graph_bpb": 2.2,
            "triggered": True,
            "restart_policy": "beginning",
        },
        "checkpoint_evidence": {
            "checkpoint_available": False,
            "safe_for_checkpoint_backed_restart": False,
            "checkpoint_unavailable_reason": "checkpoint_file_is_empty",
            "warnings": ["checkpoint_summary_unavailable:checkpoint_file_is_empty"],
        },
        "execution_readiness": {
            "execution_requested": False,
            "ready": False,
            "issues": ["empty_checkpoint:checkpoints/unit.latest.pt"],
        },
        "advanced_bpb_contract": {
            "safe_to_use_for_step0_bpb_restart": False,
            "failed_gates": ["graph_tokenization_enabled"],
        },
        "artifact_inventory": {
            "latest_got_audit_dir": "outputs/unit/periodic/step_00005000/got_audit",
            "latest_periodic_dir": "outputs/unit/periodic/step_00005000",
            "advanced_sidecars_tail": [
                str(tropical_fan_path),
                str(toric_sidecar_path),
                "got_audit/betti_table.json",
                "got_audit/certificate_indexed_cas_evidence.json",
                str(trajectory_payload_path),
                str(embedding_payload_path),
                str(bivariate_module_path),
                str(two_parameter_visual_path),
                str(persistence_landscape_path),
                "got_audit/analogical_memory_report.json",
                str(analogical_maps_path),
                str(analogical_memory_path),
                str(analogical_simplex_path),
                str(full_complex_payload_path),
                str(full_slider_path),
                str(probability_slider_path),
                str(full_simplex_poset_path),
                str(probability_simplex_poset_path),
                str(step_manifest_path),
                str(trajectory_topological_algebra_path),
                str(trajectory_growth_topology_path),
                str(inference_topology_path),
                str(inference_algebra_path),
                str(inference_scaling_tree_path),
                str(tropical_support_path),
                str(graphcg_direction_path),
                str(nll_density_path),
                "got_audit/chart_bundle_metrics.json",
                str(chart_bundle_path),
                "got_audit/derived_category_chain_map_report.json",
            ],
        },
        "restart_evidence_gate": {
            "restart_action": "blocked_missing_required_evidence_no_restart",
            "step0_restart_allowed": False,
            "blocked": True,
            "blockers": [
                "checkpoint_unavailable:checkpoint_file_is_empty:checkpoints/unit.latest.pt",
                "post_5k_review_commands_missing_results:eval_validation_visualizations",
            ],
        },
        "command_results": [
            {
                "name": "interactive_audit_validator_01",
                "command": f"python TropicalGT-I/scripts/validate_interactive_audit_artifacts.py --audit-root got_audit --json-output {validator_json}",
                "stdout_log": "stdout.log",
                "stderr_log": "stderr.log",
                "returncode": 1,
                "timed_out": False,
            }
        ],
    }
    bundle_path = tmp_path / "review_bundle_step_00005000.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    markdown_path = tmp_path / "herschel_report.md"
    json_path = tmp_path / "herschel_report.json"
    html_path = tmp_path / "herschel_report.html"

    summary = module.write_herschel_report(bundle_path, markdown_path, json_path, html_path)

    assert summary["schema_version"] == "tropicalgt.herschel_5k_report_summary.v1"
    assert summary["primary_metrics"]["bpb"] == 1.31
    assert summary["restart_decision"]["action"] == "blocked_missing_required_evidence_no_restart"
    assert summary["restart_decision"]["step0_restart_allowed"] is False
    assert "checkpoint_file_is_empty" in " ".join(summary["restart_decision"]["blockers"])
    groups = summary["artifact_evidence"]["sidecar_groups"]
    assert groups["cas_algebra"] == 2
    assert groups["topology_persistence"] == 9
    assert groups["analogical_memory"] == 4
    assert groups["other"] >= 2
    assert groups["tropical_toric"] == 3
    assert groups["graphcg"] == 1
    assert groups["gflownet"] == 1
    assert groups["nll_density"] == 1
    assert groups["chart_bundle"] == 2
    assert groups["vector_bundle"] == 1
    assert groups["sheaf_derived"] == 1
    validator_gaps = summary["artifact_evidence"]["validator_gap_evidence"]
    assert validator_gaps["available"] is True
    assert validator_gaps["source_count"] == 1
    assert validator_gaps["combined_category_counts"] == {"analogical_memory": 2, "persistence_landscapes": 1}
    assert validator_gaps["ranked_categories"][0]["category"] == "analogical_memory"
    assert validator_gaps["ranked_categories"][0]["count"] == 2
    assert validator_gaps["ranked_categories"][0]["required_action"].startswith("regenerate analogical")
    assert validator_gaps["ranked_categories"][0]["examples"][0]["example"] == "row 0 missing json analogical_simplex_tree_analogy.json"
    assert validator_gaps["top_examples"][0]["category"] == "analogical_memory"
    assert validator_gaps["top_examples"][0]["required_action"].startswith("regenerate analogical")
    assert validator_gaps["sources"][0]["gap_count"] == 3
    assert validator_gaps["sources"][0]["validator_ok"] is False
    gflownet_branch = summary["artifact_evidence"]["gflownet_branch_selection_evidence"]
    assert gflownet_branch["schema_version"] == "tropicalgt.herschel_gflownet_branch_selection_evidence.v1"
    assert gflownet_branch["available"] is True
    assert gflownet_branch["source_count"] == 1
    assert gflownet_branch["available_source_count"] == 1
    assert gflownet_branch["total_valid_branch_selection_rows"] == 1
    assert gflownet_branch["total_selected_actions"] == 2
    assert gflownet_branch["policy_counts"] == {"ranked_diverse_action_sweep": 1}
    assert gflownet_branch["sources"][0]["no_proxy_or_fallback"] is True
    assert gflownet_branch["sources"][0]["valid_branch_selection_row_count"] == 1
    tropical_support = summary["artifact_evidence"]["tropical_support_evidence"]
    assert tropical_support["schema_version"] == "tropicalgt.herschel_tropical_support_evidence.v1"
    assert tropical_support["available"] is True
    assert tropical_support["source_count"] == 1
    assert tropical_support["available_source_count"] == 1
    assert tropical_support["total_token_count"] == 5
    assert tropical_support["total_valid_support_assignment_count"] == 5
    assert tropical_support["support_probability_source_counts"] == {"model_tropical_support_probabilities": 1}
    assert tropical_support["low_strict_wall_interpretation_status_counts"] == {"strict_wall_margin_events_observed": 1}
    assert tropical_support["sources"][0]["strict_wall_hit_rate"] == 0.2
    assert tropical_support["sources"][0]["near_wall_hit_rate"] == 0.4
    assert tropical_support["sources"][0]["normal_fan_wall_crossing_certified"] is False
    assert tropical_support["sources"][0]["no_proxy_or_fallback"] is True
    graphcg_direction = summary["artifact_evidence"]["graphcg_direction_evidence"]
    assert graphcg_direction["schema_version"] == "tropicalgt.herschel_graphcg_direction_evidence.v1"
    assert graphcg_direction["available"] is True
    assert graphcg_direction["source_count"] == 1
    assert graphcg_direction["available_source_count"] == 1
    assert graphcg_direction["total_direction_count"] == 4
    assert graphcg_direction["total_candidate_count"] == 3
    assert graphcg_direction["basis_source_counts"] == {"effective_full_rank_qr": 3}
    assert graphcg_direction["sources"][0]["projection_basis"] == "effective_full_rank_qr"
    assert graphcg_direction["sources"][0]["active_rank_nonzero_mean_abs"] == 4
    assert graphcg_direction["sources"][0]["all_direction_panels_available"] is True
    assert graphcg_direction["sources"][0]["no_proxy_or_fallback"] is True
    nll_density = summary["artifact_evidence"]["nll_density_evidence"]
    assert nll_density["schema_version"] == "tropicalgt.herschel_nll_density_evidence.v1"
    assert nll_density["available"] is True
    assert nll_density["source_count"] == 1
    assert nll_density["available_source_count"] == 1
    assert nll_density["total_actual_model_anchor_count"] == 3
    assert nll_density["total_support_sample_count"] == 9
    assert nll_density["visible_density_layer_counts"] == {"actual_model_anchor_markers": 1, "anchor_gaussian_neighborhoods": 1, "density_volume": 1}
    assert nll_density["sources"][0]["support_sample_trace_visibility"] == "legendonly"
    assert nll_density["sources"][0]["support_samples_are_model_states"] is False
    assert nll_density["sources"][0]["no_proxy_or_fallback"] is True
    persistence_landscape = summary["artifact_evidence"]["persistence_landscape_evidence"]
    assert persistence_landscape["schema_version"] == "tropicalgt.herschel_persistence_landscape_evidence.v1"
    assert persistence_landscape["available"] is True
    assert persistence_landscape["source_count"] == 1
    assert persistence_landscape["available_source_count"] == 1
    assert persistence_landscape["verified_unavailable_source_count"] == 0
    assert persistence_landscape["total_landscape_row_count"] == 2
    assert persistence_landscape["total_curve_trace_count"] == 2
    assert persistence_landscape["total_finite_persistence_interval_count"] == 3
    assert persistence_landscape["backend_counts"] == {"gudhi.representations.Landscape": 1}
    assert persistence_landscape["sources"][0]["not_nll_fitness_landscape"] is True
    assert persistence_landscape["sources"][0]["not_norm_only_summary"] is True
    assert persistence_landscape["sources"][0]["no_proxy_or_fallback"] is True
    bivariate_module = summary["artifact_evidence"]["bivariate_module_evidence"]
    assert bivariate_module["schema_version"] == "tropicalgt.herschel_bivariate_module_evidence.v1"
    assert bivariate_module["available"] is True
    assert bivariate_module["source_count"] == 1
    assert bivariate_module["module_available_source_count"] == 1
    assert bivariate_module["certified_real_free_resolution_source_count"] == 0
    assert bivariate_module["safe_unavailable_real_free_resolution_source_count"] == 1
    assert bivariate_module["total_fiber_rank_profile_count"] == 1
    assert bivariate_module["total_structure_map_count"] == 1
    assert bivariate_module["resolution_status_counts"] == {"certificate_failed": 1}
    assert bivariate_module["sources"][0]["chain_presentation_not_a_free_resolution"] is True
    assert bivariate_module["sources"][0]["real_free_resolution_certified"] is False
    assert bivariate_module["sources"][0]["safe_unavailable_real_free_resolution"] is True
    two_parameter_visual = summary["artifact_evidence"]["two_parameter_bifiltration_visual_evidence"]
    assert two_parameter_visual["schema_version"] == "tropicalgt.herschel_two_parameter_bifiltration_visual_evidence.v1"
    assert two_parameter_visual["available"] is True
    assert two_parameter_visual["source_count"] == 1
    assert two_parameter_visual["available_source_count"] == 1
    assert two_parameter_visual["total_staircase_card_count"] == 2
    assert two_parameter_visual["total_primary_staircase_card_count"] == 1
    assert two_parameter_visual["total_actual_generator_bidegree_count"] == 3
    assert two_parameter_visual["total_minimal_antichain_count"] == 2
    assert two_parameter_visual["total_generator_label_count"] == 2
    assert two_parameter_visual["total_upward_closed_region_count"] == 2
    assert two_parameter_visual["total_quotient_basis_lattice_count"] == 5
    assert two_parameter_visual["total_hilbert_numerator_term_count"] == 4
    assert two_parameter_visual["total_adjacent_lcm_syzygy_count"] == 1
    assert two_parameter_visual["total_structure_map_count"] == 2
    assert two_parameter_visual["total_rank_invariant_sample_count"] == 2
    assert two_parameter_visual["total_grid_fiber_row_count"] == 4
    assert two_parameter_visual["status_counts"] == {"two_parameter_bifiltration_visual_available": 1}
    assert two_parameter_visual["sources"][0]["axes_horizontal"] == "x_radius"
    assert two_parameter_visual["sources"][0]["axes_vertical"] == "x_level"
    assert two_parameter_visual["sources"][0]["no_proxy_or_fallback"] is True
    trajectory_embedding = summary["artifact_evidence"]["trajectory_embedding_visual_evidence"]
    assert trajectory_embedding["schema_version"] == "tropicalgt.herschel_trajectory_embedding_visual_evidence.v1"
    assert trajectory_embedding["available"] is True
    assert trajectory_embedding["paired_payloads_available"] is True
    assert trajectory_embedding["source_count"] == 2
    assert trajectory_embedding["available_source_count"] == 2
    assert trajectory_embedding["total_trajectory_node_count"] == 3
    assert trajectory_embedding["total_embedding_node_count"] == 3
    assert trajectory_embedding["total_edge_count"] == 4
    assert trajectory_embedding["total_filtered_simplicial_object_count"] == 6
    assert trajectory_embedding["total_raw_model_embedding_count"] == 6
    assert trajectory_embedding["total_parent_child_transition_count"] == 2
    assert trajectory_embedding["pca_quality_warning_source_count"] == 2
    assert trajectory_embedding["status_counts"] == {"trajectory_embedding_visual_available": 2}
    assert trajectory_embedding["coordinate_source_counts"] == {"model graph_state embeddings": 2}
    assert {row["kind"] for row in trajectory_embedding["sources"]} == {"trajectory_payload", "embedding_map_payload"}
    assert all(row["pca_pairwise_distance_correlation"] == 0.98 for row in trajectory_embedding["sources"])
    chart_bundle = summary["artifact_evidence"]["chart_bundle_transport_evidence"]
    assert chart_bundle["schema_version"] == "tropicalgt.herschel_chart_bundle_transport_evidence.v1"
    assert chart_bundle["available"] is True
    assert chart_bundle["source_count"] == 1
    assert chart_bundle["available_source_count"] == 1
    assert chart_bundle["paper_ready_source_count"] == 1
    assert chart_bundle["total_chart_count"] == 2
    assert chart_bundle["total_transport_count"] == 1
    assert chart_bundle["completeness_tier_counts"] == {"paper_ready": 1}
    assert chart_bundle["sources"][0]["paper_ready"] is True
    assert chart_bundle["sources"][0]["safe_to_render_as_toric_embedding_certificate"] is False
    assert chart_bundle["sources"][0]["safe_to_use_as_vector_bundle_theorem_certificate"] is False
    assert chart_bundle["sources"][0]["no_proxy_or_fallback"] is True
    toric_tropical = summary["artifact_evidence"]["toric_tropical_cas_evidence"]
    assert toric_tropical["schema_version"] == "tropicalgt.herschel_toric_tropical_cas_evidence.v1"
    assert toric_tropical["available"] is True
    assert toric_tropical["source_count"] == 2
    assert toric_tropical["available_source_count"] == 2
    assert toric_tropical["toric_source_count"] == 1
    assert toric_tropical["tropical_fan_source_count"] == 1
    assert toric_tropical["certified_finite_toric_ideal_count"] == 1
    assert toric_tropical["certified_tropical_fan_count"] == 1
    assert toric_tropical["forbidden_global_claim_count"] == 0
    assert toric_tropical["total_tropical_ray_count"] == 2
    assert toric_tropical["status_counts"] == {"toric_embedding_sidecar:certified": 1, "tropical_fan_diagnostics:certified": 1}
    assert toric_tropical["sources"][0]["no_proxy_or_fallback"] is True
    assert toric_tropical["sources"][1]["no_proxy_or_fallback"] is True
    analogical_query = summary["artifact_evidence"]["analogical_query_context_evidence"]
    assert analogical_query["schema_version"] == "tropicalgt.herschel_analogical_query_context_evidence.v1"
    assert analogical_query["available"] is True
    assert analogical_query["source_count"] == 1
    assert analogical_query["available_source_count"] == 1
    assert analogical_query["sources"][0]["topk_embeds_same_contract"] is True
    assert analogical_query["sources"][0]["selected_query_complex_source"] == "trajectory_probability_filtered_simplicial_object"
    assert analogical_query["sources"][0]["selected_query_complex_available"] is True
    assert analogical_query["sources"][0]["selected_query_probability_vertex_count"] == 3
    assert analogical_query["sources"][0]["conversion_status"] == "valid_query_probability_trajectory_complex"
    assert analogical_query["sources"][0]["rejected_query_context_keys"] == ["probability_filtered_simplicial_object"]
    assert analogical_query["sources"][0]["embedding_only_assignment_allowed"] is False
    analogical_memory = summary["artifact_evidence"]["analogical_memory_evidence"]
    assert analogical_memory["schema_version"] == "tropicalgt.herschel_analogical_memory_evidence.v1"
    assert analogical_memory["available"] is True
    assert analogical_memory["source_count"] == 3
    assert analogical_memory["available_source_count"] == 3
    assert analogical_memory["verified_insufficient_memory_source_count"] == 0
    assert analogical_memory["total_bank_size"] == 3
    assert analogical_memory["total_retrieved_count"] == 1
    assert analogical_memory["total_top_k_rendered"] == 1
    assert analogical_memory["total_simplex_tree_pair_count"] == 1
    assert analogical_memory["total_checked_simplices"] == 2
    assert analogical_memory["total_preserved_simplices"] == 2
    assert analogical_memory["probability_vector_contract_source_count"] == 1
    assert analogical_memory["quality_gate_reason_counts"] == {"below_quality_threshold": 1}
    assert analogical_memory["status_counts"] == {"retrieval_available": 1, "simplex_tree_analogy_available": 1, "topk_maps_available": 1}
    simplicial_complex = summary["artifact_evidence"]["simplicial_complex_evidence"]
    assert simplicial_complex["schema_version"] == "tropicalgt.herschel_simplicial_complex_evidence.v1"
    assert simplicial_complex["available"] is True
    assert simplicial_complex["source_count"] == 6
    assert simplicial_complex["available_source_count"] == 6
    assert simplicial_complex["total_view_count"] == 2
    assert simplicial_complex["total_step_count"] == 2
    assert simplicial_complex["total_radius_slider_contracts"] == 2
    assert simplicial_complex["total_simplex_tree_poset_contracts"] == 2
    assert simplicial_complex["total_vertices"] == 12
    assert simplicial_complex["total_edges"] == 12
    assert simplicial_complex["total_faces"] == 4
    assert simplicial_complex["total_source_simplices"] == 28
    assert simplicial_complex["total_displayed_simplices"] == 10
    assert simplicial_complex["status_counts"] == {
        "full_trajectory_complex_available": 1,
        "radius_slider_available": 2,
        "reasoning_step_manifest_available": 1,
        "simplex_tree_poset_available": 2,
    }
    topological_algebra = summary["artifact_evidence"]["topological_algebra_evidence"]
    assert topological_algebra["schema_version"] == "tropicalgt.herschel_topological_algebra_evidence.v1"
    assert topological_algebra["available"] is True
    assert topological_algebra["source_count"] == 4
    assert topological_algebra["available_source_count"] == 3
    assert topological_algebra["total_growth_rows"] == 2
    assert topological_algebra["total_topology_reports"] == 4
    assert topological_algebra["total_probability_topology_reports"] == 2
    assert topological_algebra["total_persistence_intervals"] == 12
    assert topological_algebra["total_finite_intervals"] == 6
    assert topological_algebra["total_chain_group_rank_entries"] == 12
    assert topological_algebra["total_boundary_maps"] == 6
    assert topological_algebra["total_multiparameter_fiber_rows"] == 6
    assert topological_algebra["total_rank_invariant_samples"] == 6
    assert topological_algebra["total_chain_module_generators"] == 6
    assert topological_algebra["status_counts"] == {
        "inference_algebra_empty_unavailable": 1,
        "inference_topology_available": 1,
        "trajectory_growth_topology_available": 1,
        "trajectory_topological_algebra_available": 1,
    }
    assert "no training" in summary["policy"]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Herschel 5K Evidence Report" in markdown
    assert "flowchart TD" in markdown
    assert "## Validator Evidence Gaps" in markdown
    assert "### Required Actions" in markdown
    assert "analogical_memory" in markdown
    assert "row 0 missing json analogical_simplex_tree_analogy.json" in markdown
    assert "regenerate analogical memory sidecars" in markdown
    assert "## GFlowNet Branch Selection Evidence" in markdown
    assert "ranked_diverse_action_sweep" in markdown
    assert "## Tropical Support Evidence" in markdown
    assert "model_tropical_support_probabilities" in markdown
    assert "strict_wall_margin_events_observed" in markdown
    assert "## GraphCG Direction Evidence" in markdown
    assert "effective_full_rank_qr" in markdown
    assert "## NLL Density Evidence" in markdown
    assert "density_volume" in markdown
    assert "## Trajectory Embedding Visual Evidence" in markdown
    assert "trajectory_embedding_visual_available" in markdown
    assert "## Bivariate Module Evidence" in markdown
    assert "certificate_failed" in markdown
    assert "## Two-Parameter Bifiltration Visual Evidence" in markdown
    assert "two_parameter_bifiltration_visual_available" in markdown
    assert "## Persistence Landscape Evidence" in markdown
    assert "gudhi.representations.Landscape" in markdown
    assert "## Chart/Vector-Bundle Evidence" in markdown
    assert "paper_ready" in markdown
    assert "## Toric/Tropical CAS Evidence" in markdown
    assert "toric_embedding_sidecar:certified" in markdown
    assert "## Analogical Query Context Evidence" in markdown
    assert "## Analogical Memory Evidence" in markdown
    assert "## Simplicial Complex And Simplex-Tree Evidence" in markdown
    assert "## Topological Algebra Evidence" in markdown
    assert "trajectory_growth_topology_available" in markdown
    assert "inference_algebra_empty_unavailable" in markdown
    assert "topk_maps_available" in markdown
    assert "trajectory_probability_filtered_simplicial_object" in markdown
    assert "probability_filtered_simplicial_object" in markdown
    assert "checkpoint_file_is_empty" in markdown
    assert "blocked_missing_required_evidence_no_restart" in markdown
    html = html_path.read_text(encoding="utf-8")
    assert "Herschel 5K Visual Evidence Report" in html
    assert "data-chart='sidecar-groups'" in html
    assert "data-chart='validator-gap-counts'" in html
    assert "Restart Decision Flow" in html
    assert "Validator Gap Actions" in html
    assert "GFlowNet Branch Selection Evidence" in html
    assert "data-chart='gflownet-branch-selection-policies'" in html
    assert "ranked_diverse_action_sweep" in html
    assert "Tropical Support Evidence" in html
    assert "data-chart='tropical-support-probability-sources'" in html
    assert "strict_wall_margin_events_observed" in html
    assert "GraphCG Direction Evidence" in html
    assert "data-chart='graphcg-projection-basis-sources'" in html
    assert "effective_full_rank_qr" in html
    assert "NLL Density Evidence" in html
    assert "data-chart='nll-density-visible-layers'" in html
    assert "density_volume" in html
    assert "Trajectory Embedding Visual Evidence" in html
    assert "data-chart='trajectory-embedding-visual-statuses'" in html
    assert "trajectory_embedding_visual_available" in html
    assert "Bivariate Module Evidence" in html
    assert "data-chart='bivariate-module-resolution-statuses'" in html
    assert "certificate_failed" in html
    assert "Two-Parameter Bifiltration Visual Evidence" in html
    assert "data-chart='two-parameter-bifiltration-visual-statuses'" in html
    assert "two_parameter_bifiltration_visual_available" in html
    assert "Persistence Landscape Evidence" in html
    assert "data-chart='persistence-landscape-backends'" in html
    assert "gudhi.representations.Landscape" in html
    assert "Chart/Vector-Bundle Evidence" in html
    assert "data-chart='chart-vector-bundle-completeness-tiers'" in html
    assert "paper_ready" in html
    assert "Toric/Tropical CAS Evidence" in html
    assert "data-chart='toric-tropical-cas-statuses'" in html
    assert "toric_embedding_sidecar" in html
    assert "Analogical Query Context Evidence" in html
    assert "Analogical Memory Evidence" in html
    assert "analogical-memory-statuses" in html
    assert "Simplicial Complex And Simplex-Tree Evidence" in html
    assert "simplicial-complex-statuses" in html
    assert "Topological Algebra Evidence" in html
    assert "topological-algebra-statuses" in html
    assert "trajectory_growth_topology_available" in html
    assert "valid_query_probability_trajectory_complex" in html
    assert "row 0 missing json analogical_simplex_tree_analogy.json" in html
    assert "sidecar-filter" in html
    assert "analogical_memory" in html
    assert json.loads(json_path.read_text(encoding="utf-8"))["restart_decision"] == summary["restart_decision"]


def test_prepare_review_bundle_writes_herschel_report_artifacts(tmp_path: Path):
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    report_path = output_dir / "train_report.json"
    report_path.write_text(
        json.dumps({"final_step": 5000, "eval": {"bpb": 1.3, "graph_bpb": 2.0}}),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "run_name": "unit_run",
                "batch_size": 2,
                "seq_len": 16,
                "model": {"dim": 32},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=cfg_path,
        report=report_path,
        checkpoint=None,
        stop_record=None,
        output_dir=tmp_path / "bundle",
        boundary_step=5000,
        target_bpb=1.12,
        metric="eval.bpb",
        graph_metric="eval.graph_bpb",
        python="python",
        split="validation",
        details_limit=2,
        viz_limit=3,
        audit_level="full",
        audit_ph_backend="gudhi",
        audit_max_simplices=128,
    )

    bundle = module.prepare_review_bundle(args)

    artifacts = bundle["artifacts"]
    assert "herschel_report_markdown" in artifacts
    assert "herschel_report_json" in artifacts
    assert "herschel_report_html" in artifacts
    assert (module.ROOT / artifacts["herschel_report_markdown"]).exists()
    assert (module.ROOT / artifacts["herschel_report_json"]).exists()
    assert (module.ROOT / artifacts["herschel_report_html"]).exists()
    assert bundle["herschel_report_summary"]["restart_decision"]["action"] == bundle["restart_evidence_gate"]["restart_action"]
    assert bundle["herschel_report_summary"]["checkpoint_evidence"]["restart_safe"] is False
    persisted_bundle = json.loads((module.ROOT / artifacts["bundle_json"]).read_text(encoding="utf-8"))
    assert persisted_bundle["artifacts"]["herschel_report_markdown"] == artifacts["herschel_report_markdown"]
    assert persisted_bundle["artifacts"]["herschel_report_json"] == artifacts["herschel_report_json"]
    assert persisted_bundle["artifacts"]["herschel_report_html"] == artifacts["herschel_report_html"]
    assert persisted_bundle["herschel_report_summary"]["restart_decision"] == bundle["herschel_report_summary"]["restart_decision"]

    report_markdown = (module.ROOT / artifacts["herschel_report_markdown"]).read_text(encoding="utf-8")
    assert "# Herschel 5K Evidence Report" in report_markdown
    assert "checkpoint_summary_unavailable" in report_markdown
    report_html = (module.ROOT / artifacts["herschel_report_html"]).read_text(encoding="utf-8")
    assert "Herschel 5K Visual Evidence Report" in report_html
    assert "Restart Decision Flow" in report_html
    bundle_markdown = (module.ROOT / artifacts["bundle_markdown"]).read_text(encoding="utf-8")
    assert "Herschel Report Summary" in bundle_markdown
    assert "herschel_report_markdown" in bundle_markdown
    assert "herschel_report_html" in bundle_markdown
