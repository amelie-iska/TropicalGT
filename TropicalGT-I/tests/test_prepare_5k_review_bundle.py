import argparse
import importlib.util
import json
import sys

import pytest
from pathlib import Path


def _load_bundle_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_5k_review_bundle.py"
    spec = importlib.util.spec_from_file_location("prepare_5k_review_bundle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_review_bundle_writes_prompt_contract_and_commands(tmp_path: Path):
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    report = {
        "final_step": 5000,
        "metrics": {"loss": 1.1, "nll": 1.0, "bpb": 1.45, "eval_bpb": 1.30, "eval_graph_bpb": 2.2},
        "eval": {"bpb": 1.30, "graph_bpb": 2.2},
        "history": [],
    }
    report_path = output_dir / "train_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
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
    stop_record = tmp_path / "stop.json"
    stop_record.write_text(json.dumps({"action": "target_reached_terminate", "target_step": 5000}), encoding="utf-8")
    args = argparse.Namespace(
        config=cfg_path,
        report=report_path,
        checkpoint=None,
        stop_record=stop_record,
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
    assert bundle["decision"]["bpb"] == 1.30
    assert bundle["decision"]["triggered"] is True
    assert "eval_tropicalgt_i.py" in bundle["commands"]["eval_validation_visualizations"]
    assert "--render-visualizations" in bundle["commands"]["eval_validation_visualizations"]
    assert bundle["command_results"] == []
    assert bundle["execution_readiness"]["execution_requested"] is False
    assert bundle["execution_readiness"]["ready"] is False
    assert any(issue.startswith("missing_checkpoint:") for issue in bundle["execution_readiness"]["issues"])
    assert bundle["commands"]["interactive_audit_backfills"] == []
    assert bundle["advanced_bpb_contract"]["section"]["required"] is False
    assert bundle["advanced_bpb_contract"]["safe_to_use_for_step0_bpb_restart"] is True
    checkpoint_evidence = bundle["checkpoint_evidence"]
    assert checkpoint_evidence["schema_version"] == "tropicalgt.checkpoint_evidence.v1"
    assert checkpoint_evidence["checkpoint_available"] is False
    assert checkpoint_evidence["safe_for_checkpoint_backed_restart"] is False
    assert any(warning.startswith("checkpoint_summary_unavailable:") for warning in checkpoint_evidence["warnings"])
    gate = bundle["restart_evidence_gate"]
    assert gate["restart_action"] == "blocked_missing_required_evidence_no_restart"
    assert gate["step0_restart_allowed"] is False
    assert gate["checkpoint_evidence_safe"] is False
    assert any(blocker.startswith("checkpoint_unavailable:checkpoint_unavailable") for blocker in gate["blockers"])
    assert any(blocker.startswith("checkpoint_evidence:checkpoint_summary_unavailable:") for blocker in gate["blockers"])
    assert any(blocker.startswith("execution_readiness:missing_checkpoint:") for blocker in gate["blockers"])
    assert bundle["restart_decision_schema"]["config_patch_contract"]["requires_evidence_paths"] is True
    assert "spawn_or_assign_codex_subagent_when_available" in bundle["review_requirements"]
    assert "review_metrics_advanced_sidecars_topological_geometric_algebraic_visualizations" in bundle["review_requirements"]
    assert "run_legacy_audit_backfill_before_strict_validation_when_available" in bundle["review_requirements"]
    prompt_text = (module.ROOT / artifacts["codex_prompt"]).read_text(encoding="utf-8")
    assert "spawn or assign a fresh Codex subagent" in prompt_text
    assert "topological, geometric, algebraic" in prompt_text
    assert "Restart from step 0" in prompt_text
    assert "No proxies or fallbacks" in prompt_text
    assert "restart_decision_schema" in prompt_text
    bundle_markdown = (module.ROOT / artifacts["bundle_markdown"]).read_text(encoding="utf-8")
    assert "Checkpoint Evidence" in bundle_markdown
    assert "Advanced BPB Contract" in bundle_markdown
    assert "Restart Evidence Gate" in bundle_markdown
    for key in ("contract_json", "contract_markdown", "advanced_bpb_contract_json", "codex_prompt", "bundle_json", "bundle_markdown"):
        assert (module.ROOT / artifacts[key]).exists()

def test_prepare_review_bundle_defaults_to_periodic_validation_artifacts(tmp_path: Path):
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    periodic_dir = output_dir / "periodic" / "step_00005000"
    query_context_contract = {
        "schema_version": "tropicalgt.analogical_query_context_conversion.v1",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "observed_query_context_keys": ["trajectory_probability_filtered_simplicial_object", "topological_algebra"],
        "accepted_query_complex_keys": ["trajectory_probability_filtered_simplicial_object"],
        "selected_query_complex_source": "trajectory_probability_filtered_simplicial_object",
        "selected_query_complex_available": True,
        "selected_query_probability_vertex_count": 4,
        "query_topological_algebra_source": "topological_algebra",
        "query_topological_algebra_available": True,
        "rejected_query_context_keys": [
            {
                "key": "probability_filtered_simplicial_object",
                "reason": "non_trajectory_probability_complex_not_accepted_as_query_fallback",
                "has_real_probability_filtration": True,
                "probability_vertex_count": 4,
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
    (periodic_dir / "got_audit").mkdir(parents=True)
    checkpoint_dir.mkdir()
    periodic_report = {
        "step": 5000,
        "validation": str(periodic_dir / "validation_report.json"),
        "metrics": {
            "eval_nll": 0.93,
            "eval_bpb": 1.25,
            "eval_graph_bpb": 2.05,
            "eval_parameter_golf_source_rate": 0.5,
        },
        "visualizations": {"got_audit": str(periodic_dir / "got_audit" / "inference_audit.html")},
    }
    (periodic_dir / "periodic_validation_artifacts.json").write_text(json.dumps(periodic_report), encoding="utf-8")
    (periodic_dir / "got_audit" / "analogical_simplicial_maps.json").write_text(
        json.dumps(
            {
                "available": False,
                "reason": "no_non_self_model_memory",
                "reason_detail": "No non-self model-probability analogical memories retrieved; no analogical correspondence certificate is rendered.",
                "maps": [],
                "query_context_contract": query_context_contract,
                "topk_contract": {
                    "schema_version": "tropicalgt.analogical_topk.v1",
                    "status": "unavailable_no_non_self_model_memory",
                    "no_proxy_or_fallback": True,
                    "retrieval_requires_model_probability_vectors": True,
                    "embedding_only_assignment_allowed": False,
                    "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors",
                    "query_complex_required": "trajectory_probability_filtered_simplicial_object",
                    "codomain_complex_required": "trajectory_probability_filtered_simplicial_object",
                    "query_context_contract_schema_version": "tropicalgt.analogical_query_context_conversion.v1",
                    "query_context_contract": query_context_contract,
                    "top_k_requested": 12,
                    "top_k_rendered": 0,
                    "raw_retrieved_count": 0,
                    "qualified_model_probability_memory_count": 0,
                    "rejected_retrieved_count": 0,
                    "readability_contract": {
                        "schema_version": "tropicalgt.analogical_topk_readability.v1",
                        "no_proxy_or_fallback": True,
                        "insufficient_memory_state_explicit": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (periodic_dir / "got_audit" / "analogical_memory_retrieval.json").write_text(
        json.dumps(
            {
                "bank_path": str(output_dir / "memory_bank" / "trajectory_memories.jsonl"),
                "bank_size": 0,
                "records_added": 0,
                "top_k": 12,
                "retrieved": [],
                "quality_gate": {
                    "candidate_count": 0,
                    "eligible_count": 0,
                    "rejected_count": 0,
                    "policy": "store memories only above quality threshold",
                    "reason_counts": {"no_non_self_model_memory": 1},
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
    (periodic_dir / "got_audit" / "analogical_simplex_tree_analogy.json").write_text(
        json.dumps(
            {
                "contract": {
                    "schema_version": "tropicalgt.analogical_simplex_tree_analogy.v1",
                    "available": False,
                    "status": "unavailable_no_non_self_model_memory",
                    "no_proxy_or_fallback": True,
                    "compares_query_and_memory_simplex_trees": True,
                    "renders_hasse_face_to_coface_rows": True,
                    "preserved_face_coface_chains_highlighted": True,
                    "failed_or_distorted_chains_labeled_not_maps": True,
                    "chain_map_claim_requires_certified_filtered_simplicial_map": True,
                    "persistence_module_morphism_claim_requires_certified_filtered_simplicial_map": True,
                    "source": "probability_simplicial_map.simplex_tree_map.rows",
                    "pair_count": 0,
                    "total_checked_simplices": 0,
                    "total_preserved_simplices": 0,
                    "reason_detail": "No certified probability simplicial maps available for simplex-tree analogy rows.",
                },
                "pairs": [],
            }
        ),
        encoding="utf-8",
    )
    (periodic_dir / "got_audit" / "got_full_trajectory_complex_payload.json").write_text(
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
                    "embedding_view": {"available": True, "actual_data_only": True, "no_proxy_or_fallback": True, "safe_to_render_overlay_semantics": True},
                    "probability_view": {"available": True, "actual_data_only": True, "no_proxy_or_fallback": True, "safe_to_render_overlay_semantics": True},
                },
                "filtered_simplicial_object": {
                    "available": True,
                    "summary": {"num_vertices": 4, "num_edges": 5, "num_two_simplices": 2},
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "available": True, "num_simplices": 11},
                },
                "probability_filtered_simplicial_object": {
                    "available": True,
                    "summary": {"num_vertices": 4, "num_edges": 5, "num_two_simplices": 2},
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "available": True, "num_simplices": 11},
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
        "threshold_count": 5,
        "frame_count": 5,
        "first_frame_vertex_count": 4,
        "last_frame_solid_edge_count": 5,
        "last_frame_filled_face_count": 2,
    }
    (periodic_dir / "got_audit" / "got_full_trajectory_complex_slider_contract.json").write_text(json.dumps(slider_payload), encoding="utf-8")
    (periodic_dir / "got_audit" / "got_full_trajectory_complex_jensen_shannon_slider_contract.json").write_text(json.dumps(slider_payload), encoding="utf-8")
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
        "displayed_simplex_count": 8,
        "source_simplex_count": 11,
        "actual_face_to_coface_cover_edges": 9,
        "empty_simplex_root_present": True,
        "truncated": False,
    }
    (periodic_dir / "got_audit" / "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json").write_text(json.dumps(simplex_poset_payload), encoding="utf-8")
    (periodic_dir / "got_audit" / "got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json").write_text(json.dumps(simplex_poset_payload), encoding="utf-8")
    (periodic_dir / "got_audit" / "reasoning_step_complex_maps").mkdir()
    (periodic_dir / "got_audit" / "reasoning_step_complex_maps" / "manifest.json").write_text(
        json.dumps(
            {
                "contract": {
                    "schema_version": "tropicalgt.reasoning_step_complex_maps.v1",
                    "available": True,
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "step_count": 3,
                    "rendered_complex_pages": 3,
                    "rendered_simplex_tree_pages": 3,
                    "rendered_slider_contracts": 3,
                    "rendered_simplex_tree_poset_contracts": 3,
                    "gudhi_simplex_tree_step_count": 3,
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
                "steps": [{"index": 0}, {"index": 1}, {"index": 2}],
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
        "ranked_candidate_count": 3,
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
    (periodic_dir / "got_audit" / "inference_scaling_tree.json").write_text(
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
                                    {"branch_rank": 0, "action": "expand", "probability": 0.6, "audit_selection_score": 0.6, "action_selection_contract": action_selection_contract},
                                    {"branch_rank": 1, "action": "verify", "probability": 0.3, "audit_selection_score": 0.3, "action_selection_contract": action_selection_contract},
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
    (periodic_dir / "got_audit" / "tropical_support_payload.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "available": True,
                    "token_count": 6,
                    "unique_support_count": 2,
                    "effective_supports": 1.7,
                    "support_entropy_bits": 0.68,
                    "top_support_collapse_rate": 0.5,
                    "valid_support_assignment_count": 6,
                    "invalid_support_count": 0,
                    "support_probability_source": "model_tropical_support_probabilities",
                    "wall_margin_audit": {
                        "wall_margin_threshold": 0.001,
                        "near_wall_margin_threshold": 0.01,
                        "strict_wall_hit_count": 0,
                        "near_wall_hit_count": 1,
                        "near_wall_only_count": 1,
                        "strict_wall_hit_rate": 0.0,
                        "near_wall_hit_rate": 0.1666666667,
                        "near_wall_only_rate": 0.1666666667,
                        "metric_scope": "margin_threshold_audit_not_certified_normal_fan_wall_crossing",
                        "low_strict_wall_interpretation_status": "low_strict_expected_near_wall_ambiguity",
                    },
                    "strict_wall_hit_rate": 0.0,
                    "near_wall_hit_rate": 0.1666666667,
                    "near_wall_only_rate": 0.1666666667,
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
                    "token_count": 6,
                    "valid_support_assignment_count": 6,
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
    (periodic_dir / "got_audit" / "graphcg_direction_cosines_payload.json").write_text(
        json.dumps(
            {
                "available": True,
                "matrix_shape": [2, 3],
                "full_rank_direction_count": 3,
                "active_rank_nonzero_mean_abs": 3,
                "panel_count": 5,
                "mean_abs_min": 0.2,
                "mean_abs_max": 0.8,
                "mean_abs_p90": 0.7,
                "graphcg_direction_evidence_contract": {
                    "schema_version": "tropicalgt.graphcg_direction_evidence.v1",
                    "source": "candidate.graphcg_projection.all_direction_cosines",
                    "no_proxy_or_fallback": True,
                    "all_model_directions_have_rows": True,
                    "direction_count": 3,
                    "direction_row_count": 3,
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
                    "basis_source_counts": {"effective_full_rank_qr": 2},
                    "candidate_count": 2,
                    "direction_count": 3,
                    "all_candidates_have_all_direction_cosines": True,
                },
                "direction_rows": [
                    {"direction_id": 0, "no_proxy_or_fallback": True},
                    {"direction_id": 1, "no_proxy_or_fallback": True},
                    {"direction_id": 2, "no_proxy_or_fallback": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (periodic_dir / "got_audit" / "got_nll_density_cloud_payload.json").write_text(
        json.dumps(
            {
                "available": True,
                "render_contract": "Gaussian cloud points are not model states; actual model anchors remain distinct.",
                "anchor_count": 4,
                "actual_model_anchor_count": 4,
                "support_sample_count": 12,
                "support_samples_hidden_as_model_states": True,
                "sample_points_are_model_states": False,
                "kernel_bandwidth": 0.2,
                "nll_range": {"min": 0.9, "max": 1.3, "span": 0.4},
                "visual_layer_contract": {
                    "schema_version": "tropicalgt.nll_density_render.v1",
                    "page": "standalone_density_cloud",
                    "z_axis_policy": "z is PC3(graph_state embedding); raw NLL is encoded by color",
                    "actual_model_anchor_count": 4,
                    "support_sample_count": 12,
                    "kernel_bandwidth": 0.2,
                    "actual_anchor_layer_visible_by_default": True,
                    "support_sample_trace_visibility": "legendonly",
                    "support_samples_are_model_states": False,
                    "support_samples_hidden_as_model_states": True,
                    "visible_density_layers": ["density_volume", "actual_model_anchor_markers"],
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
    (periodic_dir / "got_audit" / "trajectory_level_radius_bifiltration.json").write_text(
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
    topological_report = {
        "enabled": True,
        "audit_level": "full",
        "chain_complex": {"chain_group_ranks": {"0": 2, "1": 1}, "boundary_maps": {"d1": [[1], [0]]}},
        "graph_metrics": {"backend": "networkx"},
        "persistence": {"available": True, "backend": "gudhi", "intervals": [[0.0, 0.5], [0.3, "inf"]]},
        "multiparameter_persistence": {
            "num_parameters": 2,
            "fiber_rank_profile": [{"grade": [0, 0], "chain_group_ranks": {"0": 2}}],
            "rank_invariant_samples": [{"source_grade": [0, 0], "target_grade": [1, 0], "rank": 1}],
            "chain_module_generators": [{"simplex": ["v0"], "homological_degree": 0, "multidegree": [0, 0]}],
        },
        "persistence_representations": {"available": True},
    }
    (periodic_dir / "got_audit" / "trajectory_topological_algebra.json").write_text(json.dumps(topological_report), encoding="utf-8")
    (periodic_dir / "got_audit" / "inference_topology.json").write_text(json.dumps(topological_report), encoding="utf-8")
    (periodic_dir / "got_audit" / "inference_algebra.json").write_text(json.dumps({}), encoding="utf-8")
    (periodic_dir / "got_audit" / "trajectory_growth_topology.json").write_text(
        json.dumps([{"topological_algebra": topological_report, "probability_topological_algebra": topological_report}]),
        encoding="utf-8",
    )
    (periodic_dir / "got_audit" / "trajectory_persistence").mkdir(parents=True, exist_ok=True)
    (periodic_dir / "got_audit" / "trajectory_persistence" / "persistence_landscapes.json").write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.persistence_landscape_visual_contract.v1",
                "available": False,
                "source": "topology.persistence_representations.methods[*].landscape",
                "landscape_backend": "unavailable",
                "backend_provenance": {"available": False, "backends": []},
                "actual_data_only": True,
                "no_proxy_or_fallback": True,
                "not_nll_fitness_landscape": True,
                "not_norm_only_summary": False,
                "safe_to_render_actual_landscape_functions": False,
                "curve_trace_count": 0,
                "finite_persistence_interval_count": 0,
                "growth_row_count": 1,
                "homology_dimensions": [],
                "small_multiples_available": False,
                "heatmap_available": False,
                "landscape_rows": [],
                "unavailable_reasons": ["no_finite_persistence_intervals_for_gudhi_landscape"],
                "unavailable_state_verified_by_intervals": True,
            }
        ),
        encoding="utf-8",
    )
    (periodic_dir / "got_audit" / "chart_bundle_transport_sidecar.json").write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.chart_bundle_transport_sidecar.v1",
                "available": True,
                "source_path": "unit.chart_bundle_transport_metadata",
                "reason": "exported_chart_bundle_transport_metadata_available",
                "metadata": {
                    "schema_version": "tropicalgt.chart_bundle_transport_metadata.v1",
                    "available": True,
                    "source": "unit.fixture",
                    "chart_ids": ["chart_00", "chart_01", "chart_02"],
                    "overlap_pairs": [{"id": "chart_00__to__chart_01"}, {"id": "chart_01__to__chart_02"}],
                    "overlap_triples": [{"id": "chart_00__to__chart_01__to__chart_02"}],
                },
                "chart_ids": ["chart_00", "chart_01", "chart_02"],
                "overlap_pair_count": 2,
                "overlap_triple_count": 1,
                "monomial_transport_contract": {
                    "schema_version": "tropicalgt.monomial_transport_head.v1",
                    "transport_ids": ["chart_00__to__chart_01", "chart_01__to__chart_02"],
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                },
                "bundle_matroid_contract": {
                    "schema_version": "tropicalgt.bundle_matroid_flat_incidence.v1",
                    "flat_incidence_shape": [3, 5],
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                },
                "vector_bundle_paper_sidecar": {
                    "schema_version": "tropicalgt.vector_bundle_paper_sidecar.v1",
                    "available": True,
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "chart_ids": ["chart_00", "chart_01", "chart_02"],
                    "monomial_transport_ids": ["chart_00__to__chart_01", "chart_01__to__chart_02"],
                    "completeness_tier": "telemetry_partial",
                    "safe_to_use_as_vector_bundle_paper_ready_evidence": False,
                    "safe_to_use_as_vector_bundle_theorem_certificate": False,
                    "safe_to_use_as_toric_or_tropical_embedding_certificate": False,
                    "completeness_contract": {
                        "schema_version": "tropicalgt.vector_bundle_paper_sidecar_completeness.v1",
                        "tier": "telemetry_partial",
                        "paper_ready": False,
                        "basic_telemetry_available": True,
                        "required_groups": {
                            "chart_and_monomial_transport_ids": True,
                            "configured_toric_active_rows": True,
                            "flat_incidence_diagnostics": True,
                            "graphcg_toric_agreement": False,
                            "transported_persistence_landscapes": True,
                        },
                        "missing_required_groups": ["graphcg_toric_agreement"],
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
        ),
        encoding="utf-8",
    )
    (periodic_dir / "got_audit" / "toric_embedding_sidecar.json").write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.toric_embedding_sidecar_visual_audit.v1",
                "available": False,
                "source_path": "unavailable",
                "exponent_matrix_spec": None,
                "cas_input_contract": {
                    "schema_version": "tropicalgt.toric_embedding_input_contract.v1",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "proxy_substitution_allowed": False,
                    "explicit_cas_input_present": False,
                    "safe_to_render_certificate": False,
                    "input_sha256": None,
                },
                "diagnostics": {
                    "schema_version": "tropicalgt.cas_toric_embedding.v1",
                    "available": False,
                    "status": "unavailable_no_explicit_exponent_matrix",
                    "backend": "Macaulay2",
                    "certificate_attached": False,
                    "toric_ideal_certified": False,
                    "safe_to_render_as_toric_embedding": False,
                    "safe_to_render_as_tropical_variety_embedding": False,
                    "safe_to_render_as_global_toric_variety_embedding": False,
                    "safe_to_use_as_normal_fan_certificate": False,
                },
                "safe_to_render_as_finite_toric_ideal_sidecar": False,
                "safe_to_render_as_tropical_variety_embedding": False,
                "safe_to_render_as_global_toric_variety_embedding": False,
                "safe_to_use_as_normal_fan_certificate": False,
            }
        ),
        encoding="utf-8",
    )
    (periodic_dir / "got_audit" / "tropical_fan_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "tropicalgt.tropical_fan_visual_audit.v1",
                "available": False,
                "source_path": "unavailable",
                "ideal_spec": None,
                "cas_input_contract": {
                    "schema_version": "tropicalgt.tropical_fan_input_contract.v1",
                    "actual_data_only": True,
                    "no_proxy_or_fallback": True,
                    "proxy_substitution_allowed": False,
                    "explicit_cas_input_present": False,
                    "safe_to_render_certificate": False,
                    "input_sha256": None,
                },
                "diagnostics": {
                    "schema_version": "tropicalgt.cas_tropical_fan.v1",
                    "available": False,
                    "status": "unavailable_no_model_derived_tropical_ideal",
                    "backend": "Macaulay2",
                    "certificate_attached": False,
                    "fan_diagnostics_certified": False,
                    "tropical_cycle_certified": False,
                    "safe_to_render_as_tropical_fan": False,
                    "fan_summary": {"ray_count": 0, "ambient_dimension": 0},
                },
                "safe_to_render_as_tropical_fan": False,
            }
        ),
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
        report=None,
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
    assert bundle["report"].endswith("periodic/step_00005000/periodic_validation_artifacts.json")
    assert bundle["decision"]["bpb"] == 1.25
    assert bundle["decision"]["graph_bpb"] == 2.05
    contract = json.loads((module.ROOT / bundle["artifacts"]["contract_json"]).read_text(encoding="utf-8"))
    assert contract["compression_metrics"]["eval_bpb"] == 1.25
    assert contract["compression_metrics"]["eval_graph_bpb"] == 2.05
    assert bundle["commands"]["interactive_audit_backfills"]
    assert "backfill_interactive_audit_artifacts.py" in bundle["commands"]["interactive_audit_backfills"][0]
    assert bundle["commands"]["interactive_audit_validators"]
    sidecars = bundle["artifact_inventory"]["advanced_sidecars_tail"]
    assert any(path.endswith("periodic/step_00005000/got_audit/analogical_simplicial_maps.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/analogical_memory_retrieval.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/analogical_simplex_tree_analogy.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/got_full_trajectory_complex_payload.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/got_full_trajectory_complex_slider_contract.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/got_full_trajectory_complex_jensen_shannon_slider_contract.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/reasoning_step_complex_maps/manifest.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/inference_scaling_tree.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/tropical_support_payload.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/graphcg_direction_cosines_payload.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/got_nll_density_cloud_payload.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/chart_bundle_transport_sidecar.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/trajectory_persistence/persistence_landscapes.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/trajectory_level_radius_bifiltration.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/trajectory_topological_algebra.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/trajectory_growth_topology.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/inference_topology.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/inference_algebra.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/toric_embedding_sidecar.json") for path in sidecars)
    assert any(path.endswith("periodic/step_00005000/got_audit/tropical_fan_diagnostics.json") for path in sidecars)
    assert bundle["artifact_inventory"]["herschel_required_sidecars_present"]
    persisted_contract = json.loads((module.ROOT / bundle["artifacts"]["contract_json"]).read_text(encoding="utf-8"))
    assert any(
        path.endswith("periodic/step_00005000/got_audit/analogical_simplicial_maps.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/analogical_memory_retrieval.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/analogical_simplex_tree_analogy.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/got_full_trajectory_complex_payload.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/reasoning_step_complex_maps/manifest.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/tropical_support_payload.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/graphcg_direction_cosines_payload.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/got_nll_density_cloud_payload.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/chart_bundle_transport_sidecar.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/trajectory_persistence/persistence_landscapes.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/trajectory_level_radius_bifiltration.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/trajectory_topological_algebra.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/trajectory_growth_topology.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/inference_topology.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/inference_algebra.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/toric_embedding_sidecar.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    assert any(
        path.endswith("periodic/step_00005000/got_audit/tropical_fan_diagnostics.json")
        for path in persisted_contract["artifact_inventory"]["advanced_sidecars_tail"]
    )
    gflownet_branch = bundle["herschel_report_summary"]["artifact_evidence"]["gflownet_branch_selection_evidence"]
    assert gflownet_branch["available"] is True
    assert gflownet_branch["policy_counts"] == {"ranked_diverse_action_sweep": 1}
    assert gflownet_branch["total_selected_actions"] == 2
    tropical_support = bundle["herschel_report_summary"]["artifact_evidence"]["tropical_support_evidence"]
    assert tropical_support["available"] is True
    assert tropical_support["support_probability_source_counts"] == {"model_tropical_support_probabilities": 1}
    assert tropical_support["total_token_count"] == 6
    assert tropical_support["total_valid_support_assignment_count"] == 6
    assert tropical_support["low_strict_wall_interpretation_status_counts"] == {"low_strict_expected_near_wall_ambiguity": 1}
    graphcg_direction = bundle["herschel_report_summary"]["artifact_evidence"]["graphcg_direction_evidence"]
    assert graphcg_direction["available"] is True
    assert graphcg_direction["basis_source_counts"] == {"effective_full_rank_qr": 2}
    assert graphcg_direction["total_direction_count"] == 3
    assert graphcg_direction["total_candidate_count"] == 2
    assert graphcg_direction["sources"][0]["all_model_directions_have_rows"] is True
    nll_density = bundle["herschel_report_summary"]["artifact_evidence"]["nll_density_evidence"]
    assert nll_density["available"] is True
    assert nll_density["total_actual_model_anchor_count"] == 4
    assert nll_density["total_support_sample_count"] == 12
    assert nll_density["visible_density_layer_counts"] == {"actual_model_anchor_markers": 1, "density_volume": 1}
    assert nll_density["sources"][0]["support_sample_trace_visibility"] == "legendonly"
    persistence_landscape = bundle["herschel_report_summary"]["artifact_evidence"]["persistence_landscape_evidence"]
    assert persistence_landscape["available"] is False
    assert persistence_landscape["source_count"] == 1
    assert persistence_landscape["verified_unavailable_source_count"] == 1
    assert persistence_landscape["unavailable_reason_counts"]["no_finite_persistence_intervals_for_gudhi_landscape"] == 1
    assert persistence_landscape["sources"][0]["not_nll_fitness_landscape"] is True
    assert persistence_landscape["sources"][0]["verified_unavailable"] is True
    bivariate_module = bundle["herschel_report_summary"]["artifact_evidence"]["bivariate_module_evidence"]
    assert bivariate_module["available"] is True
    assert bivariate_module["module_available_source_count"] == 1
    assert bivariate_module["safe_unavailable_real_free_resolution_source_count"] == 1
    assert bivariate_module["resolution_status_counts"] == {"certificate_failed": 1}
    assert bivariate_module["sources"][0]["real_free_resolution_certified"] is False
    chart_bundle = bundle["herschel_report_summary"]["artifact_evidence"]["chart_bundle_transport_evidence"]
    assert chart_bundle["available"] is True
    assert chart_bundle["paper_ready_source_count"] == 0
    assert chart_bundle["total_chart_count"] == 3
    assert chart_bundle["total_transport_count"] == 2
    assert chart_bundle["completeness_tier_counts"] == {"telemetry_partial": 1}
    assert chart_bundle["missing_required_group_counts"] == {"graphcg_toric_agreement": 1}
    assert chart_bundle["sources"][0]["safe_to_render_as_toric_embedding_certificate"] is False
    toric_tropical = bundle["herschel_report_summary"]["artifact_evidence"]["toric_tropical_cas_evidence"]
    assert toric_tropical["available"] is False
    assert toric_tropical["source_count"] == 2
    assert toric_tropical["toric_source_count"] == 1
    assert toric_tropical["tropical_fan_source_count"] == 1
    assert toric_tropical["status_counts"] == {
        "toric_embedding_sidecar:unavailable_no_explicit_exponent_matrix": 1,
        "tropical_fan_diagnostics:unavailable_no_model_derived_tropical_ideal": 1,
    }
    assert toric_tropical["unavailable_reason_counts"]["toric_embedding_sidecar_unavailable"] == 1
    assert toric_tropical["unavailable_reason_counts"]["tropical_fan_diagnostics_unavailable"] == 1
    analogical_query = bundle["herschel_report_summary"]["artifact_evidence"]["analogical_query_context_evidence"]
    assert analogical_query["available"] is True
    assert analogical_query["sources"][0]["selected_query_complex_source"] == "trajectory_probability_filtered_simplicial_object"
    assert analogical_query["sources"][0]["selected_query_probability_vertex_count"] == 4
    analogical_memory = bundle["herschel_report_summary"]["artifact_evidence"]["analogical_memory_evidence"]
    assert analogical_memory["available"] is False
    assert analogical_memory["source_count"] == 3
    assert analogical_memory["verified_insufficient_memory_source_count"] == 3
    assert analogical_memory["total_bank_size"] == 0
    assert analogical_memory["total_retrieved_count"] == 0
    assert analogical_memory["total_top_k_rendered"] == 0
    assert analogical_memory["total_simplex_tree_pair_count"] == 0
    assert analogical_memory["quality_gate_reason_counts"] == {"no_non_self_model_memory": 1}
    assert analogical_memory["status_counts"] == {"unavailable_no_non_self_model_memory": 2, "unavailable_insufficient_model_probability_memory": 1}
    simplicial_complex = bundle["herschel_report_summary"]["artifact_evidence"]["simplicial_complex_evidence"]
    assert simplicial_complex["available"] is True
    assert simplicial_complex["source_count"] == 6
    assert simplicial_complex["available_source_count"] == 6
    assert simplicial_complex["total_view_count"] == 2
    assert simplicial_complex["total_step_count"] == 3
    assert simplicial_complex["total_radius_slider_contracts"] == 2
    assert simplicial_complex["total_simplex_tree_poset_contracts"] == 2
    assert simplicial_complex["total_source_simplices"] == 44
    assert simplicial_complex["total_displayed_simplices"] == 16
    topological_algebra = bundle["herschel_report_summary"]["artifact_evidence"]["topological_algebra_evidence"]
    assert topological_algebra["available"] is True
    assert topological_algebra["source_count"] == 4
    assert topological_algebra["available_source_count"] == 3
    assert topological_algebra["total_growth_rows"] == 1
    assert topological_algebra["total_topology_reports"] == 3
    assert topological_algebra["total_probability_topology_reports"] == 1
    assert topological_algebra["status_counts"] == {
        "inference_algebra_empty_unavailable": 1,
        "inference_topology_available": 1,
        "trajectory_growth_topology_available": 1,
        "trajectory_topological_algebra_available": 1,
    }

def test_prepare_review_bundle_blocks_command_execution_without_checkpoint(tmp_path: Path):
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    got_audit = output_dir / "periodic" / "step_00005000" / "got_audit"
    got_audit.mkdir(parents=True)
    checkpoint_dir.mkdir()
    report_path = output_dir / "train_report.json"
    report_path.write_text(json.dumps({"final_step": 5000, "eval": {"bpb": 1.3}}), encoding="utf-8")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"output_dir": str(output_dir), "checkpoint_dir": str(checkpoint_dir), "run_name": "unit_run", "model": {}}),
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
        run_eval_visualizations=True,
        run_legacy_audit_backfill=True,
        run_interactive_audit_validators=True,
        command_timeout_seconds=5,
    )
    with pytest.raises(RuntimeError, match="missing_checkpoint"):
        module.prepare_review_bundle(args)



def test_prepare_review_bundle_blocks_command_execution_with_empty_checkpoint(tmp_path: Path):
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    got_audit = output_dir / "periodic" / "step_00005000" / "got_audit"
    got_audit.mkdir(parents=True)
    checkpoint_dir.mkdir()
    report_path = output_dir / "train_report.json"
    report_path.write_text(json.dumps({"final_step": 5000, "eval": {"bpb": 1.3}}), encoding="utf-8")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"output_dir": str(output_dir), "checkpoint_dir": str(checkpoint_dir), "run_name": "unit_run", "model": {}}),
        encoding="utf-8",
    )
    (checkpoint_dir / "unit_run.latest.pt").write_bytes(b"")
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
        run_eval_visualizations=True,
        run_legacy_audit_backfill=True,
        run_interactive_audit_validators=True,
        command_timeout_seconds=5,
    )
    with pytest.raises(RuntimeError, match="empty_checkpoint"):
        module.prepare_review_bundle(args)


def test_prepare_review_bundle_records_empty_checkpoint_restart_gate_without_running_commands(tmp_path: Path):
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    report_path = output_dir / "train_report.json"
    report_path.write_text(json.dumps({"final_step": 5000, "eval": {"bpb": 1.3, "graph_bpb": 2.0}}), encoding="utf-8")
    (checkpoint_dir / "unit_run.latest.pt").write_bytes(b"")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"output_dir": str(output_dir), "checkpoint_dir": str(checkpoint_dir), "run_name": "unit_run", "model": {}}),
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
    checkpoint_evidence = bundle["checkpoint_evidence"]
    assert checkpoint_evidence["checkpoint_available"] is False
    assert checkpoint_evidence["safe_for_checkpoint_backed_restart"] is False
    assert "checkpoint_summary_unavailable:checkpoint_file_is_empty" in checkpoint_evidence["warnings"]
    gate = bundle["restart_evidence_gate"]
    assert gate["checkpoint_available"] is False
    assert gate["checkpoint_evidence_safe"] is False
    assert gate["restart_action"] == "blocked_missing_required_evidence_no_restart"
    assert gate["step0_restart_allowed"] is False
    assert any("checkpoint_file_is_empty" in blocker for blocker in gate["blockers"])
    assert any(blocker.startswith("checkpoint_evidence:checkpoint_summary_unavailable:checkpoint_file_is_empty") for blocker in gate["blockers"])
    assert any(blocker.startswith("execution_readiness:empty_checkpoint:") for blocker in gate["blockers"])


def test_prepare_review_bundle_blocks_restart_without_post_5k_command_results(tmp_path: Path):
    torch = pytest.importorskip("torch")
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    report_path = output_dir / "train_report.json"
    report_path.write_text(json.dumps({"final_step": 5000, "eval": {"bpb": 1.3, "graph_bpb": 2.0}}), encoding="utf-8")
    torch.save(
        {"model": {}, "config": {}, "step": 5000, "metrics": {"eval_bpb": 1.3, "eval_graph_bpb": 2.0}},
        checkpoint_dir / "unit_run.latest.pt",
    )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"output_dir": str(output_dir), "checkpoint_dir": str(checkpoint_dir), "run_name": "unit_run", "model": {}}),
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
    gate = bundle["restart_evidence_gate"]
    assert bundle["checkpoint_evidence"]["checkpoint_available"] is True
    assert bundle["checkpoint_evidence"]["safe_for_checkpoint_backed_restart"] is True
    assert bundle["checkpoint_evidence"]["checkpoint_step"] == 5000
    assert gate["checkpoint_available"] is True
    assert gate["checkpoint_evidence_safe"] is True
    assert gate["execution_evidence_ready"] is True
    assert gate["advanced_bpb_contract_safe"] is True
    assert gate["restart_action"] == "blocked_missing_required_evidence_no_restart"
    assert gate["step0_restart_allowed"] is False
    assert "post_5k_review_commands_missing_results:eval_validation_visualizations" in gate["blockers"]


def test_prepare_review_bundle_records_failed_advanced_bpb_contract(tmp_path: Path):
    module = _load_bundle_module()
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpts"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    report_path = output_dir / "train_report.json"
    report_path.write_text(json.dumps({"final_step": 5000, "eval": {"bpb": 1.3, "graph_bpb": 2.0}}), encoding="utf-8")
    cfg_path = tmp_path / "bad_bpb_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "run_name": "bad_bpb_review_bundle",
                "parameter_golf_bpb_focus": True,
                "target_bpb": 1.12,
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "batch_size": 1,
                "seq_len": 32,
                "tokengt": {"feature_dim": 48},
                "model": {"dim": 16, "hidden_dim": 16, "graph_feature_dim": 48},
                "wandb": {"enabled": False},
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
    contract = bundle["advanced_bpb_contract"]
    assert contract["section"]["required"] is True
    assert contract["safe_to_use_for_step0_bpb_restart"] is False
    assert "advanced_bpb_real_data_required" in contract["failed_gates"]
    assert "advanced_bpb_wandb_online_project" in contract["failed_gates"]
    gate = bundle["restart_evidence_gate"]
    assert gate["restart_action"] == "blocked_missing_required_evidence_no_restart"
    assert any(blocker.startswith("advanced_bpb_contract_failed:") for blocker in gate["blockers"])
    artifact_contract = json.loads((module.ROOT / bundle["artifacts"]["advanced_bpb_contract_json"]).read_text(encoding="utf-8"))
    assert artifact_contract["failed_gates"] == contract["failed_gates"]
    markdown = (module.ROOT / bundle["artifacts"]["bundle_markdown"]).read_text(encoding="utf-8")
    assert "advanced_bpb_real_data_required" in markdown


def test_run_shell_command_records_logs_and_return_code(tmp_path: Path):
    module = _load_bundle_module()
    command = f"{sys.executable} -c \"import sys; print(\\\"ok\\\"); print(\\\"warn\\\", file=sys.stderr)\""
    result = module._run_shell_command(command, tmp_path / "logs", "unit command", timeout_seconds=5)
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert Path(result["stdout_log"]).read_text(encoding="utf-8").strip() == "ok"
    assert Path(result["stderr_log"]).read_text(encoding="utf-8").strip() == "warn"


def test_requested_commands_run_backfill_before_validators(tmp_path: Path):
    module = _load_bundle_module()
    args = argparse.Namespace(
        run_eval_visualizations=False,
        run_legacy_audit_backfill=True,
        run_interactive_audit_validators=True,
        command_timeout_seconds=5,
    )
    commands = {
        "eval_validation_visualizations": f"{sys.executable} -c \"print('eval')\"",
        "interactive_audit_backfills": [f"{sys.executable} -c \"print('backfill')\""],
        "interactive_audit_validators": [f"{sys.executable} -c \"print('validator')\""],
    }
    results = module._run_requested_commands(args, commands, tmp_path / "bundle")
    assert [row["name"] for row in results] == ["interactive_audit_backfill_01", "interactive_audit_validator_01"]
