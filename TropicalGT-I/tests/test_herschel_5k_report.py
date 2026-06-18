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
        json.dumps({"query_context_contract": query_context_contract, "topk_contract": {"query_context_contract_schema_version": "tropicalgt.analogical_query_context_conversion.v1", "query_context_contract": query_context_contract}}),
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
                "got_audit/tropical_fan_diagnostics.json",
                "got_audit/betti_table.json",
                "got_audit/certificate_indexed_cas_evidence.json",
                "got_audit/persistence_landscape.json",
                "got_audit/analogical_memory_report.json",
                str(analogical_maps_path),
                str(inference_scaling_tree_path),
                str(tropical_support_path),
                str(graphcg_direction_path),
                str(nll_density_path),
                "got_audit/chart_bundle_metrics.json",
                "got_audit/chart_bundle_transport_sidecar.json",
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
    assert groups["topology_persistence"] == 1
    assert groups["analogical_memory"] == 2
    assert groups["tropical_toric"] == 2
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
    assert "## Analogical Query Context Evidence" in markdown
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
    assert "Analogical Query Context Evidence" in html
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
