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
                "got_audit/graphcg_report.json",
                "got_audit/nll_density_grid.json",
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
    assert groups["tropical_toric"] == 1
    assert groups["graphcg"] == 1
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
