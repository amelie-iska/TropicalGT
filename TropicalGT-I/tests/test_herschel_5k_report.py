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
        "command_results": [],
    }
    bundle_path = tmp_path / "review_bundle_step_00005000.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    markdown_path = tmp_path / "herschel_report.md"
    json_path = tmp_path / "herschel_report.json"

    summary = module.write_herschel_report(bundle_path, markdown_path, json_path)

    assert summary["schema_version"] == "tropicalgt.herschel_5k_report_summary.v1"
    assert summary["primary_metrics"]["bpb"] == 1.31
    assert summary["restart_decision"]["action"] == "blocked_missing_required_evidence_no_restart"
    assert summary["restart_decision"]["step0_restart_allowed"] is False
    assert "checkpoint_file_is_empty" in " ".join(summary["restart_decision"]["blockers"])
    groups = summary["artifact_evidence"]["sidecar_groups"]
    assert groups["cas_algebra"] == 2
    assert groups["topology_persistence"] == 1
    assert groups["analogical_memory"] == 1
    assert groups["tropical_toric"] == 1
    assert groups["graphcg"] == 1
    assert groups["nll_density"] == 1
    assert groups["chart_bundle"] == 2
    assert groups["vector_bundle"] == 1
    assert groups["sheaf_derived"] == 1
    assert "no training" in summary["policy"]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Herschel 5K Evidence Report" in markdown
    assert "flowchart TD" in markdown
    assert "checkpoint_file_is_empty" in markdown
    assert "blocked_missing_required_evidence_no_restart" in markdown
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
    assert (module.ROOT / artifacts["herschel_report_markdown"]).exists()
    assert (module.ROOT / artifacts["herschel_report_json"]).exists()
    assert bundle["herschel_report_summary"]["restart_decision"]["action"] == bundle["restart_evidence_gate"]["restart_action"]
    assert bundle["herschel_report_summary"]["checkpoint_evidence"]["restart_safe"] is False
    persisted_bundle = json.loads((module.ROOT / artifacts["bundle_json"]).read_text(encoding="utf-8"))
    assert persisted_bundle["artifacts"]["herschel_report_markdown"] == artifacts["herschel_report_markdown"]
    assert persisted_bundle["artifacts"]["herschel_report_json"] == artifacts["herschel_report_json"]
    assert persisted_bundle["herschel_report_summary"]["restart_decision"] == bundle["herschel_report_summary"]["restart_decision"]

    report_markdown = (module.ROOT / artifacts["herschel_report_markdown"]).read_text(encoding="utf-8")
    assert "# Herschel 5K Evidence Report" in report_markdown
    assert "checkpoint_summary_unavailable" in report_markdown
    bundle_markdown = (module.ROOT / artifacts["bundle_markdown"]).read_text(encoding="utf-8")
    assert "Herschel Report Summary" in bundle_markdown
    assert "herschel_report_markdown" in bundle_markdown
