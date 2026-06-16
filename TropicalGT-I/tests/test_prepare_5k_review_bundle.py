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
