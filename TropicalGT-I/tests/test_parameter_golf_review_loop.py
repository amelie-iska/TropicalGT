import argparse
import importlib.util
from pathlib import Path

import pytest
import torch


def _load_review_loop():
    path = Path(__file__).resolve().parents[1] / "scripts" / "parameter_golf_codex_review_loop.py"
    spec = importlib.util.spec_from_file_location("parameter_golf_codex_review_loop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_review_loop_metric_lookup_and_train_command():
    loop = _load_review_loop()
    report = {"eval": {"bpb": 1.3}, "metrics": {"eval_graph_bpb": 1.9}}
    assert loop._metric_value(report, {}, "eval.bpb") == 1.3
    assert loop._metric_value(report, {}, "eval.graph_bpb") == 1.9
    cmd = loop._train_command("python", Path("train.py"), Path("cfg.json"), 5000, Path("ckpt.pt"))
    assert cmd == ["python", "train.py", "--config", "cfg.json", "--max-steps", "5000", "--resume-from", "ckpt.pt"]


def test_active_training_contract_reports_losses_and_graph_order_metrics():
    loop = _load_review_loop()
    cfg = {
        "batch_size": 128,
        "seq_len": 1024,
        "lr": 3e-4,
        "model": {"dim": 1760, "gflownet_weight": 0.02, "graphcg_weight": 0.02},
        "hybrid_data": {"enabled": True},
    }
    report = {
        "metrics": {
            "nll": 2.0,
            "loss": 2.1,
            "bpb": 2.88,
            "graph_bpb": 3.0,
            "gflownet_tb": 0.1,
            "graphcg_loss": 0.2,
            "sequence_tropical_margin_mean": 0.3,
            "causal_dag_ar_rate": 0.75,
            "random_graph_ar_rate": 0.25,
            "gpu_mem_mb": 18000.0,
        },
        "eval": {"bpb": 1.4, "graph_bpb": 1.6},
        "history": [],
    }
    contract = loop._active_training_contract(cfg, report, {}, 5000)
    assert contract["compression_metrics"]["eval_bpb"] == 1.4
    assert contract["objective"]["primary_target"] == 1.12
    assert contract["active_losses"]["gflownet_trajectory_balance"] == 0.1
    assert contract["data_metrics"]["causal_dag_ar_rate"] == 0.75
    assert contract["tropical_metrics"]["sequence_tropical_margin_mean"] == 0.3
    assert contract["restart_decision_schema"]["config_patch_contract"]["requires_evidence_paths"] is True
    assert "artifact_inventory" in contract
    markdown = loop._active_training_contract_markdown(contract)
    assert "artifact_inventory" in markdown
    assert "Restart Decision Schema" in markdown
    assert "Active Losses" in markdown


def test_active_training_contract_surfaces_checkpoint_evidence_for_unavailable_checkpoint():
    loop = _load_review_loop()
    report = {
        "final_step": 5000,
        "metrics": {"eval_bpb": 1.43, "eval_graph_bpb": 20.1},
        "latest_checkpoint_integrity": {
            "available": False,
            "path": "bad.latest.pt",
            "unavailable_reason": "checkpoint_file_empty:bad.latest.pt",
            "expected_step": 5000,
            "verified_load": True,
        },
    }
    checkpoint = {"available": False, "path": "bad.latest.pt", "unavailable_reason": "checkpoint_file_is_empty"}

    contract = loop._active_training_contract(
        {"model": {}, "batch_size": 4, "seq_len": 32},
        report,
        checkpoint,
        5000,
        checkpoint_path=Path("bad.latest.pt"),
    )

    evidence = contract["checkpoint_evidence"]
    assert evidence["schema_version"] == "tropicalgt.checkpoint_evidence.v1"
    assert evidence["checkpoint_available"] is False
    assert evidence["safe_for_checkpoint_backed_restart"] is False
    assert "checkpoint_summary_unavailable:checkpoint_file_is_empty" in evidence["warnings"]
    assert any(warning.startswith("latest_checkpoint_integrity_unavailable:checkpoint_file_empty") for warning in evidence["warnings"])
    markdown = loop._active_training_contract_markdown(contract)
    assert "Checkpoint Evidence" in markdown
    assert "checkpoint_file_is_empty" in markdown


def test_checkpoint_evidence_warns_on_stale_checkpoint_and_missing_eval_metrics():
    loop = _load_review_loop()
    report = {
        "final_step": 5000,
        "metrics": {"eval_bpb": 1.2, "eval_graph_bpb": 2.3},
        "latest_checkpoint_integrity": {"available": True, "path": "fresh.latest.pt", "observed_step": 4750},
    }
    checkpoint = {"available": True, "path": "stale.latest.pt", "step": 4750, "metrics": {"loss": 1.0}}

    evidence = loop._checkpoint_evidence_summary(report, checkpoint, Path("stale.latest.pt"), 5000)

    assert evidence["checkpoint_available"] is True
    assert evidence["safe_for_checkpoint_backed_restart"] is False
    assert "checkpoint_step_before_boundary:4750<5000" in evidence["warnings"]
    assert "checkpoint_step_mismatch_boundary:4750!=5000" in evidence["warnings"]
    assert "latest_checkpoint_integrity_path_mismatch:fresh.latest.pt!=stale.latest.pt" in evidence["warnings"]
    assert "latest_checkpoint_integrity_observed_step_mismatch_boundary:4750!=5000" in evidence["warnings"]
    assert "checkpoint_missing_report_metric:eval_bpb" in evidence["warnings"]
    assert "checkpoint_missing_report_metric:eval_graph_bpb" in evidence["warnings"]


def test_active_training_contract_inventories_latest_periodic_artifacts(tmp_path: Path):
    loop = _load_review_loop()
    output_dir = tmp_path / "run"
    got_audit = output_dir / "periodic" / "step_00005000" / "got_audit"
    got_audit.mkdir(parents=True)
    (got_audit / "tropical_fan_diagnostics.json").write_text("{}", encoding="utf-8")
    (got_audit / "tropical_fan_diagnostics.html").write_text("<html></html>", encoding="utf-8")
    (got_audit / "toric_embedding_sidecar.json").write_text("{}", encoding="utf-8")
    (got_audit / "toric_embedding_sidecar.html").write_text("<html></html>", encoding="utf-8")
    (output_dir / "periodic" / "step_00002500").mkdir(parents=True)
    report_path = output_dir / "train_report.json"
    report_path.write_text("{}", encoding="utf-8")
    contract = loop._active_training_contract(
        {"output_dir": str(output_dir)},
        {"visualizations": {"metrics": "metrics/training_metrics.html"}},
        {},
        5000,
        report_path=report_path,
    )
    inventory = contract["artifact_inventory"]
    assert inventory["latest_periodic_dir"].endswith("step_00005000")
    assert inventory["latest_got_audit_dir"].endswith("got_audit")
    assert any(path.endswith("tropical_fan_diagnostics.json") for path in inventory["advanced_sidecars_tail"])
    assert any(path.endswith("toric_embedding_sidecar.json") for path in inventory["advanced_sidecars_tail"])
    assert inventory["interactive_audit_backfill_commands"]
    assert inventory["interactive_audit_validator_commands"]
    assert "backfill_interactive_audit_artifacts.py" in inventory["interactive_audit_backfill_commands"][0]
    assert "validate_interactive_audit_artifacts.py" in inventory["interactive_audit_validator_commands"][0]
    assert "generated artifacts are not staged or copied" in inventory["inventory_policy"]


def test_checkpoint_step_returns_zero_for_invalid_checkpoint(tmp_path: Path):
    loop = _load_review_loop()
    checkpoint = tmp_path / "empty.pt"
    checkpoint.write_bytes(b"")

    assert loop._checkpoint_step(checkpoint) == 0


def test_load_checkpoint_summary_reports_empty_checkpoint_unavailable(tmp_path: Path):
    loop = _load_review_loop()
    checkpoint = tmp_path / "empty.pt"
    checkpoint.write_bytes(b"")
    summary = loop._load_checkpoint_summary(checkpoint)
    assert summary["available"] is False
    assert summary["unavailable_reason"] == "checkpoint_file_is_empty"
    assert summary["path"].endswith("empty.pt")

def test_load_checkpoint_summary_reports_malformed_checkpoint_unavailable(tmp_path: Path):
    loop = _load_review_loop()
    checkpoint = tmp_path / "malformed.pt"
    torch.save({"step": 5, "metrics": {"eval_bpb": 1.2}}, checkpoint)

    summary = loop._load_checkpoint_summary(checkpoint)

    assert summary["available"] is False
    assert summary["unavailable_reason"] == "checkpoint_invalid_payload:missing_model,config"


def test_snapshot_checkpoint_status_refuses_empty_checkpoint(tmp_path: Path):
    loop = _load_review_loop()
    checkpoint = tmp_path / "empty.latest.pt"
    output_dir = tmp_path / "review"
    checkpoint.write_bytes(b"")

    status = loop._snapshot_checkpoint_status(checkpoint, output_dir, 5000)

    assert status["available"] is False
    assert status["snapshot"] == ""
    assert status["unavailable_reason"] == "checkpoint_file_is_empty"
    assert not list((output_dir / "checkpoints").glob("*.pt"))


def test_snapshot_checkpoint_status_copies_only_valid_training_checkpoint(tmp_path: Path):
    loop = _load_review_loop()
    checkpoint = tmp_path / "valid.latest.pt"
    output_dir = tmp_path / "review"
    torch.save({"model": {}, "config": {}, "step": 5000, "metrics": {"eval_bpb": 1.2}}, checkpoint)

    status = loop._snapshot_checkpoint_status(checkpoint, output_dir, 5000)

    assert status["available"] is True
    assert status["step"] == 5000
    snapshot = Path(status["snapshot"])
    assert snapshot.exists()
    assert snapshot.stat().st_size > 0
    assert loop._checkpoint_step(snapshot) == 5000


def test_checkpoint_snapshot_block_records_unavailable_reason():
    loop = _load_review_loop()
    block = loop._checkpoint_snapshot_block(
        {"path": "bad.latest.pt", "unavailable_reason": "checkpoint_file_is_empty", "boundary_step": 5000},
        {"boundary_step": 5000, "bpb": 1.0, "target_bpb": 1.12},
        "target_met_requires_loadable_boundary_checkpoint",
    )

    assert block["restart_action"] == "blocked_missing_loadable_boundary_checkpoint"
    assert block["unavailable_reason"] == "checkpoint_file_is_empty"
    assert "nonempty, loadable" in block["policy"]


def test_review_prompt_requires_subagent_evidence_review_and_step0_restart():
    loop = _load_review_loop()
    prompt = loop._review_prompt(
        cfg={"model": {}, "output_dir": "TropicalGT-I/outputs/unit"},
        report={"metrics": {}, "eval": {}},
        checkpoint={"history_tail": []},
        active_contract={"artifact_inventory": {"advanced_sidecars_tail": ["got_audit/tropical_fan_diagnostics.json"]}},
        report_path=Path("report.json"),
        checkpoint_path=Path("checkpoint.pt"),
        previous_boundary_checkpoint=None,
        boundary_step=5000,
        metric="eval.bpb",
        bpb=1.3,
        graph_metric="eval.graph_bpb",
        graph_bpb=2.0,
        target_bpb=1.12,
        restart_policy="beginning",
    )
    assert "spawn or assign a fresh Codex subagent" in prompt
    assert "advanced sidecars" in prompt
    assert "topological, geometric, algebraic" in prompt
    assert "Restart from step 0" in prompt
    assert "restart_decision_schema" in prompt
    assert "evidence_paths" in prompt
    assert "No proxies or fallbacks" in prompt

def test_review_loop_blocks_same_config_restart_after_target_miss_by_default():
    loop = _load_review_loop()
    args = argparse.Namespace(allow_same_config_restart_after_triggered_review=False)
    decision = {"triggered": True, "boundary_step": 5000, "bpb": 1.3, "target_bpb": 1.12, "prompt": "review.md"}

    block = loop._triggered_restart_block(args, decision)

    assert block is not None
    assert block["restart_action"] == "blocked_pending_evidence_backed_config_patch"
    assert block["boundary_step"] == 5000
    assert block["bpb"] == 1.3
    assert "reviewed config patch" in block["policy"]


def test_review_loop_same_config_restart_requires_explicit_opt_in():
    loop = _load_review_loop()
    decision = {"triggered": True, "boundary_step": 5000, "bpb": 1.3, "target_bpb": 1.12}
    blocked_args = argparse.Namespace(allow_same_config_restart_after_triggered_review=False)
    opt_in_args = argparse.Namespace(allow_same_config_restart_after_triggered_review=True)

    assert loop._triggered_restart_block(blocked_args, decision) is not None
    assert loop._triggered_restart_block(opt_in_args, decision) is None
    assert loop._triggered_restart_block(blocked_args, {"triggered": False}) is None


def test_restart_decision_schema_requires_real_evidence_paths():
    loop = _load_review_loop()
    schema = loop._restart_decision_schema(1.12)
    assert schema["primary_metric"] == "eval.bpb"
    assert schema["primary_target"] == 1.12
    assert schema["config_patch_contract"]["requires_real_metric_or_artifact_for_each_change"] is True
    assert "reviewed_config_patch_before_any_same_config_restart" in schema["required_evidence"]
    assert "blocked_pending_evidence_backed_config_patch" in schema["allowed_actions"]
    assert "blocked_missing_required_evidence_no_restart" in schema["allowed_actions"]
    assert "halt on a missed target" in schema["no_proxy_policy"]
    assert "Unavailable CAS" in schema["no_proxy_policy"]


def test_review_loop_reads_periodic_and_validation_report_metrics():
    loop = _load_review_loop()
    periodic_report = {"metrics": {"eval_bpb": 1.29, "eval_graph_bpb": 2.4}}
    validation_report = {"bpb": 1.31, "bpb_exact": 1.31, "graph_bpb": 2.6, "nll": 0.91}
    assert loop._metric_value(periodic_report, {}, "eval.bpb") == 1.29
    assert loop._metric_value(periodic_report, {}, "eval.graph_bpb") == 2.4
    assert loop._metric_value(validation_report, {}, "eval.bpb") == 1.31
    assert loop._metric_value(validation_report, {}, "eval.graph_bpb") == 2.6


def test_review_loop_enforces_advanced_contract_before_training_wrapper_launch():
    loop = _load_review_loop()
    cfg = {
        "run_name": "bad_bpb_wrapper_gate",
        "parameter_golf_bpb_focus": True,
        "target_bpb": 1.12,
        "fixture_size": 2,
        "train_limit": 2,
        "val_limit": 1,
        "seq_len": 32,
        "batch_size": 1,
        "device": "cpu",
        "tokengt": {"feature_dim": 48},
        "model": {"dim": 16, "hidden_dim": 16, "graph_feature_dim": 48},
        "wandb": {"enabled": False},
    }

    with pytest.raises(RuntimeError, match="Advanced BPB training contract failed") as exc:
        loop._enforce_train_launch_contract(cfg)

    assert "advanced_bpb_real_data_required" in str(exc.value)
    assert "advanced_bpb_wandb_online_project" in str(exc.value)


def test_active_training_contract_reads_top_level_validation_metrics():
    loop = _load_review_loop()
    report = {
        "bpb": 1.27,
        "graph_bpb": 2.1,
        "graph_sideinfo_bpb": 2.3,
        "ppl": 2.4,
        "nll": 0.88,
        "parameter_golf_source_rate": 0.5,
        "causal_dag_ar_rate": 1.0,
    }
    contract = loop._active_training_contract({"model": {}, "batch_size": 4, "seq_len": 32}, report, {}, 5000)
    compression = contract["compression_metrics"]
    assert compression["eval_bpb"] == 1.27
    assert compression["eval_graph_bpb"] == 2.1
    assert compression["eval_graph_sideinfo_bpb"] == 2.3
    assert compression["eval_ppl"] == 2.4
    assert contract["active_losses"]["nll"] == 0.88
    assert contract["data_metrics"]["parameter_golf_source_rate"] == 0.5
    assert contract["data_metrics"]["causal_dag_ar_rate"] == 1.0
