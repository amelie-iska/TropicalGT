import importlib.util
from pathlib import Path


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


def test_active_training_contract_inventories_latest_periodic_artifacts(tmp_path: Path):
    loop = _load_review_loop()
    output_dir = tmp_path / "run"
    got_audit = output_dir / "periodic" / "step_00005000" / "got_audit"
    got_audit.mkdir(parents=True)
    (got_audit / "tropical_fan_diagnostics.json").write_text("{}", encoding="utf-8")
    (got_audit / "tropical_fan_diagnostics.html").write_text("<html></html>", encoding="utf-8")
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
    assert inventory["interactive_audit_backfill_commands"]
    assert inventory["interactive_audit_validator_commands"]
    assert "backfill_interactive_audit_artifacts.py" in inventory["interactive_audit_backfill_commands"][0]
    assert "validate_interactive_audit_artifacts.py" in inventory["interactive_audit_validator_commands"][0]
    assert "generated artifacts are not staged or copied" in inventory["inventory_policy"]


def test_load_checkpoint_summary_reports_empty_checkpoint_unavailable(tmp_path: Path):
    loop = _load_review_loop()
    checkpoint = tmp_path / "empty.pt"
    checkpoint.write_bytes(b"")
    summary = loop._load_checkpoint_summary(checkpoint)
    assert summary["available"] is False
    assert summary["unavailable_reason"] == "checkpoint_file_is_empty"
    assert summary["path"].endswith("empty.pt")

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

def test_restart_decision_schema_requires_real_evidence_paths():
    loop = _load_review_loop()
    schema = loop._restart_decision_schema(1.12)
    assert schema["primary_metric"] == "eval.bpb"
    assert schema["primary_target"] == 1.12
    assert schema["config_patch_contract"]["requires_real_metric_or_artifact_for_each_change"] is True
    assert "blocked_missing_required_evidence_no_restart" in schema["allowed_actions"]
    assert "Unavailable CAS" in schema["no_proxy_policy"]


def test_review_loop_reads_periodic_and_validation_report_metrics():
    loop = _load_review_loop()
    periodic_report = {"metrics": {"eval_bpb": 1.29, "eval_graph_bpb": 2.4}}
    validation_report = {"bpb": 1.31, "bpb_exact": 1.31, "graph_bpb": 2.6, "nll": 0.91}
    assert loop._metric_value(periodic_report, {}, "eval.bpb") == 1.29
    assert loop._metric_value(periodic_report, {}, "eval.graph_bpb") == 2.4
    assert loop._metric_value(validation_report, {}, "eval.bpb") == 1.31
    assert loop._metric_value(validation_report, {}, "eval.graph_bpb") == 2.6


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
