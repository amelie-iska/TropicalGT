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
    assert contract["active_losses"]["gflownet_trajectory_balance"] == 0.1
    assert contract["data_metrics"]["causal_dag_ar_rate"] == 0.75
    assert contract["tropical_metrics"]["sequence_tropical_margin_mean"] == 0.3
    assert "Active Losses" in loop._active_training_contract_markdown(contract)
