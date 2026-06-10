import json
from pathlib import Path

from tropicalgt.run import train


def test_training_history_contains_certificate_and_throughput_metrics(tmp_path: Path):
    cfg = {
        "run_name": "metrics_test",
        "data_root": None,
        "fixture_size": 4,
        "train_limit": 4,
        "val_limit": 2,
        "seq_len": 32,
        "batch_size": 2,
        "max_steps": 1,
        "lr": 0.0005,
        "grad_clip": 1.0,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "checkpoint_every": 1,
        "tokengt": {"max_nodes": 16, "max_edges": 32, "node_id_dim": 8, "feature_dim": 48, "graph_token": True},
        "model": {
            "dim": 16,
            "hidden_dim": 16,
            "graph_feature_dim": 48,
            "num_actions": 8,
            "gflownet_weight": 0.02,
            "graphcg_weight": 0.02,
            "certificate_weight": 0.001,
        },
        "wandb": {"enabled": False},
    }
    config_path = tmp_path / "metrics.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    report = train(config_path)
    row = report["history"][0]
    for key in [
        "certificate_loss",
        "certificate_agreement",
        "wall_hit_rate",
        "examples_per_sec",
        "tokens_per_sec",
        "graph_tokens_per_sec",
        "grad_norm",
        "optimizer_lr",
        "loss_regularizer_total",
        "bpb",
        "text_bpb",
        "graph_bpb",
        "graph_sideinfo_bpb",
        "graph_conditioned_bpb_no_side_cost",
        "graph_token_structural_bytes",
        "explicit_graph_json_bytes",
        "analogical_memory_query_norm",
    ]:
        assert key in row
        assert row[key] == row[key]
    assert report["eval"]["bpb"] == report["eval"]["bpb_proxy"]
    assert report["eval"]["graph_bpb"] == report["eval"]["graph_bpb"]
