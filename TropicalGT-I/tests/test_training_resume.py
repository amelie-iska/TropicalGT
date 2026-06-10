import json
from pathlib import Path

from tropicalgt.run import train


def test_training_checkpoint_resume(tmp_path: Path):
    cfg = {
        "run_name": "resume_test",
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
        "model": {"dim": 16, "hidden_dim": 16, "graph_feature_dim": 48, "num_actions": 8, "gflownet_weight": 0.02, "graphcg_weight": 0.02},
        "wandb": {"enabled": False},
    }
    config_path = tmp_path / "resume.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    first = train(config_path)
    assert first["final_step"] == 1
    assert Path(first["checkpoint"]).exists()
    assert Path(first["latest_checkpoint"]).exists()

    cfg["max_steps"] = 2
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    second = train(config_path, resume_from=first["checkpoint"])
    assert second["resumed"] is True
    assert second["start_step"] == 1
    assert second["final_step"] == 2
    assert len(second["history"]) == 2
