import json
from pathlib import Path

import pytest
import torch

import tropicalgt.run as run_module
from tropicalgt.run import build_model, train, _save_training_checkpoint


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
    assert first["advanced_bpb_contract"]["required"] is False
    assert not [gate for gate in first["advanced_bpb_contract_gates"] if gate["status"] == "fail"]
    assert Path(first["checkpoint"]).exists()
    latest_checkpoint = Path(first["latest_checkpoint"])
    assert latest_checkpoint.exists()
    assert latest_checkpoint.stat().st_size > 0
    latest_obj = torch.load(latest_checkpoint, map_location="cpu")
    assert latest_obj["step"] == 1
    assert latest_obj["run_name"] == "resume_test"

    cfg["max_steps"] = 2
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    second = train(config_path, resume_from=first["checkpoint"])
    assert second["resumed"] is True
    assert second["start_step"] == 1
    assert second["final_step"] == 2
    assert len(second["history"]) == 2



def test_short_max_steps_override_does_not_fail_full_budget_gate(tmp_path: Path):
    cfg = {
        "run_name": "budget_override_test",
        "data_root": None,
        "fixture_size": 4,
        "train_limit": 4,
        "val_limit": 2,
        "seq_len": 16,
        "batch_size": 2,
        "max_steps": 2,
        "min_available_train_token_slots": 64,
        "min_training_token_slots": 64,
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
    config_path = tmp_path / "budget_override.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    report = train(config_path, max_steps_override=1)

    assert report["final_step"] == 1
    assert report["data_budget"]["configured_training_token_slots"] == 64
    assert report["data_budget"]["effective_training_token_slots"] == 32


def test_train_blocks_bpb_config_that_fails_advanced_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = {
        "run_name": "bad_bpb_5k_gate",
        "parameter_golf_bpb_focus": True,
        "target_bpb": 1.12,
        "fixture_size": 2,
        "train_limit": 2,
        "val_limit": 1,
        "seq_len": 32,
        "batch_size": 1,
        "max_steps": 1,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "tokengt": {"feature_dim": 48},
        "model": {"dim": 16, "hidden_dim": 16, "graph_feature_dim": 48},
        "wandb": {"enabled": False},
    }
    config_path = tmp_path / "bad_bpb.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("training helper should not run after failed advanced BPB contract")

    monkeypatch.setattr(run_module, "make_dataset_from_config", fail_if_called)
    monkeypatch.setattr(run_module, "build_model", fail_if_called)
    monkeypatch.setattr(run_module, "setup_wandb", fail_if_called)
    monkeypatch.setattr(run_module, "load_checkpoint", fail_if_called)

    with pytest.raises(RuntimeError, match="Advanced BPB training contract failed") as exc:
        train(config_path)

    assert "advanced_bpb_real_data_required" in str(exc.value)
    assert "advanced_bpb_wandb_online_project" in str(exc.value)
    assert not (tmp_path / "outputs" / "train_report.json").exists()


def test_checkpoint_save_does_not_replace_existing_target_with_empty_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = {
        "run_name": "atomic_save_test",
        "device": "cpu",
        "tokengt": {"max_nodes": 16, "max_edges": 32, "node_id_dim": 8, "feature_dim": 48, "graph_token": True},
        "model": {"dim": 16, "hidden_dim": 16, "graph_feature_dim": 48, "num_actions": 8},
    }
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    target = tmp_path / "atomic.latest.pt"
    target.write_bytes(b"previous-checkpoint")

    def write_empty_checkpoint(_payload, path):
        Path(path).write_bytes(b"")

    monkeypatch.setattr(run_module.torch, "save", write_empty_checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint_file_empty"):
        _save_training_checkpoint(target, model, opt, cfg, {"step": 7.0}, [{"step": 7.0}], 7, "atomic_save_test")

    assert target.read_bytes() == b"previous-checkpoint"
    assert not list(tmp_path.glob(".atomic.latest.pt.tmp.*"))
