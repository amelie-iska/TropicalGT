import json
from pathlib import Path
import subprocess
import sys


def test_bpb_ablation_grid_dry_run_writes_isolated_configs(tmp_path: Path):
    cfg = {
        "run_name": "grid_test",
        "seed": 123,
        "data_root": "unused",
        "require_data": True,
        "train_limit": 4,
        "val_limit": 2,
        "max_steps": 1,
        "output_dir": str(tmp_path / "base"),
        "checkpoint_dir": str(tmp_path / "base_ckpt"),
        "memory_bank_path": str(tmp_path / "base_memory.jsonl"),
        "tokengt": {"feature_dim": 48},
        "model": {
            "dim": 16,
            "hidden_dim": 16,
            "graph_feature_dim": 48,
            "graphcg_weight": 0.02,
            "gflownet_weight": 0.02,
            "certificate_weight": 0.001,
        },
        "wandb": {"enabled": True},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_bpb_ablation_grid.py"
    out_dir = tmp_path / "grid"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
            "--variants",
            "baseline,chart_bundle_telemetry,chart_bundle_toric_0p1x,no_chart_bundle_toric",
            "--fixture",
            "--device",
            "cpu",
            "--max-steps",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = json.loads(result.stdout)
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["ran_training"] is False
    assert len(manifest["variants"]) == 4
    baseline_cfg = json.loads(Path(manifest["variants"][0]["config"]).read_text(encoding="utf-8"))
    telemetry_cfg = json.loads(Path(manifest["variants"][1]["config"]).read_text(encoding="utf-8"))
    toric_cfg = json.loads(Path(manifest["variants"][2]["config"]).read_text(encoding="utf-8"))
    no_chart_cfg = json.loads(Path(manifest["variants"][3]["config"]).read_text(encoding="utf-8"))
    assert baseline_cfg["data_root"] is None
    assert baseline_cfg["wandb"]["enabled"] is False
    assert baseline_cfg["memory_bank_path"].endswith("baseline/analogical_memory/reasoning_memory.jsonl")
    assert telemetry_cfg["model"]["enable_chart_bundle_auxiliary"] is True
    assert telemetry_cfg["model"]["bundle_transport_weight"] == 0.0
    assert telemetry_cfg["model"]["chart_bpb_consistency_weight"] == 0.0
    assert toric_cfg["model"]["enable_chart_bundle_auxiliary"] is True
    assert toric_cfg["model"]["bundle_transport_weight"] == 0.0001
    assert toric_cfg["model"]["bundle_cocycle_weight"] == 0.0001
    assert toric_cfg["model"]["bundle_flat_rank_weight"] == 0.0001
    assert toric_cfg["model"]["toric_normal_fan_weight"] == 0.0001
    assert toric_cfg["model"]["graphcg_toric_cell_agreement_weight"] == 0.0001
    assert toric_cfg["model"]["chart_bpb_consistency_weight"] == 0.0001
    assert toric_cfg["model"]["bundle_atom_stability_weight"] == 0.0001
    assert no_chart_cfg["model"]["enable_chart_bundle_auxiliary"] is False
    assert no_chart_cfg["model"]["bundle_transport_weight"] == 0.0
    assert no_chart_cfg["seed"] == baseline_cfg["seed"] == 123


def test_bpb_ablation_grid_blocks_contract_breaking_bpb_configs_before_writing(tmp_path: Path):
    cfg = {
        "run_name": "grid_bpb_5k_gate",
        "parameter_golf_bpb_focus": True,
        "target_bpb": 1.12,
        "seed": 123,
        "data_root": "unused",
        "require_data": True,
        "train_limit": 4,
        "val_limit": 2,
        "max_steps": 1,
        "output_dir": str(tmp_path / "base"),
        "checkpoint_dir": str(tmp_path / "base_ckpt"),
        "tokengt": {"feature_dim": 48},
        "model": {
            "dim": 16,
            "hidden_dim": 16,
            "graph_feature_dim": 48,
            "graphcg_weight": 0.02,
            "gflownet_weight": 0.02,
            "certificate_weight": 0.001,
        },
        "wandb": {"enabled": False},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_bpb_ablation_grid.py"
    out_dir = tmp_path / "grid"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
            "--variants",
            "no_graphcg",
            "--fixture",
            "--device",
            "cpu",
            "--max-steps",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "advanced BPB contract failed before ablation configs were written" in result.stderr
    assert not (out_dir / "ablation_grid_manifest.json").exists()
    assert not (out_dir / "configs").exists()

    allowed = subprocess.run(
        result.args + ["--allow-contract-breaking-ablation-configs"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = json.loads(allowed.stdout)
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    variant = manifest["variants"][0]
    assert variant["contract_safe_to_run"] is False
    assert "advanced_bpb_graphcg_weight_positive" in variant["advanced_bpb_contract"]["failed_gates"]
    assert "advanced_bpb_wandb_online_project" in variant["advanced_bpb_contract"]["failed_gates"]
