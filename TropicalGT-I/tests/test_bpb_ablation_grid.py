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
            "baseline,no_graphcg",
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
    assert len(manifest["variants"]) == 2
    baseline_cfg = json.loads(Path(manifest["variants"][0]["config"]).read_text(encoding="utf-8"))
    no_graphcg_cfg = json.loads(Path(manifest["variants"][1]["config"]).read_text(encoding="utf-8"))
    assert baseline_cfg["data_root"] is None
    assert baseline_cfg["wandb"]["enabled"] is False
    assert baseline_cfg["memory_bank_path"].endswith("baseline/analogical_memory/reasoning_memory.jsonl")
    assert no_graphcg_cfg["model"]["graphcg_weight"] == 0.0
    assert no_graphcg_cfg["seed"] == baseline_cfg["seed"] == 123
