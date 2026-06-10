import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_golf_size_estimator_reports_cap_margin(tmp_path: Path):
    config = {
        "model": {
            "dim": 16,
            "hidden_dim": 16,
            "graph_feature_dim": 48,
            "num_actions": 8,
            "memory_dim": 8,
        }
    }
    config_path = tmp_path / "tiny_config.json"
    output_path = tmp_path / "estimate.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "estimate_parameter_golf_export_size.py"),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT.parent, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert output_path.exists()
    assert report["within_cap"] is True
    assert report["parameter_count"] > 0
    assert report["total_competition_bytes"] == report["artifact_bytes"] + report["code_bytes"]
    assert report["cap_margin_bytes"] > 0
