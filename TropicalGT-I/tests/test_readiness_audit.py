from pathlib import Path

from audit_tropicalgt_i_readiness import build_readiness_report, render_markdown
from tropicalgt.run import load_keys


def test_load_keys_accepts_colon_and_aliases(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("GH:github-value\nHF:hf-value\nwandb:wandb-value\n", encoding="utf-8")
    loaded = load_keys(keys)
    assert loaded["wandb"] == "wandb-value"
    assert loaded["github"] == "github-value"
    assert loaded["huggingface"] == "hf-value"


def test_readiness_audit_fixture_without_checkpoint(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "run_name": "audit_fixture",
  "fixture_size": 4,
  "train_limit": 4,
  "val_limit": 4,
  "batch_size": 2,
  "seq_len": 32,
  "seed": 1729,
  "device": "cpu",
  "output_dir": "%s",
  "model": {"dim": 32, "hidden_dim": 32, "graph_feature_dim": 48},
  "tokengt": {"feature_dim": 48}
}
"""
        % (tmp_path / "outputs"),
        encoding="utf-8",
    )
    report = build_readiness_report(
        config_path=config,
        checkpoint_path=None,
        split="validation",
        sample_limit=4,
        details_limit=1,
        trace_limit=4,
        scale_depth=0,
        scale_width=2,
        scale_branch_factor=2,
        require_cuda=False,
        require_checkpoint=False,
        render_visualizations=False,
    )
    assert report["status"] == "ready"
    assert report["data"]["sample_records"] == 4
    assert report["data"]["graph_json_fallback_rate"] == 0
    assert not report["failed_gates"]
    markdown = render_markdown(report)
    assert "TropicalGT-I Readiness Audit" in markdown
    assert "| config_loads | pass |" in markdown


def test_readiness_audit_train_dry_run_fixture(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "run_name": "audit_dry_run",
  "fixture_size": 4,
  "train_limit": 4,
  "val_limit": 4,
  "batch_size": 2,
  "seq_len": 32,
  "seed": 1729,
  "device": "cpu",
  "lr": 0.0003,
  "output_dir": "%s",
  "model": {"dim": 32, "hidden_dim": 32, "graph_feature_dim": 48},
  "tokengt": {"feature_dim": 48},
  "wandb": {"enabled": false}
}
"""
        % (tmp_path / "outputs"),
        encoding="utf-8",
    )
    report = build_readiness_report(
        config_path=config,
        checkpoint_path=None,
        split="train",
        sample_limit=4,
        details_limit=1,
        trace_limit=4,
        scale_depth=0,
        scale_width=2,
        scale_branch_factor=2,
        train_dry_run=True,
        check_wandb_key=True,
        require_cuda=False,
        require_checkpoint=False,
        render_visualizations=False,
    )
    assert report["status"] == "ready"
    assert report["train_dry_run"]["loss"] == report["train_dry_run"]["loss"]
    assert report["train_dry_run"]["compression"]["bpb"] > 0
    assert any(gate["name"] == "train_dry_run_forward_backward" and gate["status"] == "pass" for gate in report["gates"])
