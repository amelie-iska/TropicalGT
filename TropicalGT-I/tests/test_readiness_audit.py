from pathlib import Path

from audit_tropicalgt_i_readiness import build_readiness_report, render_markdown


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
