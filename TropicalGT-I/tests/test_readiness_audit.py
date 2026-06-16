import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import audit_tropicalgt_i_readiness as readiness_module

from audit_tropicalgt_i_readiness import advanced_bpb_contract_report, build_readiness_report, render_markdown
from tropicalgt.run import load_keys


def test_load_keys_accepts_colon_and_aliases(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("GH:github-value\nHF:hf-value\nwandb:wandb-value\n", encoding="utf-8")
    loaded = load_keys(keys)
    assert loaded["wandb"] == "wandb-value"
    assert loaded["github"] == "github-value"
    assert loaded["huggingface"] == "hf-value"


def test_validate_tropicalgt_i_reports_legacy_graph_json_guardrail_alias(tmp_path):
    config = tmp_path / "config.json"
    output = tmp_path / "validate.json"
    config.write_text(
        json.dumps(
            {
                "fixture_size": 4,
                "train_limit": 4,
                "val_limit": 4,
                "batch_size": 2,
                "seq_len": 32,
                "tokengt": {"max_nodes": 16, "max_edges": 32, "node_id_dim": 8, "feature_dim": 48, "graph_token": True},
            }
        ),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_tropicalgt_i.py"

    subprocess.run(
        [sys.executable, str(script), "--config", str(config), "--split", "train", "--output", str(output)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["graph_json_fallback_records"] == 0
    assert report["legacy_graph_json_substitution_guardrail_records"] == 0
    assert report["legacy_graph_json_substitution_guardrail_rate"] == report["invalid_graph_rate"] == 0.0
    assert report["samples"][0]["graph_json_fallback"] is False
    assert report["samples"][0]["legacy_graph_json_substitution_guardrail"] is False


def test_advanced_bpb_contract_passes_current_b54_gate_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train_full_dataset_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    section, gates = advanced_bpb_contract_report(cfg)

    assert section["required"] is True
    assert section["target_bpb"] == 1.12
    assert not [gate for gate in gates if gate["status"] == "fail"]


def test_advanced_bpb_contract_blocks_mismatched_wandb_run_name():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train_full_dataset_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["wandb_run_name"] = "stale_or_wrong_wandb_name"

    _, gates = advanced_bpb_contract_report(cfg)
    failed = {gate["name"] for gate in gates if gate["status"] == "fail"}

    assert "advanced_bpb_wandb_run_name_matches_config" in failed


def test_advanced_bpb_contract_blocks_disabled_advanced_methods():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train_full_dataset_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["visualization_every_steps"] = 1000
    cfg["periodic_interactive_artifacts_enabled"] = False
    cfg["model"]["graphcg_weight"] = 0.0
    cfg["model"]["graphcg_num_directions"] = cfg["model"]["dim"] - 1
    cfg["meet_in_middle"]["enabled"] = False
    cfg["meet_in_middle"]["noncausal_use_random_order_autoregression"] = False
    cfg["memory_quality_min_probability_simplices"] = 0
    cfg["hybrid_data"]["sources"][0]["name"] = "hf_only"
    cfg.pop("wandb_name", None)
    cfg.pop("wandb_run_name", None)
    cfg["wandb"].pop("entity", None)

    _, gates = advanced_bpb_contract_report(cfg)
    failed = {gate["name"] for gate in gates if gate["status"] == "fail"}

    assert "advanced_bpb_visual_audit_cadence_250" in failed
    assert "advanced_bpb_periodic_interactive_artifacts" in failed
    assert "advanced_bpb_hf_reasoning_required_source" in failed
    assert "advanced_bpb_graphcg_weight_positive" in failed
    assert "advanced_bpb_graphcg_full_rank_directions" in failed
    assert "advanced_bpb_meet_in_middle_enabled" in failed
    assert "advanced_bpb_meet_in_middle_roar_random_order" in failed
    assert "advanced_bpb_memory_quality_probability_complex" in failed
    assert "advanced_bpb_wandb_online_project" in failed
    assert "advanced_bpb_wandb_entity_configured" in failed
    assert "advanced_bpb_wandb_run_name_matches_config" in failed


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
    assert report["data"]["graph_json_parse_unavailable_rate"] == 0
    assert report["data"]["graph_json_derived_text_graph_rate"] == 0.0
    assert not report["failed_gates"]
    markdown = render_markdown(report)
    assert "TropicalGT-I Readiness Audit" in markdown
    assert "| config_loads | pass |" in markdown
    assert "legacy graph-json substitution guardrail records" in markdown
    assert "legacy fallback records" not in markdown
    assert "Legacy graph-json substitution guardrail rate" in markdown
    assert "Legacy fallback rate" not in markdown


def test_readiness_audit_blocks_cross_run_memory_bank_path(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "run_name": "audit_cross_run_memory",
  "fixture_size": 4,
  "train_limit": 4,
  "val_limit": 4,
  "batch_size": 2,
  "seq_len": 32,
  "seed": 1729,
  "device": "cpu",
  "output_dir": "%s",
  "memory_bank_path": "%s",
  "model": {"dim": 32, "hidden_dim": 32, "graph_feature_dim": 48},
  "tokengt": {"feature_dim": 48}
}
"""
        % (tmp_path / "outputs" / "current", tmp_path / "outputs" / "old" / "memory.jsonl"),
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
    assert report["status"] == "blocked"
    assert report["memory"]["memory_bank_scoped_to_output_dir"] is False
    assert "memory_bank_path_scoped_to_output_dir" in report["failed_gates"]


def test_readiness_audit_blocks_malformed_explicit_graph_json(tmp_path):
    data_root = tmp_path / "shards" / "train"
    data_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"record_id": "bad", "text": "abc", "graph_json": "{not-json}"},
        ]
    ).to_parquet(data_root / "train-000.parquet")
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "run_name": "audit_malformed_graph_json",
  "data_root": "%s",
  "require_data": true,
  "train_limit": 1,
  "val_limit": 1,
  "batch_size": 1,
  "seq_len": 32,
  "seed": 1729,
  "device": "cpu",
  "output_dir": "%s",
  "model": {"dim": 32, "hidden_dim": 32, "graph_feature_dim": 48},
  "tokengt": {"feature_dim": 48}
}
"""
        % (tmp_path / "shards", tmp_path / "outputs"),
        encoding="utf-8",
    )
    report = build_readiness_report(
        config_path=config,
        checkpoint_path=None,
        split="train",
        sample_limit=1,
        details_limit=1,
        trace_limit=4,
        scale_depth=0,
        scale_width=2,
        scale_branch_factor=2,
        require_cuda=False,
        require_checkpoint=False,
        render_visualizations=False,
    )
    assert report["status"] == "blocked"
    assert report["data"]["graph_json_fallback_records"] == 0
    assert report["data"]["graph_json_parse_unavailable_records"] == 1
    assert report["data"]["graph_json_parse_unavailable_rate"] == 1.0
    assert "graph_json_parse_unavailable_zero" in report["failed_gates"]


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


def test_readiness_audit_train_dry_run_short_circuits_failed_bpb_contract(tmp_path, monkeypatch):
    config = tmp_path / "bad_bpb_config.json"
    config.write_text(
        """
{
  "run_name": "bad_bpb_readiness_gate",
  "parameter_golf_bpb_focus": true,
  "target_bpb": 1.12,
  "fixture_size": 4,
  "train_limit": 4,
  "val_limit": 4,
  "batch_size": 1,
  "seq_len": 32,
  "seed": 1729,
  "device": "cpu",
  "output_dir": "%s",
  "model": {"dim": 32, "hidden_dim": 32, "graph_feature_dim": 48},
  "tokengt": {"feature_dim": 48},
  "wandb": {"enabled": false}
}
"""
        % (tmp_path / "outputs"),
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("training preflight helper should not run after failed advanced BPB contract")

    monkeypatch.setattr(readiness_module, "make_dataset_from_config", fail_if_called)
    monkeypatch.setattr(readiness_module, "build_model", fail_if_called)
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
        require_cuda=False,
        require_checkpoint=False,
        render_visualizations=False,
    )

    assert report["status"] == "blocked"
    assert report["data"]["unavailable_reason"] == "advanced_bpb_contract_failed_before_train_preflight"
    assert report["train_dry_run"]["skipped"] is True
    assert "advanced_bpb_contract_blocks_train_preflight" in report["failed_gates"]
