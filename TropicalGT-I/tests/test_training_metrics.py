import json
import math
import sys
import types
from pathlib import Path

import torch

from tropicalgt.data import encode_bytes
from tropicalgt.metrics import aggregate_bpb_metrics, batch_bpb_metrics, explicit_graph_json_bytes, graph_token_structural_bytes
from tropicalgt.records import GraphRecord
from tropicalgt.run import organize_wandb_metrics, train
from tropicalgt.tokenizer import TokenGTTokenizer


def test_bpb_and_graph_bpb_formulas_are_exact():
    record = GraphRecord.from_mapping(
        {
            "record_id": "formula",
            "text": "abc",
            "question": "abc",
            "graph_json": {"nodes": [{"id": "n", "type": "problem", "text": "abc"}], "edges": []},
        }
    )
    tok = TokenGTTokenizer(feature_dim=48)
    graph_batch = tok.batch_encode([record])
    _, y = encode_bytes(record.text, seq_len=8)
    nll = torch.tensor(2.0)
    metrics = batch_bpb_metrics(nll, y.unsqueeze(0), graph_batch, [record], graph_side_weight=0.5)
    target_bytes = int(y.ne(0).sum().item())
    nll_bits = 2.0 * target_bytes / math.log(2.0)
    graph_bytes = graph_token_structural_bytes(graph_batch)
    explicit_bytes = explicit_graph_json_bytes(record)
    assert metrics["bpb"] == nll_bits / target_bytes
    assert metrics["graph_bpb"] == (nll_bits + 4.0 * explicit_bytes) / (target_bytes + graph_bytes)
    aggregate = aggregate_bpb_metrics(nll_bits, target_bytes, graph_bytes, explicit_bytes, graph_side_weight=0.5)
    assert aggregate["graph_sideinfo_bpb"] == metrics["graph_sideinfo_bpb"]


def test_derived_text_graph_is_not_marked_fallback_or_charged_as_side_information():
    record = GraphRecord.from_mapping({"record_id": "derived", "text": "abc", "question": "abc"})
    assert record.metadata["graph_json_fallback"] is False
    assert record.metadata["graph_json_source"] == "derived_text_graph"
    assert record.metadata["graph_json_derived_from_text"] is True
    assert explicit_graph_json_bytes(record) == 0


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
        "certificate_objective_loss",
        "certificate_diagnostic_penalty",
        "certificate_loss_reconstruction_error",
        "certificate_valid_token_count",
        "certificate_allowed_target_count_mean",
        "certificate_allowed_target_count_min",
        "tropical_margin_loss",
        "tropical_margin_signed_loss",
        "tropical_margin_signed_objective",
        "tropical_margin_reward",
        "tropical_margin_shortfall_loss",
        "tropical_margin_shortfall_rate",
        "loss_tropical_margin_signed_weighted",
        "loss_tropical_margin_shortfall_weighted",
        "certificate_agreement",
        "certificate_allowed_mass_mean",
        "certificate_disallowed_support_rate",
        "certificate_node_graph_support_rate",
        "loss_certificate_objective_weighted",
        "loss_certificate_diagnostic_penalty_weighted",
        "support_transition_rate",
        "wall_hit_rate",
        "strict_wall_hit_rate",
        "near_wall_hit_rate",
        "wall_margin_threshold",
        "near_wall_margin_threshold",
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
        "graph_json_fallback_rate",
        "legacy_graph_json_substitution_guardrail_rate",
        "graph_json_derived_text_graph_rate",
        "graph_json_parse_unavailable_rate",
        "graph_json_sequentialized_rate",
        "analogical_memory_query_norm",
        "causal_dag_ar_rate",
        "random_graph_ar_rate",
        "graph_autoregressive_decoding_enabled",
        "sequence_tropical_tokens_mean",
        "sequence_tropical_margin_mean",
        "graphcg_requested_num_directions",
        "graphcg_effective_num_directions",
        "graphcg_embedding_span_rank_target",
        "graphcg_embedding_span_full_rank",
        "graphcg_direction_bank_clamped_to_embedding_dim",
    ]:
        assert key in row
        assert row[key] == row[key]
    assert report["eval"]["bpb"] == report["eval"]["bpb_exact"]
    assert report["eval"]["legacy_graph_json_substitution_guardrail_records"] == report["eval"]["graph_json_fallback_records"] == 0
    assert report["eval"]["legacy_graph_json_substitution_guardrail_rate"] == report["eval"]["invalid_graph_rate"] == 0.0
    assert report["eval"]["graph_bpb"] == report["eval"]["graph_bpb"]


def test_wandb_metrics_are_namespaced_by_priority():
    payload = organize_wandb_metrics(
        {
            "step": 5,
            "eval_bpb": 1.25,
            "bpb": 1.3,
            "loss": 2.0,
            "gflownet_tb": 0.1,
            "tropical_margin_signed_objective": -0.4,
            "loss_tropical_margin_signed_weighted": -0.01,
            "loss_tropical_margin_shortfall_weighted": 0.0,
            "graphcg_full_rank": 1.0,
            "graphcg_direction_bank_clamped_to_embedding_dim": 1.0,
            "graphcg_requested_num_directions": 4096.0,
            "graphcg_effective_num_directions": 1760.0,
            "graphcg_embedding_span_rank_target": 1760.0,
            "graphcg_embedding_span_full_rank": 1.0,
            "graphcg_active_rank_fraction": 1.0,
            "sequence_tropical_margin_mean": 0.4,
            "certificate_allowed_mass_mean": 0.9,
            "certificate_objective_loss": 0.2,
            "certificate_diagnostic_penalty": 0.0,
            "certificate_loss_reconstruction_error": 0.0,
            "support_transition_rate": 0.25,
            "analogical_memory_rejected": 2.0,
            "graph_json_fallback_rate": 0.0,
            "graph_json_derived_text_graph_rate": 1.0,
            "causal_dag_ar_rate": 0.75,
            "gpu_mem_mb": 21484.0,
        }
    )
    assert list(payload)[:4] == ["step", "00_primary/eval_bpb", "00_primary/bpb", "00_primary/loss"]
    assert payload["01_losses/gflownet_tb"] == 0.1
    assert payload["01_losses/tropical_margin_signed_objective"] == -0.4
    assert payload["01_losses/loss_tropical_margin_signed_weighted"] == -0.01
    assert payload["01_losses/loss_tropical_margin_shortfall_weighted"] == 0.0
    assert payload["03_tropical/sequence_tropical_margin_mean"] == 0.4
    assert payload["01_losses/certificate_objective_loss"] == 0.2
    assert payload["01_losses/certificate_diagnostic_penalty"] == 0.0
    assert payload["01_losses/certificate_loss_reconstruction_error"] == 0.0
    assert payload["03_tropical/certificate_allowed_mass_mean"] == 0.9
    assert payload["03_tropical/support_transition_rate"] == 0.25
    assert payload["08_memory/analogical_memory_rejected"] == 2.0
    assert payload["05_graphcg/graphcg_full_rank"] == 1.0
    graphcg_keys = [key for key in payload if key.startswith("05_graphcg/")]
    assert graphcg_keys[:6] == [
        "05_graphcg/graphcg_embedding_span_full_rank",
        "05_graphcg/graphcg_direction_bank_clamped_to_embedding_dim",
        "05_graphcg/graphcg_requested_num_directions",
        "05_graphcg/graphcg_effective_num_directions",
        "05_graphcg/graphcg_embedding_span_rank_target",
        "05_graphcg/graphcg_full_rank",
    ]
    assert payload["05_graphcg/graphcg_embedding_span_full_rank"] == 1.0
    assert "06_graph_data/graph_json_fallback_rate" not in payload
    assert payload["06_graph_data/legacy_graph_json_substitution_guardrail_rate"] == 0.0
    assert payload["06_graph_data/graph_json_derived_text_graph_rate"] == 1.0
    assert payload["06_graph_data/causal_dag_ar_rate"] == 0.75
    assert payload["00_primary/gpu_mem_mb"] == 21484.0


def test_browser_metric_visualization_prioritizes_graphcg_rank_audit(tmp_path: Path):
    from tropicalgt.visualization import GRAPHCG_BROWSER_PRIORITY_METRICS, write_metric_visualizations

    priority_keys = [
        "graphcg_loss",
        "graphcg_embedding_span_full_rank",
        "graphcg_direction_bank_clamped_to_embedding_dim",
        "graphcg_requested_num_directions",
        "graphcg_effective_num_directions",
        "graphcg_embedding_span_rank_target",
        "graphcg_num_directions",
        "graphcg_embedding_dim",
        "graphcg_active_directions",
        "graphcg_full_rank",
        "graphcg_active_full_rank",
        "graphcg_active_rank_fraction",
        "graphcg_direction_effective_rank",
        "graphcg_direction_numerical_rank",
        "graphcg_direction_rank_target",
        "graphcg_direction_singular_min",
        "graphcg_direction_singular_max",
        "graphcg_direction_svd_condition_number",
    ]
    assert [key for key in GRAPHCG_BROWSER_PRIORITY_METRICS if key in priority_keys] == priority_keys

    row = {"step": 1, "loss": 1.0, "nll": 0.9, "graph_json_fallback_rate": 0.0}
    row.update({key: float(idx + 1) for idx, key in enumerate(priority_keys)})
    paths = write_metric_visualizations([row], tmp_path)

    html = Path(paths["metrics"]).read_text(encoding="utf-8")
    positions = [html.index('"name":"' + key + '"') for key in priority_keys]
    assert positions == sorted(positions)
    assert '"name":"graph_json_fallback_rate"' not in html
    assert '"name":"legacy graph-json substitution guardrail (must remain zero)"' in html


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged = []

    def log(self, payload, step=None) -> None:
        self.logged.append((payload, step))


def test_wandb_html_artifacts_disabled_by_default(tmp_path: Path, monkeypatch):
    from tropicalgt.run import _log_wandb_html_artifacts

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Html=lambda text: ("html", text)))
    html_path = tmp_path / "plot.html"
    html_path.write_text("<html>plot</html>", encoding="utf-8")
    wb = _FakeWandbRun()

    _log_wandb_html_artifacts(wb, {"plot": str(html_path)}, {}, prefix="periodic", step=1)

    assert wb.logged == []


def test_wandb_html_artifacts_require_explicit_opt_in(tmp_path: Path, monkeypatch):
    from tropicalgt.run import _log_wandb_html_artifacts

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Html=lambda text: ("html", text)))
    html_path = tmp_path / "plot.html"
    html_path.write_text("<html>plot</html>", encoding="utf-8")
    wb = _FakeWandbRun()

    _log_wandb_html_artifacts(
        wb,
        {"plot": str(html_path)},
        {"wandb": {"log_interactive_artifacts": True, "html_artifact_limit": 1}},
        prefix="periodic",
        step=1,
    )

    assert wb.logged
    assert "periodic/plot" in wb.logged[0][0]


def test_wandb_html_artifacts_need_positive_limit(tmp_path: Path, monkeypatch):
    from tropicalgt.run import _log_wandb_html_artifacts

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Html=lambda text: ("html", text)))
    html_path = tmp_path / "plot.html"
    html_path.write_text("<html>plot</html>", encoding="utf-8")
    wb = _FakeWandbRun()

    _log_wandb_html_artifacts(
        wb,
        {"plot": str(html_path)},
        {"wandb": {"log_interactive_artifacts": True, "html_artifact_limit": 0}},
        prefix="periodic",
        step=1,
    )

    assert wb.logged == []


def test_periodic_got_visualization_requires_complete_steps_by_default(tmp_path: Path, monkeypatch):
    import tropicalgt.run as run_mod

    captured = {}
    monkeypatch.setattr(run_mod, "evaluate_model", lambda *args, **kwargs: {"nll": 1.0, "bpb": 1.0})
    monkeypatch.setattr(run_mod, "write_reasoning_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "write_metric_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "write_graphcg_training_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "_select_got_audit_records", lambda *args, **kwargs: [(0, types.SimpleNamespace(record_id="audit-record"))])

    def fake_scaling(*args, **kwargs):
        captured.update(kwargs)
        return {"enabled": True, "best": {}, "candidates": []}

    def fake_artifacts(_result, output_dir, render_html):
        path = Path(output_dir) / "audit.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>audit</html>", encoding="utf-8")
        return {"audit": str(path)}

    monkeypatch.setattr(run_mod, "run_inference_scaling", fake_scaling)
    monkeypatch.setattr(run_mod, "write_inference_audit_artifacts", fake_artifacts)

    report = run_mod._run_periodic_validation_round(
        model=object(),
        val_ds=[object()],
        tokenizer=object(),
        seq_len=1,
        batch_size=1,
        device=torch.device("cpu"),
        out_dir=tmp_path,
        cfg={"periodic_viz_got_scaling": True},
        history=[],
        step=10,
        seed=1729,
        graph_bpb_side_weight=1.0,
        graph_autoregressive=True,
        run_name="periodic-test",
        memory_bank=None,
        memory_records_added=0,
        render_visualizations=True,
        details_limit=1,
        viz_limit=1,
        audit_level="none",
        ph_backend="none",
        audit_max_simplices=8,
    )

    assert captured["require_complete_reasoning_steps"] is True
    assert report["visualizations"]["got_audit_audit"].endswith("audit.html")


def test_periodic_got_visualization_bounds_training_safe_failure_policy(tmp_path: Path, monkeypatch):
    import tropicalgt.run as run_mod

    captured = {}
    monkeypatch.setattr(run_mod, "evaluate_model", lambda *args, **kwargs: {"nll": 1.0, "bpb": 1.0})
    monkeypatch.setattr(run_mod, "write_reasoning_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "write_metric_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "write_graphcg_training_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "_select_got_audit_records", lambda *args, **kwargs: [(0, types.SimpleNamespace(record_id="audit-record"))])

    def fake_scaling(*args, **kwargs):
        captured.update(kwargs)
        return {"enabled": True, "best": {}, "candidates": []}

    def fake_artifacts(result, output_dir, render_html):
        path = Path(output_dir) / "audit.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        return {"audit": str(path)}

    monkeypatch.setattr(run_mod, "run_inference_scaling", fake_scaling)
    monkeypatch.setattr(run_mod, "write_inference_audit_artifacts", fake_artifacts)

    report = run_mod._run_periodic_validation_round(
        model=object(),
        val_ds=[object()],
        tokenizer=object(),
        seq_len=1,
        batch_size=1,
        device=torch.device("cpu"),
        out_dir=tmp_path,
        cfg={
            "periodic_viz_got_scaling": True,
            "periodic_viz_failure_policy": "record_incomplete_without_fabrication",
            "periodic_viz_scale_depth": 12,
            "periodic_viz_scale_width": 18,
            "periodic_viz_scale_branch_factor": 6,
            "viz_trace_limit": 2048,
        },
        history=[],
        step=10,
        seed=1729,
        graph_bpb_side_weight=1.0,
        graph_autoregressive=True,
        run_name="periodic-test",
        memory_bank=None,
        memory_records_added=0,
        render_visualizations=True,
        details_limit=1,
        viz_limit=1,
        audit_level="none",
        ph_backend="none",
        audit_max_simplices=8,
    )

    assert captured["depth"] == 3
    assert captured["width"] == 4
    assert captured["branch_factor"] == 3
    assert captured["trace_limit"] == 256
    budget = report["periodic_got_scaling_budgets"][0]
    assert budget["requested_depth"] == 12
    assert budget["requested_width"] == 18
    assert budget["requested_branch_factor"] == 6
    assert budget["requested_trace_limit"] == 2048
    assert budget["effective_trace_limit"] == 256
    assert budget["bounded_for_training_survivability"] is True
    assert report["metrics"]["periodic_got_scaling_requested_trace_limit"] == 2048.0
    assert report["metrics"]["periodic_got_scaling_effective_trace_limit"] == 256.0
    assert report["metrics"]["periodic_got_scaling_bounded_for_training"] == 1.0
    assert report["metrics"]["periodic_got_scaling_available"] == 1.0


def test_periodic_got_audit_retention_prunes_only_unprotected_generated_audits(tmp_path: Path):
    import tropicalgt.run as run_mod

    out_dir = tmp_path / "run"
    for step in (100, 200, 300):
        step_dir = out_dir / "periodic" / f"step_{step:08d}"
        audit_dir = step_dir / "got_audit"
        audit_dir.mkdir(parents=True)
        (audit_dir / "large_payload.json").write_text("{}", encoding="utf-8")
        (step_dir / "validation_report.json").write_text(json.dumps({"step": step}), encoding="utf-8")

    actions = run_mod._prune_periodic_generated_audits(
        out_dir,
        {"periodic_prune_got_audit_keep_latest": 1, "periodic_prune_got_audit_keep_steps": [100]},
    )

    assert [row["step"] for row in actions] == [200]
    assert (out_dir / "periodic" / "step_00000100" / "got_audit").exists()
    assert not (out_dir / "periodic" / "step_00000200" / "got_audit").exists()
    assert (out_dir / "periodic" / "step_00000300" / "got_audit").exists()
    assert (out_dir / "periodic" / "step_00000200" / "validation_report.json").exists()
    manifest = out_dir / "periodic" / "got_audit_retention_manifest.jsonl"
    assert manifest.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["step"] == 200


def test_periodic_got_audit_retention_can_cap_configured_keep_steps(tmp_path: Path):
    import tropicalgt.run as run_mod

    out_dir = tmp_path / "run"
    for step in (250, 500, 750, 1000):
        audit_dir = out_dir / "periodic" / f"step_{step:08d}" / "got_audit"
        audit_dir.mkdir(parents=True)
        (audit_dir / "large_payload.json").write_text("{}", encoding="utf-8")

    actions = run_mod._prune_periodic_generated_audits(
        out_dir,
        {
            "periodic_prune_got_audit_keep_latest": 2,
            "periodic_prune_got_audit_keep_steps": [250, 500, 1000],
            "periodic_prune_got_audit_max_retained_steps": 2,
        },
    )

    assert [row["step"] for row in actions] == [250, 500]
    assert not (out_dir / "periodic" / "step_00000250" / "got_audit").exists()
    assert not (out_dir / "periodic" / "step_00000500" / "got_audit").exists()
    assert (out_dir / "periodic" / "step_00000750" / "got_audit").exists()
    assert (out_dir / "periodic" / "step_00001000" / "got_audit").exists()
    assert all("max_retained_steps capped" in row["reason"] for row in actions)
    assert actions[0]["retention_policy"]["capped_configured_steps"] == [250, 500]


def test_periodic_got_visualization_records_unavailable_on_failure_policy(tmp_path: Path, monkeypatch):
    import tropicalgt.run as run_mod

    monkeypatch.setattr(run_mod, "evaluate_model", lambda *args, **kwargs: {"nll": 1.0, "bpb": 1.0})
    monkeypatch.setattr(run_mod, "write_reasoning_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "write_metric_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "write_graphcg_training_visualizations", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_mod, "_select_got_audit_records", lambda *args, **kwargs: [(0, types.SimpleNamespace(record_id="audit-record"))])

    def fake_scaling(*args, **kwargs):
        raise RuntimeError("periodic audit blew budget")

    monkeypatch.setattr(run_mod, "run_inference_scaling", fake_scaling)

    report = run_mod._run_periodic_validation_round(
        model=object(),
        val_ds=[object()],
        tokenizer=object(),
        seq_len=1,
        batch_size=1,
        device=torch.device("cpu"),
        out_dir=tmp_path,
        cfg={
            "periodic_viz_got_scaling": True,
            "periodic_viz_failure_policy": "record_incomplete_without_fabrication",
            "periodic_viz_scale_depth": 12,
            "periodic_viz_scale_width": 18,
            "periodic_viz_scale_branch_factor": 6,
        },
        history=[],
        step=10,
        seed=1729,
        graph_bpb_side_weight=1.0,
        graph_autoregressive=True,
        run_name="periodic-test",
        memory_bank=None,
        memory_records_added=0,
        render_visualizations=True,
        details_limit=1,
        viz_limit=1,
        audit_level="none",
        ph_backend="none",
        audit_max_simplices=8,
    )

    assert report["metrics"]["periodic_got_scaling_available"] == 0.0
    assert report["periodic_got_scaling_failures"][0]["available"] is False
    assert report["periodic_got_scaling_failures"][0]["status"] == "unavailable_periodic_got_scaling_failed"
    unavailable_path = Path(report["visualizations"]["got_audit_unavailable_json"])
    assert unavailable_path.exists()
    payload = json.loads(unavailable_path.read_text(encoding="utf-8"))
    assert payload["reason"].endswith("no proxy artifact was substituted.")


def test_validation_wandb_logs_only_new_eval_scalars(tmp_path: Path, monkeypatch):
    import tropicalgt.run as run_mod

    wb = _FakeWandbRun()
    wb.finish = lambda: None
    wb.define_metric = lambda *args, **kwargs: None
    monkeypatch.setattr(run_mod, "setup_wandb", lambda _cfg, _run_name: wb)
    cfg = {
        "run_name": "wandb_eval_test",
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
        "validation_every_steps": 1,
        "final_interactive_artifacts_enabled": False,
        "periodic_interactive_artifacts_enabled": False,
        "tokengt": {"max_nodes": 16, "max_edges": 32, "node_id_dim": 8, "feature_dim": 48, "graph_token": True},
        "model": {"dim": 16, "hidden_dim": 16, "graph_feature_dim": 48, "num_actions": 8},
        "wandb": {"enabled": True, "mode": "disabled", "log_interactive_artifacts": False, "html_artifact_limit": 0},
    }
    config_path = tmp_path / "wandb_eval.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    train(config_path)

    assert len(wb.logged) >= 3
    train_payload = wb.logged[0][0]
    validation_payload = wb.logged[1][0]
    final_payload = wb.logged[2][0]
    assert "00_primary/loss" in train_payload
    assert "00_primary/loss" not in validation_payload
    assert "00_primary/loss" not in final_payload
    assert any(key.startswith("00_primary/eval_") or key.startswith("02_bpb/eval_") for key in validation_payload)
    assert any(key.startswith("00_primary/eval_") or key.startswith("02_bpb/eval_") for key in final_payload)
