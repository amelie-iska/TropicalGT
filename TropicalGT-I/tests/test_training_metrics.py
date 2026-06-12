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


def test_derived_fallback_graph_is_not_charged_as_side_information():
    record = GraphRecord.from_mapping({"record_id": "derived", "text": "abc", "question": "abc"})
    assert record.metadata["graph_json_fallback"] is True
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
        "certificate_agreement",
        "certificate_allowed_mass_mean",
        "certificate_disallowed_support_rate",
        "certificate_node_graph_support_rate",
        "support_transition_rate",
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
        "causal_dag_ar_rate",
        "random_graph_ar_rate",
        "graph_autoregressive_decoding_enabled",
        "sequence_tropical_tokens_mean",
        "sequence_tropical_margin_mean",
    ]:
        assert key in row
        assert row[key] == row[key]
    assert report["eval"]["bpb"] == report["eval"]["bpb_proxy"]
    assert report["eval"]["graph_bpb"] == report["eval"]["graph_bpb"]


def test_wandb_metrics_are_namespaced_by_priority():
    payload = organize_wandb_metrics(
        {
            "step": 5,
            "eval_bpb": 1.25,
            "bpb": 1.3,
            "loss": 2.0,
            "gflownet_tb": 0.1,
            "graphcg_full_rank": 1.0,
            "sequence_tropical_margin_mean": 0.4,
            "certificate_allowed_mass_mean": 0.9,
            "support_transition_rate": 0.25,
            "analogical_memory_rejected": 2.0,
            "causal_dag_ar_rate": 0.75,
            "gpu_mem_mb": 21484.0,
        }
    )
    assert list(payload)[:4] == ["step", "00_primary/eval_bpb", "00_primary/bpb", "00_primary/loss"]
    assert payload["01_losses/gflownet_tb"] == 0.1
    assert payload["03_tropical/sequence_tropical_margin_mean"] == 0.4
    assert payload["03_tropical/certificate_allowed_mass_mean"] == 0.9
    assert payload["03_tropical/support_transition_rate"] == 0.25
    assert payload["08_memory/analogical_memory_rejected"] == 2.0
    assert payload["05_graphcg/graphcg_full_rank"] == 1.0
    assert payload["06_graph_data/causal_dag_ar_rate"] == 0.75
    assert payload["00_primary/gpu_mem_mb"] == 21484.0


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
