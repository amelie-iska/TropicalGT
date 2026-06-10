import json
import math
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
            "causal_dag_ar_rate": 0.75,
            "gpu_mem_mb": 21484.0,
        }
    )
    assert list(payload)[:4] == ["step", "00_primary/eval_bpb", "00_primary/bpb", "00_primary/loss"]
    assert payload["01_losses/gflownet_tb"] == 0.1
    assert payload["03_tropical/sequence_tropical_margin_mean"] == 0.4
    assert payload["05_graphcg/graphcg_full_rank"] == 1.0
    assert payload["06_graph_data/causal_dag_ar_rate"] == 0.75
    assert payload["00_primary/gpu_mem_mb"] == 21484.0
