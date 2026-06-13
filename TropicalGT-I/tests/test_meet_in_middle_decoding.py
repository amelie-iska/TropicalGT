import torch

from tropicalgt.data import FixtureGraphDataset, encode_bytes
from tropicalgt.decoding import (
    encode_record_bytes_reverse,
    meet_in_middle_batch,
    meet_in_middle_config,
)
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.records import GraphRecord
from tropicalgt.tokenizer import TokenGTTokenizer


def test_meet_in_middle_config_defaults_off():
    cfg = meet_in_middle_config({})
    assert cfg.enabled is False
    assert cfg.mode == "shared_weight_reverse_pass"
    assert cfg.agreement_weight == 0.0
    assert cfg.reverse_nll_weight == 0.0
    assert cfg.agreement_kind == "total_variation"
    assert cfg.max_meet_points == 8
    assert cfg.ngram_size == 4


def test_reverse_encoding_uses_same_shifted_byte_convention():
    record = FixtureGraphDataset(1)[0]
    _, forward_y = encode_bytes(record.text, 16)
    x_rev, y_rev = encode_record_bytes_reverse(record, 16)
    ids = [b + 1 for b in record.text.encode("utf-8", "ignore")]
    ids.reverse()
    assert x_rev[0].item() == ids[0]
    assert y_rev[0].item() == ids[1]
    assert forward_y.ne(0).sum().item() == y_rev.ne(0).sum().item()




def test_graph_reverse_encoding_uses_reverse_graph_order_not_byte_reversed_forward_order():
    record = GraphRecord.from_mapping(
        {
            "record_id": "causal-reverse-encoding",
            "text": "one two three",
            "graph_json": {
                "nodes": [
                    {"id": "a", "type": "problem", "text": "one"},
                    {"id": "b", "type": "reasoning_step", "text": "two"},
                    {"id": "c", "type": "answer", "text": "three"},
                ],
                "edges": [
                    {"source": "a", "target": "b", "type": "depends_on"},
                    {"source": "b", "target": "c", "type": "supports_answer"},
                ],
            },
        }
    )
    forward_text = record.autoregressive_text(seed=0, direction="forward")
    reverse_text = record.autoregressive_text(seed=0, direction="reverse")
    forward_lines = forward_text.splitlines()
    reverse_lines = reverse_text.splitlines()
    assert forward_lines[0].startswith("[problem] one")
    assert forward_lines[-1].startswith("[answer] three")
    assert reverse_lines[0].startswith("[answer] three")
    assert reverse_lines[-1].startswith("[problem] one")

    x_rev, y_rev = encode_record_bytes_reverse(record, 48, graph_autoregressive=True)
    reverse_ids = [b + 1 for b in reverse_text.encode("utf-8", "ignore")]
    forward_ids = [b + 1 for b in forward_text.encode("utf-8", "ignore")]
    byte_reversed_forward = list(reversed(forward_ids))
    assert x_rev[0].item() == reverse_ids[0]
    assert y_rev[0].item() == reverse_ids[1]
    assert reverse_ids[:16] != byte_reversed_forward[:16]


def test_meet_in_middle_batch_reports_shared_weight_reverse_pass():
    records = [FixtureGraphDataset(2)[0], FixtureGraphDataset(2)[1]]
    tokenizer = TokenGTTokenizer(feature_dim=48)
    xs, ys = zip(*(encode_bytes(record.text, 40) for record in records))
    graph_batch = tokenizer.batch_encode(records)
    model = TropicalGTModel(
        TropicalGTConfig(
            dim=32,
            hidden_dim=32,
            graph_feature_dim=48,
            use_sequence_tropical=False,
        )
    )
    out = model(torch.stack(xs), graph_batch, torch.stack(ys))
    report = meet_in_middle_batch(
        model,
        records,
        tokenizer,
        seq_len=40,
        device=torch.device("cpu"),
        graph_autoregressive=True,
        seed=7,
        config={"enabled": True, "split_ratio": 0.5, "agreement_weight": 0.0, "reverse_nll_weight": 0.0},
        forward_logits=out["logits"],
        forward_nll=out["nll"],
    )
    assert report["enabled"] is True
    assert report["mode"] == "shared_weight_reverse_pass"
    assert report["metrics"]["mim_enabled"] == 1.0
    assert report["metrics"]["mim_shared_weight_reverse_pass"] == 1.0
    assert report["metrics"]["mim_candidate_count"] == 2.0
    assert report["metrics"]["mim_max_meet_points"] == 8.0
    assert report["metrics"]["mim_selected_meet_points_mean"] >= 1.0
    assert 0.0 <= report["metrics"]["mim_join_token_match_rate"] <= 1.0
    assert 0.0 <= report["metrics"]["mim_ngram_candidate_rate"] <= 1.0
    assert 0.0 <= report["metrics"]["mim_truth_verification_rate"] <= 1.0
    assert len(report["records"]) == 2
    assert {"context_mode", "reverse_context_mode", "meet_points"} <= set(report["records"][0])
    assert {"meet_index", "forward_position", "reverse_position", "true_token", "reverse_true_token", "same_true_token_alignment"} <= set(report["records"][0]["meet_points"][0])


def test_causal_graph_records_have_forward_and_reverse_causal_orders():
    record = GraphRecord.from_mapping(
        {
            "record_id": "causal",
            "text": "one two three",
            "graph_json": {
                "nodes": [
                    {"id": "a", "type": "problem", "text": "one"},
                    {"id": "b", "type": "reasoning_step", "text": "two"},
                    {"id": "c", "type": "answer", "text": "three"},
                ],
                "edges": [
                    {"source": "a", "target": "b", "type": "depends_on"},
                    {"source": "b", "target": "c", "type": "supports_answer"},
                ],
            },
        }
    )
    metadata = record.metadata or {}
    assert metadata["causal_edge_inferred_count"] >= 2
    assert metadata["decoding_order_kind"] == "causal_dag"
    assert metadata["decoding_reverse_order_kind"] == "reverse_causal_dag"
    original_forward = [node for node in metadata["decoding_node_order"] if node in {"a", "b", "c"}]
    original_reverse = [node for node in metadata["decoding_reverse_node_order"] if node in {"a", "b", "c"}]
    assert original_forward == ["a", "b", "c"]
    assert original_reverse == ["c", "b", "a"]
    assert record.graph_json["edges"][0]["causal"] is True
    assert record.graph_json["edges"][0]["directed"] is True




def test_explicit_noncausal_edge_overrides_directed_flag_for_roar():
    from tropicalgt.scaling import _decoding_order_report

    record = GraphRecord.from_mapping(
        {
            "record_id": "noncausal-directed",
            "text": "similar nodes are not a causal DAG",
            "graph_json": {
                "nodes": [
                    {"id": "a", "text": "alpha"},
                    {"id": "b", "text": "beta"},
                ],
                "edges": [
                    {"source": "a", "target": "b", "type": "similar", "directed": True, "causal": False},
                ],
            },
        }
    )
    metadata = record.metadata or {}
    assert record.graph_json["edges"][0]["causal"] is False
    assert record.graph_json["edges"][0]["directed"] is False
    assert metadata["noncausal_edge_count"] == 1
    assert metadata["decoding_order_kind"] == "random_autoregressive"
    assert metadata["decoding_reverse_order_kind"] == "reverse_random_autoregressive"
    overlay_report = _decoding_order_report(record)
    assert not any(
        edge.get("source") == "a" and edge.get("target") == "b"
        for edge in overlay_report["causal_edges"]
    )
    assert overlay_report["noncausal_edge_count"] >= 1


def test_cyclic_or_noncausal_graph_records_use_roar_random_order():
    record = GraphRecord.from_mapping(
        {
            "record_id": "cycle",
            "text": "cycle",
            "graph_json": {
                "nodes": [{"id": "a", "text": "a"}, {"id": "b", "text": "b"}, {"id": "c", "text": "c"}],
                "edges": [
                    {"source": "a", "target": "b", "type": "depends_on"},
                    {"source": "b", "target": "a", "type": "depends_on"},
                    {"source": "b", "target": "c", "type": "similar"},
                ],
            },
        }
    )
    metadata = record.metadata or {}
    assert metadata["decoding_order_kind"] == "random_autoregressive"
    assert metadata["decoding_reverse_order_kind"] == "reverse_random_autoregressive"
    assert sorted(metadata["decoding_node_order"]) == ["a", "b", "c", "seq_000", "seq_root"]
    assert metadata["decoding_reverse_node_order"] == list(reversed(metadata["decoding_node_order"]))
