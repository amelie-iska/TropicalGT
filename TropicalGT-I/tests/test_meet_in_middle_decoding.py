import torch

from tropicalgt.data import FixtureGraphDataset, encode_bytes
from tropicalgt.decoding import (
    encode_record_bytes_reverse,
    meet_in_middle_batch,
    meet_in_middle_config,
)
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.tokenizer import TokenGTTokenizer


def test_meet_in_middle_config_defaults_off():
    cfg = meet_in_middle_config({})
    assert cfg.enabled is False
    assert cfg.mode == "shared_weight_reverse_pass"
    assert cfg.agreement_weight == 0.0
    assert cfg.reverse_nll_weight == 0.0


def test_reverse_encoding_uses_same_shifted_byte_convention():
    record = FixtureGraphDataset(1)[0]
    _, forward_y = encode_bytes(record.text, 16)
    x_rev, y_rev = encode_record_bytes_reverse(record, 16)
    ids = [b + 1 for b in record.text.encode("utf-8", "ignore")]
    ids.reverse()
    assert x_rev[0].item() == ids[0]
    assert y_rev[0].item() == ids[1]
    assert forward_y.ne(0).sum().item() == y_rev.ne(0).sum().item()


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
    assert 0.0 <= report["metrics"]["mim_join_token_match_rate"] <= 1.0
    assert len(report["records"]) == 2
    assert {"meet_index", "forward_position", "reverse_position", "true_token"} <= set(report["records"][0])
