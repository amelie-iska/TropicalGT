import torch

from tropicalgt.data import FixtureGraphDataset, encode_bytes
from tropicalgt.diagnostics import (
    gflownet_diagnostics,
    graph_token_trace,
    graphcg_diagnostics,
    per_record_nll,
    record_diagnostics,
)
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.records import GraphRecord
from tropicalgt.tokenizer import TokenGTTokenizer


def test_per_record_nll_and_trace_are_finite():
    ds = FixtureGraphDataset(2)
    records = [ds[0], ds[1]]
    tok = TokenGTTokenizer(feature_dim=48)
    gb = tok.batch_encode(records)
    xs, ys = zip(*(encode_bytes(r.text, 32) for r in records))
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    out = model(torch.stack(xs), gb, torch.stack(ys))
    nll, counts = per_record_nll(out["logits"], torch.stack(ys))
    assert nll.shape == (2,)
    assert torch.isfinite(nll).all()
    assert counts.min() > 0
    traces = graph_token_trace(records, gb, out["support"], out["margin"], tok)
    assert traces[0]["tokens"][0]["kind"] == "graph"
    assert "active_support_index" in traces[0]["tokens"][0]


def test_record_gfn_graphcg_diagnostics():
    ds = FixtureGraphDataset(1)
    record = ds[0]
    tok = TokenGTTokenizer(feature_dim=48)
    gb = tok.batch_encode([record])
    x, y = encode_bytes(record.text, 32)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    out = model(x[None, :], gb, y[None, :])
    rows = record_diagnostics([record], gb, out, tok, target_ids=y[None, :])
    assert rows[0]["filtered_simplicial_object"]["summary"]["num_vertices"] >= 1
    assert rows[0]["tropical"]["margin_mean"] >= 0
    gfn = gflownet_diagnostics(model, out["graph_state"])
    graphcg = graphcg_diagnostics(model, out["graph_state"])
    assert gfn["top_actions"][0]
    assert graphcg["direction_norms"]


def test_graph_record_marks_fallback_metadata():
    real = GraphRecord.from_mapping({"record_id": "ok", "text": "x", "graph_json": '{"nodes":[],"edges":[]}'})
    fallback = GraphRecord.from_mapping({"record_id": "bad", "text": "x", "graph_json": "{not-json}"})
    assert real.metadata["graph_json_fallback"] is False
    assert fallback.metadata["graph_json_fallback"] is True
