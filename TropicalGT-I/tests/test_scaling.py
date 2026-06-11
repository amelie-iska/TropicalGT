import torch

from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.scaling import apply_reasoning_action, run_inference_scaling
from tropicalgt.tokenizer import TokenGTTokenizer


def test_apply_reasoning_action_extends_graph():
    record = FixtureGraphDataset(1)[0]
    expanded = apply_reasoning_action(record, "verify", rank=0)
    assert expanded.record_id.endswith("|verify0")
    assert len(expanded.graph_json["nodes"]) == len(record.graph_json["nodes"]) + 4
    assert len(expanded.graph_json["edges"]) >= len(record.graph_json["edges"]) + 5
    assert expanded.metadata["reasoning_microstep_count"] == 3
    assert "[got:verify:1]" in expanded.text
    assert any(node.get("type") == "verification_check" for node in expanded.graph_json["nodes"])
    stopped = apply_reasoning_action(record, "stop", rank=1)
    assert len(stopped.graph_json["nodes"]) == len(record.graph_json["nodes"])


def test_inference_scaling_returns_best_candidate():
    record = FixtureGraphDataset(1)[0]
    tok = TokenGTTokenizer(feature_dim=48)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    report = run_inference_scaling(
        model,
        record,
        tok,
        seq_len=32,
        device=torch.device("cpu"),
        depth=1,
        width=2,
        branch_factor=2,
        trace_limit=4,
    )
    assert report["evaluated_candidates"] >= 2
    assert report["levels"][0]["candidates"]
    assert "best" in report
    assert "filtered_simplicial_object" in report["best"]
    assert "gflownet_action_probs" in report["best"]
