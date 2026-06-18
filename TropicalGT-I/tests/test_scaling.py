import torch

from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.scaling import _select_branch_actions, apply_reasoning_action, run_inference_scaling
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


def test_inference_scaling_uses_deeper_non_stop_audit_branches():
    record = FixtureGraphDataset(1)[0]
    tok = TokenGTTokenizer(feature_dim=48)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    report = run_inference_scaling(
        model,
        record,
        tok,
        seq_len=32,
        device=torch.device("cpu"),
        depth=2,
        width=3,
        branch_factor=3,
        trace_limit=4,
        allow_stop=False,
        diverse_actions=True,
    )
    assert report["depth"] == 2
    assert report["evaluated_candidates"] > 3
    assert any(len(row.get("path", [])) >= 2 for row in report["candidates"])
    assert all("stop" not in row.get("path", []) for row in report["candidates"])
    branch_selection = report["levels"][0]["branch_selection"]
    assert branch_selection
    first = branch_selection[0]
    assert first["schema_version"] == "tropicalgt.gflownet_branch_selection_audit.v1"
    assert first["source"] == "run_inference_scaling._select_branch_actions"
    assert first["no_proxy_or_fallback"] is True
    contract = first["action_selection_contract"]
    assert contract["schema_version"] == "tropicalgt.gflownet_action_selection_contract.v1"
    assert contract["source"] == "gflownet_action_probs"
    assert contract["selection_policy"] == "ranked_diverse_action_sweep"
    assert contract["selected_from_real_model_action_probabilities"] is True
    assert contract["not_a_policy_quality_certificate"] is True
    assert contract["no_proxy_or_fallback"] is True
    assert first["selected_action_count"] == len(first["selected_actions"])
    assert first["selected_action_count"] == contract["selected_action_count"]
    assert all(action["action_selection_contract"] == contract for action in first["selected_actions"])


def test_select_branch_actions_filters_stop_and_preserves_diversity():
    probs = [
        {"action": "stop", "probability": 0.9},
        {"action": "merge", "probability": 0.8},
        {"action": "expand", "probability": 0.7},
        {"action": "verify", "probability": 0.6},
    ]
    selected = _select_branch_actions(probs, branch_factor=3, allow_stop=False, diverse_actions=True)
    actions = [row["action"] for row in selected]
    assert actions == ["merge", "expand", "verify"]
    contracts = [row["action_selection_contract"] for row in selected]
    assert {contract["schema_version"] for contract in contracts} == {"tropicalgt.gflownet_action_selection_contract.v1"}
    assert {contract["selection_policy"] for contract in contracts} == {"ranked_diverse_action_sweep"}
    assert all(contract["source"] == "gflownet_action_probs" for contract in contracts)
    assert all(contract["selected_from_real_model_action_probabilities"] is True for contract in contracts)
    assert all(contract["not_a_policy_quality_certificate"] is True for contract in contracts)
    assert all(contract["no_proxy_or_fallback"] is True for contract in contracts)
    assert all(contract["selected_action_count"] == len(selected) for contract in contracts)


def test_select_branch_actions_supports_reproducible_stochastic_sampling():
    probs = [
        {"action": "expand", "probability": 0.45},
        {"action": "verify", "probability": 0.25},
        {"action": "retrieve", "probability": 0.15},
        {"action": "refine", "probability": 0.10},
        {"action": "merge", "probability": 0.05},
    ]
    sampled_a = _select_branch_actions(
        probs,
        branch_factor=3,
        allow_stop=False,
        diverse_actions=True,
        stochastic=True,
        temperature=2.0,
        exploration=0.35,
        seed=1234,
    )
    sampled_b = _select_branch_actions(
        probs,
        branch_factor=3,
        allow_stop=False,
        diverse_actions=True,
        stochastic=True,
        temperature=2.0,
        exploration=0.35,
        seed=1234,
    )
    assert [row["action"] for row in sampled_a] == [row["action"] for row in sampled_b]
    assert all("sampling_weight" in row for row in sampled_a)
    assert len({row["action"] for row in sampled_a}) == 3
    stochastic_contract = sampled_a[0]["action_selection_contract"]
    assert stochastic_contract["schema_version"] == "tropicalgt.gflownet_action_selection_contract.v1"
    assert stochastic_contract["selection_policy"] == "stochastic_without_replacement"
    assert stochastic_contract["stochastic"] is True
    assert stochastic_contract["temperature"] == 2.0
    assert stochastic_contract["exploration"] == 0.35
    assert stochastic_contract["selected_from_real_model_action_probabilities"] is True
    assert stochastic_contract["no_proxy_or_fallback"] is True
    assert all(row["action_selection_contract"] == stochastic_contract for row in sampled_a)
