import pytest
from types import SimpleNamespace

from tropicalgt.scaling import _reasoning_step_audit, _public_candidate, run_inference_scaling


def _complex(metric: str, source: str) -> dict:
    transform = {"kind": "model_supplied", "source": source} if metric == "jensen_shannon" else None
    return {
        "available": True,
        "summary": {
            "num_vertices": 3,
            "num_edges": 3,
            "num_two_simplices": 1,
            "embedding_metric": metric,
            "embedding_source": source,
            "filtration_model": f"model_{metric}_vietoris_rips_2_skeleton",
            "probability_transform": transform,
        },
        "simplices": [{"simplex": ["v0"], "dimension": 0, "filtration": 0.0}],
    }


def _complete_record() -> SimpleNamespace:
    graph_json = {
        "nodes": [
            {"id": "seed", "type": "problem", "text": "seed"},
            {"id": "expand_001", "type": "reasoning_step", "action": "expand", "microstep_count": 3},
            {
                "id": "expand_parse_002",
                "type": "reasoning_step_parse",
                "action": "expand",
                "reasoning_step_id": "expand_001",
                "microstep_index": 0,
            },
            {
                "id": "expand_subgoal_003",
                "type": "reasoning_step_subgoal",
                "action": "expand",
                "reasoning_step_id": "expand_001",
                "microstep_index": 1,
            },
            {
                "id": "expand_proposal_004",
                "type": "reasoning_step_proposal",
                "action": "expand",
                "reasoning_step_id": "expand_001",
                "microstep_index": 2,
            },
        ],
        "edges": [
            {"source": "seed", "target": "expand_001", "type": "expand_transition"},
            {"source": "expand_001", "target": "expand_parse_002", "type": "starts_microstep"},
            {"source": "expand_parse_002", "target": "expand_subgoal_003", "type": "next_microstep"},
            {"source": "expand_subgoal_003", "target": "expand_proposal_004", "type": "next_microstep"},
        ],
    }
    return SimpleNamespace(
        metadata={
            "scaling_action": "expand",
            "reasoning_microstep_count": 3,
            "reasoning_microsteps": [
                {"id": "expand_parse_002", "text": "parse"},
                {"id": "expand_subgoal_003", "text": "subgoal"},
                {"id": "expand_proposal_004", "text": "proposal"},
            ],
        },
        graph_json=graph_json,
    )


def _row() -> dict:
    return {
        "record": _complete_record(),
        "record_id": "candidate-0",
        "path": ["expand"],
        "parent": "seed",
        "level": 1,
        "score": -1.0,
        "nll": 1.25,
        "token_count": 128,
        "graph_tokens": 3,
        "node_tokens": 2,
        "edge_tokens": 1,
        "margin_mean": 0.5,
        "gflownet_action_probs": [{"action": "expand", "probability": 1.0}],
        "action_probability_vector": [1.0],
        "action_probability_source": "TropicalGTModel.gfn(graph_state).softmax",
        "embedding": [0.1, 0.2, 0.3],
        "embedding_source": "TropicalGTModel.graph_state",
        "input_text": "Question: test\nAnswer:",
        "target_text": "target",
        "decoded_argmax": "decoded",
        "graph_json_summary": {"node_count": 2, "edge_count": 1},
        "graphcg_projection": {"basis": "effective_full_rank_qr", "all_direction_cosines": [0.1, -0.2, 0.3]},
        "graph_token_trace": {
            "graph_token_count": 3,
            "tokens": [{"index": 0}, {"index": 1}, {"index": 2}],
            "truncated": False,
        },
        "filtered_simplicial_object": _complex("jensen_shannon", "model_tropical_support_probabilities"),
        "filtered_simplicial_object_source": "probability_filtered_simplicial_object",
        "probability_filtered_simplicial_object": _complex("jensen_shannon", "model_tropical_support_probabilities"),
        "probability_filtered_simplicial_object_source": "TropicalGTModel.graph_token_support_probabilities+jensen_shannon",
        "embedding_filtered_simplicial_object": _complex("euclidean", "TropicalGTModel.graph_token_embeddings"),
        "embedding_filtered_simplicial_object_source": "TropicalGTModel.graph_token_embeddings+euclidean",
        "record_diagnostics": {"topological_algebra": {"available": True}},
    }


def test_public_candidate_marks_complete_model_derived_reasoning_step():
    candidate = _public_candidate(_row())

    assert candidate["reasoning_step_complete"] is True
    assert candidate["reasoning_step_completeness"]["missing"] == []
    assert candidate["reasoning_step_model_data"]["probability_distance"] == "jensen_shannon"
    assert candidate["reasoning_step_model_data"]["embedding_distance"] == "euclidean"
    assert candidate["reasoning_step_structure"]["microstep_count"] == 3


def test_public_candidate_refuses_truncated_reasoning_step_trace():
    row = _row()
    row["graph_token_trace"]["tokens"] = row["graph_token_trace"]["tokens"][:1]
    row["graph_token_trace"]["truncated"] = True

    candidate = _public_candidate(row)

    assert candidate["reasoning_step_complete"] is False
    assert "graph_token_trace_complete" in candidate["reasoning_step_completeness"]["missing"]


def test_public_candidate_refuses_incomplete_reasoning_step_structure():
    row = _row()
    row["record"] = SimpleNamespace(metadata={"scaling_action": "expand"}, graph_json={"nodes": [], "edges": []})

    candidate = _public_candidate(row)

    assert candidate["reasoning_step_complete"] is False
    assert "complete_reasoning_step_structure" in candidate["reasoning_step_completeness"]["missing"]


def test_reasoning_step_audit_summarizes_incomplete_candidates():
    complete = _public_candidate(_row())
    incomplete_row = _row()
    incomplete_row["record_id"] = "candidate-1"
    incomplete_row["embedding"] = []
    incomplete = _public_candidate(incomplete_row)

    audit = _reasoning_step_audit([complete, incomplete], require_complete=True)

    assert audit["require_complete"] is True
    assert audit["candidate_count"] == 2
    assert audit["complete_candidate_count"] == 1
    assert audit["incomplete_candidate_count"] == 1
    assert audit["missing_counts"]["graph_state_embedding"] == 1
    assert audit["incomplete_candidates"][0]["record_id"] == "candidate-1"


def test_inference_scaling_can_fail_closed_when_complete_steps_required(monkeypatch):
    import tropicalgt.scaling as scaling

    row = _row()
    row["graph_token_trace"]["tokens"] = row["graph_token_trace"]["tokens"][:1]
    row["graph_token_trace"]["truncated"] = True
    row["record"] = object()

    def fake_score_records(*args, **kwargs):
        return [dict(row)]

    monkeypatch.setattr(scaling, "score_records", fake_score_records)

    with pytest.raises(ValueError, match="requires complete model-derived reasoning steps"):
        run_inference_scaling(
            model=object(),
            seed_record=object(),
            tokenizer=object(),
            seq_len=1,
            device=object(),
            depth=0,
            width=1,
            branch_factor=1,
            require_complete_reasoning_steps=True,
        )
