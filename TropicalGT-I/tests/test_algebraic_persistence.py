import torch

from tropicalgt.algebra import compute_topological_algebra_report, summarize_algebra_reports
from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.records import GraphRecord
from tropicalgt.scaling import run_inference_scaling
from tropicalgt.simplicial import build_filtered_simplicial_object, build_reasoning_trajectory_complex
from tropicalgt.tokenizer import TokenGTTokenizer


def test_topological_algebra_report_has_multiparameter_data():
    record = FixtureGraphDataset(1)[0]
    filtered = build_filtered_simplicial_object(record)
    report = compute_topological_algebra_report(filtered, audit_level="full", ph_backend="gudhi", max_simplices=128)
    assert report["enabled"] is True
    assert report["chain_complex"]["field"] == "F2"
    assert report["multiparameter_persistence"]["num_parameters"] == 3
    assert report["multiparameter_persistence"]["fiber_rank_profile"]
    reps = report["persistence_representations"]
    assert reps["backend"] == "gudhi.representations"
    assert "decision_policy" in reps
    if reps["available"]:
        assert reps["summary"]["landscape_l2_norm"] >= 0.0
        assert reps["summary"]["topological_vector_l2_norm"] >= 0.0
        assert any(row.get("method") == "Landscape" for row in reps["decision_policy"])
    assert "commutative_algebra" in report
    proxy = report["commutative_algebra"]["multiparameter_free_resolution_proxy"]
    assert proxy["ring"] == "F2[x_filtration,x_dimension,x_position]"
    assert proxy["free_chain_modules"]
    assert proxy["determinantal_ideals"]["available"] is True
    assert proxy["fitting_ideals"]["available"] is True
    assert proxy["buchsbaum_eisenbud"]["available"] is True
    assert "maps" in proxy["determinantal_ideals"]
    assert "maps" in proxy["fitting_ideals"]
    assert "rank_exactness_checks" in proxy["buchsbaum_eisenbud"]
    assert proxy["minimal_free_resolution"]["available"] is False
    summary = summarize_algebra_reports([report])
    assert summary["algebra_reports"] == 1.0


def test_ripser_backend_is_selected_when_requested():
    record = FixtureGraphDataset(1)[0]
    filtered = build_filtered_simplicial_object(record)
    report = compute_topological_algebra_report(filtered, audit_level="topology", ph_backend="ripser", max_simplices=128)
    assert report["persistence"]["backend"] == "ripser"
    assert "available" in report["persistence"]


def test_reasoning_trajectory_complex_grows_by_level():
    candidates = [
        {"record_id": "root", "level": 0, "score": 0.0, "nll": 1.0, "path": [], "parent": None, "embedding": [0.0, 0.0, 0.0]},
        {"record_id": "a", "level": 1, "score": 0.1, "nll": 0.9, "path": ["expand"], "parent": "root", "embedding": [1.0, 0.0, 0.0]},
        {"record_id": "b", "level": 2, "score": 0.2, "nll": 0.8, "path": ["expand", "verify"], "parent": "a", "embedding": [0.0, 1.0, 0.0]},
    ]
    level_one = build_reasoning_trajectory_complex(candidates, up_to_level=1)
    full = build_reasoning_trajectory_complex(candidates)
    assert level_one["summary"]["num_vertices"] == 2
    assert full["summary"]["num_vertices"] == 3
    assert full["summary"]["num_two_simplices"] == 1


def test_inference_scaling_emits_step_and_trajectory_algebra():
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
        audit_level="full",
        ph_backend="gudhi",
        audit_max_simplices=128,
    )
    assert report["candidates"][0]["topological_algebra"]["multiparameter_persistence"]["num_parameters"] == 3
    assert report["trajectory_topological_algebra"]["multiparameter_persistence"]["fiber_rank_profile"]
    assert report["trajectory_growth"]


def test_sequential_text_is_always_graphified():
    record = GraphRecord.from_mapping({"record_id": "plain", "text": "alpha beta gamma delta"})
    types = {node["type"] for node in record.graph_json["nodes"]}
    assert "sequence_chunk" in types
    assert record.metadata["graph_json_sequentialized"] is True
