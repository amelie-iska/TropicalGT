from dataclasses import asdict
import json
import math

import torch

from tropicalgt.data import FixtureGraphDataset, encode_bytes
from tropicalgt.memory import (
    AnalogicalMemoryBank,
    AnalogicalMemoryQualityGate,
    AnalogicalMemoryRecord,
    memory_quality_gate_summary,
    memory_records_from_scaling_report,
    persistence_landscape_vector_similarity,
    persistence_vector_representation_similarity,
    probability_simplicial_map_diagnostics,
    query_probability_complex_from_report,
    query_signature_from_report,
)
from tropicalgt.metrics import batch_bpb_metrics, explicit_graph_json_bytes, graph_token_structural_bytes
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.scaling import run_inference_scaling
from tropicalgt.tokenizer import TokenGTTokenizer


def test_bpb_metrics_account_for_text_and_graph_bytes():
    records = [FixtureGraphDataset(1)[0]]
    tok = TokenGTTokenizer(feature_dim=48)
    graph_batch = tok.batch_encode(records)
    _, y = encode_bytes(records[0].text, 32)
    metrics = batch_bpb_metrics(1.0, y[None, :], graph_batch, records)
    assert math.isclose(metrics["bpb"], 1.0 / math.log(2.0), rel_tol=1e-6)
    assert metrics["target_bytes"] > 0
    assert metrics["graph_token_structural_bytes"] == graph_token_structural_bytes(graph_batch)
    assert metrics["graph_bpb"] == metrics["graph_bpb"]


def test_explicit_graph_bytes_ignore_derived_text_graph_tokenization():
    record = FixtureGraphDataset(1)[0]
    assert explicit_graph_json_bytes(record) > 0
    record.metadata = {"graph_json_fallback": False, "graph_json_derived_from_text": True, "graph_json_sequentialized": True}
    assert explicit_graph_json_bytes(record) == 0


def test_analogical_memory_bank_roundtrip_and_retrieval(tmp_path):
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
        audit_level="full",
        ph_backend="gudhi",
        audit_max_simplices=128,
    )
    records = memory_records_from_scaling_report(report, max_records=2)
    assert records
    path = tmp_path / "memory.jsonl"
    bank = AnalogicalMemoryBank(path, max_records=8)
    bank.extend(records)
    bank.save()
    loaded = AnalogicalMemoryBank(path, max_records=8)
    query_embedding, query_signature = query_signature_from_report({"inference_scaling": report})
    retrieved = loaded.retrieve(query_embedding, query_signature, top_k=2)
    assert retrieved
    assert "filtered_summary" in retrieved[0]
    trajectory_summary = report["trajectory_filtered_simplicial_object"]["summary"]
    candidate_summaries = {row["record_id"]: row["filtered_simplicial_object"]["summary"] for row in report["candidates"]}
    stored_summaries = [record.filtered_simplicial_object["summary"] for record in records]
    assert all(record.filtered_simplicial_object["summary"] == candidate_summaries[record.record_id] for record in records)
    assert any(summary != trajectory_summary for summary in stored_summaries)
    serialized = [json.dumps(asdict(record)) for record in records]
    assert max(len(row) for row in serialized) < 1_000_000
    for record in records:
        assert record.metadata["memory_payload_policy"] == "compact_real_trajectory_payload_no_duplicate_full_complexes"
        compact_trajectory = record.metadata["trajectory_filtered_simplicial_object"]
        compact_probability = record.metadata["trajectory_probability_filtered_simplicial_object"]
        assert compact_trajectory["summary"] == trajectory_summary
        assert "summary" in compact_probability
        assert len(json.dumps(compact_trajectory)) < 500_000
        assert len(json.dumps(compact_probability)) < 500_000
        assert "summary" in record.filtered_simplicial_object
    assert "record_family" in retrieved[0]
    assert "base_retrieval_score" in retrieved[0]
    probability_vertices = [
        simplex
        for record in records
        for simplex in record.probability_filtered_simplicial_object.get("simplices", [])
        if simplex.get("dimension") == 0
    ]
    assert any(simplex.get("probability") or simplex.get("model_probability_vector") or simplex.get("probability_vector") for simplex in probability_vertices)
    probability_summary = retrieved[0]["trajectory_probability_filtered_simplicial_object"]["summary"]
    assert retrieved[0]["trajectory_probability_filtered_summary"] == probability_summary
    assert "jensen_shannon" in probability_summary["filtration_model"]
    assert retrieved[0]["trajectory_probability_filtered_simplicial_object"].get("available") is not False

    source_bank = AnalogicalMemoryBank(tmp_path / "source_memory.jsonl", max_records=8)
    source_a = memory_records_from_scaling_report(report, source="source-a", max_records=1)
    source_b = memory_records_from_scaling_report(report, source="source-b", max_records=1)
    assert source_a and source_b and source_a[0].record_id == source_b[0].record_id
    source_bank.extend(source_a + source_b)
    source_hits = source_bank.retrieve(query_embedding, query_signature, top_k=4, exclude_sources={"source-a"})
    assert source_hits
    assert all(row["trajectory_source"] != "source-a" for row in source_hits)
    assert "trajectory_probability_filtered_simplicial_object" in source_hits[0]
    assert "jensen_shannon" in source_hits[0]["trajectory_probability_filtered_summary"]["filtration_model"]
    assert any(row["trajectory_source"] == "source-b" for row in source_hits)
    assert source_bank.retrieve(query_embedding, query_signature, top_k=4, exclude_sources={"source-a", "source-b"}) == []


def _landscape_topology(vector, *, extra_vectors=None):
    row = {
        "available": True,
        "landscape": {
            "source": "gudhi.representations.Landscape.vector",
            "vector": [float(value) for value in vector],
        },
    }
    if extra_vectors:
        row.update(extra_vectors)
    return {
        "persistence_representations": {
            "available": True,
            "source": "gudhi.representations",
            "methods": {"0": row},
        }
    }


def _memory_record(record_id, topology):
    return AnalogicalMemoryRecord(
        memory_id=f"mem-{record_id}",
        record_id=record_id,
        score=1.0,
        nll=1.0,
        embedding=[1.0, 0.0, 0.0],
        signature_vector=[1.0, 0.0, 0.0],
        trajectory_embeddings=[[0.0, 0.0, 0.0]],
        trajectory_edges=[],
        trajectory_paths=[[]],
        filtered_simplicial_object={"summary": {"simplices": 1}, "simplices": []},
        probability_filtered_simplicial_object={"summary": {"filtration_model": "probability_jensen_shannon", "simplices": 1}, "simplices": []},
        topological_algebra=topology,
        derived_signature={"betti_vector": [1, 0]},
        metadata={"source": record_id, "trajectory_probability_topological_algebra": topology},
    )


def _topology_with_certified_real_resolution(*, input_hash="hash-a", fitt0="ideal(x_level,x_radius)", multiplier_matrix="matrix {{1}}"):
    free_modules = [
        {"homological_degree": 0, "multidegree": [0, 0], "rank": 1, "display": "F_0 contains S(-0,0)^1"},
        {"homological_degree": 1, "multidegree": [1, 0], "rank": 1, "display": "F_1 contains S(-1,0)^1"},
    ]
    real = {
        "schema_version": "tropicalgt.real_free_resolution.v1",
        "available": True,
        "status": "certified",
        "backend": "Macaulay2",
        "coefficient_ring": "F2[x_level,x_radius]",
        "input_sha256": input_hash,
        "certificate_attached": True,
        "exactness_certified": True,
        "minimality_certified": True,
        "real_free_resolution_certified": True,
        "multigraded_free_resolution_certified": True,
        "safe_to_render_as_multigraded_free_resolution": True,
        "free_resolution_summary": {
            "available": True,
            "safe_for_multigraded_claims": True,
            "not_multigraded": False,
            "betti_by_homological_and_multidegree": {"0": {"0,0": 1}, "1": {"1,0": 1}},
            "free_modules": free_modules,
        },
        "cas_artifacts": {
            "differentials": [
                {
                    "homological_degree": 1,
                    "rows": 1,
                    "cols": 1,
                    "source_degrees": [[1, 0]],
                    "target_degrees": [[0, 0]],
                    "matrix_text": "matrix {{x_level}}",
                }
            ],
            "fitting_ideals": {"Fitt0": fitt0, "Fitt1": "ideal 1"},
            "minors": {"minors_1": fitt0},
            "buchsbaum_eisenbud_diagnostics": {
                "available": True,
                "multiplier_output_available": True,
                "bemultipliers_status": "computed_aMultiplier_1",
                "a_multiplier_1_shape": "1x1",
                "a_multiplier_1_matrix": multiplier_matrix,
            },
        },
    }
    return {
        "persistence": {"intervals": [{"dimension": 0, "birth": 0.0, "death": None, "infinite": True}]},
        "derived_equivalence_signature": {
            "betti_vector": [1, 0, 0, 0],
            "persistence_finite_interval_count": 0,
            "persistence_infinite_interval_count": 1,
            "persistence_total_finite_length": 0.0,
            "multiparameter_grid_points": 1,
            "multiparameter_h0_rank_sample": [{"h0_rank": 1}],
        },
        "commutative_algebra": {
            "two_parameter_chain_presentation_diagnostics": {
                "ring": "F2[x_level,x_radius]",
                "free_chain_modules": [
                    {"homological_degree": 0, "rank": 1},
                    {"homological_degree": 1, "rank": 1},
                ],
                "real_free_resolution": real,
            }
        },
    }


def test_analogical_memory_retrieval_scores_only_matching_certified_cas_evidence(tmp_path):
    query_topology = _topology_with_certified_real_resolution()
    matching_topology = _topology_with_certified_real_resolution()
    mismatched_topology = _topology_with_certified_real_resolution(
        input_hash="hash-b",
        fitt0="ideal(x_level)",
        multiplier_matrix="matrix {{0}}",
    )
    bank = AnalogicalMemoryBank(tmp_path / "certified_cas_memory.jsonl", max_records=8)
    bank.extend(
        [
            _memory_record("mismatch", mismatched_topology),
            _memory_record("match", matching_topology),
        ]
    )
    hits = bank.retrieve(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        top_k=2,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=0.0,
        vector_representation_weight=0.0,
        probability_map_weight=0.0,
        certified_cas_weight=1.0,
        diversity_weight=0.0,
        query_topology=query_topology,
    )
    assert [row["record_id"] for row in hits] == ["match", "mismatch"]
    assert hits[0]["retrieval_weights"]["certified_cas_weight"] == 1.0
    assert hits[0]["certified_cas_evidence_available"] is True
    assert hits[0]["certified_cas_evidence_match"] is True
    assert hits[0]["certified_cas_retrieval_similarity"] == 1.0
    assert hits[0]["certified_cas_score_contribution"] == 1.0
    assert hits[0]["retrieval_score_components"]["certified_cas_evidence"] == 1.0
    assert hits[0]["certified_cas_evidence"]["safe_for_derived_category_claims"] is False
    assert hits[0]["certified_cas_evidence"]["derived_category_claim"] == "not_asserted_by_retrieval_scoring"
    assert math.isclose(hits[0]["retrieval_score"], hits[0]["certified_cas_score_contribution"], rel_tol=1e-9)

    assert hits[1]["certified_cas_evidence_available"] is True
    assert hits[1]["certified_cas_evidence_match"] is False
    assert hits[1]["certified_cas_score_contribution"] == 0.0
    assert hits[1]["certified_cas_retrieval_similarity"] == 0.0
    assert 0.0 < hits[1]["certified_cas_evidence_similarity"] < 1.0
    assert {"input_sha256", "artifact_hash", "fitting_ideals", "minors", "buchsbaum_eisenbud"}.issubset(
        set(hits[1]["certified_cas_mismatched_components"])
    )

    unavailable_hits = bank.retrieve(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        top_k=2,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=0.0,
        vector_representation_weight=0.0,
        probability_map_weight=0.0,
        certified_cas_weight=1.0,
        diversity_weight=0.0,
        query_topology={},
    )
    assert unavailable_hits
    assert all(row["certified_cas_score_contribution"] == 0.0 for row in unavailable_hits)
    assert all(row["certified_cas_evidence_available"] is False for row in unavailable_hits)



def _probability_complex(prefix, *, probabilities=None):
    labels = [f"{prefix}0", f"{prefix}1", f"{prefix}2"]
    probabilities = probabilities or [
        [0.90, 0.08, 0.02],
        [0.05, 0.90, 0.05],
        [0.02, 0.08, 0.90],
    ]
    simplices = []
    for index, (label, probs) in enumerate(zip(labels, probabilities)):
        simplices.append(
            {
                "dimension": 0,
                "label": label,
                "simplex": [label],
                "level": index,
                "filtration": 0.0,
                "probability": [float(value) for value in probs],
                "probability_source": "model_logits_softmax",
            }
        )
    for edge in ((0, 1), (0, 2), (1, 2)):
        simplices.append({"dimension": 1, "simplex": [labels[edge[0]], labels[edge[1]]], "filtration": 0.25})
    simplices.append({"dimension": 2, "simplex": labels, "filtration": 0.50})
    return {
        "available": True,
        "summary": {
            "filtration_model": "model_probability_jensen_shannon_vietoris_rips_2_skeleton",
            "vertices": 3,
            "simplices": len(simplices),
        },
        "simplices": simplices,
    }


def _probability_memory_record(record_id, probability_complex):
    return AnalogicalMemoryRecord(
        memory_id=f"mem-{record_id}",
        record_id=record_id,
        score=0.0,
        nll=0.0,
        embedding=[0.0, 0.0, 0.0],
        signature_vector=[0.0, 0.0, 0.0],
        trajectory_embeddings=[],
        trajectory_edges=[],
        trajectory_paths=[],
        filtered_simplicial_object={"summary": {"simplices": 0}, "simplices": []},
        probability_filtered_simplicial_object=probability_complex,
        topological_algebra={},
        derived_signature={},
        metadata={
            "source": record_id,
            "trajectory_probability_filtered_simplicial_object": probability_complex,
        },
    )

def test_analogical_memory_retrieval_uses_gudhi_vector_representation_family(tmp_path):
    query_topology = _landscape_topology(
        [0.0, 0.25, 0.75, 0.25, 0.0],
        extra_vectors={
            "betti_curve": {"values": [0.0, 1.0, 1.0, 0.0]},
            "silhouette": {"values": [0.0, 0.4, 0.2, 0.0]},
            "entropy": {"vector": [0.0, 0.1, 0.1, 0.0]},
            "persistence_lengths": {"values": [0.75, 0.25]},
            "topological_vector": {"values": [0.2, 0.3, 0.5]},
            "persistence_image": {"values": [[0.0, 0.2], [0.1, 0.0]]},
        },
    )
    matching_topology = _landscape_topology(
        [0.0, 0.24, 0.74, 0.26, 0.0],
        extra_vectors={
            "betti_curve": {"values": [0.0, 1.0, 0.9, 0.0]},
            "silhouette": {"values": [0.0, 0.39, 0.21, 0.0]},
            "entropy": {"vector": [0.0, 0.11, 0.1, 0.0]},
            "persistence_lengths": {"values": [0.74, 0.26]},
            "topological_vector": {"values": [0.21, 0.29, 0.49]},
            "persistence_image": {"values": [[0.0, 0.21], [0.1, 0.0]]},
        },
    )
    mismatched_topology = _landscape_topology(
        [0.9, 0.1, 0.0, 0.1, 0.9],
        extra_vectors={
            "betti_curve": {"values": [3.0, 0.0, 0.0, 3.0]},
            "silhouette": {"values": [1.0, 0.0, 0.0, 1.0]},
            "entropy": {"vector": [1.0, 0.0, 0.0, 1.0]},
            "persistence_lengths": {"values": [0.05, 0.04]},
            "topological_vector": {"values": [2.0, -1.0, 0.0]},
            "persistence_image": {"values": [[1.0, 0.0], [0.0, 1.0]]},
        },
    )
    similarity = persistence_vector_representation_similarity(query_topology, matching_topology)
    mismatch = persistence_vector_representation_similarity(query_topology, mismatched_topology)
    assert similarity["available"] is True
    assert similarity["source"] == "gudhi.representations.vector_methods"
    assert set(similarity["available_methods"]) >= {"landscape", "betti_curve", "silhouette", "persistence_image", "topological_vector"}
    assert similarity["aggregate_similarity"] > mismatch["aggregate_similarity"]
    assert similarity["components"]["persistence_image"]["overlap_dim"] == 4

    bank = AnalogicalMemoryBank(tmp_path / "vector_family_memory.jsonl", max_records=8)
    bank.extend([
        _memory_record("mismatch", mismatched_topology),
        _memory_record("match", matching_topology),
    ])
    hits = bank.retrieve(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        top_k=2,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=0.25,
        vector_representation_weight=1.0,
        diversity_weight=0.0,
        query_topology=query_topology,
    )
    assert [row["record_id"] for row in hits] == ["match", "mismatch"]
    assert hits[0]["retrieval_weights"]["persistence_landscape_weight"] == 0.25
    assert hits[0]["retrieval_weights"]["persistence_vector_weight"] == 1.0
    assert hits[0]["retrieval_weights"]["persistence_vector_includes_landscape"] is False
    assert hits[0]["retrieval_weights"]["legacy_landscape_weight_alias_mode"] is False
    assert hits[0]["base_retrieval_score"] == 0.0
    assert hits[0]["persistence_landscape_score_contribution"] > 0.0
    assert hits[0]["persistence_vector_score_contribution"] > 0.0
    components = hits[0]["retrieval_score_components"]
    assert components["persistence_landscape"] == hits[0]["persistence_landscape_score_contribution"]
    assert components["persistence_vector_family"] == hits[0]["persistence_vector_score_contribution"]
    assert math.isclose(
        hits[0]["retrieval_score"],
        hits[0]["base_retrieval_score"]
        + hits[0]["persistence_landscape_score_contribution"]
        + hits[0]["persistence_vector_score_contribution"],
        rel_tol=1e-9,
    )
    assert hits[0]["persistence_vector_representation_similarity"]["available"] is True
    assert hits[0]["persistence_vector_representation_similarity"]["includes_landscape"] is False
    assert hits[0]["persistence_vector_aggregate_similarity"] > hits[1]["persistence_vector_aggregate_similarity"]
    assert "persistence_image" in hits[0]["persistence_vector_available_methods"]
    assert "landscape" not in hits[0]["persistence_vector_available_methods"]


def test_analogical_memory_retrieval_uses_persistence_landscape_vectors(tmp_path):
    query_topology = _landscape_topology([0.0, 0.25, 0.75, 0.25, 0.0])
    matching_topology = _landscape_topology([0.0, 0.24, 0.74, 0.26, 0.0])
    mismatched_topology = _landscape_topology([0.9, 0.1, 0.0, 0.1, 0.9])
    similarity = persistence_landscape_vector_similarity(query_topology, matching_topology)
    mismatch = persistence_landscape_vector_similarity(query_topology, mismatched_topology)
    assert similarity["available"] is True
    assert similarity["source"] == "gudhi.representations.Landscape.vector"
    assert similarity["l2_similarity"] > mismatch["l2_similarity"]

    bank = AnalogicalMemoryBank(tmp_path / "landscape_memory.jsonl", max_records=8)
    bank.extend([
        _memory_record("mismatch", mismatched_topology),
        _memory_record("match", matching_topology),
    ])
    hits = bank.retrieve(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        top_k=2,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=1.0,
        vector_representation_weight=0.0,
        diversity_weight=0.0,
        query_topology=query_topology,
    )
    assert [row["record_id"] for row in hits] == ["match", "mismatch"]
    assert hits[0]["retrieval_weights"]["persistence_landscape_weight"] == 1.0
    assert hits[0]["retrieval_weights"]["persistence_vector_weight"] == 0.0
    assert hits[0]["retrieval_weights"]["persistence_vector_includes_landscape"] is False
    assert hits[0]["persistence_landscape_vector_similarity"]["available"] is True
    assert hits[0]["persistence_landscape_l2_similarity"] > hits[1]["persistence_landscape_l2_similarity"]
    assert hits[0]["persistence_vector_score_contribution"] == 0.0
    assert hits[0]["persistence_landscape_overlap_dim"] == 5


def test_probability_simplicial_map_diagnostics_certifies_filtered_chain_map():
    query_complex = _probability_complex("q")
    memory_complex = _probability_complex("m")
    report = probability_simplicial_map_diagnostics(query_complex, memory_complex)
    assert report["available"] is True
    assert report["map_source"] == "model_probability_jensen_shannon_assignment"
    assert report["simplex_tree_map_checked"] == 7
    assert report["simplex_tree_map_preserved"] == 7
    assert math.isclose(report["simplex_tree_map_preservation_rate"], 1.0)
    assert report["chain_map_diagnostics"]["available"] is True
    assert report["chain_map_diagnostics"]["boundary_commutation_certified"] is True
    assert report["persistence_module_morphism_diagnostics"]["morphism_certified"] is True
    assert report["safe_to_render_as_simplicial_map"] is True
    assert report["safe_to_render_as_chain_map"] is True
    assert report["safe_to_render_as_persistence_module_morphism"] is True
    assert report["map_render_claim"] == "certified_filtered_simplicial_map"
    assert report["map_claim_failure_reason"] is None
    assert report["no_proxy_or_fallback"] is True
    assert report["simplex_tree_map"]["filtered_simplicial_map_certified"] is True
    assert "certifies a filtered simplicial map" in report["simplex_tree_map"]["interpretation"]
    wrapped = {"inference_scaling": {"trajectory_probability_filtered_simplicial_object": query_complex}}
    assert query_probability_complex_from_report(wrapped) == query_complex


def test_analogical_memory_retrieval_uses_probability_simplicial_map_weight(tmp_path):
    query_complex = _probability_complex("q")
    matching_complex = _probability_complex("m")
    uniform_complex = _probability_complex("u", probabilities=[[1.0 / 3.0] * 3, [1.0 / 3.0] * 3, [1.0 / 3.0] * 3])
    bank = AnalogicalMemoryBank(tmp_path / "probability_map_memory.jsonl", max_records=8)
    bank.extend(
        [
            _probability_memory_record("uniform", uniform_complex),
            _probability_memory_record("match", matching_complex),
        ]
    )
    hits = bank.retrieve(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        top_k=2,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=0.0,
        vector_representation_weight=0.0,
        probability_map_weight=1.0,
        diversity_weight=0.0,
        query_probability_complex=query_complex,
    )
    assert [row["record_id"] for row in hits] == ["match", "uniform"]
    assert hits[0]["probability_simplicial_map_available"] is True
    assert hits[0]["probability_simplicial_map_source"] == "model_probability_jensen_shannon_assignment"
    assert hits[0]["retrieval_weights"]["probability_simplicial_map_weight"] == 1.0
    assert hits[0]["retrieval_score_components"]["probability_simplicial_map"] > hits[1]["retrieval_score_components"]["probability_simplicial_map"]
    assert hits[0]["probability_simplicial_map_similarity"] > hits[1]["probability_simplicial_map_similarity"]
    assert hits[0]["probability_simplicial_map"]["chain_map_diagnostics"]["safe_to_use_as_persistence_module_morphism"] is True
    assert hits[0]["probability_simplicial_map_scoring_policy"].startswith("positive score only")
    assert hits[0]["probability_simplicial_map_vertex_assignment_count"] == 3
    assert hits[0]["probability_simplicial_map_checked_edges"] == 3
    assert hits[0]["probability_simplicial_map_preserved_edges"] == 3
    assert math.isclose(hits[0]["probability_simplicial_map_edge_preservation_rate"], 1.0)
    assert hits[0]["probability_simplicial_map_checked_two_simplices"] == 1
    assert hits[0]["probability_simplicial_map_preserved_two_simplices"] == 1
    assert hits[0]["probability_simplicial_map_chain_map_certified"] is True
    assert hits[0]["probability_simplicial_map_persistence_morphism_certified"] is True


def test_analogical_memory_probability_map_must_preserve_simplex_tree_to_score(tmp_path):
    query_complex = _probability_complex("q")
    vertex_only_complex = _probability_complex("v")
    vertex_only_complex["simplices"] = [row for row in vertex_only_complex["simplices"] if row.get("dimension") == 0]
    vertex_only_complex["summary"] = {**vertex_only_complex["summary"], "simplices": len(vertex_only_complex["simplices"])}
    bank = AnalogicalMemoryBank(tmp_path / "probability_map_no_proxy_memory.jsonl", max_records=4)
    bank.extend([_probability_memory_record("vertices-only", vertex_only_complex)])
    hits = bank.retrieve(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        top_k=1,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=0.0,
        vector_representation_weight=0.0,
        probability_map_weight=1.0,
        diversity_weight=0.0,
        query_probability_complex=query_complex,
    )
    assert hits[0]["record_id"] == "vertices-only"
    assert hits[0]["probability_simplicial_map_available"] is False
    assert hits[0]["probability_simplicial_map_similarity"] == 0.0
    assert hits[0]["probability_simplicial_map_score_contribution"] == 0.0
    assert hits[0]["probability_simplicial_map_checked_edges"] == 3
    assert hits[0]["probability_simplicial_map_preserved_edges"] == 0
    assert hits[0]["probability_simplicial_map_missing_edges"] == 3
    assert hits[0]["probability_simplicial_map_checked_two_simplices"] == 1
    assert hits[0]["probability_simplicial_map_preserved_two_simplices"] == 0
    assert hits[0]["probability_simplicial_map_missing_two_simplices"] == 1
    assert hits[0]["probability_simplicial_map_chain_map_certified"] is False
    assert hits[0]["probability_simplicial_map_persistence_morphism_certified"] is False
    assert hits[0]["probability_simplicial_map_safe_to_render_as_simplicial_map"] is False
    assert hits[0]["probability_simplicial_map_safe_to_render_as_chain_map"] is False
    assert hits[0]["probability_simplicial_map_safe_to_render_as_persistence_module_morphism"] is False
    assert hits[0]["probability_simplicial_map_render_claim"] == "probability_correspondence_not_a_simplicial_map"
    assert hits[0]["probability_simplicial_map_claim_failure_reason"] == "simplex_tree_map_not_fully_preserved"
    assert hits[0]["probability_simplicial_map_no_proxy_or_fallback"] is True
    assert hits[0]["probability_simplicial_map"]["simplex_tree_map"]["filtered_simplicial_map_certified"] is False
    assert "no simplicial map" in hits[0]["probability_simplicial_map"]["simplex_tree_map"]["interpretation"]


def test_analogical_memory_retrieval_reports_transported_landscape_diagnostics(tmp_path):
    query_complex = _probability_complex("q")
    matching_complex = _probability_complex("m")
    query_topology = _landscape_topology([0.0, 0.25, 0.75, 0.25, 0.0])
    matching_topology = _landscape_topology([0.0, 0.24, 0.74, 0.26, 0.0])

    record = _probability_memory_record("match", matching_complex)
    record.topological_algebra = matching_topology
    record.metadata["trajectory_probability_topological_algebra"] = matching_topology
    bank = AnalogicalMemoryBank(tmp_path / "transported_landscape_memory.jsonl", max_records=8)
    bank.extend([record])

    hits = bank.retrieve(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        top_k=1,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=0.0,
        vector_representation_weight=0.0,
        probability_map_weight=1.0,
        diversity_weight=0.0,
        query_topology=query_topology,
        query_probability_complex=query_complex,
    )
    assert hits[0]["transported_landscape_available"] is True
    assert hits[0]["transported_landscape"]["source"] == "probability_simplicial_map_plus_gudhi_landscape"
    assert hits[0]["transported_landscape"]["chain_map_certified"] is True
    assert hits[0]["transported_landscape_l2"] >= 0.0
    assert hits[0]["transported_landscape_cosine"] > 0.0
    assert hits[0]["transported_landscape_l2_similarity"] > 0.0

    unavailable_hits = bank.retrieve(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        top_k=1,
        embedding_weight=0.0,
        signature_weight=0.0,
        score_weight=0.0,
        landscape_weight=0.0,
        vector_representation_weight=0.0,
        probability_map_weight=0.0,
        diversity_weight=0.0,
        query_topology=query_topology,
        query_probability_complex={},
    )
    assert unavailable_hits[0]["transported_landscape_available"] is False
    assert unavailable_hits[0]["transported_landscape_reason"] == "probability_simplicial_map_unavailable"


def test_analogical_memory_quality_gate_rejects_low_quality_storage(tmp_path):
    report = {
        "candidates": [
            {
                "record_id": "root",
                "level": 0,
                "score": -1.0,
                "nll": 4.0,
                "margin_mean": 0.2,
                "embedding": [1.0, 0.0],
                "filtered_simplicial_object": {"summary": {"simplices": 1}, "simplices": []},
                "probability_filtered_simplicial_object": {"available": False, "summary": {}, "simplices": []},
                "topological_algebra": {},
            },
            {
                "record_id": "better",
                "level": 1,
                "score": 1.0,
                "nll": 3.5,
                "margin_mean": 0.3,
                "embedding": [0.0, 1.0],
                "filtered_simplicial_object": {"summary": {"simplices": 1}, "simplices": []},
                "probability_filtered_simplicial_object": {
                    "available": True,
                    "summary": {"filtration_model": "model_tropical_support_probability_jensen_shannon_vietoris_rips_2_skeleton", "simplices": 4},
                    "simplices": [
                        {"dimension": 0, "label": "a", "probability": [0.8, 0.2]},
                        {"dimension": 0, "label": "b", "probability": [0.7, 0.3]},
                        {"dimension": 1, "simplex": ["a", "b"], "filtration": 0.2},
                        {"dimension": 1, "simplex": ["b", "a"], "filtration": 0.2},
                    ],
                },
                "topological_algebra": {"persistence_summary": {"intervals": 1}, "derived_equivalence_signature": {"betti_vector": [2, 1]}},
            },
        ]
    }
    gate = AnalogicalMemoryQualityGate(
        min_nll_improvement=0.0,
        require_probability_complex=True,
        min_probability_vertices=2,
        min_probability_simplices=3,
        require_topological_algebra=True,
    )
    records = memory_records_from_scaling_report(report, quality_gate=gate, max_records=2)
    assert [record.record_id for record in records] == ["better"]
    assert records[0].metadata["quality_gate"]["passed"] is True
    summary = memory_quality_gate_summary(report, gate, max_records=2)
    assert summary["candidate_count"] == 2
    assert summary["eligible_count"] == 1
    assert summary["rejected_count"] == 1
    assert "probability_complex_unavailable" in summary["reason_counts"]
