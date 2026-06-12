from dataclasses import asdict
import json
import math

import torch

from tropicalgt.data import FixtureGraphDataset, encode_bytes
from tropicalgt.memory import AnalogicalMemoryBank, memory_records_from_scaling_report, query_signature_from_report
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


def test_explicit_graph_bytes_ignore_derived_sequence_for_fallbacks():
    record = FixtureGraphDataset(1)[0]
    assert explicit_graph_json_bytes(record) > 0
    record.metadata = {"graph_json_fallback": True, "graph_json_sequentialized": True}
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
        assert "trajectory_filtered_simplicial_object" not in record.metadata
        assert "trajectory_probability_filtered_simplicial_object" not in record.metadata
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
