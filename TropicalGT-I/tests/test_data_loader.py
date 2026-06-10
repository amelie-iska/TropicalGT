from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from tropicalgt.data import ChunkShuffleSampler, ParameterGolfBinGraphDataset, ParquetGraphDataset, make_dataset, make_dataset_from_config, parquet_manifest
from tropicalgt.records import GraphRecord, graph_decoding_order


def test_parquet_loader_reads_graph_json(tmp_path: Path):
    root = tmp_path / "shards" / "train"
    root.mkdir(parents=True)
    pd.DataFrame([{"record_id": "r", "text": "abc", "graph_json": '{"nodes":[{"id":"a"}],"edges":[]}'}]).to_parquet(root / "train-000.parquet")
    ds = ParquetGraphDataset(tmp_path / "shards", "train")
    assert len(ds) == 1
    rec = ds[0]
    assert rec.graph_json["nodes"][0]["id"] == "a"


def test_parquet_loader_indexes_multiple_shards_with_limit(tmp_path: Path):
    root = tmp_path / "shards" / "train"
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"record_id": "r0", "text": "a", "graph_json": '{"nodes":[{"id":"a"}],"edges":[]}'},
            {"record_id": "r1", "text": "b", "graph_json": '{"nodes":[{"id":"b"}],"edges":[]}'},
        ]
    ).to_parquet(root / "train-000.parquet")
    pd.DataFrame(
        [
            {"record_id": "r2", "text": "c", "graph_json": '{"nodes":[{"id":"c"}],"edges":[]}'},
            {"record_id": "r3", "text": "d", "graph_json": '{"nodes":[{"id":"d"}],"edges":[]}'},
        ]
    ).to_parquet(root / "train-001.parquet")
    ds = ParquetGraphDataset(tmp_path / "shards", "train", limit=3, cache_shards=1)
    assert len(ds) == 3
    assert ds[0].record_id == "r0"
    assert ds[2].record_id == "r2"
    assert ds.manifest()["files"] == 2
    assert ds.manifest()["rows"] == 3


def test_chunk_shuffle_sampler_is_deterministic_and_chunk_local(tmp_path: Path):
    root = tmp_path / "shards" / "train"
    root.mkdir(parents=True)
    for shard in range(3):
        rows = [
            {"record_id": f"r{shard}-{i}", "text": f"text {shard}-{i}", "graph_json": '{"nodes":[{"id":"a"}],"edges":[]}'}
            for i in range(2)
        ]
        pd.DataFrame(rows).to_parquet(root / f"train-{shard:03d}.parquet")

    ds = ParquetGraphDataset(tmp_path / "shards", "train", cache_shards=1)
    indices_a = list(ChunkShuffleSampler(ds, seed=7, shuffle_rows=False))
    indices_b = list(ChunkShuffleSampler(ds, seed=7, shuffle_rows=False))

    assert indices_a == indices_b
    assert sorted(indices_a) == list(range(len(ds)))

    positions = {index: position for position, index in enumerate(indices_a)}
    for start, end in ds.chunk_bounds():
        chunk_positions = [positions[index] for index in range(start, end)]
        assert chunk_positions == list(range(min(chunk_positions), max(chunk_positions) + 1))

    within_chunk = list(ChunkShuffleSampler(ds, seed=7, shuffle_rows=True))
    assert sorted(within_chunk) == list(range(len(ds)))


def test_parquet_manifest_counts_rows_without_loading_records(tmp_path: Path):
    root = tmp_path / "shards"
    train = root / "train"
    validation = root / "validation"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    pd.DataFrame([{"record_id": "r0", "text": "a"}]).to_parquet(train / "train-000.parquet")
    pd.DataFrame([{"record_id": "v0", "text": "b"}, {"record_id": "v1", "text": "c"}]).to_parquet(validation / "validation-000.parquet")
    manifest = parquet_manifest(root, ("train", "validation", "test"))
    assert manifest["splits"]["train"]["rows"] == 1
    assert manifest["splits"]["validation"]["rows"] == 2
    assert manifest["splits"]["test"]["files"] == 0


def test_required_data_does_not_fall_back_to_fixture(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        make_dataset(missing, "train", require_data=True)
    fallback = make_dataset(missing, "train", require_data=False, fixture_size=3)
    assert len(fallback) == 3


def test_parameter_golf_bin_loader_emits_graph_structured_dag_records(tmp_path: Path):
    data_root = tmp_path / "pg"
    tok_root = tmp_path / "tokenizers"
    data_root.mkdir()
    tok_root.mkdir()
    _write_parameter_golf_bin(data_root / "fineweb_train_000000.bin", b"abcdef")
    tokenizer_path = tok_root / "fineweb_pure_byte_260.json"
    tokenizer_path.write_text(
        json.dumps({"tokenizer_type": "pure_byte", "config": {"byte_offset": 4}, "vocab_size": 260}),
        encoding="utf-8",
    )

    ds = ParameterGolfBinGraphDataset(data_root, "train", window_tokens=4, tokenizer_path=tokenizer_path)
    record = ds[0]

    assert record.text == "abcd"
    assert (record.metadata or {})["source"] == "parameter_golf_bin"
    assert (record.metadata or {})["decoding_order_kind"] == "causal_dag"
    assert record.graph_json["nodes"][0]["type"] == "token_window"
    assert record.graph_json["edges"]


def test_make_dataset_from_config_mixes_graph_parquet_and_parameter_golf(tmp_path: Path):
    parquet_root = tmp_path / "shards" / "train"
    parquet_root.mkdir(parents=True)
    pd.DataFrame([{"record_id": "r", "text": "abc", "graph_json": '{"nodes":[{"id":"a"}],"edges":[]}'}]).to_parquet(parquet_root / "train-000.parquet")
    pg_root = tmp_path / "pg"
    pg_root.mkdir()
    _write_parameter_golf_bin(pg_root / "fineweb_train_000000.bin", b"abcdef")
    cfg = {
        "seed": 7,
        "seq_len": 4,
        "data_root": str(tmp_path / "shards"),
        "hybrid_data": {
            "enabled": True,
            "sources": [
                {"kind": "parquet", "name": "graph_parquet", "root": str(tmp_path / "shards"), "weight": 1.0, "required": True},
                {"kind": "parameter_golf_bin", "name": "openai_parameter_golf", "root": str(pg_root), "weight": 1.0, "required": True, "window_tokens": 4},
            ],
        },
    }
    ds = make_dataset_from_config(cfg, "train")
    assert len(ds) >= 1
    sources = {(ds[i].metadata or {}).get("hybrid_source") for i in range(min(len(ds), 8))}
    assert sources <= {"graph_parquet", "openai_parameter_golf"}
    assert sources


def test_graph_decoding_order_uses_topological_order_for_dag_and_random_for_cycles():
    dag = {"nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}], "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]}
    assert graph_decoding_order(dag)["decoding_node_order"] == ["a", "b", "c"]
    cyclic = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    order_a = graph_decoding_order(cyclic, seed=11, record_id="x")
    order_b = graph_decoding_order(cyclic, seed=11, record_id="x")
    assert order_a["decoding_order_kind"] == "random_autoregressive"
    assert order_a["decoding_node_order"] == order_b["decoding_node_order"]
    record = GraphRecord.from_mapping(
        {
            "record_id": "dag",
            "text": "fallback",
            "graph_json": {"nodes": [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}], "edges": [{"source": "a", "target": "b"}]},
        }
    )
    assert record.autoregressive_text().splitlines()[0] == "[node] first"


def _write_parameter_golf_bin(path: Path, payload: bytes) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    tokens = np.frombuffer(payload, dtype=np.uint8).astype("<u2") + 4
    header[2] = len(tokens)
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.tobytes())
