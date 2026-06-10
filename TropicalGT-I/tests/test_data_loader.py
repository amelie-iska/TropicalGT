from pathlib import Path
import pandas as pd
import pytest

from tropicalgt.data import ChunkShuffleSampler, ParquetGraphDataset, make_dataset, parquet_manifest


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
