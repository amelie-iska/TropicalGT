from pathlib import Path
import pandas as pd
from tropicalgt.data import ParquetGraphDataset


def test_parquet_loader_reads_graph_json(tmp_path: Path):
    root = tmp_path / "shards" / "train"
    root.mkdir(parents=True)
    pd.DataFrame([{"record_id": "r", "text": "abc", "graph_json": '{"nodes":[{"id":"a"}],"edges":[]}'}]).to_parquet(root / "train-000.parquet")
    ds = ParquetGraphDataset(tmp_path / "shards", "train")
    assert len(ds) == 1
    rec = ds[0]
    assert rec.graph_json["nodes"][0]["id"] == "a"
