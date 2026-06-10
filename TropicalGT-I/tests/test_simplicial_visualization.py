import json
from pathlib import Path

import torch

from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.simplicial import build_filtered_simplicial_object
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import write_reasoning_visualizations


def test_filtered_simplicial_object_contains_path_faces():
    record = FixtureGraphDataset(1)[0]
    obj = build_filtered_simplicial_object(record)
    assert obj["record_id"] == record.record_id
    assert obj["summary"]["num_vertices"] >= 2
    assert obj["summary"]["num_edges"] >= 1
    assert "simplices" in obj
    assert all("filtration" in simplex for simplex in obj["simplices"])


def test_visualization_payload_contains_filtered_objects(tmp_path: Path):
    ds = FixtureGraphDataset(2)
    tok = TokenGTTokenizer(feature_dim=48)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    paths = write_reasoning_visualizations(model, ds, tok, seq_len=32, device=torch.device("cpu"), output_dir=tmp_path, limit=2)
    payload = json.loads(Path(paths["payloads"]).read_text(encoding="utf-8"))
    assert set(payload) == {"hover", "points", "filtered_simplicial_objects"}
    assert len(payload["filtered_simplicial_objects"]) == 2
    assert payload["points"][0]["filtered_summary"]["num_vertices"] >= 1
