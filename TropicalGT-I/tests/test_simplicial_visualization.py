import json
from pathlib import Path

import torch

from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.simplicial import build_filtered_simplicial_object
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import _simplicial_object_svg, write_got_trajectory_visualization, write_reasoning_visualizations


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
    assert set(payload) == {"hover", "points", "filtered_simplicial_objects", "nll_surface", "model_io"}
    assert len(payload["filtered_simplicial_objects"]) == 2
    assert payload["points"][0]["filtered_summary"]["num_vertices"] >= 1
    assert "input_text" in payload["points"][0]
    assert "decoded_argmax" in payload["points"][0]
    assert payload["nll_surface"]["nll_height"]["touches_points"] is True
    assert payload["nll_surface"]["nll_height"]["surface_kind"] in {
        "smooth_exact_rbf_surface",
        "degenerate_interpolating_polyline",
        "sparse_exact_triangular_nll_mesh",
    }
    assert payload["nll_surface"]["nll_height"]["max_point_residual"] < 1e-5
    html = Path(paths["pca_nll"]).read_text(encoding="utf-8")
    assert "simplicial-object-panel" in html
    assert "hover-simplicial-card" in html
    assert "filtration-slider" in html
    assert "data-filtration" in html
    assert 'color-scheme: dark' in html
    assert "plotly_hover" in html
    assert "renderHoverCard" in html
    assert "<svg" in html
    assert "zero-simplex" in html
    assert "filtration-layer" in html
    assert "Interpolating NLL surface" in html


def test_visualization_payload_contains_topology_when_audited(tmp_path: Path):
    ds = FixtureGraphDataset(2)
    tok = TokenGTTokenizer(feature_dim=48)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    paths = write_reasoning_visualizations(
        model,
        ds,
        tok,
        seq_len=32,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        limit=2,
        audit_level="topology",
        audit_max_simplices=128,
    )
    payload = json.loads(Path(paths["payloads"]).read_text(encoding="utf-8"))
    assert "topological_algebra_diagnostics" in payload
    assert payload["topological_algebra_diagnostics"][0]["topological_algebra"]["multiparameter_persistence"]["num_parameters"] == 3


def test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    obj = build_filtered_simplicial_object(record)
    scaling = {
        "candidates": [
            {"record_id": "root", "embedding": [0.0, 0.0, 0.0], "score": 0.1, "nll": 2.0, "level": 0, "path": [], "filtered_simplicial_object": obj},
            {"record_id": "child", "parent": "root", "embedding": [1.0, 0.3, 0.2], "score": 0.5, "nll": 1.3, "level": 1, "path": ["verify"], "input_text": "input", "decoded_argmax": "output", "filtered_simplicial_object": obj},
            {"record_id": "sibling", "parent": "root", "embedding": [0.8, -0.3, 0.1], "score": 0.4, "nll": 1.5, "level": 1, "path": ["expand"], "filtered_simplicial_object": obj},
            {"record_id": "leaf", "parent": "child", "embedding": [1.4, 0.7, 0.4], "score": 0.7, "nll": 0.9, "level": 2, "path": ["verify", "refine"], "filtered_simplicial_object": obj},
        ]
    }
    paths = write_got_trajectory_visualization(scaling, tmp_path)
    html = Path(paths["got_trajectory_3d"]).read_text(encoding="utf-8")
    payload = json.loads(Path(paths["got_payloads"]).read_text(encoding="utf-8"))
    assert "simplicial-object-panel" in html
    assert "hover-simplicial-card" in html
    assert 'aria-label="hovered filtered simplicial object"' in html
    assert "<svg" in html
    assert "filtration-layer" in html
    assert "Interpolating trajectory NLL surface" in html
    assert payload["nll_surface"]["available"] is True
    assert payload["nll_surface"]["touches_points"] is True
    assert payload["nll_surface"]["surface_kind"] in {"smooth_exact_rbf_surface", "sparse_exact_triangular_nll_mesh"}
    assert payload["nll_surface"]["max_point_residual"] < 1e-5
    assert len(payload["edges"]) == 3
    assert sum(1 for edge in payload["edges"] if edge["source"] == "root") == 2
    assert payload["nodes"][1]["input_text"] == "input"
    assert payload["nodes"][1]["decoded_argmax"] == "output"
    assert payload["nodes"][1]["level"] == 1
    assert payload["nodes"][1]["filtered_simplicial_object"]["summary"]["num_vertices"] >= 1


def test_simplicial_svg_wraps_long_topological_paths():
    simplices = []
    for idx in range(50):
        simplices.append({"simplex": [f"v{idx:02d}"], "dimension": 0, "filtration": idx / 49})
    for idx in range(49):
        simplices.append({"simplex": [f"v{idx:02d}", f"v{idx + 1:02d}"], "dimension": 1, "filtration": (idx + 1) / 49})
    svg = _simplicial_object_svg(
        {
            "summary": {"num_vertices": 50, "num_edges": 49, "num_two_simplices": 0},
            "simplices": simplices,
            "thresholds": [idx / 10 for idx in range(11)],
        }
    )
    assert "layout=wrapped_topological_path" in svg
    assert svg.count("zero-simplex") == 50
    assert svg.count("one-simplex") == 49
