import json
from pathlib import Path

import torch

from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.scaling import apply_reasoning_action
from tropicalgt.simplicial import build_filtered_simplicial_object
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import (
    _simplicial_object_svg,
    write_analogical_memory_visualization,
    write_got_trajectory_visualization,
    write_persistence_visualizations,
    write_reasoning_visualizations,
)


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
        "sample_supported_local_idw_surface",
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
    assert "pca-radius-filtered-complex" in html
    assert "3D PCA radius-filtered simplicial complex" in html
    assert "pca-radius-node" in html
    assert "data-pca-z" in html
    assert "zero-simplex" in html
    assert "filtration-layer" in html
    assert "sample-supported" in html or "Smoothed NLL surface" in html or "Interpolating NLL surface" in html


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


def test_simplicial_object_svg_uses_3d_pca_radius_filtration():
    obj = {
        "summary": {"num_vertices": 4, "num_edges": 4, "num_two_simplices": 1},
        "thresholds": [0.0, 0.2, 0.4],
        "simplices": [
            {"simplex": ["a"], "dimension": 0, "filtration": 0.0, "type": "reasoning-step", "text": "alpha"},
            {"simplex": ["b"], "dimension": 0, "filtration": 0.1, "type": "reasoning-step", "text": "beta"},
            {"simplex": ["c"], "dimension": 0, "filtration": 0.2, "type": "reasoning-step", "text": "gamma"},
            {"simplex": ["d"], "dimension": 0, "filtration": 0.3, "type": "reasoning-step", "text": "delta"},
            {"simplex": ["a", "b"], "dimension": 1, "filtration": 0.15, "type": "radius-edge"},
            {"simplex": ["b", "c"], "dimension": 1, "filtration": 0.25, "type": "radius-edge"},
            {"simplex": ["c", "d"], "dimension": 1, "filtration": 0.35, "type": "radius-edge"},
            {"simplex": ["a", "c"], "dimension": 1, "filtration": 0.4, "type": "radius-edge"},
            {"simplex": ["a", "b", "c"], "dimension": 2, "filtration": 0.45, "type": "radius-face"},
        ],
    }
    html = _simplicial_object_svg(obj)
    assert "pca-radius-filtered-complex" in html
    assert "3D PCA radius-filtered simplicial complex" in html
    assert html.count("pca-radius-node") == 4
    assert html.count("pca-radius-edge") == 4
    assert html.count("pca-radius-face") == 1
    assert "data-pca-z" in html
    assert "PCA=(" in html
    assert "radius filtration" in html


def test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    obj = build_filtered_simplicial_object(record)
    child_record = apply_reasoning_action(record, "verify", rank=0)
    sibling_record = apply_reasoning_action(record, "expand", rank=1)
    leaf_record = apply_reasoning_action(child_record, "refine", rank=0)
    child_obj = build_filtered_simplicial_object(child_record)
    sibling_obj = build_filtered_simplicial_object(sibling_record)
    leaf_obj = build_filtered_simplicial_object(leaf_record)
    scaling = {
        "candidates": [
            {"record_id": record.record_id, "embedding": [0.0, 0.0, 0.0], "score": 0.1, "nll": 2.0, "level": 0, "path": [], "filtered_simplicial_object": obj},
            {"record_id": child_record.record_id, "parent": record.record_id, "embedding": [1.0, 0.3, 0.2], "score": 0.5, "nll": 1.3, "level": 1, "path": ["verify"], "input_text": "input", "decoded_argmax": "output", "filtered_simplicial_object": child_obj},
            {"record_id": sibling_record.record_id, "parent": record.record_id, "embedding": [0.2, 1.1, -0.4], "score": 0.4, "nll": 1.5, "level": 1, "path": ["expand"], "filtered_simplicial_object": sibling_obj},
            {"record_id": leaf_record.record_id, "parent": child_record.record_id, "embedding": [1.4, -0.8, 0.9], "score": 0.7, "nll": 0.9, "level": 2, "path": ["verify", "refine"], "filtered_simplicial_object": leaf_obj},
        ]
    }
    paths = write_got_trajectory_visualization(scaling, tmp_path)
    html = Path(paths["got_trajectory_3d"]).read_text(encoding="utf-8")
    embedding_map_html = Path(paths["got_embedding_map_3d"]).read_text(encoding="utf-8")
    full_complex_html = Path(paths["got_full_trajectory_complex"]).read_text(encoding="utf-8")
    step_index_html = Path(paths["got_reasoning_step_complex_index"]).read_text(encoding="utf-8")
    step_manifest = json.loads(Path(paths["got_reasoning_step_complex_manifest"]).read_text(encoding="utf-8"))
    payload = json.loads(Path(paths["got_payloads"]).read_text(encoding="utf-8"))
    assert "simplicial-object-panel" in html
    assert "hover-simplicial-card" in html
    assert 'aria-label="hovered filtered simplicial object"' in html
    assert "<svg" in html
    assert "filtration-layer" in html
    assert "Smooth projected NLL" in html
    assert "fitness landscape" in html
    assert "Exact GoT NLL anchor mesh" in html
    assert "NLL surface anchors" in html
    assert payload["nll_surface"]["available"] is True
    assert payload["nll_surface"]["touches_points"] is True
    assert payload["nll_surface"]["surface_kind"] in {"sample_supported_local_idw_surface", "sparse_exact_triangular_nll_mesh", "exact_delaunay_nll_mesh"}
    assert payload["nll_surface"]["z_axis"] == "projected_nll_fitness_energy"
    assert payload["nll_surface"]["surrogate_landscape_layer"]["surface_kind"] == "smooth_projected_nll_fitness_landscape"
    assert payload["nll_surface"]["surrogate_landscape_layer"]["point_count"] >= len(scaling["candidates"])
    assert payload["nll_surface"]["local_interpolating_sheet"]["surface_kind"] == "local_interpolating_nll_sheet"
    assert payload["nll_surface"]["local_interpolating_sheet"]["point_count"] >= len(scaling["candidates"])
    assert payload["nll_surface"]["smooth_landscape_microstep_anchor_count"] > 0
    assert payload["nll_surface"]["max_point_residual"] < 1e-5
    assert payload["nll_progress"]["edge_count"] == 3
    assert payload["nll_progress"]["improving_edge_fraction"] > 0.0
    assert len(payload["edges"]) == 3
    assert sum(1 for edge in payload["edges"] if edge["source"] == record.record_id) == 2
    assert payload["microstep_nodes"]
    assert any(row["type"] == "verification_check" for row in payload["microstep_nodes"])
    assert "reasoning microstep" in html
    assert payload["nodes"][1]["input_text"] == "input"
    assert payload["nodes"][1]["decoded_argmax"] == "output"
    assert payload["nodes"][1]["level"] == 1
    assert payload["nodes"][1]["filtered_simplicial_object"]["summary"]["num_vertices"] >= 1
    assert payload["embedding_pca_diagnostics"]["coordinate_source"] == "model graph_state embeddings"
    assert "Graph-of-thought embedding-space trajectory map" in embedding_map_html
    assert "actual graph_state PCA" in embedding_map_html
    assert "distance corr" in embedding_map_html
    assert "Full graph-of-thought trajectory filtered simplicial complex" in full_complex_html
    assert "Filtration radius" in full_complex_html
    assert "play filtration" in full_complex_html
    assert "Reasoning step filtered simplicial complex maps" in step_index_html
    assert len(step_manifest["steps"]) == 4
    first_step = tmp_path / "reasoning_step_complex_maps" / "reasoning_step_000.html"
    assert first_step.exists()
    first_step_html = first_step.read_text(encoding="utf-8")
    assert "Filtration radius" in first_step_html
    assert "play filtration" in first_step_html


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
    assert "layout=3d_pca_radius_projection" in svg
    assert "method=classical_mds_pcoa" in svg
    assert "stress=" in svg
    assert "pca-radius-filtered-complex" in svg
    assert "data-pca-z" in svg
    assert svg.count("zero-simplex") == 50
    assert svg.count("one-simplex") == 49


def test_trajectory_persistence_uses_growth_and_free_resolution(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    obj = build_filtered_simplicial_object(record)
    topo0 = _toy_topology(intervals=[{"dimension": 0, "birth": 0.0, "death": 0.45, "infinite": False}])
    topo1 = _toy_topology(
        intervals=[
            {"dimension": 0, "birth": 0.0, "death": 0.45, "infinite": False},
            {"dimension": 1, "birth": 0.2, "death": 0.82, "infinite": False},
        ]
    )
    paths = write_persistence_visualizations(
        topo1,
        tmp_path,
        growth=[
            {"level": 0, "filtered_simplicial_object": obj, "topological_algebra": topo0},
            {"level": 1, "filtered_simplicial_object": obj, "topological_algebra": topo1},
        ],
        title_prefix="Trajectory ",
    )
    barcode_html = Path(paths["persistence_barcode"]).read_text(encoding="utf-8")
    module_html = Path(paths["persistence_module_betti"]).read_text(encoding="utf-8")
    reps_html = Path(paths["persistence_representations"]).read_text(encoding="utf-8")
    landscapes_html = Path(paths["persistence_landscapes"]).read_text(encoding="utf-8")
    assert "persistent homology growth barcode" in barcode_html
    assert "trajectory growth level" in barcode_html
    assert "multiparameter persistence and free-resolution growth" in module_html
    assert "free-resolution proxy" in module_html
    assert "simplicial-object-panel" in module_html
    assert "GUDHI persistence vectorization growth" in reps_html
    assert "Fast train" in reps_html
    assert "eval features" in reps_html
    assert "Actual GUDHI persistence landscape functions" in landscapes_html
    assert "lambda_1(t)" in landscapes_html
    assert "not norm-only summaries" in landscapes_html
    assert '<input id="filtration-slider"' not in barcode_html
    assert '<div class="filtration-controls"' not in barcode_html
    assert '<input id="filtration-slider"' not in module_html
    assert '<div class="filtration-controls"' not in module_html


def test_analogical_memory_visualization_renders_simplicial_maps(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    obj = build_filtered_simplicial_object(record)
    topo = _toy_topology(intervals=[{"dimension": 0, "birth": 0.0, "death": None, "infinite": True}])
    memory = {
        "bank_path": "",
        "retrieved": [
            {
                "memory_id": "mem0",
                "record_id": "rec0",
                "retrieval_score": 0.9,
                "embedding_similarity": 0.8,
                "signature_similarity": 0.7,
                "quality_score": 0.6,
                "filtered_simplicial_object": obj,
                "topological_algebra": topo,
                "derived_signature": topo["derived_equivalence_signature"],
            },
            {
                "memory_id": "mem1",
                "record_id": "rec1",
                "retrieval_score": 0.7,
                "embedding_similarity": 0.6,
                "signature_similarity": 0.5,
                "quality_score": 0.4,
                "filtered_simplicial_object": obj,
                "topological_algebra": topo,
                "derived_signature": topo["derived_equivalence_signature"],
            }
        ],
    }
    paths = write_analogical_memory_visualization(
        memory,
        tmp_path,
        query_context={"filtered_simplicial_object": obj, "topological_algebra": topo, "label": "query trajectory"},
    )
    html = Path(paths["analogical_memory_retrieval_html"]).read_text(encoding="utf-8")
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    assert "Analogical reasoning memory retrieval as simplicial maps" in html
    assert "simplicial map candidate" in html
    assert "persistent homology similarity" in html
    assert "free-resolution similarity" in html
    assert "simplicial-object-panel" in html
    assert '<input id="filtration-slider"' in html
    assert '<div class="filtration-controls"' in html
    assert "memory 2" not in html
    assert "analogical_memory_topk_index_html" in paths
    assert "analogical_memory_map_02_html" in paths
    index_html = Path(paths["analogical_memory_topk_index_html"]).read_text(encoding="utf-8")
    rank2_html = Path(paths["analogical_memory_map_02_html"]).read_text(encoding="utf-8")
    assert "top-k retrieval as separate filtered simplicial maps" in index_html
    assert "rank 2" in rank2_html
    assert '<input id="filtration-slider"' in rank2_html
    assert len(maps["maps"]) == 2
    assert maps["maps"][1]["pair_page"].endswith("analogical_memory_map_02.html")
    assert maps["maps"][0]["edge_preservation_rate"] >= 0.0
    assert maps["maps"][0]["derived_signature_similarity"] >= 0.0


def _toy_topology(intervals):
    return {
        "persistence": {"backend": "toy", "available": True, "intervals": intervals},
        "persistence_module": {
            "states": [
                {"threshold": 0.0, "chain_group_ranks": {"0": 1}, "betti": {"0": 1}, "euler_characteristic": 1},
                {"threshold": 1.0, "chain_group_ranks": {"0": 2, "1": 1}, "betti": {"0": 1, "1": 0}, "euler_characteristic": 1},
            ]
        },
        "derived_equivalence_signature": {
            "betti_vector": [1, 0, 0, 0],
            "persistence_finite_interval_count": 0,
            "persistence_infinite_interval_count": 1,
            "persistence_total_finite_length": 0.0,
            "multiparameter_grid_points": 2,
            "multiparameter_h0_rank_sample": [{"h0_rank": 1}],
        },
        "commutative_algebra": {
            "multiparameter_free_resolution_proxy": {
                "ring": "F2[x_filtration,x_dimension,x_position]",
                "free_chain_modules": [
                    {"homological_degree": 0, "rank": 2},
                    {"homological_degree": 1, "rank": 1},
                ],
            }
        },
    }
