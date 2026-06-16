import json
from pathlib import Path

import plotly.graph_objects as go
import torch

from tropicalgt import cas_toric
from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.records import GraphRecord
from tropicalgt.scaling import _has_real_probability_complex, apply_reasoning_action
from tropicalgt.memory import probability_simplicial_map_diagnostics
from tropicalgt.simplicial import build_embedding_radius_simplicial_object, build_filtered_simplicial_object, build_reasoning_trajectory_complex
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import (
    _attach_graph_token_direction_overlay,
    _cas_real_resolution_display,
    _complex_slider_frame_contract,
    _complex_slider_traces,
    _derived_invariant_comparison,
    _gudhi_canonical_complex,
    _has_real_probability_filtration,
    _write_simplex_tree_3d_map,
    _m2_be_diagnostic_columns,
    _m2_certificate_columns,
    _m2_ideal_diagnostic_columns,
    _simplicial_object_svg,
    _simplicial_map_between_complexes,
    _simplicial_panel_items,
    write_analogical_memory_visualization,
    write_graphcg_trajectory_visualization,
    write_got_trajectory_visualization,
    write_inference_audit_artifacts,
    write_persistence_visualizations,
    write_reasoning_visualizations,
    write_tropical_fan_diagnostics,
    write_toric_embedding_sidecar,
    write_tropical_support_heatmap,
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
    assert set(payload) == {
        "hover",
        "points",
        "filtered_simplicial_objects",
        "embedding_filtered_simplicial_objects",
        "nll_surface",
        "model_io",
        "reasoning_visualization_diagnostics",
    }
    assert len(payload["filtered_simplicial_objects"]) == 2
    assert payload["filtered_simplicial_objects"][0]["summary"]["filtration_model"] == "model_tropical_support_probability_jensen_shannon_vietoris_rips_2_skeleton"
    assert payload["filtered_simplicial_objects"][0]["summary"]["probability_transform"] == {
        "kind": "model_supplied",
        "source": "model_tropical_support_probabilities",
        "temperature": None,
    }
    assert payload["filtered_simplicial_objects"][0]["simplex_tree"]["backend"] == "gudhi.SimplexTree"
    assert payload["filtered_simplicial_objects"][0]["summary"]["simplex_tree_available"] is True
    probability_vertices = [row for row in payload["filtered_simplicial_objects"][0]["simplices"] if row["dimension"] == 0]
    assert probability_vertices and probability_vertices[0]["probability_source"] == "model_tropical_support_probabilities"
    assert probability_vertices and probability_vertices[0]["gudhi_simplex_tree"] is True
    assert payload["embedding_filtered_simplicial_objects"][0]["summary"]["filtration_model"] == "model_graph_token_embedding_vietoris_rips_2_skeleton"
    assert payload["embedding_filtered_simplicial_objects"][0]["summary"]["embedding_source"] == "TropicalGTModel.graph_token_embeddings"
    assert payload["embedding_filtered_simplicial_objects"][0]["summary"]["probability_transform"] is None
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
    assert "if (panelSvg) panelSvg.innerHTML" in html
    assert 'aria-label="interactive selected filtered simplicial complex"' in html
    assert '<details class="static-preview">' not in html
    assert "Static SVG fallback preview" not in html
    assert '<details class="static-preview" open>' not in html
    assert "<svg" in html
    assert "pca-radius-filtered-complex" in html
    assert "3D PCA radius-filtered simplicial complex" in html
    assert "pca-radius-node" in html
    assert "data-pca-z" in html
    assert "zero-simplex" in html
    assert "filtration-layer" in html
    assert "sample-supported" in html or "Smoothed NLL surface" in html or "Interpolating NLL surface" in html


def test_reasoning_visualization_does_not_duplicate_single_state(tmp_path: Path):
    ds = FixtureGraphDataset(1)
    tok = TokenGTTokenizer(feature_dim=48)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    paths = write_reasoning_visualizations(model, ds, tok, seq_len=32, device=torch.device("cpu"), output_dir=tmp_path, limit=1)
    payload = json.loads(Path(paths["payloads"]).read_text(encoding="utf-8"))

    assert len(payload["points"]) == 1
    assert len(payload["filtered_simplicial_objects"]) == 1
    assert len(payload["embedding_filtered_simplicial_objects"]) == 1
    diagnostics = payload["reasoning_visualization_diagnostics"]
    assert diagnostics["source_state_count"] == 1
    assert diagnostics["contrived_duplicate_for_pca"] is False
    assert diagnostics["single_state_degenerate_pca"] is True
    assert "no synthetic duplicate points" in diagnostics["pca_point_policy"]

    html = Path(paths["pca_nll"]).read_text(encoding="utf-8")
    assert "single-state degenerate PCA" in html


def test_missing_model_probabilities_and_embeddings_are_unavailable_diagnostics():
    record = FixtureGraphDataset(1)[0]
    descriptors = [
        {"index": 0, "kind": "node", "node_id": "a", "text": "alpha"},
        {"index": 1, "kind": "node", "node_id": "b", "text": "beta"},
    ]
    embeddings = [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]]
    missing_probabilities = build_embedding_radius_simplicial_object(
        record,
        descriptors,
        embeddings,
        token_probabilities=None,
        metric="jensen_shannon",
    )
    assert missing_probabilities["available"] is False
    assert missing_probabilities["summary"]["filtration_model"] == "unavailable_no_model_probabilities"
    assert missing_probabilities["simplices"] == []

    missing_embeddings = build_embedding_radius_simplicial_object(record, descriptors, [], metric="euclidean")
    assert missing_embeddings["available"] is False
    assert missing_embeddings["summary"]["filtration_model"] == "unavailable_no_model_embeddings"
    assert missing_embeddings["simplices"] == []


def test_real_probability_filtration_requires_vertex_probability_vectors():
    labelled_but_empty = {
        "available": True,
        "summary": {
            "filtration_model": "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton",
            "radius_filtration": True,
            "num_edges": 1,
        },
        "simplices": [
            {"simplex": ["q0"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["q1"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["q0", "q1"], "dimension": 1, "filtration": 0.25},
        ],
    }
    with_probabilities = {
        **labelled_but_empty,
        "simplices": [
            {"simplex": ["q0"], "dimension": 0, "filtration": 0.0, "probability": [0.7, 0.3]},
            {"simplex": ["q1"], "dimension": 0, "filtration": 0.0, "probability": [0.2, 0.8]},
            {"simplex": ["q0", "q1"], "dimension": 1, "filtration": 0.25},
        ],
    }

    assert _has_real_probability_filtration(labelled_but_empty) is False
    assert _has_real_probability_filtration(with_probabilities) is True


def test_analogical_simplicial_map_uses_model_probability_vectors():
    query = {
        "available": True,
        "summary": {"filtration_model": "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton"},
        "simplex_tree": {"backend": "gudhi.SimplexTree"},
        "simplices": [
            {"simplex": ["q0"], "dimension": 0, "filtration": 0.0, "model_probability_vector": [0.8, 0.1, 0.1]},
            {"simplex": ["q1"], "dimension": 0, "filtration": 0.0, "model_probability_vector": [0.1, 0.8, 0.1]},
            {"simplex": ["q0", "q1"], "dimension": 1, "filtration": 0.2},
        ],
    }
    memory = {
        "available": True,
        "summary": {"filtration_model": "model_tropical_support_probability_jensen_shannon_vietoris_rips_2_skeleton"},
        "simplex_tree": {"backend": "gudhi.SimplexTree"},
        "simplices": [
            {"simplex": ["m0"], "dimension": 0, "filtration": 0.0, "probability_vector": [0.78, 0.12, 0.10]},
            {"simplex": ["m1"], "dimension": 0, "filtration": 0.0, "probability_vector": [0.12, 0.78, 0.10]},
            {"simplex": ["m0", "m1"], "dimension": 1, "filtration": 0.1},
        ],
    }
    report = _simplicial_map_between_complexes(query, memory)
    assert report["map_source"] == "model_probability_jensen_shannon_assignment"
    assert len(report["vertex_map"]) == 2
    assert report["jensen_shannon_distance_summary"]["count"] == 2
    assert report["assignment_cost_summary"]["count"] == 2
    assert report["preserved_edge_pairs"]
    assert report["chain_map_diagnostics"]["available"] is True
    assert report["chain_map_diagnostics"]["boundary_commutation_certified"] is True
    assert report["persistence_module_morphism_diagnostics"]["available"] is True
    assert report["persistence_module_morphism_diagnostics"]["ring"] == "F2[x_level,x_radius]"


def test_analogical_simplicial_map_preserves_plain_probability_after_gudhi_canonicalization():
    query = {
        "available": True,
        "summary": {"filtration_model": "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton"},
        "simplices": [
            {
                "simplex": ["inference"],
                "dimension": 0,
                "filtration": 0.0,
                "probability": [0.62, 0.37, 0.01],
                "probability_source": "TropicalGTModel.gfn(graph_state).softmax_action_probability_vector",
            },
            {
                "simplex": ["inference|expand0"],
                "dimension": 0,
                "filtration": 0.0,
                "probability": [0.22, 0.73, 0.05],
                "probability_source": "TropicalGTModel.gfn(graph_state).softmax_action_probability_vector",
            },
            {"simplex": ["inference", "inference|expand0"], "dimension": 1, "filtration": 0.2},
        ],
    }
    memory = {
        "available": True,
        "summary": {"filtration_model": "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton"},
        "simplices": [
            {
                "simplex": ["memory"],
                "dimension": 0,
                "filtration": 0.0,
                "probability": [0.61, 0.38, 0.01],
                "probability_source": "TropicalGTModel.gfn(graph_state).softmax_action_probability_vector",
            },
            {
                "simplex": ["memory|expand0"],
                "dimension": 0,
                "filtration": 0.0,
                "probability": [0.21, 0.74, 0.05],
                "probability_source": "TropicalGTModel.gfn(graph_state).softmax_action_probability_vector",
            },
            {"simplex": ["memory", "memory|expand0"], "dimension": 1, "filtration": 0.2},
        ],
    }

    report = _simplicial_map_between_complexes(
        _gudhi_canonical_complex(query),
        _gudhi_canonical_complex(memory),
    )

    assert report["map_source"] == "model_probability_jensen_shannon_assignment"
    assert report["jensen_shannon_distance_summary"]["count"] == 2
    assert report["assignment_cost_summary"]["count"] == 2
    assert report["vertex_map"][0]["query_probability_source"] == "TropicalGTModel.gfn(graph_state).softmax_action_probability_vector"
    assert report["simplicial_map_certificate"]["source"] == "finite_filtered_complex_check"


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


def test_reasoning_trajectory_radius_complex_starts_with_disjoint_vertex_simplex_tree():
    candidate = {
        "record_id": "root",
        "embedding": [0.0, 0.0, 0.0],
        "action_probability_vector": [0.8, 0.2],
        "level": 0,
        "score": 0.1,
        "nll": 1.0,
        "path": [],
    }
    embedding_complex = build_reasoning_trajectory_complex([candidate])
    assert embedding_complex["available"] is True
    assert embedding_complex["summary"]["radius_filtration"] is True
    assert embedding_complex["summary"]["single_vertex_radius_filtration"] is True
    assert embedding_complex["summary"]["num_vertices"] == 1
    assert embedding_complex["summary"]["num_edges"] == 0
    assert embedding_complex["thresholds"] == [0.0]
    assert embedding_complex["simplex_tree"]["backend"] == "gudhi.SimplexTree"
    assert embedding_complex["simplex_tree"]["available"] is True
    assert embedding_complex["simplex_tree"]["num_vertices"] == 1

    probability_complex = build_reasoning_trajectory_complex([candidate], metric="jensen_shannon")
    assert probability_complex["available"] is True
    assert probability_complex["summary"]["radius_filtration"] is True
    assert probability_complex["summary"]["embedding_metric"] == "jensen_shannon"
    assert probability_complex["summary"]["single_vertex_radius_filtration"] is True
    assert probability_complex["summary"]["probability_transform"]["kind"] == "model_candidate_probability_vector"
    assert _has_real_probability_filtration(probability_complex) is True
    assert _has_real_probability_complex(probability_complex) is True

    missing_probability = build_reasoning_trajectory_complex(
        [{**candidate, "record_id": "missing-prob", "action_probability_vector": []}],
        metric="jensen_shannon",
    )
    assert missing_probability["available"] is False
    assert missing_probability["reason"] == "unavailable_missing_jensen_shannon_radius_vertices"
    assert missing_probability["summary"]["radius_filtration"] is False
    assert _has_real_probability_filtration(missing_probability) is False


def test_gudhi_canonical_complex_marks_simplex_tree_available():
    obj = {
        "summary": {
            "filtration_model": "embedding_vietoris_rips_2_skeleton",
            "radius_filtration": True,
        },
        "simplices": [
            {"simplex": ["a"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["b"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["a", "b"], "dimension": 1, "filtration": 0.25},
        ],
    }
    canonical = _gudhi_canonical_complex(obj)
    assert canonical["simplex_tree"]["backend"] == "gudhi.SimplexTree"
    assert canonical["simplex_tree"]["available"] is True
    assert canonical["summary"]["simplex_tree_backend"] == "gudhi.SimplexTree"
    assert canonical["summary"]["simplex_tree_available"] is True
    assert canonical["summary"]["simplex_tree_closure_inserted_simplices"] == 0
    assert canonical["simplex_tree"]["closure_inserted_simplices"] == 0
    assert canonical["summary"]["num_vertices"] == 2
    assert canonical["summary"]["num_edges"] == 1


def test_gudhi_canonical_complex_discloses_closure_inserted_faces():
    obj = {
        "summary": {
            "filtration_model": "embedding_vietoris_rips_2_skeleton",
            "radius_filtration": True,
        },
        "simplices": [
            {"simplex": ["a"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["b"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["c"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["a", "b", "c"], "dimension": 2, "filtration": 0.4},
        ],
    }
    canonical = _gudhi_canonical_complex(obj)
    inserted = [row for row in canonical["simplices"] if row.get("gudhi_closure_inserted")]
    assert canonical["summary"]["simplex_tree_closure_inserted_simplices"] == 3
    assert canonical["simplex_tree"]["closure_inserted_simplices"] == 3
    assert len(inserted) == 3
    assert {tuple(row["simplex"]) for row in inserted} == {("a", "b"), ("a", "c"), ("b", "c")}
    assert all(row["filtration_source"] == "gudhi_simplex_tree_closure" for row in inserted)


def test_simplex_tree_page_is_unavailable_without_gudhi_not_raw_json(monkeypatch, tmp_path: Path):
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gudhi":
            raise ImportError("forced missing gudhi for no-fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    obj = {
        "summary": {"filtration_model": "embedding_vietoris_rips_2_skeleton", "radius_filtration": True},
        "simplices": [
            {"simplex": ["a"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["b"], "dimension": 0, "filtration": 0.0},
            {"simplex": ["a", "b"], "dimension": 1, "filtration": 0.25},
        ],
    }

    canonical = _gudhi_canonical_complex(obj)
    assert canonical["simplex_tree"]["available"] is False
    assert canonical["simplex_tree"]["backend"] == "unavailable_gudhi_simplex_tree"
    assert canonical["simplex_tree"]["no_proxy_or_fallback"] is True
    assert canonical["simplex_tree"]["safe_to_render_simplex_tree"] is False
    assert canonical["summary"]["simplex_tree_available"] is False
    assert canonical["summary"]["simplex_tree_no_proxy_or_fallback"] is True

    out = tmp_path / "tree.html"
    _write_simplex_tree_3d_map(out, obj, "forced missing GUDHI tree")
    html = out.read_text(encoding="utf-8")
    assert "unavailable_gudhi_simplex_tree" in html
    assert "No simplex-tree/trie or face-coface poset is rendered without a real GUDHI SimplexTree" in html
    assert "actual face-to-coface covers" not in html
    assert "optional sorted-label trie prefix links" not in html


def test_simplex_tree_poset_contract_records_actual_face_coface_covers(tmp_path: Path):
    obj = {
        "summary": {
            "filtration_model": "embedding_vietoris_rips_2_skeleton",
            "radius_filtration": True,
        },
        "simplices": [
            {"simplex": ["a"], "dimension": 0, "filtration": 0.0, "type": "vertex"},
            {"simplex": ["b"], "dimension": 0, "filtration": 0.0, "type": "vertex"},
            {"simplex": ["c"], "dimension": 0, "filtration": 0.0, "type": "vertex"},
            {"simplex": ["a", "b"], "dimension": 1, "filtration": 0.25, "type": "radius_edge"},
            {"simplex": ["a", "c"], "dimension": 1, "filtration": 0.5, "type": "radius_edge"},
            {"simplex": ["b", "c"], "dimension": 1, "filtration": 0.75, "type": "radius_edge"},
            {"simplex": ["a", "b", "c"], "dimension": 2, "filtration": 1.0, "type": "radius_face"},
        ],
    }
    out = tmp_path / "simplex_tree_poset.html"
    _write_simplex_tree_3d_map(out, obj, "contracted simplex tree")
    html = out.read_text(encoding="utf-8")
    compact = html.replace(" ", "")
    assert "simplex_tree_poset_contract" in html
    assert '"schema_version":"tropicalgt.simplex_tree_poset.v1"' in compact
    assert '"layout":"model_embedding_barycentric_face_coface_poset"' in compact
    assert '"not_disconnected_simplex_columns":true' in compact
    assert '"empty_simplex_root_present":true' in compact
    assert '"displayed_simplex_count":7' in compact
    assert '"actual_face_to_coface_cover_edges":12' in compact
    assert '"empty_root_vertex_cover_edges":3' in compact
    assert '"optional_sorted_label_trie_prefix_edges":7' in compact
    assert '"primary_edges":"actual_face_to_coface_covers"' in compact
    assert '"optional_prefix_links_visible":"legendonly"' in compact
    assert "actual face-to-coface covers (12)" in html
    assert "optional sorted-label trie prefix links (7)" in html
    assert "not disconnected simplex columns" in html


def test_complex_slider_contract_starts_with_disjoint_vertices_and_monotone_growth():
    obj = {
        "summary": {
            "filtration_model": "model_graph_token_embedding_vietoris_rips_2_skeleton",
            "radius_filtration": True,
        },
        "thresholds": [0.0, 0.5, 1.0],
        "simplices": [
            {"simplex": ["a"], "dimension": 0, "filtration": 0.0, "type": "state"},
            {"simplex": ["b"], "dimension": 0, "filtration": 0.0, "type": "state"},
            {"simplex": ["c"], "dimension": 0, "filtration": 0.0, "type": "state"},
            {"simplex": ["a", "b"], "dimension": 1, "filtration": 0.0, "type": "zero_radius_edge"},
            {"simplex": ["b", "c"], "dimension": 1, "filtration": 0.5, "type": "radius_edge"},
            {"simplex": ["a", "b", "c"], "dimension": 2, "filtration": 1.0, "type": "radius_face"},
        ],
        "trajectory_overlay": {
            "edges": [{"source": "a", "target": "b", "filtration": 0.0, "action": "expand"}],
        },
        "graph_token_direction_overlay": {
            "edges": [{"source": "b", "target": "c", "filtration": 0.0, "role": "source-node-to-edge-token"}],
        },
        "decoding_causal_overlay": {
            "edges": [{"source": "a", "target": "c", "filtration": 0.0, "role": "forward_decoding_order"}],
        },
    }
    coords = {"a": (0.0, 0.0, 0.0), "b": (1.0, 0.0, 0.0), "c": (0.0, 1.0, 0.0)}

    contract = _complex_slider_frame_contract(obj, thresholds=[0.0, 0.5, 1.0])
    assert contract["radius_filtration"] is True
    assert contract["first_frame_disjoint_vertices_only"] is True
    assert contract["monotone_visible_counts"] is True
    first, middle, final = contract["frames"]
    assert first == {
        "threshold": 0.0,
        "initial_radius_frame": True,
        "vertices": 3,
        "solid_edges": 0,
        "filled_faces": 0,
        "dotted_trajectory_overlays": 0,
        "dotted_direction_overlays": 0,
        "dotted_decoding_overlays": 0,
        "dotted_overlays": 0,
    }
    assert middle["solid_edges"] == 2
    assert final["filled_faces"] == 1
    assert final["dotted_overlays"] == 3

    def non_null_count(trace) -> int:
        return sum(1 for value in list(trace.x) if value is not None)

    first_traces = {trace.name: trace for trace in _complex_slider_traces(obj, coords, threshold=0.0)}
    assert non_null_count(first_traces["solid radius/simplicial edges induced from the same embeddings"]) == 0
    assert non_null_count(first_traces["faint GoT parent-child trajectory overlay"]) == 0
    assert non_null_count(first_traces["faint directed graph-token overlay"]) == 0
    assert non_null_count(first_traces["dotted causal/decoding order overlay"]) == 0
    assert non_null_count(first_traces["0-simplices"]) == 3

    final_traces = {trace.name: trace for trace in _complex_slider_traces(obj, coords, threshold=1.0)}
    assert non_null_count(final_traces["solid radius/simplicial edges induced from the same embeddings"]) == 4
    assert non_null_count(final_traces["faint GoT parent-child trajectory overlay"]) == 2
    assert non_null_count(final_traces["faint directed graph-token overlay"]) == 2
    assert non_null_count(final_traces["dotted causal/decoding order overlay"]) > 2
    mesh = final_traces["filled 2-simplices gated by radius slider"]
    assert len(mesh.i) == 1


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


def test_decoding_causal_overlay_uses_graph_decoding_order_report():
    from tropicalgt.scaling import _decoding_order_report
    from tropicalgt.visualization import _attach_decoding_causal_overlay, _simplicial_plot_payload

    record = FixtureGraphDataset(1)[0]
    obj = build_filtered_simplicial_object(record)
    report = _decoding_order_report(record)
    assert report["decoding_node_order"]

    updated = _attach_decoding_causal_overlay(obj, {"level": 0, "decoding_order_report": report})
    overlay = updated["decoding_causal_overlay"]
    assert str(overlay["source"]).startswith("GraphRecord.metadata")
    assert overlay["edge_count"] > 0
    assert all(edge["style"] == "dotted" for edge in overlay["edges"])
    assert {edge["role"] for edge in overlay["edges"]} & {"forward_decoding_order", "reverse_decoding_order", "causal_graph_edge"}

    payload = _simplicial_plot_payload(updated)
    assert payload["directed_edge_count"] == overlay["edge_count"]
    assert all(edge["style"] == "dotted" for edge in payload["directed_edges"])


def test_decoding_causal_overlay_dots_roar_edges_for_cyclic_graph():
    from tropicalgt.scaling import _decoding_order_report
    from tropicalgt.visualization import _attach_decoding_causal_overlay, _simplicial_plot_payload

    record = GraphRecord.from_mapping(
        {
            "record_id": "cyclic-overlay",
            "text": "cycle",
            "graph_json": {
                "nodes": [
                    {"id": "a", "text": "alpha"},
                    {"id": "b", "text": "beta"},
                    {"id": "c", "text": "gamma"},
                ],
                "edges": [
                    {"source": "a", "target": "b", "type": "depends_on"},
                    {"source": "b", "target": "a", "type": "depends_on"},
                ],
            },
        }
    )
    obj = build_filtered_simplicial_object(record)
    report = _decoding_order_report(record)
    assert report["decoding_order_kind"] == "random_autoregressive"
    assert report["decoding_reverse_order_kind"] == "reverse_random_autoregressive"

    updated = _attach_decoding_causal_overlay(obj, {"level": 0, "decoding_order_report": report})
    overlay = updated["decoding_causal_overlay"]
    roles = {edge["role"] for edge in overlay["edges"]}
    assert {"forward_decoding_order", "reverse_decoding_order"} <= roles
    assert all(edge["style"] == "dotted" for edge in overlay["edges"])

    payload = _simplicial_plot_payload(updated)
    assert all(edge["style"] == "dotted" for edge in payload["directed_edges"])
    assert {edge["role"] for edge in payload["directed_edges"]} >= {"forward_decoding_order", "reverse_decoding_order"}


def test_graph_token_direction_overlay_uses_model_trace_edges():
    obj = {
        "summary": {"num_vertices": 4, "num_edges": 0, "num_two_simplices": 0},
        "thresholds": [0.0],
        "simplices": [
            {"simplex": ["0:graph:graph"], "dimension": 0, "filtration": 0.0, "token_index": 0, "token_kind": "graph"},
            {
                "simplex": ["1:node:problem"],
                "dimension": 0,
                "filtration": 0.0,
                "token_index": 1,
                "token_kind": "node",
                "node_id": "problem-0",
            },
            {
                "simplex": ["2:node:answer"],
                "dimension": 0,
                "filtration": 0.0,
                "token_index": 2,
                "token_kind": "node",
                "node_id": "answer-1",
            },
            {
                "simplex": ["3:edge:problem-0->answer-1"],
                "dimension": 0,
                "filtration": 0.0,
                "token_index": 3,
                "token_kind": "edge",
                "source": "problem-0",
                "target": "answer-1",
            },
        ],
    }
    row = {
        "graph_token_trace": {
            "tokens": [
                {"index": 0, "kind": "graph", "label": "graph"},
                {"index": 1, "kind": "node", "node_id": "problem-0", "label": "problem"},
                {"index": 2, "kind": "node", "node_id": "answer-1", "label": "answer"},
                {
                    "index": 3,
                    "kind": "edge",
                    "edge_type": "supports_answer",
                    "label": "problem-0->answer-1",
                    "source": "problem-0",
                    "target": "answer-1",
                    "margin": 0.42,
                },
            ]
        }
    }
    overlay = _attach_graph_token_direction_overlay(obj, row)["graph_token_direction_overlay"]
    assert overlay["source"] == "graph_token_trace_directed_edges"
    assert overlay["edge_count"] == 2
    assert overlay["edges"][0]["source"] == "1:node:problem"
    assert overlay["edges"][0]["target"] == "3:edge:problem-0->answer-1"
    assert overlay["edges"][0]["role"] == "source-node-to-edge-token"
    assert overlay["edges"][1]["source"] == "3:edge:problem-0->answer-1"
    assert overlay["edges"][1]["target"] == "2:node:answer"
    assert overlay["edges"][1]["role"] == "edge-token-to-target-node"
    assert overlay["edges"][0]["margin"] == 0.42


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
    full_tree_html = Path(paths["got_full_trajectory_simplex_tree_3d"]).read_text(encoding="utf-8")
    probability_tree_html = Path(paths["got_full_trajectory_simplex_tree_3d_jensen_shannon"]).read_text(encoding="utf-8")
    density_cloud_html = Path(paths["got_nll_density_cloud_pca_3d"]).read_text(encoding="utf-8")
    step_index_html = Path(paths["got_reasoning_step_complex_index"]).read_text(encoding="utf-8")
    step_manifest = json.loads(Path(paths["got_reasoning_step_complex_manifest"]).read_text(encoding="utf-8"))
    payload = json.loads(Path(paths["got_payloads"]).read_text(encoding="utf-8"))
    full_complex_payload = json.loads(Path(paths["got_full_trajectory_complex_payload"]).read_text(encoding="utf-8"))
    density_cloud_payload = json.loads(Path(paths["got_nll_density_cloud_payload"]).read_text(encoding="utf-8"))
    assert "simplicial-object-panel" in html
    assert "hover-simplicial-card" in html
    assert 'aria-label="hovered filtered simplicial object"' in html
    assert "<svg" in html
    assert "filtration-layer" in html
    assert payload["nll_surface"]["available"] is True
    assert payload["nll_surface"]["touches_points"] is True
    assert payload["nll_surface"].get("actual_landscape_layer") is False
    assert payload["nll_surface"].get("sparse_observed_anchor_layer") is True
    assert payload["nll_surface"].get("dense_model_evaluated_field") is False
    assert payload["nll_surface"]["surface_kind"] in {"sparse_exact_triangular_nll_mesh", "sparse_observed_state_nll_anchor_mesh"}
    assert payload["nll_surface"]["z_axis"] == "projected_nll_fitness_energy"
    assert payload["nll_surface"]["provenance"] == "computed only from observed model-evaluated GoT state embeddings and their measured raw NLL values"
    assert payload["nll_surface"]["surface_contact_contract"].startswith("disabled for the main trajectory page")
    assert payload["nll_surface"]["trajectory_point_surface_residual_max"] == 0.0
    projected_by_id = payload["nll_surface"]["surface_projected_z_by_record_id"]
    for idx, node in enumerate(payload["nodes"]):
        rid = node["record_id"]
        assert node["plot"]["touches_nll_surface"] is False
        assert node["plot"]["z_surface"] is None
        assert rid in projected_by_id
        assert abs(node["plot"]["z_centered_scaled_nll"] - projected_by_id[rid]) < 1e-9
        assert node["plot"]["z"] == node["pca"]["pc3"]
        assert "raw_centered_scaled_nll" in node["plot"]
        assert node["reasoning_step_index"] == idx
        assert node["step_complex_href"] == f"reasoning_step_complex_maps/reasoning_step_{idx:03d}.html"
        assert node["step_simplex_tree_href"] == f"reasoning_step_complex_maps/reasoning_step_{idx:03d}_simplex_tree.html"
        assert node["step_complex_contract"].startswith("this GoT state maps")
    surrogate_layer = payload["nll_surface"].get("surrogate_landscape_layer", {})
    if surrogate_layer.get("available") is True:
        assert surrogate_layer["surface_kind"] == "smooth_projected_nll_fitness_landscape"
        assert surrogate_layer["point_count"] >= len(scaling["candidates"])
    local_sheet = payload["nll_surface"].get("local_interpolating_sheet", {})
    assert local_sheet.get("available") is False
    assert local_sheet.get("reason") == "disabled_to_preserve_exact_reasoning_point_surface_contact"
    assert payload["nll_surface"]["max_point_residual"] < 1e-5
    assert "3D PCA NLL density cloud around actual GoT embeddings" in density_cloud_html
    assert "not a model state" in density_cloud_html
    assert "nll_density_render_contract" in html
    assert "nll_density_render_contract" in density_cloud_html
    assert "audit samples from the Gaussian NLL field (hidden by default)" in html
    assert "audit samples from the Gaussian NLL field (hidden by default)" in density_cloud_html
    assert density_cloud_payload["available"] is True
    density_cloud = density_cloud_payload["density_cloud"]
    assert density_cloud["available"] is True
    assert density_cloud["source"] == "actual model-evaluated graph_state PCA anchors and measured raw NLL values"
    assert density_cloud["support_samples_are_not_model_states"] is True
    assert density_cloud["exact_anchor_layer"] is True
    assert density_cloud["anchor_count"] == len(scaling["candidates"])
    assert density_cloud["local_nll_rule"].startswith("kernel-weighted mean")
    assert density_cloud["density_volume"]["support_samples_are_not_model_states"] is True
    main_density_cloud = payload["nll_density_cloud"]
    assert main_density_cloud["support_sample_trace_visibility"] == "legendonly"
    assert main_density_cloud["sample_points_are_model_states"] is False
    assert main_density_cloud["support_samples_hidden_as_model_states"] is True
    assert main_density_cloud["visual_layer_contract"]["schema_version"] == "tropicalgt.nll_density_render.v1"
    assert main_density_cloud["visual_layer_contract"]["support_sample_trace_visibility"] == "legendonly"
    assert main_density_cloud["visual_layer_contract"]["support_samples_are_model_states"] is False
    assert "density_volume" in main_density_cloud["visual_layer_contract"]["visible_density_layers"]
    assert density_cloud_payload["render_contract"] == density_cloud["render_contract"]
    assert density_cloud_payload["visual_layer_contract"]["schema_version"] == "tropicalgt.nll_density_render.v1"
    assert density_cloud_payload["visual_layer_contract"]["page"] == "standalone_density_cloud"
    assert density_cloud_payload["visual_layer_contract"]["support_sample_trace_visibility"] == "legendonly"
    assert density_cloud_payload["visual_layer_contract"]["support_samples_are_model_states"] is False
    assert density_cloud_payload["visual_layer_contract"]["actual_model_anchor_count"] == len(scaling["candidates"])
    assert density_cloud_payload["density_contract"]["actual_model_anchor_layer"] is True
    assert density_cloud_payload["density_contract"]["support_sample_trace_visibility"] == "legendonly"
    assert density_cloud_payload["density_contract"]["z_axis_policy"].startswith("z is PC3")
    assert density_cloud_payload["density_contract"]["sample_points_are_model_states"] is False
    assert density_cloud_payload["support_samples_hidden_as_model_states"] is True
    assert density_cloud_payload["sample_points_are_model_states"] is False
    assert density_cloud_payload["anchor_count"] == len(scaling["candidates"])
    assert density_cloud_payload["actual_model_anchor_count"] == len(scaling["candidates"])
    assert density_cloud_payload["support_sample_count"] == density_cloud["sample_count"]
    assert density_cloud_payload["kernel_bandwidth"] == density_cloud["sigma"]
    assert density_cloud_payload["nll_range"]["span"] > 0.0
    assert density_cloud_payload["local_nll_summary"]["count"] == density_cloud_payload["support_sample_count"]
    assert density_cloud_payload["support_samples"]["visible_by_default"] is False
    assert density_cloud_payload["support_samples"]["visible_as_model_states"] is False
    assert density_cloud_payload["density_volume"]["support_samples_are_not_model_states"] is True
    assert len(density_cloud_payload["anchors"]) == len(scaling["candidates"])
    assert len(density_cloud_payload["nodes"]) == len(scaling["candidates"])
    assert len(density_cloud_payload["edges"]) == 3
    assert all("nll_delta" in edge for edge in density_cloud_payload["edges"])
    assert all("source_nll" in edge and "target_nll" in edge for edge in density_cloud_payload["edges"])
    assert density_cloud_payload["edge_nll_delta_summary"]["count"] == 3
    assert density_cloud_payload["terminal_nll_progress"]["best_terminal_improvement_from_root"] > 0.0
    assert payload["nll_progress"]["edge_count"] == 3
    assert payload["nll_progress"]["improving_edge_fraction"] > 0.0
    assert len(payload["edges"]) == 3
    assert sum(1 for edge in payload["edges"] if edge["source"] == record.record_id) == 2
    assert payload["microstep_nodes"] == []
    assert payload["nll_surface"]["rendered_microsteps_policy"].startswith("disabled")
    assert "reasoning microstep" not in html
    assert "projected surface z=%{z:.4f}" in html
    assert "centered scaled NLL=%{z:.4f}" not in html
    assert payload["nll_surface"]["z_axis_label"].startswith("observed-state NLL anchor z; raw centered NLL")
    assert "open interactive reasoning-step complex page" in html
    assert "if (panelSvg) panelSvg.innerHTML" in html
    assert 'aria-label="interactive selected filtered simplicial complex"' in html
    assert '<details class="static-preview">' not in html
    assert "Static SVG fallback preview" not in html
    assert '<details class="static-preview" open>' not in html
    assert "reasoning_step_complex_maps/reasoning_step_000.html" in html
    assert payload["nodes"][1]["input_text"] == "input"
    assert payload["nodes"][1]["decoded_argmax"] == "output"
    assert payload["nodes"][1]["level"] == 1
    assert payload["nodes"][1]["filtered_simplicial_object"]["summary"]["num_vertices"] >= 1
    assert payload["embedding_pca_diagnostics"]["coordinate_source"] == "model graph_state embeddings"
    assert "Graph-of-thought embedding-space trajectory map" in embedding_map_html
    assert "actual graph_state PCA" in embedding_map_html
    assert "distance corr" in embedding_map_html
    overlay = full_complex_payload["filtered_simplicial_object"]["trajectory_overlay"]
    assert overlay["source"] == "graph_of_thought_parent_edges"
    assert overlay["semantic_note"].startswith("Radius topology is induced")
    assert len(overlay["edges"]) == 3
    assert {edge["target"] for edge in overlay["edges"]} == {
        child_record.record_id,
        sibling_record.record_id,
        leaf_record.record_id,
    }
    decoding_overlay = full_complex_payload["filtered_simplicial_object"]["decoding_causal_overlay"]
    assert decoding_overlay["source"] == "graph_of_thought_parent_decoding_order"
    assert decoding_overlay["edge_count"] == 3
    assert all(edge["style"] == "dotted" for edge in decoding_overlay["edges"])
    assert {edge["target"] for edge in decoding_overlay["edges"]} == {
        child_record.record_id,
        sibling_record.record_id,
        leaf_record.record_id,
    }
    assert "faint GoT parent-child trajectory overlay" in full_complex_html
    assert "graph_of_thought_parent_decoding_order" in full_complex_html
    assert "got_parent_child_decoding_order" in full_complex_html
    assert "solid radius/simplicial edges induced from the same embeddings" in full_complex_html
    assert "solid radius edges and filled radius-gated 2-simplices" in full_complex_html
    assert "filled 2-simplices gated by radius slider" in full_complex_html
    assert "dotted causal and decoding overlays" in full_complex_html
    assert "play filtration min-to-max" in full_complex_html
    assert "Left is the smallest visible filtration; right is the full selected complex" in full_complex_html
    assert "Full graph-of-thought trajectory filtered simplicial complex" in full_complex_html
    assert "actual face-to-coface covers" in full_tree_html
    assert "optional sorted-label trie prefix links" in full_tree_html
    assert "not disconnected simplex columns" in full_tree_html
    assert "empty simplex" in full_tree_html
    if "no Jensen-Shannon radius complex or simplex tree was rendered" in probability_tree_html:
        assert "probability vectors were not present" in probability_tree_html
    else:
        assert "Jensen-Shannon probability SimplexTree" in probability_tree_html
        assert "actual face-to-coface covers" in probability_tree_html
    assert "Filtration radius" in full_complex_html
    assert "play filtration" in full_complex_html
    assert "Reasoning step filtered simplicial complex maps" in step_index_html
    assert "no proxy" in step_index_html
    assert "not reconstructed from the global trajectory PCA surface" in step_index_html
    assert step_manifest["contract"]["schema_version"] == "tropicalgt.reasoning_step_complex_maps.v1"
    assert step_manifest["contract"]["no_proxy_or_fallback"] is True
    assert step_manifest["contract"]["actual_data_only"] is True
    assert step_manifest["contract"]["one_page_per_model_evaluated_reasoning_step"] is True
    assert step_manifest["contract"]["embedding_trajectory_map_is_not_a_step_complex"] is True
    assert step_manifest["contract"]["step_count"] == 4
    assert step_manifest["contract"]["rendered_complex_pages"] == 4
    assert step_manifest["contract"]["rendered_simplex_tree_pages"] == 4
    assert step_manifest["contract"]["gudhi_simplex_tree_step_count"] + step_manifest["contract"]["simplex_tree_unavailable_count"] == 4
    assert len(step_manifest["steps"]) == 4
    assert [row["record_id"] for row in step_manifest["steps"]] == [node["record_id"] for node in payload["nodes"]]
    assert step_manifest["steps"][0]["complex_render_contract"].startswith("actual per-step radius-filtered complex")
    assert step_manifest["steps"][0]["simplex_tree_render_contract"].startswith("actual GUDHI SimplexTree")
    first_step = tmp_path / "reasoning_step_complex_maps" / "reasoning_step_000.html"
    assert first_step.exists()
    first_step_html = first_step.read_text(encoding="utf-8")
    assert "Filtration radius" in first_step_html
    assert "play filtration min-to-max" in first_step_html
    assert "solid radius edges and filled radius-gated 2-simplices" in first_step_html
    assert "filled 2-simplices gated by radius slider" in first_step_html
    assert "faint directed graph-token overlay" in first_step_html



def test_tropical_fan_diagnostics_unavailable_without_explicit_ideal(tmp_path: Path):
    paths = write_tropical_fan_diagnostics({}, tmp_path)
    payload = json.loads(Path(paths["tropical_fan_diagnostics_payload"]).read_text(encoding="utf-8"))
    markup = Path(paths["tropical_fan_diagnostics"]).read_text(encoding="utf-8")
    assert payload["available"] is False
    assert payload["safe_to_render_as_tropical_fan"] is False
    assert payload["diagnostics"]["status"] == "unavailable_no_model_derived_tropical_ideal"
    assert "No support-token" in payload["diagnostics"]["certificate_contract"]["no_proxy_policy"]
    assert "support-token proxies" in payload["render_contract"]
    assert "Tropical fan diagnostics unavailable" in markup
    assert "Sage tropical polynomial" in markup
    assert "one dimensional cones" in markup
    assert "not a multigraded free-resolution" in markup
    assert "Plotly.newPlot" in markup


def test_tropical_fan_diagnostics_renders_certified_explicit_ideal(tmp_path: Path, monkeypatch):
    import tropicalgt.cas_tropical as cas_tropical

    calls = []

    def fake_try_compute(spec, *, timeout_s=15.0, use_cache=None):
        calls.append((spec, timeout_s, use_cache))
        return {
            "schema_version": "tropicalgt.cas_tropical_fan.v1",
            "available": True,
            "status": "certified",
            "backend": "Macaulay2",
            "certificate_attached": True,
            "tropical_cycle_certified": True,
            "fan_diagnostics_certified": True,
            "safe_to_render_as_tropical_fan": True,
            "certificate_contract": {
                "certificate_source": "Macaulay2 Tropical tropicalVariety on an explicit QQ polynomial ideal",
                "ordinary_to_laurent_torus_scope": "explicit QQ ideal interpreted as a torus-side Laurent certificate",
                "sage_scope": (
                    "Sage tropical polynomial/variety APIs may support exact polynomial checks, but are not a substitute certificate."
                ),
                "no_proxy_policy": "No support-token proxies or chain diagnostics may substitute for this certificate.",
            },
            "ideal_schema": {"variables": ["x", "y"], "generators": ["x+y+1"]},
            "cas_artifacts": {"raw_tagged_output": "rays=matrix {{1,-1,0},{0,-1,1}}"},
            "tropical_basis_check": {"available": True, "is_tropical_basis": True, "error": None},
            "tropical_prevariety_summary": {"available": True, "rays": [[1, -1, 0], [0, -1, 1]], "max_cones": [[1], [0], [2]]},
            "fan_summary": {
                "rays": [[1, -1, 0], [0, -1, 1]],
                "max_cones": [[1], [0], [2]],
                "multiplicities": [1, 1, 1],
                "ray_count": 3,
                "ambient_dimension": 2,
                "max_cone_count": 3,
                "is_balanced": True,
                "is_pure": True,
                "is_simplicial": True,
                "one_dimensional_cone_language": "rays are one dimensional cones",
            },
            "render_warning": "Certified tropical fan diagnostics only. This is not a multigraded free-resolution or derived-equivalence certificate.",
        }

    monkeypatch.setattr(cas_tropical, "try_compute_tropical_fan_diagnostics", fake_try_compute)
    paths = write_tropical_fan_diagnostics(
        {"graph_token_trace": {"model_derived_tropical_ideal": {"variables": ["x", "y"], "generators": ["x+y+1"]}}},
        tmp_path,
        timeout_s=3.0,
    )
    payload = json.loads(Path(paths["tropical_fan_diagnostics_payload"]).read_text(encoding="utf-8"))
    markup = Path(paths["tropical_fan_diagnostics"]).read_text(encoding="utf-8")
    assert calls == [({"variables": ["x", "y"], "generators": ["x+y+1"]}, 3.0, None)]
    assert payload["available"] is True
    assert payload["source_path"] == "result.graph_token_trace.model_derived_tropical_ideal"
    assert payload["diagnostics"]["fan_summary"]["ray_count"] == 3
    assert payload["safe_to_render_as_tropical_fan"] is True
    assert "Tropical fan diagnostics: real Macaulay2 certificate" in markup
    assert "rho_0" in markup
    assert "tropical basis check" in markup
    assert "prevariety rays" in markup
    assert "one dimensional cones" in markup
    assert "not a multigraded free-resolution" in markup
    assert "Sage tropical polynomial" in markup
    assert "No support-token proxies" in markup



def _certified_toric_sidecar_fixture() -> dict[str, object]:
    schema = cas_toric.canonicalize_toric_exponent_matrix(
        {"exponent_matrix": [[1, 1, 1], [0, 1, 2]], "variable_names": ["z_0", "z_1", "z_2"], "source": "unit_test_rational_normal_curve"}
    )
    tagged = "\n".join([
        "backend=Macaulay2",
        "quasidegrees_package_available=true",
        "certificate_type=Macaulay2 Quasidegrees toricIdeal finite monomial-map certificate",
        "embedding_scope=finite_monomial_map_toric_ideal_certificate_only",
        "toric_embedding_certified=true",
        "toric_ideal_certified=true",
        "tropical_variety_embedding_certified=false",
        "global_toric_variety_embedding_certified=false",
        "ring=QQ[z_0..z_2]",
        "exponent_matrix=| 1 1 1 | || | 0 1 2 |",
        "toric_ideal_text=ideal(z_1^2-z_0*z_2)",
        "toric_ideal_generators={z_1^2-z_0*z_2}",
        "generator_count=1",
        "codimension=1",
        "dimension=2",
    ])
    parsed = {
        "quasidegrees_package_available": "true",
        "certificate_type": "Macaulay2 Quasidegrees toricIdeal finite monomial-map certificate",
        "embedding_scope": "finite_monomial_map_toric_ideal_certificate_only",
        "toric_embedding_certified": "true",
        "toric_ideal_certified": "true",
        "tropical_variety_embedding_certified": "false",
        "global_toric_variety_embedding_certified": "false",
        "ring": "QQ[z_0..z_2]",
        "toric_ideal_text": "ideal(z_1^2-z_0*z_2)",
        "toric_ideal_generators": "{z_1^2-z_0*z_2}",
        "generator_count": "1",
        "codimension": "1",
        "dimension": "2",
    }
    return cas_toric._certified_toric_embedding_result(schema, parsed, tagged, attempts=[{"backend": "Macaulay2", "status": "ran"}])


def test_toric_embedding_sidecar_unavailable_without_exponent_matrix(tmp_path: Path):
    paths = write_toric_embedding_sidecar({}, tmp_path)
    payload = json.loads(Path(paths["toric_embedding_sidecar_payload"]).read_text(encoding="utf-8"))
    html = Path(paths["toric_embedding_sidecar"]).read_text(encoding="utf-8")
    assert payload["schema_version"] == "tropicalgt.toric_embedding_sidecar_visual_audit.v1"
    assert payload["available"] is False
    assert payload["safe_to_render_as_finite_toric_ideal_sidecar"] is False
    assert payload["safe_to_render_as_tropical_variety_embedding"] is False
    assert payload["safe_to_use_as_normal_fan_certificate"] is False
    assert payload["diagnostics"]["schema_version"] == "tropicalgt.cas_toric_embedding.v1"
    assert "chart-bundle" in payload["render_contract"]
    assert "No finite monomial-map toric ideal" in html
    assert "Toric embedding sidecar unavailable" in html


def test_toric_embedding_sidecar_renders_precomputed_finite_toric_ideal_certificate(tmp_path: Path):
    report = _certified_toric_sidecar_fixture()
    paths = write_toric_embedding_sidecar({"toric_embedding_certificate": report}, tmp_path)
    payload = json.loads(Path(paths["toric_embedding_sidecar_payload"]).read_text(encoding="utf-8"))
    html = Path(paths["toric_embedding_sidecar"]).read_text(encoding="utf-8")
    assert payload["available"] is True
    assert payload["safe_to_render_as_finite_toric_ideal_sidecar"] is True
    assert payload["safe_to_render_as_tropical_variety_embedding"] is False
    assert payload["safe_to_render_as_global_toric_variety_embedding"] is False
    assert payload["safe_to_use_as_normal_fan_certificate"] is False
    assert payload["diagnostics"]["toric_ideal_certified"] is True
    assert payload["diagnostics"]["monomial_map_summary"]["exponent_matrix"] == [[1, 1, 1], [0, 1, 2]]
    assert "finite monomial-map toric ideal certificate" in html
    assert "not a normal-fan" in html
    assert "z_1^2-z_0*z_2" in html
    assert "toric_embedding_sidecar_contract" in html


def test_tropical_support_heatmap_does_not_fabricate_invalid_supports(tmp_path: Path):
    result = {
        "graph_token_trace": {
            "tokens": [
                {"index": 0, "text": "alpha", "kind": "node", "active_support_index": -1, "margin": 0.2},
                {"index": 1, "text": "beta", "kind": "node", "active_support_index": None, "margin": 0.3},
                {"index": 2, "text": "gamma", "kind": "edge", "active_support_index": 99, "margin": 0.4},
            ]
        }
    }
    paths = write_tropical_support_heatmap(result, tmp_path)
    payload = json.loads(Path(paths["tropical_support_payload"]).read_text(encoding="utf-8"))
    html = Path(paths["tropical_support_heatmap"]).read_text(encoding="utf-8")
    assert payload["metrics"]["available"] is False
    assert payload["metrics"]["reason"] == "no_valid_model_active_support_indices"
    assert payload["support_flow_edges"] == []
    assert payload["supports"] == []
    assert "No valid model active-support indices" in html


def test_inference_audit_materializes_unavailable_bifiltration_for_legacy_nonempty_scaling(tmp_path: Path):
    result = {
        "inference_scaling": {
            "candidates": [
                {
                    "record_id": "root",
                    "embedding": [0.0, 0.0, 0.0],
                    "score": 0.0,
                    "nll": 1.0,
                    "path": [],
                    "level": 0,
                }
            ]
        }
    }
    paths = write_inference_audit_artifacts(result, tmp_path, render_html=False)
    assert "trajectory_level_radius_bifiltration" in paths
    payload = json.loads(Path(paths["trajectory_level_radius_bifiltration"]).read_text(encoding="utf-8"))
    assert payload["available"] is False
    assert payload["coefficient_ring"] == "F2[x_level,x_radius]"
    assert payload["fiber_rank_profile"] == []
    assert payload["structure_maps"] == []
    assert payload["object_key_selected"] == "unavailable"
    assert payload["reason"] == "trajectory_level_radius_bifiltration_missing_for_nonempty_scaling_report_without_trajectory_growth"
    assert payload["grid_fiber_provenance"]["reason"] == payload["reason"]


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


def test_trajectory_persistence_uses_growth_and_chain_presentation_diagnostics(tmp_path: Path):
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
    assert "multiparameter persistence and chain-presentation diagnostics" in module_html
    assert "chain-presentation diagnostic" in module_html
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


def test_non_growth_persistence_landscape_is_explicitly_unavailable(tmp_path: Path):
    topo = _toy_topology(intervals=[{"dimension": 0, "birth": 0.0, "death": 0.5, "infinite": False}])
    paths = write_persistence_visualizations(topo, tmp_path)
    assert "persistence_landscapes" in paths
    landscape_html = Path(paths["persistence_landscapes"]).read_text(encoding="utf-8")
    assert "redirect to trajectory-growth artifact" in landscape_html
    assert "Open trajectory growth persistence landscapes" in landscape_html
    assert "trajectory_persistence/persistence_landscapes.html" in landscape_html


def test_tropical_support_heatmap_layout_keeps_legend_out_of_margin(tmp_path: Path):
    result = {
        "graph_token_trace": {
            "tokens": [
                {"index": 0, "text": "graph", "kind": "graph", "node_type": "graph", "active_support_index": 0, "margin": 12.0, "active_support_probability": 0.91, "support_probability_entropy_bits": 0.3, "top_model_support_probabilities": [{"index": 0, "probability": 0.91}], "support_probability_source": "model_tropical_support_probabilities"},
                {"index": 1, "text": "problem", "kind": "node", "node_type": "problem", "active_support_index": 0, "margin": 11.5, "active_support_probability": 0.82, "support_probability_entropy_bits": 0.5, "top_model_support_probabilities": [{"index": 0, "probability": 0.82}], "support_probability_source": "model_tropical_support_probabilities"},
                {"index": 2, "text": "answer", "kind": "node", "node_type": "answer", "active_support_index": 2, "margin": 0.0004, "active_support_probability": 0.63, "support_probability_entropy_bits": 1.1, "top_model_support_probabilities": [{"index": 2, "probability": 0.63}], "support_probability_source": "model_tropical_support_probabilities"},
                {"index": 3, "text": "edge", "kind": "edge", "active_support_index": 3, "margin": 0.004, "active_support_probability": 0.57, "support_probability_entropy_bits": 1.3, "top_model_support_probabilities": [{"index": 3, "probability": 0.57}], "support_probability_source": "model_tropical_support_probabilities"},
            ]
        }
    }
    paths = write_tropical_support_heatmap(result, tmp_path)
    html = Path(paths["tropical_support_heatmap"]).read_text(encoding="utf-8")
    payload = json.loads(Path(paths["tropical_support_payload"]).read_text(encoding="utf-8"))
    assert payload["metrics"]["support_probability_source"] == "model_tropical_support_probabilities"
    assert payload["metrics"]["active_support_probability_summary"]["available"] is True
    assert payload["metrics"]["support_probability_entropy_bits_summary"]["available"] is True
    assert {row["group"] for row in payload["metrics"]["query_token_group_summary"]} >= {"graph:graph", "node:problem", "node:answer", "edge:edge"}
    assert {row["group"] for row in payload["metrics"]["support_token_group_summary"]} >= {"graph:graph", "node:answer", "edge:edge"}
    assert payload["metrics"]["top_support_summary"]["support_index"] == 0
    assert payload["metrics"]["top_support_summary"]["selected_query_count"] == 2
    assert payload["metrics"]["top_support_summary"]["support_group"] == "graph:graph"
    assert payload["metrics"]["grouped_token_label_policy"].startswith("query/support labels are grouped")
    assert payload["support_flow_edges"][0]["active_support_probability"] == 0.91
    assert payload["support_flow_edges"][0]["support_assignment_status"] == "selected_observed_support"
    assert payload["support_assignment_status_by_token"][0]["rendered_as_assignment_cell"] is True
    assert payload["metrics"]["render_contract"].startswith("assignment_matrix is binary model argmax support")
    contract = payload["tropical_support_render_contract"]
    assert contract["schema_version"] == "tropicalgt.tropical_support_render.v1"
    assert contract["support_columns_policy"] == "observed_valid_active_support_indices_only"
    assert contract["assignment_matrix_binary"] is True
    assert contract["normal_fan_wall_crossing_certified"] is False
    assert contract["invalid_support_count"] == 0
    assert payload["metrics"]["render_contract_schema_version"] == contract["schema_version"]
    assert payload["metrics"]["no_proxy_or_fallback"] is True
    audit = payload["metrics"]["wall_margin_audit"]
    assert audit["strict_wall_hit_count"] == 1
    assert audit["near_wall_hit_count"] == 2
    assert audit["near_wall_only_count"] == 1
    assert audit["metric_scope"] == "margin_threshold_audit_not_certified_normal_fan_wall_crossing"
    assert audit["low_strict_wall_interpretation_status"] == "strict_wall_margin_events_observed"
    assert audit["metric_issue"] is False
    assert payload["support_flow_edges"][2]["wall_margin_bucket"] == "strict_wall"
    assert payload["support_flow_edges"][3]["wall_margin_bucket"] == "near_wall"
    assert "active support probability" in html
    assert "Grouped token labels" in html
    assert "near-wall hit rate" in html
    assert "Wall audit scope" in html
    assert "strict wall threshold" in html
    assert "Tropical active-support audit" in html
    assert "No support-token proxies" in html
    assert "tropical_support_render_contract" in html
    compact = html.replace(" ", "")
    assert '"showlegend":false' in compact
    assert '"r":190' in compact
    assert "Support frequency and mean selected margin" in html


def test_tropical_support_mixed_invalid_active_support_indices_are_not_fabricated(tmp_path: Path):
    result = {
        "graph_token_trace": {
            "tokens": [
                {"index": 0, "text": "valid root", "kind": "node", "node_type": "root", "active_support_index": 0, "margin": 0.7},
                {"index": 1, "text": "invalid negative", "kind": "node", "node_type": "leaf", "active_support_index": -1, "margin": 0.2},
                {"index": 2, "text": "invalid high", "kind": "edge", "edge_type": "causal", "active_support_index": 99, "margin": 0.3},
                {"index": 3, "text": "valid self", "kind": "node", "node_type": "answer", "active_support_index": 3, "margin": 0.4},
            ]
        }
    }
    paths = write_tropical_support_heatmap(result, tmp_path)
    html = Path(paths["tropical_support_heatmap"]).read_text(encoding="utf-8")
    payload = json.loads(Path(paths["tropical_support_payload"]).read_text(encoding="utf-8"))
    assert payload["metrics"]["available"] is True
    assert payload["metrics"]["support_indices"] == [0, 3]
    assert payload["metrics"]["invalid_support_count"] == 2
    assert payload["metrics"]["valid_support_assignment_count"] == 2
    assert payload["tropical_support_render_contract"]["invalid_support_count"] == 2
    assert payload["tropical_support_render_contract"]["valid_support_assignment_count"] == 2
    assert payload["tropical_support_render_contract"]["no_proxy_or_fallback"] is True
    assert payload["support_assignment_status_by_token"][1]["status"] == "invalid_active_support_index"
    assert payload["support_assignment_status_by_token"][1]["rendered_as_assignment_cell"] is False
    assert payload["support_flow_edges"][1]["support_assignment_status"] == "invalid_active_support_index"
    assert payload["support_flow_edges"][1]["rendered_as_assignment_cell"] is False
    assert payload["support_flow_edges"][2]["support_label"] == "invalid"
    assert payload["assignment_matrix"][1] == [0.0, 0.0]
    assert payload["selected_margin_matrix"][1] == [None, None]
    assert "invalid_active_support_index" in html
    assert "No support-token proxies" in html


def test_tropical_support_wall_audit_preserves_explicit_zero_threshold(tmp_path: Path):
    result = {
        "wall_margin_threshold": 0.0,
        "near_wall_margin_threshold": 0.0001,
        "graph_token_trace": {
            "tokens": [
                {"index": 0, "text": "zero", "kind": "node", "node_type": "root", "active_support_index": 0, "margin": 0.0},
                {"index": 1, "text": "near", "kind": "node", "node_type": "leaf", "active_support_index": 1, "margin": 0.0001},
                {"index": 2, "text": "inside", "kind": "edge", "edge_type": "causal", "active_support_index": 1, "margin": 0.01},
            ]
        },
    }
    paths = write_tropical_support_heatmap(result, tmp_path)
    html = Path(paths["tropical_support_heatmap"]).read_text(encoding="utf-8")
    payload = json.loads(Path(paths["tropical_support_payload"]).read_text(encoding="utf-8"))
    audit = payload["metrics"]["wall_margin_audit"]
    assert audit["wall_margin_threshold"] == 0.0
    assert audit["near_wall_margin_threshold"] == 0.0001
    assert audit["strict_wall_hit_count"] == 1
    assert audit["near_wall_hit_count"] == 2
    assert audit["near_wall_only_count"] == 1
    assert audit["metric_scope"] == "margin_threshold_audit_not_certified_normal_fan_wall_crossing"
    assert audit["metric_issue"] is False
    assert payload["support_flow_edges"][0]["wall_margin_bucket"] == "strict_wall"
    assert payload["support_flow_edges"][1]["wall_margin_bucket"] == "near_wall"
    assert "Wall audit scope" in html


def test_tropical_support_high_collapse_uses_compact_diagnostic(tmp_path: Path):
    tokens = []
    for idx in range(7):
        tokens.append(
            {
                "index": idx,
                "text": f"token-{idx}",
                "kind": "node" if idx < 5 else "edge",
                "node_type": "problem" if idx < 5 else "edge",
                "active_support_index": 0 if idx < 5 else 6,
                "margin": 20.0 if idx < 5 else 0.02,
            }
        )
    paths = write_tropical_support_heatmap({"graph_token_trace": {"tokens": tokens}}, tmp_path)
    html = Path(paths["tropical_support_heatmap"]).read_text(encoding="utf-8")
    payload = json.loads(Path(paths["tropical_support_payload"]).read_text(encoding="utf-8"))
    assert payload["metrics"]["layout_mode"] == "collapse_diagnostic"
    assert payload["metrics"]["top_support_collapse_rate"] > 0.7
    assert payload["metrics"]["top_support_summary"]["selected_query_count"] == 5
    assert payload["metrics"]["raw_token_labels_truncated"] is True
    assert "Tropical active-support collapse diagnostic" in html
    assert "Grouped token labels" in html
    assert "token groups" in html
    assert "top support" in html
    assert "captures" in html


def test_plotly_dark_html_promotes_static_preview_for_webgl_failures(tmp_path: Path):
    obj = {
        "record_id": "panel",
        "summary": {"num_vertices": 1, "num_edges": 0, "num_two_simplices": 0},
        "simplices": [{"simplex": ["panel"], "dimension": 0, "filtration": 0.0, "embedding": [0.0, 0.0, 0.0]}],
    }
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                mode="markers",
                customdata=[0],
            )
        ]
    )
    from tropicalgt.visualization import _write_plotly_dark_html  # local import keeps public imports tidy

    path = tmp_path / "webgl_fallback.html"
    _write_plotly_dark_html(path, fig, "WebGL fallback test", _simplicial_panel_items([obj], ["panel hover"]))
    html = path.read_text(encoding="utf-8")
    assert 'class="static-preview"' in html
    assert "Static SVG fallback preview" in html
    assert "same filtered-complex payload" in html


def test_graphcg_visualization_preserves_projection_basis_certificate(tmp_path: Path):
    scaling_report = {
        "candidates": [
            {
                "record_id": f"node-{idx}",
                "path": ["expand", str(idx)],
                "level": idx,
                "nll": 1.0 + 0.1 * idx,
                "score": 0.2 * idx,
                "graphcg_projection": {
                    "basis": "effective_full_rank_qr",
                    "mean_abs_offdiag_cosine": 0.01 + 0.001 * idx,
                    "max_abs_offdiag_cosine": 0.03 + 0.001 * idx,
                    "all_direction_cosines": [0.4 + 0.01 * idx, -0.2, 0.1, 0.05],
                },
            }
            for idx in range(3)
        ]
    }
    paths = write_graphcg_trajectory_visualization(scaling_report, tmp_path)
    payload = json.loads(Path(paths["graphcg_direction_cosines_payload"]).read_text(encoding="utf-8"))
    html = Path(paths["graphcg_direction_cosines"]).read_text(encoding="utf-8")
    cert = payload["projection_basis_certificate"]
    assert cert["source"] == "candidate.graphcg_projection"
    assert cert["projection_basis"] == "effective_full_rank_qr"
    assert cert["basis_source_counts"] == {"effective_full_rank_qr": 3}
    assert cert["all_candidates_have_all_direction_cosines"] is True
    assert cert["direction_count"] == 4
    assert cert["max_abs_offdiag_cosine_max"] > 0.0
    assert payload["visible_direction_tick_label_limit"] == 8
    assert payload["exact_direction_labels_available_in_hover_and_payload"] is True
    assert payload["readability_contract"].startswith("four coordinated panels render all model GraphCG directions")
    assert payload["panel_names"] == [
        "all_direction_heatmap",
        "full_rank_activity_spectrum",
        "candidate_activity_by_observed_got_state",
        "direction_signed_bias",
    ]
    assert payload["panel_count"] == 4
    assert payload["directions_sampled_for_heatmap"] is False
    assert payload["all_direction_heatmap_available"] is True
    assert payload["direction_spectrum_panel_available"] is True
    assert payload["candidate_activity_panel_available"] is True
    assert payload["direction_signed_bias_panel_available"] is True
    assert len(payload["candidate_effective_direction_count"]) == 3
    assert len(payload["direction_signed_mean_sorted"]) == 4
    assert "basis=effective_full_rank_qr" in html
    assert "heatmap shows all" in html
    assert "Readable full-rank heatmap" in html
    assert "Full-rank activity spectrum" in html
    assert "Candidate activity by observed GoT state" in html
    assert "Signed bias for every direction" in html
    assert "direction rank by activity" in html


def test_analogical_memory_visualization_renders_simplicial_maps(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    descriptors = [
        {"index": 0, "kind": "node", "node_id": "a", "text": "alpha"},
        {"index": 1, "kind": "node", "node_id": "b", "text": "beta"},
        {"index": 2, "kind": "node", "node_id": "c", "text": "gamma"},
    ]
    embeddings = [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.2, 1.0, 0.1]]
    probabilities = [
        [0.82, 0.12, 0.06],
        [0.10, 0.78, 0.12],
        [0.08, 0.14, 0.78],
    ]
    obj = build_embedding_radius_simplicial_object(
        record,
        descriptors,
        embeddings,
        token_probabilities=probabilities,
        metric="jensen_shannon",
    )
    assert obj["summary"]["filtration_model"] == "model_tropical_support_probability_jensen_shannon_vietoris_rips_2_skeleton"
    retrieval_probability_map = probability_simplicial_map_diagnostics(obj, obj)
    assert retrieval_probability_map["available"] is True
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
                "base_retrieval_score": 0.5,
                "persistence_landscape_score_contribution": 0.1,
                "persistence_vector_score_contribution": 0.3,
                "probability_simplicial_map_score_contribution": 0.18,
                "probability_simplicial_map_similarity": 0.9,
                "probability_simplicial_map_available": True,
                "probability_simplicial_map_preservation_rate": 1.0,
                "probability_simplicial_map_source": "model_probability_jensen_shannon_assignment",
                "probability_simplicial_map": retrieval_probability_map,
                "retrieval_score_components": {
                    "embedding": 0.2,
                    "signature": 0.2,
                    "quality": 0.1,
                    "persistence_landscape": 0.1,
                    "persistence_vector_family": 0.3,
                    "probability_simplicial_map": 0.18,
                },
                "retrieval_weights": {
                    "persistence_landscape_weight": 0.08,
                    "persistence_vector_weight": 0.18,
                    "probability_simplicial_map_weight": 0.20,
                    "persistence_vector_includes_landscape": False,
                },
                "quality_score": 0.6,
                "trajectory_probability_filtered_simplicial_object": obj,
                "topological_algebra": topo,
                "derived_signature": topo["derived_equivalence_signature"],
            },
            {
                "memory_id": "mem1",
                "record_id": "rec1",
                "retrieval_score": 0.7,
                "embedding_similarity": 0.6,
                "signature_similarity": 0.5,
                "base_retrieval_score": 0.4,
                "persistence_landscape_score_contribution": 0.08,
                "persistence_vector_score_contribution": 0.22,
                "probability_simplicial_map_score_contribution": 0.10,
                "probability_simplicial_map_similarity": 0.5,
                "probability_simplicial_map_available": True,
                "probability_simplicial_map_preservation_rate": 1.0,
                "probability_simplicial_map_source": "model_probability_jensen_shannon_assignment",
                "probability_simplicial_map": retrieval_probability_map,
                "retrieval_score_components": {
                    "embedding": 0.16,
                    "signature": 0.14,
                    "quality": 0.1,
                    "persistence_landscape": 0.08,
                    "persistence_vector_family": 0.22,
                    "probability_simplicial_map": 0.10,
                },
                "retrieval_weights": {"persistence_landscape_weight": 0.08, "persistence_vector_weight": 0.18, "probability_simplicial_map_weight": 0.20},
                "quality_score": 0.4,
                "trajectory_probability_filtered_simplicial_object": obj,
                "topological_algebra": topo,
                "derived_signature": topo["derived_equivalence_signature"],
            }
        ],
    }
    paths = write_analogical_memory_visualization(
        memory,
        tmp_path,
        query_context={"trajectory_probability_filtered_simplicial_object": obj, "topological_algebra": topo, "label": "query trajectory"},
    )
    html = Path(paths["analogical_memory_retrieval_html"]).read_text(encoding="utf-8")
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    assert "Analogical probability-matched correspondence" in html
    assert "filtered-complex certificate from model-probability Jensen-Shannon assignment" in html
    assert "certified filtered simplicial map" in html
    assert "model_probability_jensen_shannon_assignment" in html
    assert "persistent homology similarity" in html
    assert "chain-presentation diagnostic similarity" in html
    assert "persistence-landscape L2 similarity" in html
    assert "persistence-landscape cosine" in html
    assert "vertex-only correspondences" in html
    assert "base retrieval" in html
    assert "landscape contribution" in html
    assert "vector-family contribution" in html
    assert "probability-map contribution" in html
    assert "retrieval probability map" in html
    assert "retrieval weights" in html
    assert "preserved 1-simplex correspondence" in html
    assert "certificate diagnostic" in html
    assert "preserved 1-simplex correspondences" in html
    assert "vertex-only correspondences (legend only)" in html
    compact = html.replace(" ", "")
    assert '"domain":{"x":[0.0,0.62],"y":[0.24,0.98]}' in compact
    assert '"domain":{"x":[0.64,0.995],"y":[0.24,0.96]}' in compact
    assert "simplicial-object-panel" in html
    assert '<input id="filtration-slider"' in html
    assert '<div class="filtration-controls"' in html
    assert "memory 2" not in html
    assert "analogical_memory_topk_index_html" in paths
    assert "analogical_memory_map_02_html" in paths
    index_html = Path(paths["analogical_memory_topk_index_html"]).read_text(encoding="utf-8")
    rank2_html = Path(paths["analogical_memory_map_02_html"]).read_text(encoding="utf-8")
    assert "Analogical top-k probability correspondences" in index_html
    assert "Edge, face, and filtration preservation can fail" in index_html
    assert "landscape L2 sim" in index_html
    assert "landscape cosine" in index_html
    assert "vector aggregate" in index_html
    assert "prob-map contrib." in index_html
    assert "prob-map source" in index_html
    assert "map claim" in index_html
    assert "certified_filtered_simplicial_map" in index_html
    assert "retrieval-side probability-map score contribution" in index_html
    assert "vectorized GUDHI family" in index_html
    assert "BettiCurve, Silhouette" in index_html
    assert "landscape contrib." in index_html
    assert "vector contrib." in index_html
    assert "rank 2" in rank2_html
    assert '<input id="filtration-slider"' in rank2_html
    assert len(maps["maps"]) == 2
    assert maps["topk_contract"]["schema_version"] == "tropicalgt.analogical_topk.v1"
    assert maps["topk_contract"]["no_proxy_or_fallback"] is True
    assert maps["topk_contract"]["retrieval_requires_model_probability_vectors"] is True
    assert maps["topk_contract"]["embedding_only_assignment_allowed"] is False
    assert maps["topk_contract"]["assignment_metric"] == "jensen_shannon_distance_on_model_probability_vectors"
    assert maps["topk_contract"]["qualified_model_probability_memory_count"] == 2
    assert maps["topk_contract"]["top_k_rendered"] == 2
    assert maps["topk_contract"]["query_complex_source"] == "trajectory_probability_filtered_simplicial_object"
    assert maps["maps"][1]["pair_page"].endswith("analogical_memory_map_02.html")
    assert not Path(maps["maps"][0]["pair_page"]).is_absolute()
    assert not Path(maps["maps"][1]["pair_page"]).is_absolute()
    assert maps["maps"][0]["edge_preservation_rate"] >= 0.0
    assert maps["maps"][0]["map_source"] == "model_probability_jensen_shannon_assignment"
    assert maps["maps"][0]["map_certificate_source"] == "retrieval_probability_simplicial_map_certificate"
    assert maps["maps"][0]["retrieval_probability_certificate_available"] is True
    assert maps["maps"][0]["simplicial_map_certificate"]["no_proxy_or_fallback"] is True
    assert maps["maps"][0]["query_complex_source"] == "trajectory_probability_filtered_simplicial_object"
    assert maps["maps"][0]["codomain_complex_source"] == "trajectory_probability_filtered_simplicial_object"
    assert maps["maps"][0]["simplicial_map_certificate"]["source"] == "retrieval_probability_simplicial_map_certificate"
    assert maps["maps"][0]["derived_signature_similarity"] >= 0.0
    assert maps["maps"][0]["persistence_vector_representation_similarity"]["includes_landscape"] is False
    assert maps["maps"][0]["persistence_landscape_vector_available"] == 1.0
    assert maps["maps"][0]["persistence_landscape_l2_similarity"] > 0.0
    assert maps["maps"][0]["persistence_landscape_overlap_dim"] > 0.0
    assert maps["maps"][0]["persistence_vector_component_count"] >= 2.0
    assert maps["maps"][0]["base_retrieval_score"] == 0.5
    assert maps["maps"][0]["persistence_landscape_score_contribution"] == 0.1
    assert maps["maps"][0]["persistence_vector_score_contribution"] == 0.3
    assert maps["maps"][0]["retrieval_score_components"]["persistence_vector_family"] == 0.3
    assert maps["maps"][0]["retrieval_weights"]["persistence_vector_weight"] == 0.18
    assert "persistence_vector_components" in maps["maps"][0]
    component_methods = {row["method"] for row in maps["maps"][0]["persistence_vector_components"]}
    assert "landscape" not in component_methods
    assert {"betti_curve", "silhouette"}.issubset(component_methods)
    assert "persistence_vector_comparison_space" in maps["maps"][0]
    assert "differentiable" in maps["maps"][0]["persistence_vector_differentiable_note"]
    assert maps["maps"][0]["derived_invariant_comparison"]["persistence_landscape_vector_available"] is True
    assert "is_simplicial_on_displayed_skeleton" in maps["maps"][0]
    assert maps["maps"][0]["chain_map_diagnostics"]["available"] is True
    assert maps["maps"][0]["persistence_module_morphism_diagnostics"]["available"] is True
    assert maps["maps"][0]["safe_to_render_as_simplicial_map"] is True
    assert maps["maps"][0]["safe_to_render_as_chain_map"] is True
    assert maps["maps"][0]["safe_to_render_as_persistence_module_morphism"] is True
    assert maps["maps"][0]["map_render_claim"] == "certified_filtered_simplicial_map"
    assert maps["maps"][0]["map_claim_failure_reason"] is None
    assert maps["maps"][0]["simplex_tree_map"]["filtered_simplicial_map_certified"] is True
    assert maps["maps"][0]["persistence_module_morphism_diagnostics"]["free_resolution_required"] is False
    assert maps["maps"][0]["derived_invariant_comparison"]["real_free_resolution_comparison"]["safe_for_derived_category_claims"] is False
    assert "free_resolution_similarity_interpretation" in maps["maps"][0]["derived_invariant_comparison"]
    assert isinstance(maps["maps"][0]["preserved_edge_pairs"], list)
    assert isinstance(maps["maps"][0]["failed_edge_pairs"], list)
    assert isinstance(maps["maps"][0]["preserved_edge_query_vertices"], list)



def _topology_with_certified_real_resolution(*, input_hash: str = "hash-a", fitt0: str = "ideal(x_level,x_radius)", multiplier_matrix: str = "matrix {{1}}"):
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
            "certificate_summary": {
                "available": True,
                "backend": "Macaulay2",
                "certificate_type": "Macaulay2 res coker presentation over multigraded F2 polynomial ring",
                "certificate_attached": True,
                "exactness_certified": True,
                "minimality_certified": True,
                "homogeneous_presentation": True,
                "input_sha256": input_hash,
                "safe_to_render_as_multigraded_free_resolution": True,
                "no_proxy_policy": "Only exact CAS certificates are rendered as resolutions.",
            },
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
                "safe_to_render_multiplier_output": True,
                "is_resolution_backend": False,
                "requires_certified_macaulay2_chain_complex": True,
                "safe_to_substitute_for_resolution": False,
                "diagnostic_contract": {
                    "safe_to_use_as_resolution_certificate": False,
                    "safe_to_substitute_for_resolution": False,
                },
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


def test_certified_cas_diagnostic_tables_require_explicit_structured_certificates():
    topology = _topology_with_certified_real_resolution()
    real = topology["commutative_algebra"]["two_parameter_chain_presentation_diagnostics"]["real_free_resolution"]

    ideal_headers, ideal_columns = _m2_ideal_diagnostic_columns({"real_free_resolution": real})
    assert ideal_headers == ["kind", "name", "order/index", "ideal", "source/method"]
    assert ideal_columns[0] == ["unavailable"]
    assert "lacks ideal_diagnostics certificate" in ideal_columns[4][0]
    assert "Fitt0" not in ideal_columns[1]
    assert "minors_1" not in ideal_columns[1]

    real_with_structured = json.loads(json.dumps(real))
    real_with_structured["cas_artifacts"]["ideal_diagnostics"] = {
        "available": True,
        "presentation_shape": [1, 2],
        "fitting_invariants": [
            {
                "name": "Fitt0",
                "fitting_index": 0,
                "determinantal_order": 1,
                "ideal_text": "ideal(x_level,x_radius)",
                "method": "Fitt_j(coker(PM)) = I_{rows-j}(PM)",
            }
        ],
        "determinantal_minors": [
            {
                "name": "minors_1",
                "minor_order": 1,
                "ideal_text": "ideal(x_level,x_radius)",
                "method": "determinantal ideal generated by order-k minors",
            }
        ],
    }
    real_with_structured["cas_artifacts"]["buchsbaum_eisenbud_rank_conditions"] = {
        "available": True,
        "image_rank_estimates_by_differential": {"d1": 1},
        "differential_shapes": {"d1": [1, 1]},
        "shape_bounds_hold": True,
        "is_independent_certificate": False,
        "paper_method_note": "For an exact finite free complex, Buchsbaum-Eisenbud rank equalities apply.",
    }
    real_with_structured["cas_artifacts"]["grade_depth_regular_diagnostics"] = {
        "available": True,
        "backend": "Macaulay2",
        "source": "Macaulay2 codim/depth/rank ideals on certified resolution differentials",
        "ambient_ring_dimension": 2,
        "rank_ideal_diagnostics": [
            {
                "homological_degree": 1,
                "rank": 1,
                "rank_ideal": "ideal(x_level)",
                "rank_ideal_codim": 1,
                "rank_ideal_depth": 1,
                "grade_lower_bound_holds": True,
            }
        ],
        "regular_element_certificate_available": False,
        "regular_element_certificate_reason": "No independent regular-sequence certificate is substituted.",
        "not_a_resolution_certificate_by_itself": True,
    }

    _, structured_ideal_columns = _m2_ideal_diagnostic_columns({"real_free_resolution": real_with_structured})
    ideal_rows = list(zip(*structured_ideal_columns))
    assert any(row[0] == "Fitting invariant" and row[1] == "Fitt0" for row in ideal_rows)
    assert any(row[0] == "determinantal minors" and row[1] == "minors_1" for row in ideal_rows)

    _, be_columns = _m2_be_diagnostic_columns({"real_free_resolution": real_with_structured})
    be_rows = list(zip(*be_columns))
    assert any(row[0] == "CAS exactness" and row[2] == "True" for row in be_rows)
    assert any(row[0] == "BEMultipliers" and row[2] == "1x1" for row in be_rows)
    assert any(row[0] == "BEMultipliers contract" and "substitute=False" in row[2] and "resolution_backend=False" in row[3] for row in be_rows)
    assert any(row[0] == "BE rank condition" and row[1] == "d1" and "rank=1" in row[2] for row in be_rows)
    assert any(row[0] == "grade/depth diagnostic" and row[1] == "d1" and "codim=1" in row[2] and "regular_element_certificate=False" in row[3] for row in be_rows)
    assert any(row[0] == "regular-element note" and row[1] == "CAS grade/depth" for row in be_rows)
    assert any(row[0] == "method note" and row[1] == "Buchsbaum-Eisenbud" for row in be_rows)

    display = _cas_real_resolution_display(real_with_structured)
    assert display["certificate_summary"]["certificate_type"].startswith("Macaulay2 res")
    assert display["buchsbaum_eisenbud_diagnostics"]["safe_to_render_multiplier_output"] is True
    assert display["buchsbaum_eisenbud_diagnostics"]["is_resolution_backend"] is False
    assert display["buchsbaum_eisenbud_diagnostics"]["safe_to_substitute_for_resolution"] is False
    assert display["grade_depth_regular_diagnostics"]["rank_ideal_diagnostics"][0]["rank_ideal_codim"] == 1
    cert_headers, cert_columns = _m2_certificate_columns({"real_free_resolution": real_with_structured}, {"module_ring": "F2[x_level,x_radius]"})
    assert cert_headers == ["diagnostic", "value"]
    cert_rows = dict(zip(cert_columns[0], cert_columns[1]))
    assert cert_rows["CAS certificate type"].startswith("Macaulay2 res")
    assert cert_rows["CAS homogeneous presentation"] == "True"
    assert cert_rows["CAS input sha256"] == real_with_structured["input_sha256"]
    assert "Only exact CAS certificates" in cert_rows["CAS no-proxy policy"]
    assert cert_rows["BEMultipliers safe render"] == "True"
    assert cert_rows["BEMultipliers is resolution backend"] == "False"
    assert cert_rows["BEMultipliers substitute for resolution"] == "False"
    assert "rank_ideal_codim" in cert_rows["grade/depth diagnostics"]
    assert cert_rows["regular-element certificate"] == "False"


def test_derived_comparison_requires_matching_certified_cas_artifacts():
    query = _topology_with_certified_real_resolution()
    matching = _topology_with_certified_real_resolution()
    comparison = _derived_invariant_comparison(
        query,
        matching,
        sim={"derived_signature_similarity": 1.0, "chain_presentation_similarity": 1.0, "derived_algebraic_similarity": 1.0},
    )
    real = comparison["real_free_resolution_comparison"]
    assert real["available"] is True
    assert real["safe_for_derived_category_claims"] is True
    assert real["certified_cas_evidence_match"] is True
    assert real["certified_cas_evidence_similarity"] == 1.0
    assert comparison["certified_cas_evidence_match"] is True
    assert comparison["derived_equivalence_claim"] == "cas_certified_matching_real_resolution_witness"
    assert real["component_matches"]["fitting_ideals"] is True
    assert real["component_matches"]["buchsbaum_eisenbud"] is True
    assert real["diagnostic_only_components"] == ["buchsbaum_eisenbud"]
    assert real["mismatch_explanations"] == []
    assert real["comparison_policy"] == "no_proxy_no_fallback_exact_cas_components_only"
    assert "diagnostic-only" in real["component_explanations"]["buchsbaum_eisenbud"]
    assert "not substitute resolution evidence" in real["interpretation"]

    mismatched = _topology_with_certified_real_resolution(
        input_hash="hash-b",
        fitt0="ideal(x_radius^2)",
        multiplier_matrix="matrix {{x_radius}}",
    )
    mismatch = _derived_invariant_comparison(
        query,
        mismatched,
        sim={"derived_signature_similarity": 1.0, "chain_presentation_similarity": 1.0, "derived_algebraic_similarity": 1.0},
    )
    real_mismatch = mismatch["real_free_resolution_comparison"]
    assert real_mismatch["available"] is True
    assert real_mismatch["safe_for_derived_category_claims"] is False
    assert real_mismatch["certified_cas_evidence_match"] is False
    assert 0.0 < real_mismatch["certified_cas_evidence_similarity"] < 1.0
    assert "input_sha256" in real_mismatch["mismatched_components"]
    assert "fitting_ideals" in real_mismatch["mismatched_components"]
    assert "buchsbaum_eisenbud" in real_mismatch["mismatched_components"]
    be_mismatch = next(row for row in real_mismatch["mismatch_explanations"] if row["component"] == "buchsbaum_eisenbud")
    assert be_mismatch["diagnostic_only"] is True
    assert "cannot substitute" in be_mismatch["explanation"]
    assert real_mismatch["comparison_policy"] == "no_proxy_no_fallback_exact_cas_components_only"
    assert "cannot substitute" in real_mismatch["interpretation"]
    assert mismatch["derived_equivalence_claim"] == "compatible_finite_invariant_witness"
    assert "do not match" in mismatch["free_resolution_similarity_interpretation"]
    assert "diagnostic-only BEMultipliers" in mismatch["free_resolution_similarity_interpretation"]

    unavailable = _derived_invariant_comparison(
        {"commutative_algebra": {}},
        matching,
        sim={"derived_signature_similarity": 1.0, "chain_presentation_similarity": 1.0, "derived_algebraic_similarity": 1.0},
    )
    real_unavailable = unavailable["real_free_resolution_comparison"]
    assert real_unavailable["available"] is False
    assert real_unavailable["comparison_policy"] == "no_proxy_no_fallback_exact_cas_components_only"
    assert "unavailable, not estimated" in real_unavailable["unavailable_explanation"]
    assert real_unavailable["unavailable_reasons"]["query"]
    assert real_unavailable["mismatch_explanations"] == []


def test_analogical_memory_visualization_requires_retrieval_probability_map_certificate(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    descriptors = [
        {"index": 0, "kind": "node", "node_id": "a", "text": "alpha"},
        {"index": 1, "kind": "node", "node_id": "b", "text": "beta"},
    ]
    embeddings = [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]]
    probabilities = [[0.82, 0.18], [0.16, 0.84]]
    obj = build_embedding_radius_simplicial_object(
        record,
        descriptors,
        embeddings,
        token_probabilities=probabilities,
        metric="jensen_shannon",
    )
    topo = _toy_topology(intervals=[{"dimension": 0, "birth": 0.0, "death": None, "infinite": True}])
    memory = {
        "bank_path": "",
        "retrieved": [
            {
                "memory_id": "mem-missing-cert",
                "record_id": "rec-missing-cert",
                "retrieval_score": 0.8,
                "trajectory_probability_filtered_simplicial_object": obj,
                "topological_algebra": topo,
            }
        ],
    }
    paths = write_analogical_memory_visualization(
        memory,
        tmp_path,
        query_context={"trajectory_probability_filtered_simplicial_object": obj, "topological_algebra": topo},
    )
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    report = maps["maps"][0]
    assert report["retrieval_probability_certificate_available"] is False
    assert report["simplicial_map_failure_reason"] == "missing_retrieval_probability_simplicial_map_certificate"
    assert report["simplicial_map_certificate"]["no_proxy_or_fallback"] is True
    assert report["chain_map_diagnostics"]["available"] is False
    assert report["persistence_module_morphism_diagnostics"]["available"] is False
    assert report["safe_to_render_as_simplicial_map"] is False
    assert report["safe_to_render_as_chain_map"] is False
    assert report["safe_to_render_as_persistence_module_morphism"] is False
    assert report["map_render_claim"] == "unavailable_probability_correspondence_certificate"
    assert report["map_claim_failure_reason"] == "missing_retrieval_probability_simplicial_map_certificate"


def test_analogical_memory_visualization_labels_failed_probability_correspondence_not_map(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    descriptors = [
        {"index": 0, "kind": "node", "node_id": "a", "text": "alpha"},
        {"index": 1, "kind": "node", "node_id": "b", "text": "beta"},
        {"index": 2, "kind": "node", "node_id": "c", "text": "gamma"},
    ]
    embeddings = [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.2, 1.0, 0.1]]
    probabilities = [[0.82, 0.12, 0.06], [0.10, 0.78, 0.12], [0.08, 0.14, 0.78]]
    query_obj = build_embedding_radius_simplicial_object(
        record,
        descriptors,
        embeddings,
        token_probabilities=probabilities,
        metric="jensen_shannon",
    )
    vertex_only = {
        **query_obj,
        "simplices": [row for row in query_obj["simplices"] if row.get("dimension") == 0],
        "summary": {**query_obj["summary"], "simplices": 3, "edges": 0, "faces": 0},
    }
    failed_probability_map = probability_simplicial_map_diagnostics(query_obj, vertex_only)
    assert failed_probability_map["available"] is False
    topo = _toy_topology(intervals=[{"dimension": 0, "birth": 0.0, "death": None, "infinite": True}])
    memory = {
        "bank_path": "",
        "retrieved": [
            {
                "memory_id": "mem-failed-map",
                "record_id": "rec-failed-map",
                "retrieval_score": 0.6,
                "probability_simplicial_map": failed_probability_map,
                "probability_simplicial_map_source": "model_probability_jensen_shannon_assignment",
                "probability_simplicial_map_available": False,
                "trajectory_probability_filtered_simplicial_object": vertex_only,
                "topological_algebra": topo,
            }
        ],
    }
    paths = write_analogical_memory_visualization(
        memory,
        tmp_path,
        query_context={"trajectory_probability_filtered_simplicial_object": query_obj, "topological_algebra": topo},
    )
    html = Path(paths["analogical_memory_retrieval_html"]).read_text(encoding="utf-8")
    index_html = Path(paths["analogical_memory_topk_index_html"]).read_text(encoding="utf-8")
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    report = maps["maps"][0]
    assert ("probability correspondence only; no simplicial/chain/persistence morphism is asserted" in html) or ("probability correspondence only; no simplicial\\u002fchain\\u002fpersistence morphism is asserted" in html)
    assert "not a simplicial map on the displayed skeleton" in html
    assert "map claim" in index_html
    assert "probability_correspondence_not_a_simplicial_map" in index_html
    assert report["retrieval_probability_certificate_available"] is True
    assert report["is_filtered_simplicial_map"] is False
    assert report["safe_to_render_as_simplicial_map"] is False
    assert report["safe_to_render_as_chain_map"] is False
    assert report["safe_to_render_as_persistence_module_morphism"] is False
    assert report["map_render_claim"] == "probability_correspondence_not_a_simplicial_map"
    assert report["map_claim_failure_reason"] == "simplex_tree_map_not_fully_preserved"
    assert report["chain_map_diagnostics"]["available"] is False
    assert report["persistence_module_morphism_diagnostics"]["available"] is False
    assert report["simplex_tree_map"]["filtered_simplicial_map_certified"] is False
    assert "no simplicial map" in report["simplex_tree_map"]["interpretation"]


def test_analogical_memory_visualization_rejects_non_trajectory_probability_fallback(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    descriptors = [
        {"index": 0, "kind": "node", "node_id": "a", "text": "alpha"},
        {"index": 1, "kind": "node", "node_id": "b", "text": "beta"},
    ]
    obj = build_embedding_radius_simplicial_object(
        record,
        descriptors,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        token_probabilities=[[0.8, 0.2], [0.3, 0.7]],
        metric="jensen_shannon",
    )
    memory = {"bank_path": "", "retrieved": [{"memory_id": "mem0", "record_id": "rec0", "probability_filtered_simplicial_object": obj}]}
    paths = write_analogical_memory_visualization(
        memory,
        tmp_path,
        query_context={"probability_filtered_simplicial_object": obj},
    )
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    assert maps["available"] is False
    assert maps["reason"] == "missing_model_probability_query_complex"
    assert maps["reason_detail"].startswith("No model probability filtered query trajectory complex")
    assert maps["topk_contract"]["no_proxy_or_fallback"] is True
    assert maps["topk_contract"]["embedding_only_assignment_allowed"] is False
    assert maps["topk_contract"]["raw_retrieved_count"] == 1
    assert maps["topk_contract"]["qualified_model_probability_memory_count"] == 0
    assert maps["maps"] == []


def test_analogical_memory_without_retrieval_emits_unavailable_surfaces(tmp_path: Path):
    paths = write_analogical_memory_visualization({"bank_path": "", "retrieved": []}, tmp_path, query_context={})
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    index_html = Path(paths["analogical_memory_topk_index_html"]).read_text(encoding="utf-8")
    map_html = Path(paths["analogical_memory_map_02_html"]).read_text(encoding="utf-8")
    assert maps["available"] is False
    assert maps["reason"] == "no_non_self_model_memory"
    assert maps["reason_detail"].startswith("No non-self model-probability analogical memories retrieved")
    assert maps["topk_contract"]["status"] == "no_non_self_model_memory"
    assert maps["topk_contract"]["no_proxy_or_fallback"] is True
    assert maps["topk_contract"]["retrieval_requires_model_probability_vectors"] is True
    assert maps["topk_contract"]["top_k_rendered"] == 0
    assert maps["maps"] == []
    assert "Analogical top-k probability correspondences" in index_html
    assert "Insufficient model-probability memory" in index_html
    assert "No retrieved memories" in index_html
    assert "Embedding-only assignments are rejected" in index_html
    assert "Analogical probability-matched correspondence filtered-complex certificate unavailable" in map_html
    assert "No vertex assignment" in map_html


def test_analogical_memory_without_query_probabilities_is_unavailable_not_fallback(tmp_path: Path):
    record = FixtureGraphDataset(1)[0]
    obj = build_filtered_simplicial_object(record)
    topo = _toy_topology(intervals=[{"dimension": 0, "birth": 0.0, "death": None, "infinite": True}])
    paths = write_analogical_memory_visualization(
        {
            "bank_path": "",
            "retrieved": [
                {
                    "memory_id": "mem0",
                    "record_id": "rec0",
                    "retrieval_score": 0.9,
                    "filtered_simplicial_object": obj,
                    "topological_algebra": topo,
                }
            ],
        },
        tmp_path,
        query_context={"filtered_simplicial_object": obj, "topological_algebra": topo, "label": "query trajectory"},
    )
    html = Path(paths["analogical_memory_retrieval_html"]).read_text(encoding="utf-8")
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    assert maps["available"] is False
    assert maps["reason"] == "missing_model_probability_query_complex"
    assert maps["reason_detail"].startswith("No model probability filtered query trajectory complex")
    assert maps["topk_contract"]["no_proxy_or_fallback"] is True
    assert maps["topk_contract"]["embedding_only_assignment_allowed"] is False
    assert maps["topk_contract"]["raw_retrieved_count"] == 1
    assert maps["topk_contract"]["qualified_model_probability_memory_count"] == 0
    assert maps["maps"] == []
    assert "analogical_memory_topk_index_html" in paths
    assert "analogical_memory_map_02_html" in paths
    assert "No model probability filtered query trajectory complex was available" in html
    assert "model_probability_jensen_shannon_assignment" not in html


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
        "persistence_representations": {
            "available": True,
            "backend": "gudhi.representations",
            "methods": {
                "0": {
                    "available": True,
                    "landscape": {
                        "num_landscapes": 2,
                        "resolution": 4,
                        "vector": [0.0, 0.2, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0],
                    },
                    "betti_curve": {"values": [0.0, 1.0, 1.0, 0.0]},
                    "silhouette": {"values": [0.0, 0.4, 0.2, 0.0]},
                    "entropy": {"vector": [0.0, 0.1, 0.1, 0.0]},
                    "persistence_lengths": {"values": [0.5, 0.25]},
                    "topological_vector": {"values": [0.2, 0.3, 0.5]},
                    "persistence_image": {"values": [[0.0, 0.2], [0.1, 0.0]]},
                },
                "1": {
                    "available": True,
                    "landscape": {
                        "num_landscapes": 2,
                        "resolution": 4,
                        "vector": [0.0, 0.1, 0.3, 0.0, 0.0, 0.02, 0.0, 0.0],
                    },
                    "betti_curve": {"values": [0.0, 0.8, 0.9, 0.0]},
                    "silhouette": {"values": [0.0, 0.2, 0.5, 0.0]},
                    "entropy": {"vector": [0.0, 0.05, 0.15, 0.0]},
                    "persistence_lengths": {"values": [0.4, 0.22]},
                    "topological_vector": {"values": [0.15, 0.35, 0.5]},
                    "persistence_image": {"values": [[0.0, 0.1], [0.2, 0.0]]},
                },
            },
            "summary": {"landscape_l2_norm": 0.4},
        },
        "commutative_algebra": {
            "multiparameter_chain_presentation_diagnostics": {
                "ring": "F2[x_filtration,x_dimension,x_position]",
                "free_chain_modules": [
                    {"homological_degree": 0, "rank": 2},
                    {"homological_degree": 1, "rank": 1},
                ],
                "real_free_resolution": {
                    "available": False,
                    "safe_to_render_as_multigraded_free_resolution": False,
                    "multigraded_free_resolution_certified": False,
                    "exactness_certified": False,
                },
            }
        },
    }
