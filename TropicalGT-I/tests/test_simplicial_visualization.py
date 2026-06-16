import json
from pathlib import Path

import plotly.graph_objects as go
import torch

from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.scaling import apply_reasoning_action
from tropicalgt.simplicial import build_embedding_radius_simplicial_object, build_filtered_simplicial_object
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import (
    _attach_graph_token_direction_overlay,
    _derived_invariant_comparison,
    _gudhi_canonical_complex,
    _has_real_probability_filtration,
    _simplicial_object_svg,
    _simplicial_map_between_complexes,
    _simplicial_panel_items,
    write_analogical_memory_visualization,
    write_graphcg_trajectory_visualization,
    write_got_trajectory_visualization,
    write_persistence_visualizations,
    write_reasoning_visualizations,
    write_tropical_fan_diagnostics,
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
    assert 'aria-label="interactive selected filtered simplicial complex"' in html
    assert '<details class="static-preview">' in html
    assert "Static SVG fallback preview" in html
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
    assert density_cloud_payload["available"] is True
    density_cloud = density_cloud_payload["density_cloud"]
    assert density_cloud["available"] is True
    assert density_cloud["source"] == "actual model-evaluated graph_state PCA anchors and measured raw NLL values"
    assert density_cloud["support_samples_are_not_model_states"] is True
    assert density_cloud["exact_anchor_layer"] is True
    assert density_cloud["anchor_count"] == len(scaling["candidates"])
    assert density_cloud["local_nll_rule"].startswith("kernel-weighted mean")
    assert density_cloud["density_volume"]["support_samples_are_not_model_states"] is True
    assert density_cloud_payload["render_contract"] == density_cloud["render_contract"]
    assert density_cloud_payload["density_contract"]["actual_model_anchor_layer"] is True
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
    assert 'aria-label="interactive selected filtered simplicial complex"' in html
    assert '<details class="static-preview">' in html
    assert "Static SVG fallback preview" in html
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
    assert "radius/simplicial edges induced from the same embeddings" in full_complex_html
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
    assert len(step_manifest["steps"]) == 4
    assert [row["record_id"] for row in step_manifest["steps"]] == [node["record_id"] for node in payload["nodes"]]
    first_step = tmp_path / "reasoning_step_complex_maps" / "reasoning_step_000.html"
    assert first_step.exists()
    first_step_html = first_step.read_text(encoding="utf-8")
    assert "Filtration radius" in first_step_html
    assert "play filtration" in first_step_html
    assert "faint directed graph-token overlay" in first_step_html



def test_tropical_fan_diagnostics_unavailable_without_explicit_ideal(tmp_path: Path):
    paths = write_tropical_fan_diagnostics({}, tmp_path)
    payload = json.loads(Path(paths["tropical_fan_diagnostics_payload"]).read_text(encoding="utf-8"))
    markup = Path(paths["tropical_fan_diagnostics"]).read_text(encoding="utf-8")
    assert payload["available"] is False
    assert payload["safe_to_render_as_tropical_fan"] is False
    assert payload["diagnostics"]["status"] == "unavailable_no_model_derived_tropical_ideal"
    assert "support-token proxies" in payload["render_contract"]
    assert "Tropical fan diagnostics unavailable" in markup
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
            "ideal_schema": {"variables": ["x", "y"], "generators": ["x+y+1"]},
            "cas_artifacts": {"raw_tagged_output": "rays=matrix {{1,-1,0},{0,-1,1}}"},
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
    assert "one dimensional cones" in markup
    assert "not a multigraded free-resolution" in markup



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
    assert payload["support_flow_edges"][0]["active_support_probability"] == 0.91
    assert payload["metrics"]["render_contract"].startswith("assignment_matrix is binary model argmax support")
    audit = payload["metrics"]["wall_margin_audit"]
    assert audit["strict_wall_hit_count"] == 1
    assert audit["near_wall_hit_count"] == 2
    assert audit["near_wall_only_count"] == 1
    assert payload["support_flow_edges"][2]["wall_margin_bucket"] == "strict_wall"
    assert payload["support_flow_edges"][3]["wall_margin_bucket"] == "near_wall"
    assert "active support probability" in html
    assert "near-wall hit rate" in html
    assert "strict wall threshold" in html
    assert "Tropical active-support audit" in html
    compact = html.replace(" ", "")
    assert '"showlegend":false' in compact
    assert '"r":190' in compact
    assert "Support frequency and mean selected margin" in html


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
    assert payload["metrics"]["raw_token_labels_truncated"] is True
    assert "Tropical active-support collapse diagnostic" in html
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
    assert "basis=effective_full_rank_qr" in html
    assert "heatmap shows all" in html
    assert "Readable full-rank heatmap" in html
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
    assert "retrieval-side probability-map score contribution" in index_html
    assert "vectorized GUDHI family" in index_html
    assert "BettiCurve, Silhouette" in index_html
    assert "landscape contrib." in index_html
    assert "vector contrib." in index_html
    assert "rank 2" in rank2_html
    assert '<input id="filtration-slider"' in rank2_html
    assert len(maps["maps"]) == 2
    assert maps["maps"][1]["pair_page"].endswith("analogical_memory_map_02.html")
    assert not Path(maps["maps"][0]["pair_page"]).is_absolute()
    assert not Path(maps["maps"][1]["pair_page"]).is_absolute()
    assert maps["maps"][0]["edge_preservation_rate"] >= 0.0
    assert maps["maps"][0]["map_source"] == "model_probability_jensen_shannon_assignment"
    assert maps["maps"][0]["query_complex_source"] == "trajectory_probability_filtered_simplicial_object"
    assert maps["maps"][0]["codomain_complex_source"] == "trajectory_probability_filtered_simplicial_object"
    assert maps["maps"][0]["simplicial_map_certificate"]["source"] == "finite_filtered_complex_check"
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
    assert mismatch["derived_equivalence_claim"] == "compatible_finite_invariant_witness"
    assert "do not match" in mismatch["free_resolution_similarity_interpretation"]


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
    assert maps["maps"] == []


def test_analogical_memory_without_retrieval_emits_unavailable_surfaces(tmp_path: Path):
    paths = write_analogical_memory_visualization({"bank_path": "", "retrieved": []}, tmp_path, query_context={})
    maps = json.loads(Path(paths["analogical_simplicial_maps"]).read_text(encoding="utf-8"))
    index_html = Path(paths["analogical_memory_topk_index_html"]).read_text(encoding="utf-8")
    map_html = Path(paths["analogical_memory_map_02_html"]).read_text(encoding="utf-8")
    assert maps == {"available": False, "reason": "no_non_self_model_memory", "maps": []}
    assert "Analogical top-k probability correspondences" in index_html
    assert "No retrieved memories" in index_html
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
    assert maps == {"available": False, "reason": "missing_model_probability_query_complex", "maps": []}
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
