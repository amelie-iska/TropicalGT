from __future__ import annotations

import html
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_interactive_audit_artifacts.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_interactive_audit_artifacts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _html(title: str, extra: str = "Plotly.newPlot play filtration Filtration radius") -> str:
    return f"<!doctype html><title>{title}</title><script src='plotly.min.js'></script><body>{title} {extra}</body>"


def _row(root: Path, name: str) -> Path:
    row = root if name == "." else root / name
    row.mkdir(parents=True, exist_ok=True)
    candidates = [
        {"record_id": "root", "path": [], "parent": None, "level": 0, "nll": 1.0, "embedding": [0.0, 0.0, 0.0]},
        {"record_id": "a", "path": ["expand"], "parent": "root", "level": 1, "nll": 0.9, "embedding": [1.0, 0.0, 0.0]},
        {"record_id": "b", "path": ["verify"], "parent": "root", "level": 1, "nll": 1.1, "embedding": [0.0, 1.0, 0.0]},
        {"record_id": "c", "path": ["expand", "refine"], "parent": "a", "level": 2, "nll": 0.8, "embedding": [1.0, 1.0, 0.0]},
    ]
    nodes = []
    for idx, candidate in enumerate(candidates):
        nodes.append(
            {
                "record_id": candidate["record_id"],
                "level": candidate["level"],
                "nll": candidate["nll"],
                "embedding": candidate["embedding"],
                "embedding_pca": {"pc1": candidate["embedding"][0], "pc2": candidate["embedding"][1], "pc3": candidate["embedding"][2]},
                "plot": {
                    "x": candidate["embedding"][0],
                    "y": candidate["embedding"][1],
                    "z": candidate["nll"],
                    "z_surface": candidate["nll"],
                    "z_centered_scaled_nll": candidate["nll"],
                    "raw_centered_scaled_nll": candidate["nll"],
                    "raw_nll": candidate["nll"],
                    "touches_nll_surface": True,
                },
                "reasoning_step_index": idx,
                "step_complex_href": f"reasoning_step_complex_maps/reasoning_step_{idx:03d}.html",
                "step_simplex_tree_href": f"reasoning_step_complex_maps/reasoning_step_{idx:03d}_simplex_tree.html",
                "step_complex_contract": "this GoT state maps to a per-step interactive filtered simplicial complex page and simplex-tree page",
                "filtered_simplicial_object": {
                    "summary": {"num_vertices": 1, "num_edges": 0, "num_two_simplices": 0},
                    "simplices": [{"simplex": [candidate["record_id"]], "dimension": 0, "filtration": 0.0}],
                },
            }
        )
    payload = {
        "nodes": nodes,
        "edges": [
            {"source": "root", "target": "a"},
            {"source": "root", "target": "b"},
            {"source": "a", "target": "c"},
        ],
        "embedding_pca_diagnostics": {
            "coordinate_source": "model graph_state embeddings",
            "n_samples": 4,
            "embedding_dim": 3,
            "pairwise_distance_correlation": 1.0,
            "normalized_stress": 0.0,
            "explained_variance_ratio_sum3": 1.0,
        },
        "nll_surface": {
            "available": True,
            "touches_points": True,
            "surface_kind": "sample_supported_local_idw_surface",
            "max_point_residual": 0.0,
            "z_axis": "projected_nll_fitness_energy",
            "raw_nll_range": 0.3,
            "exact_anchor_layer": True,
            "sparse_observed_anchor_layer": True,
            "actual_landscape_layer": False,
            "support_radius": 0.5,
            "surface_contact_contract": "every rendered GoT state marker and trajectory edge endpoint uses plot.z/plot.z_surface sampled from the displayed NLL energy surface",
            "trajectory_point_surface_residual_max": 0.0,
            "surface_projected_z_by_record_id": {row["record_id"]: row["nll"] for row in candidates},
            "local_interpolating_sheet": {"available": False, "reason": "disabled_to_preserve_exact_reasoning_point_surface_contact"},
            "surrogate_landscape_layer": {"available": False, "reason": "disabled_by_default_not_model_evaluated"},
        },
        "nll_progress": {
            "edge_count": 3,
            "improving_edge_fraction": 2 / 3,
            "mean_edge_delta": -0.0666666667,
            "best_terminal_improvement_from_root": 0.2,
            "by_level": [
                {"level": 0, "count": 1, "mean_nll": 1.0},
                {"level": 1, "count": 2, "mean_nll": 1.0},
                {"level": 2, "count": 1, "mean_nll": 0.8},
            ],
        },
    }
    _write(row / "inference_scaling_tree.json", json.dumps({"stochastic_actions": True, "sampling_temperature": 2.0, "sampling_exploration": 0.4, "candidates": candidates}))
    _write(row / "got_trajectory_payloads.json", json.dumps(payload))
    _write(
        row / "got_embedding_map_payloads.json",
        json.dumps(
            {
                "coordinate_source": "PCA of model graph_state embeddings; no level/tree layout coordinates are used",
                "nodes": nodes,
                "filtered_simplicial_objects": [node["filtered_simplicial_object"] for node in nodes],
            }
        ),
    )
    _write(
        row / "got_full_trajectory_complex_payload.json",
        json.dumps(
            {
                "filtered_simplicial_object": {
                    "summary": {"num_vertices": 4, "num_edges": 3, "num_two_simplices": 0},
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "dimension": 1, "num_simplices": 7},
                    "simplices": [
                        {"simplex": ["root"], "dimension": 0, "embedding": [0, 0, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["a"], "dimension": 0, "embedding": [1, 0, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["b"], "dimension": 0, "embedding": [0, 1, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["c"], "dimension": 0, "embedding": [1, 1, 0], "input_text": "i", "decoded_argmax": "o"},
                        {"simplex": ["root", "a"], "dimension": 1},
                        {"simplex": ["root", "b"], "dimension": 1},
                        {"simplex": ["a", "c"], "dimension": 1},
                    ],
                },
                "probability_filtered_simplicial_object": {
                    "available": True,
                    "summary": {
                        "num_vertices": 4,
                        "num_edges": 3,
                        "num_two_simplices": 0,
                        "filtration_model": "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton",
                    },
                    "simplex_tree": {"backend": "gudhi.SimplexTree", "dimension": 1, "num_simplices": 7},
                    "simplices": [
                        {"simplex": ["root"], "dimension": 0, "model_probability_vector": [0.7, 0.2, 0.1]},
                        {"simplex": ["a"], "dimension": 0, "model_probability_vector": [0.2, 0.7, 0.1]},
                        {"simplex": ["b"], "dimension": 0, "model_probability_vector": [0.2, 0.1, 0.7]},
                        {"simplex": ["c"], "dimension": 0, "model_probability_vector": [0.4, 0.4, 0.2]},
                        {"simplex": ["root", "a"], "dimension": 1},
                        {"simplex": ["root", "b"], "dimension": 1},
                        {"simplex": ["a", "c"], "dimension": 1},
                    ],
                }
            }
        ),
    )
    steps = [
        {
            "record_id": candidates[idx]["record_id"],
            "file": f"reasoning_step_{idx:03d}.html",
            "simplex_tree_file": f"reasoning_step_{idx:03d}_simplex_tree.html",
            "summary": {"num_vertices": 1},
            "simplex_tree": {"backend": "gudhi.SimplexTree"},
        }
        for idx in range(4)
    ]
    _write(row / "reasoning_step_complex_maps/manifest.json", json.dumps({"steps": steps}))
    _write(row / "analogical_simplicial_maps.json", json.dumps({"maps": [
        {
            "query_complex_source": "trajectory_probability_filtered_simplicial_object",
            "codomain_complex_source": "trajectory_probability_filtered_simplicial_object",
            "map_source": "model_probability_jensen_shannon_assignment",
            "is_identity_self_map": False,
            "pair_page": "analogical_memory_retrieval.html",
            "jensen_shannon_distance_mean": 0.12,
            "assignment_cost_mean": 0.17,
            "jensen_shannon_distance_summary": {"count": 2, "min": 0.1, "max": 0.14, "mean": 0.12, "std": 0.02},
            "assignment_cost_summary": {"count": 2, "min": 0.15, "max": 0.19, "mean": 0.17, "std": 0.02},
            "filtration_distortion_summary": {"count": 1, "min": 0.0, "max": 0.07, "mean": 0.07, "std": 0.0},
            "edge_preservation_rate": 0.25,
            "domain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "codomain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "displayed_domain_vertices": 2,
            "displayed_codomain_vertices": 2,
            "is_simplicial_on_displayed_skeleton": False,
            "preserved_edge_pairs": [{"query_edge": ["a", "b"], "memory_edge": ["x", "y"]}],
            "failed_edge_pairs": [{"query_edge": ["b", "c"], "memory_edge": ["y", "z"]}],
            "preserved_edge_query_vertices": ["a", "b"],
        },
        {
            "query_complex_source": "trajectory_probability_filtered_simplicial_object",
            "codomain_complex_source": "trajectory_probability_filtered_simplicial_object",
            "map_source": "model_probability_jensen_shannon_assignment",
            "is_identity_self_map": False,
            "pair_page": "analogical_memory_map_02.html",
            "jensen_shannon_distance_mean": 0.23,
            "assignment_cost_mean": 0.27,
            "jensen_shannon_distance_summary": {"count": 2, "min": 0.2, "max": 0.26, "mean": 0.23, "std": 0.03},
            "assignment_cost_summary": {"count": 2, "min": 0.24, "max": 0.3, "mean": 0.27, "std": 0.03},
            "filtration_distortion_summary": {"count": 1, "min": 0.0, "max": 0.02, "mean": 0.02, "std": 0.0},
            "edge_preservation_rate": 0.5,
            "domain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "codomain_simplex_tree": {"backend": "gudhi.SimplexTree"},
            "displayed_domain_vertices": 2,
            "displayed_codomain_vertices": 2,
            "is_simplicial_on_displayed_skeleton": False,
            "preserved_edge_pairs": [{"query_edge": ["a", "b"], "memory_edge": ["x", "y"]}],
            "failed_edge_pairs": [],
            "preserved_edge_query_vertices": ["a", "b"],
        },
    ]}))
    _write(
        row / "tropical_support_payload.json",
        json.dumps(
            {
                "metrics": {
                    "available": True,
                    "token_count": 4,
                    "unique_support_count": 1,
                    "effective_supports": 1.0,
                    "support_entropy_bits": 0.0,
                    "top_support_collapse_rate": 1.0,
                    "margin_summary": {"min": 0.1, "max": 0.4, "mean": 0.25, "std": 0.1, "p05": 0.1, "p50": 0.25, "p95": 0.4},
                    "interpretation": "Uniform blocks indicate true active-support collapse or nearly constant margins.",
                },
                "support_flow_edges": [
                    {"query_index": idx, "query_label": f"q{idx}", "support_index": 0, "support_label": "q0", "margin": 0.1 * (idx + 1)}
                    for idx in range(4)
                ],
            }
        ),
    )
    _write(
        row / "graphcg_direction_cosines_payload.json",
        json.dumps(
            {
                "available": True,
                "matrix_shape": [4, 8],
                "display_count": 4,
                "full_rank_direction_count": 8,
                "active_rank_nonzero_mean_abs": 8,
                "candidate_effective_direction_count": [4.0, 4.0, 4.0, 4.0],
                "direction_activity_sorted": [0.2 for _ in range(8)],
                "projection_basis_certificate": {
                    "source": "candidate.graphcg_projection",
                    "available": True,
                    "projection_basis": "effective_full_rank_qr",
                    "basis_sources": ["effective_full_rank_qr"],
                    "basis_source_counts": {"effective_full_rank_qr": 4},
                    "candidate_count": 4,
                    "direction_count": 8,
                    "all_candidates_have_all_direction_cosines": True,
                    "mean_abs_offdiag_cosine_values": [0.01, 0.01, 0.01, 0.01],
                    "max_abs_offdiag_cosine_values": [0.02, 0.02, 0.02, 0.02],
                    "mean_abs_offdiag_cosine_max": 0.01,
                    "max_abs_offdiag_cosine_max": 0.02,
                },
                "interpretation": "Heatmap colors encode absolute cosine activity; signed cosine values are preserved in hover.",
            }
        ),
    )
    _write(
        row / "trajectory_level_radius_bifiltration.json",
        json.dumps(
            {
                "available": True,
                "coefficient_ring": "F2[x_level,x_radius]",
                "num_parameters": 2,
                "parameters": [
                    {"name": "trajectory_level", "meaning": "reasoning growth level"},
                    {"name": "radius", "meaning": "scalar radius threshold"},
                ],
                "grid_axes": [[0, 1, 2], [0, 1, 2]],
                "levels": [0, 1, 2],
                "radii": [0.0, 0.5, 1.0],
                "radius_grade_policy": "exact_sorted_radius_grid_index_no_bucket_collision",
                "radius_grade_values": {"0": 0.0, "1": 0.5, "2": 1.0},
                "rank_invariant_samples": [
                    {"source_grade": [0, 0], "target_grade": [0, 1], "h0_rank": 1},
                    {"source_grade": [0, 0], "target_grade": [1, 1], "h0_rank": 1},
                ],
                "boundary_monomials": {"d1": [{"monomial": "x_level", "monomial_exponent": [1, 0]}]},
                "chain_presentation_diagnostics": {
                    "ring": "F2[x_level,x_radius]",
                    "real_free_resolution_certified": False,
                    "resolution_status": "computed finite presentation; minimality not certified",
                },
            }
        ),
    )
    _write(row / "inference_audit.json", "{}")
    html_files = {
        "got_embedding_map_3d.html": _html(
            "Graph-of-thought embedding-space trajectory map actual graph_state PCA",
            "Plotly.newPlot simplicial-object-panel simplicial-object-plot selected-complex-graph hover-simplicial-card plotly_click",
        ),
        "got_trajectory_pca_3d.html": _html(
            "Graph-of-thought branching trajectory with observed NLL anchors",
            "Plotly.newPlot selected-complex-graph plotly_click open interactive reasoning-step complex page",
        ),
        "got_nll_density_cloud_pca_3d.html": _html("3D PCA NLL density cloud", "Gaussian cloud actual model GoT state anchors not a model state Plotly.newPlot plotly.min.js"),
        "got_full_trajectory_complex.html": _html("Full graph-of-thought trajectory filtered simplicial complex", "Plotly.newPlot play filtration min-to-max Filtration radius model input model output filtration backend= simplicial-object-plot selected-complex-graph plotly_click"),
        "got_full_trajectory_simplex_tree_3d.html": _html("Full graph-of-thought trajectory GUDHI SimplexTree face-coface poset", "Plotly.newPlot face-coface poset view not a literal trie layout"),
        "got_full_trajectory_complex_jensen_shannon.html": _html("Full graph-of-thought trajectory probability filtered simplicial complex", "Plotly.newPlot Jensen-Shannon probability filtered simplicial complex"),
        "got_full_trajectory_simplex_tree_3d_jensen_shannon.html": _html("Full graph-of-thought trajectory probability SimplexTree", "Plotly.newPlot Jensen-Shannon probability SimplexTree"),
        "reasoning_step_complex_maps/index.html": _html("Reasoning step filtered simplicial complex maps", "table"),
        "tropical_support_heatmap.html": _html("Tropical active support", "Plotly.newPlot observed supports only top-support collapse rate"),
        "graphcg_direction_cosines.html": _html("GraphCG full-rank direction audit", "Plotly.newPlot Readable top-direction heatmap"),
        "analogical_memory_topk_index.html": "<!doctype html><title>Analogical top-k probability correspondences</title><body>Analogical top-k probability correspondences <a href='analogical_memory_retrieval.html'>rank 1</a> <a href='analogical_memory_map_02.html'>rank 2</a></body>",
        "analogical_memory_retrieval.html": _html("Analogical probability-matched correspondence filtered-complex certificate", "Plotly.newPlot query trajectory complex retrieved memory complex slider filters domain and codomain sliders vertex-only correspondences preserved 1-simplex map simplicial-object-plot selected-complex-graph plotly_click"),
        "analogical_memory_map_02.html": _html("Analogical probability-matched correspondence filtered-complex certificate", "Plotly.newPlot query trajectory complex retrieved memory complex slider filters domain and codomain sliders vertex-only correspondences preserved 1-simplex map simplicial-object-plot selected-complex-graph plotly_click"),
        "trajectory_persistence/persistence_barcode.html": _html("Trajectory persistence barcode", "Plotly.newPlot simplicial-object-plot selected-complex-graph plotly_click"),
        "trajectory_persistence/two_parameter_bifiltration.html": _html("Trajectory 2-parameter persistence over F2[x_level,x_radius]", "Plotly.newPlot 2-parameter module fibers Miller-Sturmfels staircase H0 fiber rank"),
        "trajectory_persistence/persistence_module_betti.html": _html("Trajectory persistence Betti", "Plotly.newPlot 2D matrix decorative 3D simplicial-object-plot selected-complex-graph plotly_click"),
        "trajectory_persistence/persistence_representations.html": _html("Trajectory GUDHI persistence vectorization", "Plotly.newPlot Fast train/eval features"),
        "trajectory_persistence/persistence_landscapes.html": _html("Trajectory Actual GUDHI persistence landscape functions", "Plotly.newPlot lambda_1(t) not norm-only summaries"),
    }
    for rel, content in html_files.items():
        _write(row / rel, content)
    for step in steps:
        _write(row / "reasoning_step_complex_maps" / step["file"], _html("Reasoning step filtered simplicial complex map"))
        _write(row / "reasoning_step_complex_maps" / step["simplex_tree_file"], _html("Reasoning step GUDHI simplex tree", "Plotly.newPlot simplex-tree inclusion"))
    return row

def _browser_samples(audit: Path, sample_names: list[str]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for idx, name in enumerate(sample_names):
        sample_dir = audit if name == "." else audit / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write(sample_dir / "browser_index.html", "<!doctype html><a href='got_trajectory_pca_3d.html'>trajectory</a>")
        prefix = "" if name == "." else f"{name}/"
        samples.append(
            {
                "index": idx,
                "label": "root sample" if name == "." else name,
                "dir": name,
                "artifacts": [
                    {"src": f"{prefix}got_trajectory_pca_3d.html", "label": "Trajectory", "tag": "plot"},
                    {"src": f"{prefix}analogical_memory_topk_index.html", "label": "Analogical top-k", "tag": "index"},
                ],
            }
        )
    return samples


def _codex_browser_html(samples: list[dict[str, object]], *, omit_button_src: str | None = None) -> str:
    sample_sections = []
    for sample in samples:
        idx = str(sample["index"])
        sample_dir = str(sample["dir"])
        open_href = "browser_index.html" if sample_dir == "." else f"{sample_dir}/browser_index.html"
        buttons = []
        for artifact in sample["artifacts"]:
            src = str(artifact["src"])
            if src == omit_button_src:
                continue
            label = str(sample["label"]) + " / " + str(artifact["label"])
            buttons.append(
                f'<button class="artifact" data-sample="{html.escape(idx, quote=True)}" '
                f'data-src="{html.escape(src, quote=True)}" '
                f'data-label="{html.escape(label, quote=True)}">'
                f'<span>{html.escape(str(artifact["label"]))}</span></button>'
            )
        sample_sections.append(
            f'<section class="sample" data-sample="{html.escape(idx, quote=True)}">'
            f'<a class="open-sample" href="{html.escape(open_href, quote=True)}">open sample</a>'
            f'{"".join(buttons)}</section>'
        )
    first_src = str(samples[0]["artifacts"][0]["src"])
    payload = html.escape(json.dumps(samples), quote=True)
    return f'<!doctype html><body data-samples="{payload}">Sample-first audit {"".join(sample_sections)}<a id="open" href="{html.escape(first_src, quote=True)}">open full page</a><iframe src="{html.escape(first_src, quote=True)}"></iframe></body>'


def test_validate_audit_root_accepts_three_interactive_rows(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    _row(audit, ".")
    _row(audit, "example_01")
    _row(audit, "example_02")
    samples = _browser_samples(audit, [".", "example_01", "example_02"])
    _write(audit / "codex_browser_index.html", _codex_browser_html(samples))
    _write(tmp_path / "step_00000001" / "validation_report.json", json.dumps({"bpb": 1.5, "graph_bpb": 2.5, "invalid_graph_rate": 0.0}))
    report = validator.validate_audit_root(audit, min_rows=3, min_candidates=4, min_depth=2)
    assert report["ok"], report["errors"]
    assert report["rows_checked"] == 3
    assert report["validation_metrics"]["bpb"] == 1.5
    assert all(row["step_complex_maps"] == 4 for row in report["row_reports"])


def test_validate_audit_root_rejects_absolute_analogical_pair_pages(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    maps_path = row / "analogical_simplicial_maps.json"
    payload = json.loads(maps_path.read_text(encoding="utf-8"))
    payload["maps"][0]["pair_page"] = str(row / "analogical_memory_retrieval.html")
    maps_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("relative pair_page links" in err for err in report["errors"])


def test_validate_audit_root_rejects_analogical_probability_js_provenance_gaps(tmp_path: Path):
    validator = _load_validator()
    cases = [
        (
            "wrong_query_source",
            lambda row: row.__setitem__("query_complex_source", "trajectory_embedding_filtered_simplicial_object"),
            "query trajectory-level model-probability complexes",
        ),
        (
            "wrong_codomain_source",
            lambda row: row.__setitem__("codomain_complex_source", "trajectory_embedding_filtered_simplicial_object"),
            "codomain trajectory-level model-probability complexes",
        ),
        (
            "wrong_map_source",
            lambda row: row.__setitem__("map_source", "embedding_nearest_neighbor_assignment"),
            "derived from model probability vectors",
        ),
        (
            "missing_js_summary",
            lambda row: row.pop("jensen_shannon_distance_summary"),
            "missing Jensen-Shannon distance summaries",
        ),
        (
            "missing_domain_simplex_tree_provenance",
            lambda row: row.__setitem__("domain_simplex_tree", {"backend": "networkx"}),
            "missing domain GUDHI SimplexTree provenance",
        ),
    ]
    for case_name, mutate, expected in cases:
        audit = tmp_path / case_name / "got_audit"
        row = _row(audit, ".")
        maps_path = row / "analogical_simplicial_maps.json"
        payload = json.loads(maps_path.read_text(encoding="utf-8"))
        mutate(payload["maps"][0])
        maps_path.write_text(json.dumps(payload), encoding="utf-8")
        _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
        report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
        assert not report["ok"], case_name
        assert any(expected in err for err in report["errors"]), (case_name, report["errors"])


def test_validate_audit_root_rejects_missing_graphcg_basis_certificate(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    row = _row(audit, ".")
    payload_path = row / "graphcg_direction_cosines_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("projection_basis_certificate")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(audit / "codex_browser_index.html", _codex_browser_html(_browser_samples(audit, ["."])))
    report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("projection-basis certificate" in err for err in report["errors"])


def test_codex_browser_index_requires_artifact_button_for_each_payload_item(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    _row(audit, ".")
    _row(audit, "example_01")
    _row(audit, "example_02")
    samples = _browser_samples(audit, [".", "example_01", "example_02"])
    missing_src = str(samples[1]["artifacts"][0]["src"])
    _write(audit / "codex_browser_index.html", _codex_browser_html(samples, omit_button_src=missing_src))
    report = validator.validate_audit_root(audit, min_rows=3, min_candidates=4, min_depth=2)
    assert not report["ok"]
    assert any("missing artifact button" in err and missing_src in err for err in report["errors"])


def test_validate_bundle_root_accepts_sample_directories(tmp_path: Path):
    validator = _load_validator()
    bundle = tmp_path / "multi_sample_browser" / "latest"
    _row(bundle, "sample_000")
    _row(bundle, "sample_001")
    _write(bundle / "browser_index.html", "<!doctype html><body>sample bundle</body>")
    report = validator.validate_audit_root(bundle, min_rows=2, min_candidates=4, min_depth=2)
    assert report["ok"], report["errors"]
    assert report["rows_checked"] == 2


def test_validate_audit_root_rejects_broken_nll_surface_contact_fields(tmp_path: Path):
    validator = _load_validator()
    cases = [
        ("missing_projected_map", lambda payload: payload["nll_surface"].pop("surface_projected_z_by_record_id"), "missing per-record projected z values"),
        ("touch_flag_false", lambda payload: payload["nodes"][0]["plot"].__setitem__("touches_nll_surface", False), "not marked as touching"),
        ("missing_z", lambda payload: payload["nodes"][0]["plot"].pop("z"), "missing finite plotted z"),
        ("missing_z_surface", lambda payload: payload["nodes"][0]["plot"].pop("z_surface"), "missing finite z_surface"),
        ("missing_raw_centered", lambda payload: payload["nodes"][0]["plot"].pop("raw_centered_scaled_nll"), "missing raw centered/scaled NLL z"),
        ("projected_z_mismatch", lambda payload: payload["nodes"][0]["plot"].__setitem__("z", 999.0), "plotted z does not equal z_surface"),
        ("projected_z_surface_mismatch", lambda payload: payload["nodes"][0]["plot"].__setitem__("z_surface", 999.0), "plotted z/z_surface does not match the displayed NLL surface projection"),
    ]
    for case_name, mutate, expected in cases:
        audit = tmp_path / case_name / "got_audit"
        _row(audit, ".")
        payload_path = audit / "got_trajectory_payloads.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        mutate(payload)
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        report = validator.validate_audit_root(audit, min_rows=1, min_candidates=4, min_depth=2)
        assert not report["ok"], case_name
        assert any(expected in err for err in report["errors"]), (case_name, report["errors"])


def test_validator_cli_writes_json_and_markdown(tmp_path: Path):
    audit = tmp_path / "step_00000001" / "got_audit"
    _row(audit, ".")
    _row(audit, "example_01")
    _row(audit, "example_02")
    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--audit-root",
            str(audit),
            "--min-rows",
            "3",
            "--min-candidates",
            "4",
            "--min-depth",
            "2",
            "--json-output",
            str(json_out),
            "--markdown-output",
            str(md_out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(json_out.read_text(encoding="utf-8"))["ok"] is True
    assert "Overall status: PASS" in md_out.read_text(encoding="utf-8")
