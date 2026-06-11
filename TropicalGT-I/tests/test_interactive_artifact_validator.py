from __future__ import annotations

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
    return f"<!doctype html><title>{title}</title><body>{title} {extra}</body>"


def _row(root: Path, name: str) -> Path:
    row = root if name == "." else root / name
    row.mkdir(parents=True, exist_ok=True)
    candidates = [
        {"record_id": "root", "path": [], "parent": None, "level": 0, "nll": 1.0, "embedding": [0.0, 0.0, 0.0]},
        {"record_id": "a", "path": ["expand"], "parent": "root", "level": 1, "nll": 0.9, "embedding": [1.0, 0.0, 0.0]},
        {"record_id": "b", "path": ["verify"], "parent": "root", "level": 1, "nll": 1.1, "embedding": [0.0, 1.0, 0.0]},
        {"record_id": "c", "path": ["expand", "refine"], "parent": "a", "level": 2, "nll": 0.8, "embedding": [1.0, 1.0, 0.0]},
    ]
    nodes = [
        {
            "record_id": row["record_id"],
            "level": row["level"],
            "nll": row["nll"],
            "embedding": row["embedding"],
            "embedding_pca": {"pc1": row["embedding"][0], "pc2": row["embedding"][1], "pc3": row["embedding"][2]},
            "filtered_simplicial_object": {"summary": {"num_vertices": 1, "num_edges": 0, "num_two_simplices": 0}},
        }
        for row in candidates
    ]
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
            "surface_kind": "exact_delaunay_nll_mesh",
            "max_point_residual": 0.0,
        },
    }
    _write(row / "inference_scaling_tree.json", json.dumps({"stochastic_actions": True, "sampling_temperature": 2.0, "sampling_exploration": 0.4, "candidates": candidates}))
    _write(row / "got_trajectory_payloads.json", json.dumps(payload))
    _write(row / "got_embedding_map_payloads.json", json.dumps({"coordinate_source": "PCA of model graph_state embeddings; no level/tree layout coordinates are used", "nodes": nodes}))
    _write(
        row / "got_full_trajectory_complex_payload.json",
        json.dumps({"filtered_simplicial_object": {"summary": {"num_vertices": 4, "num_edges": 3, "num_two_simplices": 0}, "simplices": []}}),
    )
    steps = [{"file": f"reasoning_step_{idx:03d}.html", "summary": {"num_vertices": 1}} for idx in range(4)]
    _write(row / "reasoning_step_complex_maps/manifest.json", json.dumps({"steps": steps}))
    _write(row / "analogical_simplicial_maps.json", json.dumps({"maps": [
        {"codomain_complex_source": "trajectory_filtered_simplicial_object", "edge_preservation_rate": 0.25},
        {"codomain_complex_source": "trajectory_filtered_simplicial_object", "edge_preservation_rate": 0.5},
    ]}))
    _write(row / "inference_audit.json", "{}")
    html_files = {
        "got_embedding_map_3d.html": _html("Graph-of-thought embedding-space trajectory map actual graph_state PCA"),
        "got_trajectory_pca_3d.html": _html("Graph-of-thought branching trajectory NLL"),
        "got_full_trajectory_complex.html": _html("Full graph-of-thought trajectory filtered simplicial complex", "Plotly.newPlot play filtration Filtration radius model input model output"),
        "reasoning_step_complex_maps/index.html": _html("Reasoning step filtered simplicial complex maps", "table"),
        "tropical_support_heatmap.html": _html("Tropical active support", "Plotly.newPlot observed supports only top-support collapse rate"),
        "graphcg_direction_cosines.html": _html("GraphCG direction cosines"),
        "analogical_memory_topk_index.html": _html("Analogical top-k retrieval", "table"),
        "analogical_memory_map_02.html": _html("Analogical simplicial map trajectory-complex map", "Plotly.newPlot slider filters both complexes sliders"),
        "trajectory_persistence/persistence_barcode.html": _html("Trajectory persistence barcode"),
        "trajectory_persistence/persistence_module_betti.html": _html("Trajectory persistence Betti"),
    }
    for rel, content in html_files.items():
        _write(row / rel, content)
    for step in steps:
        _write(row / "reasoning_step_complex_maps" / step["file"], _html("Reasoning step filtered simplicial complex map"))
    return row


def test_validate_audit_root_accepts_three_interactive_rows(tmp_path: Path):
    validator = _load_validator()
    audit = tmp_path / "step_00000001" / "got_audit"
    _row(audit, ".")
    _row(audit, "example_01")
    _row(audit, "example_02")
    _write(tmp_path / "step_00000001" / "validation_report.json", json.dumps({"bpb": 1.5, "graph_bpb": 2.5, "invalid_graph_rate": 0.0}))
    report = validator.validate_audit_root(audit, min_rows=3, min_candidates=4, min_depth=2)
    assert report["ok"], report["errors"]
    assert report["rows_checked"] == 3
    assert report["validation_metrics"]["bpb"] == 1.5
    assert all(row["step_complex_maps"] == 4 for row in report["row_reports"])


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
