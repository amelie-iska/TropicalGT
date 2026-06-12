from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_index_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_sample_browser_index.py"
    spec = importlib.util.spec_from_file_location("build_sample_browser_index_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_sample(root: Path, name: str = "sample_000") -> Path:
    sample = root / name
    step_dir = sample / "reasoning_step_complex_maps"
    step_dir.mkdir(parents=True)
    (sample / "inference_scaling_tree.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"record_id": "q0", "nll": 1.0, "level": 0},
                    {"record_id": "q1", "nll": 0.9, "level": 1, "parent": "q0"},
                ],
                "levels": [[0], [1]],
            }
        ),
        encoding="utf-8",
    )
    (sample / "got_trajectory_pca_3d.html").write_text("<html>nll</html>", encoding="utf-8")
    (sample / "got_full_trajectory_complex_payload.json").write_text("{}", encoding="utf-8")
    for idx in range(3):
        (step_dir / f"reasoning_step_{idx:03d}.html").write_text(f"<html>step {idx}</html>", encoding="utf-8")
        (step_dir / f"reasoning_step_{idx:03d}_simplex_tree.html").write_text(
            f"<html>tree {idx}</html>",
            encoding="utf-8",
        )
    return sample


def test_build_index_writes_catalog_and_prominent_link(tmp_path: Path):
    mod = _load_index_module()
    _write_minimal_sample(tmp_path)
    output = tmp_path / "browser_index.html"

    mod.build_index(tmp_path, output, "Audit")

    catalog = tmp_path / "interactive_visualization_catalog.html"
    assert catalog.exists()
    html = output.read_text(encoding="utf-8")
    assert "open complete artifact catalog" in html
    assert "interactive_visualization_catalog.html" in html
    catalog_html = catalog.read_text(encoding="utf-8")
    assert "sample_000/reasoning_step_complex_maps/reasoning_step_000.html" in catalog_html
    assert "sample_000/got_full_trajectory_complex_payload.json" in catalog_html


def test_step_artifacts_are_collapsed_to_rollup_links(tmp_path: Path):
    mod = _load_index_module()
    _write_minimal_sample(tmp_path)
    output = tmp_path / "browser_index.html"

    mod.build_index(tmp_path, output, "Audit")

    html = output.read_text(encoding="utf-8")
    assert "All reasoning step complexes" in html
    assert "All reasoning step simplex trees" in html
    assert "reasoning_step_complex_maps/complexes_catalog.html" in html
    assert "reasoning_step_complex_maps/simplex_trees_catalog.html" in html
    assert "Reasoning step 000 complex" not in html
    assert "Reasoning step 000 simplex tree" not in html
