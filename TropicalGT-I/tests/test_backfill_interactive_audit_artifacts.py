import importlib.util
import json
from pathlib import Path


def _load_backfill():
    path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_interactive_audit_artifacts.py"
    spec = importlib.util.spec_from_file_location("backfill_interactive_audit_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backfill_writes_unavailable_fan_and_bifiltration_visual_contract(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    raw = audit / "trajectory_level_radius_bifiltration.json"
    raw.write_text(
        json.dumps(
            {
                "available": True,
                "coefficient_ring": "F2[x_level,x_radius]",
                "num_parameters": 2,
                "parameters": [{"name": "trajectory_level"}, {"name": "radius"}],
                "grid_axes": [[0, 1], [0, 1]],
                "levels": [0, 1],
                "radii": [0.0, 0.5],
                "radius_grade_policy": "exact_sorted_radius_grid_index_no_bucket_collision",
                "fiber_rank_profile": [
                    {"grade": [0, 0], "betti": {"0": 1, "1": 0}},
                    {"grade": [1, 1], "betti": {"0": 1, "1": 1}},
                ],
                "chain_module_generators": [
                    {"multidegree": [0, 0], "homological_degree": 0, "simplex": ["root"]},
                    {"multidegree": [1, 1], "homological_degree": 1, "simplex": ["a", "b"]},
                ],
                "structure_maps": [],
                "chain_presentation_diagnostics": {"ring": "F2[x_level,x_radius]"},
            }
        ),
        encoding="utf-8",
    )
    report = module.backfill_audit_root(audit)
    kinds = {row["kind"] for row in report["actions"]}
    assert "tropical_fan_unavailable_backfill" in kinds
    assert "two_parameter_bifiltration_visual_contract_backfill" in kinds
    fan = json.loads((audit / "tropical_fan_diagnostics.json").read_text(encoding="utf-8"))
    assert fan["available"] is False
    assert fan["safe_to_render_as_tropical_fan"] is False
    visual = json.loads((audit / "trajectory_persistence" / "two_parameter_bifiltration.json").read_text(encoding="utf-8"))
    assert visual["primary_view"] == "miller_sturmfels_bivariate_staircase"
    assert visual["rank_surface_primary"] is False
    assert "Backfills only explicit unavailable diagnostics" in report["policy"]
