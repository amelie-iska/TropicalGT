import json
import sys

import torch

import tropicalgt.cas_free_resolution as cas_free_resolution
import tropicalgt.cas_tropical as cas_tropical
import tropicalgt.cas_toric as cas_toric
from tropicalgt.algebra import (
    _bivariate_staircase_resolution_from_points,
    compute_level_radius_bifiltration_report,
    compute_topological_algebra_report,
    summarize_algebra_reports,
)
from tropicalgt.cas_free_resolution import build_singular_script, canonicalize_module, try_compute_real_free_resolution
from tropicalgt.data import FixtureGraphDataset
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
from tropicalgt.records import GraphRecord
from tropicalgt.scaling import run_inference_scaling
from tropicalgt.simplicial import build_filtered_simplicial_object, build_reasoning_trajectory_complex
from tropicalgt.tokenizer import TokenGTTokenizer



def _assert_real_resolution_guard(real, expected_ring):
    assert real["schema_version"] == "tropicalgt.real_free_resolution.v1"
    assert real["module_summary"]["coefficient_ring"] == expected_ring
    for key in (
        "real_free_resolution_certified",
        "total_graded_resolution_certified",
        "ungraded_resolution_certified",
        "multigraded_free_resolution_certified",
        "safe_to_render_as_real_free_resolution",
        "safe_to_render_as_total_graded_resolution",
        "safe_to_render_as_multigraded_free_resolution",
    ):
        assert key in real
    if real["available"]:
        assert real["status"] == "certified"
        assert real["backend"] in {"Macaulay2", "Singular", "sage"}
        assert real["certificate_attached"] is True
        assert real["real_free_resolution_certified"] is True
        assert real["exactness_certified"] is True
        assert real["cas_artifacts"]
        assert real["cas_artifacts"].get("raw_tagged_output")
        summary = real["free_resolution_summary"]
        if summary.get("safe_for_multigraded_claims") is True:
            assert real["multigraded_free_resolution_certified"] is True
            assert real["safe_to_render_as_multigraded_free_resolution"] is True
        else:
            assert real["multigraded_free_resolution_certified"] is False
            assert real["safe_to_render_as_multigraded_free_resolution"] is False
            assert "not a multigraded" in real.get("render_warning", "")
            if expected_ring == "F2[x_level,x_radius]":
                assert real["safe_to_render_as_real_free_resolution"] is False
        if real["total_graded_resolution_certified"]:
            assert real["safe_to_render_as_total_graded_resolution"] is True
            assert summary.get("not_multigraded") is True
        if real["cas_artifacts"].get("betti_table_ungraded", {}).get("available"):
            assert real["cas_artifacts"].get("betti_table_ungraded", {}).get("not_multigraded") is True
    else:
        assert real["status"] in {
            "unavailable_no_certificate",
            "backend_not_installed",
            "certificate_failed",
            "disabled_by_environment",
            "complexity_guard",
            "timeout",
            "parse_error",
            "backend_error",
            "unsupported_ring",
            "invalid_grading",
        }
        assert real["certificate_attached"] is False
        assert real["real_free_resolution_certified"] is False
        assert real["total_graded_resolution_certified"] is False
        assert real["ungraded_resolution_certified"] is False
        assert real["multigraded_free_resolution_certified"] is False
        assert real["minimality_certified"] is False
        assert real["exactness_certified"] is False
        assert real["safe_to_render_as_real_free_resolution"] is False
        assert real["safe_to_render_as_total_graded_resolution"] is False
        assert real["safe_to_render_as_multigraded_free_resolution"] is False
        assert real["cas_artifacts"] == {}


def _small_free_resolution_module():
    return {
        "coefficient_ring": "F2[x_level,x_radius]",
        "chain_module_generators": [
            {"generator_id": "v0", "simplex": ["v0"], "homological_degree": 0, "multidegree": [0, 0]},
            {"generator_id": "e01", "simplex": ["v0", "v1"], "homological_degree": 1, "multidegree": [1, 0]},
            {"generator_id": "e02", "simplex": ["v0", "v2"], "homological_degree": 1, "multidegree": [0, 1]},
        ],
        "boundary_monomials": {
            "d1": [
                {"source_generator_id": "e01", "target_generator_id": "v0", "exponent": [1, 0]},
                {"source_generator_id": "e02", "target_generator_id": "v0", "exponent": [0, 1]},
            ]
        },
    }


def test_real_cas_free_resolution_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("TROPICALGT_DISABLE_CAS_FREE_RESOLUTION", "1")
    real = try_compute_real_free_resolution(_small_free_resolution_module(), timeout_s=1)
    _assert_real_resolution_guard(real, "F2[x_level,x_radius]")
    assert real["available"] is False
    assert real["status"] == "disabled_by_environment"
    assert real["cache"]["enabled"] is False
    assert real["cache"]["hit"] is False
    assert "TROPICALGT_DISABLE_CAS_FREE_RESOLUTION" in real["reason"]
    assert real["command_templates"]["macaulay2"]


def test_cas_backend_probe_reports_detected_executable_paths():
    probe = cas_free_resolution.probe_cas_backends()
    backends = {row["name"]: row for row in probe["backends"]}
    assert set(backends) == {"M2", "Singular", "sage"}
    assert probe["preferred_order"] == ["M2", "sage", "Singular"]
    for name in ("M2", "Singular", "sage"):
        expected = cas_free_resolution._candidate_executable(name)
        assert backends[name]["available"] is (expected is not None)
        assert backends[name]["executable"] == expected
        if expected is not None:
            assert backends[name]["version"]
    bem = cas_free_resolution.probe_bemultipliers()
    assert bem["is_resolution_backend"] is False
    assert "never substitute" in bem["execution_policy"]


def test_bemultipliers_probe_reports_local_macaulay2_loader():
    probe = cas_free_resolution.probe_bemultipliers()
    assert probe["is_resolution_backend"] is False
    assert probe["execution_policy"].startswith("load only after a CAS-certified free resolution")
    assert "macaulay2_loader" in probe
    if probe["local_macaulay2_package"]:
        assert probe["macaulay2_loader"].startswith('load "')
        assert probe["local_macaulay2_package"].endswith("BuchsbaumEisenbudMultipliers.m2")


def test_failed_cas_certificate_preserves_module_provenance_without_artifacts():
    schema = canonicalize_module(_small_free_resolution_module())
    tagged = "\n".join([
        "TROPICALGT_RESOLUTION_BEGIN",
        "backend=Macaulay2",
        "exactness_certified=false",
        "minimality_certified=false",
        "certificate_type=failed smoke certificate",
        "presentation_shape=1x2",
        "betti_table_begin",
        "not a certified table",
        "betti_table_end",
        "TROPICALGT_RESOLUTION_END",
    ])
    parsed = cas_free_resolution._parse_tagged_output(tagged)
    assert parsed is not None
    real = cas_free_resolution._certified_result(
        schema,
        {
            "available": True,
            "backend": "Macaulay2",
            "parsed": parsed,
            "tagged_output": parsed["_raw"],
            "certificate_attached": False,
        },
        attempts=[{"backend": "M2", "status": "ran", "returncode": 0}],
    )
    assert real["available"] is False
    assert real["status"] == "certificate_failed"
    assert real["input_sha256"] == schema["input_sha256"]
    assert real["module_summary"]["generators"] == len(schema["generators"])
    assert real["module_summary"]["boundary_monomials"] == len(schema["boundary_monomials"])
    assert real["cas_artifacts"] == {}
    assert real["certificate_attached"] is False


def test_real_cas_free_resolution_caches_deterministic_unavailable_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("TROPICALGT_CAS_FREE_RESOLUTION_CACHE_DIR", str(tmp_path / "cas-cache"))
    monkeypatch.setattr(cas_free_resolution, "_candidate_executable", lambda name: None)
    module = _small_free_resolution_module()
    first = cas_free_resolution.try_compute_real_free_resolution(module, timeout_s=1)
    second = cas_free_resolution.try_compute_real_free_resolution(module, timeout_s=1)
    _assert_real_resolution_guard(first, "F2[x_level,x_radius]")
    _assert_real_resolution_guard(second, "F2[x_level,x_radius]")
    assert first["status"] == "backend_not_installed"
    assert first["cache"]["enabled"] is True
    assert first["cache"]["hit"] is False
    assert first["cache"]["written"] is True
    assert second["status"] == "backend_not_installed"
    assert second["cache"]["enabled"] is True
    assert second["cache"]["hit"] is True
    assert second["cache"]["key"] == first["cache"]["key"]
    unavailable = first["unavailable_diagnostic"]
    assert unavailable["status"] == "backend_not_installed"
    assert unavailable["safe_to_render_only_as_unavailable"] is True
    assert unavailable["available_backends"] == []
    assert unavailable["attempt_statuses"] == []
    assert unavailable["bemultipliers_is_resolution_backend"] is False
    assert "Install or activate Macaulay2" in unavailable["action"]
    assert first["unavailable_dependency_action"] == unavailable["action"]
    assert "Do not substitute chain diagnostics" in unavailable["no_proxy_policy"]


def test_real_cas_free_resolution_complexity_guard_caches_deterministic_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("TROPICALGT_CAS_FREE_RESOLUTION_CACHE_DIR", str(tmp_path / "cas-cache"))
    monkeypatch.setenv("TROPICALGT_CAS_MAX_PRESENTATION_CELLS", "4")
    monkeypatch.setenv("TROPICALGT_CAS_MAX_DETERMINANT_ORDER", "2")
    monkeypatch.setattr(cas_free_resolution, "_candidate_executable", lambda name: "/bin/false")
    module = {
        "coefficient_ring": "F2[x_level,x_radius]",
        "chain_module_generators": [
            {"generator_id": "v0", "simplex": ["v0"], "homological_degree": 0, "multidegree": [0, 0]},
            {"generator_id": "v1", "simplex": ["v1"], "homological_degree": 0, "multidegree": [0, 0]},
            {"generator_id": "v2", "simplex": ["v2"], "homological_degree": 0, "multidegree": [0, 0]},
            {"generator_id": "e0", "simplex": ["v0", "v1"], "homological_degree": 1, "multidegree": [1, 0]},
            {"generator_id": "e1", "simplex": ["v1", "v2"], "homological_degree": 1, "multidegree": [0, 1]},
            {"generator_id": "e2", "simplex": ["v0", "v2"], "homological_degree": 1, "multidegree": [1, 1]},
        ],
        "boundary_monomials": {
            "d1": [
                {"source_generator_id": "e0", "target_generator_id": "v0", "exponent": [1, 0]},
                {"source_generator_id": "e0", "target_generator_id": "v1", "exponent": [1, 0]},
                {"source_generator_id": "e1", "target_generator_id": "v1", "exponent": [0, 1]},
                {"source_generator_id": "e1", "target_generator_id": "v2", "exponent": [0, 1]},
                {"source_generator_id": "e2", "target_generator_id": "v0", "exponent": [1, 1]},
                {"source_generator_id": "e2", "target_generator_id": "v2", "exponent": [1, 1]},
            ]
        },
    }
    first = cas_free_resolution.try_compute_real_free_resolution(module, timeout_s=1)
    second = cas_free_resolution.try_compute_real_free_resolution(module, timeout_s=1)
    _assert_real_resolution_guard(first, "F2[x_level,x_radius]")
    _assert_real_resolution_guard(second, "F2[x_level,x_radius]")
    assert first["status"] == "complexity_guard"
    assert "complexity limits" in first["reason"]
    assert {attempt["status"] for attempt in first["backend_attempts"]} == {"skipped_complexity_guard"}
    assert all(attempt["presentation_shape"] == [3, 3] for attempt in first["backend_attempts"])
    assert first["cache"]["written"] is True
    assert second["cache"]["hit"] is True


def test_tagged_cas_script_timeout_records_bounded_attempt():
    result = cas_free_resolution._run_tagged_cas_script(
        name="python",
        executable=sys.executable,
        script="import time\nprint('starting slow CAS probe', flush=True)\ntime.sleep(30)\n",
        suffix=".py",
        timeout_s=0.1,
    )
    assert result["available"] is False
    attempt = result["attempt"]
    assert attempt["status"] == "timeout"
    assert attempt["timeout_s"] == 0.1
    assert attempt["stdout_tail"]


def test_singular_certified_result_structures_ungraded_betti_rows_without_multigraded_claims():
    schema = canonicalize_module(_small_free_resolution_module())
    tagged = "\n".join([
        "TROPICALGT_RESOLUTION_BEGIN",
        "backend=Singular",
        "exactness_certified=true",
        "minimality_certified=true",
        "certificate_type=Singular mres image-submodule resolution prepended to the displayed cokernel presentation",
        "presentation_shape=1x2",
        "betti_table_begin",
        "2     1",
        "betti_table_end",
        "singular_resolution_text_begin",
        "1      2      1",
        "singular_resolution_text_end",
        "fitting_ideals_begin",
        "Fitt0=x_level,x_radius",
        "Fitt1=1",
        "fitting_ideals_end",
        "minors_begin",
        "minors_1=x_level,x_radius",
        "minors_end",
        "buchsbaum_eisenbud_diagnostics_begin",
        "backend_diagnostics_available=true",
        "exactness_certified=true",
        "minimality_certified=true",
        "be_exactness_source=Singular mres plus exact determinantal/Fitting ideals for the displayed presentation matrix",
        "multiplier_output_available=false",
        "bemultipliers_status=unsupported_in_singular_adapter",
        "reason=Singular output is not Buchsbaum-Eisenbud multiplier output",
        "buchsbaum_eisenbud_diagnostics_end",
        "TROPICALGT_RESOLUTION_END",
    ])
    parsed = cas_free_resolution._parse_tagged_output(tagged)
    assert parsed is not None
    real = cas_free_resolution._certified_result(
        schema,
        {
            "available": True,
            "backend": "Singular",
            "parsed": parsed,
            "tagged_output": parsed["_raw"],
            "certificate_attached": True,
        },
        attempts=[{"backend": "Singular", "status": "ran"}],
    )
    _assert_real_resolution_guard(real, "F2[x_level,x_radius]")
    assert real["backend"] == "Singular"
    assert real["ungraded_resolution_certified"] is True
    assert real["multigraded_free_resolution_certified"] is False
    assert real["safe_to_render_as_multigraded_free_resolution"] is False
    ungraded = real["cas_artifacts"]["betti_table_ungraded"]
    assert ungraded["available"] is True
    assert ungraded["not_multigraded"] is True
    assert ungraded["safe_for_multigraded_claims"] is False
    assert ungraded["homological_column_ranks"] == [2, 1]
    assert ungraded["free_modules"] == [
        {
            "homological_degree": 0,
            "rank": 2,
            "display": "F_0 = S^2",
            "grading": "ungraded_total_rank",
            "multidegree_shifts_available": False,
        },
        {
            "homological_degree": 1,
            "rank": 1,
            "display": "F_1 = S",
            "grading": "ungraded_total_rank",
            "multidegree_shifts_available": False,
        },
    ]
    assert ungraded["betti_table_rows"] == [
        {
            "homological_degree": 0,
            "matrix_row": 0,
            "multidegree": [],
            "shift_display": "ungraded row 0",
            "rank": 2,
            "multiplicity": 2,
            "grading": "ungraded_total_rank",
            "source": "Singular_betti_matrix",
            "not_multigraded": True,
            "multidegree_shifts_available": False,
            "safe_for_multigraded_claims": False,
        },
        {
            "homological_degree": 1,
            "matrix_row": 0,
            "multidegree": [],
            "shift_display": "ungraded row 0",
            "rank": 1,
            "multiplicity": 1,
            "grading": "ungraded_total_rank",
            "source": "Singular_betti_matrix",
            "not_multigraded": True,
            "multidegree_shifts_available": False,
            "safe_for_multigraded_claims": False,
        },
    ]
    assert real["free_resolution_summary"]["betti_table_rows"] == ungraded["betti_table_rows"]
    assert real["cas_artifacts"]["fitting_ideals"]["Fitt0"] == "x_level,x_radius"
    assert real["cas_artifacts"]["minors"]["minors_1"] == "x_level,x_radius"
    ideal_diag = real["cas_artifacts"]["ideal_diagnostics"]
    assert ideal_diag["available"] is True
    assert ideal_diag["presentation_shape"] == [1, 2]
    assert ideal_diag["fitting_invariants"][0]["fitting_index"] == 0
    assert ideal_diag["fitting_invariants"][0]["determinantal_order"] == 1
    assert ideal_diag["determinantal_minors"][0]["minor_order"] == 1
    assert ideal_diag["not_a_resolution_certificate_by_itself"] is True
    grade_depth = real["cas_artifacts"]["grade_depth_regular_diagnostics"]
    assert grade_depth["available"] is False
    assert "did not include" in grade_depth["reason"]
    be_rank = real["cas_artifacts"]["buchsbaum_eisenbud_rank_conditions"]
    assert be_rank["available"] is True
    assert be_rank["free_module_ranks_by_homological_degree"] == {"0": 2, "1": 1}
    assert be_rank["image_rank_estimates_by_differential"] == {"d1": 1}
    assert be_rank["nonnegative_rank_conditions"] is True
    assert be_rank["exactness_certified_by_backend"] is True
    assert be_rank["is_independent_certificate"] is False


def test_macaulay2_rejects_nonhomogeneous_stored_multigrading(monkeypatch):
    module = {
        "coefficient_ring": "F2[x_level,x_radius]",
        "chain_module_generators": [
            {"simplex": ["a"], "homological_degree": 0, "multidegree": [0, 0]},
            {"simplex": ["b"], "homological_degree": 0, "multidegree": [0, 0]},
            {"simplex": ["a", "b"], "homological_degree": 1, "multidegree": [1, 2]},
        ],
        "boundary_monomials": [
            {"source_simplex": ["a", "b"], "target_face": ["a"], "monomial_exponent": [1, 0]},
            {"source_simplex": ["a", "b"], "target_face": ["b"], "monomial_exponent": [0, 1]},
        ],
    }
    schema = canonicalize_module(module)
    script = cas_free_resolution.build_macaulay2_script(schema)
    assert "presentation matrix is not homogeneous for stored multidegrees" in script
    m2_executable = cas_free_resolution._candidate_executable("M2")
    if m2_executable is None:
        return

    monkeypatch.setattr(cas_free_resolution, "_candidate_executable", lambda name: m2_executable if name == "M2" else None)
    real = cas_free_resolution.try_compute_real_free_resolution(module, timeout_s=10, use_cache=False)
    assert real["available"] is False
    assert real["status"] == "certificate_failed"
    assert real["input_sha256"] == schema["input_sha256"]
    assert real["module_summary"]["presentation_shape"] == [2, 1]
    assert real["cas_artifacts"] == {}
    assert real["certificate_attached"] is False
    assert real["backend_attempts"][0]["status"] == "ran"


def test_real_cas_free_resolution_smoke_when_backend_available():
    module = {
        "coefficient_ring": "F2[x_level,x_radius]",
        "chain_module_generators": [
            {"simplex": ["a"], "homological_degree": 0, "multidegree": [0, 1]},
            {"simplex": ["b"], "homological_degree": 0, "multidegree": [1, 0]},
            {"simplex": ["a", "b"], "homological_degree": 1, "multidegree": [1, 1]},
        ],
        "boundary_monomials": [
            {"source_simplex": ["a", "b"], "target_face": ["a"], "monomial_exponent": [1, 0]},
            {"source_simplex": ["a", "b"], "target_face": ["b"], "monomial_exponent": [0, 1]},
        ],
    }
    schema = canonicalize_module(module)
    singular_script = build_singular_script(schema)
    assert "matrix PM[2][1]" in singular_script
    assert "minor(PM,1)" in singular_script
    assert "Fitt0=" in singular_script
    assert "minors_1=" in singular_script
    m2_script = cas_free_resolution.build_macaulay2_script(schema)
    assert "F0 = R^{{0,-1},{-1,0}}" in m2_script
    assert "F1 = R^{{-1,-1}}" in m2_script
    assert "homogeneousPresentation = isHomogeneous M" in m2_script
    real = try_compute_real_free_resolution(module, timeout_s=10, use_cache=False)
    _assert_real_resolution_guard(real, "F2[x_level,x_radius]")
    if real["available"]:
        assert real["backend"] == "Singular" or real["backend"] in {"Macaulay2", "sage"}
        assert real["cas_artifacts"].get("betti_table_text")
        ungraded = real["cas_artifacts"]["betti_table_ungraded"]
        assert ungraded["available"] is True
        assert ungraded["homological_column_ranks"][:2] == [2, 1]
        assert [row["display"] for row in ungraded["free_modules"][:2]] == ["F_0 = S^2", "F_1 = S"]
        assert ungraded["betti_table_rows"]
        assert all(row["safe_for_multigraded_claims"] is False for row in ungraded["betti_table_rows"])
        assert ungraded["not_multigraded"] is True
        if real["backend"] == "Singular":
            assert real["cas_artifacts"]["singular_determinantal"]["available"] is True
            assert real["cas_artifacts"]["fitting_ideals"]
            assert real["cas_artifacts"]["minors"]
            assert "Fitt" in " ".join(real["cas_artifacts"]["fitting_ideals"].keys())
            assert "minors_1" in real["cas_artifacts"]["minors"]
        if real["backend"] == "Macaulay2":
            summary = real["free_resolution_summary"]
            assert real["multigraded_free_resolution_certified"] is True
            assert real["safe_to_render_as_multigraded_free_resolution"] is True
            assert summary["grading"] == "multigraded_bidegree_shifts_over_F2_polynomial_ring"
            assert summary["not_multigraded"] is False
            assert summary["safe_for_multigraded_claims"] is True
            betti_rows = {(row["homological_degree"], tuple(row["multidegree"]), row["rank"]) for row in summary["betti_table_rows"]}
            assert (0, (0, 1), 1) in betti_rows
            assert (0, (1, 0), 1) in betti_rows
            assert (1, (1, 1), 1) in betti_rows
            assert real["cas_artifacts"]["macaulay2_multigraded"]["safe_for_multigraded_claims"] is True
            cert = real["cas_artifacts"]["certificate_summary"]
            assert cert["certificate_type"] == "Macaulay2 res coker presentation over multigraded F2 polynomial ring"
            assert cert["exactness_certified"] is True
            assert cert["minimality_certified"] is True
            assert cert["homogeneous_presentation"] is True
            assert cert["input_sha256"] == real["input_sha256"]
            assert cert["safe_to_render_as_multigraded_free_resolution"] is True
            assert summary["free_modules"]
            assert real["cas_artifacts"]["differentials"]
            assert real["cas_artifacts"]["fitting_ideals"]
            assert real["cas_artifacts"]["minors"]
            be = real["cas_artifacts"]["buchsbaum_eisenbud_diagnostics"]
            assert be["available"] is True
            if cas_free_resolution.probe_bemultipliers().get("macaulay2_loadable"):
                assert be["multiplier_output_available"] is True
                assert be["bemultipliers_status"] == "computed_aMultiplier_1"
                assert "x" in be["a_multiplier_1_shape"]
                assert be["a_multiplier_1_matrix"]
                assert "matrix" in be["a_multiplier_1_matrix"]


def test_certified_cas_result_surfaces_buchsbaum_eisenbud_diagnostics():
    schema = canonicalize_module(_small_free_resolution_module())
    tagged = "\n".join([
        "TROPICALGT_RESOLUTION_BEGIN",
        "backend=Macaulay2",
        "exactness_certified=true",
        "minimality_certified=true",
        "homogeneous_presentation=true",
        "certificate_type=Macaulay2 res coker presentation over multigraded F2 polynomial ring",
        "presentation_shape=1x2",
        "betti_table_begin",
        "       0 1 2",
        "total: 1 2 1",
        "betti_table_end",
        "macaulay2_free_modules_begin",
        "F0_degrees={{0,0}}",
        "F1_degrees={{1,0},{0,1}}",
        "F2_degrees={{1,1}}",
        "macaulay2_free_modules_end",
        "macaulay2_differentials_begin",
        "d1_shape=1x2",
        "d1_source_degrees={{1,0},{0,1}}",
        "d1_target_degrees={{0,0}}",
        "d1_matrix=| x_level x_radius |",
        "d2_shape=2x1",
        "d2_source_degrees={{1,1}}",
        "d2_target_degrees={{1,0},{0,1}}",
        "d2_matrix=| x_radius || x_level |",
        "macaulay2_differentials_end",
        "fitting_ideals_begin",
        "Fitt0=ideal(x_level,x_radius)",
        "Fitt1=ideal 1_R",
        "fitting_ideals_end",
        "minors_begin",
        "minors_1=ideal(x_level,x_radius)",
        "minors_end",
        "grade_depth_regular_diagnostics_begin",
        "backend=Macaulay2",
        "source=Macaulay2 codim/depth/rank ideals on certified resolution differentials",
        "ambient_ring_dimension=2",
        "d1_rank=1",
        "d1_rank_ideal=ideal(x_level,x_radius)",
        "d1_rank_ideal_codim=2",
        "d1_rank_ideal_depth=1",
        "d1_grade_lower_bound_holds=true",
        "d2_rank=1",
        "d2_rank_ideal=ideal(x_level,x_radius)",
        "d2_rank_ideal_codim=2",
        "d2_rank_ideal_depth=1",
        "d2_grade_lower_bound_holds=true",
        "regular_element_certificate_available=false",
        "regular_element_certificate_reason=Macaulay2 emitted codim/depth diagnostics but no independent regular-sequence certificate is substituted",
        "not_a_resolution_certificate_by_itself=true",
        "grade_depth_regular_diagnostics_end",
        "buchsbaum_eisenbud_diagnostics_begin",
        "backend_diagnostics_available=true",
        "exactness_certified=true",
        "minimality_certified=true",
        "be_exactness_source=Macaulay2 res/HH exactness certificate for the displayed cokernel presentation",
        "multiplier_output_available=false",
        "bemultipliers_status=unavailable_no_package_path",
        "bemultipliers_repository=https://github.com/amelie-iska/BEMultipliers.git",
        "reason=Buchsbaum-Eisenbud multiplier output is rendered only after an explicit BEMultipliers run; no multiplier data is substituted",
        "buchsbaum_eisenbud_diagnostics_end",
        "TROPICALGT_RESOLUTION_END",
    ])
    parsed = cas_free_resolution._parse_tagged_output(tagged)
    assert parsed is not None
    real = cas_free_resolution._certified_result(
        schema,
        {
            "available": True,
            "backend": "M2",
            "parsed": parsed,
            "tagged_output": parsed["_raw"],
            "certificate_attached": True,
        },
        attempts=[{"backend": "M2", "status": "ran"}],
    )
    _assert_real_resolution_guard(real, "F2[x_level,x_radius]")
    assert real["backend"] == "Macaulay2"
    assert real["safe_to_render_as_multigraded_free_resolution"] is True
    cert = real["cas_artifacts"]["certificate_summary"]
    assert cert["available"] is True
    assert cert["backend"] == "Macaulay2"
    assert cert["certificate_attached"] is True
    assert cert["exactness_certified"] is True
    assert cert["minimality_certified"] is True
    assert cert["homogeneous_presentation"] is True
    assert cert["input_sha256"] == schema["input_sha256"]
    be = real["cas_artifacts"]["buchsbaum_eisenbud_diagnostics"]
    assert be["available"] is True
    assert be["exactness_certified"] is True
    assert be["minimality_certified"] is True
    assert be["multiplier_output_available"] is False
    assert be["safe_to_render_multiplier_output"] is False
    assert be["is_resolution_backend"] is False
    assert be["safe_to_substitute_for_resolution"] is False
    assert be["diagnostic_contract"]["requires_certified_macaulay2_chain_complex"] is True
    assert be["bemultipliers_status"] == "unavailable_no_package_path"
    assert "not substituted" in be["interpretation"]
    assert cert["bemultipliers_status"] == "unavailable_no_package_path"
    assert cert["bemultipliers_safe_to_render_multiplier_output"] is False
    assert cert["bemultipliers_is_resolution_backend"] is False
    assert real["cas_artifacts"]["fitting_ideals"]["Fitt0"] == "ideal(x_level,x_radius)"
    assert real["cas_artifacts"]["minors"]["minors_1"] == "ideal(x_level,x_radius)"
    ideal_diag = real["cas_artifacts"]["ideal_diagnostics"]
    assert ideal_diag["available"] is True
    assert ideal_diag["presentation_shape"] == [1, 2]
    assert ideal_diag["fitting_invariants"][0]["method"].startswith("Fitt_j")
    assert ideal_diag["determinantal_minors"][0]["minor_order"] == 1
    grade_depth = real["cas_artifacts"]["grade_depth_regular_diagnostics"]
    assert grade_depth["available"] is True
    assert grade_depth["ambient_ring_dimension"] == 2
    assert grade_depth["regular_element_certificate_available"] is False
    assert grade_depth["not_a_resolution_certificate_by_itself"] is True
    assert grade_depth["rank_ideal_diagnostics"] == [
        {
            "homological_degree": 1,
            "rank": 1,
            "rank_ideal": "ideal(x_level,x_radius)",
            "rank_ideal_codim": 2,
            "rank_ideal_depth": 1,
            "grade_lower_bound_holds": True,
        },
        {
            "homological_degree": 2,
            "rank": 1,
            "rank_ideal": "ideal(x_level,x_radius)",
            "rank_ideal_codim": 2,
            "rank_ideal_depth": 1,
            "grade_lower_bound_holds": True,
        },
    ]
    assert "never replace" in grade_depth["no_proxy_policy"]
    be_rank = real["cas_artifacts"]["buchsbaum_eisenbud_rank_conditions"]
    assert be_rank["available"] is True
    assert be_rank["free_module_ranks_by_homological_degree"] == {"0": 1, "1": 2, "2": 1}
    assert be_rank["image_rank_estimates_by_differential"] == {"d2": 1, "d1": 1}
    assert be_rank["shape_bounds_hold"] is True
    assert be_rank["paper_method_note"].startswith("For an exact finite free complex")
    assert real["free_resolution_summary"]["ideal_diagnostics"] == ideal_diag
    assert real["free_resolution_summary"]["buchsbaum_eisenbud_rank_conditions"] == be_rank
    assert real["free_resolution_summary"]["grade_depth_regular_diagnostics"] == grade_depth
    assert cert["grade_depth_regular_diagnostics_available"] is True
    assert cert["regular_element_certificate_available"] is False
    assert real["free_resolution_summary"]["betti_table_rows"] == [
        {
            "homological_degree": 0,
            "multidegree": [0, 0],
            "shift_display": "(0,0)",
            "rank": 1,
            "multiplicity": 1,
            "grading": "multigraded_bidegree_shift",
            "source": "macaulay2_free_module_degree_block",
            "not_multigraded": False,
            "multidegree_shifts_available": True,
            "safe_for_multigraded_claims": True,
        },
        {
            "homological_degree": 1,
            "multidegree": [0, 1],
            "shift_display": "(0,1)",
            "rank": 1,
            "multiplicity": 1,
            "grading": "multigraded_bidegree_shift",
            "source": "macaulay2_free_module_degree_block",
            "not_multigraded": False,
            "multidegree_shifts_available": True,
            "safe_for_multigraded_claims": True,
        },
        {
            "homological_degree": 1,
            "multidegree": [1, 0],
            "shift_display": "(1,0)",
            "rank": 1,
            "multiplicity": 1,
            "grading": "multigraded_bidegree_shift",
            "source": "macaulay2_free_module_degree_block",
            "not_multigraded": False,
            "multidegree_shifts_available": True,
            "safe_for_multigraded_claims": True,
        },
        {
            "homological_degree": 2,
            "multidegree": [1, 1],
            "shift_display": "(1,1)",
            "rank": 1,
            "multiplicity": 1,
            "grading": "multigraded_bidegree_shift",
            "source": "macaulay2_free_module_degree_block",
            "not_multigraded": False,
            "multidegree_shifts_available": True,
            "safe_for_multigraded_claims": True,
        },
    ]
    m2_script = cas_free_resolution.build_macaulay2_script(schema)
    singular_script = build_singular_script(schema)
    sage_script = cas_free_resolution.build_sage_python_script(schema)
    assert "grade_depth_regular_diagnostics_begin" in m2_script
    assert "rank_ideal_codim" in m2_script
    assert "regular_element_certificate_available=false" in m2_script
    assert "buchsbaum_eisenbud_diagnostics_begin" in m2_script
    assert "BEMultipliers" in m2_script
    assert "aMultiplier(1,C,ComputeRanks=>true)" in m2_script
    assert "bemultipliers_is_resolution_backend=false" in m2_script
    assert "requires_certified_macaulay2_chain_complex=true" in m2_script
    assert "safe_to_substitute_for_resolution=false" in m2_script
    assert "buchsbaum_eisenbud_diagnostics_begin" in singular_script
    assert "buchsbaum_eisenbud_diagnostics_begin" in sage_script


def test_bemultipliers_computed_output_is_post_resolution_diagnostic_only():
    tagged = "\n".join([
        "TROPICALGT_RESOLUTION_BEGIN",
        "backend=Macaulay2",
        "exactness_certified=true",
        "minimality_certified=true",
        "homogeneous_presentation=true",
        "certificate_type=Macaulay2 res coker presentation over multigraded F2 polynomial ring",
        "presentation_shape=1x1",
        "betti_table_begin",
        "total: 1 1",
        "betti_table_end",
        "macaulay2_free_modules_begin",
        "F0_degrees={{0,0}}",
        "F1_degrees={{1,0}}",
        "macaulay2_free_modules_end",
        "macaulay2_differentials_begin",
        "d1_shape=1x1",
        "d1_source_degrees={{1,0}}",
        "d1_target_degrees={{0,0}}",
        "d1_matrix=| x_level |",
        "macaulay2_differentials_end",
        "buchsbaum_eisenbud_diagnostics_begin",
        "backend_diagnostics_available=true",
        "exactness_certified=true",
        "minimality_certified=true",
        "be_exactness_source=Macaulay2 res/HH exactness certificate",
        "multiplier_output_available=true",
        "bemultipliers_status=computed_aMultiplier_1",
        "aMultiplier_1_shape=1x1",
        "aMultiplier_1_matrix=matrix {{1}}",
        "bemultipliers_is_resolution_backend=false",
        "requires_certified_macaulay2_chain_complex=true",
        "safe_to_substitute_for_resolution=false",
        "buchsbaum_eisenbud_diagnostics_end",
        "TROPICALGT_RESOLUTION_END",
    ])
    schema = canonicalize_module(_small_free_resolution_module())
    parsed = cas_free_resolution._parse_tagged_output(tagged)
    assert parsed is not None
    real = cas_free_resolution._certified_result(
        schema,
        {
            "available": True,
            "backend": "M2",
            "parsed": parsed,
            "tagged_output": parsed["_raw"],
            "certificate_attached": True,
        },
        attempts=[{"backend": "M2", "status": "ran"}],
    )
    be = real["cas_artifacts"]["buchsbaum_eisenbud_diagnostics"]
    assert be["multiplier_output_available"] is True
    assert be["safe_to_render_multiplier_output"] is True
    assert be["is_resolution_backend"] is False
    assert be["safe_to_substitute_for_resolution"] is False
    assert be["diagnostic_contract"]["safe_to_use_as_resolution_certificate"] is False
    cert = real["cas_artifacts"]["certificate_summary"]
    assert cert["buchsbaum_eisenbud_multiplier_output_available"] is True
    assert cert["bemultipliers_status"] == "computed_aMultiplier_1"
    assert cert["bemultipliers_safe_to_render_multiplier_output"] is True
    assert cert["bemultipliers_is_resolution_backend"] is False


def test_macaulay2_tropical_fan_diagnostic_parser_and_script():
    schema = cas_tropical.canonicalize_tropical_ideal({"variables": ["x", "y"], "generators": ["x+y+1"]})
    script = cas_tropical.build_macaulay2_tropical_script(schema)
    assert 'needsPackage "Tropical"' in script
    assert "tropicalVariety I" in script
    assert "rays T" in script
    tagged = "\n".join([
        "backend=Macaulay2",
        "tropical_package_available=true",
        "certificate_type=Macaulay2 Tropical tropicalVariety fan diagnostics",
        "tropical_cycle_certified=true",
        "class=TropicalCycle",
        "rays=matrix {{1, -1, 0}, {0, -1, 1}}",
        "max_cones={{1}, {0}, {2}}",
        "lineality_space=matrix {{}, {}}",
        "multiplicities={1, 1, 1}",
        "is_balanced=true",
        "is_pure=true",
        "is_simplicial=true",
        "fan_text=Fan{...1...}",
        "tropical_basis_check_available=true",
        "is_tropical_basis=true",
        "tropical_prevariety_available=true",
        "prevariety_class=Fan",
        "prevariety_rays=matrix {{1, -1, 0}, {0, -1, 1}}",
        "prevariety_max_cones={{1}, {0}, {2}}",
        "prevariety_lineality_space=matrix {{}, {}}",
        "prevariety_multiplicities={1, 1, 1}",
        "prevariety_is_balanced=true",
        "prevariety_is_pure=true",
        "prevariety_is_simplicial=true",
    ])
    parsed = cas_free_resolution._parse_key_value_lines(tagged)
    report = cas_tropical._certified_tropical_result(schema, parsed, tagged, attempts=[{"backend": "Macaulay2", "status": "ran"}])
    assert report["available"] is True
    assert report["schema_version"] == "tropicalgt.cas_tropical_fan.v1"
    assert report["tropical_cycle_certified"] is True
    assert report["safe_to_render_as_tropical_fan"] is True
    summary = report["fan_summary"]
    assert summary["ambient_dimension"] == 2
    assert summary["ray_count"] == 3
    assert summary["rays"] == [[1, -1, 0], [0, -1, 1]]
    assert summary["max_cones"] == [[1], [0], [2]]
    assert summary["multiplicities"] == [1, 1, 1]
    assert summary["is_balanced"] is True
    assert "one dimensional cones" in summary["one_dimensional_cone_language"]
    assert report["tropical_basis_check"] == {
        "available": True,
        "is_tropical_basis": True,
        "error": None,
        "method": "Macaulay2 Tropical isTropicalBasis on flatten entries gens I",
    }
    pre = report["tropical_prevariety_summary"]
    assert pre["available"] is True
    assert pre["class"] == "Fan"
    assert pre["ray_count"] == 3
    assert pre["max_cones"] == [[1], [0], [2]]
    assert pre["is_simplicial"] is True
    assert "not a multigraded free-resolution" in report["render_warning"]
    contract = report["certificate_contract"]
    assert contract["certificate_source"].startswith("Macaulay2 Tropical tropicalVariety")
    assert "tropicalVariety" in contract["required_macaulay2_methods"]
    assert "tropicalPrevariety" in contract["side_diagnostic_methods"]
    assert "not accepted" in contract["sage_scope"]
    assert "No support-token" in contract["no_proxy_policy"]


def test_macaulay2_tropical_fan_diagnostic_live_or_unavailable():
    report = cas_tropical.try_compute_tropical_fan_diagnostics(
        {"variables": ["x", "y"], "generators": ["x+y+1"], "source": "unit_test_tropical_line"},
        timeout_s=20,
        use_cache=False,
    )
    assert report["schema_version"] == "tropicalgt.cas_tropical_fan.v1"
    if report["available"]:
        assert report["backend"] == "Macaulay2"
        assert report["certificate_attached"] is True
        assert report["fan_diagnostics_certified"] is True
        assert report["fan_summary"]["ray_count"] == 3
        assert report["fan_summary"]["is_balanced"] is True
        assert "available" in report["tropical_basis_check"]
        assert "available" in report["tropical_prevariety_summary"]
        assert report["cas_artifacts"]["raw_tagged_output"]
    else:
        assert report["status"] in {"backend_not_installed", "timeout", "backend_error", "certificate_failed", "invalid_input"}
        assert report["certificate_attached"] is False
        assert report["safe_to_render_as_tropical_fan"] is False
        assert "certificate_contract" in report
        assert "No support-token" in report["certificate_contract"]["no_proxy_policy"]


def test_tropical_fan_diagnostic_caches_deterministic_unavailable_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("TROPICALGT_CAS_TROPICAL_CACHE_DIR", str(tmp_path / "cas-tropical-cache"))
    monkeypatch.setattr(cas_tropical, "_candidate_executable", lambda name: None)
    ideal = {"variables": ["x", "y"], "generators": ["x+y+1"], "source": "cache_test"}

    first = cas_tropical.try_compute_tropical_fan_diagnostics(ideal, timeout_s=1)
    second = cas_tropical.try_compute_tropical_fan_diagnostics(ideal, timeout_s=1)

    for report in (first, second):
        assert report["schema_version"] == "tropicalgt.cas_tropical_fan.v1"
        assert report["available"] is False
        assert report["status"] == "backend_not_installed"
        assert report["certificate_attached"] is False
        assert report["safe_to_render_as_tropical_fan"] is False
        assert report["cache"]["enabled"] is True
        assert report["cache"]["cache_schema_version"] == cas_tropical.TROPICAL_CACHE_SCHEMA_VERSION
        assert report["cache"]["adapter_cache_version"] == cas_tropical.TROPICAL_CACHE_VERSION
        assert str(tmp_path / "cas-tropical-cache") in report["cache"]["path"]

    assert first["cache"]["hit"] is False
    assert first["cache"]["written"] is True
    assert second["cache"]["hit"] is True
    assert second["cache"]["written"] is False
    assert second["cache"]["key"] == first["cache"]["key"]
    assert first["cas_artifacts"] == {}
    assert second["cas_artifacts"] == {}


def test_macaulay2_tropical_fan_diagnostic_invalid_input_unavailable():
    report = cas_tropical.try_compute_tropical_fan_diagnostics({"variables": ["x"], "generators": []}, use_cache=False)
    assert report["available"] is False
    assert report["status"] == "invalid_input"
    assert report["cas_artifacts"] == {}

    unsafe = cas_tropical.try_compute_tropical_fan_diagnostics(
        {"variables": ["x"], "generators": ['x; print "oops"']},
        use_cache=False,
    )
    assert unsafe["available"] is False
    assert unsafe["status"] == "invalid_input"
    assert "unsupported Macaulay2 polynomial generator text" in unsafe["reason"]


def test_macaulay2_toric_embedding_certificate_parser_and_script():
    schema = cas_toric.canonicalize_toric_exponent_matrix(
        {"exponent_matrix": [[1, 1, 1], [0, 1, 2]], "variable_names": ["z_0", "z_1", "z_2"]}
    )
    script = cas_toric.build_macaulay2_toric_embedding_script(schema)
    assert 'needsPackage "Quasidegrees"' in script
    assert "toricIdeal(A,R)" in script
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
    parsed = cas_free_resolution._parse_key_value_lines(tagged)
    report = cas_toric._certified_toric_embedding_result(schema, parsed, tagged, attempts=[{"backend": "Macaulay2", "status": "ran"}])
    assert report["available"] is True
    assert report["schema_version"] == "tropicalgt.cas_toric_embedding.v1"
    assert report["toric_embedding_certified"] is True
    assert report["toric_ideal_certified"] is True
    assert report["safe_to_render_as_toric_embedding"] is True
    assert report["embedding_scope"] == "finite_monomial_map_toric_ideal_certificate_only"
    assert report["tropical_variety_embedding_certified"] is False
    assert report["global_toric_variety_embedding_certified"] is False
    assert report["safe_to_render_as_tropical_variety_embedding"] is False
    assert report["safe_to_render_as_global_toric_variety_embedding"] is False
    assert report["safe_to_use_as_normal_fan_certificate"] is False
    assert report["monomial_map_summary"]["exponent_matrix"] == [[1, 1, 1], [0, 1, 2]]
    assert report["toric_ideal_summary"]["generator_count"] == 1
    assert "z_1^2-z_0*z_2" in report["toric_ideal_summary"]["ideal_text"]
    contract = report["certificate_contract"]
    assert "toricIdeal" in contract["required_macaulay2_methods"]
    assert "No chart-bundle logits" in contract["no_proxy_policy"]
    assert "not a tropical-variety" in contract["tool_backed_embedding_scope"]


def test_macaulay2_toric_embedding_certificate_live_or_unavailable():
    report = cas_toric.try_compute_toric_embedding_certificate(
        {"exponent_matrix": [[1, 1, 1], [0, 1, 2]], "source": "unit_test_rational_normal_curve"},
        timeout_s=20,
        use_cache=False,
    )
    assert report["schema_version"] == "tropicalgt.cas_toric_embedding.v1"
    if report["available"]:
        assert report["backend"] == "Macaulay2"
        assert report["certificate_attached"] is True
        assert report["toric_embedding_certified"] is True
        assert report["toric_ideal_certified"] is True
        assert report["safe_to_render_as_toric_embedding"] is True
        assert report["safe_to_render_as_tropical_variety_embedding"] is False
        assert report["safe_to_render_as_global_toric_variety_embedding"] is False
        assert report["safe_to_use_as_normal_fan_certificate"] is False
        assert report["cas_artifacts"]["raw_tagged_output"]
    else:
        assert report["status"] in {"backend_not_installed", "timeout", "backend_error", "certificate_failed", "invalid_input", "parse_error"}
        assert report["certificate_attached"] is False
        assert report["safe_to_render_as_toric_embedding"] is False
        assert report["safe_to_render_as_tropical_variety_embedding"] is False
        assert report["safe_to_render_as_global_toric_variety_embedding"] is False
        assert "No chart-bundle logits" in report["certificate_contract"]["no_proxy_policy"]


def test_toric_embedding_certificate_caches_deterministic_unavailable_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("TROPICALGT_CAS_TORIC_CACHE_DIR", str(tmp_path / "cas-toric-cache"))
    monkeypatch.setattr(cas_toric, "_candidate_executable", lambda name: None)
    exponent_matrix = {"exponent_matrix": [[1, 1, 1], [0, 1, 2]], "source": "cache_test"}

    first = cas_toric.try_compute_toric_embedding_certificate(exponent_matrix, timeout_s=1)
    second = cas_toric.try_compute_toric_embedding_certificate(exponent_matrix, timeout_s=1)

    for report in (first, second):
        assert report["schema_version"] == "tropicalgt.cas_toric_embedding.v1"
        assert report["available"] is False
        assert report["status"] == "backend_not_installed"
        assert report["certificate_attached"] is False
        assert report["safe_to_render_as_toric_embedding"] is False
        assert report["safe_to_render_as_tropical_variety_embedding"] is False
        assert report["safe_to_render_as_global_toric_variety_embedding"] is False
        assert report["safe_to_use_as_normal_fan_certificate"] is False
        assert report["cache"]["enabled"] is True
        assert report["cache"]["cache_schema_version"] == cas_toric.TORIC_EMBEDDING_CACHE_SCHEMA_VERSION
        assert report["cache"]["adapter_cache_version"] == cas_toric.TORIC_EMBEDDING_CACHE_VERSION
        assert str(tmp_path / "cas-toric-cache") in report["cache"]["path"]

    assert first["cache"]["hit"] is False
    assert first["cache"]["written"] is True
    assert second["cache"]["hit"] is True
    assert second["cache"]["written"] is False
    assert second["cache"]["key"] == first["cache"]["key"]
    assert first["cas_artifacts"] == {}
    assert second["cas_artifacts"] == {}


def test_macaulay2_toric_embedding_certificate_invalid_input_unavailable():
    report = cas_toric.try_compute_toric_embedding_certificate({"exponent_matrix": [[1, 2, 3], [0, 1]]}, use_cache=False)
    assert report["available"] is False
    assert report["status"] == "invalid_input"
    assert report["cas_artifacts"] == {}
    assert report["safe_to_render_as_toric_embedding"] is False
    assert report["safe_to_render_as_tropical_variety_embedding"] is False
    assert report["safe_to_render_as_global_toric_variety_embedding"] is False


def test_cas_canonicalization_preserves_generator_id_boundaries():
    module = {
        "coefficient_ring": "F2[x_level,x_radius]",
        "chain_module_generators": [
            {"generator_id": "c0_v0", "simplex": ["v0"], "homological_degree": 0, "multidegree": [0, 0]},
            {"generator_id": "c1_e1", "simplex": ["v0", "v1"], "homological_degree": 1, "multidegree": [1, 0]},
            {"generator_id": "c1_e2", "simplex": ["v0", "v2"], "homological_degree": 1, "multidegree": [0, 1]},
        ],
        "boundary_monomials": {
            "d1": [
                {"source_generator_id": "c1_e1", "target_generator_id": "c0_v0", "exponent": [1, 0]},
                {"source_generator_id": "c1_e2", "target_generator_id": "c0_v0", "exponent": [0, 1]},
            ]
        },
    }
    schema = canonicalize_module(module)
    entries = schema["presentation_matrix"]["entries"]
    assert entries == [
        {"row": 0, "col": 0, "exponent": [1, 0], "monomial": "x_level"},
        {"row": 0, "col": 1, "exponent": [0, 1], "monomial": "x_radius"},
    ]


def test_real_cas_total_graded_output_is_not_multigraded_for_generator_id_boundary():
    module = {
        "coefficient_ring": "F2[x_level,x_radius]",
        "chain_module_generators": [
            {"generator_id": "c0_v0", "simplex": ["v0"], "homological_degree": 0, "multidegree": [0, 0]},
            {"generator_id": "c1_e1", "simplex": ["v0", "v1"], "homological_degree": 1, "multidegree": [1, 0]},
            {"generator_id": "c1_e2", "simplex": ["v0", "v2"], "homological_degree": 1, "multidegree": [0, 1]},
        ],
        "boundary_monomials": {
            "d1": [
                {"source_generator_id": "c1_e1", "target_generator_id": "c0_v0", "exponent": [1, 0]},
                {"source_generator_id": "c1_e2", "target_generator_id": "c0_v0", "exponent": [0, 1]},
            ]
        },
    }
    real = try_compute_real_free_resolution(module, timeout_s=90)
    _assert_real_resolution_guard(real, "F2[x_level,x_radius]")
    if real["available"] and real.get("backend") == "sage":
        assert real["total_graded_resolution_certified"] is True
        assert real["safe_to_render_as_total_graded_resolution"] is True
        assert real["multigraded_free_resolution_certified"] is False
        assert real["safe_to_render_as_multigraded_free_resolution"] is False
        assert [row["rank"] for row in real["free_resolution_summary"]["free_modules"]] == [1, 2, 1]


def test_topological_algebra_report_has_multiparameter_data():
    record = FixtureGraphDataset(1)[0]
    filtered = build_filtered_simplicial_object(record)
    report = compute_topological_algebra_report(filtered, audit_level="full", ph_backend="gudhi", max_simplices=128)
    assert report["enabled"] is True
    assert report["chain_complex"]["field"] == "F2"
    assert report["multiparameter_persistence"]["num_parameters"] == 3
    assert report["multiparameter_persistence"]["fiber_rank_profile"]
    reps = report["persistence_representations"]
    assert reps["backend"] == "gudhi.representations"
    assert "decision_policy" in reps
    if reps["available"]:
        assert reps["summary"]["landscape_l2_norm"] >= 0.0
        assert reps["summary"]["topological_vector_l2_norm"] >= 0.0
        assert any(row.get("method") == "Landscape" for row in reps["decision_policy"])
    assert "commutative_algebra" in report
    assert "multiparameter_free_resolution_legacy_alias" not in report["commutative_algebra"]
    chain = report["commutative_algebra"]["multiparameter_chain_presentation_diagnostics"]
    assert chain["ring"] == "F2[x_filtration,x_dimension,x_position]"
    assert chain["free_chain_modules"]
    assert chain["determinantal_ideals"]["available"] is True
    assert chain["fitting_ideals"]["available"] is True
    assert chain["buchsbaum_eisenbud"]["available"] is True
    assert "maps" in chain["determinantal_ideals"]
    assert "maps" in chain["fitting_ideals"]
    assert "rank_exactness_checks" in chain["buchsbaum_eisenbud"]
    assert chain["minimal_free_resolution"]["available"] is False
    assert chain["not_a_free_resolution"] is True
    assert chain["resolution_status"] == "chain_presentation_only"
    real = chain["real_free_resolution"]
    _assert_real_resolution_guard(real, "F2[x_filtration,x_dimension,x_position]")
    probed_names = {row["name"] for row in real["backend_probe"]["backends"]}
    assert probed_names >= {"M2", "Singular", "sage"}
    assert chain["not_a_free_resolution"] is True
    _assert_real_resolution_guard(chain["real_free_resolution"], "F2[x_filtration,x_dimension,x_position]")
    summary = summarize_algebra_reports([report])
    assert summary["algebra_reports"] == 1.0


def test_bivariate_staircase_resolution_known_monomial_ideals():
    variables = ["x_level", "x_radius"]

    singleton = _bivariate_staircase_resolution_from_points(
        [(2, 1)],
        variables,
        ideal_name="singleton",
        source="test",
        auxiliary=True,
    )
    assert singleton["available"] is True
    assert [module["rank"] for module in singleton["free_modules"]] == [1, 1]
    assert [row["bidegree"] for row in singleton["minimal_generators"]] == [[2, 1]]
    assert singleton["adjacent_lcm_syzygies"] == []
    assert singleton["differentials"][0]["entries"] == [
        {"row": 0, "column": 0, "entry": "x_level^2*x_radius", "exponent": [2, 1]}
    ]

    two_generator = _bivariate_staircase_resolution_from_points(
        [(2, 1), (1, 2)],
        variables,
        ideal_name="two_generator",
        source="test",
        auxiliary=True,
    )
    assert [module["rank"] for module in two_generator["free_modules"]] == [1, 2, 1]
    assert [row["lcm_bidegree"] for row in two_generator["adjacent_lcm_syzygies"]] == [[2, 2]]
    assert two_generator["differentials"][1]["entries"] == [
        {"row": 0, "column": 0, "entry": "x_radius", "exponent": [0, 1]},
        {"row": 1, "column": 0, "entry": "x_level", "exponent": [1, 0]},
    ]

    three_generator = _bivariate_staircase_resolution_from_points(
        [(3, 1), (2, 2), (1, 3), (4, 4)],
        variables,
        ideal_name="three_generator",
        source="test",
        auxiliary=True,
    )
    assert [row["bidegree"] for row in three_generator["minimal_generators"]] == [[3, 1], [2, 2], [1, 3]]
    assert [module["rank"] for module in three_generator["free_modules"]] == [1, 3, 2]
    assert [row["lcm_bidegree"] for row in three_generator["adjacent_lcm_syzygies"]] == [[3, 2], [2, 3]]
    assert [row for row in three_generator["betti_table_rows"] if row["homological_degree"] == 2] == [
        {"homological_degree": 2, "multidegree": [3, 2], "rank": 1},
        {"homological_degree": 2, "multidegree": [2, 3], "rank": 1},
    ]
    assert "one dimensional cone(s)" in three_generator["toric_exponent_chart"]["interpretation"]
    assert three_generator["derived_category_note"].startswith("This finite free complex")


def test_level_radius_bifiltration_reports_scoped_real_staircase_resolution(tmp_path):
    growth = [
        {
            "level": 0,
            "filtered_simplicial_object": {
                "simplex_tree": {"backend": "gudhi.SimplexTree", "num_simplices": 1},
                "simplices": [{"simplex": ["root"], "dimension": 0, "filtration": 0.0}],
            },
        },
        {
            "level": 1,
            "filtered_simplicial_object": {
                "simplex_tree": {"backend": "gudhi.SimplexTree", "num_simplices": 2},
                "simplices": [
                    {"simplex": ["root"], "dimension": 0, "filtration": 0.0},
                    {"simplex": ["a"], "dimension": 0, "filtration": 0.6},
                ],
            },
        },
        {
            "level": 2,
            "filtered_simplicial_object": {
                "simplex_tree": {"backend": "gudhi.SimplexTree", "num_simplices": 4},
                "simplices": [
                    {"simplex": ["root"], "dimension": 0, "filtration": 0.0},
                    {"simplex": ["b"], "dimension": 0, "filtration": 0.3},
                    {"simplex": ["a"], "dimension": 0, "filtration": 0.6},
                    {"simplex": ["a", "b"], "dimension": 1, "filtration": 0.7},
                ],
            },
        },
    ]
    report = compute_level_radius_bifiltration_report(growth, max_simplices=32)
    assert report["coefficient_ring"] == "F2[x_level,x_radius]"
    assert report["grid_axes"][0] == [0, 1, 2]
    assert report["radius_grade_values"] == {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.7}
    provenance = report["grid_fiber_provenance"]
    assert provenance["all_growth_rows_have_gudhi_simplex_tree"] is True
    assert provenance["simplex_tree_backend_counts"] == {"gudhi.SimplexTree": 3}
    first_fiber = next(row for row in report["fiber_rank_profile"] if row["grade"] == [0, 0])
    assert first_fiber["fiber_basis"]["basis_by_dim"] == {"0": [["root"]]}
    assert first_fiber["fiber_basis"]["total_basis_count"] == 1
    assert first_fiber["fiber_basis"]["truncated"] is False
    assert first_fiber["fiber_provenance"]["all_source_levels_have_gudhi_simplex_tree"] is True
    assert first_fiber["fiber_provenance"]["construction"].startswith("K_(level,radius)")
    structure_maps = report["structure_maps"]
    assert structure_maps
    assert {row["direction"] for row in structure_maps} >= {"x_level", "x_radius"}
    assert all(row["field"] == "F2" for row in structure_maps)
    assert all(set(row["homology_rank"]) >= {"0", "1", "2"} for row in structure_maps)
    assert all("rank(B_target + image(Z_source))" in row["method"] for row in structure_maps)

    from tropicalgt.visualization import write_two_parameter_bifiltration_visualization

    html_path = tmp_path / "two_parameter_bifiltration.html"
    write_two_parameter_bifiltration_visualization(html_path, report, title="test bifiltration")
    html = html_path.read_text()
    visual_payload = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert visual_payload["schema_version"] == "tropicalgt.two_parameter_bifiltration_visual.v1"
    assert visual_payload["primary_view"] == "miller_sturmfels_bivariate_staircase"
    assert visual_payload["rank_surface_primary"] is False
    assert visual_payload["axes"] == {"horizontal": "x_radius", "vertical": "x_level", "coordinate_one_dimensional_cones": ["rho_x_radius", "rho_x_level"]}
    assert visual_payload["actual_data_only"] is True
    assert visual_payload["no_proxy_resolution_claim"] is True
    assert "rank_invariant_samples_table" in visual_payload["secondary_views"]
    assert "certified_fitting_minor_tables" in visual_payload["secondary_views"]
    assert "buchsbaum_eisenbud_diagnostic_tables" in visual_payload["secondary_views"]
    assert visual_payload["rank_invariant_sample_count"] == len(report["rank_invariant_samples"])
    assert visual_payload["rank_invariant_sample_count"] > 0
    staircase_cards = visual_payload["staircase_cards"]
    assert staircase_cards
    assert any(card["primary_card"] for card in staircase_cards)
    assert all("generator_labels" in card for card in staircase_cards)
    assert all("upward_closed_regions" in card for card in staircase_cards)
    assert all("quotient_basis_lattice_points" in card for card in staircase_cards)
    assert all(card["quotient_basis_lattice_count"] == len(card["quotient_basis_lattice_points"]) for card in staircase_cards)
    assert all("hilbert_numerator_terms" in card for card in staircase_cards)
    assert any(card["adjacent_lcm_syzygies"] for card in staircase_cards)
    primary_card = next(card for card in staircase_cards if card["primary_card"])
    assert primary_card["generator_labels"][0]["label"].startswith("g")
    assert primary_card["upward_closed_regions"][0]["x_radius_max_displayed"] >= primary_card["upward_closed_regions"][0]["x_radius_min"]
    assert "not a full persistence-module free resolution" in primary_card["theorem_scope"]
    assert "horizontal lattice coordinates are x_radius" in html
    assert "The primary view is the Miller-Sturmfels staircase" in html
    assert "Columns are radius grades" in html
    assert "Adjacent structure maps persisted=" in html
    assert "diagnostic chain data is not substituted for a free resolution" in html
    assert "Rank-invariant samples over F2[x_level,x_radius]" in html
    assert "source monomial" in html and "target monomial" in html
    assert "Certified Fitting ideals and determinantal minors" in html
    assert "Buchsbaum-Eisenbud rank and multiplier diagnostics" in html
    assert "x_radius exponent" in html and "radius grade" in html
    assert "x_level exponent" in html and "reasoning growth level" in html
    chain = report["chain_presentation_diagnostics"]
    assert chain["not_a_free_resolution"] is True
    real = chain["real_free_resolution"]
    _assert_real_resolution_guard(real, "F2[x_level,x_radius]")
    assert real["module_summary"]["variables"] == ["x_level", "x_radius"]
    minimal = chain["minimal_free_resolution"]
    assert minimal["available"] is True
    assert minimal["not_full_persistence_module_resolution"] is True
    assert minimal["scope"] == "auxiliary_two_variable_staircase_monomial_ideal"
    resolution = minimal["resolution"]
    assert resolution["ring"] == "F2[x_level,x_radius]"
    assert resolution["toric_exponent_chart"]["semigroup"] == "N^2"
    assert resolution["toric_exponent_chart"]["ambient_one_dimensional_cones"] == [
        {"name": "rho_x_level", "primitive_generator": [1, 0], "variable": "x_level", "monoid_generator": "x_level"},
        {"name": "rho_x_radius", "primitive_generator": [0, 1], "variable": "x_radius", "monoid_generator": "x_radius"},
    ]
    assert "one dimensional cone(s)" in resolution["toric_exponent_chart"]["interpretation"]
    assert resolution["one_dimensional_cone_language"]["required_terminology"] == "one dimensional cone(s)"
    assert [row["display"] for row in resolution["free_modules"]] == [
        "F_0 = S",
        "F_1 = S(-2,1) + S(-1,2)",
        "F_2 = S(-2,2)",
    ]
    assert resolution["differentials"][0]["entries"] == [
        {"row": 0, "column": 0, "entry": "x_level^2*x_radius", "exponent": [2, 1]},
        {"row": 0, "column": 1, "entry": "x_level*x_radius^2", "exponent": [1, 2]},
    ]
    assert resolution["differentials"][1]["entries"] == [
        {"row": 0, "column": 0, "entry": "x_radius", "exponent": [0, 1]},
        {"row": 1, "column": 0, "entry": "x_level", "exponent": [1, 0]},
    ]
    assert resolution["buchsbaum_eisenbud_diagnostics"]["exactness_certified"] is True
    assert "one dimensional cone(s)" in resolution["buchsbaum_eisenbud_diagnostics"]["certificate"]
    assert "D^b(gr-F2[x,y])" in resolution["derived_category_note"]
    assert "one dimensional cone(s)" in resolution["derived_category_note"]



def test_ripser_backend_is_selected_when_requested():
    record = FixtureGraphDataset(1)[0]
    filtered = build_filtered_simplicial_object(record)
    report = compute_topological_algebra_report(filtered, audit_level="topology", ph_backend="ripser", max_simplices=128)
    assert report["persistence"]["backend"] == "ripser"
    assert "available" in report["persistence"]


def test_reasoning_trajectory_complex_grows_by_level():
    candidates = [
        {"record_id": "root", "level": 0, "score": 0.0, "nll": 1.0, "path": [], "parent": None, "embedding": [0.0, 0.0, 0.0]},
        {"record_id": "a", "level": 1, "score": 0.1, "nll": 0.9, "path": ["expand"], "parent": "root", "embedding": [1.0, 0.0, 0.0]},
        {"record_id": "b", "level": 2, "score": 0.2, "nll": 0.8, "path": ["expand", "verify"], "parent": "a", "embedding": [0.0, 1.0, 0.0]},
    ]
    level_one = build_reasoning_trajectory_complex(candidates, up_to_level=1)
    full = build_reasoning_trajectory_complex(candidates)
    assert level_one["summary"]["num_vertices"] == 2
    assert full["summary"]["num_vertices"] == 3
    assert full["summary"]["num_two_simplices"] == 1
    assert full["simplex_tree"]["backend"] == "gudhi.SimplexTree"
    assert full["summary"]["simplex_tree_available"] is True
    assert all(simplex.get("gudhi_simplex_tree") for simplex in full["simplices"])


def test_inference_scaling_emits_step_and_trajectory_algebra():
    record = FixtureGraphDataset(1)[0]
    tok = TokenGTTokenizer(feature_dim=48)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    report = run_inference_scaling(
        model,
        record,
        tok,
        seq_len=32,
        device=torch.device("cpu"),
        depth=1,
        width=2,
        branch_factor=2,
        trace_limit=4,
        audit_level="full",
        ph_backend="gudhi",
        audit_max_simplices=128,
    )
    assert report["candidates"][0]["topological_algebra"]["multiparameter_persistence"]["num_parameters"] == 3
    assert report["trajectory_topological_algebra"]["multiparameter_persistence"]["fiber_rank_profile"]
    assert report["trajectory_growth"]
    assert report["trajectory_level_radius_bifiltration"]["available"] is True
    assert report["trajectory_level_radius_bifiltration"]["grid_fiber_provenance"]["all_growth_rows_have_gudhi_simplex_tree"] is True
    assert report["trajectory_level_radius_bifiltration"]["fiber_rank_profile"][0]["fiber_basis"]["total_basis_count"] >= 1

    light_report = run_inference_scaling(
        model,
        record,
        tok,
        seq_len=32,
        device=torch.device("cpu"),
        depth=1,
        width=2,
        branch_factor=2,
        trace_limit=4,
        audit_level="none",
        ph_backend="gudhi",
        audit_max_simplices=128,
    )
    assert light_report["trajectory_topological_algebra"] is None
    assert light_report["trajectory_growth"]
    assert light_report["trajectory_level_radius_bifiltration"]["available"] is True
    assert light_report["trajectory_level_radius_bifiltration"]["coefficient_ring"] == "F2[x_level,x_radius]"
    assert light_report["trajectory_level_radius_bifiltration"]["grid_fiber_provenance"]["all_growth_rows_have_gudhi_simplex_tree"] is True
    assert light_report["trajectory_level_radius_bifiltration"]["grid_fiber_provenance"]["object_key"] in {
        "filtered_simplicial_object",
        "probability_filtered_simplicial_object",
    }


def test_inference_scaling_bifiltration_uses_vertex_only_probability_start_state():
    record = FixtureGraphDataset(1)[0]
    tok = TokenGTTokenizer(feature_dim=48)
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    report = run_inference_scaling(
        model,
        record,
        tok,
        seq_len=32,
        device=torch.device("cpu"),
        depth=0,
        width=1,
        branch_factor=1,
        trace_limit=4,
        audit_level="none",
        ph_backend="gudhi",
        audit_max_simplices=128,
    )

    growth = report["trajectory_growth"]
    assert len(growth) == 1
    prob_complex = growth[0]["probability_filtered_simplicial_object"]
    assert prob_complex["available"] is True
    assert prob_complex["summary"]["radius_filtration"] is True
    assert prob_complex["summary"]["single_vertex_radius_filtration"] is True
    assert prob_complex["summary"]["num_vertices"] == 1
    assert prob_complex["summary"]["num_edges"] == 0

    bif = report["trajectory_level_radius_bifiltration"]
    assert bif["available"] is True
    assert bif["coefficient_ring"] == "F2[x_level,x_radius]"
    assert bif["object_key_selected"] == "probability_filtered_simplicial_object"
    assert "vertex-only start states" in bif["object_key_policy"]
    assert bif["grid_fiber_provenance"]["object_key"] == "probability_filtered_simplicial_object"
    assert bif["grid_fiber_provenance"]["all_growth_rows_have_gudhi_simplex_tree"] is True
    assert bif["fiber_rank_profile"][0]["fiber_basis"]["total_basis_count"] == 1


def test_sequential_text_is_always_graphified():
    record = GraphRecord.from_mapping({"record_id": "plain", "text": "alpha beta gamma delta"})
    types = {node["type"] for node in record.graph_json["nodes"]}
    assert "sequence_chunk" in types
    assert record.metadata["graph_json_sequentialized"] is True
