import torch

from tropicalgt.algebra import compute_level_radius_bifiltration_report, compute_topological_algebra_report, summarize_algebra_reports
from tropicalgt.cas_free_resolution import canonicalize_module, try_compute_real_free_resolution
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
        assert real["status"] in {"unavailable_no_certificate", "backend_not_installed", "certificate_failed"}
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


def test_real_cas_free_resolution_smoke_when_backend_available():
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
    real = try_compute_real_free_resolution(module, timeout_s=10)
    _assert_real_resolution_guard(real, "F2[x_level,x_radius]")
    if real["available"]:
        assert real["backend"] == "Singular" or real["backend"] in {"Macaulay2", "sage"}
        assert real["cas_artifacts"].get("betti_table_text")
        ungraded = real["cas_artifacts"]["betti_table_ungraded"]
        assert ungraded["available"] is True
        assert ungraded["homological_column_ranks"][:2] == [2, 1]
        assert [row["display"] for row in ungraded["free_modules"][:2]] == ["F_0 = S^2", "F_1 = S"]
        assert ungraded["not_multigraded"] is True


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


def test_level_radius_bifiltration_reports_scoped_real_staircase_resolution():
    growth = [
        {
            "level": 0,
            "filtered_simplicial_object": {
                "simplices": [{"simplex": ["root"], "dimension": 0, "filtration": 0.0}],
            },
        },
        {
            "level": 1,
            "filtered_simplicial_object": {
                "simplices": [
                    {"simplex": ["root"], "dimension": 0, "filtration": 0.0},
                    {"simplex": ["a"], "dimension": 0, "filtration": 0.6},
                ],
            },
        },
        {
            "level": 2,
            "filtered_simplicial_object": {
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


def test_sequential_text_is_always_graphified():
    record = GraphRecord.from_mapping({"record_id": "plain", "text": "alpha beta gamma delta"})
    types = {node["type"] for node in record.graph_json["nodes"]}
    assert "sequence_chunk" in types
    assert record.metadata["graph_json_sequentialized"] is True
