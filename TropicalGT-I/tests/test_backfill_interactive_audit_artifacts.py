import importlib.util
import json
from pathlib import Path

from tropicalgt.simplicial import build_embedding_radius_simplicial_object


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
    assert "toric_embedding_sidecar_unavailable_backfill" in kinds
    assert "two_parameter_bifiltration_visual_contract_backfill" in kinds
    assert "inference_audit_dashboard_rebuilt" in kinds
    fan = json.loads((audit / "tropical_fan_diagnostics.json").read_text(encoding="utf-8"))
    assert fan["available"] is False
    assert fan["safe_to_render_as_tropical_fan"] is False
    toric = json.loads((audit / "toric_embedding_sidecar.json").read_text(encoding="utf-8"))
    assert toric["available"] is False
    assert toric["safe_to_render_as_finite_toric_ideal_sidecar"] is False
    assert toric["safe_to_use_as_normal_fan_certificate"] is False
    assert "chart-bundle" in toric["render_contract"]
    dashboard = (audit / "inference_audit.html").read_text(encoding="utf-8")
    assert "toric_embedding_sidecar.html" in dashboard
    assert "tropical_fan_diagnostics.html" in dashboard
    visual = json.loads((audit / "trajectory_persistence" / "two_parameter_bifiltration.json").read_text(encoding="utf-8"))
    assert visual["primary_view"] == "miller_sturmfels_bivariate_staircase"
    assert visual["rank_surface_primary"] is False
    assert "Backfills only explicit unavailable diagnostics" in report["policy"]
    assert "toric ideals" in report["policy"]


def test_backfill_rebuilds_stale_dashboard_when_sidecars_already_exist(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    (audit / "tropical_fan_diagnostics.json").write_text("{}", encoding="utf-8")
    (audit / "tropical_fan_diagnostics.html").write_text("<html>fan</html>", encoding="utf-8")
    (audit / "toric_embedding_sidecar.json").write_text("{}", encoding="utf-8")
    (audit / "toric_embedding_sidecar.html").write_text("<html>toric</html>", encoding="utf-8")
    (audit / "inference_audit.html").write_text("<html><a href='tropical_fan_diagnostics.html'>fan</a></html>", encoding="utf-8")

    report = module.backfill_audit_root(audit)

    assert [row["kind"] for row in report["actions"]] == ["inference_audit_dashboard_rebuilt"]
    dashboard = (audit / "inference_audit.html").read_text(encoding="utf-8")
    assert "tropical_fan_diagnostics.html" in dashboard
    assert "toric_embedding_sidecar.html" in dashboard



def test_backfill_regenerates_reasoning_step_contracts_from_stored_candidate_complexes(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    descriptors = [
        {"kind": "state", "index": 0, "text": "root"},
        {"kind": "reasoning", "index": 1, "text": "expand"},
        {"kind": "output", "index": 2, "text": "answer"},
    ]
    record = type("Record", (), {"record_id": "q0"})()
    simplicial = build_embedding_radius_simplicial_object(
        record,
        descriptors,
        [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [0.2, 0.4, 0.1]],
        metric="euclidean",
    )
    (audit / "inference_scaling_tree.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "record_id": "q0",
                        "level": 0,
                        "path": [],
                        "embedding": [0.0, 0.0, 0.0],
                        "filtered_simplicial_object": simplicial,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (audit / "got_embedding_map_payloads.json").write_text(
        json.dumps(
            {
                "layout_contract": {
                    "schema_version": "tropicalgt.embedding_trajectory_identity.v1",
                    "no_proxy_or_fallback": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (audit / "got_nll_density_cloud_payload.json").write_text(
        json.dumps(
            {
                "visual_layer_contract": {
                    "schema_version": "tropicalgt.nll_density_render.v1",
                    "sample_points_are_model_states": False,
                }
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "got_full_trajectory_complex_slider_contract.json",
        "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json",
        "got_full_trajectory_complex_jensen_shannon_slider_contract.json",
        "got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json",
    ):
        (audit / name).write_text("{}", encoding="utf-8")
    stale_dir = audit / "reasoning_step_complex_maps"
    stale_dir.mkdir()
    (stale_dir / "manifest.json").write_text(json.dumps({"steps": [{"file": "reasoning_step_000.html"}]}), encoding="utf-8")

    report = module.backfill_audit_root(audit)

    kinds = {row["kind"] for row in report["actions"]}
    assert "reasoning_step_complex_contract_backfill" in kinds
    manifest = json.loads((stale_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"]["schema_version"] == "tropicalgt.reasoning_step_complex_maps.v1"
    assert manifest["contract"]["all_steps_have_source_contracts"] is True
    assert manifest["contract"]["all_steps_have_radius_slider_contracts"] is True
    assert manifest["contract"]["all_steps_have_simplex_tree_poset_contracts"] is True
    step = manifest["steps"][0]
    assert step["step_complex_source_contract"]["schema_version"] == "tropicalgt.reasoning_step_complex_source_contract.v1"
    assert step["step_complex_source_contract"]["no_proxy_or_fallback"] is True
    assert step["radius_slider_contract"]["safe_to_render_radius_filtration"] is True
    assert step["simplex_tree_poset_contract"]["schema_version"] == "tropicalgt.simplex_tree_poset.v1"
    assert step["simplex_tree_poset_contract"]["no_proxy_or_fallback"] is True
    assert (stale_dir / "reasoning_step_000_slider_contract.json").exists()
    assert (stale_dir / "reasoning_step_000_simplex_tree_simplex_tree_poset_contract.json").exists()
    dashboard = (audit / "inference_audit.html").read_text(encoding="utf-8")
    assert "reasoning_step_complex_maps/manifest.json" in dashboard



def test_backfill_regenerates_got_trajectory_contracts_from_stored_scaling_tree(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    descriptors = [
        {"kind": "state", "index": 0, "text": "root"},
        {"kind": "reasoning", "index": 1, "text": "expand"},
        {"kind": "output", "index": 2, "text": "answer"},
    ]
    record = type("Record", (), {"record_id": "q0"})()
    simplicial = build_embedding_radius_simplicial_object(
        record,
        descriptors,
        [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [0.2, 0.4, 0.1]],
        metric="euclidean",
    )
    (audit / "inference_scaling_tree.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "record_id": "q0",
                        "level": 0,
                        "path": [],
                        "embedding": [0.0, 0.0, 0.0],
                        "score": 0.2,
                        "nll": 1.0,
                        "filtered_simplicial_object": simplicial,
                    },
                    {
                        "record_id": "q1",
                        "parent": "q0",
                        "level": 1,
                        "path": ["expand"],
                        "embedding": [0.8, 0.2, 0.1],
                        "score": 0.4,
                        "nll": 0.8,
                        "filtered_simplicial_object": simplicial,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = module.backfill_audit_root(audit)

    kinds = {row["kind"] for row in report["actions"]}
    assert "got_trajectory_contract_backfill" in kinds
    embedding_payload = json.loads((audit / "got_embedding_map_payloads.json").read_text(encoding="utf-8"))
    assert embedding_payload["layout_contract"]["schema_version"] == "tropicalgt.embedding_trajectory_identity.v1"
    assert embedding_payload["layout_contract"]["coordinate_source"] == "model graph_state embeddings"
    density_payload = json.loads((audit / "got_nll_density_cloud_payload.json").read_text(encoding="utf-8"))
    assert density_payload["visual_layer_contract"]["schema_version"] == "tropicalgt.nll_density_render.v1"
    assert density_payload["visual_layer_contract"]["support_samples_are_model_states"] is False
    assert density_payload["sample_points_are_model_states"] is False
    assert (audit / "got_full_trajectory_complex_slider_contract.json").exists()
    manifest = json.loads((audit / "reasoning_step_complex_maps" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"]["schema_version"] == "tropicalgt.reasoning_step_complex_maps.v1"



def test_backfill_regenerates_unavailable_analogical_contracts_without_proxy_maps(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    (audit / "analogical_memory_retrieval.json").write_text(
        json.dumps(
            {
                "bank_path": "",
                "bank_size": 0,
                "quality_gate": {"min_quality": 0.8},
                "top_k": 5,
                "retrieved": [],
            }
        ),
        encoding="utf-8",
    )
    stale_topk = {
        "schema_version": "tropicalgt.analogical_topk.v1",
        "readability_contract": {"schema_version": "tropicalgt.analogical_topk_readability.v1"},
    }
    stale_analogy_contract = {
        "schema_version": "tropicalgt.analogical_simplex_tree_analogy.v1",
        "available": False,
        "pair_count": 0,
        "no_proxy_or_fallback": True,
    }
    (audit / "analogical_simplicial_maps.json").write_text(
        json.dumps(
            {
                "available": False,
                "reason": "legacy",
                "topk_contract": stale_topk,
                "simplex_tree_analogy_contract": stale_analogy_contract,
            }
        ),
        encoding="utf-8",
    )
    (audit / "analogical_simplex_tree_analogy.json").write_text(json.dumps({"contract": stale_analogy_contract, "pairs": []}), encoding="utf-8")
    for stale_name in (
        "analogical_memory_retrieval.html",
        "analogical_memory_topk_index.html",
        "analogical_memory_map_02.html",
        "analogical_simplex_tree_analogy.html",
    ):
        (audit / stale_name).write_text("static legacy html without local plotly asset", encoding="utf-8")

    report = module.backfill_audit_root(audit)

    kinds = {row["kind"] for row in report["actions"]}
    assert "analogical_memory_contract_backfill" in kinds
    maps = json.loads((audit / "analogical_simplicial_maps.json").read_text(encoding="utf-8"))
    assert maps["available"] is False
    assert maps["reason"] == "no_non_self_model_memory"
    assert maps["topk_contract"]["schema_version"] == "tropicalgt.analogical_topk.v1"
    assert maps["topk_contract"]["no_proxy_or_fallback"] is True
    assert maps["topk_contract"]["top_k_rendered"] == 0
    assert maps["topk_contract"]["readability_contract"]["schema_version"] == "tropicalgt.analogical_topk_readability.v1"
    analogy = json.loads((audit / "analogical_simplex_tree_analogy.json").read_text(encoding="utf-8"))
    assert analogy["contract"]["schema_version"] == "tropicalgt.analogical_simplex_tree_analogy.v1"
    assert analogy["contract"]["available"] is False
    assert analogy["contract"]["pair_count"] == 0
    assert analogy["contract"]["no_proxy_or_fallback"] is True
    assert analogy["contract"]["renders_interactive_plotly_table"] is True
    analogy_html = (audit / "analogical_simplex_tree_analogy.html").read_text(encoding="utf-8")
    assert "script src" in analogy_html
    assert "plotly.min.js" in analogy_html
    assert "Plotly.newPlot" in analogy_html
    assert "finite simplex-tree rows" in analogy_html
    assert "preserved face-to-coface chains" in analogy_html
    assert (audit / "analogical_memory_topk_index.html").exists()
    assert (audit / "analogical_memory_map_02.html").exists()
    dashboard = (audit / "inference_audit.html").read_text(encoding="utf-8")
    assert "analogical_simplex_tree_analogy.html" in dashboard


def test_backfill_regenerates_tropical_support_contracts_from_stored_payload(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    tokens = [
        {
            "index": 0,
            "text": "graph root token",
            "kind": "graph",
            "node_type": "graph",
            "active_support_index": 0,
            "margin": 0.0004,
            "active_support_probability": 0.91,
            "support_probability_entropy_bits": 0.3,
            "top_model_support_probabilities": [{"index": 0, "probability": 0.91}],
            "support_probability_source": "model_tropical_support_probabilities",
        },
        {
            "index": 1,
            "text": "problem token",
            "kind": "node",
            "node_type": "problem",
            "active_support_index": 0,
            "margin": 0.004,
            "active_support_probability": 0.81,
            "support_probability_entropy_bits": 0.5,
            "top_model_support_probabilities": [{"index": 0, "probability": 0.81}],
            "support_probability_source": "model_tropical_support_probabilities",
        },
        {
            "index": 2,
            "text": "answer token",
            "kind": "node",
            "node_type": "answer",
            "active_support_index": 2,
            "margin": 0.12,
            "active_support_probability": 0.72,
            "support_probability_entropy_bits": 0.7,
            "top_model_support_probabilities": [{"index": 2, "probability": 0.72}],
            "support_probability_source": "model_tropical_support_probabilities",
        },
        {
            "index": 3,
            "text": "edge token",
            "kind": "edge",
            "edge_type": "reason",
            "active_support_index": 2,
            "margin": 0.2,
            "active_support_probability": 0.64,
            "support_probability_entropy_bits": 0.9,
            "top_model_support_probabilities": [{"index": 2, "probability": 0.64}],
            "support_probability_source": "model_tropical_support_probabilities",
        },
    ]
    (audit / "tropical_support_payload.json").write_text(
        json.dumps({"metrics": {"available": True, "token_count": len(tokens), "invalid_support_count": 0}, "tokens": tokens}),
        encoding="utf-8",
    )
    (audit / "tropical_support_heatmap.html").write_text("static legacy tropical support page", encoding="utf-8")

    report = module.backfill_audit_root(audit)

    kinds = {row["kind"] for row in report["actions"]}
    assert "tropical_support_contract_backfill" in kinds
    payload = json.loads((audit / "tropical_support_payload.json").read_text(encoding="utf-8"))
    render_contract = payload["tropical_support_render_contract"]
    readability = payload["tropical_support_readability_contract"]
    assert render_contract["schema_version"] == "tropicalgt.tropical_support_render.v1"
    assert render_contract["no_proxy_or_fallback"] is True
    assert render_contract["assignment_matrix_binary"] is True
    assert render_contract["support_columns_policy"] == "observed_valid_active_support_indices_only"
    assert render_contract["normal_fan_wall_crossing_certified"] is False
    assert render_contract["token_count"] == len(tokens)
    assert readability["schema_version"] == "tropicalgt.tropical_support_readability.v1"
    assert readability["no_proxy_or_fallback"] is True
    assert readability["panels_are_separate"] is True
    assert readability["assignment_and_margin_panels_separated"] is True
    assert payload["metrics"]["render_contract"].startswith("assignment_matrix is binary model argmax support")
    assert len(payload["support_assignment_status_by_token"]) == len(tokens)
    assert len(payload["support_flow_edges"]) == len(tokens)
    html = (audit / "tropical_support_heatmap.html").read_text(encoding="utf-8")
    assert "plotly.min.js" in html
    assert "Plotly.newPlot" in html
    assert "tropical_support_render_contract" in html
    assert "tropical_support_readability_contract" in html
    assert module._tropical_support_contracts_need_backfill(audit) is False


def test_backfill_writes_unavailable_persistence_landscape_contract_from_no_finite_intervals(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    (audit / "trajectory_topological_algebra.json").write_text(
        json.dumps(
            {
                "persistence": {
                    "backend": "gudhi",
                    "available": True,
                    "intervals": [
                        {"dimension": 0, "birth": 0.0, "death": None, "infinite": True},
                        {"dimension": 1, "birth": 0.0, "death": None, "infinite": True},
                    ],
                },
                "persistence_representations": {
                    "available": False,
                    "backend": "gudhi.representations",
                    "reason": "no finite persistence intervals",
                },
            }
        ),
        encoding="utf-8",
    )

    report = module.backfill_audit_root(audit)

    kinds = {row["kind"] for row in report["actions"]}
    assert "persistence_landscape_contract_backfill" in kinds
    payload = json.loads((audit / "trajectory_persistence" / "persistence_landscapes.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "tropicalgt.persistence_landscape_visual_contract.v1"
    assert payload["available"] is False
    assert payload["actual_data_only"] is True
    assert payload["no_proxy_or_fallback"] is True
    assert payload["not_nll_fitness_landscape"] is True
    assert payload["finite_persistence_interval_count"] == 0
    assert payload["unavailable_state_verified_by_intervals"] is True
    assert "no_finite_persistence_intervals_for_gudhi_landscape" in payload["unavailable_reasons"]
    html = (audit / "trajectory_persistence" / "persistence_landscapes.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "No finite persistence intervals" in html
    assert module._persistence_landscapes_need_backfill(audit) is False


def test_backfill_hydrates_stale_unavailable_cas_resolution_guard(tmp_path: Path):
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
                "grid_axes": [[0], [0]],
                "levels": [0],
                "radii": [0.0],
                "radius_grade_policy": "exact_sorted_radius_grid_index_no_bucket_collision",
                "fiber_rank_profile": [{"grade": [0, 0], "betti": {"0": 1}}],
                "chain_module_generators": [{"multidegree": [0, 0], "homological_degree": 0, "simplex": ["root"]}],
                "boundary_monomials": {},
                "structure_maps": [],
                "chain_presentation_diagnostics": {
                    "ring": "F2[x_level,x_radius]",
                    "real_free_resolution_certified": False,
                    "real_free_resolution": {
                        "schema_version": "tropicalgt.real_free_resolution.v1",
                        "available": False,
                        "status": "certificate_failed",
                        "reason": "legacy stale unavailable guard",
                        "coefficient_ring": "F2[x_level,x_radius]",
                        "module_schema_version": "tropicalgt.level_radius_module.v1",
                        "input_sha256": "abc",
                        "backend_attempts": [{"backend": "M2", "status": "skipped_complexity_guard"}],
                        "backend_probe": {"backends": [], "preferred_order": ["M2", "sage", "Singular"]},
                        "bemultipliers_probe": {"is_resolution_backend": False},
                        "cas_artifacts": {},
                        "command_templates": {},
                        "certificate_attached": False,
                        "real_free_resolution_certified": False,
                        "total_graded_resolution_certified": False,
                        "ungraded_resolution_certified": False,
                        "multigraded_free_resolution_certified": False,
                        "exactness_certified": False,
                        "minimality_certified": False,
                        "safe_to_render_as_real_free_resolution": False,
                        "safe_to_render_as_total_graded_resolution": False,
                        "safe_to_render_as_multigraded_free_resolution": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = module.backfill_audit_root(audit)

    kinds = {row["kind"] for row in report["actions"]}
    assert "cas_resolution_guard_contract_backfill" in kinds
    payload = json.loads(raw.read_text(encoding="utf-8"))
    real = payload["chain_presentation_diagnostics"]["real_free_resolution"]
    assert real["certificate_contract"]["schema_version"] == "tropicalgt.cas_free_resolution_contract.v1"
    assert real["certificate_contract"]["no_proxy_or_fallback"] is True
    assert real["paper_method_contract"]["schema_version"] == "tropicalgt.be_fitting_method_contract.v1"
    assert real["paper_method_contract"]["arxiv_id"] == "2210.11433v1"
    assert real["cas_execution_manifest"]["schema_version"] == "tropicalgt.cas_execution_manifest.v1"
    assert real["cas_execution_manifest"]["no_proxy_or_fallback"] is True
    assert real["cas_execution_manifest"]["backend_entries"]
    assert real["safe_unavailable_render"] is True
    assert real["unavailable_diagnostic"]["safe_to_render_only_as_unavailable"] is True
    assert "Do not substitute chain diagnostics" in real["unavailable_diagnostic"]["no_proxy_policy"]
    assert module._real_resolution_guard_needs_backfill(real) is False


def test_backfill_regenerates_stale_graphcg_direction_contracts_from_scaling_tree(tmp_path: Path):
    module = _load_backfill()
    audit = tmp_path / "got_audit"
    audit.mkdir()
    (audit / "inference_scaling_tree.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "record_id": "q0",
                        "level": 0,
                        "path": ["root"],
                        "score": 0.2,
                        "nll": 1.0,
                        "graphcg_projection": {
                            "basis": "effective_full_rank_qr",
                            "all_direction_cosines": [0.1, -0.2, 0.3],
                            "mean_abs_offdiag_cosine": 0.01,
                            "max_abs_offdiag_cosine": 0.02,
                        },
                    },
                    {
                        "record_id": "q1",
                        "level": 1,
                        "path": ["root", "accept"],
                        "score": 0.4,
                        "nll": 0.8,
                        "graphcg_projection": {
                            "basis": "effective_full_rank_qr",
                            "all_direction_cosines": [0.4, 0.0, -0.5],
                            "mean_abs_offdiag_cosine": 0.03,
                            "max_abs_offdiag_cosine": 0.04,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (audit / "graphcg_direction_cosines_payload.json").write_text(
        json.dumps(
            {
                "available": True,
                "matrix_shape": [2, 3],
                "mean_abs": [0.25, 0.1, 0.4],
                "signed_mean": [0.25, -0.1, -0.1],
            }
        ),
        encoding="utf-8",
    )
    (audit / "graphcg_direction_cosines.html").write_text("<html>legacy static graphcg</html>", encoding="utf-8")

    assert module._graphcg_contracts_need_backfill(audit) is True
    report = module.backfill_audit_root(audit)

    kinds = {row["kind"] for row in report["actions"]}
    assert "graphcg_direction_contract_backfill" in kinds
    payload = json.loads((audit / "graphcg_direction_cosines_payload.json").read_text(encoding="utf-8"))
    assert payload["graphcg_readability_contract"]["schema_version"] == "tropicalgt.graphcg_direction_readability.v1"
    assert payload["graphcg_readability_contract"]["all_model_directions_rendered"] is True
    assert payload["graphcg_readability_contract"]["directions_sampled_for_heatmap"] is False
    evidence = payload["graphcg_direction_evidence_contract"]
    assert evidence["schema_version"] == "tropicalgt.graphcg_direction_evidence.v1"
    assert evidence["source"] == "candidate.graphcg_projection.all_direction_cosines"
    assert evidence["no_proxy_or_fallback"] is True
    assert evidence["direction_count"] == 3
    assert evidence["direction_row_count"] == 3
    assert evidence["safe_to_render_full_rank_direction_evidence"] is True
    assert [row["direction_id"] for row in payload["direction_rows"]] == [0, 1, 2]
    assert all(row["rendered_in_all_direction_heatmap"] for row in payload["direction_rows"])
    assert all(row["rendered_in_full_rank_activity_spectrum"] for row in payload["direction_rows"])
    assert all(row["rendered_in_signed_bias_panel"] for row in payload["direction_rows"])
    assert payload["top_active_direction_rows"]
    assert len(payload["candidate_hover_rows"]) == 2
    html = (audit / "graphcg_direction_cosines.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "plotly.min.js" in html
    assert "GraphCG full-rank direction audit" in html
    assert "Readable full-rank heatmap" in html
    assert module._graphcg_contracts_need_backfill(audit) is False
