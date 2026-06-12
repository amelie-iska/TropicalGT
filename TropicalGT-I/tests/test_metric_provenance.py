from pathlib import Path

from tropicalgt.provenance import provenance_by_name, write_provenance_audit


def test_metric_provenance_registry_covers_current_risky_terms(tmp_path: Path):
    registry = provenance_by_name()
    for key in [
        "bpb",
        "bpb_proxy",
        "smooth_projected_nll_fitness_landscape",
        "local_interpolating_nll_sheet",
        "persistence_landscape",
        "multiparameter_free_resolution_proxy",
        "graphcg_direction_svd_condition_proxy",
        "synthetic_h0_fallback",
        "json_fallback_graph_trace",
        "parameter_golf_token_id_fallback",
        "training_data_budget_estimate",
        "config_default_fallback",
        "simplicial_projection_feature_fallback",
        "gudhi_vectorizer_autograd_boundary",
    ]:
        assert key in registry
        assert registry[key]["kind"]
        assert registry[key]["replacement_or_guardrail"]
        assert "match_terms" in registry[key]
    assert registry["bpb"]["optimize_directly"] is True
    assert registry["bpb_proxy"]["kind"] == "legacy_exact_alias"
    assert registry["smooth_projected_nll_fitness_landscape"]["optimize_directly"] is False
    assert registry["persistence_landscape"]["kind"] == "fast_vectorized_topology"


def test_metric_provenance_audit_writes_json_and_markdown(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text(
        "graphcg_direction_svd_condition_proxy = 1.0\n"
        "surrogate_name = 'smooth_projected_nll_fitness_landscape'\n",
        encoding="utf-8",
    )
    report = write_provenance_audit(
        [source],
        tmp_path / "provenance.json",
        tmp_path / "provenance.md",
    )
    assert report["finding_count"] == 2
    assert report["uncovered_finding_count"] == 0
    assert {row["matched_entry"] for row in report["covered_findings"]} == {
        "graphcg_direction_svd_condition_proxy",
        "smooth_projected_nll_fitness_landscape",
    }
    assert "graphcg_direction_svd_condition_proxy" in (tmp_path / "provenance.md").read_text(encoding="utf-8")


def test_metric_provenance_audit_classifies_aliases_and_excludes_self(tmp_path: Path):
    source = tmp_path / "source.py"
    self_file = tmp_path / "provenance.py"
    source.write_text(
        "metadata['graph_json_fallback'] = True\n"
        "coords, fallback_stats = _feature_pca3_with_jitter(features, labels)\n"
        "autograd_note = 'GUDHI vectorizers are NumPy/scikit-learn transforms unless replaced by a torch-native differentiable surrogate'\n",
        encoding="utf-8",
    )
    self_file.write_text("RISK_WORDS = ('proxy', 'surrogate', 'fallback')\n", encoding="utf-8")
    report = write_provenance_audit(
        [tmp_path],
        tmp_path / "provenance.json",
        tmp_path / "provenance.md",
        excluded_suffixes=("provenance.py",),
    )
    assert report["finding_count"] == 3
    assert report["uncovered_finding_count"] == 0
    assert {row["matched_entry"] for row in report["covered_findings"]} == {
        "json_fallback_graph_trace",
        "simplicial_projection_feature_fallback",
        "gudhi_vectorizer_autograd_boundary",
    }
