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
    ]:
        assert key in registry
        assert registry[key]["kind"]
        assert registry[key]["replacement_or_guardrail"]
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
    assert "graphcg_direction_svd_condition_proxy" in (tmp_path / "provenance.md").read_text(encoding="utf-8")
