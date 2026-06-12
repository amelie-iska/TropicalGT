import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
        "browser_static_preview_rendering_fallback",
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
    assert registry["browser_static_preview_rendering_fallback"]["kind"] == "rendering_fallback"


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


def test_metric_provenance_audit_covers_webgl_rendering_fallback_without_metric_substitution(tmp_path: Path):
    source = tmp_path / "viz.py"
    source.write_text(
        "fallback.className = 'webgl-fallback main-fallback'\n"
        "fallback.innerHTML = 'WebGL unavailable: static complex preview shown from the selected real filtered-complex payload'\n"
        "function staticFallbackMarkup(item, reason) { return reason }\n",
        encoding="utf-8",
    )

    report = write_provenance_audit(
        [source],
        tmp_path / "provenance.json",
        tmp_path / "provenance.md",
    )

    assert report["finding_count"] == 3
    assert report["uncovered_finding_count"] == 0
    assert {row["matched_entry"] for row in report["covered_findings"]} == {
        "browser_static_preview_rendering_fallback",
    }
    registry = provenance_by_name()
    guardrail = registry["browser_static_preview_rendering_fallback"]["replacement_or_guardrail"]
    assert "never substitutes model probabilities" in guardrail


def _audit_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "audit_metric_provenance.py"


def _load_audit_script_module():
    spec = importlib.util.spec_from_file_location("audit_metric_provenance_script", _audit_script_path())
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_metric_provenance_script_covers_webgl_static_preview_only(tmp_path: Path):
    module = _load_audit_script_module()
    source = tmp_path / "TropicalGT-I" / "src" / "tropicalgt" / "visualization.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        ".webgl-fallback { color: white; }\n"
        "fallback.innerHTML = `<h2>WebGL unavailable: static complex preview shown</h2><p>The static preview below comes from the selected real filtered-complex payload.</p>`\n"
        "fake_metric_fallback = 'unregistered metric fallback'\n",
        encoding="utf-8",
    )
    report = write_provenance_audit(
        [tmp_path / "TropicalGT-I" / "src" / "tropicalgt"],
        tmp_path / "provenance.json",
        tmp_path / "provenance.md",
    )

    covered = module._apply_script_local_rendering_coverage(report)

    assert any(
        row.get("matched_entry") == "browser_static_preview_rendering_fallback"
        for row in covered["covered_findings"]
    )
    assert covered["uncovered_finding_count"] == 1
    assert covered["uncovered_findings"][0]["text"] == "fake_metric_fallback = 'unregistered metric fallback'"


def test_audit_metric_provenance_script_fail_gate_accepts_static_preview(tmp_path: Path):
    source = tmp_path / "TropicalGT-I" / "src" / "tropicalgt" / "visualization.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "function renderPanelStaticFallback(item, reason) { return staticFallbackMarkup(item, reason); }\n"
        "fallback.className = 'webgl-fallback main-fallback';\n"
        "fallback.innerHTML = 'WebGL unavailable: static complex preview shown from the selected real filtered-complex payload.';\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_audit_script_path()),
            "--scan",
            str(source),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(tmp_path / "audit.md"),
            "--fail-on-uncovered",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["uncovered_finding_count"] == 0
    assert {row["matched_entry"] for row in report["covered_findings"]} == {"browser_static_preview_rendering_fallback"}
