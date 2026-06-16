import json
from pathlib import Path

from tropicalgt.ablation import build_bpb_ablation_report, write_bpb_ablation_artifacts


def _write_report(
    path: Path,
    offset: float = 0.0,
    *,
    seed: int = 123,
    final_step: int | None = None,
    ablation_variant: str | None = None,
    ablation_overrides: dict[str, float] | None = None,
) -> Path:
    history = []
    for step in range(6):
        history.append(
            {
                "step": step + 1,
                "bpb": 6.0 - 0.2 * step + offset,
                "graph_bpb": 9.0 - 0.1 * step + offset,
                "margin_mean": 0.1 + 0.2 * step,
                "support_entropy": 2.0 - 0.1 * step,
                "graphcg_loss": 0.5 + 0.05 * step,
            }
        )
    report = {
        "checkpoint": str(path.with_suffix(".pt")),
        "final_step": final_step or len(history),
        "device": "cpu",
        "seed": seed,
        "ablation_variant": ablation_variant,
        "ablation_overrides": ablation_overrides or {},
        "history": history,
        "metrics": history[-1],
        "eval": {
            "bpb": history[-1]["bpb"] + 0.1,
            "graph_bpb": history[-1]["graph_bpb"] + 0.2,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_bpb_ablation_report_ranks_correlated_metrics(tmp_path: Path):
    first = _write_report(tmp_path / "baseline_train_report.json", offset=0.0)
    second = _write_report(
        tmp_path / "variant_train_report.json",
        offset=-0.3,
        ablation_variant="chart_bundle_toric_0p1x",
        ablation_overrides={"model.bundle_transport_weight": 0.0001, "model.chart_bpb_consistency_weight": 0.0001},
    )
    third = _write_report(
        tmp_path / "third_train_report.json",
        offset=0.4,
        seed=999,
        ablation_variant="chart_bundle_toric_bad_seed",
        ablation_overrides={"model.bundle_transport_weight": 0.0001},
    )
    report = build_bpb_ablation_report([first, second, third], top_k=20)
    assert report["baseline"].endswith("baseline_train_report.json")
    assert len(report["runs"]) == 3
    assert report["deltas_vs_baseline"][1]["deltas"]["delta_bpb"] < 0
    best = {row["target"]: row for row in report["best_by_target"]}
    assert best["bpb"]["name"].endswith("variant_train_report")
    assert best["bpb"]["delta_vs_baseline"] < 0
    assert best["eval_graph_bpb"]["runner_up_name"].endswith("baseline_train_report")
    ranked = {(row["target"], row["metric"]): row for row in report["aggregate_metric_rankings"]}
    assert ("bpb", "margin_mean") in ranked
    assert ranked[("bpb", "margin_mean")]["mean_spearman"] < 0
    assert any(row["target"] == "eval_bpb" and row.get("scope") == "final" for row in report["history_correlations"])
    gate = report["advanced_auxiliary_promotion_gate"]
    assert gate["policy"] == "no_proxy_no_fallback_matched_seed_eval_bpb_and_eval_graph_bpb_required_before_promoting_advanced_auxiliary_coefficients"
    assert gate["candidate_count"] == 2
    assert gate["promotable_count"] == 1
    assert gate["promotable_variants"] == ["chart_bundle_toric_0p1x"]
    rows = {row["ablation_variant"]: row for row in gate["rows"] if row.get("advanced_auxiliary_coefficients")}
    assert rows["chart_bundle_toric_0p1x"]["status"] == "promotable_matched_eval_bpb_eval_graph_bpb_improvement"
    assert rows["chart_bundle_toric_0p1x"]["required_target_deltas"]["eval_bpb"] < 0.0
    assert rows["chart_bundle_toric_0p1x"]["required_target_deltas"]["eval_graph_bpb"] < 0.0
    assert rows["chart_bundle_toric_bad_seed"]["status"] == "blocked_unmatched_ablation_run"
    assert any("seed mismatch" in issue for issue in rows["chart_bundle_toric_bad_seed"]["match_issues"])


def test_bpb_ablation_artifacts_write_json_markdown_and_html(tmp_path: Path):
    report_path = _write_report(tmp_path / "train_report.json")
    paths = write_bpb_ablation_artifacts([report_path], tmp_path / "ablation", render_html=True)
    assert Path(paths["json"]).exists()
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert markdown.startswith("# TropicalGT-I BPB")
    assert "## Best By Target" in markdown
    assert "## Advanced Auxiliary Promotion Gate" in markdown
    html = Path(paths["html"]).read_text(encoding="utf-8")
    assert "color-scheme: dark" in html
    assert "BPB/graph-BPB metric correlation screen" in html
