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
    certificate_shift: float = 0.0,
    wall_shift: float = 0.0,
) -> Path:
    history = []
    for step in range(6):
        history.append(
            {
                "step": step + 1,
                "bpb": 6.0 - 0.2 * step + offset,
                "graph_bpb": 9.0 - 0.1 * step + offset,
                "margin_mean": 0.1 + 0.2 * step + max(certificate_shift, 0.0),
                "support_entropy": 2.0 - 0.1 * step,
                "graphcg_loss": 0.5 + 0.05 * step,
                "certificate_agreement": 0.70 + 0.01 * step + certificate_shift,
                "certificate_disallowed_support_rate": 0.20 - 0.01 * step - certificate_shift,
                "certificate_loss": 0.40 - 0.02 * step - certificate_shift,
                "strict_wall_hit_rate": 0.10 + wall_shift,
                "near_wall_hit_rate": 0.25 + wall_shift,
            }
        )
    final_metrics = history[-1]
    report = {
        "checkpoint": str(path.with_suffix(".pt")),
        "final_step": final_step or len(history),
        "device": "cpu",
        "seed": seed,
        "ablation_variant": ablation_variant,
        "ablation_overrides": ablation_overrides or {},
        "history": history,
        "metrics": final_metrics,
        "eval": {
            "bpb": final_metrics["bpb"] + 0.1,
            "graph_bpb": final_metrics["graph_bpb"] + 0.2,
            "certificate_agreement": final_metrics["certificate_agreement"],
            "certificate_coverage": final_metrics["certificate_agreement"] - 0.05,
            "certificate_disallowed_support_rate": final_metrics["certificate_disallowed_support_rate"],
            "certificate_loss": final_metrics["certificate_loss"],
            "certificate_objective_loss": final_metrics["certificate_loss"] + 0.02,
            "certificate_diagnostic_penalty": max(0.0, 0.04 - certificate_shift),
            "margin_mean": final_metrics["margin_mean"],
            "sequence_tropical_margin_mean": final_metrics["margin_mean"] + 0.01,
            "wall_hit_rate": final_metrics["strict_wall_hit_rate"],
            "strict_wall_hit_rate": final_metrics["strict_wall_hit_rate"],
            "near_wall_hit_rate": final_metrics["near_wall_hit_rate"],
            "near_wall_only_rate": max(final_metrics["near_wall_hit_rate"] - final_metrics["strict_wall_hit_rate"], 0.0),
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
        certificate_shift=0.05,
        wall_shift=-0.03,
    )
    third = _write_report(
        tmp_path / "third_train_report.json",
        offset=0.4,
        seed=999,
        ablation_variant="chart_bundle_toric_bad_seed",
        ablation_overrides={"model.bundle_transport_weight": 0.0001},
        certificate_shift=0.02,
        wall_shift=-0.01,
    )
    fourth = _write_report(
        tmp_path / "guardrail_regression_train_report.json",
        offset=-0.1,
        ablation_variant="aux_guardrail_regression",
        ablation_overrides={"model.gflownet_weight": 0.01, "model.certificate_weight": 0.001},
        certificate_shift=-0.20,
        wall_shift=0.08,
    )
    report = build_bpb_ablation_report([first, second, third, fourth], top_k=20)
    assert report["baseline"].endswith("baseline_train_report.json")
    assert len(report["runs"]) == 4
    assert report["deltas_vs_baseline"][1]["deltas"]["delta_bpb"] < 0
    best = {row["target"]: row for row in report["best_by_target"]}
    assert best["bpb"]["name"].endswith("variant_train_report")
    assert best["bpb"]["delta_vs_baseline"] < 0
    assert best["eval_graph_bpb"]["runner_up_name"].endswith("guardrail_regression_train_report")
    ranked = {(row["target"], row["metric"]): row for row in report["aggregate_metric_rankings"]}
    assert ("bpb", "margin_mean") in ranked
    assert ranked[("bpb", "margin_mean")]["mean_spearman"] < 0
    assert any(row["target"] == "eval_bpb" and row.get("scope") == "final" for row in report["history_correlations"])
    gate = report["advanced_auxiliary_promotion_gate"]
    assert gate["policy"] == "no_proxy_no_fallback_matched_seed_eval_bpb_eval_graph_bpb_certificate_and_tropical_wall_required_before_promoting_advanced_auxiliary_coefficients"
    assert gate["candidate_count"] == 3
    assert gate["promotable_count"] == 1
    assert gate["promotable_variants"] == ["chart_bundle_toric_0p1x"]
    rows = {row["ablation_variant"]: row for row in gate["rows"] if row.get("advanced_auxiliary_coefficients")}
    assert rows["chart_bundle_toric_0p1x"]["status"] == "promotable_matched_eval_bpb_eval_graph_bpb_certificate_tropical_wall_safe"
    assert rows["chart_bundle_toric_0p1x"]["required_target_deltas"]["eval_bpb"] < 0.0
    assert rows["chart_bundle_toric_0p1x"]["required_target_deltas"]["eval_graph_bpb"] < 0.0
    assert rows["chart_bundle_toric_0p1x"]["guardrail_status"]["safe_for_promotion"] is True
    assert rows["chart_bundle_toric_bad_seed"]["status"] == "blocked_unmatched_ablation_run"
    assert any("seed mismatch" in issue for issue in rows["chart_bundle_toric_bad_seed"]["match_issues"])
    assert rows["aux_guardrail_regression"]["status"] == "blocked_guardrail_regression_or_missing_certificate_tropical_evidence"
    assert rows["aux_guardrail_regression"]["required_target_deltas"]["eval_bpb"] < 0.0
    assert rows["aux_guardrail_regression"]["guardrail_status"]["safe_for_promotion"] is False
    assert set(rows["aux_guardrail_regression"]["guardrail_status"]["failed_groups"]) == {"certificate", "tropical_wall"}
    cert_regressions = rows["aux_guardrail_regression"]["guardrail_status"]["groups"]["certificate"]["regressions"]
    wall_regressions = rows["aux_guardrail_regression"]["guardrail_status"]["groups"]["tropical_wall"]["regressions"]
    assert any("certificate_agreement" in item for item in cert_regressions)
    assert any("strict_wall_hit_rate" in item for item in wall_regressions)


def test_bpb_ablation_promotion_blocks_match_contract_mismatch(tmp_path: Path):
    baseline = _write_report(tmp_path / "baseline_train_report.json")
    candidate = _write_report(
        tmp_path / "candidate_train_report.json",
        offset=-0.2,
        ablation_variant="matched_contract_mismatch",
        ablation_overrides={"model.graphcg_weight": 0.01},
    )
    baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    base_contract = {
        "schema_version": "tropicalgt.bpb_ablation_match_contract.v1",
        "match_group_id": "group_a",
        "boundary_steps": 5000,
        "requested_max_steps": 5000,
        "seed": 123,
        "base_config_fingerprint": "abc",
        "data_root": "/data/train",
        "require_data": True,
        "train_limit": None,
        "val_limit": None,
        "graph_bpb_side_weight": 1.0,
    }
    baseline_payload["ablation_match_contract"] = dict(base_contract)
    candidate_contract = dict(base_contract)
    candidate_contract["match_group_id"] = "group_b"
    candidate_payload["ablation_match_contract"] = candidate_contract
    baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")

    report = build_bpb_ablation_report([baseline, candidate])

    row = next(row for row in report["advanced_auxiliary_promotion_gate"]["rows"] if row.get("ablation_variant") == "matched_contract_mismatch")
    assert row["status"] == "blocked_unmatched_ablation_run"
    assert row["guardrail_status"]["safe_for_promotion"] is True
    assert any("ablation_match_contract match_group_id mismatch" in issue for issue in row["match_issues"])


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
