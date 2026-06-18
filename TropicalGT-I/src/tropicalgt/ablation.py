from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TARGETS = ("bpb", "graph_bpb", "eval_bpb", "eval_graph_bpb")
PROMOTION_REQUIRED_TARGETS = ("eval_bpb", "eval_graph_bpb")
ADVANCED_AUXILIARY_COEFFICIENT_KEYS = (
    "gflownet_weight",
    "graphcg_weight",
    "margin_weight",
    "entropy_weight",
    "certificate_weight",
    "sequence_tropical_weight",
    "bundle_transport_weight",
    "bundle_cocycle_weight",
    "bundle_flat_rank_weight",
    "toric_normal_fan_weight",
    "graphcg_toric_cell_agreement_weight",
    "chart_bpb_consistency_weight",
    "bundle_atom_stability_weight",
)
PROMOTION_GUARDRAIL_GROUPS = {
    "certificate": (
        ("eval_certificate_agreement", "higher"),
        ("eval_certificate_coverage", "higher"),
        ("eval_certificate_disallowed_support_rate", "lower"),
        ("eval_certificate_loss", "lower"),
        ("eval_certificate_objective_loss", "lower"),
        ("eval_certificate_diagnostic_penalty", "lower"),
    ),
    "tropical_wall": (
        ("eval_margin_mean", "higher"),
        ("eval_sequence_tropical_margin_mean", "higher"),
        ("eval_wall_hit_rate", "lower"),
        ("eval_strict_wall_hit_rate", "lower"),
        ("eval_near_wall_hit_rate", "lower"),
        ("eval_near_wall_only_rate", "lower"),
    ),
}


@dataclass(frozen=True)
class ReportBundle:
    path: Path
    report: dict[str, Any]

    @property
    def name(self) -> str:
        metrics = self.report.get("metrics", {})
        checkpoint = self.report.get("checkpoint") or self.path.stem
        return str(metrics.get("run_name") or Path(str(checkpoint)).stem or self.path.stem)


def load_report_bundle(path: str | Path) -> ReportBundle:
    p = Path(path)
    return ReportBundle(path=p, report=json.loads(p.read_text(encoding="utf-8")))


def build_bpb_ablation_report(
    report_paths: Iterable[str | Path],
    targets: Iterable[str] = DEFAULT_TARGETS,
    baseline: str | Path | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    bundles = [load_report_bundle(path) for path in report_paths]
    if not bundles:
        raise ValueError("at least one train_report.json path is required")
    target_names = tuple(targets)
    baseline_bundle = _select_baseline(bundles, baseline)
    runs = [_run_row(bundle, target_names) for bundle in bundles]
    correlations = []
    for bundle in bundles:
        correlations.extend(_history_correlations(bundle, target_names))
    correlations.extend(_final_metric_correlations(bundles, target_names))
    aggregate = _aggregate_correlations(correlations, top_k=top_k)
    deltas = [_delta_row(baseline_bundle, bundle, target_names) for bundle in bundles]
    baseline_metrics = _flatten_report_metrics(baseline_bundle.report)
    promotion_gate = _advanced_auxiliary_promotion_gate(baseline_bundle, bundles)
    return {
        "version": 1,
        "targets": list(target_names),
        "baseline": str(baseline_bundle.path),
        "runs": runs,
        "deltas_vs_baseline": deltas,
        "best_by_target": _best_by_target(runs, target_names, baseline_metrics),
        "history_correlations": correlations,
        "aggregate_metric_rankings": aggregate,
        "advanced_auxiliary_promotion_gate": promotion_gate,
        "interpretation": {
            "primary_metric": "bpb",
            "graph_primary_metric": "graph_bpb",
            "warning": "Per-step correlations use train history; eval targets are screened across matched final reports when at least three runs are available. Promote a component only after matched-seed ablations improve held-out BPB and graph-BPB without regressing certificate or tropical-wall guardrails.",
        },
    }


def write_bpb_ablation_artifacts(
    report_paths: Iterable[str | Path],
    output_dir: str | Path,
    targets: Iterable[str] = DEFAULT_TARGETS,
    baseline: str | Path | None = None,
    top_k: int = 20,
    render_html: bool = True,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_bpb_ablation_report(report_paths, targets=targets, baseline=baseline, top_k=top_k)
    json_path = output / "bpb_ablation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = output / "bpb_ablation_report.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    paths = {"json": str(json_path), "markdown": str(md_path)}
    if render_html:
        html_path = output / "bpb_metric_correlations.html"
        _write_correlation_html(report, html_path)
        paths["html"] = str(html_path)
    return paths


def _run_row(bundle: ReportBundle, targets: tuple[str, ...]) -> dict[str, Any]:
    metrics = _flatten_report_metrics(bundle.report)
    return {
        "name": bundle.name,
        "path": str(bundle.path),
        "seed": bundle.report.get("seed"),
        "final_step": bundle.report.get("final_step"),
        "device": bundle.report.get("device"),
        "ablation_variant": bundle.report.get("ablation_variant"),
        "ablation_overrides": bundle.report.get("ablation_overrides", {}),
        "ablation_match_contract": bundle.report.get("ablation_match_contract", {}),
        "advanced_auxiliary_coefficients": _nonzero_advanced_auxiliary_coefficients(bundle.report),
        "targets": {target: metrics.get(target) for target in targets},
        "metrics": {key: value for key, value in metrics.items() if _is_finite_number(value)},
        "sampler": bundle.report.get("sampler", {}),
        "analogical_memory": bundle.report.get("analogical_memory", {}),
    }


def _delta_row(baseline: ReportBundle, bundle: ReportBundle, targets: tuple[str, ...]) -> dict[str, Any]:
    base = _flatten_report_metrics(baseline.report)
    current = _flatten_report_metrics(bundle.report)
    deltas = {}
    for target in targets:
        if _is_finite_number(base.get(target)) and _is_finite_number(current.get(target)):
            deltas[f"delta_{target}"] = float(current[target]) - float(base[target])
    return {
        "name": bundle.name,
        "path": str(bundle.path),
        "baseline_path": str(baseline.path),
        "deltas": deltas,
        "improves_bpb": bool(deltas.get("delta_bpb", 0.0) < 0.0) if "delta_bpb" in deltas else None,
        "improves_graph_bpb": bool(deltas.get("delta_graph_bpb", 0.0) < 0.0) if "delta_graph_bpb" in deltas else None,
    }


def _advanced_auxiliary_promotion_gate(baseline: ReportBundle, bundles: list[ReportBundle]) -> dict[str, Any]:
    baseline_metrics = _flatten_report_metrics(baseline.report)
    baseline_seed = baseline.report.get("seed")
    baseline_step = baseline.report.get("final_step")
    baseline_manifest_hash = _stable_json_hash(baseline.report.get("dataset_manifest")) if baseline.report.get("dataset_manifest") else ""
    rows = []
    for bundle in bundles:
        if bundle.path == baseline.path:
            continue
        coeffs = _nonzero_advanced_auxiliary_coefficients(bundle.report)
        if not coeffs:
            rows.append(
                {
                    "name": bundle.name,
                    "path": str(bundle.path),
                    "ablation_variant": bundle.report.get("ablation_variant"),
                    "advanced_auxiliary_coefficients": {},
                    "promotable": False,
                    "status": "not_applicable_no_nonzero_advanced_auxiliary_coefficients",
                    "matched_run": False,
                    "match_issues": ["no nonzero chart-bundle/toric auxiliary coefficient in ablation_overrides"],
                    "required_target_deltas": {},
                    "policy": "telemetry-only or unrelated variants cannot promote advanced coefficients",
                }
            )
            continue
        current_metrics = _flatten_report_metrics(bundle.report)
        target_deltas = _target_deltas(baseline_metrics, current_metrics, PROMOTION_REQUIRED_TARGETS)
        guardrails = _promotion_guardrail_status(baseline_metrics, current_metrics)
        match_issues = _matched_run_issues(baseline, bundle, baseline_seed, baseline_step, baseline_manifest_hash)
        missing_targets = [target for target in PROMOTION_REQUIRED_TARGETS if target not in target_deltas]
        improves_required = bool(not missing_targets and all(target_deltas[target] < 0.0 for target in PROMOTION_REQUIRED_TARGETS))
        matched_run = not match_issues
        guardrails_safe = bool(guardrails["safe_for_promotion"])
        promotable = bool(matched_run and improves_required and guardrails_safe)
        if not matched_run:
            status = "blocked_unmatched_ablation_run"
        elif missing_targets:
            status = "blocked_missing_required_eval_bpb_eval_graph_bpb_deltas"
        elif not improves_required:
            status = "blocked_no_matched_eval_bpb_eval_graph_bpb_improvement"
        elif not guardrails_safe:
            status = "blocked_guardrail_regression_or_missing_certificate_tropical_evidence"
        else:
            status = "promotable_matched_eval_bpb_eval_graph_bpb_certificate_tropical_wall_safe"
        rows.append(
            {
                "name": bundle.name,
                "path": str(bundle.path),
                "ablation_variant": bundle.report.get("ablation_variant"),
                "advanced_auxiliary_coefficients": coeffs,
                "promotable": promotable,
                "status": status,
                "matched_run": matched_run,
                "match_issues": match_issues,
                "required_target_deltas": target_deltas,
                "missing_required_targets": missing_targets,
                "guardrail_status": guardrails,
                "policy": "promote nonzero advanced auxiliary coefficients only after matched-seed held-out eval BPB and eval graph-BPB both improve and certificate/tropical-wall guardrails have real no-regression evidence; otherwise keep them zero/default or telemetry-only",
            }
        )
    candidates = [row for row in rows if row.get("advanced_auxiliary_coefficients")]
    promotable = [row for row in candidates if row.get("promotable")]
    return {
        "available": bool(candidates),
        "policy": "no_proxy_no_fallback_matched_seed_eval_bpb_eval_graph_bpb_certificate_and_tropical_wall_required_before_promoting_advanced_auxiliary_coefficients",
        "baseline": str(baseline.path),
        "baseline_seed": baseline_seed,
        "baseline_final_step": baseline_step,
        "required_targets": list(PROMOTION_REQUIRED_TARGETS),
        "required_guardrail_groups": {group: [metric for metric, _direction in metrics] for group, metrics in PROMOTION_GUARDRAIL_GROUPS.items()},
        "advanced_auxiliary_coefficient_keys": list(ADVANCED_AUXILIARY_COEFFICIENT_KEYS),
        "candidate_count": len(candidates),
        "promotable_count": len(promotable),
        "promotable_variants": [str(row.get("ablation_variant") or row.get("name")) for row in promotable],
        "rows": rows,
        "unavailable_reason": None if candidates else "no matched reports with nonzero chart-bundle/toric auxiliary coefficients were provided",
    }


def _nonzero_advanced_auxiliary_coefficients(report: dict[str, Any]) -> dict[str, float]:
    overrides = report.get("ablation_overrides") if isinstance(report.get("ablation_overrides"), dict) else {}
    coeffs: dict[str, float] = {}
    for raw_key, value in overrides.items():
        key = str(raw_key)
        normalized = key[6:] if key.startswith("model.") else key
        if normalized not in ADVANCED_AUXILIARY_COEFFICIENT_KEYS:
            continue
        if _is_finite_number(value) and abs(float(value)) > 0.0:
            coeffs[key] = float(value)
    return coeffs


def _target_deltas(base: dict[str, float], current: dict[str, float], targets: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for target in targets:
        if _is_finite_number(base.get(target)) and _is_finite_number(current.get(target)):
            out[str(target)] = float(current[target]) - float(base[target])
    return out


def _promotion_guardrail_status(base: dict[str, float], current: dict[str, float]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    failed_groups: list[str] = []
    for group, metrics in PROMOTION_GUARDRAIL_GROUPS.items():
        rows = []
        regressions = []
        for metric, direction in metrics:
            if not (_is_finite_number(base.get(metric)) and _is_finite_number(current.get(metric))):
                rows.append({"metric": metric, "direction": direction, "available": False})
                continue
            baseline_value = float(base[metric])
            candidate_value = float(current[metric])
            delta = candidate_value - baseline_value
            if direction == "higher":
                regressed = delta < 0.0
            else:
                regressed = delta > 0.0
            row = {
                "metric": metric,
                "direction": direction,
                "available": True,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "regressed": bool(regressed),
            }
            rows.append(row)
            if regressed:
                regressions.append(f"{metric}:delta={delta:.6g}:direction={direction}")
        available_rows = [row for row in rows if row.get("available")]
        group_safe = bool(available_rows and not regressions)
        if not group_safe:
            failed_groups.append(group)
        groups[group] = {
            "safe": group_safe,
            "available_metric_count": len(available_rows),
            "missing_metrics": [row["metric"] for row in rows if not row.get("available")],
            "regressions": regressions,
            "metrics": rows,
            "policy": "at least one real metric in this group must be present for baseline and candidate, and no present metric may regress",
        }
    return {
        "safe_for_promotion": not failed_groups,
        "failed_groups": failed_groups,
        "groups": groups,
        "policy": "No proxy/fallback guardrail: advanced coefficients cannot be promoted if held-out eval certificate or tropical-wall evidence is missing or regresses.",
    }


def _matched_run_issues(
    baseline: ReportBundle,
    bundle: ReportBundle,
    baseline_seed: object,
    baseline_step: object,
    baseline_manifest_hash: str,
) -> list[str]:
    issues: list[str] = []
    if baseline_seed is None or bundle.report.get("seed") != baseline_seed:
        issues.append(f"seed mismatch: baseline={baseline_seed} candidate={bundle.report.get('seed')}")
    if baseline_step is None or bundle.report.get("final_step") != baseline_step:
        issues.append(f"final_step mismatch: baseline={baseline_step} candidate={bundle.report.get('final_step')}")
    candidate_manifest_hash = _stable_json_hash(bundle.report.get("dataset_manifest")) if bundle.report.get("dataset_manifest") else ""
    if baseline_manifest_hash and candidate_manifest_hash and candidate_manifest_hash != baseline_manifest_hash:
        issues.append("dataset_manifest mismatch")
    baseline_contract = baseline.report.get("ablation_match_contract") if isinstance(baseline.report.get("ablation_match_contract"), dict) else {}
    candidate_contract = bundle.report.get("ablation_match_contract") if isinstance(bundle.report.get("ablation_match_contract"), dict) else {}
    if baseline_contract:
        if not candidate_contract:
            issues.append("ablation_match_contract missing on candidate")
        else:
            for key in (
                "match_group_id",
                "boundary_steps",
                "requested_max_steps",
                "seed",
                "base_config_fingerprint",
                "data_root",
                "require_data",
                "train_limit",
                "val_limit",
                "graph_bpb_side_weight",
            ):
                if candidate_contract.get(key) != baseline_contract.get(key):
                    issues.append(
                        f"ablation_match_contract {key} mismatch: baseline={baseline_contract.get(key)} candidate={candidate_contract.get(key)}"
                    )
    return issues


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()


def _best_by_target(
    runs: list[dict[str, Any]],
    targets: tuple[str, ...],
    baseline_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    for target in targets:
        ranked = sorted(
            (
                (float((row.get("targets") or {}).get(target)), row)
                for row in runs
                if _is_finite_number((row.get("targets") or {}).get(target))
            ),
            key=lambda pair: pair[0],
        )
        if not ranked:
            continue
        value, row = ranked[0]
        baseline_value = baseline_metrics.get(target)
        delta_vs_baseline = value - float(baseline_value) if _is_finite_number(baseline_value) else None
        runner_up_value = ranked[1][0] if len(ranked) > 1 else None
        best_rows.append(
            {
                "target": target,
                "name": row.get("name"),
                "path": row.get("path"),
                "value": value,
                "baseline_value": float(baseline_value) if _is_finite_number(baseline_value) else None,
                "delta_vs_baseline": delta_vs_baseline,
                "improves_baseline": bool(delta_vs_baseline < 0.0) if delta_vs_baseline is not None else None,
                "runner_up_name": ranked[1][1].get("name") if len(ranked) > 1 else None,
                "runner_up_value": runner_up_value,
                "margin_to_runner_up": (runner_up_value - value) if runner_up_value is not None else None,
                "lower_is_better": True,
            }
        )
    return best_rows


def _history_correlations(bundle: ReportBundle, targets: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = bundle.report.get("history", [])
    if not isinstance(rows, list) or len(rows) < 3:
        return []
    numeric = _numeric_columns(rows)
    out = []
    for target in targets:
        if target not in numeric:
            continue
        target_values = numeric[target]
        for metric, values in numeric.items():
            if metric == target or metric.startswith("eval_"):
                continue
            pearson = _pearson(values, target_values)
            spearman = _spearman(values, target_values)
            if not math.isfinite(pearson) and not math.isfinite(spearman):
                continue
            out.append(
                {
                    "run": bundle.name,
                    "path": str(bundle.path),
                    "scope": "history",
                    "target": target,
                    "metric": metric,
                    "n": int(len(values)),
                    "pearson": pearson,
                    "spearman": spearman,
                    "direction": "helps_when_lower" if pearson > 0 else "helps_when_higher",
                }
            )
    return sorted(out, key=lambda row: abs(row.get("spearman", 0.0)), reverse=True)


def _final_metric_correlations(bundles: list[ReportBundle], targets: tuple[str, ...]) -> list[dict[str, Any]]:
    if len(bundles) < 3:
        return []
    rows = [_flatten_report_metrics(bundle.report) for bundle in bundles]
    numeric = _numeric_columns(rows)
    out = []
    for target in targets:
        if target not in numeric:
            continue
        target_values = numeric[target]
        for metric, values in numeric.items():
            if metric == target:
                continue
            pearson = _pearson(values, target_values)
            spearman = _spearman(values, target_values)
            if not math.isfinite(pearson) and not math.isfinite(spearman):
                continue
            out.append(
                {
                    "run": "matched_final_reports",
                    "path": "",
                    "scope": "final",
                    "target": target,
                    "metric": metric,
                    "n": int(np.isfinite(values).sum()),
                    "pearson": pearson,
                    "spearman": spearman,
                    "direction": "helps_when_lower" if pearson > 0 else "helps_when_higher",
                }
            )
    return sorted(out, key=lambda row: abs(row.get("spearman", 0.0)), reverse=True)


def _aggregate_correlations(correlations: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in correlations:
        grouped.setdefault((str(row["target"]), str(row["metric"])), []).append(row)
    aggregate = []
    for (target, metric), rows in grouped.items():
        pearson = [float(row["pearson"]) for row in rows if math.isfinite(float(row["pearson"]))]
        spearman = [float(row["spearman"]) for row in rows if math.isfinite(float(row["spearman"]))]
        if not pearson and not spearman:
            continue
        mean_spearman = float(np.mean(spearman)) if spearman else float("nan")
        mean_abs_spearman = float(np.mean(np.abs(spearman))) if spearman else float("nan")
        mean_pearson = float(np.mean(pearson)) if pearson else float("nan")
        aggregate.append(
            {
                "target": target,
                "metric": metric,
                "runs": len(rows),
                "mean_pearson": mean_pearson,
                "mean_spearman": mean_spearman,
                "mean_abs_spearman": mean_abs_spearman,
                "candidate_interpretation": "lower metric tends to lower target" if mean_spearman > 0 else "higher metric tends to lower target",
            }
        )
    return sorted(aggregate, key=lambda row: row["mean_abs_spearman"], reverse=True)[: int(top_k)]


def _flatten_report_metrics(report: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in (report.get("metrics") or {}).items():
        if _is_finite_number(value):
            flat[str(key)] = float(value)
    for key, value in (report.get("eval") or {}).items():
        if _is_finite_number(value):
            flat[f"eval_{key}"] = float(value)
            if key in {"bpb", "graph_bpb", "graph_sideinfo_bpb", "graph_conditioned_bpb_no_side_cost"}:
                flat.setdefault(f"eval_{key}", float(value))
    return flat


def _numeric_columns(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = sorted({key for row in rows if isinstance(row, dict) for key in row.keys()})
    cols = {}
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if _is_finite_number(value):
                values.append(float(value))
            else:
                values.append(float("nan"))
        arr = np.asarray(values, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() >= 3 and np.nanstd(arr) > 0:
            cols[str(key)] = arr
    return cols


def _select_baseline(bundles: list[ReportBundle], baseline: str | Path | None) -> ReportBundle:
    if baseline is None:
        return bundles[0]
    target = str(baseline)
    for bundle in bundles:
        if str(bundle.path) == target or bundle.path.name == target or bundle.name == target:
            return bundle
    return load_report_bundle(target)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    x = left[mask]
    y = right[mask]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    return _pearson(_rank(left[mask]), _rank(right[mask]))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TropicalGT-I BPB Ablation Report",
        "",
        f"Baseline: `{report['baseline']}`",
        "",
        "## Runs",
        "",
        "| run | bpb | graph_bpb | eval_bpb | eval_graph_bpb |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        targets = row.get("targets", {})
        lines.append(
            "| {name} | {bpb} | {graph_bpb} | {eval_bpb} | {eval_graph_bpb} |".format(
                name=row.get("name"),
                bpb=_fmt(targets.get("bpb")),
                graph_bpb=_fmt(targets.get("graph_bpb")),
                eval_bpb=_fmt(targets.get("eval_bpb")),
                eval_graph_bpb=_fmt(targets.get("eval_graph_bpb")),
            )
        )
    lines.extend(["", "## Best By Target", "", "| target | best run | value | delta vs baseline | runner-up margin |", "|---|---|---:|---:|---:|"])
    for row in report.get("best_by_target", []):
        lines.append(
            "| `{target}` | {name} | {value} | {delta} | {margin} |".format(
                target=row.get("target"),
                name=row.get("name"),
                value=_fmt(row.get("value")),
                delta=_fmt(row.get("delta_vs_baseline")),
                margin=_fmt(row.get("margin_to_runner_up")),
            )
        )
    lines.extend(["", "## Top Correlation Screens", "", "| target | metric | mean Spearman | interpretation |", "|---|---|---:|---|"])
    for row in report["aggregate_metric_rankings"][:20]:
        lines.append(
            f"| `{row['target']}` | `{row['metric']}` | {_fmt(row.get('mean_spearman'))} | {row.get('candidate_interpretation')} |"
        )
    gate = report.get("advanced_auxiliary_promotion_gate") if isinstance(report.get("advanced_auxiliary_promotion_gate"), dict) else {}
    lines.extend(["", "## Advanced Auxiliary Promotion Gate", ""])
    lines.append(str(gate.get("policy", "no promotion gate available")))
    lines.extend(["", "| variant | status | promotable | delta eval_bpb | delta eval_graph_bpb | guardrails |", "|---|---|---:|---:|---:|---|"])
    for row in gate.get("rows", []):
        if not isinstance(row, dict) or not row.get("advanced_auxiliary_coefficients"):
            continue
        deltas = row.get("required_target_deltas", {}) if isinstance(row.get("required_target_deltas"), dict) else {}
        lines.append(
            "| {name} | `{status}` | {promotable} | {bpb} | {graph_bpb} | {guardrails} |".format(
                name=row.get("ablation_variant") or row.get("name"),
                status=row.get("status"),
                promotable=str(bool(row.get("promotable"))),
                bpb=_fmt(deltas.get("eval_bpb")),
                graph_bpb=_fmt(deltas.get("eval_graph_bpb")),
                guardrails=_guardrail_summary_label(row.get("guardrail_status")),
            )
        )
    if not gate.get("candidate_count"):
        lines.append("| no nonzero advanced auxiliary variants | `unavailable` | False | NA | NA | unavailable |")
    lines.extend(
        [
            "",
            "## Discipline",
            "",
            "Use this report to choose ablation candidates, not to claim causal wins. A metric is promoted only when matched-seed validation improves held-out `bpb` and `graph_bpb` while held-out certificate and tropical-wall guardrails remain non-regressing with real evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_correlation_html(report: dict[str, Any], path: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        _write_dark_html(path, "<p>Plotly unavailable.</p>")
        return
    rows = report.get("aggregate_metric_rankings", [])
    if not rows:
        _write_dark_html(path, "<p>No correlations available.</p>")
        return
    labels = [f"{row['target']}::{row['metric']}" for row in rows]
    values = [float(row.get("mean_spearman", 0.0)) for row in rows]
    fig = go.Figure(
        data=go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#54d6be" if value < 0 else "#ff7aa2" for value in values],
            hovertext=[json.dumps(row, indent=2) for row in rows],
            hoverinfo="text+x",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090b12",
        plot_bgcolor="#090b12",
        font=dict(color="#e8eef8"),
        title="BPB/graph-BPB metric correlation screen",
        xaxis_title="mean Spearman correlation with target",
        yaxis_title="target::metric",
        height=max(420, 22 * len(rows)),
    )
    _write_dark_html(
        path,
        "<h1>BPB/graph-BPB metric correlation screen</h1>"
        + fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True}),
    )


def _guardrail_summary_label(status: Any) -> str:
    if not isinstance(status, dict):
        return "unavailable"
    if status.get("safe_for_promotion"):
        return "pass"
    failed = status.get("failed_groups") if isinstance(status.get("failed_groups"), list) else []
    return "failed: " + ", ".join(str(group) for group in failed) if failed else "failed"


def _write_dark_html(path: Path, body: str) -> None:
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      color-scheme: dark;
      background: #090b12;
      color: #e8eef8;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #090b12;
      color: #e8eef8;
    }}
    p {{
      margin: 24px;
      color: #cbd5e1;
    }}
    h1 {{
      margin: 24px 24px 0;
      font-size: 24px;
      letter-spacing: 0;
      color: #e8eef8;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
""",
        encoding="utf-8",
    )


def _fmt(value: Any) -> str:
    return "NA" if not _is_finite_number(value) else f"{float(value):.6g}"
