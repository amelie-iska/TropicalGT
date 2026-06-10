from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TARGETS = ("bpb", "graph_bpb", "eval_bpb", "eval_graph_bpb")


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
    return {
        "version": 1,
        "targets": list(target_names),
        "baseline": str(baseline_bundle.path),
        "runs": runs,
        "deltas_vs_baseline": deltas,
        "history_correlations": correlations,
        "aggregate_metric_rankings": aggregate,
        "interpretation": {
            "primary_metric": "bpb",
            "graph_primary_metric": "graph_bpb",
            "warning": "Per-step correlations use train history; eval targets are screened across matched final reports when at least three runs are available. Promote a component only after matched-seed ablations improve held-out bpb or graph_bpb.",
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
        "final_step": bundle.report.get("final_step"),
        "device": bundle.report.get("device"),
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
    lines.extend(["", "## Top Correlation Screens", "", "| target | metric | mean Spearman | interpretation |", "|---|---|---:|---|"])
    for row in report["aggregate_metric_rankings"][:20]:
        lines.append(
            f"| `{row['target']}` | `{row['metric']}` | {_fmt(row.get('mean_spearman'))} | {row.get('candidate_interpretation')} |"
        )
    lines.extend(
        [
            "",
            "## Discipline",
            "",
            "Use this report to choose ablation candidates, not to claim causal wins. A metric is promoted only when matched-seed validation improves held-out `bpb` or `graph_bpb`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_correlation_html(report: dict[str, Any], path: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        path.write_text("<html><body><p>Plotly unavailable.</p></body></html>", encoding="utf-8")
        return
    rows = report.get("aggregate_metric_rankings", [])
    if not rows:
        path.write_text("<html><body><p>No correlations available.</p></body></html>", encoding="utf-8")
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
        title="BPB/graph-BPB metric correlation screen",
        xaxis_title="mean Spearman correlation with target",
        yaxis_title="target::metric",
        height=max(420, 22 * len(rows)),
    )
    fig.write_html(path)


def _fmt(value: Any) -> str:
    return "NA" if not _is_finite_number(value) else f"{float(value):.6g}"
