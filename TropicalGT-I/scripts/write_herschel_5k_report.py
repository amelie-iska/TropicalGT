#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shlex
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "unavailable"
    return str(value)



def _json_output_paths(command: str) -> list[Path]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    paths: list[Path] = []
    for index, part in enumerate(parts):
        if part == "--json-output" and index + 1 < len(parts):
            paths.append(Path(parts[index + 1]))
        elif part.startswith("--json-output="):
            paths.append(Path(part.split("=", 1)[1]))
    return paths


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _validator_gap_evidence(bundle: dict[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    combined: dict[str, int] = {}
    ranked_categories: dict[str, dict[str, Any]] = {}
    top_examples: list[dict[str, Any]] = []
    command_results = [row for row in bundle.get("command_results", []) if isinstance(row, dict)]
    for row in command_results:
        name = str(row.get("name", ""))
        command = str(row.get("command", ""))
        if "interactive_audit_validator" not in name and "validate_interactive_audit_artifacts.py" not in command:
            continue
        paths = _json_output_paths(command)
        if not paths:
            sources.append(
                {
                    "name": name,
                    "available": False,
                    "reason": "validator_command_lacks_json_output_path",
                    "returncode": row.get("returncode"),
                    "timed_out": bool(row.get("timed_out", False)),
                }
            )
            continue
        for path in paths:
            resolved = _resolve_output_path(path)
            source: dict[str, Any] = {
                "name": name,
                "path": _project_path(resolved),
                "available": False,
                "returncode": row.get("returncode"),
                "timed_out": bool(row.get("timed_out", False)),
            }
            if not resolved.exists():
                source["reason"] = "validator_json_output_missing"
                sources.append(source)
                continue
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - parser message is platform-dependent
                source["reason"] = f"validator_json_parse_error:{exc}"
                sources.append(source)
                continue
            inventory = payload.get("evidence_gap_inventory", {}) if isinstance(payload, dict) else {}
            categories = inventory.get("categories", []) if isinstance(inventory.get("categories"), list) else []
            category_counts = inventory.get("category_counts", {}) if isinstance(inventory.get("category_counts"), dict) else {}
            for category, count in category_counts.items():
                try:
                    combined[str(category)] = combined.get(str(category), 0) + int(count)
                except (TypeError, ValueError):
                    continue
            handled_categories: set[str] = set()
            for category_row in categories:
                if not isinstance(category_row, dict):
                    continue
                category = str(category_row.get("category", "other") or "other")
                handled_categories.add(category)
                try:
                    count_value = int(category_row.get("count", category_counts.get(category, 0)) or 0)
                except (TypeError, ValueError):
                    count_value = 0
                required_action = str(category_row.get("required_action", "") or "")
                examples_raw = category_row.get("examples", [])
                examples = [str(item) for item in examples_raw if str(item)] if isinstance(examples_raw, list) else []
                aggregate = ranked_categories.setdefault(
                    category,
                    {"category": category, "count": 0, "required_action": required_action, "examples": [], "source_names": []},
                )
                aggregate["count"] = int(aggregate.get("count", 0) or 0) + count_value
                if required_action and not aggregate.get("required_action"):
                    aggregate["required_action"] = required_action
                source_names = aggregate.setdefault("source_names", [])
                if isinstance(source_names, list) and name and name not in source_names:
                    source_names.append(name)
                for example in examples[:5]:
                    example_row = {"source": name, "path": source.get("path", ""), "example": example}
                    category_examples = aggregate.setdefault("examples", [])
                    if isinstance(category_examples, list) and len(category_examples) < 8:
                        category_examples.append(example_row)
                    if len(top_examples) < 24:
                        top_examples.append({"category": category, "required_action": required_action, **example_row})
            for category, count in category_counts.items():
                category_name = str(category)
                if category_name in handled_categories:
                    continue
                try:
                    count_value = int(count)
                except (TypeError, ValueError):
                    count_value = 0
                aggregate = ranked_categories.setdefault(
                    category_name,
                    {"category": category_name, "count": 0, "required_action": "", "examples": [], "source_names": []},
                )
                aggregate["count"] = int(aggregate.get("count", 0) or 0) + count_value
            source.update(
                {
                    "available": bool(inventory),
                    "validator_ok": bool(payload.get("ok")) if isinstance(payload, dict) else False,
                    "error_count": len(payload.get("errors", [])) if isinstance(payload, dict) and isinstance(payload.get("errors"), list) else None,
                    "gap_count": inventory.get("gap_count"),
                    "category_counts": category_counts,
                    "categories": categories[:16],
                    "policy": inventory.get("policy", ""),
                }
            )
            if not inventory:
                source["reason"] = "validator_json_lacks_evidence_gap_inventory"
            sources.append(source)
    ranked_category_rows = sorted(
        ranked_categories.values(), key=lambda row: (-int(row.get("count", 0) or 0), str(row.get("category", "")))
    )
    return {
        "schema_version": "tropicalgt.herschel_validator_gap_evidence.v1",
        "available": any(source.get("available") for source in sources),
        "source_count": len(sources),
        "combined_category_counts": {category: count for category, count in sorted(combined.items())},
        "ranked_categories": ranked_category_rows,
        "top_examples": top_examples,
        "sources": sources,
        "policy": "Herschel reads validator gap inventories only from recorded validator JSON outputs; missing JSON is unavailable and does not justify a restart or artifact pass.",
    }


def _sidecar_groups(paths: list[str]) -> dict[str, int]:
    groups = {
        "cas_algebra": 0,
        "topology_persistence": 0,
        "analogical_memory": 0,
        "tropical_toric": 0,
        "graphcg": 0,
        "nll_density": 0,
        "chart_bundle": 0,
        "vector_bundle": 0,
        "sheaf_derived": 0,
        "other": 0,
    }
    for path in paths:
        lower = path.lower()
        matched = False
        if any(term in lower for term in ("cas", "betti", "fitting", "minor", "free_resolution", "buchsbaum", "be_", "certificate_indexed")):
            groups["cas_algebra"] += 1
            matched = True
        if any(term in lower for term in ("persistence", "bifiltration", "simplex", "barcode", "landscape")):
            groups["topology_persistence"] += 1
            matched = True
        if "analogical" in lower or "memory" in lower:
            groups["analogical_memory"] += 1
            matched = True
        if any(term in lower for term in ("tropical", "toric", "fan")):
            groups["tropical_toric"] += 1
            matched = True
        if "graphcg" in lower:
            groups["graphcg"] += 1
            matched = True
        if "nll" in lower or "density" in lower:
            groups["nll_density"] += 1
            matched = True
        if "chart_bundle" in lower or "bundle" in lower:
            groups["chart_bundle"] += 1
            matched = True
        if any(term in lower for term in ("vector_bundle", "vector-bundle", "chart_bundle_transport_sidecar", "bundle_toric", "paper_sidecar")):
            groups["vector_bundle"] += 1
            matched = True
        if any(term in lower for term in ("sheaf", "derived", "chain_map", "derived_category")):
            groups["sheaf_derived"] += 1
            matched = True
        if not matched:
            groups["other"] += 1
    return groups


def summarize_bundle(bundle: dict[str, Any], *, bundle_path: Path | None = None) -> dict[str, Any]:
    decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
    gate = bundle.get("restart_evidence_gate") if isinstance(bundle.get("restart_evidence_gate"), dict) else {}
    checkpoint = bundle.get("checkpoint_evidence") if isinstance(bundle.get("checkpoint_evidence"), dict) else {}
    readiness = bundle.get("execution_readiness") if isinstance(bundle.get("execution_readiness"), dict) else {}
    inventory = bundle.get("artifact_inventory") if isinstance(bundle.get("artifact_inventory"), dict) else {}
    advanced = bundle.get("advanced_bpb_contract") if isinstance(bundle.get("advanced_bpb_contract"), dict) else {}
    sidecars = [str(path) for path in inventory.get("advanced_sidecars_tail", []) if str(path)]
    validator_gap_evidence = _validator_gap_evidence(bundle)
    command_results = [row for row in bundle.get("command_results", []) if isinstance(row, dict)]
    failed_commands = [row for row in command_results if row.get("returncode") not in (0, None) or row.get("timed_out")]
    blockers = [str(item) for item in gate.get("blockers", []) if str(item)]
    checkpoint_warnings = [str(item) for item in checkpoint.get("warnings", []) if str(item)]
    execution_issues = [str(item) for item in readiness.get("issues", []) if str(item)]
    failed_gates = [str(item) for item in advanced.get("failed_gates", []) if str(item)]
    return {
        "schema_version": "tropicalgt.herschel_5k_report_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_path": _project_path(bundle_path) if bundle_path else "",
        "run_identity": {
            "config": bundle.get("config", ""),
            "report": bundle.get("report", ""),
            "checkpoint": bundle.get("checkpoint", ""),
            "stop_record": bundle.get("stop_record", ""),
            "boundary_step": bundle.get("boundary_step"),
            "target_bpb": bundle.get("target_bpb"),
        },
        "primary_metrics": {
            "bpb": decision.get("bpb"),
            "graph_bpb": decision.get("graph_bpb"),
            "target_missed": bool(decision.get("triggered", False)),
            "restart_policy": decision.get("restart_policy", ""),
        },
        "checkpoint_evidence": {
            "available": bool(checkpoint.get("checkpoint_available", False)),
            "restart_safe": bool(checkpoint.get("safe_for_checkpoint_backed_restart", False)),
            "unavailable_reason": checkpoint.get("checkpoint_unavailable_reason", ""),
            "warnings": checkpoint_warnings,
        },
        "execution_evidence": {
            "requested": bool(readiness.get("execution_requested", False)),
            "ready": bool(readiness.get("ready", False)),
            "issues": execution_issues,
            "command_results": len(command_results),
            "failed_commands": failed_commands,
        },
        "advanced_bpb_contract": {
            "safe_for_restart": bool(advanced.get("safe_to_use_for_step0_bpb_restart", False)),
            "failed_gates": failed_gates,
        },
        "artifact_evidence": {
            "latest_got_audit_dir": inventory.get("latest_got_audit_dir", ""),
            "latest_periodic_dir": inventory.get("latest_periodic_dir", ""),
            "advanced_sidecar_count": len(sidecars),
            "sidecar_groups": _sidecar_groups(sidecars),
            "advanced_sidecars_tail": sidecars[:120],
            "validator_gap_evidence": validator_gap_evidence,
        },
        "restart_decision": {
            "action": gate.get("restart_action", "unavailable"),
            "step0_restart_allowed": bool(gate.get("step0_restart_allowed", False)),
            "blocked": bool(gate.get("blocked", False)),
            "blockers": blockers,
        },
        "policy": "CPU-only Herschel report generated from an existing post-5K review bundle; no training, eval, browser, validator, checkpoint load beyond the bundle contents, or GPU command is executed.",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    run = summary["run_identity"]
    metrics = summary["primary_metrics"]
    checkpoint = summary["checkpoint_evidence"]
    execution = summary["execution_evidence"]
    advanced = summary["advanced_bpb_contract"]
    artifacts = summary["artifact_evidence"]
    restart = summary["restart_decision"]
    lines = [
        "# Herschel 5K Evidence Report",
        "",
        f"- Generated: `{summary.get('generated_at')}`",
        f"- Bundle: `{summary.get('bundle_path')}`",
        f"- Boundary step: `{run.get('boundary_step')}`",
        f"- Config: `{run.get('config')}`",
        f"- Report: `{run.get('report')}`",
        f"- Checkpoint: `{run.get('checkpoint')}`",
        f"- Stop record: `{run.get('stop_record')}`",
        "",
        "## Primary Metrics",
        "",
        f"- Target BPB: `< {_fmt(run.get('target_bpb'))}`",
        f"- Observed BPB: `{_fmt(metrics.get('bpb'))}`",
        f"- Observed graph-BPB: `{_fmt(metrics.get('graph_bpb'))}`",
        f"- Target missed: `{metrics.get('target_missed')}`",
        f"- Restart policy: `{metrics.get('restart_policy')}`",
        "",
        "## Evidence Status",
        "",
        f"- Checkpoint available: `{checkpoint.get('available')}`",
        f"- Checkpoint restart-safe: `{checkpoint.get('restart_safe')}`",
        f"- Checkpoint unavailable reason: `{checkpoint.get('unavailable_reason') or 'n/a'}`",
        f"- Execution evidence requested: `{execution.get('requested')}`",
        f"- Execution evidence ready: `{execution.get('ready')}`",
        f"- Command results recorded: `{execution.get('command_results')}`",
        f"- Advanced BPB contract safe: `{advanced.get('safe_for_restart')}`",
        "",
        "## Artifact Evidence",
        "",
        f"- Latest periodic dir: `{artifacts.get('latest_periodic_dir')}`",
        f"- Latest GoT audit dir: `{artifacts.get('latest_got_audit_dir')}`",
        f"- Advanced sidecar count: `{artifacts.get('advanced_sidecar_count')}`",
        "",
        "```json",
        json.dumps(artifacts.get("sidecar_groups", {}), indent=2),
        "```",
        "",
        "## Validator Evidence Gaps",
        "",
    ]
    validator_gaps = artifacts.get("validator_gap_evidence", {}) if isinstance(artifacts.get("validator_gap_evidence"), dict) else {}
    lines.extend(
        [
            f"- Available: `{validator_gaps.get('available', False)}`",
            f"- Sources: `{validator_gaps.get('source_count', 0)}`",
            "",
            "```json",
            json.dumps(validator_gaps.get("combined_category_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    ranked_validator_categories = validator_gaps.get("ranked_categories", []) if isinstance(validator_gaps.get("ranked_categories"), list) else []
    if ranked_validator_categories:
        lines.extend(["### Required Actions", ""])
        for row in ranked_validator_categories[:12]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- `{row.get('category')}` count=`{row.get('count')}` action={row.get('required_action') or 'unavailable'}")
            for example_row in row.get("examples", [])[:3] if isinstance(row.get("examples"), list) else []:
                if not isinstance(example_row, dict):
                    continue
                lines.append(f"  - example: `{example_row.get('example')}`")
    for source in validator_gaps.get("sources", []) if isinstance(validator_gaps.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(f"- `{source.get('name', 'validator')}` path=`{source.get('path', '')}` available=`{source.get('available')}` gaps=`{source.get('gap_count')}`")
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    lines.extend(
        [
            "",
            "## Restart Decision",
        "",
        f"- Action: `{restart.get('action')}`",
        f"- Step-0 restart allowed: `{restart.get('step0_restart_allowed')}`",
        f"- Blocked: `{restart.get('blocked')}`",
        "",
        "```mermaid",
        "flowchart TD",
        "  A[5K review bundle] --> B{BPB target missed?}",
        f"  B -->|{metrics.get('target_missed')}| C{{Checkpoint restart-safe?}}",
        f"  C -->|{checkpoint.get('restart_safe')}| D{{Execution evidence ready?}}",
        f"  D -->|{execution.get('ready')}| E{{Advanced BPB contract safe?}}",
        f"  E -->|{advanced.get('safe_for_restart')}| F[{restart.get('action')}]",
        "```",
        "",
        "## Blockers And Warnings",
        "",
        ]
    )
    for label, values in (
        ("checkpoint warnings", checkpoint.get("warnings", [])),
        ("execution issues", execution.get("issues", [])),
        ("advanced BPB failed gates", advanced.get("failed_gates", [])),
        ("restart blockers", restart.get("blockers", [])),
    ):
        lines.append(f"### {label.title()}")
        if values:
            lines.extend(f"- `{value}`" for value in values)
        else:
            lines.append("- `none`")
        lines.append("")
    lines.extend(
        [
            "## Advanced Sidecars Tail",
            "",
            *[f"- `{path}`" for path in artifacts.get("advanced_sidecars_tail", [])[:80]],
            "",
            "## Policy",
            "",
            summary.get("policy", ""),
            "",
        ]
    )
    return chr(10).join(lines)




def _bar_chart_svg(values: dict[str, Any], *, title: str, chart_id: str) -> str:
    numeric: list[tuple[str, int]] = []
    for key, value in values.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            numeric.append((str(key), count))
    if not numeric:
        return f"<section class='panel' data-chart='{html.escape(chart_id)}'><h2>{html.escape(title)}</h2><p class='muted'>No recorded counts.</p></section>"
    numeric.sort(key=lambda item: (-item[1], item[0]))
    max_count = max(count for _, count in numeric) or 1
    width = 880
    left = 250
    bar_max = width - left - 96
    row_h = 34
    height = 44 + row_h * len(numeric)
    rows = [f"<svg role='img' aria-label='{html.escape(title)}' viewBox='0 0 {width} {height}'>"]
    rows.append(f"<title>{html.escape(title)}</title>")
    rows.append(f"<text x='0' y='20' class='svg-title'>{html.escape(title)}</text>")
    for index, (label, count) in enumerate(numeric):
        y = 40 + index * row_h
        bar_w = max(2, int(bar_max * (count / max_count)))
        rows.append(f"<text x='0' y='{y + 18}' class='axis-label'>{html.escape(label)}</text>")
        rows.append(f"<rect x='{left}' y='{y}' width='{bar_w}' height='22' rx='3'><title>{html.escape(label)}: {count}</title></rect>")
        rows.append(f"<text x='{left + bar_w + 10}' y='{y + 17}' class='value-label'>{count}</text>")
    rows.append("</svg>")
    return f"<section class='panel' data-chart='{html.escape(chart_id)}'><h2>{html.escape(title)}</h2>{''.join(rows)}</section>"


def _html_list(values: list[Any]) -> str:
    if not values:
        return "<li class='muted'>none</li>"
    return "".join(f"<li>{html.escape(str(value))}</li>" for value in values)


def render_html(summary: dict[str, Any]) -> str:
    run = summary["run_identity"]
    metrics = summary["primary_metrics"]
    checkpoint = summary["checkpoint_evidence"]
    execution = summary["execution_evidence"]
    advanced = summary["advanced_bpb_contract"]
    artifacts = summary["artifact_evidence"]
    restart = summary["restart_decision"]
    validator_gaps = artifacts.get("validator_gap_evidence", {}) if isinstance(artifacts.get("validator_gap_evidence"), dict) else {}
    sidecar_groups = artifacts.get("sidecar_groups", {}) if isinstance(artifacts.get("sidecar_groups"), dict) else {}
    validator_counts = validator_gaps.get("combined_category_counts", {}) if isinstance(validator_gaps.get("combined_category_counts"), dict) else {}
    validator_sources = validator_gaps.get("sources", []) if isinstance(validator_gaps.get("sources"), list) else []
    sidecars = [str(path) for path in artifacts.get("advanced_sidecars_tail", [])]
    source_rows = []
    for source in validator_sources:
        if not isinstance(source, dict):
            continue
        source_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('name', 'validator')))}</td>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('gap_count')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not source_rows:
        source_rows.append("<tr><td colspan='5' class='muted'>No validator JSON sources recorded.</td></tr>")
    validator_ranked = validator_gaps.get("ranked_categories", []) if isinstance(validator_gaps.get("ranked_categories"), list) else []
    action_rows = []
    for row in validator_ranked[:20]:
        if not isinstance(row, dict):
            continue
        examples = row.get("examples", [])
        example_text = "; ".join(
            str(example.get("example", "")) for example in examples[:3] if isinstance(example, dict)
        ) if isinstance(examples, list) else ""
        action_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('category', 'other')))}</td>"
            f"<td>{html.escape(str(row.get('count', '')))}</td>"
            f"<td>{html.escape(str(row.get('required_action', '')))}</td>"
            f"<td><code>{html.escape(example_text)}</code></td>"
            "</tr>"
        )
    if not action_rows:
        action_rows.append("<tr><td colspan='4' class='muted'>No concrete validator gap examples recorded.</td></tr>")
    sidecar_items = "".join(f"<li data-path='{html.escape(path.lower())}'>{html.escape(path)}</li>" for path in sidecars[:160]) or "<li class='muted'>No sidecar paths recorded.</li>"
    restart_safe = checkpoint.get("restart_safe") and execution.get("ready") and advanced.get("safe_for_restart") and restart.get("step0_restart_allowed")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Herschel 5K Visual Evidence Report</title>
<style>
:root {{ color-scheme: light; --ink:#151923; --muted:#5b6472; --line:#d8dee8; --panel:#ffffff; --bg:#f5f7fb; --accent:#2457d6; --warn:#b42318; --ok:#087443; }}
body {{ margin:0; font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
header, main {{ max-width:1180px; margin:0 auto; padding:24px; }}
header {{ padding-bottom:10px; }}
h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
h2 {{ margin:0 0 12px; font-size:18px; letter-spacing:0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
.card, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 1px 2px rgba(10,20,40,.04); }}
.card b {{ display:block; font-size:22px; margin-top:5px; }}
.muted {{ color:var(--muted); }}
.badge {{ display:inline-block; padding:3px 8px; border-radius:999px; border:1px solid var(--line); background:#f8fafc; margin-right:6px; }}
.badge.ok {{ color:var(--ok); border-color:#9bd3b7; background:#effaf4; }} .badge.warn {{ color:var(--warn); border-color:#f2aaa4; background:#fff3f1; }}
section {{ margin:14px 0; }}
svg {{ width:100%; height:auto; }}
rect {{ fill:var(--accent); }}
.svg-title {{ font-weight:700; font-size:16px; fill:var(--ink); }} .axis-label,.value-label {{ font-size:12px; fill:var(--ink); }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }} th {{ color:var(--muted); font-weight:600; }}
.flow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px; align-items:stretch; }} .flow div {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcff; }}
input[type=search] {{ width:100%; padding:10px; border:1px solid var(--line); border-radius:6px; margin-bottom:8px; }}
code {{ white-space:break-spaces; }}
</style>
</head>
<body>
<header>
<h1>Herschel 5K Visual Evidence Report</h1>
<p class="muted">Generated {html.escape(str(summary.get('generated_at')))} from <code>{html.escape(str(summary.get('bundle_path')))}</code>. Evidence-only rendering: no training, validation, checkpoint loading, browser control, or GPU command is executed by this report.</p>
<span class="badge {'ok' if restart_safe else 'warn'}">restart_safe={html.escape(str(bool(restart_safe)))}</span>
<span class="badge {'ok' if validator_gaps.get('available') else 'warn'}">validator_gap_evidence={html.escape(str(bool(validator_gaps.get('available'))))}</span>
</header>
<main>
<section class="grid">
<div class="card">Observed BPB<b>{html.escape(_fmt(metrics.get('bpb')))}</b><span class="muted">target &lt; {html.escape(_fmt(run.get('target_bpb')))}</span></div>
<div class="card">Observed graph-BPB<b>{html.escape(_fmt(metrics.get('graph_bpb')))}</b><span class="muted">graph-conditioned gate</span></div>
<div class="card">Checkpoint restart-safe<b>{html.escape(str(checkpoint.get('restart_safe')))}</b><span class="muted">{html.escape(str(checkpoint.get('unavailable_reason') or 'n/a'))}</span></div>
<div class="card">Restart action<b>{html.escape(str(restart.get('action')))}</b><span class="muted">step-0 allowed={html.escape(str(restart.get('step0_restart_allowed')))}</span></div>
</section>
<section class="panel"><h2>Restart Decision Flow</h2><div class="flow"><div>5K bundle<br><b>step {html.escape(str(run.get('boundary_step')))}</b></div><div>Target missed<br><b>{html.escape(str(metrics.get('target_missed')))}</b></div><div>Checkpoint safe<br><b>{html.escape(str(checkpoint.get('restart_safe')))}</b></div><div>Execution ready<br><b>{html.escape(str(execution.get('ready')))}</b></div><div>Advanced gate<br><b>{html.escape(str(advanced.get('safe_for_restart')))}</b></div><div>Action<br><b>{html.escape(str(restart.get('action')))}</b></div></div></section>
{_bar_chart_svg(sidecar_groups, title='Advanced Sidecar Groups', chart_id='sidecar-groups')}
{_bar_chart_svg(validator_counts, title='Strict Validator Evidence Gaps', chart_id='validator-gap-counts')}
<section class="panel"><h2>Validator Sources</h2><table><thead><tr><th>Name</th><th>JSON path</th><th>Available</th><th>Gaps</th><th>Reason</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></section>
<section class="panel"><h2>Validator Gap Actions</h2><table><thead><tr><th>Category</th><th>Count</th><th>Required action</th><th>Examples</th></tr></thead><tbody>{''.join(action_rows)}</tbody></table></section>
<section class="panel"><h2>Blockers And Warnings</h2><div class="grid"><div><h3>Checkpoint</h3><ul>{_html_list(checkpoint.get('warnings', []))}</ul></div><div><h3>Execution</h3><ul>{_html_list(execution.get('issues', []))}</ul></div><div><h3>Advanced BPB</h3><ul>{_html_list(advanced.get('failed_gates', []))}</ul></div><div><h3>Restart</h3><ul>{_html_list(restart.get('blockers', []))}</ul></div></div></section>
<section class="panel"><h2>Advanced Sidecars Tail</h2><input id="sidecar-filter" type="search" placeholder="Filter sidecar paths"><ul id="sidecar-list">{sidecar_items}</ul></section>
</main>
<script>
const input = document.getElementById('sidecar-filter');
const items = Array.from(document.querySelectorAll('#sidecar-list li[data-path]'));
if (input) {{ input.addEventListener('input', () => {{ const q = input.value.toLowerCase(); items.forEach(li => li.style.display = li.dataset.path.includes(q) ? '' : 'none'); }}); }}
</script>
</body>
</html>
"""


def write_herschel_report(
    bundle_path: Path,
    output_path: Path,
    json_output: Path | None = None,
    html_output: Path | None = None,
) -> dict[str, Any]:
    bundle = _read_json(bundle_path)
    summary = summarize_bundle(bundle, bundle_path=bundle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(summary), encoding="utf-8")
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if html_output is not None:
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(render_html(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a CPU-only Herschel 5K evidence report from an existing review bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args(argv)
    summary = write_herschel_report(args.bundle, args.output, args.json_output, args.html_output)
    print(
        json.dumps(
            {
                "output": _project_path(args.output),
                "json_output": _project_path(args.json_output) if args.json_output else "",
                "html_output": _project_path(args.html_output) if args.html_output else "",
                "restart_action": summary["restart_decision"]["action"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
