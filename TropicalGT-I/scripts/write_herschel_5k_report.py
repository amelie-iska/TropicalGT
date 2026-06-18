#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    return {
        "schema_version": "tropicalgt.herschel_validator_gap_evidence.v1",
        "available": any(source.get("available") for source in sources),
        "source_count": len(sources),
        "combined_category_counts": {category: count for category, count in sorted(combined.items())},
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


def write_herschel_report(bundle_path: Path, output_path: Path, json_output: Path | None = None) -> dict[str, Any]:
    bundle = _read_json(bundle_path)
    summary = summarize_bundle(bundle, bundle_path=bundle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(summary), encoding="utf-8")
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a CPU-only Herschel 5K evidence report from an existing review bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    summary = write_herschel_report(args.bundle, args.output, args.json_output)
    print(json.dumps({"output": _project_path(args.output), "json_output": _project_path(args.json_output) if args.json_output else "", "restart_action": summary["restart_decision"]["action"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
