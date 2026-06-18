#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "TropicalGT-I" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import parameter_golf_codex_review_loop as review_loop  # noqa: E402
from tropicalgt.readiness_contracts import advanced_bpb_contract_report  # noqa: E402
from tropicalgt.run import load_config  # noqa: E402


HERSCHEL_REQUIRED_AUDIT_SIDECARS = (
    "analogical_simplicial_maps.json",
    "analogical_memory_retrieval.json",
    "analogical_simplex_tree_analogy.json",
    "got_full_trajectory_complex_payload.json",
    "got_full_trajectory_complex_slider_contract.json",
    "got_full_trajectory_complex_jensen_shannon_slider_contract.json",
    "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json",
    "got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json",
    "reasoning_step_complex_maps/manifest.json",
    "inference_scaling_tree.json",
    "tropical_support_payload.json",
    "graphcg_direction_cosines_payload.json",
    "got_nll_density_cloud_payload.json",
    "chart_bundle_transport_sidecar.json",
    "toric_embedding_sidecar.json",
    "tropical_fan_diagnostics.json",
    "trajectory_persistence/persistence_landscapes.json",
    "trajectory_persistence/two_parameter_bifiltration.json",
    "trajectory_level_radius_bifiltration.json",
    "trajectory_topological_algebra.json",
    "trajectory_growth_topology.json",
    "inference_topology.json",
    "inference_algebra.json",
)


def _project_path(value: str | Path | None, default: str | Path) -> Path:
    raw = Path(str(value or default))
    return raw if raw.is_absolute() else ROOT / raw


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def _default_checkpoint(cfg: dict[str, Any]) -> Path:
    checkpoint_dir = _project_path(cfg.get("checkpoint_dir"), ROOT / "TropicalGT-I" / "checkpoints")
    run_name = str(cfg.get("run_name", "tropicalgt_i_train"))
    final_path = checkpoint_dir / f"{run_name}.pt"
    latest_path = checkpoint_dir / f"{run_name}.latest.pt"
    if final_path.exists():
        return final_path
    return latest_path


def _default_report(output_dir: Path, boundary_step: int) -> Path:
    train_report = output_dir / "train_report.json"
    if train_report.exists():
        return train_report
    step_dir = output_dir / "periodic" / f"step_{int(boundary_step):08d}"
    for candidate in (step_dir / "periodic_validation_artifacts.json", step_dir / "validation_report.json"):
        if candidate.exists():
            return candidate
    return train_report


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_herschel_report_artifacts(
    *,
    bundle_path: Path,
    bundle_dir: Path,
    boundary_step: int,
) -> tuple[dict[str, Any], Path, Path, Path]:
    module_path = ROOT / "TropicalGT-I" / "scripts" / "write_herschel_5k_report.py"
    spec = importlib.util.spec_from_file_location("write_herschel_5k_report", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_herschel_report_module:{_relative_project_path(module_path)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    markdown_path = bundle_dir / f"herschel_5k_evidence_report_step_{boundary_step:08d}.md"
    json_path = bundle_dir / f"herschel_5k_evidence_report_step_{boundary_step:08d}.json"
    html_path = bundle_dir / f"herschel_5k_evidence_report_step_{boundary_step:08d}.html"
    summary = module.write_herschel_report(bundle_path, markdown_path, json_path, html_path)
    return summary, markdown_path, json_path, html_path


def _shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _eval_command(args: argparse.Namespace, checkpoint_path: Path, bundle_dir: Path) -> str:
    cmd: list[str | Path] = [
        args.python,
        ROOT / "TropicalGT-I" / "scripts" / "eval_tropicalgt_i.py",
        "--config",
        args.config,
        "--checkpoint",
        checkpoint_path,
        "--split",
        args.split,
        "--details-limit",
        str(args.details_limit),
        "--audit-level",
        args.audit_level,
        "--audit-ph-backend",
        args.audit_ph_backend,
        "--audit-max-simplices",
        str(args.audit_max_simplices),
        "--render-visualizations",
        "--visualization-output-dir",
        bundle_dir / "eval_visualizations",
        "--viz-limit",
        str(args.viz_limit),
    ]
    return "PYTHONPATH=TropicalGT-I/src " + _shell_join(cmd)


def _execution_requested(args: argparse.Namespace) -> bool:
    return any(
        bool(getattr(args, name, False))
        for name in ("run_eval_visualizations", "run_legacy_audit_backfill", "run_interactive_audit_validators")
    )


def _report_step(report: dict[str, Any]) -> int | None:
    for key in ("final_step", "step", "global_step"):
        value = report.get(key)
        if isinstance(value, int):
            return int(value)
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("step") or metrics.get("global_step")
        if isinstance(value, int):
            return int(value)
    return None


def _merge_unique_paths(existing: Any, additions: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    source_values = existing if isinstance(existing, list) else []
    for raw in [*source_values, *additions]:
        text = str(raw or "")
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _augment_artifact_inventory_for_herschel(inventory: Any) -> dict[str, Any]:
    if not isinstance(inventory, dict):
        return {}
    augmented = dict(inventory)
    latest_got_audit = str(augmented.get("latest_got_audit_dir") or "")
    discovered: list[str] = []
    if latest_got_audit:
        audit_dir = _project_path(latest_got_audit, latest_got_audit)
        if audit_dir.is_dir():
            for filename in HERSCHEL_REQUIRED_AUDIT_SIDECARS:
                candidate = audit_dir / filename
                if candidate.is_file():
                    discovered.append(_relative_project_path(candidate))
    augmented["advanced_sidecars_tail"] = _merge_unique_paths(augmented.get("advanced_sidecars_tail", []), discovered)
    augmented["herschel_required_sidecars_present"] = discovered
    return augmented


def _execution_readiness(args: argparse.Namespace, report_path: Path, checkpoint_path: Path, report: dict[str, Any], inventory: dict[str, Any], boundary_step: int) -> dict[str, Any]:
    issues: list[str] = []
    if not report_path.exists():
        issues.append(f"missing_report:{_relative_project_path(report_path)}")
    observed_step = _report_step(report)
    if observed_step is None:
        issues.append("missing_report_step_for_execution")
    elif int(observed_step) < int(boundary_step):
        issues.append(f"report_step_before_boundary:{observed_step}< {int(boundary_step)}")
    if not checkpoint_path.exists():
        issues.append(f"missing_checkpoint:{_relative_project_path(checkpoint_path)}")
    elif not checkpoint_path.is_file():
        issues.append(f"invalid_checkpoint_not_file:{_relative_project_path(checkpoint_path)}")
    else:
        try:
            checkpoint_size = checkpoint_path.stat().st_size
        except OSError as exc:
            issues.append(f"unreadable_checkpoint:{_relative_project_path(checkpoint_path)}:{exc.__class__.__name__}")
        else:
            if checkpoint_size <= 0:
                issues.append(f"empty_checkpoint:{_relative_project_path(checkpoint_path)}")
    needs_audit = bool(getattr(args, "run_legacy_audit_backfill", False) or getattr(args, "run_interactive_audit_validators", False))
    latest_audit = str(inventory.get("latest_got_audit_dir") or "")
    if needs_audit and not latest_audit:
        issues.append("missing_latest_got_audit_for_interactive_commands")
    elif needs_audit:
        audit_path = _project_path(latest_audit, latest_audit)
        if not audit_path.is_dir():
            issues.append(f"missing_latest_got_audit_dir:{latest_audit}")
    return {
        "execution_requested": _execution_requested(args),
        "ready": not issues,
        "issues": issues,
        "boundary_step": int(boundary_step),
        "observed_report_step": observed_step,
        "report": _relative_project_path(report_path),
        "checkpoint": _relative_project_path(checkpoint_path),
        "latest_got_audit_dir": latest_audit,
    }


def _restart_evidence_gate(
    *,
    decision: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_evidence: dict[str, Any] | None,
    execution_readiness: dict[str, Any],
    advanced_bpb_contract: dict[str, Any],
    commands: dict[str, Any],
    command_results: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint_evidence = checkpoint_evidence or {}
    blockers: list[str] = []
    bpb = decision.get("bpb")
    target_missed = bool(decision.get("triggered", False))
    if bpb is None:
        blockers.append("missing_primary_bpb_metric")
    if target_missed and not bool(checkpoint.get("available", False)):
        reason = str(checkpoint.get("unavailable_reason") or "checkpoint_unavailable")
        path = str(checkpoint.get("path") or execution_readiness.get("checkpoint") or "")
        if path:
            blockers.append(f"checkpoint_unavailable:{reason}:{path}")
        else:
            blockers.append(f"checkpoint_unavailable:{reason}")
    if target_missed and not bool(execution_readiness.get("ready", False)):
        blockers.extend(f"execution_readiness:{issue}" for issue in execution_readiness.get("issues", []))
    checkpoint_evidence_warnings = [str(warning) for warning in checkpoint_evidence.get("warnings", []) if str(warning)]
    checkpoint_evidence_safe = bool(checkpoint_evidence.get("safe_for_checkpoint_backed_restart", False))
    if target_missed and not checkpoint_evidence_safe:
        if checkpoint_evidence_warnings:
            blockers.extend(f"checkpoint_evidence:{warning}" for warning in checkpoint_evidence_warnings)
        else:
            blockers.append("checkpoint_evidence:not_safe_for_checkpoint_backed_restart")
    failed_gates = [str(gate) for gate in advanced_bpb_contract.get("failed_gates", []) if str(gate)]
    if target_missed and failed_gates:
        blockers.append("advanced_bpb_contract_failed:" + ",".join(failed_gates))
    if target_missed:
        expected_command_names = ["eval_validation_visualizations"]
        expected_command_names.extend(
            f"interactive_audit_backfill_{index:02d}"
            for index, _command in enumerate(commands.get("interactive_audit_backfills", []), start=1)
        )
        expected_command_names.extend(
            f"interactive_audit_validator_{index:02d}"
            for index, _command in enumerate(commands.get("interactive_audit_validators", []), start=1)
        )
        result_by_name = {str(result.get("name", "")): result for result in command_results}
        missing_results = [name for name in expected_command_names if name not in result_by_name]
        if missing_results:
            blockers.append("post_5k_review_commands_missing_results:" + ",".join(missing_results))
        failed_results = []
        for name, result in result_by_name.items():
            returncode = result.get("returncode")
            timed_out = bool(result.get("timed_out", False))
            if timed_out or returncode != 0:
                failed_results.append(f"{name}:returncode={returncode}:timed_out={timed_out}")
        if failed_results:
            blockers.append("post_5k_review_commands_failed:" + ",".join(failed_results))
    blockers = sorted(dict.fromkeys(blockers))
    if not target_missed:
        action = "not_needed_target_met"
    elif blockers:
        action = "blocked_missing_required_evidence_no_restart"
    else:
        action = "allowed_for_evidence_backed_step0_restart_proposal"
    return {
        "schema_version": "tropicalgt.restart_evidence_gate.v1",
        "target_missed": target_missed,
        "restart_action": action,
        "step0_restart_allowed": action == "allowed_for_evidence_backed_step0_restart_proposal",
        "blocked": bool(blockers),
        "blockers": blockers,
        "checkpoint_available": bool(checkpoint.get("available", False)),
        "checkpoint_evidence_safe": bool(checkpoint_evidence.get("safe_for_checkpoint_backed_restart", False)),
        "checkpoint_evidence_warnings": checkpoint_evidence_warnings,
        "execution_evidence_ready": bool(execution_readiness.get("ready", False)),
        "advanced_bpb_contract_safe": bool(advanced_bpb_contract.get("safe_to_use_for_step0_bpb_restart", False)),
        "policy": (
            "A missed BPB target is not a restart authorization. A step-0 restart proposal is allowed only when "
            "the checkpoint is nonempty and loadable, post-5K command evidence has succeeded, and the advanced BPB contract passes. "
            "Missing evidence must remain blocked with explicit reasons; no proxies or fallbacks."
        ),
    }


def _run_shell_command(command: str, log_dir: Path, name: str, timeout_seconds: float = 0.0) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("_") or "command"
    stdout_path = log_dir / f"{safe_name}.stdout.log"
    stderr_path = log_dir / f"{safe_name}.stderr.log"
    timeout = float(timeout_seconds or 0.0)
    result: dict[str, Any] = {
        "name": name,
        "command": command,
        "stdout_log": _relative_project_path(stdout_path),
        "stderr_log": _relative_project_path(stderr_path),
        "returncode": None,
        "timed_out": False,
    }
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout if timeout > 0 else None,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        result["returncode"] = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result["returncode"] = 124
        result["timed_out"] = True
    return result


def _run_requested_commands(args: argparse.Namespace, commands: dict[str, Any], bundle_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    log_dir = bundle_dir / "command_logs"
    timeout = float(getattr(args, "command_timeout_seconds", 0.0) or 0.0)
    if bool(getattr(args, "run_eval_visualizations", False)):
        results.append(_run_shell_command(commands["eval_validation_visualizations"], log_dir, "eval_validation_visualizations", timeout))
    if bool(getattr(args, "run_legacy_audit_backfill", False)):
        for index, command in enumerate(commands.get("interactive_audit_backfills", []), start=1):
            results.append(_run_shell_command(command, log_dir, f"interactive_audit_backfill_{index:02d}", timeout))
    if bool(getattr(args, "run_interactive_audit_validators", False)):
        for index, command in enumerate(commands.get("interactive_audit_validators", []), start=1):
            results.append(_run_shell_command(command, log_dir, f"interactive_audit_validator_{index:02d}", timeout))
    return results


def _bundle_markdown(bundle: dict[str, Any]) -> str:
    decision = bundle.get("decision", {})
    inventory = bundle.get("artifact_inventory", {})
    lines = [
        "# TropicalGT-I 5K Review Bundle",
        "",
        f"- Boundary step: `{bundle.get('boundary_step')}`",
        f"- Target BPB: `< {bundle.get('target_bpb')}`",
        f"- Observed BPB: `{decision.get('bpb')}`",
        f"- Observed graph BPB: `{decision.get('graph_bpb')}`",
        f"- Config: `{bundle.get('config')}`",
        f"- Report: `{bundle.get('report')}`",
        f"- Checkpoint: `{bundle.get('checkpoint')}`",
        f"- Stop record: `{bundle.get('stop_record')}`",
        "",
        "## Commands",
        "```bash",
        bundle.get("commands", {}).get("eval_validation_visualizations", ""),
    ]
    lines.extend(bundle.get("commands", {}).get("interactive_audit_backfills", []))
    lines.extend(bundle.get("commands", {}).get("interactive_audit_validators", []))
    lines.extend(["```", ""])
    if bundle.get("command_results"):
        lines.extend(["## Command Results", "```json", json.dumps(bundle.get("command_results", []), indent=2), "```", ""])
    lines.extend(
        [
            "## Bundle Artifacts",
            "```json",
            json.dumps(bundle.get("artifacts", {}), indent=2),
            "```",
            "",
            "## Artifact Inventory",
            "```json",
            json.dumps(inventory, indent=2),
            "```",
            "",
            "## Herschel Report Summary",
            "```json",
            json.dumps(bundle.get("herschel_report_summary", {}), indent=2),
            "```",
            "",
            "## Checkpoint Evidence",
            "```json",
            json.dumps(bundle.get("checkpoint_evidence", {}), indent=2),
            "```",
            "",
            "## Execution Readiness",
            "```json",
            json.dumps(bundle.get("execution_readiness", {}), indent=2),
            "```",
            "",
            "## Advanced BPB Contract",
            "```json",
            json.dumps(bundle.get("advanced_bpb_contract", {}), indent=2),
            "```",
            "",
            "## Restart Evidence Gate",
            "```json",
            json.dumps(bundle.get("restart_evidence_gate", {}), indent=2),
            "```",
            "",
            "## Restart Decision Schema",
            "```json",
            json.dumps(bundle.get("restart_decision_schema", {}), indent=2),
            "```",
            "",
            "## Stop Record",
            "```json",
            json.dumps(bundle.get("stop_record_payload", {}), indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_review_bundle(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    output_dir = _project_path(cfg.get("output_dir"), ROOT / "TropicalGT-I" / "outputs" / "train")
    report_path = _project_path(args.report, _default_report(output_dir, int(args.boundary_step or 5000)))
    checkpoint_path = _project_path(args.checkpoint, _default_checkpoint(cfg))
    bundle_dir = _project_path(args.output_dir, output_dir / "post_5k_review_bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stop_record_path = _project_path(args.stop_record, "") if args.stop_record else None

    report = review_loop._load_report(report_path)
    checkpoint = review_loop._load_checkpoint_summary(checkpoint_path)
    boundary_step = int(args.boundary_step or report.get("final_step") or checkpoint.get("step") or 5000)
    bpb = review_loop._metric_value(report, checkpoint, args.metric)
    graph_bpb = review_loop._metric_value(report, checkpoint, args.graph_metric)
    contract = review_loop._active_training_contract(
        cfg,
        report,
        checkpoint,
        boundary_step,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        target_bpb=args.target_bpb,
    )
    inventory = _augment_artifact_inventory_for_herschel(contract.get("artifact_inventory", {}))
    contract["artifact_inventory"] = inventory
    checkpoint_evidence = contract.get("checkpoint_evidence", {})
    advanced_bpb_section, advanced_bpb_gates = advanced_bpb_contract_report(cfg)
    advanced_bpb_failed = [gate for gate in advanced_bpb_gates if gate.get("status") == "fail"]
    advanced_bpb_contract = {
        "section": advanced_bpb_section,
        "gates": advanced_bpb_gates,
        "failed_gates": [str(gate.get("name", "")) for gate in advanced_bpb_failed],
        "safe_to_use_for_step0_bpb_restart": not advanced_bpb_failed,
        "policy": "Config-only advanced BPB restart contract; failed gates must be resolved before a step-0 BPB restart config is launched.",
    }
    prompt = review_loop._review_prompt(
        cfg=cfg,
        report=report,
        checkpoint=checkpoint,
        active_contract=contract,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        previous_boundary_checkpoint=None,
        boundary_step=boundary_step,
        metric=args.metric,
        bpb=bpb,
        graph_metric=args.graph_metric,
        graph_bpb=graph_bpb,
        target_bpb=args.target_bpb,
        restart_policy="beginning",
    )
    contract_path = bundle_dir / f"active_training_contract_step_{boundary_step:08d}.json"
    contract_md_path = bundle_dir / f"active_training_contract_step_{boundary_step:08d}.md"
    advanced_bpb_contract_path = bundle_dir / f"advanced_bpb_contract_step_{boundary_step:08d}.json"
    prompt_path = bundle_dir / f"codex_review_step_{boundary_step:08d}.md"
    bundle_path = bundle_dir / f"review_bundle_step_{boundary_step:08d}.json"
    markdown_path = bundle_dir / f"review_bundle_step_{boundary_step:08d}.md"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    contract_md_path.write_text(review_loop._active_training_contract_markdown(contract), encoding="utf-8")
    advanced_bpb_contract_path.write_text(json.dumps(advanced_bpb_contract, indent=2), encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")

    commands = {
        "eval_validation_visualizations": _eval_command(args, checkpoint_path, bundle_dir),
        "interactive_audit_backfills": inventory.get("interactive_audit_backfill_commands", []),
        "interactive_audit_validators": inventory.get("interactive_audit_validator_commands", []),
    }
    execution_readiness = _execution_readiness(args, report_path, checkpoint_path, report, inventory, boundary_step)
    if execution_readiness["execution_requested"] and not execution_readiness["ready"]:
        raise RuntimeError(
            "Cannot execute post-5K review commands before required evidence exists: "
            + ", ".join(execution_readiness["issues"])
        )
    command_results = _run_requested_commands(args, commands, bundle_dir)
    decision = {
        "bpb": bpb,
        "graph_bpb": graph_bpb,
        "triggered": bpb is None or bpb > args.target_bpb,
        "restart_policy": "beginning",
    }
    restart_evidence_gate = _restart_evidence_gate(
        decision=decision,
        checkpoint=checkpoint,
        checkpoint_evidence=checkpoint_evidence,
        execution_readiness=execution_readiness,
        advanced_bpb_contract=advanced_bpb_contract,
        commands=commands,
        command_results=command_results,
    )
    bundle = {
        "schema_version": "tropicalgt.post_5k_review_bundle.v1",
        "boundary_step": boundary_step,
        "target_bpb": args.target_bpb,
        "metric": args.metric,
        "graph_metric": args.graph_metric,
        "config": _relative_project_path(_project_path(args.config, args.config)),
        "report": _relative_project_path(report_path),
        "checkpoint": _relative_project_path(checkpoint_path),
        "stop_record": _relative_project_path(stop_record_path) if stop_record_path else "",
        "stop_record_payload": _read_json(stop_record_path),
        "decision": decision,
        "advanced_bpb_contract": advanced_bpb_contract,
        "checkpoint_evidence": checkpoint_evidence,
        "restart_evidence_gate": restart_evidence_gate,
        "restart_decision_schema": review_loop._restart_decision_schema(args.target_bpb),
        "review_requirements": [
            "spawn_or_assign_codex_subagent_when_available",
            "review_metrics_advanced_sidecars_topological_geometric_algebraic_visualizations",
            "run_legacy_audit_backfill_before_strict_validation_when_available",
            "restart_from_step_0_with_adjusted_hyperparameters_if_target_not_met",
            "no_proxies_or_fallbacks_for_unavailable_evidence",
        ],
        "artifacts": {
            "contract_json": _relative_project_path(contract_path),
            "contract_markdown": _relative_project_path(contract_md_path),
            "advanced_bpb_contract_json": _relative_project_path(advanced_bpb_contract_path),
            "codex_prompt": _relative_project_path(prompt_path),
            "bundle_json": _relative_project_path(bundle_path),
            "bundle_markdown": _relative_project_path(markdown_path),
        },
        "artifact_inventory": inventory,
        "commands": commands,
        "execution_readiness": execution_readiness,
        "command_results": command_results,
        "policy": "Path-only review bundle by default; explicit command execution writes logs under the generated review bundle. Legacy audit backfill must run before strict validators when requested and may only write explicit unavailable diagnostics or rerender visual contracts from raw payloads. Do not stage generated run artifacts, checkpoints, datasets, caches, W&B data, or secrets.",
    }
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    herschel_summary, herschel_md_path, herschel_json_path, herschel_html_path = _write_herschel_report_artifacts(
        bundle_path=bundle_path,
        bundle_dir=bundle_dir,
        boundary_step=boundary_step,
    )
    bundle["artifacts"]["herschel_report_markdown"] = _relative_project_path(herschel_md_path)
    bundle["artifacts"]["herschel_report_json"] = _relative_project_path(herschel_json_path)
    bundle["artifacts"]["herschel_report_html"] = _relative_project_path(herschel_html_path)
    bundle["herschel_report_summary"] = herschel_summary
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    markdown_path.write_text(_bundle_markdown(bundle), encoding="utf-8")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a path-only TropicalGT-I post-5K Codex review bundle for an already trained run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-record", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--boundary-step", type=int, default=5000)
    parser.add_argument("--target-bpb", type=float, default=1.12)
    parser.add_argument("--metric", default="eval.bpb")
    parser.add_argument("--graph-metric", default="eval.graph_bpb")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--details-limit", type=int, default=4)
    parser.add_argument("--viz-limit", type=int, default=8)
    parser.add_argument("--audit-level", choices=["none", "basic", "topology", "algebra", "full"], default="full")
    parser.add_argument("--audit-ph-backend", choices=["auto", "gudhi", "ripser", "none"], default="gudhi")
    parser.add_argument("--audit-max-simplices", type=int, default=256)
    parser.add_argument("--run-eval-visualizations", action="store_true", help="Execute the generated eval/visualization command and record stdout/stderr logs in the review bundle.")
    parser.add_argument("--run-legacy-audit-backfill", action="store_true", help="Execute generated legacy interactive-audit backfill commands before strict validators and record stdout/stderr logs in the review bundle.")
    parser.add_argument("--run-interactive-audit-validators", action="store_true", help="Execute generated interactive-audit validator commands and record stdout/stderr logs in the review bundle.")
    parser.add_argument("--command-timeout-seconds", type=float, default=0.0, help="Optional timeout for each executed review command; 0 disables the timeout.")
    args = parser.parse_args(argv)
    bundle = prepare_review_bundle(args)
    print(json.dumps({"bundle": bundle.get("artifacts", {}), "decision": bundle.get("decision", {}), "command_results": bundle.get("command_results", [])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
