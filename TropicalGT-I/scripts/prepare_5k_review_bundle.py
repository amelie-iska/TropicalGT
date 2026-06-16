#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "TropicalGT-I" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import parameter_golf_codex_review_loop as review_loop  # noqa: E402
from tropicalgt.run import load_config  # noqa: E402


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


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    lines.extend(bundle.get("commands", {}).get("interactive_audit_validators", []))
    lines.extend(
        [
            "```",
            "",
            "## Artifact Inventory",
            "```json",
            json.dumps(inventory, indent=2),
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
    report_path = _project_path(args.report, output_dir / "train_report.json")
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
        target_bpb=args.target_bpb,
    )
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
    prompt_path = bundle_dir / f"codex_review_step_{boundary_step:08d}.md"
    bundle_path = bundle_dir / f"review_bundle_step_{boundary_step:08d}.json"
    markdown_path = bundle_dir / f"review_bundle_step_{boundary_step:08d}.md"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    contract_md_path.write_text(review_loop._active_training_contract_markdown(contract), encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")

    inventory = contract.get("artifact_inventory", {})
    commands = {
        "eval_validation_visualizations": _eval_command(args, checkpoint_path, bundle_dir),
        "interactive_audit_validators": inventory.get("interactive_audit_validator_commands", []),
    }
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
        "decision": {
            "bpb": bpb,
            "graph_bpb": graph_bpb,
            "triggered": bpb is None or bpb > args.target_bpb,
            "restart_policy": "beginning",
        },
        "review_requirements": [
            "spawn_or_assign_codex_subagent_when_available",
            "review_metrics_advanced_sidecars_topological_geometric_algebraic_visualizations",
            "restart_from_step_0_with_adjusted_hyperparameters_if_target_not_met",
            "no_proxies_or_fallbacks_for_unavailable_evidence",
        ],
        "artifacts": {
            "contract_json": _relative_project_path(contract_path),
            "contract_markdown": _relative_project_path(contract_md_path),
            "codex_prompt": _relative_project_path(prompt_path),
            "bundle_json": _relative_project_path(bundle_path),
            "bundle_markdown": _relative_project_path(markdown_path),
        },
        "artifact_inventory": inventory,
        "commands": commands,
        "policy": "Path-only review bundle; do not stage generated run artifacts, checkpoints, datasets, caches, W&B data, or secrets.",
    }
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
    args = parser.parse_args(argv)
    bundle = prepare_review_bundle(args)
    print(json.dumps({"bundle": bundle.get("artifacts", {}), "decision": bundle.get("decision", {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
