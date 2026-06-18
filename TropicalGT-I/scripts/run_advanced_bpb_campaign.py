#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
from typing import Any

TGI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TGI_ROOT.parent
SRC = TGI_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tropicalgt.launch_safety import enforce_gpu_launch_safety, gpu_launch_safety_contract  # noqa: E402
from tropicalgt.readiness_contracts import advanced_bpb_contract_report  # noqa: E402


@dataclass
class RunDecision:
    run_name: str
    config_path: Path
    output_dir: Path
    checkpoint_path: Path
    report_path: Path
    log_path: Path
    step_boundary: int
    returncode: int | None = None
    bpb: float | None = None
    graph_bpb: float | None = None
    reached_target: bool = False
    failed: bool = False
    failure_reason: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive 5K-step BPB campaign supervisor for TropicalGT-I")
    parser.add_argument("--base-config", type=Path, default=_default_base_config())
    parser.add_argument("--target-bpb", type=float, default=1.12)
    parser.add_argument("--boundary-steps", type=int, default=5000)
    parser.add_argument("--max-primary-runs", type=int, default=25)
    parser.add_argument("--followup-runs", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=64)
    parser.add_argument("--dataset-root", type=Path, default=TGI_ROOT / "data")
    parser.add_argument("--oai-root", type=Path, default=PROJECT_ROOT / "external" / "oai-parameter-golf" / "data" / "datasets" / "fineweb10B_sp1024")
    parser.add_argument("--oai-tokenizer", type=Path, default=PROJECT_ROOT / "external" / "oai-parameter-golf" / "data" / "tokenizers" / "fineweb_1024_bpe.model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=TGI_ROOT / "scripts" / "train_tropicalgt_i.py")
    parser.add_argument("--prepare-review-script", type=Path, default=TGI_ROOT / "scripts" / "prepare_5k_review_bundle.py")
    parser.add_argument("--output-root", type=Path, default=TGI_ROOT / "outputs")
    parser.add_argument("--config-root", type=Path, default=TGI_ROOT / "outputs" / "launch_configs")
    parser.add_argument("--notes-root", type=Path, default=TGI_ROOT / "training_notes")
    parser.add_argument("--state-path", type=Path, default=TGI_ROOT / "training_notes" / "advanced_bpb_campaign_state.json")
    parser.add_argument("--codex-command", default=os.environ.get("CODEX_REVIEW_COMMAND", "codex exec"))
    parser.add_argument("--invoke-codex", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Generate the next campaign config and exit without training.")
    parser.add_argument("--dry-run", action="store_true", help="Print train commands without executing them.")
    parser.add_argument("--cuda-alloc-conf", default=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
    parser.add_argument("--allow-gpu-launch", action="store_true", help="Explicitly authorize this campaign supervisor to launch GPU training work.")
    parser.add_argument("--gpu-memory-budget-mb", type=int, default=None, help="Declared safe GPU memory budget for this campaign launch; requires --gpu-clearance-note or TROPICALGT_GPU_CLEARANCE_NOTE.")
    parser.add_argument("--gpu-clearance-note", default=os.environ.get("TROPICALGT_GPU_CLEARANCE_NOTE", ""), help="Human-readable note documenting who/what cleared GPU launch or budget use.")
    args = parser.parse_args()

    args.config_root.mkdir(parents=True, exist_ok=True)
    args.notes_root.mkdir(parents=True, exist_ok=True)
    launch_safety = gpu_launch_safety_contract(
        allow_gpu_launch=bool(args.allow_gpu_launch),
        gpu_memory_budget_mb=args.gpu_memory_budget_mb,
        gpu_clearance_note=args.gpu_clearance_note,
        action="advanced_bpb_campaign_train_launch",
    )
    launch_safety_path = args.notes_root / "advanced_bpb_campaign_gpu_launch_safety.json"
    _write_json(launch_safety_path, launch_safety)
    if not args.prepare_only and not args.dry_run:
        enforce_gpu_launch_safety(
            allow_gpu_launch=bool(args.allow_gpu_launch),
            gpu_memory_budget_mb=args.gpu_memory_budget_mb,
            gpu_clearance_note=args.gpu_clearance_note,
            action="advanced_bpb_campaign_train_launch",
        )
    state = _load_json(args.state_path)
    state.setdefault("schema_version", "tropicalgt.advanced_bpb_campaign.v1")
    state.setdefault("started_at", _now_iso())
    state.setdefault("target_bpb", args.target_bpb)
    state["gpu_launch_safety_contract"] = launch_safety
    state["gpu_launch_safety_contract_path"] = str(launch_safety_path)
    state.setdefault("runs", [])
    state.setdefault("synopses", [])
    _write_json(args.state_path, state)

    total_runs = int(args.max_primary_runs) + int(args.followup_runs)
    for campaign_offset in range(total_runs):
        run_number = len(state.get("runs", [])) + 1
        phase = "primary" if run_number <= int(args.max_primary_runs) else "followup"
        run_index = int(args.start_index) + run_number - 1
        cfg, config_path = _write_next_config(args, state, run_index=run_index, run_number=run_number, phase=phase)
        if args.prepare_only:
            print(json.dumps({"prepared_config": str(config_path), "run_name": cfg["run_name"]}, indent=2))
            return
        if args.dry_run:
            _run_one_boundary(args, cfg, config_path)
            return
        decision = _run_one_boundary(args, cfg, config_path)
        _write_boundary_report(args, state, cfg, decision, phase=phase, run_number=run_number)
        state.setdefault("runs", []).append(_decision_json(decision, phase=phase, run_number=run_number))
        _write_json(args.state_path, state)
        if decision.reached_target:
            _write_campaign_synopsis(args, state, reason="target_reached")
            return
        if run_number == int(args.max_primary_runs):
            _write_campaign_synopsis(args, state, reason="primary_25_runs_without_target")
        if decision.failed and _hard_failure(decision):
            _write_campaign_synopsis(args, state, reason="hard_training_failure")
            return
    _write_campaign_synopsis(args, state, reason="campaign_exhausted_without_target")


def _default_base_config() -> Path:
    preferred = TGI_ROOT / "outputs" / "launch_configs" / "tropicalgt_i_pg_bpb_step0_full24b_b63_20260616T174503Z_fresh_bpb112_alwayson_5k_gate.json"
    if preferred.exists():
        return preferred
    return TGI_ROOT / "configs" / "train_full_dataset_pg_bpb_step0_full24b_b55_v11_bpb_5k_gate.json"


def _write_next_config(args: argparse.Namespace, state: dict[str, Any], *, run_index: int, run_number: int, phase: str) -> tuple[dict[str, Any], Path]:
    base = _load_json(args.base_config)
    if not base:
        raise RuntimeError(f"Could not load base config: {args.base_config}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"tropicalgt_i_pg_bpb_step0_full24b_b{run_index:02d}_{timestamp}_adv500_bpb112_{phase}"
    out_dir = args.output_root / run_name
    cfg = json.loads(json.dumps(base))
    cfg["run_name"] = run_name
    cfg["wandb_run_name"] = run_name
    cfg["output_dir"] = str(out_dir)
    cfg["checkpoint_dir"] = str(TGI_ROOT / "checkpoints")
    cfg["memory_bank_path"] = str(out_dir / "memory_bank" / "trajectory_memories.jsonl")
    cfg["data_root"] = str(args.dataset_root / "toricgt" / "curated_hf_shards")
    cfg["keys_path"] = str(PROJECT_ROOT / "keys.txt")
    cfg["target_bpb"] = float(args.target_bpb)
    cfg["validation_every_steps"] = 500
    cfg["visualization_every_steps"] = 500
    cfg["checkpoint_every"] = 500
    cfg["algebra_audit_interval"] = 500
    cfg["advanced_bpb_max_visual_audit_interval"] = 500
    cfg["require_graphcg_active_full_rank"] = True
    cfg["require_nontrivial_bundle_toric_losses"] = True
    cfg["parameter_golf_bpb_focus"] = True
    cfg["require_data"] = True
    cfg["device"] = "auto"
    cfg["seq_len"] = int(base.get("seq_len", 1024) or 1024)
    cfg["batch_size"] = _adaptive_batch_size(state, run_number)
    cfg["max_steps"] = max(
        int(args.boundary_steps),
        int(base.get("max_steps", args.boundary_steps)),
        _required_config_horizon_steps(cfg),
    )
    cfg["lr"] = _adaptive_float(state, "lr", default=1.55e-4, low=1.05e-4, high=2.15e-4, run_number=run_number)
    cfg["weight_decay"] = _adaptive_float(state, "weight_decay", default=0.024, low=0.014, high=0.04, run_number=run_number)
    cfg["grad_clip"] = _adaptive_float(state, "grad_clip", default=0.55, low=0.45, high=0.85, run_number=run_number)
    cfg["graph_bpb_side_weight"] = _adaptive_float(state, "graph_bpb_side_weight", default=0.82, low=0.55, high=1.05, run_number=run_number)
    cfg["train_limit"] = None
    cfg["val_limit"] = 128
    cfg["audit_level"] = "topology"
    cfg["audit_ph_backend"] = "gudhi"
    cfg["audit_max_simplices"] = 256
    cfg["periodic_audit_level"] = "topology"
    cfg["periodic_audit_ph_backend"] = "gudhi"
    cfg["periodic_audit_max_simplices"] = 128
    cfg["periodic_eval_details_limit"] = 3
    cfg["periodic_viz_limit"] = 1
    cfg["periodic_viz_got_examples"] = 1
    cfg["periodic_viz_got_record_pool"] = 12
    cfg["periodic_viz_got_scaling"] = True
    cfg["periodic_viz_failure_policy"] = "record_incomplete_without_fabrication"
    cfg["periodic_viz_got_training_safe_bounds"] = True
    cfg["periodic_viz_scale_depth"] = 18
    cfg["periodic_viz_scale_width"] = 24
    cfg["periodic_viz_scale_branch_factor"] = 8
    cfg["periodic_viz_trace_limit"] = 768
    cfg["periodic_viz_got_max_depth"] = 18
    cfg["periodic_viz_got_max_width"] = 24
    cfg["periodic_viz_got_max_branch_factor"] = 8
    cfg["periodic_viz_got_max_trace_limit"] = 768
    cfg["periodic_prune_got_audit_keep_latest"] = 1
    cfg["periodic_prune_got_audit_keep_steps"] = [5000]
    cfg["periodic_prune_got_audit_max_retained_steps"] = 2
    cfg["viz_got_scaling"] = True
    cfg["viz_scale_depth"] = 20
    cfg["viz_scale_width"] = 28
    cfg["viz_scale_branch_factor"] = 8
    cfg["viz_trace_limit"] = 1024
    cfg["final_interactive_artifacts_enabled"] = True
    cfg["periodic_interactive_artifacts_enabled"] = True
    cfg["periodic_browser_publish"] = True
    cfg["wandb_log_interactive_artifacts"] = False
    cfg["memory_max_records"] = 512
    cfg["memory_records_per_audit"] = 6
    cfg["memory_quality_min_nll_improvement"] = 0.01
    cfg["memory_quality_min_margin_mean"] = 0.00025
    cfg["memory_quality_min_probability_vertices"] = 4
    cfg["memory_quality_min_probability_simplices"] = 4
    cfg["memory_quality_require_probability_complex"] = True
    cfg["memory_quality_require_topological_algebra"] = True
    cfg["periodic_memory_retrieve_top_k"] = 24
    cfg["inference_memory_retrieve_top_k"] = 24
    cfg["memory_retrieval_landscape_weight"] = _adaptive_float(state, "memory_retrieval_landscape_weight", default=0.12, low=0.08, high=0.18, run_number=run_number)
    cfg["memory_retrieval_vector_weight"] = _adaptive_float(state, "memory_retrieval_vector_weight", default=0.22, low=0.14, high=0.30, run_number=run_number)
    cfg["memory_retrieval_probability_map_weight"] = _adaptive_float(state, "memory_retrieval_probability_map_weight", default=0.28, low=0.18, high=0.36, run_number=run_number)
    cfg["memory_retrieval_certified_cas_weight"] = _adaptive_float(state, "memory_retrieval_certified_cas_weight", default=0.16, low=0.10, high=0.22, run_number=run_number)
    cfg["restart_policy"] = {
        "review_every_steps": int(args.boundary_steps),
        "minimum_steps_before_restart": int(args.boundary_steps),
        "restart_only_after_review": True,
        "target_bpb": float(args.target_bpb),
        "max_primary_runs": int(args.max_primary_runs),
        "followup_runs": int(args.followup_runs),
        "restart_mode": "step0_adaptive_campaign",
    }
    cfg["campaign_policy"] = {
        "schema_version": "tropicalgt.advanced_bpb_campaign_policy.v1",
        "run_number": run_number,
        "run_index": run_index,
        "phase": phase,
        "target_bpb": float(args.target_bpb),
        "boundary_steps": int(args.boundary_steps),
        "dataset_root_requested": str(args.dataset_root),
        "adaptive_sweep": True,
        "codex_review_requested": bool(args.invoke_codex),
    }
    cfg["hybrid_data"] = _hybrid_data_config(cfg.get("hybrid_data", {}), args)
    cfg["required_hybrid_sources"] = ["tropicalgt_hf_reasoning", "openai_parameter_golf"]
    cfg["hybrid_source_requirements"] = {
        "openai_parameter_golf": {"min_files": 195, "min_raw_tokens": 19_473_201_340},
        "tropicalgt_hf_reasoning": {"min_examples": 4_633_582, "min_files": 117},
    }
    wandb_cfg = cfg.get("wandb", {}) if isinstance(cfg.get("wandb"), dict) else {}
    wandb_cfg.update({"enabled": True, "mode": "online", "project": "TropicalGT-I", "entity": "amelie-iska-math", "run_name": run_name})
    cfg["wandb"] = wandb_cfg
    model = cfg.setdefault("model", {})
    dim = int(model.get("dim", 1760) or 1760)
    model["graphcg_num_directions"] = dim
    model["graphcg_active_directions"] = dim
    model["enable_chart_bundle_auxiliary"] = True
    model["use_sequence_tropical"] = True
    model["sequence_tropical_weight"] = _adaptive_float(state, "sequence_tropical_weight", default=0.030, low=0.020, high=0.040, run_number=run_number)
    model["gflownet_weight"] = _adaptive_float(state, "gflownet_weight", default=0.0040, low=0.0025, high=0.0065, run_number=run_number)
    model["graphcg_weight"] = _adaptive_float(state, "graphcg_weight", default=0.0030, low=0.0020, high=0.0055, run_number=run_number)
    model["certificate_weight"] = _adaptive_float(state, "certificate_weight", default=6.0e-5, low=3.0e-5, high=1.2e-4, run_number=run_number)
    model["bundle_transport_weight"] = _adaptive_float(state, "bundle_transport_weight", default=1.5e-5, low=6.0e-6, high=3.5e-5, run_number=run_number)
    model["bundle_cocycle_weight"] = _adaptive_float(state, "bundle_cocycle_weight", default=1.0e-5, low=4.0e-6, high=2.5e-5, run_number=run_number)
    model["bundle_flat_rank_weight"] = _adaptive_float(state, "bundle_flat_rank_weight", default=1.0e-5, low=4.0e-6, high=2.5e-5, run_number=run_number)
    model["toric_normal_fan_weight"] = _adaptive_float(state, "toric_normal_fan_weight", default=2.5e-5, low=8.0e-6, high=5.0e-5, run_number=run_number)
    model["graphcg_toric_cell_agreement_weight"] = _adaptive_float(state, "graphcg_toric_cell_agreement_weight", default=1.5e-5, low=5.0e-6, high=3.5e-5, run_number=run_number)
    model["chart_bpb_consistency_weight"] = _adaptive_float(state, "chart_bpb_consistency_weight", default=1.0e-5, low=3.5e-6, high=2.5e-5, run_number=run_number)
    model["bundle_atom_stability_weight"] = _adaptive_float(state, "bundle_atom_stability_weight", default=1.0e-5, low=3.5e-6, high=2.5e-5, run_number=run_number)
    cfg["source_mix_note"] = (
        "OAI Parameter-Golf FineWeb is upweighted relative to older configs to improve BPB, "
        "while curated HF graph/reasoning shards remain required for graph-token and advanced auxiliary training."
    )

    section, gates = advanced_bpb_contract_report(cfg)
    failed = [gate for gate in gates if gate.get("status") == "fail"]
    if failed:
        detail = "\n".join(f"- {gate.get('name')}: {gate.get('detail', '')}" for gate in failed)
        raise RuntimeError(f"Generated campaign config violates advanced BPB contract:\n{detail}")
    cfg["generated_advanced_bpb_contract"] = section
    cfg["generated_advanced_bpb_contract_gates"] = gates
    config_path = args.config_root / f"{run_name}.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg, config_path


def _hybrid_data_config(existing: Any, args: argparse.Namespace) -> dict[str, Any]:
    cfg = existing if isinstance(existing, dict) else {}
    return {
        "enabled": True,
        "seed": int(cfg.get("seed", 1729) or 1729),
        "sources": [
            {
                "kind": "parquet",
                "name": "tropicalgt_hf_reasoning",
                "required": True,
                "root": str(args.dataset_root / "toricgt" / "curated_hf_shards"),
                "weight": 0.45,
            },
            {
                "allow_token_id_fallback": False,
                "kind": "parameter_golf_bin",
                "max_graph_chunks": 96,
                "name": "openai_parameter_golf",
                "required": True,
                "root": str(args.oai_root),
                "tokenizer_path": str(args.oai_tokenizer),
                "weight": 0.55,
                "window_tokens": 1025,
            },
        ],
    }


def _run_one_boundary(args: argparse.Namespace, cfg: dict[str, Any], config_path: Path) -> RunDecision:
    run_name = str(cfg["run_name"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "campaign_train.log"
    checkpoint_path = Path(cfg["checkpoint_dir"]) / f"{run_name}.pt"
    report_path = output_dir / "train_report.json"
    cmd = [args.python, str(args.train_script), "--config", str(config_path), "--max-steps", str(int(args.boundary_steps))]
    decision = RunDecision(
        run_name=run_name,
        config_path=config_path,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        log_path=log_path,
        step_boundary=int(args.boundary_steps),
    )
    command_json = output_dir / "campaign_train_command.json"
    command_json.write_text(json.dumps({"command": cmd, "cwd": str(TGI_ROOT), "started_at": _now_iso()}, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_name": run_name, "command": cmd, "log": str(log_path)}, indent=2))
        return decision
    env = os.environ.copy()
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=TGI_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    decision.returncode = int(proc.returncode)
    if proc.returncode != 0:
        decision.failed = True
        decision.failure_reason = _classify_failure(log_path)
        return decision
    report = _load_json(report_path)
    decision.bpb = _metric(report, ("metrics.eval_bpb", "eval.bpb", "metrics.bpb", "metrics.eval_bpb_exact"))
    decision.graph_bpb = _metric(report, ("metrics.eval_graph_bpb", "eval.graph_bpb", "metrics.graph_bpb"))
    decision.reached_target = decision.bpb is not None and decision.bpb <= float(args.target_bpb)
    if args.prepare_review_script.exists():
        _run_prepare_review(args, cfg, decision)
    return decision


def _run_prepare_review(args: argparse.Namespace, cfg: dict[str, Any], decision: RunDecision) -> None:
    bundle_dir = decision.output_dir / "post_5k_review_bundle"
    cmd = [
        args.python,
        str(args.prepare_review_script),
        "--config",
        str(decision.config_path),
        "--report",
        str(decision.report_path),
        "--checkpoint",
        str(decision.checkpoint_path),
        "--boundary-step",
        str(int(args.boundary_steps)),
        "--target-bpb",
        str(float(args.target_bpb)),
        "--output-dir",
        str(bundle_dir),
        "--run-interactive-audit-validators",
        "--command-timeout-seconds",
        "1200",
    ]
    log_path = bundle_dir / "prepare_review_bundle.log"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=TGI_ROOT, stdout=handle, stderr=subprocess.STDOUT)


def _write_boundary_report(
    args: argparse.Namespace,
    state: dict[str, Any],
    cfg: dict[str, Any],
    decision: RunDecision,
    *,
    phase: str,
    run_number: int,
) -> Path:
    report = _load_json(decision.report_path)
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    history = report.get("history", []) if isinstance(report.get("history"), list) else []
    periodic = report.get("periodic_artifacts", []) if isinstance(report.get("periodic_artifacts"), list) else []
    latest_periodic = periodic[-1] if periodic else {}
    advanced_keys = [
        "loss_bundle_transport_weighted",
        "loss_bundle_cocycle_weighted",
        "loss_bundle_flat_rank_weighted",
        "loss_toric_normal_fan_weighted",
        "loss_graphcg_toric_cell_agreement_weighted",
        "loss_chart_bpb_consistency_weighted",
        "loss_bundle_atom_stability_weighted",
        "loss_sequence_tropical_weighted",
        "loss_certificate_weighted",
        "loss_graphcg_weighted",
        "loss_gflownet_weighted",
    ]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    note_path = args.notes_root / f"{timestamp}_{decision.run_name}_5k_review.md"
    best_bpb = _best_metric(history, ("eval_bpb", "bpb"))
    recommendations = _recommendations(state, cfg, decision, metrics)
    lines = [
        f"# 5K Review: `{decision.run_name}`",
        "",
        f"- Generated: `{_now_iso()}`",
        f"- Phase/run: `{phase}` / `{run_number}`",
        f"- Config: `{decision.config_path}`",
        f"- Output: `{decision.output_dir}`",
        f"- Log: `{decision.log_path}`",
        f"- Report: `{decision.report_path}`",
        f"- Checkpoint: `{decision.checkpoint_path}`",
        f"- W&B run: `{cfg.get('wandb_run_name') or cfg.get('run_name')}`",
        f"- Return code: `{decision.returncode}`",
        "",
        "## BPB Status",
        "",
        f"- Target BPB: `{args.target_bpb:.4f}`",
        f"- Last eval BPB: `{_fmt(decision.bpb)}`",
        f"- Best observed BPB in history: `{_fmt(best_bpb)}`",
        f"- Last eval graph-BPB: `{_fmt(decision.graph_bpb)}`",
        f"- Target reached: `{decision.reached_target}`",
        f"- Failure: `{decision.failed}` {decision.failure_reason}",
        "",
        "## Advanced Loss Snapshot",
        "",
    ]
    for key in advanced_keys:
        lines.append(f"- `{key}`: `{_fmt(metrics.get(key))}`")
    lines.extend(
        [
            "",
            "## Periodic Artifact Inventory",
            "",
            f"- Periodic artifact count: `{len(periodic)}`",
            f"- Latest periodic artifact: `{latest_periodic}`",
            f"- Latest HTML files under output: `{len(list(decision.output_dir.rglob('*.html')))}`",
            f"- Latest PNG files under output: `{len(list(decision.output_dir.rglob('*.png')))}`",
            "",
            "## Hyperparameter Snapshot",
            "",
            f"- `batch_size`: `{cfg.get('batch_size')}`",
            f"- `seq_len`: `{cfg.get('seq_len')}`",
            f"- `lr`: `{cfg.get('lr')}`",
            f"- `weight_decay`: `{cfg.get('weight_decay')}`",
            f"- `graph_bpb_side_weight`: `{cfg.get('graph_bpb_side_weight')}`",
            f"- `validation_every_steps`: `{cfg.get('validation_every_steps')}`",
            f"- `visualization_every_steps`: `{cfg.get('visualization_every_steps')}`",
            f"- `GraphCG active/total`: `{cfg.get('model', {}).get('graphcg_active_directions')}/{cfg.get('model', {}).get('graphcg_num_directions')}`",
            "",
            "## Recommendations For Next Step-0 Run",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(
        [
            "",
            "## Raw Metric Keys",
            "",
            "```text",
            "\n".join(sorted(str(key) for key in metrics.keys()))[:12000],
            "```",
            "",
        ]
    )
    note_path.write_text("\n".join(lines), encoding="utf-8")
    prompt_path = args.notes_root / f"{timestamp}_{decision.run_name}_codex_prompt.md"
    prompt_path.write_text(_codex_prompt(note_path, cfg, decision, recommendations), encoding="utf-8")
    if args.invoke_codex:
        codex_out = args.notes_root / f"{timestamp}_{decision.run_name}_codex_output.md"
        try:
            result = subprocess.run(
                shlex.split(args.codex_command),
                input=prompt_path.read_text(encoding="utf-8"),
                text=True,
                cwd=TGI_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
            codex_out.write_text(result.stdout or f"codex returncode={result.returncode}\n", encoding="utf-8")
        except Exception as exc:
            codex_out.write_text(f"Codex invocation unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
    return note_path


def _write_campaign_synopsis(args: argparse.Namespace, state: dict[str, Any], *, reason: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.notes_root / f"{timestamp}_advanced_bpb_campaign_synopsis_{reason}.md"
    runs = state.get("runs", []) if isinstance(state.get("runs"), list) else []
    ranked = sorted(
        [run for run in runs if isinstance(run, dict) and run.get("bpb") is not None],
        key=lambda item: float(item.get("bpb", math.inf)),
    )
    lines = [
        f"# Advanced BPB Campaign Synopsis: {reason}",
        "",
        f"- Generated: `{_now_iso()}`",
        f"- Target BPB: `{args.target_bpb}`",
        f"- Completed runs: `{len(runs)}`",
        "",
        "## Best Runs",
        "",
    ]
    for run in ranked[:10]:
        lines.append(
            f"- `{run.get('run_name')}`: BPB `{_fmt(run.get('bpb'))}`, graph-BPB `{_fmt(run.get('graph_bpb'))}`, config `{run.get('config_path')}`"
        )
    lines.extend(
        [
            "",
            "## Next Adaptive Ideas",
            "",
            "- If BPB remains plateaued, increase OAI FineWeb source weight and reduce high-variance auxiliary weights.",
            "- If structure metrics improve while BPB worsens, anneal certificate, sequence-tropical, and bundle/toric weights down by 25-50%.",
            "- If GPU OOMs, reduce batch by 4 while preserving context length and full-rank GraphCG.",
            "- If graph-BPB improves but OAI BPB does not, lower graph side weight and raise direct LM exposure.",
            "- Consider a two-phase run: first 1K steps with advanced losses warmed up from zero to the configured nonzero values, then full-strength auxiliaries through 5K.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    state.setdefault("synopses", []).append(str(path))
    _write_json(args.state_path, state)
    return path


def _adaptive_batch_size(state: dict[str, Any], run_number: int) -> int:
    failures = [run for run in state.get("runs", []) if isinstance(run, dict) and "oom" in str(run.get("failure_reason", "")).lower()]
    base = 60
    if failures:
        return max(36, base - 4 * min(len(failures), 5))
    choices = [60, 56, 52, 64]
    return choices[(run_number - 1) % len(choices)]


def _required_config_horizon_steps(cfg: dict[str, Any]) -> int:
    """Return the configured max_steps needed to satisfy data-budget gates.

    The campaign still launches each review boundary with ``--max-steps 5000``.
    This horizon only satisfies the full-run budget contract that guards against
    underspecified Parameter-Golf runs.
    """
    min_slots = cfg.get("min_training_token_slots")
    if not isinstance(min_slots, (int, float)) or min_slots <= 0:
        return 0
    batch_size = max(int(cfg.get("batch_size", 1) or 1), 1)
    seq_len = max(int(cfg.get("seq_len", 1) or 1), 1)
    return int(math.ceil(float(min_slots) / float(batch_size * seq_len)))


def _adaptive_float(state: dict[str, Any], key: str, *, default: float, low: float, high: float, run_number: int) -> float:
    best = _best_run(state)
    center = default
    if best and isinstance(best.get("config_snapshot"), dict):
        value = _nested_get(best["config_snapshot"], key)
        if isinstance(value, (int, float)) and value > 0:
            center = float(value)
    rng = random.Random(1729 + 7919 * run_number + sum(ord(ch) for ch in key))
    if run_number == 1:
        sampled = center
    else:
        log_center = math.log(max(center, 1e-12))
        log_low = math.log(max(low, 1e-12))
        log_high = math.log(max(high, 1e-12))
        sampled = math.exp(min(log_high, max(log_low, rng.gauss(log_center, 0.22))))
    return float(min(high, max(low, sampled)))


def _best_run(state: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        run for run in state.get("runs", []) if isinstance(run, dict) and isinstance(run.get("bpb"), (int, float))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda run: float(run.get("bpb", math.inf)))


def _decision_json(decision: RunDecision, *, phase: str, run_number: int) -> dict[str, Any]:
    cfg = _load_json(decision.config_path)
    snapshot_keys = {
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "grad_clip": cfg.get("grad_clip"),
        "batch_size": cfg.get("batch_size"),
        "graph_bpb_side_weight": cfg.get("graph_bpb_side_weight"),
        "sequence_tropical_weight": cfg.get("model", {}).get("sequence_tropical_weight") if isinstance(cfg.get("model"), dict) else None,
        "gflownet_weight": cfg.get("model", {}).get("gflownet_weight") if isinstance(cfg.get("model"), dict) else None,
        "graphcg_weight": cfg.get("model", {}).get("graphcg_weight") if isinstance(cfg.get("model"), dict) else None,
        "certificate_weight": cfg.get("model", {}).get("certificate_weight") if isinstance(cfg.get("model"), dict) else None,
        "bundle_transport_weight": cfg.get("model", {}).get("bundle_transport_weight") if isinstance(cfg.get("model"), dict) else None,
        "toric_normal_fan_weight": cfg.get("model", {}).get("toric_normal_fan_weight") if isinstance(cfg.get("model"), dict) else None,
        "memory_retrieval_landscape_weight": cfg.get("memory_retrieval_landscape_weight"),
        "memory_retrieval_vector_weight": cfg.get("memory_retrieval_vector_weight"),
        "memory_retrieval_probability_map_weight": cfg.get("memory_retrieval_probability_map_weight"),
        "memory_retrieval_certified_cas_weight": cfg.get("memory_retrieval_certified_cas_weight"),
    }
    return {
        "run_number": run_number,
        "phase": phase,
        "run_name": decision.run_name,
        "config_path": str(decision.config_path),
        "output_dir": str(decision.output_dir),
        "checkpoint_path": str(decision.checkpoint_path),
        "report_path": str(decision.report_path),
        "log_path": str(decision.log_path),
        "step_boundary": decision.step_boundary,
        "returncode": decision.returncode,
        "bpb": decision.bpb,
        "graph_bpb": decision.graph_bpb,
        "reached_target": decision.reached_target,
        "failed": decision.failed,
        "failure_reason": decision.failure_reason,
        "config_snapshot": snapshot_keys,
    }


def _recommendations(state: dict[str, Any], cfg: dict[str, Any], decision: RunDecision, metrics: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if decision.failed:
        if "oom" in decision.failure_reason.lower():
            items.append("Next run should reduce batch size by 4 and keep sequence length fixed to preserve context.")
        else:
            items.append("Investigate the training log before launching another campaign run; this may be a code/data failure rather than a hyperparameter issue.")
        return items
    if decision.bpb is None:
        items.append("Primary eval BPB is missing; verify validation reporting and OAI source-specific metric emission.")
    elif decision.bpb > 1.3:
        items.append("BPB is far above target; prioritize OAI-source weight, LR stability, and lower auxiliary pressure before increasing geometric losses.")
    elif decision.bpb > 1.12:
        items.append("BPB missed target but is in range for sweeps; jitter LR, batch size, and graph-BPB side weight around the incumbent.")
    else:
        items.append("Target reached; stop the campaign and preserve the checkpoint/artifacts.")
    weighted_terms = [key for key, value in metrics.items() if key.startswith("loss_") and key.endswith("_weighted") and isinstance(value, (int, float)) and abs(float(value)) > 0]
    if not weighted_terms:
        items.append("No weighted advanced loss terms were observed; check model config and W&B grouping.")
    if cfg.get("hybrid_data", {}).get("sources"):
        items.append("Keep OAI FineWeb weight high until source-specific BPB improves; do not let graph/reasoning examples dominate early BPB optimization.")
    return items


def _codex_prompt(note_path: Path, cfg: dict[str, Any], decision: RunDecision, recommendations: list[str]) -> str:
    return "\n".join(
        [
            "Review this TropicalGT-I 5K BPB campaign boundary.",
            "",
            f"Run: {decision.run_name}",
            f"Config: {decision.config_path}",
            f"Report: {decision.report_path}",
            f"Checkpoint: {decision.checkpoint_path}",
            f"Review note: {note_path}",
            f"Target BPB: {cfg.get('target_bpb')}",
            f"Observed eval BPB: {decision.bpb}",
            f"Observed graph BPB: {decision.graph_bpb}",
            "",
            "Required output: write concise evidence-backed hyperparameter recommendations for the next step-0 run.",
            "Do not suggest training on validation data or violating Parameter-Golf constraints.",
            "",
            "Supervisor recommendations:",
            *[f"- {item}" for item in recommendations],
        ]
    )


def _hard_failure(decision: RunDecision) -> bool:
    text = decision.failure_reason.lower()
    if "oom" in text:
        return False
    return decision.failed


def _classify_failure(log_path: Path) -> str:
    if not log_path.exists():
        return "missing_log"
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:].lower()
    if "out of memory" in tail or "cuda error: out of memory" in tail or "torch.cuda.outofmemoryerror" in tail:
        return "cuda_oom"
    if "advanced bpb training contract failed" in tail:
        return "advanced_bpb_contract_failed"
    if "training data budget check failed" in tail:
        return "training_data_budget_failed"
    return "nonzero_returncode"


def _metric(report: dict[str, Any], paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value = _nested_get(report, path)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _best_metric(history: list[Any], keys: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
    return min(values) if values else None


def _nested_get(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return "missing"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
