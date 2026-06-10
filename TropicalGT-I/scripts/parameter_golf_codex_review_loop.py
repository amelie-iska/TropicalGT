#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "TropicalGT-I" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tropicalgt.run import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5K-step Codex review loop for graph-encoded Parameter-Golf BPB training")
    parser.add_argument("--config", type=Path, default=ROOT / "TropicalGT-I" / "configs" / "train.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=ROOT / "TropicalGT-I" / "scripts" / "train_tropicalgt_i.py")
    parser.add_argument("--review-every-steps", type=int, default=5000)
    parser.add_argument("--target-bpb", type=float, default=1.18)
    parser.add_argument("--metric", default="eval.bpb")
    parser.add_argument("--graph-metric", default="eval.graph_bpb")
    parser.add_argument("--max-total-steps", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--restart-policy", choices=("beginning", "latest", "previous-review"), default="beginning")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TropicalGT-I" / "outputs" / "parameter_golf_codex_reviews")
    parser.add_argument("--codex-command", default=os.environ.get("CODEX_REVIEW_COMMAND", "codex exec"))
    parser.add_argument("--cuda-alloc-conf", default=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
    parser.add_argument("--invoke-codex", action="store_true", help="Actually run the configured Codex command. Without this, prompts are written only.")
    parser.add_argument("--once", action="store_true", help="Review the current report/checkpoint once without launching training.")
    parser.add_argument("--dry-run", action="store_true", help="Print intended train/review actions without executing them.")
    parser.add_argument("--max-reviews", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.get("output_dir", "TropicalGT-I/outputs/train")) / "train_report.json"
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "TropicalGT-I/checkpoints"))
    run_name = cfg.get("run_name", "tropicalgt_i_train")
    final_checkpoint = checkpoint_dir / f"{run_name}.pt"
    latest_checkpoint = checkpoint_dir / f"{run_name}.latest.pt"
    max_total_steps = int(args.max_total_steps or cfg.get("max_steps", args.review_every_steps))
    state = {
        "config": str(args.config),
        "target_bpb": args.target_bpb,
        "metric": args.metric,
        "graph_metric": args.graph_metric,
        "review_every_steps": args.review_every_steps,
        "max_total_steps": max_total_steps,
        "restart_policy": args.restart_policy,
        "started_at": time.time(),
        "events": [],
    }

    if args.once:
        _review_boundary(
            args=args,
            state=state,
            cfg=cfg,
            report_path=report_path,
            checkpoint_path=args.resume_from or final_checkpoint,
            previous_boundary_checkpoint=None,
            boundary_step=_current_step(report_path, args.resume_from or final_checkpoint),
            output_dir=output_dir,
        )
        _write_state(output_dir, state)
        return

    resume_from = args.resume_from
    previous_boundary_checkpoint: Path | None = None
    boundary_step = args.review_every_steps
    reviews = 0
    while boundary_step <= max_total_steps and reviews < args.max_reviews:
        train_cmd = _train_command(args.python, args.train_script, args.config, boundary_step, resume_from)
        event: dict[str, Any] = {"boundary_step": boundary_step, "train_command": train_cmd, "resume_from": str(resume_from or "")}
        if args.dry_run:
            event["dry_run"] = True
            print(json.dumps(event, indent=2))
        else:
            env = os.environ.copy()
            if args.cuda_alloc_conf:
                env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
            subprocess.run(train_cmd, cwd=ROOT, check=True, env=env)
        checkpoint_path = final_checkpoint if final_checkpoint.exists() else latest_checkpoint
        snapshot_path = _snapshot_checkpoint(checkpoint_path, output_dir, boundary_step) if checkpoint_path.exists() else None
        event["checkpoint_snapshot"] = str(snapshot_path or "")
        state["events"].append(event)
        decision = _review_boundary(
            args=args,
            state=state,
            cfg=load_config(args.config),
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            previous_boundary_checkpoint=previous_boundary_checkpoint,
            boundary_step=boundary_step,
            output_dir=output_dir,
        )
        reviews += 1
        _write_state(output_dir, state)
        if decision.get("triggered"):
            if args.restart_policy == "beginning":
                resume_from = None
                boundary_step = args.review_every_steps
            elif args.restart_policy == "previous-review" and previous_boundary_checkpoint is not None:
                resume_from = previous_boundary_checkpoint
                boundary_step = _checkpoint_step(previous_boundary_checkpoint) + args.review_every_steps
            else:
                resume_from = snapshot_path or checkpoint_path
                boundary_step = _checkpoint_step(resume_from) + args.review_every_steps
        else:
            previous_boundary_checkpoint = snapshot_path or checkpoint_path
            resume_from = snapshot_path or checkpoint_path
            boundary_step += args.review_every_steps
    _write_state(output_dir, state)


def _review_boundary(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    cfg: dict[str, Any],
    report_path: Path,
    checkpoint_path: Path,
    previous_boundary_checkpoint: Path | None,
    boundary_step: int,
    output_dir: Path,
) -> dict[str, Any]:
    report = _load_report(report_path)
    checkpoint = _load_checkpoint_summary(checkpoint_path)
    bpb = _metric_value(report, checkpoint, args.metric)
    graph_bpb = _metric_value(report, checkpoint, args.graph_metric)
    triggered = bpb is None or bpb > args.target_bpb
    active_contract = _active_training_contract(cfg, report, checkpoint, boundary_step)
    contract_json = output_dir / f"active_training_contract_step_{boundary_step:08d}.json"
    contract_md = output_dir / f"active_training_contract_step_{boundary_step:08d}.md"
    contract_json.write_text(json.dumps(active_contract, indent=2), encoding="utf-8")
    contract_md.write_text(_active_training_contract_markdown(active_contract), encoding="utf-8")
    prompt = _review_prompt(
        cfg=cfg,
        report=report,
        checkpoint=checkpoint,
        active_contract=active_contract,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        previous_boundary_checkpoint=previous_boundary_checkpoint,
        boundary_step=boundary_step,
        metric=args.metric,
        bpb=bpb,
        graph_metric=args.graph_metric,
        graph_bpb=graph_bpb,
        target_bpb=args.target_bpb,
        restart_policy=args.restart_policy,
    )
    prompt_path = output_dir / f"codex_review_step_{boundary_step:08d}.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    decision = {
        "boundary_step": boundary_step,
        "metric": args.metric,
        "bpb": bpb,
        "graph_metric": args.graph_metric,
        "graph_bpb": graph_bpb,
        "target_bpb": args.target_bpb,
        "triggered": triggered,
        "prompt": str(prompt_path),
        "active_contract_json": str(contract_json),
        "active_contract_markdown": str(contract_md),
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
        "restart_policy": args.restart_policy,
    }
    if triggered and args.invoke_codex:
        decision["codex_command"] = args.codex_command
        if args.dry_run:
            decision["codex_dry_run"] = True
        else:
            result = subprocess.run(shlex.split(args.codex_command), input=prompt, text=True, cwd=ROOT)
            decision["codex_returncode"] = result.returncode
            if result.returncode != 0:
                raise SystemExit(result.returncode)
    state.setdefault("reviews", []).append(decision)
    print(json.dumps(decision, indent=2))
    return decision


def _train_command(python: str, train_script: Path, config: Path, max_steps: int, resume_from: Path | None) -> list[str]:
    cmd = [python, str(train_script), "--config", str(config), "--max-steps", str(int(max_steps))]
    if resume_from:
        cmd.extend(["--resume-from", str(resume_from)])
    return cmd


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_checkpoint_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = torch.load(path, map_location="cpu")
    metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
    history = obj.get("history", []) if isinstance(obj, dict) else []
    return {
        "path": str(path),
        "step": int(obj.get("step", metrics.get("step", 0))) if isinstance(obj, dict) else 0,
        "metrics": metrics,
        "history_tail": history[-20:] if isinstance(history, list) else [],
        "config": obj.get("config", {}) if isinstance(obj, dict) else {},
    }


def _metric_value(report: dict[str, Any], checkpoint: dict[str, Any], dotted: str) -> float | None:
    for root in (report, checkpoint):
        value = _nested_get(root, dotted)
        if isinstance(value, (int, float)):
            return float(value)
    if dotted.startswith("eval."):
        value = _nested_get(report, dotted.replace("eval.", "metrics.eval_", 1))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _nested_get(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _current_step(report_path: Path, checkpoint_path: Path) -> int:
    report = _load_report(report_path)
    if isinstance(report.get("final_step"), int):
        return int(report["final_step"])
    return _checkpoint_step(checkpoint_path)


def _checkpoint_step(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    obj = torch.load(path, map_location="cpu")
    return int(obj.get("step", obj.get("metrics", {}).get("step", 0))) if isinstance(obj, dict) else 0


def _snapshot_checkpoint(path: Path, output_dir: Path, boundary_step: int) -> Path:
    snapshot_dir = output_dir / "checkpoints"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"checkpoint_step_{boundary_step:08d}.pt"
    shutil.copy2(path, snapshot)
    return snapshot


def _review_prompt(
    *,
    cfg: dict[str, Any],
    report: dict[str, Any],
    checkpoint: dict[str, Any],
    active_contract: dict[str, Any],
    report_path: Path,
    checkpoint_path: Path,
    previous_boundary_checkpoint: Path | None,
    boundary_step: int,
    metric: str,
    bpb: float | None,
    graph_metric: str,
    graph_bpb: float | None,
    target_bpb: float,
    restart_policy: str,
) -> str:
    history_tail = checkpoint.get("history_tail") or report.get("history", [])[-20:]
    compact = {
        "boundary_step": boundary_step,
        "target_bpb": target_bpb,
        "observed_bpb": bpb,
        "observed_graph_bpb": graph_bpb,
        "metric": metric,
        "graph_metric": graph_metric,
        "restart_policy": restart_policy,
        "report_path": str(report_path),
        "checkpoint_path": str(checkpoint_path),
        "previous_boundary_checkpoint": str(previous_boundary_checkpoint or ""),
        "model": cfg.get("model", {}),
        "optimizer": {k: cfg.get(k) for k in ("lr", "weight_decay", "grad_clip", "batch_size", "seq_len", "max_steps")},
        "tokengt": cfg.get("tokengt", {}),
        "sampler": {k: cfg.get(k) for k in ("chunk_shuffle", "chunk_shuffle_seed", "shuffle_rows_within_chunk", "cache_shards")},
        "final_metrics": report.get("metrics", checkpoint.get("metrics", {})),
        "eval": report.get("eval", {}),
        "history_tail": history_tail,
        "active_training_contract": active_contract,
    }
    return (
        "# TropicalGT-I Parameter-Golf 5K Review\n\n"
        "You are the single Codex review agent for a graph-encoded OpenAI Parameter-Golf BPB run.\n"
        f"The run reached review boundary step `{boundary_step}`. The target is `{metric} < {target_bpb}`.\n"
        "If the target is not met, inspect the metrics, losses, hyperparameters, graph-token accounting, "
        "GFlowNet/GraphCG/tropical regularizers, checkpoint history, and visualization/audit artifacts. "
        "Then make concrete code/config changes that improve ordinary BPB first and graph-BPB second, while "
        "retaining useful advanced TropicalGT techniques. Prefer restarting from the beginning unless the "
        "evidence clearly supports resuming from an earlier checkpoint.\n\n"
        "Do not expose secrets. Do not commit datasets, checkpoints, W&B runs, or generated artifacts. "
        "After edits, run the relevant tests/readiness checks and leave a concise review note in `planning/`.\n\n"
        "```json\n"
        f"{json.dumps(compact, indent=2)}\n"
        "```\n"
    )


def _active_training_contract(cfg: dict[str, Any], report: dict[str, Any], checkpoint: dict[str, Any], boundary_step: int) -> dict[str, Any]:
    metrics = dict(report.get("metrics") or checkpoint.get("metrics") or {})
    eval_metrics = dict(report.get("eval") or {})
    history_tail = checkpoint.get("history_tail") or report.get("history", [])[-20:]
    model_cfg = dict(cfg.get("model", {}))
    return {
        "boundary_step": int(boundary_step),
        "objective": {
            "primary_metric": "eval.bpb",
            "primary_target": 1.18,
            "secondary_metric": "eval.graph_bpb",
            "base_loss": "byte-level autoregressive cross entropy / NLL",
            "graph_decoding": "causal DAG topological autoregressive order; deterministic random autoregressive order for non-causal or cyclic graphs",
            "dataset": "hybrid graph-structured TropicalGT reasoning data plus OpenAI Parameter Golf byte windows as sequential DAG records",
        },
        "active_losses": {
            "nll": metrics.get("nll"),
            "total_loss": metrics.get("loss"),
            "gflownet_trajectory_balance": metrics.get("gflownet_tb"),
            "graphcg_loss": metrics.get("graphcg_loss"),
            "tropical_certificate_loss": metrics.get("certificate_loss"),
            "tropical_margin_loss": metrics.get("tropical_margin_loss"),
            "tropical_entropy_loss": metrics.get("tropical_entropy_loss"),
            "regularizer_total": metrics.get("loss_regularizer_total"),
            "regularizer_ratio": metrics.get("loss_regularizer_ratio"),
        },
        "regularizer_weights": {
            "gflownet_weight": model_cfg.get("gflownet_weight"),
            "graphcg_weight": model_cfg.get("graphcg_weight"),
            "certificate_weight": model_cfg.get("certificate_weight"),
            "margin_weight": model_cfg.get("margin_weight", 0.002),
            "entropy_weight": model_cfg.get("entropy_weight", 0.001),
            "graph_bpb_side_weight": cfg.get("graph_bpb_side_weight"),
        },
        "compression_metrics": {
            "train_bpb": metrics.get("bpb"),
            "train_text_bpb": metrics.get("text_bpb"),
            "train_graph_bpb": metrics.get("graph_bpb"),
            "train_graph_sideinfo_bpb": metrics.get("graph_sideinfo_bpb"),
            "eval_bpb": eval_metrics.get("bpb", metrics.get("eval_bpb")),
            "eval_graph_bpb": eval_metrics.get("graph_bpb", metrics.get("eval_graph_bpb")),
            "eval_graph_sideinfo_bpb": eval_metrics.get("graph_sideinfo_bpb", metrics.get("eval_graph_sideinfo_bpb")),
            "eval_ppl": eval_metrics.get("ppl", metrics.get("eval_ppl")),
        },
        "tropical_metrics": _select_prefixed(metrics, ("support_", "margin_", "wall_", "certificate_", "sequence_tropical_", "tropical_")),
        "gflownet_metrics": _select_prefixed(metrics, ("gflownet_",)),
        "graphcg_metrics": _select_prefixed(metrics, ("graphcg_",)),
        "algebra_topology_metrics": _select_prefixed(metrics, ("algebra_", "homology_", "persistence_", "betti_", "syzygy_", "free_resolution_", "derived_")),
        "memory_metrics": _select_prefixed(metrics, ("analogical_memory_",)),
        "data_metrics": {
            "batch_size": cfg.get("batch_size"),
            "seq_len": cfg.get("seq_len"),
            "graph_tokens_mean": metrics.get("graph_tokens_mean"),
            "node_tokens_mean": metrics.get("node_tokens_mean"),
            "edge_tokens_mean": metrics.get("edge_tokens_mean"),
            "graph_token_node_edge_ratio": metrics.get("graph_token_node_edge_ratio"),
            "graph_json_fallback_rate": metrics.get("graph_json_fallback_rate"),
            "graph_json_sequentialized_rate": metrics.get("graph_json_sequentialized_rate"),
            "causal_dag_ar_rate": metrics.get("causal_dag_ar_rate"),
            "random_graph_ar_rate": metrics.get("random_graph_ar_rate"),
            "parameter_golf_source_rate": metrics.get("parameter_golf_source_rate"),
        },
        "optimization_hyperparameters": {
            "lr": cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay"),
            "grad_clip": cfg.get("grad_clip"),
            "max_steps": cfg.get("max_steps"),
            "checkpoint_every": cfg.get("checkpoint_every"),
            "chunk_shuffle": cfg.get("chunk_shuffle"),
            "shuffle_rows_within_chunk": cfg.get("shuffle_rows_within_chunk"),
        },
        "model_hyperparameters": model_cfg,
        "tokengt_hyperparameters": cfg.get("tokengt", {}),
        "hybrid_data": cfg.get("hybrid_data", {}),
        "throughput_and_vram": {
            "gpu_mem_mb": metrics.get("gpu_mem_mb"),
            "step_time_s": metrics.get("step_time_s"),
            "examples_per_sec": metrics.get("examples_per_sec"),
            "tokens_per_sec": metrics.get("tokens_per_sec"),
            "graph_tokens_per_sec": metrics.get("graph_tokens_per_sec"),
            "grad_norm": metrics.get("grad_norm"),
        },
        "history_tail": history_tail,
        "report_visualizations": report.get("visualizations", {}),
    }


def _active_training_contract_markdown(contract: dict[str, Any]) -> str:
    compression = contract.get("compression_metrics", {})
    objective = contract.get("objective", {})
    lines = [
        "# TropicalGT-I Active Training Contract",
        "",
        f"- Boundary step: `{contract.get('boundary_step')}`",
        f"- Primary metric: `{objective.get('primary_metric')}` target `< {objective.get('primary_target')}`",
        f"- Eval BPB: `{compression.get('eval_bpb')}`",
        f"- Eval graph BPB: `{compression.get('eval_graph_bpb')}`",
        f"- Graph decoding: {objective.get('graph_decoding')}",
        "",
        "## Active Losses",
        "```json",
        json.dumps(contract.get("active_losses", {}), indent=2),
        "```",
        "",
        "## Regularizer Weights",
        "```json",
        json.dumps(contract.get("regularizer_weights", {}), indent=2),
        "```",
        "",
        "## Metrics",
        "```json",
        json.dumps(
            {
                "compression": contract.get("compression_metrics", {}),
                "tropical": contract.get("tropical_metrics", {}),
                "gflownet": contract.get("gflownet_metrics", {}),
                "graphcg": contract.get("graphcg_metrics", {}),
                "algebra_topology": contract.get("algebra_topology_metrics", {}),
                "data": contract.get("data_metrics", {}),
                "throughput_and_vram": contract.get("throughput_and_vram", {}),
            },
            indent=2,
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _select_prefixed(metrics: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if any(key.startswith(prefix) for prefix in prefixes)}


def _write_state(output_dir: Path, state: dict[str, Any]) -> None:
    (output_dir / "review_loop_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
