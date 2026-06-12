#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from tropicalgt.data import dataset_manifest, make_dataset_from_config
from tropicalgt.diagnostics import gflownet_diagnostics, graphcg_diagnostics, record_diagnostics
from tropicalgt.metrics import batch_bpb_metrics
from tropicalgt.records import GraphRecord
from tropicalgt.run import build_model, collate_records, evaluate_model, load_checkpoint, load_config, load_keys
from tropicalgt.scaling import run_inference_scaling
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import write_inference_audit_artifacts, write_reasoning_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a TropicalGT-I readiness audit report")
    parser.add_argument("--config", default=str(ROOT / "configs" / "gpu_smoke.json"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--details-limit", type=int, default=2)
    parser.add_argument("--trace-limit", type=int, default=12)
    parser.add_argument("--scale-depth", type=int, default=1)
    parser.add_argument("--scale-width", type=int, default=2)
    parser.add_argument("--scale-branch-factor", type=int, default=2)
    parser.add_argument("--audit-level", choices=["none", "basic", "topology", "algebra", "full"], default="")
    parser.add_argument("--audit-ph-backend", choices=["auto", "gudhi", "ripser", "none"], default="")
    parser.add_argument("--audit-max-simplices", type=int, default=0)
    parser.add_argument("--check-ablation-tools", action="store_true")
    parser.add_argument("--check-wandb-key", action="store_true")
    parser.add_argument("--train-dry-run", action="store_true", help="Build the configured model and run one optimizer step on sampled records")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--render-visualizations", action="store_true")
    parser.add_argument(
        "--allow-incomplete-reasoning-visualizations",
        action="store_true",
        help="Allow rendered GoT readiness artifacts even when model-derived reasoning-step completeness checks fail.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()
    report = build_readiness_report(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        split=args.split,
        sample_limit=args.sample_limit,
        details_limit=args.details_limit,
        trace_limit=args.trace_limit,
        scale_depth=args.scale_depth,
        scale_width=args.scale_width,
        scale_branch_factor=args.scale_branch_factor,
        audit_level=args.audit_level,
        audit_ph_backend=args.audit_ph_backend,
        audit_max_simplices=args.audit_max_simplices,
        check_ablation_tools=args.check_ablation_tools,
        check_wandb_key=args.check_wandb_key,
        train_dry_run=args.train_dry_run,
        require_cuda=args.require_cuda,
        require_checkpoint=args.require_checkpoint,
        render_visualizations=args.render_visualizations,
        require_complete_reasoning_steps=not args.allow_incomplete_reasoning_visualizations,
    )
    output = Path(args.output) if args.output else Path(report["output_dir"]) / "readiness_audit.json"
    markdown = Path(args.markdown) if args.markdown else output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(output), "markdown": str(markdown), "failed_gates": report["failed_gates"]}, indent=2))
    if report["failed_gates"]:
        raise SystemExit(1)


def build_readiness_report(
    *,
    config_path: Path,
    checkpoint_path: Path | None,
    split: str,
    sample_limit: int,
    details_limit: int,
    trace_limit: int,
    scale_depth: int,
    scale_width: int,
    scale_branch_factor: int,
    audit_level: str = "",
    audit_ph_backend: str = "",
    audit_max_simplices: int = 0,
    check_ablation_tools: bool = False,
    check_wandb_key: bool = False,
    train_dry_run: bool = False,
    require_cuda: bool = False,
    require_checkpoint: bool = False,
    render_visualizations: bool = False,
    require_complete_reasoning_steps: bool = True,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = Path(cfg.get("output_dir", ROOT / "outputs" / "gpu_smoke"))
    report: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "output_dir": str(out_dir),
        "gates": [],
    }
    gates: list[dict[str, Any]] = report["gates"]
    add_gate(gates, "config_loads", True, f"Loaded {config_path}")
    add_gate(gates, "seed_configured", "seed" in cfg, str(cfg.get("seed", "")))
    report["environment"] = environment_report()
    add_gate(gates, "cuda_available", torch.cuda.is_available() or not require_cuda, "CUDA required" if require_cuda else "CUDA optional")
    required_packages = ("torch", "plotly", "sklearn", "tqdm", "wandb", "pyarrow", "networkx", "sympy")
    add_gate(
        gates,
        "required_packages",
        all(report["environment"]["packages"].get(name, {}).get("available", False) for name in required_packages),
        ",".join(required_packages),
    )
    report["tooling"] = tooling_report()
    add_gate(gates, "core_scripts_present", all(item["exists"] for item in report["tooling"]["core_scripts"].values()), "train/eval/infer/validate/render/analyze/grid")
    if check_ablation_tools:
        add_gate(gates, "ablation_tools_present", all(item["exists"] for item in report["tooling"]["ablation_scripts"].values()), "BPB analyzer and grid runner")
    if check_wandb_key:
        wandb_enabled = bool(cfg.get("wandb", {}).get("enabled", False))
        keys = load_keys(cfg.get("keys_path", "keys.txt"))
        add_gate(
            gates,
            "wandb_key_available",
            (not wandb_enabled) or bool(keys.get("wandb")),
            "enabled" if wandb_enabled else "wandb disabled",
        )

    root = cfg.get("data_root")
    require_data = bool(cfg.get("require_data", bool(root)))
    cfg_sample = dict(cfg)
    if split == "train":
        cfg_sample["train_limit"] = sample_count = max(1, int(sample_limit))
    else:
        cfg_sample["val_limit"] = sample_count = max(1, int(sample_limit))
    ds = make_dataset_from_config(cfg_sample, split)
    manifest = dataset_manifest(ds, root)
    report["data"] = {"root": root or "", "require_data": require_data, "manifest": manifest}
    add_gate(gates, "data_root_present", bool(root) or not require_data, root or "fixture mode")
    if root:
        add_gate(gates, "data_splits_present", bool(manifest), "dataset manifest resolved")
    records = [ds[i] for i in range(len(ds))]
    tokenizer = TokenGTTokenizer(**cfg.get("tokengt", {}))
    graph_batch = tokenizer.batch_encode(records[: max(1, min(len(records), int(cfg.get("batch_size", 2))))])
    fallback_count = sum(1 for record in records if (record.metadata or {}).get("graph_json_fallback", False))
    sequentialized_count = sum(1 for record in records if (record.metadata or {}).get("graph_json_sequentialized", False))
    causal_dag_count = sum(1 for record in records if (record.metadata or {}).get("decoding_order_kind") == "causal_dag")
    random_ar_count = sum(1 for record in records if (record.metadata or {}).get("decoding_order_kind") == "random_autoregressive")
    parameter_golf_count = sum(
        1 for record in records if (record.metadata or {}).get("source") == "parameter_golf_bin" or (record.metadata or {}).get("hybrid_source") == "openai_parameter_golf"
    )
    sequence_node_count = sum(
        1
        for record in records
        for node in (record.graph_json or {}).get("nodes", [])
        if str(node.get("type", "")) in {"sequence_document", "sequence_chunk"}
    )
    report["data"].update(
        {
            "split": split,
            "sample_records": len(records),
            "graph_json_fallback_records": fallback_count,
            "graph_json_fallback_rate": fallback_count / max(len(records), 1),
            "graph_json_sequentialized_records": sequentialized_count,
            "graph_json_sequentialized_rate": sequentialized_count / max(len(records), 1),
            "causal_dag_ar_records": causal_dag_count,
            "causal_dag_ar_rate": causal_dag_count / max(len(records), 1),
            "random_graph_ar_records": random_ar_count,
            "random_graph_ar_rate": random_ar_count / max(len(records), 1),
            "parameter_golf_source_records": parameter_golf_count,
            "parameter_golf_source_rate": parameter_golf_count / max(len(records), 1),
            "sequence_graph_nodes": sequence_node_count,
            "batch_graph_tokens": graph_batch.graph_token_counts.tolist(),
            "batch_node_tokens": graph_batch.node_counts.tolist(),
            "batch_edge_tokens": graph_batch.edge_counts.tolist(),
            "token_feature_shape": list(graph_batch.token_features.shape),
        }
    )
    add_gate(gates, "sample_records_loaded", len(records) > 0, f"{len(records)} {split} records")
    add_gate(gates, "graph_tokens_nonempty", bool(graph_batch.graph_token_counts.min().item() > 0), str(graph_batch.graph_token_counts.tolist()))
    add_gate(gates, "graph_json_fallback_rate_low", report["data"]["graph_json_fallback_rate"] <= float(cfg.get("max_fallback_rate", 0.05)), f"{report['data']['graph_json_fallback_rate']:.4f}")
    add_gate(gates, "sequential_text_graphs_present", sequence_node_count > 0, f"{sequence_node_count} sequence nodes")

    paper_tex = ROOT / "assets" / "tropicalgt_neurips_research_paper.tex"
    paper_pdf = paper_tex.with_suffix(".pdf")
    report["paper"] = {
        "tex": str(paper_tex),
        "pdf": str(paper_pdf),
        "tex_exists": paper_tex.exists(),
        "pdf_exists": paper_pdf.exists(),
        "tex_bytes": paper_tex.stat().st_size if paper_tex.exists() else 0,
        "pdf_bytes": paper_pdf.stat().st_size if paper_pdf.exists() else 0,
    }
    add_gate(gates, "paper_sources_present", paper_tex.exists() and paper_pdf.exists(), "TeX and PDF assets")

    if train_dry_run:
        add_train_dry_run_section(report, gates, cfg, records, tokenizer, require_cuda=require_cuda)

    if checkpoint_path is None:
        add_gate(gates, "checkpoint_reload", not require_checkpoint, "skipped: no checkpoint provided")
    else:
        add_checkpoint_sections(
            report,
            gates,
            cfg,
            checkpoint_path,
            records,
            tokenizer,
            split,
            details_limit,
            trace_limit,
            scale_depth,
            scale_width,
            scale_branch_factor,
            audit_level or str(cfg.get("audit_level", "none")),
            audit_ph_backend or str(cfg.get("ph_backend", "auto")),
            int(audit_max_simplices or cfg.get("audit_max_simplices", 512)),
            render_visualizations,
            out_dir,
        )

    failed = [gate["name"] for gate in gates if gate["status"] == "fail"]
    report["failed_gates"] = failed
    report["status"] = "ready" if not failed else "blocked"
    return report


def add_train_dry_run_section(
    report: dict[str, Any],
    gates: list[dict[str, Any]],
    cfg: dict[str, Any],
    records: list[GraphRecord],
    tokenizer: TokenGTTokenizer,
    *,
    require_cuda: bool,
) -> None:
    if not records:
        add_gate(gates, "train_dry_run_records", False, "no sampled records")
        return
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    add_gate(gates, "train_dry_run_device", (device.type == "cuda") or not require_cuda, str(device))
    try:
        model = build_model(cfg).to(device)
        model.train()
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("lr", 3e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )
        batch_size = max(1, min(int(cfg.get("batch_size", 2)), len(records)))
        x, y, graph_batch, batch_records = collate_records(
            records[:batch_size],
            int(cfg.get("seq_len", 128)),
            tokenizer,
            graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
            ar_seed=int(cfg.get("seed", 1729)),
        )
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x, graph_batch, y)
        loss = out["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        opt.step()
        compression = batch_bpb_metrics(out["nll"].detach().cpu(), y.detach().cpu(), graph_batch, batch_records, float(cfg.get("graph_bpb_side_weight", 1.0)))
        dry = {
            "device": str(device),
            "batch_size": batch_size,
            "seq_len": int(cfg.get("seq_len", 128)),
            "loss": float(loss.detach().cpu()),
            "nll": float(out["nll"].detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "graph_tokens": int(graph_batch.graph_token_counts.sum().item()),
            "node_tokens": int(graph_batch.node_counts.sum().item()),
            "edge_tokens": int(graph_batch.edge_counts.sum().item()),
            "compression": compression,
        }
        if torch.cuda.is_available():
            dry["gpu_mem_mb"] = float(torch.cuda.max_memory_allocated() / 1e6)
        report["train_dry_run"] = dry
        add_gate(gates, "train_dry_run_forward_backward", True, f"loss {dry['loss']:.4f}")
        add_gate(gates, "train_dry_run_finite_loss", _finite_number(dry["loss"]) and _finite_number(dry["nll"]), f"nll {dry['nll']:.4f}")
        add_gate(gates, "train_dry_run_finite_grad", _finite_number(dry["grad_norm"]), f"grad {dry['grad_norm']:.4f}")
        add_gate(gates, "train_dry_run_reports_bpb", _finite_number(compression.get("bpb")), f"BPB {compression.get('bpb')}")
        add_gate(gates, "train_dry_run_reports_graph_bpb", _finite_number(compression.get("graph_bpb")), f"graph BPB {compression.get('graph_bpb')}")
    except Exception as exc:
        report["train_dry_run"] = {"error": f"{type(exc).__name__}: {exc}"}
        add_gate(gates, "train_dry_run_forward_backward", False, report["train_dry_run"]["error"])


def add_checkpoint_sections(
    report: dict[str, Any],
    gates: list[dict[str, Any]],
    cfg: dict[str, Any],
    checkpoint_path: Path,
    records: list[GraphRecord],
    tokenizer: TokenGTTokenizer,
    split: str,
    details_limit: int,
    trace_limit: int,
    scale_depth: int,
    scale_width: int,
    scale_branch_factor: int,
    audit_level: str,
    audit_ph_backend: str,
    audit_max_simplices: int,
    render_visualizations: bool,
    out_dir: Path,
) -> None:
    add_gate(gates, "checkpoint_exists", checkpoint_path.exists(), str(checkpoint_path))
    if not checkpoint_path.exists():
        return
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model, obj = load_checkpoint(checkpoint_path, device)
    expected_metrics = {
        "nll",
        "certificate_loss",
        "certificate_agreement",
        "gflownet_tb",
        "graphcg_loss",
        "wall_hit_rate",
        "examples_per_sec",
        "tokens_per_sec",
        "bpb",
        "graph_bpb",
        "graph_sideinfo_bpb",
    }
    metrics = obj.get("metrics", {})
    report["checkpoint"] = {
        "path": str(checkpoint_path),
        "step": int(obj.get("step", metrics.get("step", 0))),
        "run_name": obj.get("run_name", ""),
        "metric_keys": sorted(metrics.keys()),
        "history_len": len(obj.get("history", [])),
        "missing_expected_metrics": sorted(expected_metrics.difference(metrics.keys())),
    }
    add_gate(gates, "checkpoint_reload", True, f"step {report['checkpoint']['step']}")
    add_gate(gates, "checkpoint_has_metrics", not report["checkpoint"]["missing_expected_metrics"], ",".join(report["checkpoint"]["missing_expected_metrics"]))

    eval_ds = records
    eval_report = evaluate_model(
        model,
        eval_ds,
        tokenizer,
        int(cfg.get("seq_len", 128)),
        int(cfg.get("batch_size", 2)),
        device,
        details_limit=details_limit,
        graph_bpb_side_weight=float(cfg.get("graph_bpb_side_weight", 1.0)),
        graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
        ar_seed=int(cfg.get("seed", 1729)),
    )
    report["eval"] = eval_report
    add_gate(gates, "eval_finite_nll", _finite_number(eval_report.get("nll")), f"NLL {eval_report.get('nll')}")
    add_gate(gates, "eval_reports_bpb", _finite_number(eval_report.get("bpb")), f"BPB {eval_report.get('bpb')}")
    add_gate(gates, "eval_reports_graph_bpb", _finite_number(eval_report.get("graph_bpb")), f"graph BPB {eval_report.get('graph_bpb')}")
    add_gate(gates, "eval_records_diagnostic", len(eval_report.get("records", [])) >= min(details_limit, len(records)), f"{len(eval_report.get('records', []))} records")

    prompt = "Represent the proof as a graph of tropical reasoning steps."
    rec = GraphRecord.from_mapping(
        {
            "record_id": "readiness-inference",
            "text": prompt,
            "question": prompt,
        }
    )
    x, y, graph_batch, _ = collate_records(
        [rec],
        int(cfg.get("seq_len", 128)),
        tokenizer,
        graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
        ar_seed=int(cfg.get("seed", 1729)),
    )
    with torch.no_grad():
        out = model(x.to(device), graph_batch, y.to(device))
    out_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in out.items()}
    diagnostics = record_diagnostics(
        [rec],
        graph_batch,
        out_cpu,
        tokenizer,
        target_ids=y,
        max_records=1,
        max_trace_tokens=trace_limit,
        audit_level=audit_level,
        ph_backend=audit_ph_backend,
        audit_max_simplices=audit_max_simplices,
    )[0]
    scaling = run_inference_scaling(
        model,
        rec,
        tokenizer,
        int(cfg.get("seq_len", 128)),
        device,
        depth=max(0, int(scale_depth)),
        width=max(1, int(scale_width)),
        branch_factor=max(1, int(scale_branch_factor)),
        trace_limit=trace_limit,
        audit_level=audit_level,
        ph_backend=audit_ph_backend,
        audit_max_simplices=audit_max_simplices,
        require_complete_reasoning_steps=bool(render_visualizations and require_complete_reasoning_steps),
    ) if scale_depth > 0 else {"enabled": False}
    compression = batch_bpb_metrics(out_cpu["nll"], y, graph_batch, [rec], float(cfg.get("graph_bpb_side_weight", 1.0)))
    report["inference"] = {
        "margin_mean": float(out["margin_mean"].detach().cpu()),
        "compression": compression,
        "gflownet": gflownet_diagnostics(model, out["graph_state"]),
        "graphcg": graphcg_diagnostics(model, out["graph_state"]),
        "graph_token_trace_len": len(diagnostics["graph_token_trace"]["tokens"]),
        "filtered_simplicial_summary": diagnostics["filtered_simplicial_object"]["summary"],
        "topological_algebra_summary": _topological_summary(diagnostics.get("topological_algebra")),
        "scaling": {
            "enabled": bool(scaling.get("enabled", False)),
            "evaluated_candidates": scaling.get("evaluated_candidates", 0),
            "best_path": scaling.get("best", {}).get("path", []),
            "best_nll": scaling.get("best", {}).get("nll"),
            "trajectory_has_algebra": isinstance(scaling.get("trajectory_topological_algebra"), dict),
        },
    }
    report["inference"]["scaling"]["trajectory_topological_summary"] = _topological_summary(scaling.get("trajectory_topological_algebra"))
    add_gate(gates, "inference_trace_nonempty", report["inference"]["graph_token_trace_len"] > 0, str(report["inference"]["graph_token_trace_len"]))
    add_gate(gates, "inference_reports_bpb", _finite_number(compression.get("bpb")), f"BPB {compression.get('bpb')}")
    add_gate(gates, "inference_reports_graph_bpb", _finite_number(compression.get("graph_bpb")), f"graph BPB {compression.get('graph_bpb')}")
    add_gate(gates, "inference_scaling_candidates", report["inference"]["scaling"]["evaluated_candidates"] > 0 if scale_depth > 0 else True, str(report["inference"]["scaling"]["evaluated_candidates"]))
    if (audit_level or "none").lower() != "none":
        add_gate(gates, "inference_topological_audit", isinstance(diagnostics.get("topological_algebra"), dict), audit_level)
        add_gate(gates, "scaling_topological_audit", bool(report["inference"]["scaling"]["trajectory_has_algebra"]) if scale_depth > 0 else True, audit_level)

    if render_visualizations:
        paths = write_reasoning_visualizations(
            model,
            eval_ds,
            tokenizer,
            int(cfg.get("seq_len", 128)),
            device,
            out_dir,
            limit=min(len(records), 4),
            audit_level=audit_level,
            ph_backend=audit_ph_backend,
            audit_max_simplices=audit_max_simplices,
        )
        audit_paths = write_inference_audit_artifacts(
            {
                "topological_algebra": diagnostics.get("topological_algebra"),
                "graph_token_trace": diagnostics.get("graph_token_trace", {}),
                "inference_scaling": scaling,
            },
            out_dir / "readiness_inference_audit",
            render_html=True,
        )
        paths.update({f"inference_{key}": value for key, value in audit_paths.items()})
        report["visualizations"] = paths
        add_gate(gates, "visualizations_written", all(Path(path).exists() for path in paths.values()), json.dumps(paths))


def environment_report() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "plotly", "sklearn", "tqdm", "wandb", "pyarrow", "networkx", "sympy", "gudhi", "ripser", "persim", "multipers"):
        packages[name] = {"available": importlib.util.find_spec(name) is not None}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "packages": packages,
    }


def tooling_report() -> dict[str, Any]:
    core = {
        "train": ROOT / "scripts" / "train_tropicalgt_i.py",
        "eval": ROOT / "scripts" / "eval_tropicalgt_i.py",
        "infer": ROOT / "scripts" / "infer_tropicalgt_i.py",
        "validate": ROOT / "scripts" / "validate_tropicalgt_i.py",
        "render": ROOT / "scripts" / "render_reasoning_visualizations.py",
        "readiness": ROOT / "scripts" / "audit_tropicalgt_i_readiness.py",
    }
    ablation = {
        "analyze_bpb_ablations": ROOT / "scripts" / "analyze_bpb_ablations.py",
        "run_bpb_ablation_grid": ROOT / "scripts" / "run_bpb_ablation_grid.py",
    }
    return {
        "core_scripts": {name: {"path": str(path), "exists": path.exists()} for name, path in core.items()},
        "ablation_scripts": {name: {"path": str(path), "exists": path.exists()} for name, path in ablation.items()},
    }


def add_gate(gates: list[dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    gates.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(f"| {gate['name']} | {gate['status']} | {gate.get('detail', '')} |" for gate in report["gates"])
    return "\n".join(
        [
            "# TropicalGT-I Readiness Audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Generated: `{report['generated_at']}`",
            f"- Config: `{report['config_path']}`",
            f"- Checkpoint: `{report.get('checkpoint_path', '')}`",
            "",
            "## Gates",
            "",
            "| Gate | Status | Detail |",
            "|---|---:|---|",
            rows,
            "",
            "## Summary",
            "",
            f"- Data split/sample: `{report.get('data', {}).get('split', '')}` / `{report.get('data', {}).get('sample_records', 0)}` records",
            f"- Fallback rate: `{report.get('data', {}).get('graph_json_fallback_rate', 0.0)}`",
            f"- Eval NLL: `{report.get('eval', {}).get('nll', 'n/a')}`",
            f"- Eval BPB: `{report.get('eval', {}).get('bpb', 'n/a')}`",
            f"- Eval graph-BPB: `{report.get('eval', {}).get('graph_bpb', 'n/a')}`",
            f"- Train dry-run BPB: `{report.get('train_dry_run', {}).get('compression', {}).get('bpb', 'n/a')}`",
            f"- Train dry-run graph-BPB: `{report.get('train_dry_run', {}).get('compression', {}).get('graph_bpb', 'n/a')}`",
            f"- Inference scaling candidates: `{report.get('inference', {}).get('scaling', {}).get('evaluated_candidates', 'n/a')}`",
            f"- Failed gates: `{', '.join(report.get('failed_gates', [])) or 'none'}`",
            "",
        ]
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def _topological_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"enabled": False}
    return {
        "enabled": bool(value.get("enabled", True)),
        "betti": value.get("chain_complex", {}).get("homology", {}).get("betti"),
        "multiparameter_num_parameters": value.get("multiparameter_persistence", {}).get("num_parameters"),
        "multiparameter_grid_points": len(value.get("multiparameter_persistence", {}).get("fiber_rank_profile", [])),
        "external_backends": value.get("external_backends", {}),
    }


if __name__ == "__main__":
    main()
