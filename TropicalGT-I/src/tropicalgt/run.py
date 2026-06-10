from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .algebra import compute_topological_algebra_report, summarize_algebra_reports
from .data import ChunkShuffleSampler, ParquetGraphDataset, encode_bytes, make_dataset, parquet_manifest
from .diagnostics import record_diagnostics
from .memory import AnalogicalMemoryBank, memory_records_from_scaling_report
from .metrics import aggregate_bpb_metrics, batch_bpb_metrics, explicit_graph_json_bytes, graph_token_structural_bytes
from .model import TropicalGTConfig, TropicalGTModel
from .scaling import run_inference_scaling
from .simplicial import build_filtered_simplicial_object
from .tokenizer import TokenGTTokenizer
from .visualization import write_graphcg_training_visualizations, write_inference_audit_artifacts, write_metric_visualizations, write_reasoning_visualizations


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_keys(path: str | Path = "keys.txt") -> dict[str, str]:
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def setup_wandb(cfg: dict[str, Any], run_name: str):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    keys = load_keys(cfg.get("keys_path", "keys.txt"))
    if keys.get("wandb") and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = keys["wandb"]
    if wandb_cfg.get("mode"):
        os.environ["WANDB_MODE"] = wandb_cfg["mode"]
    import wandb
    return wandb.init(project=wandb_cfg.get("project", "TropicalGT-I"), entity=wandb_cfg.get("entity"), name=run_name, config=cfg)


def collate_records(records, seq_len: int, tokenizer: TokenGTTokenizer):
    xs, ys = zip(*(encode_bytes(r.text, seq_len) for r in records))
    return torch.stack(xs), torch.stack(ys), tokenizer.batch_encode(records), list(records)


def build_model(cfg: dict[str, Any]) -> TropicalGTModel:
    return TropicalGTModel(TropicalGTConfig(**cfg.get("model", {})))


def train(config_path: str | Path, resume_from: str | Path | None = None, max_steps_override: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 1729))
    _set_seed(seed)
    if resume_from is None:
        resume_from = cfg.get("resume_from")
    root = cfg.get("data_root")
    require_data = bool(cfg.get("require_data", bool(root)))
    cache_shards = int(cfg.get("cache_shards", 2))
    train_ds = make_dataset(
        root,
        "train",
        limit=cfg.get("train_limit"),
        fixture_size=cfg.get("fixture_size", 8),
        require_data=require_data,
        cache_shards=cache_shards,
    )
    val_ds = make_dataset(
        root,
        "validation",
        limit=cfg.get("val_limit", cfg.get("train_limit", 4)),
        fixture_size=cfg.get("fixture_size", 8),
        require_data=require_data,
        cache_shards=cache_shards,
    )
    tokenizer = TokenGTTokenizer(**cfg.get("tokengt", {}))
    seq_len = int(cfg.get("seq_len", 128))
    batch_size = int(cfg.get("batch_size", 2))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    run_name = cfg.get("run_name", f"tropicalgt-i-{int(time.time())}")
    out_dir = Path(cfg.get("output_dir", "TropicalGT-I/outputs/smoke")); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.get("checkpoint_dir", "TropicalGT-I/checkpoints")); ckpt_dir.mkdir(parents=True, exist_ok=True)
    sampler = None
    loader_shuffle = bool(cfg.get("shuffle", True))
    sampler_report: dict[str, Any] = {"kind": "torch_shuffle" if loader_shuffle else "sequential"}
    if bool(cfg.get("chunk_shuffle", False)):
        if isinstance(train_ds, ParquetGraphDataset):
            sampler = ChunkShuffleSampler(
                train_ds,
                seed=int(cfg.get("chunk_shuffle_seed", 0)),
                shuffle_rows=bool(cfg.get("shuffle_rows_within_chunk", False)),
            )
            loader_shuffle = False
            sampler_report = {
                "kind": "chunk_shuffle",
                "seed": sampler.seed,
                "shuffle_rows_within_chunk": sampler.shuffle_rows,
                "chunks": len(train_ds.chunks),
                "rows": len(train_ds),
            }
        else:
            sampler_report = {
                "kind": "requested_chunk_shuffle_unavailable",
                "reason": type(train_ds).__name__,
                "fallback_shuffle": loader_shuffle,
            }
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=loader_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=lambda r: collate_records(r, seq_len, tokenizer),
    )
    max_steps = int(max_steps_override if max_steps_override is not None else cfg.get("max_steps", 5))
    metrics_last: dict[str, float] = {}
    history: list[dict[str, float]] = []
    audit_level = str(cfg.get("audit_level", "none"))
    audit_interval = int(cfg.get("algebra_audit_interval", 0))
    audit_limit = int(cfg.get("algebra_audit_limit", 2))
    audit_max_simplices = int(cfg.get("audit_max_simplices", 512))
    ph_backend = str(cfg.get("ph_backend", "auto"))
    graph_bpb_side_weight = float(cfg.get("graph_bpb_side_weight", 1.0))
    memory_bank_path = str(cfg.get("memory_bank_path", ""))
    memory_bank = AnalogicalMemoryBank(memory_bank_path, max_records=int(cfg.get("memory_max_records", 2048))) if memory_bank_path else None
    memory_records_added = 0
    step = 0
    loaded_step = 0
    resume_path = str(resume_from) if resume_from else ""
    if resume_from:
        resume_obj = torch.load(resume_from, map_location=device)
        model.load_state_dict(resume_obj["model"], strict=False)
        if "optimizer" in resume_obj:
            try:
                opt.load_state_dict(resume_obj["optimizer"])
            except ValueError:
                pass
        step = int(resume_obj.get("step", resume_obj.get("metrics", {}).get("step", 0)))
        loaded_step = step
        history = list(resume_obj.get("history", []))
        metrics_last = dict(resume_obj.get("metrics", {}))
        _restore_rng_state(resume_obj)
    wb = setup_wandb(cfg, run_name)
    model.train()
    pbar = tqdm(total=max_steps, initial=min(step, max_steps), desc="TropicalGT-I train", dynamic_ncols=True)
    checkpoint_every = int(cfg.get("checkpoint_every", 0))
    latest_ckpt_path = ckpt_dir / f"{run_name}.latest.pt"
    examples_seen = int(sum(int(row.get("batch_size", 0)) for row in history))
    tokens_seen_total = int(sum(int(row.get("tokens_seen", 0)) for row in history))
    graph_tokens_seen_total = int(sum(int(row.get("graph_tokens_seen", 0)) for row in history))
    epoch = int(cfg.get("sampler_epoch", 0))
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, y, graph_batch, _records in loader:
            step_started = time.perf_counter()
            step += 1
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, graph_batch, y)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            metrics_last = {k: float(v.detach().cpu()) for k, v in out.items() if torch.is_tensor(v) and v.ndim == 0}
            metrics_last["loss"] = float(out["loss"].detach().cpu())
            metrics_last["ppl"] = float(math.exp(min(metrics_last.get("nll", metrics_last["loss"]), 20)))
            metrics_last.update(batch_bpb_metrics(metrics_last.get("nll", metrics_last["loss"]), y, graph_batch, _records, graph_bpb_side_weight))
            metrics_last["bpb_proxy"] = metrics_last["bpb"]
            metrics_last["step"] = step
            elapsed = max(time.perf_counter() - step_started, 1e-9)
            token_count = int((y != 0).sum().item())
            example_count = len(_records)
            graph_token_count = int(graph_batch.graph_token_counts.sum().item())
            fallback_count = sum(1 for record in _records if (record.metadata or {}).get("graph_json_fallback", False))
            examples_seen += example_count
            tokens_seen_total += token_count
            graph_tokens_seen_total += graph_token_count
            metrics_last["step_time_s"] = elapsed
            metrics_last["examples_per_sec"] = example_count / elapsed
            metrics_last["tokens_per_sec"] = token_count / elapsed
            metrics_last["graph_tokens_per_sec"] = graph_token_count / elapsed
            metrics_last["tokens_seen"] = token_count
            metrics_last["tokens_seen_total"] = tokens_seen_total
            metrics_last["graph_tokens_seen"] = graph_token_count
            metrics_last["graph_tokens_seen_total"] = graph_tokens_seen_total
            metrics_last["examples_seen"] = examples_seen
            metrics_last["batch_size"] = example_count
            metrics_last["optimizer_lr"] = float(opt.param_groups[0].get("lr", 0.0))
            metrics_last["grad_norm"] = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            metrics_last["chunk_shuffle_enabled"] = 1.0 if sampler_report.get("kind") == "chunk_shuffle" else 0.0
            metrics_last["sampler_chunks"] = float(sampler_report.get("chunks", 0))
            metrics_last["shuffle_rows_within_chunk"] = 1.0 if sampler_report.get("shuffle_rows_within_chunk") else 0.0
            metrics_last["graph_json_fallback_rate"] = fallback_count / max(example_count, 1)
            metrics_last["graph_json_sequentialized_rate"] = sum(
                1 for record in _records if (record.metadata or {}).get("graph_json_sequentialized", False)
            ) / max(example_count, 1)
            metrics_last["graph_token_node_edge_ratio"] = (
                float(graph_batch.node_counts.float().sum() / graph_batch.edge_counts.float().sum().clamp_min(1.0))
            )
            if audit_level.lower() != "none" and audit_interval > 0 and step % audit_interval == 0:
                algebra_reports = [
                    compute_topological_algebra_report(
                        build_filtered_simplicial_object(record),
                        audit_level=audit_level,
                        ph_backend=ph_backend,
                        max_simplices=audit_max_simplices,
                    )
                    for record in _records[:audit_limit]
                ]
                metrics_last.update(summarize_algebra_reports(algebra_reports))
            if torch.cuda.is_available():
                metrics_last["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            history.append(dict(metrics_last))
            if wb:
                wb.log(metrics_last, step=step)
            pbar.set_postfix({"loss": f"{metrics_last['loss']:.3f}", "nll": f"{metrics_last.get('nll', 0):.3f}"})
            pbar.update(1)
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                _save_training_checkpoint(latest_ckpt_path, model, opt, cfg, metrics_last, history, step, run_name)
            if step >= max_steps:
                break
        epoch += 1
    pbar.close()
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    _save_training_checkpoint(ckpt_path, model, opt, cfg, metrics_last, history, step, run_name)
    eval_report = evaluate_model(model, val_ds, tokenizer, seq_len, batch_size, device, graph_bpb_side_weight=graph_bpb_side_weight)
    metrics_last.update({f"eval_{k}": v for k, v in eval_report.items() if isinstance(v, (int, float))})
    vis_paths = write_reasoning_visualizations(
        model,
        val_ds,
        tokenizer,
        seq_len,
        device,
        out_dir,
        limit=int(cfg.get("viz_limit", 8)),
        audit_level=audit_level if bool(cfg.get("viz_topological_algebra", False)) else "none",
        ph_backend=ph_backend,
        audit_max_simplices=audit_max_simplices,
    )
    vis_paths.update(write_metric_visualizations(history, out_dir))
    vis_paths.update(write_graphcg_training_visualizations(model, out_dir))
    if bool(cfg.get("viz_got_scaling", False)) and len(val_ds) > 0:
        scaling = run_inference_scaling(
            model,
            val_ds[0],
            tokenizer,
            seq_len,
            device,
            depth=int(cfg.get("viz_scale_depth", 1)),
            width=int(cfg.get("viz_scale_width", 3)),
            branch_factor=int(cfg.get("viz_scale_branch_factor", 2)),
            trace_limit=int(cfg.get("viz_trace_limit", 24)),
            audit_level=audit_level,
            ph_backend=ph_backend,
            audit_max_simplices=audit_max_simplices,
        )
        if memory_bank is not None:
            records = memory_records_from_scaling_report(
                scaling,
                source=f"train:{run_name}:step{step}",
                min_score=cfg.get("memory_min_score"),
                max_records=int(cfg.get("memory_records_per_audit", 8)),
            )
            memory_bank.extend(records)
            memory_bank.save()
            memory_records_added += len(records)
            metrics_last["analogical_memory_records_added"] = float(memory_records_added)
            metrics_last["analogical_memory_bank_size"] = float(len(memory_bank.records))
        vis_paths.update(
            {
                f"train_{key}": value
                for key, value in write_inference_audit_artifacts(
                    {"inference_scaling": scaling, "topological_algebra": scaling.get("best", {}).get("topological_algebra")},
                    out_dir / "train_got_audit",
                    render_html=True,
                ).items()
            }
        )
    report = {
        "checkpoint": str(ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path) if latest_ckpt_path.exists() else "",
        "resume_from": resume_path,
        "resumed": bool(resume_path),
        "start_step": loaded_step,
        "final_step": step,
        "metrics": metrics_last,
        "history": history,
        "eval": eval_report,
        "visualizations": vis_paths,
        "dataset_manifest": parquet_manifest(root, ("train", "validation"), include_shards=False) if root else {},
        "sampler": sampler_report,
        "device": str(device),
        "seed": seed,
    }
    if memory_bank is not None:
        report["analogical_memory"] = {
            "path": str(memory_bank.path),
            "records": len(memory_bank.records),
            "records_added": memory_records_added,
        }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if wb:
        html_limit = int(cfg.get("wandb_html_artifact_limit", 5))
        uploaded_html = 0
        for key, path in _wandb_html_items(vis_paths):
            if uploaded_html >= html_limit:
                break
            try:
                import wandb
                html_path = Path(path)
                max_bytes = int(cfg.get("wandb_html_max_bytes", 5_000_000))
                if html_path.suffix.lower() != ".html" or html_path.stat().st_size > max_bytes:
                    continue
                wb.log({key: wandb.Html(html_path.read_text(encoding="utf-8"))})
                uploaded_html += 1
            except Exception:
                pass
        wb.finish()
    return report


def _wandb_html_items(paths: dict[str, str]) -> list[tuple[str, str]]:
    preferred = [
        "metrics",
        "pca_3d",
        "pca_nll",
        "graphcg_direction_gram",
        "graphcg_direction_pca",
        "train_dashboard",
        "train_got_trajectory_3d",
        "train_graphcg_direction_cosines",
        "train_tropical_support_heatmap",
    ]
    order = {name: idx for idx, name in enumerate(preferred)}
    return sorted(paths.items(), key=lambda item: (order.get(item[0], len(order)), item[0]))


def evaluate_model(
    model: TropicalGTModel,
    dataset,
    tokenizer: TokenGTTokenizer,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    details_limit: int = 0,
    graph_bpb_side_weight: float = 1.0,
    audit_level: str = "none",
    ph_backend: str = "auto",
    audit_max_simplices: int = 1024,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda r: collate_records(r, seq_len, tokenizer))
    total_loss = 0.0; total_tokens = 0; batches = 0
    details: list[dict[str, Any]] = []
    graph_token_total = 0
    graph_token_structural_bytes_total = 0
    explicit_graph_json_bytes_total = 0
    node_token_total = 0
    edge_token_total = 0
    fallback_total = 0
    model.eval()
    with torch.no_grad():
        for x, y, graph_batch, records in loader:
            x = x.to(device); y = y.to(device)
            out = model(x, graph_batch, y)
            tokens = int((y != 0).sum().item())
            total_loss += float(out["nll"].detach().cpu()) * max(tokens, 1)
            total_tokens += max(tokens, 1); batches += 1
            graph_token_total += int(graph_batch.graph_token_counts.sum().item())
            graph_token_structural_bytes_total += graph_token_structural_bytes(graph_batch)
            explicit_graph_json_bytes_total += sum(explicit_graph_json_bytes(record) for record in records)
            node_token_total += int(graph_batch.node_counts.sum().item())
            edge_token_total += int(graph_batch.edge_counts.sum().item())
            fallback_total += sum(1 for record in records if (record.metadata or {}).get("graph_json_fallback", False))
            if details_limit > 0 and len(details) < details_limit:
                needed = details_limit - len(details)
                details.extend(
                    record_diagnostics(
                        records,
                        graph_batch,
                        {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in out.items()},
                        tokenizer,
                        target_ids=y.detach().cpu(),
                        max_records=needed,
                        max_trace_tokens=16,
                        audit_level=audit_level,
                        ph_backend=ph_backend,
                        audit_max_simplices=audit_max_simplices,
                    )
                )
    nll = total_loss / max(total_tokens, 1)
    bpb = aggregate_bpb_metrics(
        total_loss / math.log(2.0),
        total_tokens,
        graph_token_structural_bytes_total,
        explicit_graph_json_bytes_total,
        graph_side_weight=graph_bpb_side_weight,
    )
    report: dict[str, Any] = {
        "nll": nll,
        "ppl": math.exp(min(nll, 20)),
        "bpb_proxy": bpb["bpb"],
        "batches": batches,
        "tokens": total_tokens,
        "graph_tokens": graph_token_total,
        "node_tokens": node_token_total,
        "edge_tokens": edge_token_total,
        "graph_json_fallback_records": fallback_total,
        "invalid_graph_rate": fallback_total / max(len(dataset), 1),
        **bpb,
    }
    if details:
        report["records"] = details
    return report


def load_checkpoint(path: str | Path, device: torch.device):
    obj = torch.load(path, map_location=device)
    model = build_model(obj["config"]).to(device)
    model.load_state_dict(obj["model"], strict=False)
    model.eval()
    return model, obj


def _save_training_checkpoint(
    path: Path,
    model: TropicalGTModel,
    opt: torch.optim.Optimizer,
    cfg: dict[str, Any],
    metrics: dict[str, float],
    history: list[dict[str, float]],
    step: int,
    run_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg,
            "metrics": metrics,
            "history": history,
            "step": int(step),
            "run_name": run_name,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "saved_at": time.time(),
        },
        path,
    )


def _restore_rng_state(obj: dict[str, Any]) -> None:
    rng_state = obj.get("rng_state")
    if isinstance(rng_state, torch.Tensor):
        torch.set_rng_state(rng_state.detach().cpu().to(torch.uint8))
    cuda_states = obj.get("cuda_rng_state_all")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        normalized = [state.detach().cpu().to(torch.uint8) for state in cuda_states if isinstance(state, torch.Tensor)]
        if normalized:
            torch.cuda.set_rng_state_all(normalized)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
