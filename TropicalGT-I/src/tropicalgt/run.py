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
from .data import (
    ChunkShuffleSampler,
    ParquetGraphDataset,
    dataset_budget_report,
    dataset_manifest,
    encode_record_bytes,
    make_dataset_from_config,
    validate_dataset_budget,
)
from .decoding import meet_in_middle_batch, meet_in_middle_config
from .diagnostics import describe_graph_tokens, record_diagnostics
from .memory import AnalogicalMemoryBank, memory_records_from_scaling_report, query_signature_from_report
from .metrics import aggregate_bpb_metrics, batch_bpb_metrics, explicit_graph_json_bytes, graph_token_structural_bytes
from .model import TropicalGTConfig, TropicalGTModel
from .scaling import run_inference_scaling
from .simplicial import build_embedding_radius_simplicial_object
from .tokenizer import TokenGTTokenizer
from .visualization import write_graphcg_training_visualizations, write_inference_audit_artifacts, write_metric_visualizations, write_reasoning_visualizations


WANDB_PRIORITY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "00_primary",
        (
            "eval_bpb",
            "bpb",
            "eval_graph_bpb",
            "graph_bpb",
            "loss",
            "nll",
            "gpu_mem_mb",
        ),
    ),
    (
        "01_losses",
        (
            "loss",
            "nll",
            "loss_regularizer_total",
            "loss_regularizer_ratio",
            "gflownet_tb",
            "graphcg_loss",
            "certificate_loss",
            "tropical_margin_loss",
            "tropical_entropy_loss",
            "loss_gflownet_weighted",
            "loss_graphcg_weighted",
            "loss_margin_weighted",
            "loss_entropy_weighted",
            "loss_certificate_weighted",
        ),
    ),
    (
        "02_bpb",
        (
            "bpb",
            "text_bpb",
            "graph_bpb",
            "graph_sideinfo_bpb",
            "graph_conditioned_bpb_no_side_cost",
            "eval_bpb",
            "eval_text_bpb",
            "eval_graph_bpb",
            "eval_graph_sideinfo_bpb",
            "eval_graph_conditioned_bpb_no_side_cost",
            "target_bytes",
            "nll_bits",
            "graph_token_structural_bytes",
            "explicit_graph_json_bytes",
        ),
    ),
    (
        "03_tropical",
        (
            "support_entropy",
            "support_soft_entropy",
            "support_unique_frac",
            "self_support_rate",
            "invalid_support_rate",
            "margin_mean",
            "margin_min",
            "margin_p05",
            "positive_margin_rate",
            "wall_hit_rate",
            "support_boundary_hit_rate",
            "sequence_tropical_tokens_mean",
            "sequence_tropical_stride",
            "sequence_tropical_margin_mean",
            "sequence_tropical_margin_min",
            "sequence_tropical_support_entropy",
            "sequence_tropical_weight",
            "certificate_agreement",
            "certificate_coverage",
            "certificate_edge_agreement",
            "certificate_node_agreement",
        ),
    ),
    (
        "04_gflownet",
        (
            "gflownet_tb",
            "gflownet_tb_residual_abs_mean",
            "gflownet_log_z",
            "gflownet_reward_mean",
            "gflownet_reward_std",
            "gflownet_terminal_diversity",
        ),
    ),
    (
        "05_graphcg",
        (
            "graphcg_loss",
            "graphcg_full_rank",
            "graphcg_active_full_rank",
            "graphcg_raw_full_rank",
            "graphcg_active_rank_fraction",
            "graphcg_raw_active_rank_fraction",
            "graphcg_active_full_rank_penalty",
            "graphcg_full_rank_penalty",
            "graphcg_raw_full_rank_penalty",
            "graphcg_full_rank_possible",
            "graphcg_effective_rank",
            "graphcg_numerical_rank",
            "graphcg_rank_target",
            "graphcg_raw_effective_rank",
            "graphcg_raw_numerical_rank",
            "graphcg_raw_singular_min",
            "graphcg_raw_singular_max",
            "graphcg_min_singular_value",
            "graphcg_max_singular_value",
            "graphcg_direction_effective_rank",
            "graphcg_direction_numerical_rank",
            "graphcg_direction_singular_min",
            "graphcg_direction_singular_max",
            "graphcg_direction_svd_condition_proxy",
        ),
    ),
    (
        "06_graph_data",
        (
            "graph_tokens_mean",
            "node_tokens_mean",
            "edge_tokens_mean",
            "node_edge_ratio",
            "graph_token_node_edge_ratio",
            "graph_json_fallback_rate",
            "graph_json_sequentialized_rate",
            "causal_dag_ar_rate",
            "random_graph_ar_rate",
            "parameter_golf_source_rate",
            "tokens_seen",
            "tokens_seen_total",
            "graph_tokens_seen",
            "graph_tokens_seen_total",
            "examples_seen",
        ),
    ),
    (
        "07_algebra_topology",
        (
            "algebra_audit_records",
            "algebra_betti0_mean",
            "algebra_betti1_mean",
            "algebra_euler_mean",
            "persistence_points_mean",
            "multipersistence_grid_points_mean",
            "free_resolution_terms_mean",
            "derived_signature_dim_mean",
        ),
    ),
    (
        "08_memory",
        (
            "analogical_memory_query_norm",
            "analogical_memory_records_added",
            "analogical_memory_bank_size",
        ),
    ),
    (
        "09_meet_in_middle",
        (
            "mim_enabled",
            "mim_shared_weight_reverse_pass",
            "mim_loss",
            "mim_reverse_nll",
            "mim_bidirectional_nll",
            "mim_agreement_loss",
            "mim_join_token_match_rate",
            "mim_true_meet_logprob_mean",
            "mim_candidate_count",
            "mim_agreement_weight",
            "mim_reverse_nll_weight",
            "eval_mim_enabled",
            "eval_mim_reverse_nll",
            "eval_mim_bidirectional_nll",
            "eval_mim_agreement_loss",
            "eval_mim_join_token_match_rate",
        ),
    ),
    (
        "10_system",
        (
            "gpu_mem_mb",
            "step_time_s",
            "examples_per_sec",
            "tokens_per_sec",
            "graph_tokens_per_sec",
            "batch_size",
        ),
    ),
    (
        "11_optimization",
        (
            "optimizer_lr",
            "grad_norm",
            "chunk_shuffle_enabled",
            "sampler_chunks",
            "shuffle_rows_within_chunk",
        ),
    ),
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_keys(path: str | Path = "keys.txt") -> dict[str, str]:
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        key = k.strip()
        value = v.strip()
        out[key] = value
        lowered = key.lower()
        out[lowered] = value
        if lowered in {"wandb_api_key", "wandb-key"}:
            out["wandb"] = value
        elif lowered in {"gh", "github", "github_token"}:
            out["github"] = value
        elif lowered in {"hf", "huggingface", "huggingface_token", "hf_token"}:
            out["huggingface"] = value
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
    run = wandb.init(project=wandb_cfg.get("project", "TropicalGT-I"), entity=wandb_cfg.get("entity"), name=run_name, config=cfg)
    configure_wandb_metrics(run)
    return run


def configure_wandb_metrics(run: Any) -> None:
    """Register prioritized metric namespaces for the W&B UI."""

    try:
        run.define_metric("step")
        for group, _keys in WANDB_PRIORITY_GROUPS:
            run.define_metric(f"{group}/*", step_metric="step")
        for metric in (
            "00_primary/eval_bpb",
            "00_primary/bpb",
            "00_primary/eval_graph_bpb",
            "00_primary/graph_bpb",
            "00_primary/loss",
            "00_primary/nll",
            "01_losses/loss_regularizer_total",
            "09_meet_in_middle/mim_loss",
            "10_system/gpu_mem_mb",
        ):
            run.define_metric(metric, step_metric="step")
    except Exception:
        pass


def organize_wandb_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return ordered, namespaced W&B metrics while preserving scalar values."""

    payload: dict[str, Any] = {}
    if "step" in metrics:
        payload["step"] = metrics["step"]
    used: set[str] = set()
    for group, keys in WANDB_PRIORITY_GROUPS:
        for key in keys:
            if key in metrics and _wandb_scalar(metrics[key]):
                payload[f"{group}/{key}"] = metrics[key]
                used.add(key)
    for key, value in metrics.items():
        if key in used or key == "step" or not _wandb_scalar(value):
            continue
        payload[f"{_wandb_fallback_group(key)}/{key}"] = value
    return payload


def collate_records(records, seq_len: int, tokenizer: TokenGTTokenizer, graph_autoregressive: bool = False, ar_seed: int = 0):
    xs, ys = zip(*(encode_record_bytes(r, seq_len, graph_autoregressive=graph_autoregressive, seed=ar_seed) for r in records))
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
    train_ds = make_dataset_from_config(cfg, "train")
    val_ds = make_dataset_from_config(cfg, "validation")
    tokenizer = TokenGTTokenizer(**cfg.get("tokengt", {}))
    seq_len = int(cfg.get("seq_len", 128))
    batch_size = int(cfg.get("batch_size", 2))
    config_max_steps = int(cfg.get("max_steps", 5))
    max_steps = int(max_steps_override if max_steps_override is not None else config_max_steps)
    train_data_budget = dataset_budget_report(train_ds, seq_len=seq_len, batch_size=batch_size, max_steps=config_max_steps)
    train_data_budget["effective_max_steps"] = max_steps
    train_data_budget["effective_training_token_slots"] = batch_size * max_steps * seq_len
    data_budget_errors = validate_dataset_budget(
        train_data_budget,
        min_available_token_slots=cfg.get("min_available_train_token_slots"),
        min_training_token_slots=cfg.get("min_training_token_slots"),
        required_sources=cfg.get("required_hybrid_sources", ()),
        source_requirements=cfg.get("hybrid_source_requirements", {}),
    )
    if data_budget_errors:
        detail = "\n".join(f"- {error}" for error in data_budget_errors)
        raise RuntimeError(f"Training data budget check failed:\n{detail}")
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
        collate_fn=lambda r: collate_records(
            r,
            seq_len,
            tokenizer,
            graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
            ar_seed=seed + step,
        ),
    )
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
    periodic_artifacts: list[dict[str, Any]] = []
    validation_every = int(cfg.get("validation_every_steps", cfg.get("eval_every_steps", 0)) or 0)
    visualization_every = int(cfg.get("visualization_every_steps", validation_every) or 0)
    periodic_eval_limit = int(cfg.get("periodic_eval_details_limit", cfg.get("eval_details_limit", 4)) or 0)
    periodic_viz_limit = int(cfg.get("periodic_viz_limit", cfg.get("viz_limit", 8)) or 0)
    periodic_audit_level = str(cfg.get("periodic_audit_level", cfg.get("audit_level", "none")))
    periodic_ph_backend = str(cfg.get("periodic_ph_backend", ph_backend))
    periodic_audit_max_simplices = int(cfg.get("periodic_audit_max_simplices", audit_max_simplices))
    mim_cfg = meet_in_middle_config(cfg.get("meet_in_middle"))
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
            train_loss = out["loss"]
            mim_report: dict[str, Any] | None = None
            if mim_cfg.enabled:
                mim_report = meet_in_middle_batch(
                    model,
                    list(_records),
                    tokenizer,
                    seq_len,
                    device,
                    graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
                    seed=seed + step,
                    config=mim_cfg,
                    forward_logits=out.get("logits"),
                    forward_nll=out.get("nll"),
                    require_grad=(mim_cfg.agreement_weight > 0.0 or mim_cfg.reverse_nll_weight > 0.0),
                )
                if torch.is_tensor(mim_report.get("loss")):
                    train_loss = train_loss + mim_report["loss"]
            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            metrics_last = {k: float(v.detach().cpu()) for k, v in out.items() if torch.is_tensor(v) and v.ndim == 0}
            metrics_last["loss"] = float(train_loss.detach().cpu())
            if mim_report is not None:
                metrics_last.update(
                    {
                        key: float(value)
                        for key, value in mim_report.get("metrics", {}).items()
                        if isinstance(value, (int, float))
                    }
                )
            else:
                metrics_last["mim_enabled"] = 0.0
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
            metrics_last["causal_dag_ar_rate"] = sum(
                1 for record in _records if (record.metadata or {}).get("decoding_order_kind") == "causal_dag"
            ) / max(example_count, 1)
            metrics_last["random_graph_ar_rate"] = sum(
                1 for record in _records if (record.metadata or {}).get("decoding_order_kind") == "random_autoregressive"
            ) / max(example_count, 1)
            metrics_last["parameter_golf_source_rate"] = sum(
                1 for record in _records if (record.metadata or {}).get("source") == "parameter_golf_bin" or (record.metadata or {}).get("hybrid_source") == "openai_parameter_golf"
            ) / max(example_count, 1)
            metrics_last["graph_autoregressive_decoding_enabled"] = 1.0 if cfg.get("graph_autoregressive_decoding", True) else 0.0
            metrics_last["graph_token_node_edge_ratio"] = (
                float(graph_batch.node_counts.float().sum() / graph_batch.edge_counts.float().sum().clamp_min(1.0))
            )
            if audit_level.lower() != "none" and audit_interval > 0 and step % audit_interval == 0:
                graph_batch_cpu = graph_batch.to("cpu")
                graph_token_embeddings = out.get("graph_token_embeddings")
                graph_token_support_probabilities = out.get("graph_token_support_probabilities")
                embeddings_cpu = graph_token_embeddings.detach().cpu() if torch.is_tensor(graph_token_embeddings) else None
                probabilities_cpu = (
                    graph_token_support_probabilities.detach().cpu()
                    if torch.is_tensor(graph_token_support_probabilities)
                    else None
                )
                algebra_reports = []
                for record_idx, record in enumerate(_records[:audit_limit]):
                    graph_token_count = int(graph_batch_cpu.graph_token_counts[record_idx].item())
                    descriptors = describe_graph_tokens(record, tokenizer)[:graph_token_count]
                    embeddings = (
                        embeddings_cpu[record_idx, :graph_token_count]
                        if embeddings_cpu is not None and embeddings_cpu.ndim >= 3
                        else []
                    )
                    probabilities = (
                        probabilities_cpu[record_idx, :graph_token_count, :graph_token_count]
                        if probabilities_cpu is not None and probabilities_cpu.ndim >= 3
                        else None
                    )
                    filtered_object = build_embedding_radius_simplicial_object(
                        record,
                        descriptors,
                        embeddings,
                        token_probabilities=probabilities,
                        metric="jensen_shannon",
                    )
                    if filtered_object.get("available") is False:
                        continue
                    algebra_reports.append(
                        compute_topological_algebra_report(
                            filtered_object,
                            audit_level=audit_level,
                            ph_backend=ph_backend,
                            max_simplices=audit_max_simplices,
                        )
                    )
                metrics_last.update(summarize_algebra_reports(algebra_reports))
            if torch.cuda.is_available():
                metrics_last["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            history.append(dict(metrics_last))
            if wb:
                wb.log(organize_wandb_metrics(metrics_last), step=step)
            pbar.set_postfix({"loss": f"{metrics_last['loss']:.3f}", "nll": f"{metrics_last.get('nll', 0):.3f}"})
            pbar.update(1)
            should_validate = validation_every > 0 and step % validation_every == 0
            should_visualize = visualization_every > 0 and step % visualization_every == 0
            if should_validate or should_visualize:
                periodic_report = _run_periodic_validation_round(
                    model=model,
                    val_ds=val_ds,
                    tokenizer=tokenizer,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                    out_dir=out_dir,
                    cfg=cfg,
                    history=history,
                    step=step,
                    seed=seed,
                    graph_bpb_side_weight=graph_bpb_side_weight,
                    graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
                    run_name=run_name,
                    memory_bank=memory_bank,
                    memory_records_added=memory_records_added,
                    render_visualizations=should_visualize,
                    details_limit=periodic_eval_limit,
                    viz_limit=periodic_viz_limit,
                    audit_level=periodic_audit_level,
                    ph_backend=periodic_ph_backend,
                    audit_max_simplices=periodic_audit_max_simplices,
                )
                periodic_artifacts.append(periodic_report)
                memory_records_added = int(periodic_report.get("memory_records_added_total", memory_records_added))
                periodic_metrics = {
                    key: value
                    for key, value in periodic_report.get("metrics", {}).items()
                    if isinstance(value, (int, float))
                }
                metrics_last.update(periodic_metrics)
                history[-1].update(periodic_metrics)
                if wb:
                    wb.log(organize_wandb_metrics(metrics_last), step=step)
                    _log_wandb_html_artifacts(
                        wb,
                        periodic_report.get("visualizations", {}),
                        cfg,
                        prefix=f"periodic_eval/step_{step:08d}",
                        step=step,
                    )
                model.train()
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                _save_training_checkpoint(latest_ckpt_path, model, opt, cfg, metrics_last, history, step, run_name)
            if step >= max_steps:
                break
        epoch += 1
    pbar.close()
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    _save_training_checkpoint(ckpt_path, model, opt, cfg, metrics_last, history, step, run_name)
    eval_report = evaluate_model(
        model,
        val_ds,
        tokenizer,
        seq_len,
        batch_size,
        device,
        graph_bpb_side_weight=graph_bpb_side_weight,
        graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
        ar_seed=seed,
        meet_in_middle=cfg.get("meet_in_middle"),
    )
    metrics_last.update({f"eval_{k}": v for k, v in eval_report.items() if isinstance(v, (int, float))})
    if wb:
        wb.log(organize_wandb_metrics(metrics_last), step=step)
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
        audit_seed_record = _select_got_audit_records(
            val_ds,
            count=1,
            pool=int(cfg.get("viz_got_record_pool", cfg.get("viz_limit", 8)) or 1),
            seed=seed + step,
        )[0][1]
        scaling = run_inference_scaling(
            model,
            audit_seed_record,
            tokenizer,
            seq_len,
            device,
            depth=int(cfg.get("viz_scale_depth", 3)),
            width=int(cfg.get("viz_scale_width", 4)),
            branch_factor=int(cfg.get("viz_scale_branch_factor", 3)),
            trace_limit=int(cfg.get("viz_trace_limit", 24)),
            audit_level=audit_level,
            ph_backend=ph_backend,
            audit_max_simplices=audit_max_simplices,
            allow_stop=bool(cfg.get("viz_scale_allow_stop", False)),
            diverse_actions=bool(cfg.get("viz_scale_diverse_actions", True)),
            stochastic_actions=bool(cfg.get("viz_scale_stochastic_actions", False)),
            sampling_temperature=float(cfg.get("viz_scale_sampling_temperature", 1.0)),
            sampling_exploration=float(cfg.get("viz_scale_sampling_exploration", 0.0)),
            sampling_seed=int(cfg.get("viz_scale_sampling_seed", seed + step)),
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
        "periodic_artifacts": periodic_artifacts,
        "dataset_manifest": {
            "train": dataset_manifest(train_ds, root, ("train", "validation")),
            "validation": dataset_manifest(val_ds, root, ("train", "validation")),
        },
        "data_budget": train_data_budget,
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
        _log_wandb_html_artifacts(wb, vis_paths, cfg, prefix="final_artifacts", step=step)
        wb.finish()
    return report


def _run_periodic_validation_round(
    *,
    model: TropicalGTModel,
    val_ds,
    tokenizer: TokenGTTokenizer,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    cfg: dict[str, Any],
    history: list[dict[str, float]],
    step: int,
    seed: int,
    graph_bpb_side_weight: float,
    graph_autoregressive: bool,
    run_name: str,
    memory_bank: AnalogicalMemoryBank | None,
    memory_records_added: int,
    render_visualizations: bool,
    details_limit: int,
    viz_limit: int,
    audit_level: str,
    ph_backend: str,
    audit_max_simplices: int,
) -> dict[str, Any]:
    step_dir = out_dir / "periodic" / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    eval_report = evaluate_model(
        model,
        val_ds,
        tokenizer,
        seq_len,
        batch_size,
        device,
        details_limit=details_limit,
        graph_bpb_side_weight=graph_bpb_side_weight,
        audit_level=audit_level,
        ph_backend=ph_backend,
        audit_max_simplices=audit_max_simplices,
        graph_autoregressive=graph_autoregressive,
        ar_seed=seed + step,
        meet_in_middle=cfg.get("meet_in_middle"),
    )
    eval_path = step_dir / "validation_report.json"
    eval_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    metrics = {f"eval_{key}": value for key, value in eval_report.items() if isinstance(value, (int, float))}
    vis_paths: dict[str, str] = {}
    memory_added_this_round = 0
    if render_visualizations:
        vis_paths.update(
            {
                f"reasoning_{key}": value
                for key, value in write_reasoning_visualizations(
                    model,
                    val_ds,
                    tokenizer,
                    seq_len,
                    device,
                    step_dir / "reasoning",
                    limit=viz_limit,
                    audit_level=audit_level,
                    ph_backend=ph_backend,
                    audit_max_simplices=audit_max_simplices,
                ).items()
            }
        )
        vis_paths.update({f"metrics_{key}": value for key, value in write_metric_visualizations(history, step_dir / "metrics").items()})
        vis_paths.update({f"graphcg_{key}": value for key, value in write_graphcg_training_visualizations(model, step_dir / "graphcg").items()})
        if bool(cfg.get("periodic_viz_got_scaling", cfg.get("viz_got_scaling", False))) and len(val_ds) > 0:
            audit_records = _select_got_audit_records(
                val_ds,
                count=int(cfg.get("periodic_viz_got_examples", 3)),
                pool=int(cfg.get("periodic_viz_got_record_pool", cfg.get("periodic_viz_limit", 16)) or 1),
                seed=seed + step,
            )
            for example_idx, (record_idx, audit_record) in enumerate(audit_records):
                scaling = run_inference_scaling(
                    model,
                    audit_record,
                    tokenizer,
                    seq_len,
                    device,
                    depth=int(cfg.get("periodic_viz_scale_depth", cfg.get("viz_scale_depth", 3))),
                    width=int(cfg.get("periodic_viz_scale_width", cfg.get("viz_scale_width", 4))),
                    branch_factor=int(cfg.get("periodic_viz_scale_branch_factor", cfg.get("viz_scale_branch_factor", 3))),
                    trace_limit=int(cfg.get("viz_trace_limit", 24)),
                    audit_level=audit_level,
                    ph_backend=ph_backend,
                    audit_max_simplices=audit_max_simplices,
                    allow_stop=bool(cfg.get("periodic_viz_scale_allow_stop", cfg.get("viz_scale_allow_stop", False))),
                    diverse_actions=bool(cfg.get("periodic_viz_scale_diverse_actions", cfg.get("viz_scale_diverse_actions", True))),
                    stochastic_actions=bool(cfg.get("periodic_viz_scale_stochastic_actions", cfg.get("viz_scale_stochastic_actions", False))),
                    sampling_temperature=float(cfg.get("periodic_viz_scale_sampling_temperature", cfg.get("viz_scale_sampling_temperature", 1.0))),
                    sampling_exploration=float(cfg.get("periodic_viz_scale_sampling_exploration", cfg.get("viz_scale_sampling_exploration", 0.0))),
                    sampling_seed=int(cfg.get("periodic_viz_scale_sampling_seed", seed + step + example_idx)),
                )
                audit_result: dict[str, Any] = {
                    "inference_scaling": scaling,
                    "topological_algebra": scaling.get("best", {}).get("topological_algebra"),
                    "audit_seed_record_index": record_idx,
                    "audit_seed_record_id": audit_record.record_id,
                }
                if memory_bank is not None:
                    records = memory_records_from_scaling_report(
                        scaling,
                        source=f"periodic:{run_name}:step{step}:record{record_idx}",
                        min_score=cfg.get("memory_min_score"),
                        max_records=int(cfg.get("memory_records_per_audit", 8)),
                    )
                    memory_bank.extend(records)
                    memory_bank.save()
                    memory_added_this_round += len(records)
                    query_embedding, query_signature = query_signature_from_report(audit_result)
                    audit_result["analogical_memory_retrieval"] = {
                        "bank_path": str(memory_bank.path),
                        "bank_size": len(memory_bank.records),
                        "records_added": len(records),
                        "top_k": int(cfg.get("periodic_memory_retrieve_top_k", 5)),
                        "retrieved": memory_bank.retrieve(
                            query_embedding,
                            query_signature,
                            top_k=int(cfg.get("periodic_memory_retrieve_top_k", 5)),
                            exclude_record_ids={
                                str(row.get("record_id", ""))
                                for row in scaling.get("candidates", [])
                                if isinstance(row, dict)
                            },
                            exclude_memory_ids={record.memory_id for record in records},
                        ),
                    }
                example_dir = step_dir / "got_audit" if example_idx == 0 else step_dir / "got_audit" / f"example_{example_idx:02d}"
                prefix = "got_audit" if example_idx == 0 else f"got_audit_example_{example_idx:02d}"
                vis_paths.update(
                    {
                        f"{prefix}_{key}": value
                        for key, value in write_inference_audit_artifacts(
                            audit_result,
                            example_dir,
                            render_html=True,
                        ).items()
                    }
                )
            if memory_bank is not None:
                memory_records_added += memory_added_this_round
                metrics["analogical_memory_records_added"] = float(memory_records_added)
                metrics["analogical_memory_bank_size"] = float(len(memory_bank.records))
    report = {
        "step": step,
        "validation": str(eval_path),
        "metrics": metrics,
        "visualizations": vis_paths,
        "audit_level": audit_level,
        "ph_backend": ph_backend,
        "audit_max_simplices": audit_max_simplices,
        "memory_records_added_this_round": memory_added_this_round,
        "memory_records_added_total": memory_records_added,
    }
    report_path = step_dir / "periodic_validation_artifacts.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path = out_dir / "periodic" / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": step, "report": str(report_path), "visualization_count": len(vis_paths)}) + "\n")
    return report


def _select_got_audit_records(dataset: Any, count: int, pool: int, seed: int) -> list[tuple[int, Any]]:
    try:
        n = len(dataset)
    except Exception:
        n = 0
    if n <= 0:
        return []
    count = max(int(count), 1)
    pool = max(int(pool), count)
    pool = min(pool, n)
    offset = int(seed) % n
    indices = [int((offset + idx) % n) for idx in range(pool)]
    scored: list[tuple[float, int, Any]] = []
    for idx in indices:
        try:
            record = dataset[idx]
        except Exception:
            continue
        scored.append((_got_audit_record_score(record), idx, record))
    if not scored:
        record = dataset[0]
        return [(0, record)]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = scored[:count]
    selected.sort(key=lambda item: item[1])
    return [(idx, record) for _, idx, record in selected]


def _got_audit_record_score(record: Any) -> float:
    graph = getattr(record, "graph_json", None)
    graph = graph if isinstance(graph, dict) else {}
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []
    node_types = {str(node.get("type", "")) for node in nodes if isinstance(node, dict)}
    edge_types = {str(edge.get("type", "")) for edge in edges if isinstance(edge, dict)}
    text = str(getattr(record, "text", "") or "")
    reasoning = str(getattr(record, "reasoning", "") or "")
    answer = str(getattr(record, "answer", "") or "")
    return (
        4.0 * min(len(nodes), 96)
        + 3.0 * min(len(edges), 160)
        + 9.0 * len(node_types)
        + 4.0 * len(edge_types)
        + min(len(text), 4096) / 64.0
        + min(len(reasoning), 4096) / 40.0
        + min(len(answer), 1024) / 48.0
    )


def _log_wandb_html_artifacts(
    wb: Any,
    vis_paths: dict[str, str],
    cfg: dict[str, Any],
    prefix: str,
    step: int | None = None,
) -> None:
    html_limit = int(cfg.get("wandb_html_artifact_limit", 5))
    uploaded_html = 0
    payload: dict[str, Any] = {}
    for key, path in _wandb_html_items(vis_paths):
        if uploaded_html >= html_limit:
            break
        try:
            import wandb

            html_path = Path(path)
            max_bytes = int(cfg.get("wandb_html_max_bytes", 5_000_000))
            if html_path.suffix.lower() != ".html" or html_path.stat().st_size > max_bytes:
                continue
            payload[f"{prefix}/{key}"] = wandb.Html(html_path.read_text(encoding="utf-8"))
            uploaded_html += 1
        except Exception:
            pass
    if payload:
        if step is not None:
            payload["step"] = step
        wb.log(payload, step=step)


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


def _wandb_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, bool) or isinstance(value, bool)


def _wandb_fallback_group(key: str) -> str:
    if key.startswith("eval_"):
        return "00_primary" if key in {"eval_bpb", "eval_graph_bpb", "eval_nll", "eval_ppl"} else "02_bpb"
    if key.startswith(("loss_",)) or key in {"loss", "nll", "ppl"}:
        return "01_losses"
    if key.startswith(("graph_bpb", "text_bpb", "bpb", "graph_sideinfo", "graph_conditioned", "explicit_graph", "target_bytes", "nll_bits")):
        return "02_bpb"
    if key.startswith(("support_", "margin_", "wall_", "certificate_", "sequence_tropical_", "tropical_", "positive_margin", "self_support", "invalid_support")):
        return "03_tropical"
    if key.startswith("gflownet_"):
        return "04_gflownet"
    if key.startswith("graphcg_"):
        return "05_graphcg"
    if key.startswith(("graph_", "node_", "edge_", "causal_", "random_graph_", "parameter_golf_", "tokens_seen", "examples_seen")):
        return "06_graph_data"
    if key.startswith(("algebra_", "homology_", "persistence_", "multipersistence_", "free_resolution_", "derived_", "betti_", "syzygy_")):
        return "07_algebra_topology"
    if key.startswith("analogical_memory_"):
        return "08_memory"
    if key.startswith(("gpu_", "step_time", "examples_per_sec", "tokens_per_sec", "batch_size")):
        return "09_system"
    if key.startswith(("optimizer_", "grad_", "chunk_shuffle", "sampler_", "shuffle_")):
        return "10_optimization"
    return "99_other"


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
    graph_autoregressive: bool = False,
    ar_seed: int = 0,
    meet_in_middle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mim_cfg = meet_in_middle_config(meet_in_middle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda r: collate_records(r, seq_len, tokenizer, graph_autoregressive=graph_autoregressive, ar_seed=ar_seed),
    )
    total_loss = 0.0; total_tokens = 0; batches = 0
    details: list[dict[str, Any]] = []
    graph_token_total = 0
    graph_token_structural_bytes_total = 0
    explicit_graph_json_bytes_total = 0
    node_token_total = 0
    edge_token_total = 0
    graph_json_fallback_total = 0
    causal_dag_total = 0
    random_graph_total = 0
    parameter_golf_total = 0
    mim_metric_sums: dict[str, float] = {}
    mim_batches = 0
    mim_records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for x, y, graph_batch, records in loader:
            x = x.to(device); y = y.to(device)
            out = model(x, graph_batch, y)
            if mim_cfg.enabled:
                mim_report = meet_in_middle_batch(
                    model,
                    list(records),
                    tokenizer,
                    seq_len,
                    device,
                    graph_autoregressive=graph_autoregressive,
                    seed=ar_seed,
                    config=mim_cfg,
                    forward_logits=out.get("logits"),
                    forward_nll=out.get("nll"),
                    require_grad=False,
                )
                mim_batches += 1
                for key, value in mim_report.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        mim_metric_sums[key] = mim_metric_sums.get(key, 0.0) + float(value)
                if details_limit > 0 and len(mim_records) < details_limit:
                    mim_records.extend(mim_report.get("records", [])[: max(details_limit - len(mim_records), 0)])
            tokens = int((y != 0).sum().item())
            total_loss += float(out["nll"].detach().cpu()) * max(tokens, 1)
            total_tokens += max(tokens, 1); batches += 1
            graph_token_total += int(graph_batch.graph_token_counts.sum().item())
            graph_token_structural_bytes_total += graph_token_structural_bytes(graph_batch)
            explicit_graph_json_bytes_total += sum(explicit_graph_json_bytes(record) for record in records)
            node_token_total += int(graph_batch.node_counts.sum().item())
            edge_token_total += int(graph_batch.edge_counts.sum().item())
            graph_json_fallback_total += sum(1 for record in records if (record.metadata or {}).get("graph_json_fallback", False))
            causal_dag_total += sum(1 for record in records if (record.metadata or {}).get("decoding_order_kind") == "causal_dag")
            random_graph_total += sum(1 for record in records if (record.metadata or {}).get("decoding_order_kind") == "random_autoregressive")
            parameter_golf_total += sum(
                1 for record in records if (record.metadata or {}).get("source") == "parameter_golf_bin" or (record.metadata or {}).get("hybrid_source") == "openai_parameter_golf"
            )
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
        "graph_json_fallback_records": graph_json_fallback_total,
        "invalid_graph_rate": graph_json_fallback_total / max(len(dataset), 1),
        "causal_dag_ar_rate": causal_dag_total / max(len(dataset), 1),
        "random_graph_ar_rate": random_graph_total / max(len(dataset), 1),
        "parameter_golf_source_rate": parameter_golf_total / max(len(dataset), 1),
        "graph_autoregressive_decoding_enabled": 1.0 if graph_autoregressive else 0.0,
        **bpb,
    }
    if mim_cfg.enabled:
        report.update(
            {
                key: value / max(mim_batches, 1)
                for key, value in mim_metric_sums.items()
            }
        )
        report["meet_in_middle"] = {
            "enabled": True,
            "mode": mim_cfg.mode if not mim_cfg.reverse_model_path else "explicit_reverse_model_requested",
            "shared_weight_reverse_pass": not bool(mim_cfg.reverse_model_path),
            "split_ratio": mim_cfg.split_ratio,
            "agreement_weight": mim_cfg.agreement_weight,
            "reverse_nll_weight": mim_cfg.reverse_nll_weight,
            "batches": mim_batches,
            "records": mim_records,
            "note": "Current implementation uses a shared TropicalGT-I model on the reversed graph-byte order unless reverse_model_path is supplied.",
        }
    else:
        report["mim_enabled"] = 0.0
    if details:
        report["records"] = details
    return report


def load_checkpoint(path: str | Path, device: torch.device):
    obj = torch.load(path, map_location=device)
    model = build_model(obj["config"]).to(device)
    model.load_state_dict(obj["model"], strict=False)
    model.eval()
    # Inference/eval callers only need model/config/metrics metadata. Drop the
    # optimizer state so full training checkpoints can be audited beside a live
    # training run without keeping a second optimizer copy resident in memory.
    obj.pop("optimizer", None)
    obj.pop("rng_state", None)
    obj.pop("cuda_rng_state_all", None)
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
