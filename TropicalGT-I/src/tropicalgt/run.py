from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import encode_bytes, make_dataset
from .diagnostics import record_diagnostics
from .model import TropicalGTConfig, TropicalGTModel
from .tokenizer import TokenGTTokenizer
from .visualization import write_metric_visualizations, write_reasoning_visualizations


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


def train(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = cfg.get("data_root")
    train_ds = make_dataset(root, "train", limit=cfg.get("train_limit"), fixture_size=cfg.get("fixture_size", 8))
    val_ds = make_dataset(root, "validation", limit=cfg.get("val_limit", cfg.get("train_limit", 4)), fixture_size=cfg.get("fixture_size", 8))
    tokenizer = TokenGTTokenizer(**cfg.get("tokengt", {}))
    seq_len = int(cfg.get("seq_len", 128))
    batch_size = int(cfg.get("batch_size", 2))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    run_name = cfg.get("run_name", f"tropicalgt-i-{int(time.time())}")
    wb = setup_wandb(cfg, run_name)
    out_dir = Path(cfg.get("output_dir", "TropicalGT-I/outputs/smoke")); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.get("checkpoint_dir", "TropicalGT-I/checkpoints")); ckpt_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda r: collate_records(r, seq_len, tokenizer))
    max_steps = int(cfg.get("max_steps", 5))
    metrics_last: dict[str, float] = {}
    history: list[dict[str, float]] = []
    model.train(); step = 0
    pbar = tqdm(total=max_steps, desc="TropicalGT-I train", dynamic_ncols=True)
    while step < max_steps:
        for x, y, graph_batch, _records in loader:
            step += 1
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, graph_batch, y)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            metrics_last = {k: float(v.detach().cpu()) for k, v in out.items() if torch.is_tensor(v) and v.ndim == 0}
            metrics_last["loss"] = float(out["loss"].detach().cpu())
            metrics_last["ppl"] = float(math.exp(min(metrics_last.get("nll", metrics_last["loss"]), 20)))
            metrics_last["step"] = step
            if torch.cuda.is_available():
                metrics_last["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            history.append(dict(metrics_last))
            if wb:
                wb.log(metrics_last, step=step)
            pbar.set_postfix({"loss": f"{metrics_last['loss']:.3f}", "nll": f"{metrics_last.get('nll', 0):.3f}"})
            pbar.update(1)
            if step >= max_steps:
                break
    pbar.close()
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics_last}, ckpt_path)
    eval_report = evaluate_model(model, val_ds, tokenizer, seq_len, batch_size, device)
    metrics_last.update({f"eval_{k}": v for k, v in eval_report.items() if isinstance(v, (int, float))})
    vis_paths = write_reasoning_visualizations(model, val_ds, tokenizer, seq_len, device, out_dir, limit=int(cfg.get("viz_limit", 8)))
    vis_paths.update(write_metric_visualizations(history, out_dir))
    report = {"checkpoint": str(ckpt_path), "metrics": metrics_last, "history": history, "eval": eval_report, "visualizations": vis_paths, "device": str(device)}
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if wb:
        for key, path in vis_paths.items():
            try:
                import wandb
                wb.log({key: wandb.Html(open(path, encoding="utf-8").read())})
            except Exception:
                pass
        wb.finish()
    return report


def evaluate_model(model: TropicalGTModel, dataset, tokenizer: TokenGTTokenizer, seq_len: int, batch_size: int, device: torch.device, details_limit: int = 0) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda r: collate_records(r, seq_len, tokenizer))
    total_loss = 0.0; total_tokens = 0; batches = 0
    details: list[dict[str, Any]] = []
    graph_token_total = 0
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
                    )
                )
    nll = total_loss / max(total_tokens, 1)
    report: dict[str, Any] = {
        "nll": nll,
        "ppl": math.exp(min(nll, 20)),
        "bpb_proxy": nll / math.log(2.0),
        "batches": batches,
        "tokens": total_tokens,
        "graph_tokens": graph_token_total,
        "node_tokens": node_token_total,
        "edge_tokens": edge_token_total,
        "graph_json_fallback_records": fallback_total,
        "invalid_graph_rate": fallback_total / max(len(dataset), 1),
    }
    if details:
        report["records"] = details
    return report


def load_checkpoint(path: str | Path, device: torch.device):
    obj = torch.load(path, map_location=device)
    model = build_model(obj["config"]).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    return model, obj
