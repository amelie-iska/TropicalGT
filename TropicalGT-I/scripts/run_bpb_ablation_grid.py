#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tropicalgt.ablation import write_bpb_ablation_artifacts
from tropicalgt.run import load_config, train


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no_graphcg": {"model.graphcg_weight": 0.0},
    "no_gflownet": {"model.gflownet_weight": 0.0},
    "no_certificate": {"model.certificate_weight": 0.0},
    "no_tropical_regularizers": {
        "model.margin_weight": 0.0,
        "model.entropy_weight": 0.0,
        "model.certificate_weight": 0.0,
    },
    "no_auxiliary": {
        "model.gflownet_weight": 0.0,
        "model.graphcg_weight": 0.0,
        "model.margin_weight": 0.0,
        "model.entropy_weight": 0.0,
        "model.certificate_weight": 0.0,
    },
    "no_memory_bank": {"memory_bank_path": ""},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally run matched TropicalGT-I BPB ablations")
    parser.add_argument("--config", default=str(ROOT / "configs" / "gpu_smoke.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "bpb_ablation_grid"))
    parser.add_argument("--variants", default="baseline,no_graphcg,no_gflownet,no_certificate,no_tropical_regularizers,no_auxiliary")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Actually train each generated variant")
    parser.add_argument("--fixture", action="store_true", help="Force fixture data for quick CPU/debug ablations")
    parser.add_argument("--device", choices=["auto", "cpu"], default=None)
    parser.add_argument("--wandb", action="store_true", help="Keep W&B enabled from the base config. Defaults to disabled for ablation grids.")
    parser.add_argument("--render-html", action="store_true", help="Render Plotly correlation report after running")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    base = load_config(args.config)
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variant_names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}; known={', '.join(sorted(VARIANTS))}")

    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    report_paths: list[str] = []
    configs = []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    for idx, name in enumerate(variant_names):
        cfg = _variant_config(
            base,
            name,
            output_dir,
            run_id,
            VARIANTS[name],
            seed=args.seed,
            max_steps=args.max_steps,
            fixture=args.fixture,
            device=args.device,
            wandb=args.wandb,
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{idx:02d}_{name}.json"
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        configs.append({"variant": name, "config": str(config_path), "overrides": VARIANTS[name]})
        if args.run:
            train(config_path, max_steps_override=args.max_steps)
            report_path = Path(cfg["output_dir"]) / "train_report.json"
            report_paths.append(str(report_path))

    manifest = {
        "base_config": str(args.config),
        "run_id": run_id,
        "ran_training": bool(args.run),
        "variants": configs,
        "reports": report_paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ablation_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    paths = {"manifest": str(manifest_path)}
    if args.run and report_paths:
        paths.update(
            {
                f"analysis_{key}": value
                for key, value in write_bpb_ablation_artifacts(
                    report_paths,
                    output_dir / "analysis",
                    baseline=report_paths[0],
                    top_k=args.top_k,
                    render_html=args.render_html,
                ).items()
            }
        )
    print(json.dumps(paths, indent=2))


def _variant_config(
    base: dict[str, Any],
    name: str,
    output_dir: Path,
    run_id: str,
    overrides: dict[str, Any],
    seed: int | None,
    max_steps: int | None,
    fixture: bool,
    device: str | None,
    wandb: bool,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    base_name = str(base.get("run_name", "tropicalgt_i"))
    cfg["run_name"] = f"{base_name}_{name}_{run_id}"
    cfg["output_dir"] = str(output_dir / name)
    cfg["checkpoint_dir"] = str(output_dir / name / "checkpoints")
    if cfg.get("memory_bank_path"):
        cfg["memory_bank_path"] = str(output_dir / name / "analogical_memory" / "reasoning_memory.jsonl")
    if seed is not None:
        cfg["seed"] = int(seed)
    else:
        cfg["seed"] = int(cfg.get("seed", 1729))
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if fixture:
        cfg["data_root"] = None
        cfg["require_data"] = False
        cfg["train_limit"] = int(cfg.get("train_limit") or 8)
        cfg["val_limit"] = int(cfg.get("val_limit") or 4)
        cfg["fixture_size"] = max(int(cfg.get("fixture_size", 8)), cfg["train_limit"], cfg["val_limit"])
        cfg["chunk_shuffle"] = False
    if device:
        cfg["device"] = device
    if not wandb:
        cfg["wandb"] = {**dict(cfg.get("wandb", {})), "enabled": False}
    for dotted_key, value in overrides.items():
        _set_dotted(cfg, dotted_key, value)
    cfg["ablation_variant"] = name
    cfg["ablation_overrides"] = overrides
    return cfg


def _set_dotted(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = cfg
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


if __name__ == "__main__":
    main()
