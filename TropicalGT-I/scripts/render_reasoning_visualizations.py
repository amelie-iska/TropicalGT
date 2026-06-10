#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import torch
from tropicalgt.run import load_config, load_checkpoint
from tropicalgt.data import make_dataset
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import write_reasoning_visualizations

def main() -> None:
    parser = argparse.ArgumentParser(description="Render TropicalGT-I reasoning visualizations")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audit-level", choices=["none", "basic", "topology", "algebra", "full"], default="none")
    parser.add_argument("--ph-backend", choices=["auto", "gudhi", "ripser", "none"], default="auto")
    parser.add_argument("--audit-max-simplices", type=int, default=1024)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    root = cfg.get("data_root")
    ds = make_dataset(
        root,
        args.split,
        limit=cfg.get("val_limit", 8),
        fixture_size=cfg.get("fixture_size", 8),
        require_data=bool(cfg.get("require_data", bool(root))),
        cache_shards=int(cfg.get("cache_shards", 2)),
    )
    tok = TokenGTTokenizer(**cfg.get("tokengt", {}))
    paths = write_reasoning_visualizations(
        model,
        ds,
        tok,
        int(cfg.get("seq_len", 128)),
        device,
        args.output_dir or cfg.get("output_dir", "TropicalGT-I/outputs/smoke"),
        limit=int(args.limit if args.limit is not None else cfg.get("viz_limit", 8)),
        audit_level=args.audit_level,
        ph_backend=args.ph_backend,
        audit_max_simplices=args.audit_max_simplices,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
