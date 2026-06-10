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
    paths = write_reasoning_visualizations(model, ds, tok, int(cfg.get("seq_len", 128)), device, cfg.get("output_dir", "TropicalGT-I/outputs/smoke"), limit=int(cfg.get("viz_limit", 8)))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
