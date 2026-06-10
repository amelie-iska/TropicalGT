#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import torch
from tropicalgt.run import load_config, load_checkpoint, evaluate_model
from tropicalgt.data import make_dataset
from tropicalgt.tokenizer import TokenGTTokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TropicalGT-I checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model, obj = load_checkpoint(args.checkpoint, device)
    ds = make_dataset(cfg.get("data_root"), args.split, limit=cfg.get("val_limit", cfg.get("train_limit", 4)), fixture_size=cfg.get("fixture_size", 8))
    tok = TokenGTTokenizer(**cfg.get("tokengt", {}))
    report = evaluate_model(model, ds, tok, int(cfg.get("seq_len", 128)), int(cfg.get("batch_size", 2)), device)
    out = Path(cfg.get("output_dir", "TropicalGT-I/outputs/smoke")) / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": report, "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
