#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from tropicalgt.run import load_config
from tropicalgt.data import make_dataset
from tropicalgt.tokenizer import TokenGTTokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TropicalGT-I data and tokenization")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ds = make_dataset(cfg.get("data_root"), args.split, limit=cfg.get("train_limit", 4), fixture_size=cfg.get("fixture_size", 8))
    records = [ds[i] for i in range(min(len(ds), int(cfg.get("batch_size", 2))))]
    batch = TokenGTTokenizer(**cfg.get("tokengt", {})).batch_encode(records)
    report = {"records": len(ds), "batch_tokens": batch.graph_token_counts.tolist(), "node_counts": batch.node_counts.tolist(), "edge_counts": batch.edge_counts.tolist(), "feature_shape": list(batch.token_features.shape)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
