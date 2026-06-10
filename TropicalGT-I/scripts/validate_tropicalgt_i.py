#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from tropicalgt.run import load_config
from tropicalgt.data import make_dataset
from tropicalgt.diagnostics import describe_graph_tokens
from tropicalgt.tokenizer import TokenGTTokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TropicalGT-I data and tokenization")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-limit", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ds = make_dataset(cfg.get("data_root"), args.split, limit=cfg.get("train_limit", 4), fixture_size=cfg.get("fixture_size", 8))
    all_records = [ds[i] for i in range(len(ds))]
    records = all_records[: min(len(all_records), int(cfg.get("batch_size", 2)))]
    tok = TokenGTTokenizer(**cfg.get("tokengt", {}))
    batch = tok.batch_encode(records)
    fallback_count = sum(1 for record in all_records if (record.metadata or {}).get("graph_json_fallback", False))
    token_counts = []
    node_counts = []
    edge_counts = []
    for record in all_records:
        encoded = tok.encode(record)
        token_counts.append(int(encoded[0].shape[0]))
        node_counts.append(int(encoded[3]))
        edge_counts.append(int(encoded[4]))
    def stats(values):
        return {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": sum(values) / max(len(values), 1),
        }
    report = {
        "records": len(ds),
        "split": args.split,
        "graph_json_fallback_records": fallback_count,
        "invalid_graph_rate": fallback_count / max(len(ds), 1),
        "token_count_stats": stats(token_counts),
        "node_count_stats": stats(node_counts),
        "edge_count_stats": stats(edge_counts),
        "batch_tokens": batch.graph_token_counts.tolist(),
        "node_counts": batch.node_counts.tolist(),
        "edge_counts": batch.edge_counts.tolist(),
        "feature_shape": list(batch.token_features.shape),
        "samples": [
            {
                "record_id": record.record_id,
                "graph_json_fallback": bool((record.metadata or {}).get("graph_json_fallback", False)),
                "graph_tokens": describe_graph_tokens(record, tok)[: args.sample_limit],
            }
            for record in all_records[: args.sample_limit]
        ],
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
