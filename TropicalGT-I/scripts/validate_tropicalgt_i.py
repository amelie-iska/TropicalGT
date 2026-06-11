#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from tropicalgt.run import load_config
from tropicalgt.data import dataset_manifest, make_dataset_from_config
from tropicalgt.diagnostics import describe_graph_tokens
from tropicalgt.tokenizer import TokenGTTokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TropicalGT-I data and tokenization")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg.get("data_root")
    require_data = bool(cfg.get("require_data", bool(root)))
    cfg_sample = dict(cfg)
    limit = args.limit if args.limit is not None else cfg.get("train_limit", 4)
    if args.split == "train":
        cfg_sample["train_limit"] = limit
    else:
        cfg_sample["val_limit"] = limit
    ds = make_dataset_from_config(cfg_sample, args.split)
    all_records = [ds[i] for i in range(len(ds))]
    records = all_records[: min(len(all_records), int(cfg.get("batch_size", 2)))]
    tok = TokenGTTokenizer(**cfg.get("tokengt", {}))
    batch = tok.batch_encode(records)
    graph_json_fallback_count = sum(1 for record in all_records if (record.metadata or {}).get("graph_json_fallback", False))
    causal_dag_count = sum(1 for record in all_records if (record.metadata or {}).get("decoding_order_kind") == "causal_dag")
    random_ar_count = sum(1 for record in all_records if (record.metadata or {}).get("decoding_order_kind") == "random_autoregressive")
    parameter_golf_count = sum(
        1 for record in all_records if (record.metadata or {}).get("source") == "parameter_golf_bin" or (record.metadata or {}).get("hybrid_source") == "openai_parameter_golf"
    )
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
        "dataset_required": require_data,
        "manifest": dataset_manifest(ds, root),
        "graph_json_fallback_records": graph_json_fallback_count,
        "invalid_graph_rate": graph_json_fallback_count / max(len(ds), 1),
        "causal_dag_ar_records": causal_dag_count,
        "causal_dag_ar_rate": causal_dag_count / max(len(ds), 1),
        "random_graph_ar_records": random_ar_count,
        "random_graph_ar_rate": random_ar_count / max(len(ds), 1),
        "parameter_golf_source_records": parameter_golf_count,
        "parameter_golf_source_rate": parameter_golf_count / max(len(ds), 1),
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
                "decoding_order_kind": (record.metadata or {}).get("decoding_order_kind", ""),
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
