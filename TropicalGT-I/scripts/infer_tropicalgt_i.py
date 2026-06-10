#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import torch
from tropicalgt.diagnostics import gflownet_diagnostics, graphcg_diagnostics, record_diagnostics
from tropicalgt.run import load_config, load_checkpoint, collate_records
from tropicalgt.records import GraphRecord, conservative_graph
from tropicalgt.scaling import run_inference_scaling
from tropicalgt.tokenizer import TokenGTTokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Run TropicalGT-I inference")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Question: add 2 and 3\nAnswer:")
    parser.add_argument("--trace-limit", type=int, default=64)
    parser.add_argument("--scale-depth", type=int, default=None)
    parser.add_argument("--scale-width", type=int, default=None)
    parser.add_argument("--scale-branch-factor", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    rec = GraphRecord("inference", args.prompt, graph_json=conservative_graph(question=args.prompt, text=args.prompt))
    tok = TokenGTTokenizer(**cfg.get("tokengt", {}))
    scaling_cfg = cfg.get("inference_scaling", {})
    scale_depth = int(scaling_cfg.get("depth", 0) if args.scale_depth is None else args.scale_depth)
    scale_width = int(scaling_cfg.get("width", 3) if args.scale_width is None else args.scale_width)
    scale_branch_factor = int(scaling_cfg.get("branch_factor", 2) if args.scale_branch_factor is None else args.scale_branch_factor)
    x, y, gb, _ = collate_records([rec], int(cfg.get("seq_len", 128)), tok)
    with torch.no_grad():
        out = model(x.to(device), gb, y.to(device))
        pred = out["logits"].argmax(dim=-1)[0].detach().cpu().tolist()
    text = bytes([max(0, t - 1) for t in pred if t > 0]).decode("utf-8", "ignore")
    record_report = record_diagnostics([rec], gb, out, tok, target_ids=y, max_records=1, max_trace_tokens=args.trace_limit)[0]
    result = {
        "prompt": args.prompt,
        "decoded_argmax": text,
        "support": out["support"].detach().cpu().tolist(),
        "margin_mean": float(out["margin_mean"].detach().cpu()),
        "tropical": record_report["tropical"],
        "graph_token_trace": record_report["graph_token_trace"],
        "filtered_simplicial_object": record_report["filtered_simplicial_object"],
        "gflownet": gflownet_diagnostics(model, out["graph_state"]),
        "graphcg": graphcg_diagnostics(model, out["graph_state"]),
    }
    if scale_depth > 0:
        result["inference_scaling"] = run_inference_scaling(
            model,
            rec,
            tok,
            int(cfg.get("seq_len", 128)),
            device,
            depth=scale_depth,
            width=scale_width,
            branch_factor=scale_branch_factor,
            trace_limit=args.trace_limit,
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
