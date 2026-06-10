#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import torch
from tropicalgt.run import load_config, load_checkpoint, collate_records
from tropicalgt.records import GraphRecord, conservative_graph
from tropicalgt.tokenizer import TokenGTTokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Run TropicalGT-I inference")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Question: add 2 and 3\nAnswer:")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    rec = GraphRecord("inference", args.prompt, graph_json=conservative_graph(question=args.prompt, text=args.prompt))
    tok = TokenGTTokenizer(**cfg.get("tokengt", {}))
    x, y, gb, _ = collate_records([rec], int(cfg.get("seq_len", 128)), tok)
    with torch.no_grad():
        out = model(x.to(device), gb, y.to(device))
        pred = out["logits"].argmax(dim=-1)[0].detach().cpu().tolist()
    text = bytes([max(0, t - 1) for t in pred if t > 0]).decode("utf-8", "ignore")
    result = {"prompt": args.prompt, "decoded_argmax": text, "support": out["support"].detach().cpu().tolist(), "margin_mean": float(out["margin_mean"].detach().cpu())}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
