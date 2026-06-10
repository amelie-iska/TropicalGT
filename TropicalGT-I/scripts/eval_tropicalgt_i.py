#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import torch
from tropicalgt.run import load_config, load_checkpoint, evaluate_model
from tropicalgt.data import make_dataset_from_config
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import write_graphcg_training_visualizations, write_metric_visualizations, write_reasoning_visualizations

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TropicalGT-I checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--details-limit", type=int, default=4)
    parser.add_argument("--audit-level", choices=["none", "basic", "topology", "algebra", "full"], default="")
    parser.add_argument("--audit-ph-backend", choices=["auto", "gudhi", "ripser", "none"], default="")
    parser.add_argument("--audit-max-simplices", type=int, default=0)
    parser.add_argument("--render-visualizations", action="store_true")
    parser.add_argument("--visualization-output-dir", default="")
    parser.add_argument("--viz-limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model, obj = load_checkpoint(args.checkpoint, device)
    ds = make_dataset_from_config(cfg, args.split)
    tok = TokenGTTokenizer(**cfg.get("tokengt", {}))
    report = evaluate_model(
        model,
        ds,
        tok,
        int(cfg.get("seq_len", 128)),
        int(cfg.get("batch_size", 2)),
        device,
        details_limit=args.details_limit,
        graph_bpb_side_weight=float(cfg.get("graph_bpb_side_weight", 1.0)),
        audit_level=args.audit_level or str(cfg.get("audit_level", "none")),
        ph_backend=args.audit_ph_backend or str(cfg.get("ph_backend", "auto")),
        audit_max_simplices=int(args.audit_max_simplices or cfg.get("audit_max_simplices", 1024)),
        graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
        ar_seed=int(cfg.get("seed", 1729)),
    )
    if args.render_visualizations:
        vis_dir = Path(args.visualization_output_dir or cfg.get("output_dir", "TropicalGT-I/outputs/smoke")) / f"eval_{args.split}_visualizations"
        paths = write_reasoning_visualizations(
            model,
            ds,
            tok,
            int(cfg.get("seq_len", 128)),
            device,
            vis_dir / "reasoning",
            limit=int(args.viz_limit if args.viz_limit is not None else cfg.get("viz_limit", 8)),
            audit_level=args.audit_level or str(cfg.get("audit_level", "none")),
            ph_backend=args.audit_ph_backend or str(cfg.get("ph_backend", "auto")),
            audit_max_simplices=int(args.audit_max_simplices or cfg.get("audit_max_simplices", 1024)),
        )
        paths.update(write_metric_visualizations([report], vis_dir / "metrics"))
        paths.update(write_graphcg_training_visualizations(model, vis_dir / "graphcg"))
        report["visualizations"] = paths
    out = Path(cfg.get("output_dir", "TropicalGT-I/outputs/smoke")) / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": report, "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
