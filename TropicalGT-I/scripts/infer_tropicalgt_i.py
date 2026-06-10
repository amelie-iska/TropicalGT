#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import torch
from tropicalgt.algebra import compute_topological_algebra_report
from tropicalgt.diagnostics import gflownet_diagnostics, graphcg_diagnostics, record_diagnostics
from tropicalgt.memory import AnalogicalMemoryBank, memory_records_from_scaling_report, query_signature_from_report
from tropicalgt.metrics import batch_bpb_metrics
from tropicalgt.run import load_config, load_checkpoint, collate_records
from tropicalgt.records import GraphRecord
from tropicalgt.scaling import run_inference_scaling
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import write_inference_audit_artifacts

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
    parser.add_argument("--audit-output-dir", default="")
    parser.add_argument("--audit-level", choices=["none", "basic", "topology", "algebra", "full"], default="none")
    parser.add_argument("--audit-ph-backend", choices=["auto", "gudhi", "ripser", "none"], default="auto")
    parser.add_argument("--audit-max-simplices", type=int, default=1024)
    parser.add_argument("--audit-render-html", action="store_true")
    parser.add_argument("--memory-bank", default="", help="JSONL analogical reasoning memory bank to query or update")
    parser.add_argument("--memory-save", action="store_true", help="Append inference-scaling candidates to the memory bank")
    parser.add_argument("--memory-retrieve-top-k", type=int, default=0, help="Retrieve this many analogical memories")
    parser.add_argument("--memory-max-records", type=int, default=2048)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    rec = GraphRecord.from_mapping(
        {
            "record_id": "inference",
            "text": args.prompt,
            "question": args.prompt,
        }
    )
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
    record_report = record_diagnostics(
        [rec],
        gb,
        out,
        tok,
        target_ids=y,
        max_records=1,
        max_trace_tokens=args.trace_limit,
        audit_level=args.audit_level,
        ph_backend=args.audit_ph_backend,
        audit_max_simplices=args.audit_max_simplices,
    )[0]
    topological_algebra = record_report.get("topological_algebra")
    if topological_algebra is None and args.audit_level != "none":
        topological_algebra = compute_topological_algebra_report(
            record_report["filtered_simplicial_object"],
            audit_level=args.audit_level,
            ph_backend=args.audit_ph_backend,
            max_simplices=args.audit_max_simplices,
        )
    result = {
        "prompt": args.prompt,
        "decoded_argmax": text,
        "support": out["support"].detach().cpu().tolist(),
        "margin_mean": float(out["margin_mean"].detach().cpu()),
        "tropical": record_report["tropical"],
        "graph_token_trace": record_report["graph_token_trace"],
        "filtered_simplicial_object": record_report["filtered_simplicial_object"],
        "topological_algebra": topological_algebra,
        "gflownet": gflownet_diagnostics(model, out["graph_state"]),
        "graphcg": graphcg_diagnostics(model, out["graph_state"]),
        "compression": batch_bpb_metrics(out["nll"], y, gb, [rec], float(cfg.get("graph_bpb_side_weight", 1.0))),
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
            audit_level=args.audit_level,
            ph_backend=args.audit_ph_backend,
            audit_max_simplices=args.audit_max_simplices,
        )
    if args.memory_bank:
        bank = AnalogicalMemoryBank(args.memory_bank, max_records=args.memory_max_records)
        retrieved = []
        if args.memory_retrieve_top_k > 0:
            embedding, signature = query_signature_from_report(result)
            retrieved = bank.retrieve(embedding, signature, top_k=args.memory_retrieve_top_k)
        added = 0
        if args.memory_save and isinstance(result.get("inference_scaling"), dict):
            records = memory_records_from_scaling_report(
                result["inference_scaling"],
                source="inference",
                max_records=min(args.memory_max_records, 16),
            )
            bank.extend(records)
            bank.save()
            added = len(records)
            if args.memory_retrieve_top_k > 0 and not retrieved:
                embedding, signature = query_signature_from_report(result)
                retrieved = bank.retrieve(embedding, signature, top_k=args.memory_retrieve_top_k)
        result["analogical_memory_retrieval"] = {
            "bank_path": str(bank.path),
            "bank_size": len(bank.records),
            "records_added": added,
            "top_k": args.memory_retrieve_top_k,
            "retrieved": retrieved,
        }
    if args.audit_output_dir:
        result["audit_artifacts"] = write_inference_audit_artifacts(
            result,
            args.audit_output_dir,
            render_html=args.audit_render_html,
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
