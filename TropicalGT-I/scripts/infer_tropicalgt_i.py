#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import torch
from tropicalgt.algebra import compute_topological_algebra_report
from tropicalgt.decoding import meet_in_middle_batch
from tropicalgt.diagnostics import gflownet_diagnostics, graphcg_diagnostics, record_diagnostics
from tropicalgt.memory import AnalogicalMemoryBank, memory_records_from_scaling_report, query_signature_from_report, query_topology_from_report
from tropicalgt.metrics import batch_bpb_metrics
from tropicalgt.run import load_config, load_checkpoint, collate_records
from tropicalgt.records import GraphRecord
from tropicalgt.scaling import run_inference_scaling
from tropicalgt.tokenizer import TokenGTTokenizer
from tropicalgt.visualization import write_inference_audit_artifacts


def _inference_memory_source(prompt: str, sampling_seed: int, audit_output_dir: str) -> str:
    digest = hashlib.blake2b(
        f"{prompt}\n{sampling_seed}\n{audit_output_dir}".encode("utf-8", "ignore"),
        digest_size=10,
    ).hexdigest()
    return f"inference:{digest}"


def _current_scaling_record_ids(result: dict) -> set[str]:
    scaling = result.get("inference_scaling")
    if not isinstance(scaling, dict):
        return set()
    rows = scaling.get("candidates", [])
    return {str(row.get("record_id")) for row in rows if isinstance(row, dict) and row.get("record_id") is not None}


def _resolve_render_html(
    *,
    audit_all: bool,
    audit_output_dir: str,
    audit_render_html: bool,
    no_audit_render_html: bool,
    interactive_artifacts: bool,
) -> bool:
    del audit_output_dir
    if no_audit_render_html:
        return False
    return bool(audit_all or audit_render_html or interactive_artifacts)


def _resolve_require_complete_reasoning_steps(
    *,
    audit_all: bool,
    render_html: bool,
    require_complete_reasoning_steps: bool,
) -> bool:
    return bool(require_complete_reasoning_steps or (audit_all and render_html))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TropicalGT-I inference")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Question: add 2 and 3\nAnswer:")
    parser.add_argument("--trace-limit", type=int, default=64)
    parser.add_argument("--scale-depth", type=int, default=None)
    parser.add_argument("--scale-width", type=int, default=None)
    parser.add_argument("--scale-branch-factor", type=int, default=None)
    parser.add_argument("--scale-stochastic-actions", action="store_true")
    parser.add_argument("--scale-sampling-temperature", type=float, default=None)
    parser.add_argument("--scale-sampling-exploration", type=float, default=None)
    parser.add_argument("--scale-sampling-seed", type=int, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--audit-output-dir", default="")
    parser.add_argument("--audit-level", choices=["none", "basic", "topology", "algebra", "full"], default="none")
    parser.add_argument("--audit-ph-backend", choices=["auto", "gudhi", "ripser", "none"], default="auto")
    parser.add_argument("--audit-max-simplices", type=int, default=1024)
    parser.add_argument("--audit-render-html", action="store_true")
    parser.add_argument("--no-audit-render-html", action="store_true")
    parser.add_argument(
        "--interactive-artifacts",
        action="store_true",
        help="Render optional local interactive audit HTML pages. JSON audit output can be written without this flag.",
    )
    parser.add_argument(
        "--audit-all",
        action="store_true",
        help="Emit full optional inference artifacts: GoT PCA, NLL fields, tropical support, GraphCG, analogical memory, chain-presentation/algebra JSON, derived signatures, persistence plots, and dashboard HTML.",
    )
    parser.add_argument(
        "--require-complete-reasoning-steps",
        action="store_true",
        help="Fail inference scaling if any browserable reasoning step lacks real model NLL, embeddings, probability vectors, trace tokens, GraphCG data, or directed GoT parent edge.",
    )
    parser.add_argument("--memory-bank", default="", help="JSONL analogical reasoning memory bank to query or update")
    parser.add_argument("--memory-save", action="store_true", help="Append inference-scaling candidates to the memory bank")
    parser.add_argument("--memory-retrieve-top-k", type=int, default=0, help="Retrieve this many analogical memories")
    parser.add_argument("--memory-max-records", type=int, default=2048)
    parser.add_argument("--memory-landscape-weight", type=float, default=None, help="Weight for GUDHI persistence-landscape vector similarity during analogical memory retrieval")
    parser.add_argument("--meet-in-middle", action="store_true", help="Enable graph-aware meet-in-the-middle reverse-pass diagnostics")
    parser.add_argument("--no-meet-in-middle", action="store_true", help="Disable meet-in-the-middle diagnostics even if enabled in config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    mim_cfg = dict(cfg.get("meet_in_middle", {}))
    if args.meet_in_middle:
        mim_cfg["enabled"] = True
    if args.no_meet_in_middle:
        mim_cfg["enabled"] = False
    if args.audit_all and args.audit_level == "none":
        args.audit_level = "full"
    if args.audit_all and args.audit_output_dir == "":
        args.audit_output_dir = str(Path(cfg.get("output_dir", "TropicalGT-I/outputs/smoke")) / "inference_audit")
    if args.audit_all and args.memory_bank == "":
        args.memory_bank = str(cfg.get("memory_bank_path", ""))
    if args.audit_all and args.memory_retrieve_top_k <= 0:
        args.memory_retrieve_top_k = int(cfg.get("inference_memory_retrieve_top_k", cfg.get("periodic_memory_retrieve_top_k", 5)))
    render_html = _resolve_render_html(
        audit_all=bool(args.audit_all),
        audit_output_dir=args.audit_output_dir,
        audit_render_html=bool(args.audit_render_html),
        no_audit_render_html=bool(args.no_audit_render_html),
        interactive_artifacts=bool(args.interactive_artifacts),
    )
    require_complete_reasoning_steps = _resolve_require_complete_reasoning_steps(
        audit_all=bool(args.audit_all),
        render_html=bool(render_html),
        require_complete_reasoning_steps=bool(args.require_complete_reasoning_steps),
    )
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
    scale_stochastic = bool(scaling_cfg.get("stochastic_actions", False) or args.scale_stochastic_actions)
    scale_temperature = float(
        scaling_cfg.get("sampling_temperature", 1.0) if args.scale_sampling_temperature is None else args.scale_sampling_temperature
    )
    scale_exploration = float(
        scaling_cfg.get("sampling_exploration", 0.0) if args.scale_sampling_exploration is None else args.scale_sampling_exploration
    )
    scale_seed = int(scaling_cfg.get("sampling_seed", cfg.get("seed", 1729)) if args.scale_sampling_seed is None else args.scale_sampling_seed)
    x, y, gb, _ = collate_records(
        [rec],
        int(cfg.get("seq_len", 128)),
        tok,
        graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
        ar_seed=int(cfg.get("seed", 1729)),
    )
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
        "optional_outputs": {
            "audit_all": bool(args.audit_all),
            "audit_level": args.audit_level,
            "render_html": bool(render_html),
            "interactive_artifacts": bool(args.interactive_artifacts),
            "inference_scaling_enabled": bool(scale_depth > 0),
            "memory_bank": args.memory_bank,
            "memory_retrieve_top_k": int(args.memory_retrieve_top_k),
            "artifacts_dir": args.audit_output_dir,
            "require_complete_reasoning_steps": bool(require_complete_reasoning_steps),
        },
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
    if bool(mim_cfg.get("enabled", False)):
        mim_report = meet_in_middle_batch(
            model,
            [rec],
            tok,
            int(cfg.get("seq_len", 128)),
            device,
            graph_autoregressive=bool(cfg.get("graph_autoregressive_decoding", True)),
            seed=int(cfg.get("seed", 1729)),
            config=mim_cfg,
            forward_logits=out.get("logits"),
            forward_nll=out.get("nll"),
            require_grad=False,
        )
        result["meet_in_middle"] = {
            "enabled": True,
            "mode": mim_report.get("mode", ""),
            "shared_weight_reverse_pass": bool(mim_report.get("metrics", {}).get("mim_shared_weight_reverse_pass", 0.0)),
            "metrics": mim_report.get("metrics", {}),
            "records": mim_report.get("records", []),
            "note": "Shared-weight reverse-pass diagnostic unless a separately trained right-to-left checkpoint is configured.",
        }
    else:
        result["meet_in_middle"] = {"enabled": False}
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
            stochastic_actions=scale_stochastic,
            sampling_temperature=scale_temperature,
            sampling_exploration=scale_exploration,
            sampling_seed=scale_seed,
            require_complete_reasoning_steps=require_complete_reasoning_steps,
        )
    if args.memory_bank:
        bank = AnalogicalMemoryBank(args.memory_bank, max_records=args.memory_max_records)
        retrieved = []
        current_source = _inference_memory_source(args.prompt, scale_seed, args.audit_output_dir)
        if args.memory_retrieve_top_k > 0:
            embedding, signature = query_signature_from_report(result)
            query_topology = query_topology_from_report(result)
            landscape_weight = float(args.memory_landscape_weight if args.memory_landscape_weight is not None else cfg.get("inference_memory_landscape_weight", cfg.get("memory_retrieval_landscape_weight", 0.18)) or 0.0)
            retrieved = bank.retrieve(
                embedding,
                signature,
                top_k=args.memory_retrieve_top_k,
                exclude_sources={current_source},
                query_topology=query_topology,
                landscape_weight=landscape_weight,
            )
        added = 0
        if args.memory_save and isinstance(result.get("inference_scaling"), dict):
            records = memory_records_from_scaling_report(
                result["inference_scaling"],
                source=current_source,
                max_records=min(args.memory_max_records, 16),
            )
            bank.extend(records)
            bank.save()
            added = len(records)
            if args.memory_retrieve_top_k > 0 and not retrieved:
                embedding, signature = query_signature_from_report(result)
                query_topology = query_topology_from_report(result)
                landscape_weight = float(args.memory_landscape_weight if args.memory_landscape_weight is not None else cfg.get("inference_memory_landscape_weight", cfg.get("memory_retrieval_landscape_weight", 0.18)) or 0.0)
                retrieved = bank.retrieve(
                    embedding,
                    signature,
                    top_k=args.memory_retrieve_top_k,
                    exclude_sources={current_source},
                    exclude_memory_ids={record.memory_id for record in records},
                    query_topology=query_topology,
                    landscape_weight=landscape_weight,
                )
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
            render_html=render_html,
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
