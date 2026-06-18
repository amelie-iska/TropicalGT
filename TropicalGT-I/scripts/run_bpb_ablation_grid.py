#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tropicalgt.ablation import write_bpb_ablation_artifacts
from tropicalgt.readiness_contracts import advanced_bpb_contract_report
from tropicalgt.run import load_config, train


CORE_AUX_ZERO_OVERRIDES: dict[str, Any] = {
    "model.gflownet_weight": 0.0,
    "model.graphcg_weight": 0.0,
    "model.margin_weight": 0.0,
    "model.entropy_weight": 0.0,
    "model.certificate_weight": 0.0,
    "model.sequence_tropical_weight": 0.0,
}

CHART_BUNDLE_ZERO_OVERRIDES: dict[str, Any] = {
    "model.enable_chart_bundle_auxiliary": True,
    "model.bundle_transport_weight": 0.0,
    "model.bundle_monomial_transport_weight": 0.0,
    "model.bundle_cocycle_weight": 0.0,
    "model.bundle_flat_rank_weight": 0.0,
    "model.bundle_flat_incidence_weight": 0.0,
    "model.toric_normal_fan_weight": 0.0,
    "model.graphcg_toric_cell_agreement_weight": 0.0,
    "model.chart_bpb_consistency_weight": 0.0,
    "model.bundle_atom_stability_weight": 0.0,
}

MEMORY_RETRIEVAL_ZERO_OVERRIDES: dict[str, Any] = {
    "memory_retrieval_landscape_weight": 0.0,
    "memory_retrieval_vector_weight": 0.0,
    "memory_retrieval_probability_map_weight": 0.0,
    "memory_retrieval_certified_cas_weight": 0.0,
}

VECTOR_BUNDLE_ABLATION_MATRIX = (
    "zero_auxiliary",
    "vector_bundle_telemetry_only",
    "vector_bundle_transport_only",
    "matroid_cone_only",
    "toric_graphcg_only",
    "memory_landscape_only",
    "chart_bpb_only",
    "vector_bundle_full_stack",
)


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "aux_0p5x": {
        "model.gflownet_weight": 0.01,
        "model.graphcg_weight": 0.01,
        "model.margin_weight": 0.001,
        "model.entropy_weight": 0.0005,
        "model.certificate_weight": 0.0005,
    },
    "aux_0p25x": {
        "model.gflownet_weight": 0.005,
        "model.graphcg_weight": 0.005,
        "model.margin_weight": 0.0005,
        "model.entropy_weight": 0.00025,
        "model.certificate_weight": 0.00025,
    },
    "gflownet_0p25x": {
        "model.gflownet_weight": 0.005,
        "model.graphcg_weight": 0.0,
        "model.margin_weight": 0.0,
        "model.entropy_weight": 0.0,
        "model.certificate_weight": 0.0,
    },
    "graphcg_0p25x": {
        "model.gflownet_weight": 0.0,
        "model.graphcg_weight": 0.005,
        "model.margin_weight": 0.0,
        "model.entropy_weight": 0.0,
        "model.certificate_weight": 0.0,
    },
    "tropical_0p25x": {
        "model.gflownet_weight": 0.0,
        "model.graphcg_weight": 0.0,
        "model.margin_weight": 0.0005,
        "model.entropy_weight": 0.00025,
        "model.certificate_weight": 0.00025,
    },
    "no_graphcg": {"model.graphcg_weight": 0.0},
    "no_gflownet": {"model.gflownet_weight": 0.0},
    "no_certificate": {"model.certificate_weight": 0.0},
    "no_tropical_regularizers": {
        "model.margin_weight": 0.0,
        "model.entropy_weight": 0.0,
        "model.certificate_weight": 0.0,
    },
    "no_auxiliary": {
        "model.gflownet_weight": 0.0,
        "model.graphcg_weight": 0.0,
        "model.margin_weight": 0.0,
        "model.entropy_weight": 0.0,
        "model.certificate_weight": 0.0,
    },
    "no_memory_bank": {"memory_bank_path": ""},
    "chart_bundle_telemetry": dict(CHART_BUNDLE_ZERO_OVERRIDES),
    "chart_bundle_toric_0p1x": {
        "model.enable_chart_bundle_auxiliary": True,
        "model.bundle_transport_weight": 0.0001,
        "model.bundle_monomial_transport_weight": 0.0001,
        "model.bundle_cocycle_weight": 0.0001,
        "model.bundle_flat_rank_weight": 0.0001,
        "model.bundle_flat_incidence_weight": 0.0001,
        "model.toric_normal_fan_weight": 0.0001,
        "model.graphcg_toric_cell_agreement_weight": 0.0001,
        "model.chart_bpb_consistency_weight": 0.0001,
        "model.bundle_atom_stability_weight": 0.0001,
    },
    "no_chart_bundle_toric": {
        "model.enable_chart_bundle_auxiliary": False,
        "model.bundle_transport_weight": 0.0,
        "model.bundle_monomial_transport_weight": 0.0,
        "model.bundle_cocycle_weight": 0.0,
        "model.bundle_flat_rank_weight": 0.0,
        "model.bundle_flat_incidence_weight": 0.0,
        "model.toric_normal_fan_weight": 0.0,
        "model.graphcg_toric_cell_agreement_weight": 0.0,
        "model.chart_bpb_consistency_weight": 0.0,
        "model.bundle_atom_stability_weight": 0.0,
    },
    "zero_auxiliary": {
        **CORE_AUX_ZERO_OVERRIDES,
        "model.enable_chart_bundle_auxiliary": False,
        **{key: value for key, value in CHART_BUNDLE_ZERO_OVERRIDES.items() if key != "model.enable_chart_bundle_auxiliary"},
        **MEMORY_RETRIEVAL_ZERO_OVERRIDES,
    },
    "vector_bundle_telemetry_only": {
        **CORE_AUX_ZERO_OVERRIDES,
        **CHART_BUNDLE_ZERO_OVERRIDES,
        **MEMORY_RETRIEVAL_ZERO_OVERRIDES,
    },
    "vector_bundle_transport_only": {
        **CORE_AUX_ZERO_OVERRIDES,
        **CHART_BUNDLE_ZERO_OVERRIDES,
        "model.bundle_transport_weight": 0.0001,
        "model.bundle_monomial_transport_weight": 0.0001,
        **MEMORY_RETRIEVAL_ZERO_OVERRIDES,
    },
    "matroid_cone_only": {
        **CORE_AUX_ZERO_OVERRIDES,
        **CHART_BUNDLE_ZERO_OVERRIDES,
        "model.bundle_flat_rank_weight": 0.0001,
        "model.bundle_flat_incidence_weight": 0.0001,
        "model.toric_normal_fan_weight": 0.0001,
        **MEMORY_RETRIEVAL_ZERO_OVERRIDES,
    },
    "toric_graphcg_only": {
        **CORE_AUX_ZERO_OVERRIDES,
        **CHART_BUNDLE_ZERO_OVERRIDES,
        "model.graphcg_weight": 0.005,
        "model.toric_normal_fan_weight": 0.0001,
        "model.graphcg_toric_cell_agreement_weight": 0.0001,
        **MEMORY_RETRIEVAL_ZERO_OVERRIDES,
    },
    "memory_landscape_only": {
        **CORE_AUX_ZERO_OVERRIDES,
        "model.enable_chart_bundle_auxiliary": False,
        **{key: value for key, value in CHART_BUNDLE_ZERO_OVERRIDES.items() if key != "model.enable_chart_bundle_auxiliary"},
        **MEMORY_RETRIEVAL_ZERO_OVERRIDES,
        "memory_retrieval_landscape_weight": 0.08,
    },
    "chart_bpb_only": {
        **CORE_AUX_ZERO_OVERRIDES,
        **CHART_BUNDLE_ZERO_OVERRIDES,
        "model.chart_bpb_consistency_weight": 0.0001,
        **MEMORY_RETRIEVAL_ZERO_OVERRIDES,
    },
    "vector_bundle_full_stack": {
        "model.enable_chart_bundle_auxiliary": True,
        "model.gflownet_weight": 0.005,
        "model.graphcg_weight": 0.005,
        "model.margin_weight": 0.0005,
        "model.entropy_weight": 0.00025,
        "model.certificate_weight": 0.00025,
        "model.sequence_tropical_weight": 0.0625,
        "model.bundle_transport_weight": 0.0001,
        "model.bundle_monomial_transport_weight": 0.0001,
        "model.bundle_cocycle_weight": 0.0001,
        "model.bundle_flat_rank_weight": 0.0001,
        "model.bundle_flat_incidence_weight": 0.0001,
        "model.toric_normal_fan_weight": 0.0001,
        "model.graphcg_toric_cell_agreement_weight": 0.0001,
        "model.chart_bpb_consistency_weight": 0.0001,
        "model.bundle_atom_stability_weight": 0.0001,
        "memory_retrieval_landscape_weight": 0.08,
        "memory_retrieval_vector_weight": 0.18,
        "memory_retrieval_probability_map_weight": 0.20,
        "memory_retrieval_certified_cas_weight": 0.14,
    },
}

VARIANT_GROUPS: dict[str, tuple[str, ...]] = {
    "vector_bundle_matrix": VECTOR_BUNDLE_ABLATION_MATRIX,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally run matched TropicalGT-I BPB ablations")
    parser.add_argument("--config", default=str(ROOT / "configs" / "gpu_smoke.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "bpb_ablation_grid"))
    parser.add_argument("--variants", default="baseline,vector_bundle_matrix,no_graphcg,no_gflownet,no_certificate,no_tropical_regularizers,no_auxiliary")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training steps for quick tests; omitted configs use --boundary-steps.")
    parser.add_argument("--boundary-steps", type=int, default=5000, help="Matched ablation boundary step; defaults to the 5K gate.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Actually train each generated variant")
    parser.add_argument("--fixture", action="store_true", help="Force fixture data for quick CPU/debug ablations")
    parser.add_argument("--device", choices=["auto", "cpu"], default=None)
    parser.add_argument("--wandb", action="store_true", help="Keep W&B enabled from the base config. Defaults to disabled for ablation grids.")
    parser.add_argument("--render-html", action="store_true", help="Render Plotly correlation report after running")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--allow-contract-breaking-ablation-configs",
        action="store_true",
        help="Permit analysis-only config emission for BPB-focused variants that fail the advanced BPB contract. These configs are never run by this script.",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    variant_names = _expand_variant_names([name.strip() for name in args.variants.split(",") if name.strip()])
    unknown = [name for name in variant_names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}; known={', '.join(sorted(VARIANTS))}")

    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    report_paths: list[str] = []
    configs = []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    boundary_steps = int(args.boundary_steps or 5000)
    requested_max_steps = int(args.max_steps) if args.max_steps is not None else boundary_steps
    match_group_id = f"{str(base.get('run_name', 'tropicalgt_i'))}_matched_{boundary_steps}_step_{run_id}"
    base_config_fingerprint = _stable_json_hash(_match_fingerprint_payload(base, boundary_steps))
    variant_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(variant_names):
        cfg = _variant_config(
            base,
            name,
            output_dir,
            run_id,
            VARIANTS[name],
            seed=args.seed,
            max_steps=args.max_steps,
            boundary_steps=boundary_steps,
            requested_max_steps=requested_max_steps,
            match_group_id=match_group_id,
            base_config_fingerprint=base_config_fingerprint,
            fixture=args.fixture,
            device=args.device,
            wandb=args.wandb,
        )
        contract_summary = _advanced_contract_summary(cfg)
        variant_rows.append({"idx": idx, "name": name, "cfg": cfg, "contract": contract_summary})

    contract_breaking = [row for row in variant_rows if row["contract"]["failed_gates"]]
    if contract_breaking and args.run:
        detail = "; ".join("{}={}".format(row["name"], ",".join(row["contract"]["failed_gates"])) for row in contract_breaking)
        raise SystemExit(f"advanced BPB contract failed before runnable ablation configs were written: {detail}")
    if contract_breaking and not args.allow_contract_breaking_ablation_configs:
        detail = "; ".join("{}={}".format(row["name"], ",".join(row["contract"]["failed_gates"])) for row in contract_breaking)
        raise SystemExit(
            "advanced BPB contract failed before ablation configs were written; "
            "pass --allow-contract-breaking-ablation-configs to emit analysis-only configs: "
            f"{detail}"
        )

    for row in variant_rows:
        idx = int(row["idx"])
        name = str(row["name"])
        cfg = row["cfg"]
        contract_summary = row["contract"]
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{idx:02d}_{name}.json"
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        configs.append(
            {
                "variant": name,
                "config": str(config_path),
                "overrides": VARIANTS[name],
                "contract_safe_to_run": not contract_summary["failed_gates"],
                "advanced_bpb_contract": contract_summary,
            }
        )
        if args.run:
            train(config_path, max_steps_override=args.max_steps)
            report_path = Path(cfg["output_dir"]) / "train_report.json"
            report_paths.append(str(report_path))

    manifest = {
        "base_config": str(args.config),
        "run_id": run_id,
        "ran_training": bool(args.run),
        "match_contract": {
            "schema_version": "tropicalgt.bpb_ablation_match_contract.v1",
            "match_group_id": match_group_id,
            "boundary_steps": boundary_steps,
            "requested_max_steps": requested_max_steps,
            "base_config_fingerprint": base_config_fingerprint,
            "variant_count": len(configs),
            "policy": "Matched BPB ablations must share seed, boundary, requested steps, data identity, graph BPB side weight, and base config fingerprint before any advanced auxiliary coefficient can be promoted.",
        },
        "vector_bundle_ablation_matrix": {
            "schema_version": "tropicalgt.vector_bundle_ablation_matrix.v1",
            "required_variants": list(VECTOR_BUNDLE_ABLATION_MATRIX),
            "emitted_variants": [name for name in variant_names if name in VECTOR_BUNDLE_ABLATION_MATRIX],
            "missing_variants": [name for name in VECTOR_BUNDLE_ABLATION_MATRIX if name not in variant_names],
            "policy": "Each matrix variant is config-only until matched 5K eval BPB and eval graph-BPB improve with certificate/tropical-wall guardrails; no advanced coefficient is promoted from telemetry or missing evidence.",
        },
        "variants": configs,
        "reports": report_paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ablation_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    paths = {"manifest": str(manifest_path)}
    if args.run and report_paths:
        paths.update(
            {
                f"analysis_{key}": value
                for key, value in write_bpb_ablation_artifacts(
                    report_paths,
                    output_dir / "analysis",
                    baseline=report_paths[0],
                    top_k=args.top_k,
                    render_html=args.render_html,
                ).items()
            }
        )
    print(json.dumps(paths, indent=2))


def _expand_variant_names(names: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for name in names:
        group = VARIANT_GROUPS.get(name)
        items = group if group is not None else (name,)
        for item in items:
            if item not in seen:
                expanded.append(item)
                seen.add(item)
    return expanded


def _variant_config(
    base: dict[str, Any],
    name: str,
    output_dir: Path,
    run_id: str,
    overrides: dict[str, Any],
    seed: int | None,
    max_steps: int | None,
    boundary_steps: int,
    requested_max_steps: int,
    match_group_id: str,
    base_config_fingerprint: str,
    fixture: bool,
    device: str | None,
    wandb: bool,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    base_name = str(base.get("run_name", "tropicalgt_i"))
    cfg["run_name"] = f"{base_name}_{name}_{run_id}"
    cfg["output_dir"] = str(output_dir / name)
    cfg["checkpoint_dir"] = str(output_dir / name / "checkpoints")
    if cfg.get("memory_bank_path"):
        cfg["memory_bank_path"] = str(output_dir / name / "analogical_memory" / "reasoning_memory.jsonl")
    if seed is not None:
        cfg["seed"] = int(seed)
    else:
        cfg["seed"] = int(cfg.get("seed", 1729))
    cfg["max_steps"] = int(max_steps) if max_steps is not None else int(boundary_steps)
    if fixture:
        cfg["data_root"] = None
        cfg["require_data"] = False
        cfg["train_limit"] = int(cfg.get("train_limit") or 8)
        cfg["val_limit"] = int(cfg.get("val_limit") or 4)
        cfg["fixture_size"] = max(int(cfg.get("fixture_size", 8)), cfg["train_limit"], cfg["val_limit"])
        cfg["chunk_shuffle"] = False
    if device:
        cfg["device"] = device
    if not wandb:
        cfg["wandb"] = {**dict(cfg.get("wandb", {})), "enabled": False}
    for dotted_key, value in overrides.items():
        _set_dotted(cfg, dotted_key, value)
    cfg["ablation_variant"] = name
    cfg["ablation_overrides"] = overrides
    cfg["ablation_match_contract"] = {
        "schema_version": "tropicalgt.bpb_ablation_match_contract.v1",
        "match_group_id": match_group_id,
        "variant": name,
        "baseline_variant": "baseline",
        "boundary_steps": int(boundary_steps),
        "requested_max_steps": int(requested_max_steps),
        "seed": int(cfg.get("seed", 0)),
        "base_config_fingerprint": base_config_fingerprint,
        "data_root": cfg.get("data_root"),
        "require_data": bool(cfg.get("require_data", False)),
        "train_limit": cfg.get("train_limit"),
        "val_limit": cfg.get("val_limit"),
        "graph_bpb_side_weight": float(cfg.get("graph_bpb_side_weight", 1.0) or 1.0),
        "policy": "No promotion from this run unless all candidate reports share these matched fields with baseline and pass held-out BPB, graph-BPB, certificate, and tropical-wall gates.",
    }
    return cfg


def _match_fingerprint_payload(base: dict[str, Any], boundary_steps: int) -> dict[str, Any]:
    excluded = {"run_name", "output_dir", "checkpoint_dir", "memory_bank_path", "wandb_name", "wandb_run_name", "ablation_variant", "ablation_overrides"}
    payload = {key: value for key, value in base.items() if key not in excluded}
    payload["boundary_steps"] = int(boundary_steps)
    return payload


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()


def _set_dotted(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = cfg
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _advanced_contract_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    section, gates = advanced_bpb_contract_report(cfg)
    failed = [str(gate.get("name", "")) for gate in gates if gate.get("status") == "fail"]
    return {
        "required": bool(section.get("required", False)),
        "failed_gates": failed,
        "gate_count": len(gates),
        "policy": section.get("policy", ""),
    }


if __name__ == "__main__":
    main()
