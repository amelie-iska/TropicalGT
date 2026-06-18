#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tropicalgt.visualization import (  # noqa: E402
    _write_inference_dashboard,
    _write_reasoning_step_complex_maps,
    write_got_trajectory_visualization,
    write_toric_embedding_sidecar,
    write_tropical_fan_diagnostics,
    write_two_parameter_bifiltration_visualization,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _dashboard_artifact_paths(root: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "inference_audit.html":
            continue
        if path.suffix.lower() not in {".html", ".json"}:
            continue
        key = path.relative_to(root).as_posix().replace("/", "_").replace(".", "_")
        paths[key] = str(path)
    return paths




def _got_trajectory_contracts_need_backfill(root: Path) -> bool:
    embedding_payload = _read_json(root / "got_embedding_map_payloads.json")
    layout_contract = embedding_payload.get("layout_contract", {}) if isinstance(embedding_payload.get("layout_contract"), dict) else {}
    if layout_contract.get("schema_version") != "tropicalgt.embedding_trajectory_identity.v1":
        return True
    if layout_contract.get("no_proxy_or_fallback") is not True:
        return True
    density_payload = _read_json(root / "got_nll_density_cloud_payload.json")
    visual_contract = density_payload.get("visual_layer_contract", {}) if isinstance(density_payload.get("visual_layer_contract"), dict) else {}
    if visual_contract.get("schema_version") != "tropicalgt.nll_density_render.v1":
        return True
    if visual_contract.get("support_samples_are_model_states") is not False:
        return True
    if density_payload.get("sample_points_are_model_states") is True:
        return True
    required_sidecars = (
        root / "got_full_trajectory_complex_slider_contract.json",
        root / "got_full_trajectory_simplex_tree_3d_simplex_tree_poset_contract.json",
        root / "got_full_trajectory_complex_jensen_shannon_slider_contract.json",
        root / "got_full_trajectory_simplex_tree_3d_jensen_shannon_simplex_tree_poset_contract.json",
    )
    return any(not path.exists() for path in required_sidecars)


def _reasoning_step_contracts_need_backfill(root: Path) -> bool:
    manifest_path = root / "reasoning_step_complex_maps" / "manifest.json"
    if not manifest_path.exists():
        return True
    manifest = _read_json(manifest_path)
    contract = manifest.get("contract", {}) if isinstance(manifest.get("contract"), dict) else {}
    if contract.get("schema_version") != "tropicalgt.reasoning_step_complex_maps.v1":
        return True
    steps = manifest.get("steps", []) if isinstance(manifest.get("steps"), list) else []
    if not steps:
        return True
    directory = root / "reasoning_step_complex_maps"
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return True
        if step.get("step_complex_source_contract", {}).get("schema_version") != "tropicalgt.reasoning_step_complex_source_contract.v1":
            return True
        if step.get("radius_slider_contract", {}).get("schema_version") != "tropicalgt.reasoning_step_radius_slider_summary.v1":
            return True
        if step.get("simplex_tree_poset_contract", {}).get("schema_version") != "tropicalgt.simplex_tree_poset.v1":
            return True
        step_file = directory / str(step.get("file", f"reasoning_step_{index:03d}.html"))
        slider_file = directory / str(step.get("slider_contract_file") or f"{step_file.stem}_slider_contract.json")
        tree_file = directory / str(step.get("simplex_tree_file", f"reasoning_step_{index:03d}_simplex_tree.html"))
        poset_file = directory / str(step.get("simplex_tree_poset_contract_file") or f"{tree_file.stem}_simplex_tree_poset_contract.json")
        if not (step_file.exists() and slider_file.exists() and tree_file.exists() and poset_file.exists()):
            return True
    return False


def _dashboard_missing_links(root: Path, required_names: tuple[str, ...]) -> bool:
    dashboard = root / "inference_audit.html"
    existing_required = [name for name in required_names if (root / name).exists()]
    if not existing_required:
        return False
    if not dashboard.exists():
        return True
    try:
        markup = dashboard.read_text(encoding="utf-8")
    except Exception:
        return True
    return any(name not in markup for name in existing_required)


def backfill_audit_root(audit_root: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    root = Path(audit_root)
    actions: list[dict[str, Any]] = []
    if root.name == "got_audit":
        step_dir = root.parent
    elif (root / "got_audit").is_dir():
        step_dir = root
        root = root / "got_audit"
    else:
        step_dir = root.parent

    fan_json = root / "tropical_fan_diagnostics.json"
    fan_html = root / "tropical_fan_diagnostics.html"
    if overwrite or not (fan_json.exists() and fan_html.exists()):
        paths = write_tropical_fan_diagnostics({}, root)
        actions.append(
            {
                "kind": "tropical_fan_unavailable_backfill",
                "reason": "No explicit model-derived tropical ideal was available in this legacy audit bundle; wrote explicit unavailable diagnostics rather than a proxy fan.",
                "paths": paths,
            }
        )

    toric_json = root / "toric_embedding_sidecar.json"
    toric_html = root / "toric_embedding_sidecar.html"
    if overwrite or not (toric_json.exists() and toric_html.exists()):
        paths = write_toric_embedding_sidecar({}, root)
        actions.append(
            {
                "kind": "toric_embedding_sidecar_unavailable_backfill",
                "reason": "No explicit model-derived toric exponent matrix was available in this legacy audit bundle; wrote explicit unavailable finite toric-ideal sidecar diagnostics rather than a chart-bundle, support-token, GraphCG, embedding, or visualization proxy.",
                "paths": paths,
            }
        )


    scaling_path = root / "inference_scaling_tree.json"
    got_needed = overwrite or _got_trajectory_contracts_need_backfill(root)
    if got_needed and scaling_path.exists():
        scaling = _read_json(scaling_path)
        candidates = scaling.get("candidates", []) if isinstance(scaling.get("candidates"), list) else []
        if candidates:
            paths = write_got_trajectory_visualization(scaling, root)
            actions.append(
                {
                    "kind": "got_trajectory_contract_backfill",
                    "reason": "Regenerated GoT embedding/NLL/full-complex/reasoning-step visualization contracts from stored inference_scaling_tree.json candidates and their model outputs.",
                    "candidate_count": len(candidates),
                    "paths": paths,
                }
            )
        else:
            actions.append(
                {
                    "kind": "got_trajectory_contract_unavailable",
                    "reason": "GoT trajectory contracts were missing or stale, but inference_scaling_tree.json had no stored candidates; no embedding/NLL/full-complex contracts were fabricated.",
                    "paths": {"inference_scaling_tree": str(scaling_path)},
                }
            )

    reasoning_needed = overwrite or _reasoning_step_contracts_need_backfill(root)
    if reasoning_needed and scaling_path.exists():
        scaling = _read_json(scaling_path)
        candidates = scaling.get("candidates", []) if isinstance(scaling.get("candidates"), list) else []
        candidates_with_complex = [row for row in candidates if isinstance(row, dict) and isinstance(row.get("filtered_simplicial_object"), dict)]
        if candidates_with_complex:
            paths = _write_reasoning_step_complex_maps(candidates_with_complex, root)
            actions.append(
                {
                    "kind": "reasoning_step_complex_contract_backfill",
                    "reason": "Regenerated per-step complex pages, radius-slider contracts, SimplexTree poset contracts, source contracts, fingerprints, and manifest from stored inference_scaling_tree.json candidate filtered_simplicial_object payloads.",
                    "candidate_count": len(candidates_with_complex),
                    "paths": paths,
                }
            )
        else:
            actions.append(
                {
                    "kind": "reasoning_step_complex_contract_unavailable",
                    "reason": "Reasoning-step contracts were missing or stale, but inference_scaling_tree.json had no candidates with stored filtered_simplicial_object payloads; no per-step contracts were fabricated.",
                    "paths": {"inference_scaling_tree": str(scaling_path)},
                }
            )

    bif_raw = root / "trajectory_level_radius_bifiltration.json"
    bif_html = root / "trajectory_persistence" / "two_parameter_bifiltration.html"
    bif_sidecar = bif_html.with_suffix(".json")
    if bif_raw.exists() and (overwrite or not bif_sidecar.exists()):
        payload = _read_json(bif_raw)
        if payload:
            rendered = write_two_parameter_bifiltration_visualization(
                bif_html,
                payload,
                title="Trajectory 2-parameter persistence over F2[x_level,x_radius]",
            )
            actions.append(
                {
                    "kind": "two_parameter_bifiltration_visual_contract_backfill",
                    "reason": "Regenerated the bivariate staircase HTML/JSON contract from the raw trajectory_level_radius_bifiltration.json payload.",
                    "paths": {
                        "html": rendered,
                        "json": str(bif_sidecar),
                        "raw_bifiltration": str(bif_raw),
                    },
                }
            )
        else:
            actions.append(
                {
                    "kind": "two_parameter_bifiltration_visual_contract_unavailable",
                    "reason": "Raw trajectory_level_radius_bifiltration.json was present but could not be parsed as an object; no sidecar was fabricated.",
                    "paths": {"raw_bifiltration": str(bif_raw)},
                }
            )

    dashboard_needs_refresh = _dashboard_missing_links(
        root,
        (
            "tropical_fan_diagnostics.html",
            "toric_embedding_sidecar.html",
            "trajectory_persistence/two_parameter_bifiltration.html",
            "reasoning_step_complex_maps/manifest.json",
        ),
    )
    if actions or overwrite or dashboard_needs_refresh:
        dashboard_paths = _dashboard_artifact_paths(root)
        if dashboard_paths:
            dashboard_path = _write_inference_dashboard(dashboard_paths, root)
            actions.append(
                {
                    "kind": "inference_audit_dashboard_rebuilt",
                    "reason": "Rebuilt the local audit dashboard so explicit no-proxy sidecars are reachable from inference_audit.html.",
                    "paths": {"inference_audit": str(dashboard_path)},
                }
            )

    report = {
        "schema_version": "tropicalgt.interactive_audit_backfill.v1",
        "audit_root": str(root),
        "step_dir": str(step_dir),
        "overwrite": bool(overwrite),
        "actions": actions,
        "policy": "Backfills only explicit unavailable diagnostics or rerenders visual contracts from existing raw payloads; it does not fabricate CAS certificates, tropical fans, toric ideals, toric embeddings, normal fans, tropical-variety embeddings, or persistence modules.",
    }
    report_path = root / "backfill_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill legacy TropicalGT-I interactive audit bundles with explicit unavailable/visual-contract artifacts.")
    parser.add_argument("--audit-root", required=True, help="Path to a got_audit directory or its step directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args(argv)
    report = backfill_audit_root(args.audit_root, overwrite=args.overwrite)
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
