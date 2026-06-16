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

from tropicalgt.visualization import write_toric_embedding_sidecar, write_tropical_fan_diagnostics, write_two_parameter_bifiltration_visualization  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


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
