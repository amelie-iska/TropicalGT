#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shlex
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "unavailable"
    return str(value)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _json_output_paths(command: str) -> list[Path]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    paths: list[Path] = []
    for index, part in enumerate(parts):
        if part == "--json-output" and index + 1 < len(parts):
            paths.append(Path(parts[index + 1]))
        elif part.startswith("--json-output="):
            paths.append(Path(part.split("=", 1)[1]))
    return paths


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _validator_gap_evidence(bundle: dict[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    combined: dict[str, int] = {}
    ranked_categories: dict[str, dict[str, Any]] = {}
    top_examples: list[dict[str, Any]] = []
    command_results = [row for row in bundle.get("command_results", []) if isinstance(row, dict)]
    for row in command_results:
        name = str(row.get("name", ""))
        command = str(row.get("command", ""))
        if "interactive_audit_validator" not in name and "validate_interactive_audit_artifacts.py" not in command:
            continue
        paths = _json_output_paths(command)
        if not paths:
            sources.append(
                {
                    "name": name,
                    "available": False,
                    "reason": "validator_command_lacks_json_output_path",
                    "returncode": row.get("returncode"),
                    "timed_out": bool(row.get("timed_out", False)),
                }
            )
            continue
        for path in paths:
            resolved = _resolve_output_path(path)
            source: dict[str, Any] = {
                "name": name,
                "path": _project_path(resolved),
                "available": False,
                "returncode": row.get("returncode"),
                "timed_out": bool(row.get("timed_out", False)),
            }
            if not resolved.exists():
                source["reason"] = "validator_json_output_missing"
                sources.append(source)
                continue
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - parser message is platform-dependent
                source["reason"] = f"validator_json_parse_error:{exc}"
                sources.append(source)
                continue
            inventory = payload.get("evidence_gap_inventory", {}) if isinstance(payload, dict) else {}
            categories = inventory.get("categories", []) if isinstance(inventory.get("categories"), list) else []
            category_counts = inventory.get("category_counts", {}) if isinstance(inventory.get("category_counts"), dict) else {}
            for category, count in category_counts.items():
                try:
                    combined[str(category)] = combined.get(str(category), 0) + int(count)
                except (TypeError, ValueError):
                    continue
            handled_categories: set[str] = set()
            for category_row in categories:
                if not isinstance(category_row, dict):
                    continue
                category = str(category_row.get("category", "other") or "other")
                handled_categories.add(category)
                try:
                    count_value = int(category_row.get("count", category_counts.get(category, 0)) or 0)
                except (TypeError, ValueError):
                    count_value = 0
                required_action = str(category_row.get("required_action", "") or "")
                examples_raw = category_row.get("examples", [])
                examples = [str(item) for item in examples_raw if str(item)] if isinstance(examples_raw, list) else []
                aggregate = ranked_categories.setdefault(
                    category,
                    {"category": category, "count": 0, "required_action": required_action, "examples": [], "source_names": []},
                )
                aggregate["count"] = int(aggregate.get("count", 0) or 0) + count_value
                if required_action and not aggregate.get("required_action"):
                    aggregate["required_action"] = required_action
                source_names = aggregate.setdefault("source_names", [])
                if isinstance(source_names, list) and name and name not in source_names:
                    source_names.append(name)
                for example in examples[:5]:
                    example_row = {"source": name, "path": source.get("path", ""), "example": example}
                    category_examples = aggregate.setdefault("examples", [])
                    if isinstance(category_examples, list) and len(category_examples) < 8:
                        category_examples.append(example_row)
                    if len(top_examples) < 24:
                        top_examples.append({"category": category, "required_action": required_action, **example_row})
            for category, count in category_counts.items():
                category_name = str(category)
                if category_name in handled_categories:
                    continue
                try:
                    count_value = int(count)
                except (TypeError, ValueError):
                    count_value = 0
                aggregate = ranked_categories.setdefault(
                    category_name,
                    {"category": category_name, "count": 0, "required_action": "", "examples": [], "source_names": []},
                )
                aggregate["count"] = int(aggregate.get("count", 0) or 0) + count_value
            source.update(
                {
                    "available": bool(inventory),
                    "validator_ok": bool(payload.get("ok")) if isinstance(payload, dict) else False,
                    "error_count": len(payload.get("errors", [])) if isinstance(payload, dict) and isinstance(payload.get("errors"), list) else None,
                    "gap_count": inventory.get("gap_count"),
                    "category_counts": category_counts,
                    "categories": categories[:16],
                    "policy": inventory.get("policy", ""),
                }
            )
            if not inventory:
                source["reason"] = "validator_json_lacks_evidence_gap_inventory"
            sources.append(source)
    ranked_category_rows = sorted(
        ranked_categories.values(), key=lambda row: (-int(row.get("count", 0) or 0), str(row.get("category", "")))
    )
    return {
        "schema_version": "tropicalgt.herschel_validator_gap_evidence.v1",
        "available": any(source.get("available") for source in sources),
        "source_count": len(sources),
        "combined_category_counts": {category: count for category, count in sorted(combined.items())},
        "ranked_categories": ranked_category_rows,
        "top_examples": top_examples,
        "sources": sources,
        "policy": "Herschel reads validator gap inventories only from recorded validator JSON outputs; missing JSON is unavailable and does not justify a restart or artifact pass.",
    }


def _sidecar_groups(paths: list[str]) -> dict[str, int]:
    groups = {
        "cas_algebra": 0,
        "topology_persistence": 0,
        "analogical_memory": 0,
        "tropical_toric": 0,
        "graphcg": 0,
        "gflownet": 0,
        "nll_density": 0,
        "chart_bundle": 0,
        "vector_bundle": 0,
        "sheaf_derived": 0,
        "other": 0,
    }
    for path in paths:
        lower = path.lower()
        matched = False
        if any(term in lower for term in ("cas", "betti", "fitting", "minor", "free_resolution", "buchsbaum", "be_", "certificate_indexed")):
            groups["cas_algebra"] += 1
            matched = True
        if any(term in lower for term in ("persistence", "bifiltration", "simplex", "barcode", "landscape", "topology", "topological")):
            groups["topology_persistence"] += 1
            matched = True
        if "analogical" in lower or "memory" in lower:
            groups["analogical_memory"] += 1
            matched = True
        if any(term in lower for term in ("tropical", "toric", "fan")):
            groups["tropical_toric"] += 1
            matched = True
        if "graphcg" in lower:
            groups["graphcg"] += 1
            matched = True
        if any(term in lower for term in ("gflownet", "inference_scaling_tree", "branch_selection", "action_selection")):
            groups["gflownet"] += 1
            matched = True
        if "nll" in lower or "density" in lower:
            groups["nll_density"] += 1
            matched = True
        if "chart_bundle" in lower or "bundle" in lower:
            groups["chart_bundle"] += 1
            matched = True
        if any(term in lower for term in ("vector_bundle", "vector-bundle", "chart_bundle_transport_sidecar", "bundle_toric", "paper_sidecar")):
            groups["vector_bundle"] += 1
            matched = True
        if any(term in lower for term in ("sheaf", "derived", "chain_map", "derived_category")):
            groups["sheaf_derived"] += 1
            matched = True
        if not matched:
            groups["other"] += 1
    return groups



def _trajectory_embedding_visual_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    coordinate_source_counts: dict[str, int] = {}
    total_trajectory_nodes = 0
    total_embedding_nodes = 0
    total_edges = 0
    total_filtered_objects = 0
    total_raw_embeddings = 0
    total_parent_child_transitions = 0
    pca_quality_warning_count = 0

    def _count(bucket: dict[str, int], key: str) -> None:
        if key:
            bucket[key] = bucket.get(key, 0) + 1

    def _pca_ok(diag: dict[str, Any], node_count: int) -> tuple[bool, float | None, float | None, str]:
        source = str(diag.get("coordinate_source", "unavailable"))
        corr = _optional_float(diag.get("pairwise_distance_correlation"))
        stress = _optional_float(diag.get("normalized_stress"))
        samples = _optional_int(diag.get("n_samples"))
        ok = bool(source == "model graph_state embeddings" and samples == node_count and corr is not None and corr >= 0.75 and stress is not None and stress <= 0.75)
        return ok, corr, stress, source

    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if not (lower.endswith("got_trajectory_payloads.json") or lower.endswith("got_embedding_map_payloads.json")):
            continue
        kind = "trajectory_payload" if lower.endswith("got_trajectory_payloads.json") else "embedding_map_payload"
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "kind": kind, "available": False}
        if not resolved.exists():
            source["reason"] = f"{kind}_missing"
            _count(status_counts, source["reason"])
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser message is platform-dependent
            source["reason"] = f"{kind}_parse_error:{exc}"
            _count(status_counts, f"{kind}_parse_error")
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = f"{kind}_not_object"
            _count(status_counts, source["reason"])
            sources.append(source)
            continue
        nodes = [row for row in payload.get("nodes", []) if isinstance(row, dict)] if isinstance(payload.get("nodes"), list) else []
        edges = [row for row in payload.get("edges", []) if isinstance(row, dict)] if isinstance(payload.get("edges"), list) else []
        filtered_objects = [row for row in payload.get("filtered_simplicial_objects", []) if isinstance(row, dict)] if isinstance(payload.get("filtered_simplicial_objects"), list) else []
        pca_diag = payload.get("embedding_pca_diagnostics") if isinstance(payload.get("embedding_pca_diagnostics"), dict) else {}
        pca_ok, pca_corr, pca_stress, coordinate_source = _pca_ok(pca_diag, len(nodes))
        raw_embedding_count = sum(1 for row in nodes if isinstance(row.get("embedding"), list) and bool(row.get("embedding")))
        pca_node_count = sum(1 for row in nodes if isinstance(row.get("embedding_pca"), dict) and all(_optional_float(row["embedding_pca"].get(axis)) is not None for axis in ("pc1", "pc2", "pc3")))
        model_embedding_source_count = sum(1 for row in nodes if row.get("embedding_source") in ("model graph_state", None) and isinstance(row.get("embedding"), list) and bool(row.get("embedding")))
        node_count = len(nodes)
        edge_count = len(edges)
        filtered_count = len(filtered_objects)
        duplicate_count = _optional_int(pca_diag.get("duplicate_pca_coordinates_rounded8"))
        unique_ratio = _optional_float(pca_diag.get("unique_embedding_ratio_rounded8"))
        pca_warning = bool((duplicate_count or 0) > 0 or (unique_ratio is not None and unique_ratio < 0.75))
        common_ok = bool(node_count > 0 and edge_count >= max(0, node_count - 1) and filtered_count == node_count and raw_embedding_count == node_count and pca_ok)
        reasons: list[str] = []
        if not common_ok:
            if node_count <= 0:
                reasons.append("missing_visual_nodes")
            if edge_count < max(0, node_count - 1):
                reasons.append("parent_child_edge_count_too_small")
            if filtered_count != node_count:
                reasons.append("filtered_object_count_mismatch")
            if raw_embedding_count != node_count:
                reasons.append("missing_raw_model_embeddings")
            if not pca_ok:
                reasons.append("invalid_model_graph_state_pca_diagnostics")
        if kind == "trajectory_payload":
            surface = payload.get("nll_surface") if isinstance(payload.get("nll_surface"), dict) else {}
            progress = payload.get("nll_progress") if isinstance(payload.get("nll_progress"), dict) else {}
            surface_ok = bool(surface.get("available") is True and surface.get("touches_points") is True)
            progress_ok = isinstance(progress, dict)
            available = bool(common_ok and surface_ok and progress_ok)
            if not surface_ok:
                reasons.append("trajectory_nll_surface_not_point_anchored")
            source.update(
                {
                    "status": "trajectory_embedding_visual_available" if available else "trajectory_payload_unavailable",
                    "available": available,
                    "coordinate_source": coordinate_source,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "filtered_simplicial_object_count": filtered_count,
                    "raw_model_embedding_count": raw_embedding_count,
                    "pca_node_count": pca_node_count,
                    "pca_pairwise_distance_correlation": pca_corr,
                    "pca_normalized_stress": pca_stress,
                    "duplicate_pca_coordinate_count": duplicate_count or 0,
                    "unique_embedding_ratio": unique_ratio,
                    "pca_quality_warning": pca_warning,
                    "nll_surface_available": surface.get("available"),
                    "nll_surface_touches_points": surface.get("touches_points"),
                    "model_embedding_source_verified": model_embedding_source_count == node_count,
                }
            )
            if available:
                total_trajectory_nodes += node_count
        else:
            layout = payload.get("layout_contract") if isinstance(payload.get("layout_contract"), dict) else {}
            layout_ok = bool(
                payload.get("coordinate_source", "").startswith("PCA of model graph_state embeddings")
                and layout.get("schema_version") == "tropicalgt.embedding_trajectory_identity.v1"
                and layout.get("coordinate_source") == "model graph_state embeddings"
                and layout.get("branch_depth_metadata_present") is True
                and layout.get("parent_child_transitions_present") is True
                and layout.get("edge_source") == "graph_of_thought_parent_edges"
                and layout.get("node_embedding_source") == "model graph_state"
                and layout.get("geometric_separation_overclaim_allowed") is False
                and layout.get("no_proxy_or_fallback") is True
                and (_optional_int(layout.get("parent_child_transition_count")) or 0) == edge_count
            )
            available = bool(common_ok and layout_ok)
            if not layout_ok:
                reasons.append("missing_embedding_trajectory_identity_contract")
            source.update(
                {
                    "status": "trajectory_embedding_visual_available" if available else "embedding_map_payload_unavailable",
                    "available": available,
                    "coordinate_source": coordinate_source,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "filtered_simplicial_object_count": filtered_count,
                    "raw_model_embedding_count": raw_embedding_count,
                    "pca_node_count": pca_node_count,
                    "pca_pairwise_distance_correlation": pca_corr,
                    "pca_normalized_stress": pca_stress,
                    "duplicate_pca_coordinate_count": duplicate_count or 0,
                    "unique_embedding_ratio": unique_ratio,
                    "pca_quality_warning": bool(layout.get("pca_quality_warning") or pca_warning),
                    "layout_contract_schema": layout.get("schema_version", "unavailable"),
                    "parent_child_transition_count": _optional_int(layout.get("parent_child_transition_count")) or 0,
                    "geometric_separation_overclaim_allowed": layout.get("geometric_separation_overclaim_allowed"),
                    "no_proxy_or_fallback": layout.get("no_proxy_or_fallback") is True,
                    "sampling": payload.get("sampling", {}) if isinstance(payload.get("sampling"), dict) else {},
                }
            )
            if available:
                total_embedding_nodes += node_count
                total_parent_child_transitions += _optional_int(layout.get("parent_child_transition_count")) or 0
        status = str(source.get("status", "trajectory_embedding_visual_unavailable"))
        _count(status_counts, status)
        _count(coordinate_source_counts, coordinate_source)
        if source.get("available"):
            total_edges += edge_count
            total_filtered_objects += filtered_count
            total_raw_embeddings += raw_embedding_count
            if source.get("pca_quality_warning"):
                pca_quality_warning_count += 1
        else:
            source["reason"] = ";".join(reasons) or f"{kind}_unavailable"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    paired_payloads_available = bool(
        any(row.get("available") and row.get("kind") == "trajectory_payload" for row in sources)
        and any(row.get("available") and row.get("kind") == "embedding_map_payload" for row in sources)
    )
    return {
        "schema_version": "tropicalgt.herschel_trajectory_embedding_visual_evidence.v1",
        "available": bool(available_sources),
        "paired_payloads_available": paired_payloads_available,
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "total_trajectory_node_count": total_trajectory_nodes,
        "total_embedding_node_count": total_embedding_nodes,
        "total_edge_count": total_edges,
        "total_filtered_simplicial_object_count": total_filtered_objects,
        "total_raw_model_embedding_count": total_raw_embeddings,
        "total_parent_child_transition_count": total_parent_child_transitions,
        "pca_quality_warning_source_count": pca_quality_warning_count,
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "coordinate_source_counts": {key: coordinate_source_counts[key] for key in sorted(coordinate_source_counts)},
        "sources": sources,
        "policy": "Herschel reports GoT trajectory and embedding-map visuals only from recorded payloads with raw model graph_state embeddings, PCA diagnostics, parent-child GoT metadata, filtered simplicial objects, and the embedding trajectory identity contract. PCA duplicate-coordinate warnings are diagnostics, not geometric-separation claims.",
    }


def _inference_audit_backfill_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    action_kind_counts: dict[str, int] = {}
    total_inference_candidates = 0
    total_backfill_actions = 0
    total_backfill_candidate_count = 0
    total_quality_gate_candidates = 0
    total_quality_gate_eligible = 0
    total_quality_gate_rejected = 0
    total_records_added = 0

    def _count(bucket: dict[str, int], key: str) -> None:
        if key:
            bucket[key] = bucket.get(key, 0) + 1

    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if not (lower.endswith("inference_audit.json") or lower.endswith("backfill_report.json") or lower.endswith("backfill_report_latest.json")):
            continue
        kind = "inference_audit" if lower.endswith("inference_audit.json") else "interactive_backfill_report"
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "kind": kind, "available": False}
        if not resolved.exists():
            source["reason"] = f"{kind}_missing"
            _count(status_counts, source["reason"])
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser message is platform-dependent
            source["reason"] = f"{kind}_parse_error:{exc}"
            _count(status_counts, f"{kind}_parse_error")
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = f"{kind}_not_object"
            _count(status_counts, source["reason"])
            sources.append(source)
            continue
        if kind == "inference_audit":
            scaling = payload.get("inference_scaling") if isinstance(payload.get("inference_scaling"), dict) else {}
            topology = payload.get("topological_algebra") if isinstance(payload.get("topological_algebra"), dict) else {}
            budget = payload.get("periodic_got_scaling_budget") if isinstance(payload.get("periodic_got_scaling_budget"), dict) else {}
            memory = payload.get("analogical_memory_retrieval") if isinstance(payload.get("analogical_memory_retrieval"), dict) else {}
            quality_gate = memory.get("quality_gate") if isinstance(memory.get("quality_gate"), dict) else {}
            candidates = scaling.get("candidates") if isinstance(scaling.get("candidates"), list) else []
            best = scaling.get("best") if isinstance(scaling.get("best"), dict) else {}
            candidate_count = len([row for row in candidates if isinstance(row, dict)])
            if candidate_count == 0 and isinstance(best, dict) and best:
                candidate_count = 1
            topology_available = bool(topology)
            memory_retrieved = memory.get("retrieved") if isinstance(memory.get("retrieved"), list) else []
            gate_candidates = _optional_int(quality_gate.get("candidate_count")) or 0
            gate_eligible = _optional_int(quality_gate.get("eligible_count")) or 0
            gate_rejected = _optional_int(quality_gate.get("rejected_count")) or 0
            records_added = _optional_int(memory.get("records_added")) or 0
            available = bool(scaling and candidate_count > 0)
            status = "inference_audit_available" if available else "inference_audit_unavailable"
            _count(status_counts, status)
            if available:
                total_inference_candidates += candidate_count
                total_quality_gate_candidates += gate_candidates
                total_quality_gate_eligible += gate_eligible
                total_quality_gate_rejected += gate_rejected
                total_records_added += records_added
            source.update(
                {
                    "available": available,
                    "status": status,
                    "audit_seed_record_id": payload.get("audit_seed_record_id", "unavailable"),
                    "candidate_count": candidate_count,
                    "best_record_id": best.get("record_id", "unavailable"),
                    "stochastic_actions": scaling.get("stochastic_actions"),
                    "allow_stop": scaling.get("allow_stop"),
                    "topological_algebra_available": topology_available,
                    "periodic_got_scaling_budget_keys": sorted(str(key) for key in budget.keys()),
                    "analogical_memory_bank_size": _optional_int(memory.get("bank_size")) or 0,
                    "analogical_memory_retrieved_count": len(memory_retrieved),
                    "quality_gate_candidate_count": gate_candidates,
                    "quality_gate_eligible_count": gate_eligible,
                    "quality_gate_rejected_count": gate_rejected,
                    "records_added": records_added,
                    "quality_gate_reason_counts": quality_gate.get("reason_counts", {}) if isinstance(quality_gate.get("reason_counts"), dict) else {},
                }
            )
            if not available:
                source["reason"] = "missing_inference_scaling_candidates"
        else:
            actions = [row for row in payload.get("actions", []) if isinstance(row, dict)] if isinstance(payload.get("actions"), list) else []
            policy = str(payload.get("policy", ""))
            schema_ok = payload.get("schema_version") == "tropicalgt.interactive_audit_backfill.v1"
            policy_ok = "does not fabricate" in policy and "Backfills only explicit unavailable diagnostics" in policy
            action_count = len(actions)
            candidate_count = sum(_optional_int(row.get("candidate_count")) or 0 for row in actions)
            for row in actions:
                _count(action_kind_counts, str(row.get("kind", "unknown_backfill_action")))
            available = bool(schema_ok and policy_ok)
            status = "interactive_backfill_report_available" if available else "interactive_backfill_report_unavailable"
            _count(status_counts, status)
            if available:
                total_backfill_actions += action_count
                total_backfill_candidate_count += candidate_count
            source.update(
                {
                    "available": available,
                    "status": status,
                    "schema_version": payload.get("schema_version", "unavailable"),
                    "overwrite": payload.get("overwrite"),
                    "action_count": action_count,
                    "candidate_count": candidate_count,
                    "action_kinds": [str(row.get("kind", "unknown_backfill_action")) for row in actions],
                    "policy_no_fabrication": policy_ok,
                    "policy": policy,
                }
            )
            if not available:
                reasons = []
                if not schema_ok:
                    reasons.append("missing_interactive_backfill_schema")
                if not policy_ok:
                    reasons.append("missing_backfill_no_fabrication_policy")
                source["reason"] = ";".join(reasons) or "interactive_backfill_report_unavailable"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_inference_audit_backfill_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "inference_audit_available_source_count": sum(1 for row in available_sources if row.get("kind") == "inference_audit"),
        "backfill_available_source_count": sum(1 for row in available_sources if row.get("kind") == "interactive_backfill_report"),
        "total_inference_candidate_count": total_inference_candidates,
        "total_backfill_action_count": total_backfill_actions,
        "total_backfill_candidate_count": total_backfill_candidate_count,
        "total_quality_gate_candidate_count": total_quality_gate_candidates,
        "total_quality_gate_eligible_count": total_quality_gate_eligible,
        "total_quality_gate_rejected_count": total_quality_gate_rejected,
        "total_records_added": total_records_added,
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "action_kind_counts": {key: action_kind_counts[key] for key in sorted(action_kind_counts)},
        "sources": sources,
        "policy": "Herschel reports inference_audit as raw recorded inference evidence and interactive backfill reports as rendering/provenance repair logs only. Backfill actions prove rerendering from existing payloads or explicit unavailable diagnostics, never model quality, CAS certificates, tropical fans, toric embeddings, normal fans, or persistence modules.",
    }

def _gflownet_branch_selection_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    policy_counts: dict[str, int] = {}
    total_rows = 0
    total_actions = 0
    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if not any(term in lower for term in ("inference_scaling_tree", "branch_selection", "action_selection", "gflownet")):
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "gflownet_branch_selection_sidecar_missing"
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse message is platform-dependent
            source["reason"] = f"gflownet_branch_selection_sidecar_parse_error:{exc}"
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "gflownet_branch_selection_sidecar_not_object"
            sources.append(source)
            continue
        levels = payload.get("levels", []) if isinstance(payload.get("levels"), list) else []
        rows: list[dict[str, Any]] = []
        for level in levels:
            if not isinstance(level, dict):
                continue
            branch_rows = level.get("branch_selection", []) if isinstance(level.get("branch_selection"), list) else []
            rows.extend(row for row in branch_rows if isinstance(row, dict))
        if not rows and isinstance(payload.get("branch_selection"), list):
            rows = [row for row in payload.get("branch_selection", []) if isinstance(row, dict)]
        valid_rows: list[dict[str, Any]] = []
        source_policy_counts: dict[str, int] = {}
        source_selected_actions = 0
        for row in rows:
            contract = row.get("action_selection_contract") if isinstance(row.get("action_selection_contract"), dict) else {}
            selected_actions = row.get("selected_actions", []) if isinstance(row.get("selected_actions"), list) else []
            row_schema_ok = row.get("schema_version") == "tropicalgt.gflownet_branch_selection_audit.v1"
            contract_schema_ok = contract.get("schema_version") == "tropicalgt.gflownet_action_selection_contract.v1"
            no_proxy_ok = bool(row.get("no_proxy_or_fallback") is True and contract.get("no_proxy_or_fallback") is True)
            real_prob_ok = bool(contract.get("selected_from_real_model_action_probabilities") is True)
            policy = str(contract.get("selection_policy", "unavailable"))
            source_selected_actions += len(selected_actions)
            if row_schema_ok and contract_schema_ok and no_proxy_ok and real_prob_ok:
                valid_rows.append(row)
                source_policy_counts[policy] = source_policy_counts.get(policy, 0) + 1
                policy_counts[policy] = policy_counts.get(policy, 0) + 1
        total_rows += len(valid_rows)
        total_actions += source_selected_actions
        source.update(
            {
                "available": bool(valid_rows),
                "branch_selection_row_count": len(rows),
                "valid_branch_selection_row_count": len(valid_rows),
                "selected_action_count": source_selected_actions,
                "policy_counts": source_policy_counts,
                "required_branch_schema": "tropicalgt.gflownet_branch_selection_audit.v1",
                "required_action_contract_schema": "tropicalgt.gflownet_action_selection_contract.v1",
                "no_proxy_or_fallback": bool(valid_rows)
                and all(
                    row.get("no_proxy_or_fallback") is True
                    and isinstance(row.get("action_selection_contract"), dict)
                    and row["action_selection_contract"].get("no_proxy_or_fallback") is True
                    for row in valid_rows
                ),
            }
        )
        if not rows:
            source["reason"] = "gflownet_branch_selection_rows_missing"
        elif not valid_rows:
            source["reason"] = "gflownet_branch_selection_contract_missing_or_unsafe"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_gflownet_branch_selection_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "required_branch_schema": "tropicalgt.gflownet_branch_selection_audit.v1",
        "required_action_contract_schema": "tropicalgt.gflownet_action_selection_contract.v1",
        "total_valid_branch_selection_rows": total_rows,
        "total_selected_actions": total_actions,
        "policy_counts": {key: policy_counts[key] for key in sorted(policy_counts)},
        "sources": sources,
        "policy": "Herschel reports GFlowNet branch-selection evidence only from recorded inference_scaling_tree sidecars; missing or unsafe contracts stay unavailable and cannot justify restart decisions.",
    }


def _tropical_support_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    support_probability_source_counts: dict[str, int] = {}
    interpretation_status_counts: dict[str, int] = {}
    total_token_count = 0
    total_valid_assignments = 0
    total_invalid_assignments = 0
    for raw_path in sidecar_paths:
        if "tropical_support_payload" not in raw_path.lower():
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "tropical_support_sidecar_missing"
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse message is platform-dependent
            source["reason"] = f"tropical_support_sidecar_parse_error:{exc}"
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "tropical_support_sidecar_not_object"
            sources.append(source)
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        render_contract = payload.get("tropical_support_render_contract") if isinstance(payload.get("tropical_support_render_contract"), dict) else {}
        readability_contract = payload.get("tropical_support_readability_contract") if isinstance(payload.get("tropical_support_readability_contract"), dict) else {}
        wall_audit = metrics.get("wall_margin_audit") if isinstance(metrics.get("wall_margin_audit"), dict) else {}
        render_schema = str(render_contract.get("schema_version") or metrics.get("render_contract_schema_version") or "unavailable")
        readability_schema = str(readability_contract.get("schema_version") or metrics.get("readability_contract_schema_version") or "unavailable")
        support_probability_source = str(render_contract.get("support_probability_source") or metrics.get("support_probability_source") or "unavailable")
        token_count = _optional_int(render_contract.get("token_count", metrics.get("token_count")))
        observed_support_count = _optional_int(render_contract.get("observed_support_count", metrics.get("unique_support_count")))
        valid_assignments = _optional_int(render_contract.get("valid_support_assignment_count", metrics.get("valid_support_assignment_count")))
        invalid_assignments = _optional_int(render_contract.get("invalid_support_count", metrics.get("invalid_support_count")))
        strict_wall_hit_rate = _optional_float(metrics.get("strict_wall_hit_rate", wall_audit.get("strict_wall_hit_rate")))
        near_wall_hit_rate = _optional_float(metrics.get("near_wall_hit_rate", wall_audit.get("near_wall_hit_rate")))
        near_wall_only_rate = _optional_float(metrics.get("near_wall_only_rate", wall_audit.get("near_wall_only_rate")))
        low_strict_status = str(wall_audit.get("low_strict_wall_interpretation_status") or "unavailable")
        no_proxy_ok = bool(
            metrics.get("no_proxy_or_fallback") is True
            and render_contract.get("no_proxy_or_fallback") is True
            and readability_contract.get("no_proxy_or_fallback") is True
        )
        schema_ok = render_schema == "tropicalgt.tropical_support_render.v1"
        readability_ok = readability_schema == "tropicalgt.tropical_support_readability.v1"
        probability_ok = support_probability_source == "model_tropical_support_probabilities"
        wall_rates_ok = strict_wall_hit_rate is not None and near_wall_hit_rate is not None
        assignment_ok = bool((token_count or 0) > 0 and (valid_assignments or 0) > 0)
        available = bool(schema_ok and readability_ok and no_proxy_ok and probability_ok and wall_rates_ok and assignment_ok)
        if available:
            support_probability_source_counts[support_probability_source] = support_probability_source_counts.get(support_probability_source, 0) + 1
            interpretation_status_counts[low_strict_status] = interpretation_status_counts.get(low_strict_status, 0) + 1
            total_token_count += int(token_count or 0)
            total_valid_assignments += int(valid_assignments or 0)
            total_invalid_assignments += int(invalid_assignments or 0)
        source.update(
            {
                "available": available,
                "render_contract_schema_version": render_schema,
                "readability_contract_schema_version": readability_schema,
                "support_probability_source": support_probability_source,
                "token_count": token_count,
                "observed_support_count": observed_support_count,
                "valid_support_assignment_count": valid_assignments,
                "invalid_support_count": invalid_assignments,
                "top_support_collapse_rate": _optional_float(metrics.get("top_support_collapse_rate")),
                "effective_supports": _optional_float(metrics.get("effective_supports")),
                "support_entropy_bits": _optional_float(metrics.get("support_entropy_bits")),
                "strict_wall_hit_rate": strict_wall_hit_rate,
                "near_wall_hit_rate": near_wall_hit_rate,
                "near_wall_only_rate": near_wall_only_rate,
                "wall_margin_threshold": _optional_float(metrics.get("wall_margin_threshold", wall_audit.get("wall_margin_threshold"))),
                "near_wall_margin_threshold": _optional_float(metrics.get("near_wall_margin_threshold", wall_audit.get("near_wall_margin_threshold"))),
                "wall_margin_metric_scope": str(render_contract.get("wall_margin_metric_scope") or wall_audit.get("metric_scope") or "unavailable"),
                "low_strict_wall_interpretation_status": low_strict_status,
                "normal_fan_wall_crossing_certified": bool(
                    render_contract.get("normal_fan_wall_crossing_certified", metrics.get("normal_fan_wall_crossing_certified", False))
                ),
                "no_proxy_or_fallback": no_proxy_ok,
            }
        )
        if not available:
            reasons = []
            if not schema_ok:
                reasons.append("missing_tropical_support_render_contract_schema")
            if not readability_ok:
                reasons.append("missing_tropical_support_readability_contract_schema")
            if not no_proxy_ok:
                reasons.append("missing_tropical_support_no_proxy_contract")
            if not probability_ok:
                reasons.append("missing_model_tropical_support_probability_provenance")
            if not wall_rates_ok:
                reasons.append("missing_wall_margin_rates")
            if not assignment_ok:
                reasons.append("missing_observed_support_assignments")
            source["reason"] = ";".join(reasons) or "tropical_support_contract_unavailable"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_tropical_support_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "required_render_contract_schema": "tropicalgt.tropical_support_render.v1",
        "required_readability_contract_schema": "tropicalgt.tropical_support_readability.v1",
        "total_token_count": total_token_count,
        "total_valid_support_assignment_count": total_valid_assignments,
        "total_invalid_support_count": total_invalid_assignments,
        "support_probability_source_counts": {key: support_probability_source_counts[key] for key in sorted(support_probability_source_counts)},
        "low_strict_wall_interpretation_status_counts": {key: interpretation_status_counts[key] for key in sorted(interpretation_status_counts)},
        "sources": sources,
        "policy": "Herschel reports tropical support only from recorded tropical_support_payload sidecars with the render/readability contracts, model_tropical_support_probabilities provenance, wall-margin rates, observed support assignments, and no-proxy flags. These rates are model tropical-margin threshold audits, not certified normal-fan wall-crossing counts.",
    }


def _nll_density_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    visible_layer_counts: dict[str, int] = {}
    total_anchor_count = 0
    total_support_sample_count = 0
    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "got_nll_density_cloud_payload" not in lower:
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "nll_density_sidecar_missing"
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse message is platform-dependent
            source["reason"] = f"nll_density_sidecar_parse_error:{exc}"
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "nll_density_sidecar_not_object"
            sources.append(source)
            continue
        visual_contract = payload.get("visual_layer_contract") if isinstance(payload.get("visual_layer_contract"), dict) else {}
        density_contract = payload.get("density_contract") if isinstance(payload.get("density_contract"), dict) else {}
        density_volume = payload.get("density_volume") if isinstance(payload.get("density_volume"), dict) else {}
        nll_range = payload.get("nll_range") if isinstance(payload.get("nll_range"), dict) else {}
        anchor_count = _optional_int(payload.get("actual_model_anchor_count", visual_contract.get("actual_model_anchor_count")))
        support_sample_count = _optional_int(payload.get("support_sample_count", visual_contract.get("support_sample_count")))
        kernel_bandwidth = _optional_float(payload.get("kernel_bandwidth", visual_contract.get("kernel_bandwidth")))
        nll_span = _optional_float(nll_range.get("span"))
        visible_layers = visual_contract.get("visible_density_layers") if isinstance(visual_contract.get("visible_density_layers"), list) else []
        schema_ok = visual_contract.get("schema_version") == "tropicalgt.nll_density_render.v1"
        no_proxy_ok = visual_contract.get("no_proxy_or_fallback") is True
        anchor_ok = bool((anchor_count or 0) > 0 and visual_contract.get("actual_anchor_layer_visible_by_default") is True)
        support_ok = bool(
            support_sample_count is not None
            and anchor_count is not None
            and support_sample_count >= anchor_count
            and visual_contract.get("support_samples_are_model_states") is False
            and visual_contract.get("support_samples_hidden_as_model_states") is True
            and visual_contract.get("support_sample_trace_visibility") == "legendonly"
        )
        z_axis_ok = isinstance(visual_contract.get("z_axis_policy"), str) and str(visual_contract.get("z_axis_policy")).startswith("z is PC3")
        density_ok = bool(
            isinstance(visible_layers, list)
            and {"density_volume", "actual_model_anchor_markers"}.issubset(set(str(item) for item in visible_layers))
            and density_volume.get("support_samples_are_not_model_states") is True
        )
        contract_ok = bool(
            density_contract.get("actual_model_anchor_layer") is True
            and density_contract.get("sample_points_are_model_states") is False
            and density_contract.get("support_samples_hidden_as_model_states") is True
        )
        numeric_ok = bool(kernel_bandwidth is not None and kernel_bandwidth > 0.0 and nll_span is not None and nll_span >= 0.0)
        available = bool(payload.get("available") is True and schema_ok and no_proxy_ok and anchor_ok and support_ok and z_axis_ok and density_ok and contract_ok and numeric_ok)
        if available:
            total_anchor_count += int(anchor_count or 0)
            total_support_sample_count += int(support_sample_count or 0)
            for layer in visible_layers:
                visible_layer_counts[str(layer)] = visible_layer_counts.get(str(layer), 0) + 1
        source.update(
            {
                "available": available,
                "visual_layer_contract_schema_version": visual_contract.get("schema_version", "unavailable"),
                "actual_model_anchor_count": anchor_count,
                "support_sample_count": support_sample_count,
                "kernel_bandwidth": kernel_bandwidth,
                "nll_span": nll_span,
                "z_axis_policy": visual_contract.get("z_axis_policy", "unavailable"),
                "visible_density_layers": visible_layers,
                "actual_anchor_layer_visible_by_default": bool(visual_contract.get("actual_anchor_layer_visible_by_default", False)),
                "support_sample_trace_visibility": visual_contract.get("support_sample_trace_visibility", "unavailable"),
                "support_samples_are_model_states": visual_contract.get("support_samples_are_model_states"),
                "support_samples_hidden_as_model_states": visual_contract.get("support_samples_hidden_as_model_states"),
                "density_volume_support_samples_are_not_model_states": density_volume.get("support_samples_are_not_model_states"),
                "no_proxy_or_fallback": bool(no_proxy_ok),
            }
        )
        if not available:
            reasons = []
            if payload.get("available") is not True:
                reasons.append("nll_density_payload_unavailable")
            if not schema_ok:
                reasons.append("missing_nll_density_visual_layer_contract_schema")
            if not no_proxy_ok:
                reasons.append("missing_nll_density_no_proxy_contract")
            if not anchor_ok:
                reasons.append("missing_visible_actual_anchor_layer")
            if not support_ok:
                reasons.append("support_samples_not_hidden_as_non_model_states")
            if not z_axis_ok:
                reasons.append("missing_pc3_z_axis_policy")
            if not density_ok:
                reasons.append("missing_density_volume_or_anchor_layer")
            if not contract_ok:
                reasons.append("missing_density_contract_non_model_state_provenance")
            if not numeric_ok:
                reasons.append("missing_positive_kernel_or_nll_range")
            source["reason"] = ";".join(reasons) or "nll_density_evidence_unavailable"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_nll_density_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "required_visual_layer_contract_schema": "tropicalgt.nll_density_render.v1",
        "total_actual_model_anchor_count": total_anchor_count,
        "total_support_sample_count": total_support_sample_count,
        "visible_density_layer_counts": {key: visible_layer_counts[key] for key in sorted(visible_layer_counts)},
        "sources": sources,
        "policy": "Herschel reports NLL density evidence only from recorded got_nll_density_cloud_payload sidecars with tropicalgt.nll_density_render.v1, visible actual model anchors, legend-only non-model support samples, positive kernel bandwidth, NLL range, and no-proxy flags. Gaussian support samples are visualization support, not model states or training data.",
    }


def _graphcg_direction_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    basis_source_counts: dict[str, int] = {}
    total_direction_count = 0
    total_candidate_count = 0
    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "graphcg_direction_cosines_payload" not in lower:
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "graphcg_direction_sidecar_missing"
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse message is platform-dependent
            source["reason"] = f"graphcg_direction_sidecar_parse_error:{exc}"
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "graphcg_direction_sidecar_not_object"
            sources.append(source)
            continue
        evidence_contract = payload.get("graphcg_direction_evidence_contract") if isinstance(payload.get("graphcg_direction_evidence_contract"), dict) else {}
        readability_contract = payload.get("graphcg_readability_contract") if isinstance(payload.get("graphcg_readability_contract"), dict) else {}
        basis_certificate = payload.get("projection_basis_certificate") if isinstance(payload.get("projection_basis_certificate"), dict) else {}
        matrix_shape = payload.get("matrix_shape") if isinstance(payload.get("matrix_shape"), list) else []
        direction_count = _optional_int(evidence_contract.get("direction_count", payload.get("full_rank_direction_count")))
        direction_row_count = _optional_int(evidence_contract.get("direction_row_count", len(payload.get("direction_rows", [])) if isinstance(payload.get("direction_rows"), list) else None))
        candidate_count = _optional_int(basis_certificate.get("candidate_count", matrix_shape[0] if matrix_shape else None))
        panel_count = _optional_int(payload.get("panel_count"))
        contract_schema_ok = evidence_contract.get("schema_version") == "tropicalgt.graphcg_direction_evidence.v1"
        readability_schema_ok = readability_contract.get("schema_version") == "tropicalgt.graphcg_direction_readability.v1"
        no_proxy_ok = bool(evidence_contract.get("no_proxy_or_fallback") is True and readability_contract.get("no_proxy_or_fallback") is True)
        all_rows_ok = bool(evidence_contract.get("all_model_directions_have_rows") is True and direction_count and direction_row_count == direction_count)
        all_panels_ok = bool(
            evidence_contract.get("all_directions_rendered_in_heatmap") is True
            and evidence_contract.get("all_directions_rendered_in_activity_spectrum") is True
            and evidence_contract.get("all_directions_rendered_in_signed_bias_panel") is True
            and readability_contract.get("all_model_directions_rendered") is True
            and readability_contract.get("directions_sampled_for_heatmap") is False
        )
        basis_ok = bool(basis_certificate.get("available") is True and basis_certificate.get("all_candidates_have_all_direction_cosines") is True)
        safe_full_rank_ok = bool(evidence_contract.get("safe_to_render_full_rank_direction_evidence") is True)
        available = bool(payload.get("available") is True and contract_schema_ok and readability_schema_ok and no_proxy_ok and all_rows_ok and all_panels_ok and basis_ok and safe_full_rank_ok)
        basis_counts = basis_certificate.get("basis_source_counts") if isinstance(basis_certificate.get("basis_source_counts"), dict) else {}
        if available:
            total_direction_count += int(direction_count or 0)
            total_candidate_count += int(candidate_count or 0)
            for key, value in basis_counts.items():
                try:
                    basis_source_counts[str(key)] = basis_source_counts.get(str(key), 0) + int(value)
                except (TypeError, ValueError):
                    continue
        source.update(
            {
                "available": available,
                "direction_contract_schema_version": evidence_contract.get("schema_version", "unavailable"),
                "readability_contract_schema_version": readability_contract.get("schema_version", "unavailable"),
                "direction_count": direction_count,
                "direction_row_count": direction_row_count,
                "candidate_count": candidate_count,
                "matrix_shape": matrix_shape,
                "full_rank_direction_count": _optional_int(payload.get("full_rank_direction_count")),
                "active_rank_nonzero_mean_abs": _optional_int(payload.get("active_rank_nonzero_mean_abs")),
                "top_active_direction_panel_count": _optional_int(evidence_contract.get("top_active_direction_panel_count", payload.get("top_active_direction_limit"))),
                "panel_count": panel_count,
                "projection_basis": basis_certificate.get("projection_basis", "unavailable"),
                "basis_source_counts": basis_counts,
                "mean_abs_min": _optional_float(payload.get("mean_abs_min")),
                "mean_abs_max": _optional_float(payload.get("mean_abs_max")),
                "mean_abs_p90": _optional_float(payload.get("mean_abs_p90")),
                "safe_to_render_full_rank_direction_evidence": safe_full_rank_ok,
                "all_model_directions_have_rows": all_rows_ok,
                "all_direction_panels_available": all_panels_ok,
                "exact_direction_ids_preserved": bool(
                    evidence_contract.get("exact_direction_ids_preserved") is True
                    and readability_contract.get("exact_direction_ids_preserved_in_hover_and_payload") is True
                ),
                "no_proxy_or_fallback": no_proxy_ok,
            }
        )
        if not available:
            reasons = []
            if payload.get("available") is not True:
                reasons.append("graphcg_payload_unavailable")
            if not contract_schema_ok:
                reasons.append("missing_graphcg_direction_evidence_contract_schema")
            if not readability_schema_ok:
                reasons.append("missing_graphcg_direction_readability_contract_schema")
            if not no_proxy_ok:
                reasons.append("missing_graphcg_no_proxy_contract")
            if not all_rows_ok:
                reasons.append("missing_all_model_direction_rows")
            if not all_panels_ok:
                reasons.append("missing_full_rank_direction_panels")
            if not basis_ok:
                reasons.append("missing_projection_basis_certificate")
            if not safe_full_rank_ok:
                reasons.append("unsafe_full_rank_direction_evidence")
            source["reason"] = ";".join(reasons) or "graphcg_direction_evidence_unavailable"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_graphcg_direction_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "required_direction_contract_schema": "tropicalgt.graphcg_direction_evidence.v1",
        "required_readability_contract_schema": "tropicalgt.graphcg_direction_readability.v1",
        "total_direction_count": total_direction_count,
        "total_candidate_count": total_candidate_count,
        "basis_source_counts": {key: basis_source_counts[key] for key in sorted(basis_source_counts)},
        "sources": sources,
        "policy": "Herschel reports GraphCG direction evidence only from recorded graphcg_direction_cosines_payload sidecars with all model-derived directions rendered, exact ids preserved, projection-basis certificate present, and no-proxy/readability contracts. This is steering-basis activity evidence, not a semantic identifiability proof or toric fan certificate.",
    }


def _chart_bundle_transport_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    tier_counts: dict[str, int] = {}
    missing_group_counts: dict[str, int] = {}
    total_chart_count = 0
    total_transport_count = 0
    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "chart_bundle_transport_sidecar" not in lower or not lower.endswith(".json"):
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "chart_bundle_transport_sidecar_missing"
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse message is platform-dependent
            source["reason"] = f"chart_bundle_transport_sidecar_parse_error:{exc}"
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "chart_bundle_transport_sidecar_not_object"
            sources.append(source)
            continue
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        transport_contract = payload.get("monomial_transport_contract") if isinstance(payload.get("monomial_transport_contract"), dict) else {}
        matroid_contract = payload.get("bundle_matroid_contract") if isinstance(payload.get("bundle_matroid_contract"), dict) else {}
        paper_sidecar = payload.get("vector_bundle_paper_sidecar") if isinstance(payload.get("vector_bundle_paper_sidecar"), dict) else {}
        completeness = paper_sidecar.get("completeness_contract") if isinstance(paper_sidecar.get("completeness_contract"), dict) else {}
        chart_ids = payload.get("chart_ids") if isinstance(payload.get("chart_ids"), list) else []
        transport_ids = paper_sidecar.get("monomial_transport_ids") if isinstance(paper_sidecar.get("monomial_transport_ids"), list) else transport_contract.get("transport_ids", [])
        if not isinstance(transport_ids, list):
            transport_ids = []
        missing_groups = completeness.get("missing_required_groups") if isinstance(completeness.get("missing_required_groups"), list) else []
        schema_ok = payload.get("schema_version") == "tropicalgt.chart_bundle_transport_sidecar.v1"
        metadata_ok = metadata.get("schema_version") == "tropicalgt.chart_bundle_transport_metadata.v1" and metadata.get("available") is True
        transport_ok = bool(
            transport_contract.get("schema_version") == "tropicalgt.monomial_transport_head.v1"
            and transport_contract.get("actual_data_only") is True
            and transport_contract.get("no_proxy_or_fallback") is True
        )
        matroid_ok = bool(
            matroid_contract.get("schema_version") == "tropicalgt.bundle_matroid_flat_incidence.v1"
            and matroid_contract.get("actual_data_only") is True
            and matroid_contract.get("no_proxy_or_fallback") is True
        )
        paper_ok = bool(
            paper_sidecar.get("schema_version") == "tropicalgt.vector_bundle_paper_sidecar.v1"
            and paper_sidecar.get("actual_data_only") is True
            and paper_sidecar.get("no_proxy_or_fallback") is True
            and completeness.get("schema_version") == "tropicalgt.vector_bundle_paper_sidecar_completeness.v1"
            and completeness.get("actual_data_only") is True
            and completeness.get("no_proxy_or_fallback") is True
        )
        safety_ok = bool(
            payload.get("safe_to_render_as_toric_embedding_certificate") is False
            and payload.get("safe_to_render_as_tropical_variety_embedding") is False
            and payload.get("safe_to_render_as_global_toric_variety_embedding") is False
            and payload.get("safe_to_use_as_normal_fan_certificate") is False
            and paper_sidecar.get("safe_to_use_as_vector_bundle_theorem_certificate") is False
            and paper_sidecar.get("safe_to_use_as_toric_or_tropical_embedding_certificate", False) is False
        )
        no_proxy_ok = bool(payload.get("actual_data_only") is True and payload.get("no_proxy_or_fallback") is True)
        chart_ok = bool(len(chart_ids) > 0)
        available = bool(payload.get("available") is True and schema_ok and metadata_ok and transport_ok and matroid_ok and paper_ok and safety_ok and no_proxy_ok and chart_ok)
        completeness_tier = str(paper_sidecar.get("completeness_tier") or completeness.get("tier") or "unavailable")
        paper_ready = bool(
            completeness.get("paper_ready") is True
            and paper_sidecar.get("safe_to_use_as_vector_bundle_paper_ready_evidence") is True
            and not missing_groups
        )
        if available:
            tier_counts[completeness_tier] = tier_counts.get(completeness_tier, 0) + 1
            total_chart_count += len(chart_ids)
            total_transport_count += len(transport_ids)
            for group in missing_groups:
                missing_group_counts[str(group)] = missing_group_counts.get(str(group), 0) + 1
        source.update(
            {
                "available": available,
                "sidecar_schema_version": payload.get("schema_version", "unavailable"),
                "metadata_schema_version": metadata.get("schema_version", "unavailable"),
                "chart_count": len(chart_ids),
                "transport_count": len(transport_ids),
                "overlap_pair_count": _optional_int(payload.get("overlap_pair_count")),
                "overlap_triple_count": _optional_int(payload.get("overlap_triple_count")),
                "transport_contract_schema_version": transport_contract.get("schema_version", "unavailable"),
                "matroid_contract_schema_version": matroid_contract.get("schema_version", "unavailable"),
                "flat_incidence_shape": matroid_contract.get("flat_incidence_shape", []),
                "paper_sidecar_schema_version": paper_sidecar.get("schema_version", "unavailable"),
                "completeness_contract_schema_version": completeness.get("schema_version", "unavailable"),
                "completeness_tier": completeness_tier,
                "paper_ready": paper_ready,
                "missing_required_groups": [str(group) for group in missing_groups],
                "safe_to_render_as_toric_embedding_certificate": payload.get("safe_to_render_as_toric_embedding_certificate"),
                "safe_to_render_as_tropical_variety_embedding": payload.get("safe_to_render_as_tropical_variety_embedding"),
                "safe_to_use_as_normal_fan_certificate": payload.get("safe_to_use_as_normal_fan_certificate"),
                "safe_to_use_as_vector_bundle_theorem_certificate": paper_sidecar.get("safe_to_use_as_vector_bundle_theorem_certificate"),
                "safe_to_use_as_vector_bundle_paper_ready_evidence": paper_sidecar.get("safe_to_use_as_vector_bundle_paper_ready_evidence"),
                "actual_data_only": bool(payload.get("actual_data_only") is True and paper_sidecar.get("actual_data_only") is True),
                "no_proxy_or_fallback": bool(no_proxy_ok and paper_sidecar.get("no_proxy_or_fallback") is True),
            }
        )
        if not available:
            reasons = []
            if payload.get("available") is not True:
                reasons.append("chart_bundle_transport_sidecar_unavailable")
            if not schema_ok:
                reasons.append("missing_chart_bundle_transport_sidecar_schema")
            if not metadata_ok:
                reasons.append("missing_chart_bundle_transport_metadata")
            if not transport_ok:
                reasons.append("missing_monomial_transport_no_proxy_contract")
            if not matroid_ok:
                reasons.append("missing_bundle_matroid_no_proxy_contract")
            if not paper_ok:
                reasons.append("missing_vector_bundle_paper_sidecar_completeness_contract")
            if not safety_ok:
                reasons.append("unsafe_chart_bundle_certificate_claim")
            if not no_proxy_ok:
                reasons.append("missing_chart_bundle_no_proxy_contract")
            if not chart_ok:
                reasons.append("missing_chart_ids")
            source["reason"] = ";".join(reasons) or "chart_bundle_transport_evidence_unavailable"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    paper_ready_sources = [row for row in available_sources if row.get("paper_ready")]
    return {
        "schema_version": "tropicalgt.herschel_chart_bundle_transport_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "paper_ready_source_count": len(paper_ready_sources),
        "required_sidecar_schema": "tropicalgt.chart_bundle_transport_sidecar.v1",
        "required_metadata_schema": "tropicalgt.chart_bundle_transport_metadata.v1",
        "required_paper_sidecar_schema": "tropicalgt.vector_bundle_paper_sidecar.v1",
        "total_chart_count": total_chart_count,
        "total_transport_count": total_transport_count,
        "completeness_tier_counts": {key: tier_counts[key] for key in sorted(tier_counts)},
        "missing_required_group_counts": {key: missing_group_counts[key] for key in sorted(missing_group_counts)},
        "sources": sources,
        "policy": "Herschel reports chart/vector-bundle telemetry only from recorded chart_bundle_transport_sidecar JSON with exported metadata, monomial transport and matroid contracts, vector-bundle completeness contracts, and no-proxy flags. Paper-ready telemetry is not a vector-bundle theorem certificate, toric embedding, tropical variety, global toric variety, or normal-fan certificate.",
    }



def _two_parameter_bifiltration_visual_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    unavailable_reason_counts: dict[str, int] = {}
    total_cards = 0
    total_primary_cards = 0
    total_actual_generators = 0
    total_minimal_antichain = 0
    total_generator_labels = 0
    total_upward_regions = 0
    total_quotient_basis = 0
    total_hilbert_terms = 0
    total_adjacent_lcm_syzygies = 0
    total_structure_maps = 0
    total_rank_invariant_samples = 0
    total_grid_fibers = 0

    def _count(bucket: dict[str, int], key: str) -> None:
        if key:
            bucket[key] = bucket.get(key, 0) + 1

    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if not lower.endswith("trajectory_persistence/two_parameter_bifiltration.json"):
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "two_parameter_bifiltration_visual_sidecar_missing"
            _count(unavailable_reason_counts, source["reason"])
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser message is platform-dependent
            source["reason"] = f"two_parameter_bifiltration_visual_sidecar_parse_error:{exc}"
            _count(unavailable_reason_counts, "two_parameter_bifiltration_visual_sidecar_parse_error")
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "two_parameter_bifiltration_visual_sidecar_not_object"
            _count(unavailable_reason_counts, source["reason"])
            sources.append(source)
            continue
        axes = payload.get("axes") if isinstance(payload.get("axes"), dict) else {}
        module_contract = payload.get("module_visual_contract") if isinstance(payload.get("module_visual_contract"), dict) else {}
        structure = payload.get("structure_map_summary") if isinstance(payload.get("structure_map_summary"), dict) else {}
        primary_structure = payload.get("primary_structure_map_evidence") if isinstance(payload.get("primary_structure_map_evidence"), dict) else {}
        staircase = payload.get("miller_sturmfels_staircase_evidence") if isinstance(payload.get("miller_sturmfels_staircase_evidence"), dict) else {}
        cas_indexed = payload.get("certificate_indexed_cas_evidence") if isinstance(payload.get("certificate_indexed_cas_evidence"), dict) else {}
        grid = payload.get("grid") if isinstance(payload.get("grid"), dict) else {}
        chain_summary = payload.get("chain_generator_summary") if isinstance(payload.get("chain_generator_summary"), dict) else {}
        cards = payload.get("staircase_cards") if isinstance(payload.get("staircase_cards"), list) else []
        valid_cards = [card for card in cards if isinstance(card, dict)]
        direction_counts = structure.get("direction_counts") if isinstance(structure.get("direction_counts"), dict) else {}
        coordinate_cones = axes.get("coordinate_one_dimensional_cones") if isinstance(axes.get("coordinate_one_dimensional_cones"), list) else []
        schema_ok = payload.get("schema_version") == "tropicalgt.two_parameter_bifiltration_visual.v1"
        primary_view_ok = payload.get("primary_view") == "miller_sturmfels_bivariate_staircase"
        ring_ok = payload.get("coefficient_ring") == "F2[x_level,x_radius]" and staircase.get("coefficient_ring") == "F2[x_level,x_radius]"
        axes_ok = bool(
            axes.get("horizontal") == "x_radius"
            and axes.get("vertical") == "x_level"
            and "rho_x_radius" in coordinate_cones
            and "rho_x_level" in coordinate_cones
            and staircase.get("coordinate_axes_are_one_dimensional_cones") is True
        )
        no_proxy_ok = bool(
            payload.get("actual_data_only") is True
            and payload.get("no_proxy_resolution_claim") is True
            and module_contract.get("no_proxy_or_fallback") is True
            and structure.get("no_proxy_or_fallback") is True
            and primary_structure.get("no_proxy_or_fallback") is True
            and staircase.get("actual_data_only") is True
            and staircase.get("no_proxy_or_fallback") is True
            and cas_indexed.get("no_proxy_or_fallback") is True
        )
        module_ok = bool(
            module_contract.get("schema_version") == "tropicalgt.two_parameter_module_staircase_contract.v1"
            and module_contract.get("primary_view") == "miller_sturmfels_bivariate_staircase"
            and module_contract.get("rank_slabs_removed_as_primary_view") is True
            and module_contract.get("secondary_rank_diagnostics_labeled") is True
            and module_contract.get("x_radius_horizontal") is True
            and module_contract.get("x_level_vertical") is True
            and module_contract.get("shaded_regions_are_upward_closed_generated_submodules") is True
            and module_contract.get("white_points_are_displayed_quotient_basis_lattice_points") is True
            and module_contract.get("structure_map_summary_required") is True
            and module_contract.get("primary_structure_map_evidence_required") is True
            and "chain-presentation diagnostics are never substituted" in str(module_contract.get("safe_resolution_policy", ""))
        )
        structure_count = _optional_int(structure.get("actual_adjacent_map_count")) or 0
        valid_grade_count = _optional_int(structure.get("valid_grade_edge_count")) or 0
        structure_ok = bool(
            structure.get("schema_version") == "tropicalgt.two_parameter_structure_maps.v1"
            and structure.get("coefficient_ring") == "F2[x_level,x_radius]"
            and structure_count > 0
            and valid_grade_count == structure_count
            and structure.get("east_north_structure_maps_present") is True
            and int(direction_counts.get("x_level", 0) or 0) > 0
            and int(direction_counts.get("x_radius", 0) or 0) > 0
        )
        primary_structure_ok = bool(
            primary_structure.get("schema_version") == "tropicalgt.primary_structure_map_evidence.v1"
            and primary_structure.get("available") is True
            and primary_structure.get("source") == "bifiltration.structure_maps"
            and set(str(item) for item in primary_structure.get("directions_rendered", [])) == {"x_level", "x_radius"}
            and (_optional_int(primary_structure.get("primary_table_rows")) or 0) == structure_count
        )
        card_count = _optional_int(staircase.get("card_count")) or 0
        primary_card_count = _optional_int(staircase.get("primary_card_count")) or 0
        staircase_ok = bool(
            staircase.get("schema_version") == "tropicalgt.miller_sturmfels_staircase_evidence.v1"
            and staircase.get("source") == "staircase_cards_from_bifiltration.chain_module_generators[*].multidegree"
            and staircase.get("primary_view") == "miller_sturmfels_bivariate_staircase"
            and staircase.get("safe_to_render_miller_sturmfels_staircase") is True
            and card_count == len(valid_cards)
            and card_count > 0
            and primary_card_count == 1
            and staircase.get("all_cards_have_generator_labels") is True
            and staircase.get("all_cards_have_upward_closed_regions") is True
            and staircase.get("all_cards_have_quotient_basis_lattice_points") is True
            and staircase.get("all_cards_have_hilbert_numerator_terms") is True
            and staircase.get("all_cards_have_adjacent_lcm_syzygy_lists") is True
            and staircase.get("quotient_basis_counts_match_lattice_points") is True
            and staircase.get("theorem_scope_boundary_all_cards") is True
        )
        card_contract_ok = bool(
            valid_cards
            and all(
                card.get("schema_version") == "tropicalgt.two_parameter_staircase_card.v1"
                and card.get("x_radius_horizontal") is True
                and card.get("x_level_vertical") is True
                and card.get("shaded_regions_are_upward_closed_generated_submodules") is True
                and card.get("white_points_are_displayed_quotient_basis_lattice_points") is True
                and isinstance(card.get("generator_labels"), list)
                and isinstance(card.get("upward_closed_regions"), list)
                and isinstance(card.get("quotient_basis_lattice_points"), list)
                and isinstance(card.get("hilbert_numerator_terms"), list)
                and isinstance(card.get("adjacent_lcm_syzygies"), list)
                and "not a full persistence-module free resolution" in str(card.get("theorem_scope", ""))
                for card in valid_cards
            )
        )
        rank_policy_ok = bool(payload.get("rank_surface_primary") is False and "secondary" in str(payload.get("rank_surface_policy", "")).lower())
        cas_safe_ok = bool(cas_indexed.get("schema_version") == "tropicalgt.cas_certificate_indexed_evidence.v1" and cas_indexed.get("safe_unavailable_render") in (True, None))
        available = bool(schema_ok and primary_view_ok and ring_ok and axes_ok and no_proxy_ok and module_ok and structure_ok and primary_structure_ok and staircase_ok and card_contract_ok and rank_policy_ok and cas_safe_ok)
        status = "two_parameter_bifiltration_visual_available" if available else "two_parameter_bifiltration_visual_unavailable"
        _count(status_counts, status)
        if available:
            total_cards += card_count
            total_primary_cards += primary_card_count
            total_actual_generators += _optional_int(staircase.get("total_actual_generator_bidegree_count")) or 0
            total_minimal_antichain += _optional_int(staircase.get("total_minimal_antichain_count")) or 0
            total_generator_labels += _optional_int(staircase.get("total_generator_label_count")) or 0
            total_upward_regions += _optional_int(staircase.get("total_upward_closed_region_count")) or 0
            total_quotient_basis += _optional_int(staircase.get("total_quotient_basis_lattice_count")) or 0
            total_hilbert_terms += _optional_int(staircase.get("total_hilbert_numerator_term_count")) or 0
            total_adjacent_lcm_syzygies += _optional_int(staircase.get("total_adjacent_lcm_syzygy_count")) or 0
            total_structure_maps += structure_count
            total_rank_invariant_samples += _optional_int(payload.get("rank_invariant_sample_count")) or 0
            total_grid_fibers += _optional_int(grid.get("fiber_row_count")) or 0
        source.update(
            {
                "available": available,
                "status": status,
                "schema_version": payload.get("schema_version", "unavailable"),
                "primary_view": payload.get("primary_view", "unavailable"),
                "coefficient_ring": payload.get("coefficient_ring", "unavailable"),
                "axes_horizontal": axes.get("horizontal", "unavailable"),
                "axes_vertical": axes.get("vertical", "unavailable"),
                "coordinate_one_dimensional_cones": coordinate_cones,
                "rank_surface_primary": payload.get("rank_surface_primary"),
                "module_visual_contract_schema": module_contract.get("schema_version", "unavailable"),
                "structure_map_summary_schema": structure.get("schema_version", "unavailable"),
                "primary_structure_map_evidence_schema": primary_structure.get("schema_version", "unavailable"),
                "staircase_evidence_schema": staircase.get("schema_version", "unavailable"),
                "cas_certificate_indexed_schema": cas_indexed.get("schema_version", "unavailable"),
                "staircase_card_count": card_count,
                "primary_staircase_card_count": primary_card_count,
                "actual_generator_bidegree_count": _optional_int(staircase.get("total_actual_generator_bidegree_count")) or 0,
                "minimal_antichain_count": _optional_int(staircase.get("total_minimal_antichain_count")) or 0,
                "generator_label_count": _optional_int(staircase.get("total_generator_label_count")) or 0,
                "upward_closed_region_count": _optional_int(staircase.get("total_upward_closed_region_count")) or 0,
                "quotient_basis_lattice_count": _optional_int(staircase.get("total_quotient_basis_lattice_count")) or 0,
                "hilbert_numerator_term_count": _optional_int(staircase.get("total_hilbert_numerator_term_count")) or 0,
                "adjacent_lcm_syzygy_count": _optional_int(staircase.get("total_adjacent_lcm_syzygy_count")) or 0,
                "structure_map_count": structure_count,
                "rank_invariant_sample_count": _optional_int(payload.get("rank_invariant_sample_count")) or 0,
                "grid_fiber_row_count": _optional_int(grid.get("fiber_row_count")) or 0,
                "chain_generator_total_count": _optional_int(chain_summary.get("total_generator_count")) or 0,
                "no_proxy_or_fallback": no_proxy_ok,
            }
        )
        if not available:
            reasons = []
            if not schema_ok:
                reasons.append("missing_two_parameter_bifiltration_visual_schema")
            if not primary_view_ok:
                reasons.append("primary_view_not_miller_sturmfels_staircase")
            if not ring_ok:
                reasons.append("wrong_two_parameter_coefficient_ring")
            if not axes_ok:
                reasons.append("missing_x_radius_x_level_axes_or_coordinate_cones")
            if not no_proxy_ok:
                reasons.append("missing_two_parameter_no_proxy_contract")
            if not module_ok:
                reasons.append("missing_two_parameter_module_visual_contract")
            if not structure_ok:
                reasons.append("missing_two_parameter_structure_map_summary")
            if not primary_structure_ok:
                reasons.append("missing_primary_structure_map_evidence")
            if not staircase_ok:
                reasons.append("missing_miller_sturmfels_staircase_evidence")
            if not card_contract_ok:
                reasons.append("missing_staircase_card_contracts")
            if not rank_policy_ok:
                reasons.append("rank_surface_still_marked_primary")
            if not cas_safe_ok:
                reasons.append("unsafe_certificate_indexed_cas_state")
            source["reason"] = ";".join(reasons) or "two_parameter_bifiltration_visual_evidence_unavailable"
            for reason in reasons or [source["reason"]]:
                _count(unavailable_reason_counts, reason)
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_two_parameter_bifiltration_visual_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "total_staircase_card_count": total_cards,
        "total_primary_staircase_card_count": total_primary_cards,
        "total_actual_generator_bidegree_count": total_actual_generators,
        "total_minimal_antichain_count": total_minimal_antichain,
        "total_generator_label_count": total_generator_labels,
        "total_upward_closed_region_count": total_upward_regions,
        "total_quotient_basis_lattice_count": total_quotient_basis,
        "total_hilbert_numerator_term_count": total_hilbert_terms,
        "total_adjacent_lcm_syzygy_count": total_adjacent_lcm_syzygies,
        "total_structure_map_count": total_structure_maps,
        "total_rank_invariant_sample_count": total_rank_invariant_samples,
        "total_grid_fiber_row_count": total_grid_fibers,
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "unavailable_reason_counts": {key: unavailable_reason_counts[key] for key in sorted(unavailable_reason_counts)},
        "sources": sources,
        "policy": "Herschel reports two-parameter bifiltration visuals only from recorded tropicalgt.two_parameter_bifiltration_visual.v1 payloads whose primary view is the Miller-Sturmfels bivariate staircase over F2[x_level,x_radius]. Rank surfaces remain secondary diagnostics, and scoped adjacent-LCM staircase resolutions are not promoted to full persistence-module free resolutions or CAS certificates.",
    }

def _bivariate_module_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    resolution_status_counts: dict[str, int] = {}
    unavailable_reason_counts: dict[str, int] = {}
    total_fibers = 0
    total_structure_maps = 0
    total_rank_samples = 0
    total_generators = 0
    certified_resolution_count = 0
    safe_unavailable_resolution_count = 0
    module_available_count = 0
    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "trajectory_level_radius_bifiltration" not in lower or not lower.endswith(".json"):
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "bivariate_module_sidecar_missing"
            unavailable_reason_counts[source["reason"]] = unavailable_reason_counts.get(source["reason"], 0) + 1
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            source["reason"] = f"bivariate_module_sidecar_parse_error:{exc}"
            unavailable_reason_counts["bivariate_module_sidecar_parse_error"] = unavailable_reason_counts.get("bivariate_module_sidecar_parse_error", 0) + 1
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "bivariate_module_sidecar_not_object"
            unavailable_reason_counts[source["reason"]] = unavailable_reason_counts.get(source["reason"], 0) + 1
            sources.append(source)
            continue
        diagnostics = payload.get("chain_presentation_diagnostics") if isinstance(payload.get("chain_presentation_diagnostics"), dict) else {}
        real = diagnostics.get("real_free_resolution") if isinstance(diagnostics.get("real_free_resolution"), dict) else {}
        fiber_count = len(payload.get("fiber_rank_profile", [])) if isinstance(payload.get("fiber_rank_profile"), list) else 0
        structure_map_count = len(payload.get("structure_maps", [])) if isinstance(payload.get("structure_maps"), list) else 0
        rank_sample_count = len(payload.get("rank_invariant_samples", [])) if isinstance(payload.get("rank_invariant_samples"), list) else 0
        generator_count = len(payload.get("chain_module_generators", [])) if isinstance(payload.get("chain_module_generators"), list) else 0
        boundary_count = sum(len(v) for v in payload.get("boundary_monomials", {}).values() if isinstance(v, list)) if isinstance(payload.get("boundary_monomials"), dict) else 0
        ring_ok = payload.get("coefficient_ring") == "F2[x_level,x_radius]" and diagnostics.get("ring") == "F2[x_level,x_radius]"
        grid_ok = bool(payload.get("available") is True and payload.get("num_parameters") == 2 and fiber_count > 0 and structure_map_count > 0)
        no_proxy_ok = bool(diagnostics.get("method") == "finite_multigraded_chain_presentation_diagnostics" and diagnostics.get("not_a_free_resolution") is True)
        module_available = bool(ring_ok and grid_ok and no_proxy_ok)
        resolution_certified = bool(
            real.get("schema_version") == "tropicalgt.real_free_resolution.v1"
            and real.get("available") is True
            and real.get("certificate_attached") is True
            and real.get("real_free_resolution_certified") is True
            and real.get("exactness_certified") is True
            and real.get("multigraded_free_resolution_certified") is True
            and real.get("safe_to_render_as_multigraded_free_resolution") is True
        )
        safe_unavailable = bool(
            real.get("schema_version") == "tropicalgt.real_free_resolution.v1"
            and real.get("available") is False
            and real.get("safe_unavailable_render") is True
            and real.get("safe_to_render_as_multigraded_free_resolution") is False
        )
        resolution_status = str(real.get("status") or diagnostics.get("resolution_status") or "unavailable")
        resolution_status_counts[resolution_status] = resolution_status_counts.get(resolution_status, 0) + 1
        if module_available:
            module_available_count += 1
            total_fibers += fiber_count
            total_structure_maps += structure_map_count
            total_rank_samples += rank_sample_count
            total_generators += generator_count
        if resolution_certified:
            certified_resolution_count += 1
        if safe_unavailable:
            safe_unavailable_resolution_count += 1
        source.update(
            {
                "available": module_available,
                "coefficient_ring": payload.get("coefficient_ring", "unavailable"),
                "num_parameters": payload.get("num_parameters"),
                "object_key_selected": payload.get("object_key_selected", "unavailable"),
                "fiber_rank_profile_count": fiber_count,
                "structure_map_count": structure_map_count,
                "rank_invariant_sample_count": rank_sample_count,
                "chain_module_generator_count": generator_count,
                "boundary_monomial_entry_count": boundary_count,
                "chain_presentation_method": diagnostics.get("method", "unavailable"),
                "chain_presentation_not_a_free_resolution": bool(diagnostics.get("not_a_free_resolution", False)),
                "certificate_attached": bool(diagnostics.get("certificate_attached", False)),
                "real_free_resolution_schema_version": real.get("schema_version", "unavailable"),
                "real_free_resolution_status": resolution_status,
                "real_free_resolution_certified": resolution_certified,
                "safe_unavailable_real_free_resolution": safe_unavailable,
                "safe_to_render_as_multigraded_free_resolution": bool(real.get("safe_to_render_as_multigraded_free_resolution", False)),
                "real_free_resolution_reason": real.get("reason", ""),
                "input_sha256": real.get("input_sha256"),
                "no_proxy_or_fallback": no_proxy_ok,
            }
        )
        if not module_available:
            reasons = []
            if not ring_ok:
                reasons.append("missing_f2_level_radius_ring")
            if not grid_ok:
                reasons.append("missing_bivariate_fiber_or_structure_maps")
            if not no_proxy_ok:
                reasons.append("missing_chain_presentation_no_proxy_guard")
            source["reason"] = ";".join(reasons) or "bivariate_module_evidence_unavailable"
            for reason in reasons or [source["reason"]]:
                unavailable_reason_counts[reason] = unavailable_reason_counts.get(reason, 0) + 1
        sources.append(source)
    return {
        "schema_version": "tropicalgt.herschel_bivariate_module_evidence.v1",
        "available": bool(module_available_count),
        "source_count": len(sources),
        "module_available_source_count": module_available_count,
        "certified_real_free_resolution_source_count": certified_resolution_count,
        "safe_unavailable_real_free_resolution_source_count": safe_unavailable_resolution_count,
        "total_fiber_rank_profile_count": total_fibers,
        "total_structure_map_count": total_structure_maps,
        "total_rank_invariant_sample_count": total_rank_samples,
        "total_chain_module_generator_count": total_generators,
        "resolution_status_counts": {key: resolution_status_counts[key] for key in sorted(resolution_status_counts)},
        "unavailable_reason_counts": {key: unavailable_reason_counts[key] for key in sorted(unavailable_reason_counts)},
        "sources": sources,
        "policy": "Herschel reports bivariate F2[x_level,x_radius] module evidence from trajectory_level_radius_bifiltration.json. Finite chain-presentation diagnostics are not treated as real free resolutions; the nested tropicalgt.real_free_resolution.v1 guard must certify exactness and multigraded resolution safety before any free-resolution claim is available.",
    }


def _persistence_landscape_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    backend_counts: dict[str, int] = {}
    unavailable_reason_counts: dict[str, int] = {}
    total_rows = 0
    total_curve_traces = 0
    total_finite_intervals = 0
    total_growth_rows = 0
    verified_unavailable_count = 0
    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "persistence_landscapes" not in lower or not lower.endswith(".json"):
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = "persistence_landscape_sidecar_missing"
            unavailable_reason_counts[source["reason"]] = unavailable_reason_counts.get(source["reason"], 0) + 1
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser message is platform-dependent
            source["reason"] = f"persistence_landscape_sidecar_parse_error:{exc}"
            unavailable_reason_counts["persistence_landscape_sidecar_parse_error"] = unavailable_reason_counts.get("persistence_landscape_sidecar_parse_error", 0) + 1
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "persistence_landscape_sidecar_not_object"
            unavailable_reason_counts[source["reason"]] = unavailable_reason_counts.get(source["reason"], 0) + 1
            sources.append(source)
            continue
        rows = payload.get("landscape_rows") if isinstance(payload.get("landscape_rows"), list) else []
        unavailable_reasons = [str(reason) for reason in payload.get("unavailable_reasons", [])] if isinstance(payload.get("unavailable_reasons"), list) else []
        backend = str(payload.get("landscape_backend") or "unavailable")
        row_count = len([row for row in rows if isinstance(row, dict)])
        curve_trace_count = _optional_int(payload.get("curve_trace_count")) or 0
        finite_interval_count = _optional_int(payload.get("finite_persistence_interval_count")) or 0
        growth_row_count = _optional_int(payload.get("growth_row_count")) or 0
        schema_ok = payload.get("schema_version") == "tropicalgt.persistence_landscape_visual_contract.v1"
        no_proxy_ok = bool(payload.get("actual_data_only") is True and payload.get("no_proxy_or_fallback") is True and payload.get("not_nll_fitness_landscape") is True)
        rows_ok = bool(row_count > 0 and curve_trace_count > 0 and payload.get("safe_to_render_actual_landscape_functions") is True)
        backend_ok = bool(backend and backend != "unavailable")
        norm_only_ok = payload.get("not_norm_only_summary") is True
        available = bool(payload.get("available") is True and schema_ok and no_proxy_ok and rows_ok and backend_ok and norm_only_ok)
        verified_unavailable = bool(not available and payload.get("available") is False and payload.get("unavailable_state_verified_by_intervals") is True and finite_interval_count == 0)
        if available:
            total_rows += row_count
            total_curve_traces += curve_trace_count
            total_finite_intervals += finite_interval_count
            total_growth_rows += growth_row_count
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        if verified_unavailable:
            verified_unavailable_count += 1
        for reason in unavailable_reasons:
            unavailable_reason_counts[reason] = unavailable_reason_counts.get(reason, 0) + 1
        source.update(
            {
                "available": available,
                "verified_unavailable": verified_unavailable,
                "schema_version": payload.get("schema_version", "unavailable"),
                "landscape_backend": backend,
                "landscape_row_count": row_count,
                "curve_trace_count": curve_trace_count,
                "finite_persistence_interval_count": finite_interval_count,
                "growth_row_count": growth_row_count,
                "homology_dimensions": payload.get("homology_dimensions", []),
                "small_multiples_available": bool(payload.get("small_multiples_available", False)),
                "heatmap_available": bool(payload.get("heatmap_available", False)),
                "safe_to_render_actual_landscape_functions": bool(payload.get("safe_to_render_actual_landscape_functions", False)),
                "not_nll_fitness_landscape": bool(payload.get("not_nll_fitness_landscape", False)),
                "not_norm_only_summary": bool(payload.get("not_norm_only_summary", False)),
                "unavailable_reasons": unavailable_reasons,
                "no_proxy_or_fallback": no_proxy_ok,
            }
        )
        if not available:
            reasons = []
            if payload.get("available") is not True:
                reasons.append("persistence_landscape_unavailable")
            if not schema_ok:
                reasons.append("missing_persistence_landscape_visual_contract_schema")
            if not no_proxy_ok:
                reasons.append("missing_persistence_landscape_no_proxy_contract")
            if not rows_ok:
                reasons.append("missing_actual_gudhi_landscape_rows")
            if not backend_ok:
                reasons.append("missing_gudhi_landscape_backend")
            if not norm_only_ok:
                reasons.append("landscape_not_verified_beyond_norm_summary")
            source["reason"] = ";".join(reasons) or "persistence_landscape_evidence_unavailable"
            for reason in reasons or [source["reason"]]:
                unavailable_reason_counts[reason] = unavailable_reason_counts.get(reason, 0) + 1
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_persistence_landscape_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "verified_unavailable_source_count": verified_unavailable_count,
        "required_visual_contract_schema": "tropicalgt.persistence_landscape_visual_contract.v1",
        "total_landscape_row_count": total_rows,
        "total_curve_trace_count": total_curve_traces,
        "total_finite_persistence_interval_count": total_finite_intervals,
        "total_growth_row_count": total_growth_rows,
        "backend_counts": {key: backend_counts[key] for key in sorted(backend_counts)},
        "unavailable_reason_counts": {key: unavailable_reason_counts[key] for key in sorted(unavailable_reason_counts)},
        "sources": sources,
        "policy": "Herschel reports persistence landscapes only from recorded trajectory_persistence/persistence_landscapes.json sidecars with actual GUDHI lambda_k(t) rows, backend provenance, no-proxy flags, and an explicit not-NLL/fitness-landscape contract. No finite-interval cases remain verified unavailable and are not rendered as zero landscapes or norm-only summaries.",
    }


def _toric_tropical_cas_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    unavailable_reason_counts: dict[str, int] = {}
    toric_source_count = 0
    tropical_source_count = 0
    certified_finite_toric_ideal_count = 0
    certified_tropical_fan_count = 0
    forbidden_global_claim_count = 0
    total_tropical_ray_count = 0

    def _count(mapping: dict[str, int], key: str) -> None:
        mapping[key] = mapping.get(key, 0) + 1

    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "toric_embedding_sidecar" in lower and lower.endswith(".json"):
            kind = "toric_embedding_sidecar"
            toric_source_count += 1
        elif "tropical_fan_diagnostics" in lower and lower.endswith(".json"):
            kind = "tropical_fan_diagnostics"
            tropical_source_count += 1
        else:
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"path": _project_path(resolved), "kind": kind, "available": False}
        if not resolved.exists():
            source["reason"] = f"{kind}_missing"
            _count(unavailable_reason_counts, source["reason"])
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser message is platform-dependent
            source["reason"] = f"{kind}_parse_error:{exc}"
            _count(unavailable_reason_counts, f"{kind}_parse_error")
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = f"{kind}_not_object"
            _count(unavailable_reason_counts, source["reason"])
            sources.append(source)
            continue
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        input_contract = payload.get("cas_input_contract") if isinstance(payload.get("cas_input_contract"), dict) else {}
        status = str(diagnostics.get("status") or ("available" if payload.get("available") is True else "unavailable"))
        _count(status_counts, f"{kind}:{status}")
        if kind == "toric_embedding_sidecar":
            schema_ok = payload.get("schema_version") == "tropicalgt.toric_embedding_sidecar_visual_audit.v1"
            diag_schema_ok = diagnostics.get("schema_version") == "tropicalgt.cas_toric_embedding.v1"
            input_schema_ok = input_contract.get("schema_version") == "tropicalgt.toric_embedding_input_contract.v1"
            no_proxy_ok = bool(input_contract.get("actual_data_only") is True and input_contract.get("no_proxy_or_fallback") is True and input_contract.get("proxy_substitution_allowed") is False)
            explicit_input_ok = bool(input_contract.get("explicit_cas_input_present") is True and input_contract.get("safe_to_render_certificate") is True and bool(input_contract.get("input_sha256")))
            finite_certificate_ok = bool(payload.get("available") is True and payload.get("safe_to_render_as_finite_toric_ideal_sidecar") is True and diagnostics.get("certificate_attached") is True and diagnostics.get("toric_ideal_certified") is True and diagnostics.get("safe_to_render_as_toric_embedding") is True)
            forbidden_claim_safe = bool(payload.get("safe_to_render_as_tropical_variety_embedding") is False and payload.get("safe_to_render_as_global_toric_variety_embedding") is False and payload.get("safe_to_use_as_normal_fan_certificate") is False and diagnostics.get("safe_to_render_as_tropical_variety_embedding") is False and diagnostics.get("safe_to_render_as_global_toric_variety_embedding") is False and diagnostics.get("safe_to_use_as_normal_fan_certificate") is False)
            available = bool(schema_ok and diag_schema_ok and input_schema_ok and no_proxy_ok and explicit_input_ok and finite_certificate_ok and forbidden_claim_safe)
            if available:
                certified_finite_toric_ideal_count += 1
            if not forbidden_claim_safe:
                forbidden_global_claim_count += 1
            source.update({
                "available": available,
                "sidecar_schema_version": payload.get("schema_version", "unavailable"),
                "cas_schema_version": diagnostics.get("schema_version", "unavailable"),
                "cas_input_contract_schema_version": input_contract.get("schema_version", "unavailable"),
                "status": status,
                "backend": diagnostics.get("backend", "unavailable"),
                "certificate_attached": bool(diagnostics.get("certificate_attached", False)),
                "toric_ideal_certified": bool(diagnostics.get("toric_ideal_certified", False)),
                "finite_toric_ideal_sidecar_safe": bool(payload.get("safe_to_render_as_finite_toric_ideal_sidecar", False)),
                "safe_to_render_as_global_toric_variety_embedding": payload.get("safe_to_render_as_global_toric_variety_embedding"),
                "safe_to_render_as_tropical_variety_embedding": payload.get("safe_to_render_as_tropical_variety_embedding"),
                "safe_to_use_as_normal_fan_certificate": payload.get("safe_to_use_as_normal_fan_certificate"),
                "explicit_cas_input_present": bool(input_contract.get("explicit_cas_input_present", False)),
                "input_sha256": input_contract.get("input_sha256"),
                "no_proxy_or_fallback": no_proxy_ok,
            })
            if not available:
                reasons = []
                if payload.get("available") is not True:
                    reasons.append("toric_embedding_sidecar_unavailable")
                if not schema_ok:
                    reasons.append("missing_toric_embedding_sidecar_schema")
                if not diag_schema_ok:
                    reasons.append("missing_cas_toric_embedding_schema")
                if not input_schema_ok:
                    reasons.append("missing_toric_embedding_input_contract_schema")
                if not no_proxy_ok:
                    reasons.append("missing_toric_embedding_no_proxy_contract")
                if not explicit_input_ok:
                    reasons.append("missing_explicit_toric_exponent_matrix_input")
                if not finite_certificate_ok:
                    reasons.append("finite_toric_ideal_certificate_unavailable")
                if not forbidden_claim_safe:
                    reasons.append("unsafe_global_toric_or_tropical_claim")
                source["reason"] = ";".join(reasons) or "toric_embedding_evidence_unavailable"
                for reason in reasons or [source["reason"]]:
                    _count(unavailable_reason_counts, reason)
        else:
            schema_ok = payload.get("schema_version") == "tropicalgt.tropical_fan_visual_audit.v1"
            diag_schema_ok = diagnostics.get("schema_version") == "tropicalgt.cas_tropical_fan.v1"
            input_schema_ok = input_contract.get("schema_version") == "tropicalgt.tropical_fan_input_contract.v1"
            no_proxy_ok = bool(input_contract.get("actual_data_only") is True and input_contract.get("no_proxy_or_fallback") is True and input_contract.get("proxy_substitution_allowed") is False)
            explicit_input_ok = bool(input_contract.get("explicit_cas_input_present") is True and input_contract.get("safe_to_render_certificate") is True and bool(input_contract.get("input_sha256")))
            summary = diagnostics.get("fan_summary") if isinstance(diagnostics.get("fan_summary"), dict) else {}
            ray_count = _optional_int(summary.get("ray_count")) or 0
            fan_certificate_ok = bool(payload.get("available") is True and payload.get("safe_to_render_as_tropical_fan") is True and diagnostics.get("certificate_attached") is True and diagnostics.get("fan_diagnostics_certified") is True and diagnostics.get("safe_to_render_as_tropical_fan") is True and ray_count > 0)
            available = bool(schema_ok and diag_schema_ok and input_schema_ok and no_proxy_ok and explicit_input_ok and fan_certificate_ok)
            if available:
                certified_tropical_fan_count += 1
                total_tropical_ray_count += int(ray_count)
            source.update({
                "available": available,
                "sidecar_schema_version": payload.get("schema_version", "unavailable"),
                "cas_schema_version": diagnostics.get("schema_version", "unavailable"),
                "cas_input_contract_schema_version": input_contract.get("schema_version", "unavailable"),
                "status": status,
                "backend": diagnostics.get("backend", "unavailable"),
                "certificate_attached": bool(diagnostics.get("certificate_attached", False)),
                "fan_diagnostics_certified": bool(diagnostics.get("fan_diagnostics_certified", False)),
                "tropical_cycle_certified": bool(diagnostics.get("tropical_cycle_certified", False)),
                "safe_to_render_as_tropical_fan": bool(payload.get("safe_to_render_as_tropical_fan", False)),
                "ray_count": ray_count,
                "ambient_dimension": _optional_int(summary.get("ambient_dimension")),
                "explicit_cas_input_present": bool(input_contract.get("explicit_cas_input_present", False)),
                "input_sha256": input_contract.get("input_sha256"),
                "no_proxy_or_fallback": no_proxy_ok,
            })
            if not available:
                reasons = []
                if payload.get("available") is not True:
                    reasons.append("tropical_fan_diagnostics_unavailable")
                if not schema_ok:
                    reasons.append("missing_tropical_fan_visual_schema")
                if not diag_schema_ok:
                    reasons.append("missing_cas_tropical_fan_schema")
                if not input_schema_ok:
                    reasons.append("missing_tropical_fan_input_contract_schema")
                if not no_proxy_ok:
                    reasons.append("missing_tropical_fan_no_proxy_contract")
                if not explicit_input_ok:
                    reasons.append("missing_explicit_model_derived_tropical_ideal")
                if not fan_certificate_ok:
                    reasons.append("tropical_fan_certificate_unavailable")
                if ray_count <= 0:
                    reasons.append("missing_one_dimensional_cones")
                source["reason"] = ";".join(reasons) or "tropical_fan_evidence_unavailable"
                for reason in reasons or [source["reason"]]:
                    _count(unavailable_reason_counts, reason)
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_toric_tropical_cas_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "toric_source_count": toric_source_count,
        "tropical_fan_source_count": tropical_source_count,
        "certified_finite_toric_ideal_count": certified_finite_toric_ideal_count,
        "certified_tropical_fan_count": certified_tropical_fan_count,
        "forbidden_global_claim_count": forbidden_global_claim_count,
        "total_tropical_ray_count": total_tropical_ray_count,
        "required_toric_sidecar_schema": "tropicalgt.toric_embedding_sidecar_visual_audit.v1",
        "required_toric_cas_schema": "tropicalgt.cas_toric_embedding.v1",
        "required_tropical_fan_sidecar_schema": "tropicalgt.tropical_fan_visual_audit.v1",
        "required_tropical_fan_cas_schema": "tropicalgt.cas_tropical_fan.v1",
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "unavailable_reason_counts": {key: unavailable_reason_counts[key] for key in sorted(unavailable_reason_counts)},
        "sources": sources,
        "policy": "Herschel reports toric/tropical CAS evidence only from recorded toric_embedding_sidecar and tropical_fan_diagnostics JSON sidecars. Finite toric-ideal sidecar certificates and Macaulay2 Tropical fan certificates remain separate from global toric-variety embeddings, tropical-variety embeddings, normal-fan certificates, vector-bundle theorem evidence, and BPB restart justification.",
    }


def _analogical_query_context_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "analogical_simplicial_maps" not in lower and "analogical_query_context" not in lower:
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {
            "path": _project_path(resolved),
            "available": False,
        }
        if not resolved.exists():
            source["reason"] = "analogical_query_context_sidecar_missing"
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse message is platform-dependent
            source["reason"] = f"analogical_query_context_sidecar_parse_error:{exc}"
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = "analogical_query_context_sidecar_not_object"
            sources.append(source)
            continue
        contract = payload.get("query_context_contract") if isinstance(payload.get("query_context_contract"), dict) else {}
        topk = payload.get("topk_contract") if isinstance(payload.get("topk_contract"), dict) else {}
        topk_contract = topk.get("query_context_contract") if isinstance(topk.get("query_context_contract"), dict) else {}
        rejected = contract.get("rejected_query_context_keys", []) if isinstance(contract.get("rejected_query_context_keys"), list) else []
        schema_ok = contract.get("schema_version") == "tropicalgt.analogical_query_context_conversion.v1"
        embedded_ok = bool(topk_contract == contract and topk.get("query_context_contract_schema_version") == "tropicalgt.analogical_query_context_conversion.v1")
        source.update(
            {
                "available": bool(schema_ok),
                "schema_version": contract.get("schema_version", "unavailable"),
                "topk_embeds_same_contract": embedded_ok,
                "selected_query_complex_source": contract.get("selected_query_complex_source", "unavailable"),
                "selected_query_complex_available": bool(contract.get("selected_query_complex_available", False)),
                "selected_query_probability_vertex_count": contract.get("selected_query_probability_vertex_count", 0),
                "conversion_status": contract.get("conversion_status", "unavailable"),
                "query_topological_algebra_available": bool(contract.get("query_topological_algebra_available", False)),
                "rejected_query_context_key_count": len(rejected),
                "rejected_query_context_keys": [str(row.get("key", "")) for row in rejected if isinstance(row, dict)],
                "embedding_only_assignment_allowed": bool(contract.get("embedding_only_assignment_allowed", True)),
                "probability_assignment_metric_required": contract.get("probability_assignment_metric_required", "unavailable"),
                "no_proxy_or_fallback": bool(contract.get("no_proxy_or_fallback", False)),
            }
        )
        if not schema_ok:
            source["reason"] = "analogical_query_context_contract_missing_or_wrong_schema"
        elif not embedded_ok:
            source["reason"] = "analogical_query_context_contract_not_embedded_in_topk_contract"
        sources.append(source)
    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_analogical_query_context_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "required_contract_schema": "tropicalgt.analogical_query_context_conversion.v1",
        "sources": sources,
        "policy": "Herschel reports analogical query-domain contracts only from recorded analogical_simplicial_maps sidecars; missing sidecars remain unavailable and cannot justify a restart.",
    }


def _analogical_memory_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    quality_gate_reason_counts: dict[str, int] = {}
    total_bank_size = 0
    total_records_added = 0
    total_retrieved_count = 0
    total_top_k_requested = 0
    total_top_k_rendered = 0
    total_simplex_tree_pair_count = 0
    total_checked_simplices = 0
    total_preserved_simplices = 0
    no_proxy_contract_source_count = 0
    probability_vector_contract_source_count = 0
    verified_insufficient_memory_source_count = 0

    def _count(mapping: dict[str, int], key: str, amount: int = 1) -> None:
        if not key:
            return
        mapping[str(key)] = mapping.get(str(key), 0) + int(amount)

    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if "analogical_memory_retrieval" in lower and lower.endswith(".json"):
            kind = "retrieval"
        elif "analogical_simplicial_maps" in lower and lower.endswith(".json"):
            kind = "topk_maps"
        elif "analogical_simplex_tree_analogy" in lower and lower.endswith(".json"):
            kind = "simplex_tree_analogy"
        else:
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {
            "kind": kind,
            "path": _project_path(resolved),
            "available": False,
            "verified_insufficient_memory": False,
        }
        if not resolved.exists():
            source["reason"] = f"{kind}_sidecar_missing"
            _count(status_counts, source["reason"])
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse text is platform-dependent
            source["reason"] = f"{kind}_sidecar_parse_error:{exc}"
            _count(status_counts, f"{kind}_sidecar_parse_error")
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = f"{kind}_sidecar_not_object"
            _count(status_counts, source["reason"])
            sources.append(source)
            continue

        if kind == "retrieval":
            retrieved = payload.get("retrieved", []) if isinstance(payload.get("retrieved"), list) else []
            quality_gate = payload.get("quality_gate", {}) if isinstance(payload.get("quality_gate"), dict) else {}
            retrieval_weights = payload.get("retrieval_weights", {}) if isinstance(payload.get("retrieval_weights"), dict) else {}
            reason_counts = quality_gate.get("reason_counts", {}) if isinstance(quality_gate.get("reason_counts"), dict) else {}
            bank_size = _optional_int(payload.get("bank_size")) or 0
            records_added = _optional_int(payload.get("records_added")) or 0
            top_k = _optional_int(payload.get("top_k")) or 0
            candidate_count = _optional_int(quality_gate.get("candidate_count")) or 0
            eligible_count = _optional_int(quality_gate.get("eligible_count")) or 0
            rejected_count = _optional_int(quality_gate.get("rejected_count")) or 0
            retrieved_count = len(retrieved)
            total_bank_size += bank_size
            total_records_added += records_added
            total_retrieved_count += retrieved_count
            total_top_k_requested += top_k
            for reason, count in reason_counts.items():
                _count(quality_gate_reason_counts, str(reason), _optional_int(count) or 0)
            probability_map_rows = [row for row in retrieved if isinstance(row, dict) and row.get("probability_simplicial_map_available") is True]
            available = bool(retrieved_count > 0 and probability_map_rows)
            insufficient = bool(not available and retrieved_count == 0 and (bank_size == 0 or eligible_count == 0 or rejected_count >= candidate_count))
            status = "retrieval_available" if available else "unavailable_insufficient_model_probability_memory"
            source.update(
                {
                    "available": available,
                    "status": status,
                    "bank_path": payload.get("bank_path", ""),
                    "bank_size": bank_size,
                    "records_added": records_added,
                    "top_k_requested": top_k,
                    "retrieved_count": retrieved_count,
                    "candidate_count": candidate_count,
                    "eligible_count": eligible_count,
                    "rejected_count": rejected_count,
                    "quality_gate_policy": quality_gate.get("policy", "unavailable"),
                    "quality_gate_reason_counts": {str(key): _optional_int(value) or 0 for key, value in reason_counts.items()},
                    "retrieval_weight_keys": sorted(str(key) for key in retrieval_weights.keys()),
                    "probability_simplicial_map_available_count": len(probability_map_rows),
                    "verified_insufficient_memory": insufficient,
                }
            )
            if insufficient:
                verified_insufficient_memory_source_count += 1
                source["reason"] = "no_qualified_model_probability_memory_retrieved"
            elif not available:
                source["reason"] = "retrieved_rows_lack_probability_simplicial_maps"
            _count(status_counts, status)
        elif kind == "topk_maps":
            maps = payload.get("maps", []) if isinstance(payload.get("maps"), list) else []
            topk = payload.get("topk_contract", {}) if isinstance(payload.get("topk_contract"), dict) else {}
            schema_ok = topk.get("schema_version") == "tropicalgt.analogical_topk.v1"
            no_proxy_ok = topk.get("no_proxy_or_fallback") is True
            probability_required = topk.get("retrieval_requires_model_probability_vectors") is True
            embedding_rejected = topk.get("embedding_only_assignment_allowed") is False
            metric_ok = topk.get("assignment_metric") == "jensen_shannon_distance_on_model_probability_vectors"
            query_ok = topk.get("query_complex_required") == "trajectory_probability_filtered_simplicial_object"
            codomain_ok = topk.get("codomain_complex_required") == "trajectory_probability_filtered_simplicial_object"
            map_count = len(maps)
            rendered = _optional_int(topk.get("top_k_rendered"))
            requested = _optional_int(topk.get("top_k_requested"))
            raw_count = _optional_int(topk.get("raw_retrieved_count")) or 0
            qualified_count = _optional_int(topk.get("qualified_model_probability_memory_count")) or 0
            rejected_count = _optional_int(topk.get("rejected_retrieved_count")) or 0
            if rendered is None:
                rendered = map_count
            if requested is not None:
                total_top_k_requested += requested
            total_top_k_rendered += rendered
            if no_proxy_ok:
                no_proxy_contract_source_count += 1
            if probability_required and embedding_rejected and metric_ok:
                probability_vector_contract_source_count += 1
            available = bool(payload.get("available") is True and map_count > 0 and schema_ok and no_proxy_ok and probability_required and embedding_rejected and metric_ok and query_ok and codomain_ok)
            insufficient = bool(not available and no_proxy_ok and probability_required and embedding_rejected and qualified_count == 0)
            status = str(topk.get("status") or payload.get("reason") or ("topk_maps_available" if available else "unavailable_insufficient_model_probability_memory"))
            source.update(
                {
                    "available": available,
                    "status": status,
                    "topk_schema_version": topk.get("schema_version", "unavailable"),
                    "map_count": map_count,
                    "top_k_requested": requested,
                    "top_k_rendered": rendered,
                    "raw_retrieved_count": raw_count,
                    "qualified_model_probability_memory_count": qualified_count,
                    "rejected_retrieved_count": rejected_count,
                    "retrieval_requires_model_probability_vectors": probability_required,
                    "embedding_only_assignment_allowed": bool(topk.get("embedding_only_assignment_allowed", True)),
                    "assignment_metric": topk.get("assignment_metric", "unavailable"),
                    "query_complex_required": topk.get("query_complex_required", "unavailable"),
                    "codomain_complex_required": topk.get("codomain_complex_required", "unavailable"),
                    "query_context_contract_schema_version": topk.get("query_context_contract_schema_version", "unavailable"),
                    "no_proxy_or_fallback": no_proxy_ok,
                    "verified_insufficient_memory": insufficient,
                    "reason_detail": payload.get("reason_detail") or topk.get("reason_detail") or "",
                }
            )
            if insufficient:
                verified_insufficient_memory_source_count += 1
            if not available:
                reasons = []
                if not schema_ok:
                    reasons.append("missing_analogical_topk_schema")
                if not no_proxy_ok:
                    reasons.append("missing_analogical_topk_no_proxy_contract")
                if not probability_required:
                    reasons.append("model_probability_vector_retrieval_not_required")
                if not embedding_rejected:
                    reasons.append("embedding_only_assignment_not_rejected")
                if not metric_ok:
                    reasons.append("wrong_probability_assignment_metric")
                if not query_ok:
                    reasons.append("wrong_query_complex_requirement")
                if not codomain_ok:
                    reasons.append("wrong_codomain_complex_requirement")
                if map_count <= 0:
                    reasons.append("no_rendered_analogical_maps")
                source["reason"] = ";".join(reasons) or status
            _count(status_counts, status)
        else:
            contract = payload.get("contract", {}) if isinstance(payload.get("contract"), dict) else {}
            pairs = payload.get("pairs", []) if isinstance(payload.get("pairs"), list) else []
            schema_ok = contract.get("schema_version") == "tropicalgt.analogical_simplex_tree_analogy.v1"
            no_proxy_ok = contract.get("no_proxy_or_fallback") is True
            source_ok = contract.get("source") == "probability_simplicial_map.simplex_tree_map.rows"
            compares_trees = contract.get("compares_query_and_memory_simplex_trees") is True
            chain_guard = contract.get("chain_map_claim_requires_certified_filtered_simplicial_map") is True
            morphism_guard = contract.get("persistence_module_morphism_claim_requires_certified_filtered_simplicial_map") is True
            pair_count = _optional_int(contract.get("pair_count"))
            if pair_count is None:
                pair_count = len(pairs)
            checked = _optional_int(contract.get("total_checked_simplices")) or 0
            preserved = _optional_int(contract.get("total_preserved_simplices")) or 0
            total_simplex_tree_pair_count += pair_count
            total_checked_simplices += checked
            total_preserved_simplices += preserved
            if no_proxy_ok:
                no_proxy_contract_source_count += 1
            available = bool(contract.get("available") is True and pair_count > 0 and schema_ok and no_proxy_ok and source_ok and compares_trees and chain_guard and morphism_guard)
            insufficient = bool(not available and no_proxy_ok and pair_count == 0 and str(contract.get("status", "")).startswith("unavailable"))
            status = str(contract.get("status") or ("simplex_tree_analogy_available" if available else "unavailable_insufficient_model_probability_memory"))
            source.update(
                {
                    "available": available,
                    "status": status,
                    "contract_schema_version": contract.get("schema_version", "unavailable"),
                    "contract_available": bool(contract.get("available", False)),
                    "pair_count": pair_count,
                    "total_checked_simplices": checked,
                    "total_preserved_simplices": preserved,
                    "source_rows": contract.get("source", "unavailable"),
                    "no_proxy_or_fallback": no_proxy_ok,
                    "compares_query_and_memory_simplex_trees": compares_trees,
                    "renders_hasse_face_to_coface_rows": bool(contract.get("renders_hasse_face_to_coface_rows", False)),
                    "preserved_face_coface_chains_highlighted": bool(contract.get("preserved_face_coface_chains_highlighted", False)),
                    "failed_or_distorted_chains_labeled_not_maps": bool(contract.get("failed_or_distorted_chains_labeled_not_maps", False)),
                    "chain_map_claim_requires_certified_filtered_simplicial_map": chain_guard,
                    "persistence_module_morphism_claim_requires_certified_filtered_simplicial_map": morphism_guard,
                    "verified_insufficient_memory": insufficient,
                    "reason_detail": contract.get("reason_detail", ""),
                }
            )
            if insufficient:
                verified_insufficient_memory_source_count += 1
            if not available:
                reasons = []
                if not schema_ok:
                    reasons.append("missing_simplex_tree_analogy_schema")
                if not no_proxy_ok:
                    reasons.append("missing_simplex_tree_analogy_no_proxy_contract")
                if not source_ok:
                    reasons.append("wrong_simplex_tree_source_rows")
                if pair_count <= 0:
                    reasons.append("no_simplex_tree_analogy_pairs")
                if not chain_guard or not morphism_guard:
                    reasons.append("uncertified_chain_or_morphism_claims_allowed")
                source["reason"] = ";".join(reasons) or status
            _count(status_counts, status)
        sources.append(source)

    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_analogical_memory_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "verified_insufficient_memory_source_count": verified_insufficient_memory_source_count,
        "total_bank_size": total_bank_size,
        "total_records_added": total_records_added,
        "total_retrieved_count": total_retrieved_count,
        "total_top_k_requested": total_top_k_requested,
        "total_top_k_rendered": total_top_k_rendered,
        "total_simplex_tree_pair_count": total_simplex_tree_pair_count,
        "total_checked_simplices": total_checked_simplices,
        "total_preserved_simplices": total_preserved_simplices,
        "no_proxy_contract_source_count": no_proxy_contract_source_count,
        "probability_vector_contract_source_count": probability_vector_contract_source_count,
        "required_topk_schema": "tropicalgt.analogical_topk.v1",
        "required_simplex_tree_schema": "tropicalgt.analogical_simplex_tree_analogy.v1",
        "required_assignment_metric": "jensen_shannon_distance_on_model_probability_vectors",
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "quality_gate_reason_counts": {key: quality_gate_reason_counts[key] for key in sorted(quality_gate_reason_counts)},
        "sources": sources,
        "policy": "Herschel reports analogical memory only from recorded retrieval, top-k simplicial-map, and simplex-tree analogy sidecars. Empty or unqualified memory banks are explicit insufficient-memory evidence, not a proxy analogy or restart justification.",
    }


def _simplicial_complex_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    total_view_count = 0
    total_step_count = 0
    total_vertices = 0
    total_edges = 0
    total_faces = 0
    total_source_simplices = 0
    total_displayed_simplices = 0
    total_radius_slider_contracts = 0
    total_simplex_tree_poset_contracts = 0

    def _count(key: str) -> None:
        if key:
            status_counts[str(key)] = status_counts.get(str(key), 0) + 1

    def _int(value: Any) -> int:
        return _optional_int(value) or 0

    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if lower.endswith("got_full_trajectory_complex_payload.json"):
            kind = "full_trajectory_complex_payload"
        elif lower.endswith("slider_contract.json") and "trajectory_complex" in lower:
            kind = "radius_slider_contract"
        elif lower.endswith("simplex_tree_poset_contract.json"):
            kind = "simplex_tree_poset_contract"
        elif lower.endswith("reasoning_step_complex_maps/manifest.json"):
            kind = "reasoning_step_manifest"
        else:
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"kind": kind, "path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = f"{kind}_sidecar_missing"
            _count(source["reason"])
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser message is platform-dependent
            source["reason"] = f"{kind}_sidecar_parse_error:{exc}"
            _count(f"{kind}_sidecar_parse_error")
            sources.append(source)
            continue
        if not isinstance(payload, dict):
            source["reason"] = f"{kind}_sidecar_not_object"
            _count(source["reason"])
            sources.append(source)
            continue

        if kind == "full_trajectory_complex_payload":
            contract = payload.get("trajectory_complex_overlay_contract", {}) if isinstance(payload.get("trajectory_complex_overlay_contract"), dict) else {}
            views = [view for view in (contract.get("embedding_view"), contract.get("probability_view")) if isinstance(view, dict)]
            available_views = [view for view in views if view.get("available") is True and view.get("actual_data_only") is True and view.get("no_proxy_or_fallback") is True and view.get("safe_to_render_overlay_semantics") is True]
            objects = [payload.get("filtered_simplicial_object"), payload.get("probability_filtered_simplicial_object")]
            object_summaries = [obj.get("summary", {}) for obj in objects if isinstance(obj, dict) and isinstance(obj.get("summary"), dict)]
            simplex_trees = [obj.get("simplex_tree", {}) for obj in objects if isinstance(obj, dict) and isinstance(obj.get("simplex_tree"), dict)]
            vertex_count = sum(_int(summary.get("num_vertices")) for summary in object_summaries)
            edge_count = sum(_int(summary.get("num_edges")) for summary in object_summaries)
            face_count = sum(_int(summary.get("num_two_simplices")) for summary in object_summaries)
            source_simplices = sum(_int(tree.get("num_simplices")) for tree in simplex_trees)
            total_view_count += len(available_views)
            total_vertices += vertex_count
            total_edges += edge_count
            total_faces += face_count
            total_source_simplices += source_simplices
            schema_ok = contract.get("schema_version") == "tropicalgt.trajectory_complex_overlay_contract.v1"
            safe = bool(schema_ok and contract.get("actual_data_only") is True and contract.get("no_proxy_or_fallback") is True and contract.get("safe_to_render_available_views") is True and contract.get("solid_lines_reserved_for_radius_simplices") is True and contract.get("filled_faces_reserved_for_radius_simplices") is True and contract.get("dotted_lines_reserved_for_trajectory_decoding_order_overlays") is True and len(available_views) > 0)
            source.update(
                {
                    "available": safe,
                    "status": "full_trajectory_complex_available" if safe else "full_trajectory_complex_unavailable",
                    "contract_schema_version": contract.get("schema_version", "unavailable"),
                    "available_view_count": len(available_views),
                    "view_count": len(views),
                    "vertex_count": vertex_count,
                    "edge_count": edge_count,
                    "face_count": face_count,
                    "source_simplex_count": source_simplices,
                    "safe_to_render_available_views": bool(contract.get("safe_to_render_available_views", False)),
                    "no_proxy_or_fallback": bool(contract.get("no_proxy_or_fallback", False)),
                }
            )
            if not safe:
                source["reason"] = "trajectory_complex_overlay_contract_unavailable_or_unsafe"
            _count(str(source["status"]))
        elif kind == "radius_slider_contract":
            total_radius_slider_contracts += 1
            safe = bool(payload.get("schema_version") == "tropicalgt.radius_filtration_slider_contract.v1" and payload.get("actual_data_only") is True and payload.get("no_proxy_or_fallback") is True and payload.get("radius_filtration") is True and payload.get("threshold_order") == "ascending_min_to_max" and payload.get("thresholds_ascending") is True and payload.get("first_frame_disjoint_vertices_only") is True and payload.get("initial_radius_frame_hides_solid_edges_and_faces") is True and payload.get("monotone_visible_counts") is True and payload.get("monotone_solid_radius_edges") is True and payload.get("monotone_filled_radius_faces") is True)
            total_vertices += _int(payload.get("first_frame_vertex_count"))
            total_edges += _int(payload.get("last_frame_solid_edge_count"))
            total_faces += _int(payload.get("last_frame_filled_face_count"))
            source.update(
                {
                    "available": safe,
                    "status": "radius_slider_available" if safe else "radius_slider_unavailable",
                    "contract_schema_version": payload.get("schema_version", "unavailable"),
                    "threshold_count": _int(payload.get("threshold_count")),
                    "frame_count": _int(payload.get("frame_count")),
                    "first_frame_vertex_count": _int(payload.get("first_frame_vertex_count")),
                    "last_frame_solid_edge_count": _int(payload.get("last_frame_solid_edge_count")),
                    "last_frame_filled_face_count": _int(payload.get("last_frame_filled_face_count")),
                    "first_frame_disjoint_vertices_only": bool(payload.get("first_frame_disjoint_vertices_only", False)),
                    "thresholds_ascending": bool(payload.get("thresholds_ascending", False)),
                    "monotone_visible_counts": bool(payload.get("monotone_visible_counts", False)),
                    "no_proxy_or_fallback": bool(payload.get("no_proxy_or_fallback", False)),
                }
            )
            if not safe:
                source["reason"] = "radius_slider_contract_unavailable_or_unsafe"
            _count(str(source["status"]))
        elif kind == "simplex_tree_poset_contract":
            total_simplex_tree_poset_contracts += 1
            safe = bool(payload.get("schema_version") == "tropicalgt.simplex_tree_poset.v1" and payload.get("available") is True and payload.get("actual_data_only") is True and payload.get("no_proxy_or_fallback") is True and payload.get("backend") == "gudhi.SimplexTree" and payload.get("safe_to_render_simplex_tree") is True and payload.get("not_disconnected_simplex_columns") is True and payload.get("primary_edges") == "actual_face_to_coface_covers" and payload.get("all_non_vertex_simplices_have_face_cover_edges") is True)
            displayed = _int(payload.get("displayed_simplex_count"))
            source_count = _int(payload.get("source_simplex_count"))
            total_displayed_simplices += displayed
            total_source_simplices += source_count
            source.update(
                {
                    "available": safe,
                    "status": "simplex_tree_poset_available" if safe else "simplex_tree_poset_unavailable",
                    "contract_schema_version": payload.get("schema_version", "unavailable"),
                    "backend": payload.get("backend", "unavailable"),
                    "displayed_simplex_count": displayed,
                    "source_simplex_count": source_count,
                    "actual_face_to_coface_cover_edges": _int(payload.get("actual_face_to_coface_cover_edges")),
                    "empty_simplex_root_present": bool(payload.get("empty_simplex_root_present", False)),
                    "not_disconnected_simplex_columns": bool(payload.get("not_disconnected_simplex_columns", False)),
                    "primary_edges": payload.get("primary_edges", "unavailable"),
                    "truncated": bool(payload.get("truncated", False)),
                    "no_proxy_or_fallback": bool(payload.get("no_proxy_or_fallback", False)),
                }
            )
            if not safe:
                source["reason"] = "simplex_tree_poset_contract_unavailable_or_unsafe"
            _count(str(source["status"]))
        else:
            contract = payload.get("contract", {}) if isinstance(payload.get("contract"), dict) else {}
            steps = payload.get("steps", []) if isinstance(payload.get("steps"), list) else []
            step_count = _int(contract.get("step_count")) or len(steps)
            total_step_count += step_count
            safe = bool(contract.get("schema_version") == "tropicalgt.reasoning_step_complex_maps.v1" and contract.get("available") is True and contract.get("actual_data_only") is True and contract.get("no_proxy_or_fallback") is True and contract.get("all_steps_have_source_contracts") is True and contract.get("all_step_complex_source_contracts_safe") is True and contract.get("all_step_radius_sliders_start_disjoint_vertices") is True and contract.get("all_step_radius_sliders_monotone") is True and contract.get("all_step_radius_sliders_safe_to_render") is True and contract.get("all_step_simplex_tree_posets_use_gudhi") is True and contract.get("all_step_simplex_tree_posets_face_coface_primary") is True and contract.get("all_step_simplex_tree_posets_safe_to_render") is True and _int(contract.get("radius_slider_unavailable_count")) == 0 and _int(contract.get("simplex_tree_poset_unavailable_count")) == 0 and _int(contract.get("source_contract_unavailable_count")) == 0)
            source.update(
                {
                    "available": safe,
                    "status": "reasoning_step_manifest_available" if safe else "reasoning_step_manifest_unavailable",
                    "contract_schema_version": contract.get("schema_version", "unavailable"),
                    "step_count": step_count,
                    "rendered_complex_pages": _int(contract.get("rendered_complex_pages")),
                    "rendered_simplex_tree_pages": _int(contract.get("rendered_simplex_tree_pages")),
                    "rendered_slider_contracts": _int(contract.get("rendered_slider_contracts")),
                    "rendered_simplex_tree_poset_contracts": _int(contract.get("rendered_simplex_tree_poset_contracts")),
                    "gudhi_simplex_tree_step_count": _int(contract.get("gudhi_simplex_tree_step_count")),
                    "radius_slider_unavailable_count": _int(contract.get("radius_slider_unavailable_count")),
                    "simplex_tree_poset_unavailable_count": _int(contract.get("simplex_tree_poset_unavailable_count")),
                    "source_contract_unavailable_count": _int(contract.get("source_contract_unavailable_count")),
                    "all_step_complex_fingerprints_present": bool(contract.get("all_step_complex_fingerprints_present", False)),
                    "all_step_complex_fingerprints_unique": bool(contract.get("all_step_complex_fingerprints_unique", False)),
                    "no_proxy_or_fallback": bool(contract.get("no_proxy_or_fallback", False)),
                }
            )
            if not safe:
                source["reason"] = "reasoning_step_complex_manifest_unavailable_or_unsafe"
            _count(str(source["status"]))
        sources.append(source)

    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_simplicial_complex_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "total_view_count": total_view_count,
        "total_step_count": total_step_count,
        "total_vertices": total_vertices,
        "total_edges": total_edges,
        "total_faces": total_faces,
        "total_source_simplices": total_source_simplices,
        "total_displayed_simplices": total_displayed_simplices,
        "total_radius_slider_contracts": total_radius_slider_contracts,
        "total_simplex_tree_poset_contracts": total_simplex_tree_poset_contracts,
        "required_overlay_schema": "tropicalgt.trajectory_complex_overlay_contract.v1",
        "required_slider_schema": "tropicalgt.radius_filtration_slider_contract.v1",
        "required_simplex_tree_schema": "tropicalgt.simplex_tree_poset.v1",
        "required_step_manifest_schema": "tropicalgt.reasoning_step_complex_maps.v1",
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "sources": sources,
        "policy": "Herschel reports simplicial/radius/simplex-tree evidence only from recorded trajectory-complex payloads, radius-slider contracts, simplex-tree poset contracts, and reasoning-step complex manifests. Unsafe or missing contracts remain unavailable and are not replaced by global trajectory proxies.",
    }


def _topological_algebra_evidence(sidecar_paths: list[str]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    total_growth_rows = 0
    total_topology_reports = 0
    total_probability_topology_reports = 0
    total_persistence_intervals = 0
    total_finite_intervals = 0
    total_chain_group_rank_entries = 0
    total_boundary_maps = 0
    total_multiparameter_fiber_rows = 0
    total_rank_invariant_samples = 0
    total_chain_generators = 0

    def _count(key: str) -> None:
        if key:
            status_counts[str(key)] = status_counts.get(str(key), 0) + 1

    def _int(value: Any) -> int:
        return _optional_int(value) or 0

    def _report_stats(report: Any) -> dict[str, Any]:
        if not isinstance(report, dict) or not report:
            return {"available": False, "reason": "topological_algebra_report_missing"}
        chain = report.get("chain_complex", {}) if isinstance(report.get("chain_complex"), dict) else {}
        persistence = report.get("persistence", {}) if isinstance(report.get("persistence"), dict) else {}
        intervals = persistence.get("intervals", []) if isinstance(persistence.get("intervals"), list) else []
        finite = [row for row in intervals if isinstance(row, (list, tuple)) and len(row) >= 2 and str(row[1]).lower() not in {"inf", "infinity"}]
        multi = report.get("multiparameter_persistence", {}) if isinstance(report.get("multiparameter_persistence"), dict) else {}
        two_param = report.get("two_parameter_persistence", {}) if isinstance(report.get("two_parameter_persistence"), dict) else {}
        fiber_rows = multi.get("fiber_rank_profile", []) if isinstance(multi.get("fiber_rank_profile"), list) else []
        if not fiber_rows and isinstance(two_param.get("fiber_rank_profile"), list):
            fiber_rows = two_param.get("fiber_rank_profile", [])
        rank_samples = multi.get("rank_invariant_samples", []) if isinstance(multi.get("rank_invariant_samples"), list) else []
        if not rank_samples and isinstance(two_param.get("rank_invariant_samples"), list):
            rank_samples = two_param.get("rank_invariant_samples", [])
        generators = multi.get("chain_module_generators", []) if isinstance(multi.get("chain_module_generators"), list) else []
        if not generators and isinstance(two_param.get("chain_module_generators"), list):
            generators = two_param.get("chain_module_generators", [])
        chain_ranks = chain.get("chain_group_ranks", {}) if isinstance(chain.get("chain_group_ranks"), dict) else {}
        boundary_maps = chain.get("boundary_maps", {}) if isinstance(chain.get("boundary_maps"), dict) else {}
        available = bool(report.get("enabled", True) is not False and chain and (persistence or multi or two_param))
        return {
            "available": available,
            "audit_level": report.get("audit_level", "unavailable"),
            "graph_backend": (report.get("graph_metrics", {}) if isinstance(report.get("graph_metrics"), dict) else {}).get("backend", "unavailable"),
            "persistence_available": bool(persistence.get("available", bool(intervals))),
            "persistence_backend": persistence.get("backend", "unavailable"),
            "persistence_interval_count": len(intervals),
            "finite_persistence_interval_count": len(finite),
            "chain_group_rank_entry_count": len(chain_ranks),
            "boundary_map_count": len(boundary_maps),
            "multiparameter_num_parameters": _int(multi.get("num_parameters") or two_param.get("num_parameters")),
            "multiparameter_fiber_rank_profile_count": len(fiber_rows),
            "rank_invariant_sample_count": len(rank_samples),
            "chain_module_generator_count": len(generators),
            "persistence_representations_available": bool((report.get("persistence_representations", {}) if isinstance(report.get("persistence_representations"), dict) else {}).get("available", False)),
            "reason": "" if available else "topological_algebra_report_unavailable_or_incomplete",
        }

    for raw_path in sidecar_paths:
        lower = raw_path.lower()
        if lower.endswith("trajectory_topological_algebra.json"):
            kind = "trajectory_topological_algebra"
        elif lower.endswith("trajectory_growth_topology.json"):
            kind = "trajectory_growth_topology"
        elif lower.endswith("inference_topology.json"):
            kind = "inference_topology"
        elif lower.endswith("inference_algebra.json"):
            kind = "inference_algebra"
        else:
            continue
        resolved = _resolve_output_path(Path(raw_path))
        source: dict[str, Any] = {"kind": kind, "path": _project_path(resolved), "available": False}
        if not resolved.exists():
            source["reason"] = f"{kind}_sidecar_missing"
            _count(source["reason"])
            sources.append(source)
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser message is platform-dependent
            source["reason"] = f"{kind}_sidecar_parse_error:{exc}"
            _count(f"{kind}_sidecar_parse_error")
            sources.append(source)
            continue
        if kind == "trajectory_growth_topology":
            rows = payload if isinstance(payload, list) else []
            total_growth_rows += len(rows)
            report_stats = []
            probability_stats = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                stats = _report_stats(row.get("topological_algebra"))
                prob_stats = _report_stats(row.get("probability_topological_algebra"))
                report_stats.append(stats)
                probability_stats.append(prob_stats)
            available_reports = [row for row in report_stats if row.get("available")]
            available_probability = [row for row in probability_stats if row.get("available")]
            for stats in available_reports + available_probability:
                total_persistence_intervals += _int(stats.get("persistence_interval_count"))
                total_finite_intervals += _int(stats.get("finite_persistence_interval_count"))
                total_chain_group_rank_entries += _int(stats.get("chain_group_rank_entry_count"))
                total_boundary_maps += _int(stats.get("boundary_map_count"))
                total_multiparameter_fiber_rows += _int(stats.get("multiparameter_fiber_rank_profile_count"))
                total_rank_invariant_samples += _int(stats.get("rank_invariant_sample_count"))
                total_chain_generators += _int(stats.get("chain_module_generator_count"))
            total_topology_reports += len(available_reports)
            total_probability_topology_reports += len(available_probability)
            available = bool(rows and (available_reports or available_probability))
            source.update(
                {
                    "available": available,
                    "status": "trajectory_growth_topology_available" if available else "trajectory_growth_topology_unavailable",
                    "growth_row_count": len(rows),
                    "available_topological_algebra_reports": len(available_reports),
                    "available_probability_topological_algebra_reports": len(available_probability),
                }
            )
            if not available:
                source["reason"] = "trajectory_growth_topology_has_no_available_reports"
            _count(str(source["status"]))
        elif kind == "inference_algebra" and isinstance(payload, dict) and not payload:
            source.update({"available": False, "status": "inference_algebra_empty_unavailable", "reason": "inference_algebra_sidecar_empty"})
            _count(str(source["status"]))
        else:
            stats = _report_stats(payload)
            available = bool(stats.get("available"))
            if available:
                total_topology_reports += 1
                total_persistence_intervals += _int(stats.get("persistence_interval_count"))
                total_finite_intervals += _int(stats.get("finite_persistence_interval_count"))
                total_chain_group_rank_entries += _int(stats.get("chain_group_rank_entry_count"))
                total_boundary_maps += _int(stats.get("boundary_map_count"))
                total_multiparameter_fiber_rows += _int(stats.get("multiparameter_fiber_rank_profile_count"))
                total_rank_invariant_samples += _int(stats.get("rank_invariant_sample_count"))
                total_chain_generators += _int(stats.get("chain_module_generator_count"))
            status = f"{kind}_available" if available else f"{kind}_unavailable"
            source.update({"available": available, "status": status, **stats})
            if not available and not source.get("reason"):
                source["reason"] = stats.get("reason") or f"{kind}_unavailable"
            _count(status)
        sources.append(source)

    available_sources = [row for row in sources if row.get("available")]
    return {
        "schema_version": "tropicalgt.herschel_topological_algebra_evidence.v1",
        "available": bool(available_sources),
        "source_count": len(sources),
        "available_source_count": len(available_sources),
        "total_growth_rows": total_growth_rows,
        "total_topology_reports": total_topology_reports,
        "total_probability_topology_reports": total_probability_topology_reports,
        "total_persistence_intervals": total_persistence_intervals,
        "total_finite_intervals": total_finite_intervals,
        "total_chain_group_rank_entries": total_chain_group_rank_entries,
        "total_boundary_maps": total_boundary_maps,
        "total_multiparameter_fiber_rows": total_multiparameter_fiber_rows,
        "total_rank_invariant_samples": total_rank_invariant_samples,
        "total_chain_module_generators": total_chain_generators,
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "sources": sources,
        "policy": "Herschel reports topological algebra only from recorded trajectory, growth, and inference topology sidecars. Empty inference_algebra sidecars remain unavailable and cannot substitute for chain complexes, persistence intervals, or multiparameter module evidence.",
    }


def summarize_bundle(bundle: dict[str, Any], *, bundle_path: Path | None = None) -> dict[str, Any]:
    decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
    gate = bundle.get("restart_evidence_gate") if isinstance(bundle.get("restart_evidence_gate"), dict) else {}
    checkpoint = bundle.get("checkpoint_evidence") if isinstance(bundle.get("checkpoint_evidence"), dict) else {}
    readiness = bundle.get("execution_readiness") if isinstance(bundle.get("execution_readiness"), dict) else {}
    inventory = bundle.get("artifact_inventory") if isinstance(bundle.get("artifact_inventory"), dict) else {}
    advanced = bundle.get("advanced_bpb_contract") if isinstance(bundle.get("advanced_bpb_contract"), dict) else {}
    sidecars = [str(path) for path in inventory.get("advanced_sidecars_tail", []) if str(path)]
    validator_gap_evidence = _validator_gap_evidence(bundle)
    command_results = [row for row in bundle.get("command_results", []) if isinstance(row, dict)]
    failed_commands = [row for row in command_results if row.get("returncode") not in (0, None) or row.get("timed_out")]
    blockers = [str(item) for item in gate.get("blockers", []) if str(item)]
    checkpoint_warnings = [str(item) for item in checkpoint.get("warnings", []) if str(item)]
    execution_issues = [str(item) for item in readiness.get("issues", []) if str(item)]
    failed_gates = [str(item) for item in advanced.get("failed_gates", []) if str(item)]
    return {
        "schema_version": "tropicalgt.herschel_5k_report_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_path": _project_path(bundle_path) if bundle_path else "",
        "run_identity": {
            "config": bundle.get("config", ""),
            "report": bundle.get("report", ""),
            "checkpoint": bundle.get("checkpoint", ""),
            "stop_record": bundle.get("stop_record", ""),
            "boundary_step": bundle.get("boundary_step"),
            "target_bpb": bundle.get("target_bpb"),
        },
        "primary_metrics": {
            "bpb": decision.get("bpb"),
            "graph_bpb": decision.get("graph_bpb"),
            "target_missed": bool(decision.get("triggered", False)),
            "restart_policy": decision.get("restart_policy", ""),
        },
        "checkpoint_evidence": {
            "available": bool(checkpoint.get("checkpoint_available", False)),
            "restart_safe": bool(checkpoint.get("safe_for_checkpoint_backed_restart", False)),
            "unavailable_reason": checkpoint.get("checkpoint_unavailable_reason", ""),
            "warnings": checkpoint_warnings,
        },
        "execution_evidence": {
            "requested": bool(readiness.get("execution_requested", False)),
            "ready": bool(readiness.get("ready", False)),
            "issues": execution_issues,
            "command_results": len(command_results),
            "failed_commands": failed_commands,
        },
        "advanced_bpb_contract": {
            "safe_for_restart": bool(advanced.get("safe_to_use_for_step0_bpb_restart", False)),
            "failed_gates": failed_gates,
        },
        "artifact_evidence": {
            "latest_got_audit_dir": inventory.get("latest_got_audit_dir", ""),
            "latest_periodic_dir": inventory.get("latest_periodic_dir", ""),
            "advanced_sidecar_count": len(sidecars),
            "sidecar_groups": _sidecar_groups(sidecars),
            "advanced_sidecars_tail": sidecars[:120],
            "validator_gap_evidence": validator_gap_evidence,
            "trajectory_embedding_visual_evidence": _trajectory_embedding_visual_evidence(sidecars),
            "inference_audit_backfill_evidence": _inference_audit_backfill_evidence(sidecars),
            "gflownet_branch_selection_evidence": _gflownet_branch_selection_evidence(sidecars),
            "tropical_support_evidence": _tropical_support_evidence(sidecars),
            "graphcg_direction_evidence": _graphcg_direction_evidence(sidecars),
            "nll_density_evidence": _nll_density_evidence(sidecars),
            "chart_bundle_transport_evidence": _chart_bundle_transport_evidence(sidecars),
            "two_parameter_bifiltration_visual_evidence": _two_parameter_bifiltration_visual_evidence(sidecars),
            "bivariate_module_evidence": _bivariate_module_evidence(sidecars),
            "persistence_landscape_evidence": _persistence_landscape_evidence(sidecars),
            "toric_tropical_cas_evidence": _toric_tropical_cas_evidence(sidecars),
            "analogical_query_context_evidence": _analogical_query_context_evidence(sidecars),
            "analogical_memory_evidence": _analogical_memory_evidence(sidecars),
            "simplicial_complex_evidence": _simplicial_complex_evidence(sidecars),
            "topological_algebra_evidence": _topological_algebra_evidence(sidecars),
        },
        "restart_decision": {
            "action": gate.get("restart_action", "unavailable"),
            "step0_restart_allowed": bool(gate.get("step0_restart_allowed", False)),
            "blocked": bool(gate.get("blocked", False)),
            "blockers": blockers,
        },
        "policy": "CPU-only Herschel report generated from an existing post-5K review bundle; no training, eval, browser, validator, checkpoint load beyond the bundle contents, or GPU command is executed.",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    run = summary["run_identity"]
    metrics = summary["primary_metrics"]
    checkpoint = summary["checkpoint_evidence"]
    execution = summary["execution_evidence"]
    advanced = summary["advanced_bpb_contract"]
    artifacts = summary["artifact_evidence"]
    restart = summary["restart_decision"]
    lines = [
        "# Herschel 5K Evidence Report",
        "",
        f"- Generated: `{summary.get('generated_at')}`",
        f"- Bundle: `{summary.get('bundle_path')}`",
        f"- Boundary step: `{run.get('boundary_step')}`",
        f"- Config: `{run.get('config')}`",
        f"- Report: `{run.get('report')}`",
        f"- Checkpoint: `{run.get('checkpoint')}`",
        f"- Stop record: `{run.get('stop_record')}`",
        "",
        "## Primary Metrics",
        "",
        f"- Target BPB: `< {_fmt(run.get('target_bpb'))}`",
        f"- Observed BPB: `{_fmt(metrics.get('bpb'))}`",
        f"- Observed graph-BPB: `{_fmt(metrics.get('graph_bpb'))}`",
        f"- Target missed: `{metrics.get('target_missed')}`",
        f"- Restart policy: `{metrics.get('restart_policy')}`",
        "",
        "## Evidence Status",
        "",
        f"- Checkpoint available: `{checkpoint.get('available')}`",
        f"- Checkpoint restart-safe: `{checkpoint.get('restart_safe')}`",
        f"- Checkpoint unavailable reason: `{checkpoint.get('unavailable_reason') or 'n/a'}`",
        f"- Execution evidence requested: `{execution.get('requested')}`",
        f"- Execution evidence ready: `{execution.get('ready')}`",
        f"- Command results recorded: `{execution.get('command_results')}`",
        f"- Advanced BPB contract safe: `{advanced.get('safe_for_restart')}`",
        "",
        "## Artifact Evidence",
        "",
        f"- Latest periodic dir: `{artifacts.get('latest_periodic_dir')}`",
        f"- Latest GoT audit dir: `{artifacts.get('latest_got_audit_dir')}`",
        f"- Advanced sidecar count: `{artifacts.get('advanced_sidecar_count')}`",
        "",
        "```json",
        json.dumps(artifacts.get("sidecar_groups", {}), indent=2),
        "```",
        "",
        "## Validator Evidence Gaps",
        "",
    ]
    validator_gaps = artifacts.get("validator_gap_evidence", {}) if isinstance(artifacts.get("validator_gap_evidence"), dict) else {}
    lines.extend(
        [
            f"- Available: `{validator_gaps.get('available', False)}`",
            f"- Sources: `{validator_gaps.get('source_count', 0)}`",
            "",
            "```json",
            json.dumps(validator_gaps.get("combined_category_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    trajectory_embedding = artifacts.get("trajectory_embedding_visual_evidence", {}) if isinstance(artifacts.get("trajectory_embedding_visual_evidence"), dict) else {}
    lines.extend(
        [
            "## Trajectory Embedding Visual Evidence",
            "",
            f"- Available: `{trajectory_embedding.get('available', False)}`",
            f"- Paired trajectory+embedding payloads available: `{trajectory_embedding.get('paired_payloads_available', False)}`",
            f"- Sources: `{trajectory_embedding.get('source_count', 0)}`",
            f"- Trajectory / embedding nodes: `{trajectory_embedding.get('total_trajectory_node_count', 0)}` / `{trajectory_embedding.get('total_embedding_node_count', 0)}`",
            f"- Edges / filtered objects / raw embeddings: `{trajectory_embedding.get('total_edge_count', 0)}` / `{trajectory_embedding.get('total_filtered_simplicial_object_count', 0)}` / `{trajectory_embedding.get('total_raw_model_embedding_count', 0)}`",
            f"- Parent-child transitions: `{trajectory_embedding.get('total_parent_child_transition_count', 0)}`",
            f"- PCA warning sources: `{trajectory_embedding.get('pca_quality_warning_source_count', 0)}`",
            "",
            "```json",
            json.dumps(trajectory_embedding.get("status_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in trajectory_embedding.get("sources", []) if isinstance(trajectory_embedding.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` kind=`{source.get('kind', 'unavailable')}` available=`{source.get('available')}` "
            f"nodes=`{source.get('node_count', 0)}` edges=`{source.get('edge_count', 0)}` raw_embeddings=`{source.get('raw_model_embedding_count', 0)}` "
            f"pca_corr=`{_fmt(source.get('pca_pairwise_distance_correlation'))}` stress=`{_fmt(source.get('pca_normalized_stress'))}` "
            f"warning=`{source.get('pca_quality_warning')}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    inference_audit_backfill = artifacts.get("inference_audit_backfill_evidence", {}) if isinstance(artifacts.get("inference_audit_backfill_evidence"), dict) else {}
    lines.extend(
        [
            "## Inference Audit And Backfill Evidence",
            "",
            f"- Available: `{inference_audit_backfill.get('available', False)}`",
            f"- Sources: `{inference_audit_backfill.get('source_count', 0)}`",
            f"- Inference/backfill available sources: `{inference_audit_backfill.get('inference_audit_available_source_count', 0)}` / `{inference_audit_backfill.get('backfill_available_source_count', 0)}`",
            f"- Inference candidates: `{inference_audit_backfill.get('total_inference_candidate_count', 0)}`",
            f"- Backfill actions / candidates: `{inference_audit_backfill.get('total_backfill_action_count', 0)}` / `{inference_audit_backfill.get('total_backfill_candidate_count', 0)}`",
            f"- Quality gate candidates / eligible / rejected: `{inference_audit_backfill.get('total_quality_gate_candidate_count', 0)}` / `{inference_audit_backfill.get('total_quality_gate_eligible_count', 0)}` / `{inference_audit_backfill.get('total_quality_gate_rejected_count', 0)}`",
            "",
            "```json",
            json.dumps(inference_audit_backfill.get("status_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in inference_audit_backfill.get("sources", []) if isinstance(inference_audit_backfill.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` kind=`{source.get('kind', 'unavailable')}` available=`{source.get('available')}` "
            f"status=`{source.get('status', 'unavailable')}` candidates=`{source.get('candidate_count', 0)}` "
            f"actions=`{source.get('action_count', 0)}` policy_no_fabrication=`{source.get('policy_no_fabrication', 'n/a')}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    gflownet_branch = artifacts.get("gflownet_branch_selection_evidence", {}) if isinstance(artifacts.get("gflownet_branch_selection_evidence"), dict) else {}
    lines.extend(
        [
            "## GFlowNet Branch Selection Evidence",
            "",
            f"- Available: `{gflownet_branch.get('available', False)}`",
            f"- Sources: `{gflownet_branch.get('source_count', 0)}`",
            f"- Valid branch rows: `{gflownet_branch.get('total_valid_branch_selection_rows', 0)}`",
            f"- Selected actions: `{gflownet_branch.get('total_selected_actions', 0)}`",
            "",
            "```json",
            json.dumps(gflownet_branch.get("policy_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in gflownet_branch.get("sources", []) if isinstance(gflownet_branch.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` rows=`{source.get('valid_branch_selection_row_count', 0)}` "
            f"actions=`{source.get('selected_action_count', 0)}` policies=`{source.get('policy_counts', {})}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    tropical_support = artifacts.get("tropical_support_evidence", {}) if isinstance(artifacts.get("tropical_support_evidence"), dict) else {}
    lines.extend(
        [
            "## Tropical Support Evidence",
            "",
            f"- Available: `{tropical_support.get('available', False)}`",
            f"- Sources: `{tropical_support.get('source_count', 0)}`",
            f"- Total tokens: `{tropical_support.get('total_token_count', 0)}`",
            f"- Valid support assignments: `{tropical_support.get('total_valid_support_assignment_count', 0)}`",
            "",
            "```json",
            json.dumps(tropical_support.get("support_probability_source_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in tropical_support.get("sources", []) if isinstance(tropical_support.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` source=`{source.get('support_probability_source', 'unavailable')}` "
            f"tokens=`{source.get('token_count')}` strict=`{_fmt(source.get('strict_wall_hit_rate'))}` near=`{_fmt(source.get('near_wall_hit_rate'))}` "
            f"status=`{source.get('low_strict_wall_interpretation_status', 'unavailable')}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    graphcg_direction = artifacts.get("graphcg_direction_evidence", {}) if isinstance(artifacts.get("graphcg_direction_evidence"), dict) else {}
    lines.extend(
        [
            "## GraphCG Direction Evidence",
            "",
            f"- Available: `{graphcg_direction.get('available', False)}`",
            f"- Sources: `{graphcg_direction.get('source_count', 0)}`",
            f"- Total directions: `{graphcg_direction.get('total_direction_count', 0)}`",
            f"- Total candidates: `{graphcg_direction.get('total_candidate_count', 0)}`",
            "",
            "```json",
            json.dumps(graphcg_direction.get("basis_source_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in graphcg_direction.get("sources", []) if isinstance(graphcg_direction.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` basis=`{source.get('projection_basis', 'unavailable')}` "
            f"directions=`{source.get('direction_count')}` candidates=`{source.get('candidate_count')}` "
            f"active_nonzero=`{source.get('active_rank_nonzero_mean_abs')}` mean_abs_p90=`{_fmt(source.get('mean_abs_p90'))}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    nll_density = artifacts.get("nll_density_evidence", {}) if isinstance(artifacts.get("nll_density_evidence"), dict) else {}
    lines.extend(
        [
            "## NLL Density Evidence",
            "",
            f"- Available: `{nll_density.get('available', False)}`",
            f"- Sources: `{nll_density.get('source_count', 0)}`",
            f"- Actual model anchors: `{nll_density.get('total_actual_model_anchor_count', 0)}`",
            f"- Support samples: `{nll_density.get('total_support_sample_count', 0)}`",
            "",
            "```json",
            json.dumps(nll_density.get("visible_density_layer_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in nll_density.get("sources", []) if isinstance(nll_density.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` anchors=`{source.get('actual_model_anchor_count')}` "
            f"samples=`{source.get('support_sample_count')}` bandwidth=`{_fmt(source.get('kernel_bandwidth'))}` nll_span=`{_fmt(source.get('nll_span'))}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    bivariate_module = artifacts.get("bivariate_module_evidence", {}) if isinstance(artifacts.get("bivariate_module_evidence"), dict) else {}
    lines.extend(
        [
            "## Bivariate Module Evidence",
            "",
            f"- Module available: `{bivariate_module.get('available', False)}`",
            f"- Sources: `{bivariate_module.get('source_count', 0)}`",
            f"- Certified real free resolutions: `{bivariate_module.get('certified_real_free_resolution_source_count', 0)}`",
            f"- Safe unavailable real free resolutions: `{bivariate_module.get('safe_unavailable_real_free_resolution_source_count', 0)}`",
            f"- Total fibers: `{bivariate_module.get('total_fiber_rank_profile_count', 0)}`",
            f"- Total structure maps: `{bivariate_module.get('total_structure_map_count', 0)}`",
            "",
            "```json",
            json.dumps(bivariate_module.get("resolution_status_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in bivariate_module.get("sources", []) if isinstance(bivariate_module.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` module_available=`{source.get('available')}` ring=`{source.get('coefficient_ring', 'unavailable')}` "
            f"fibers=`{source.get('fiber_rank_profile_count', 0)}` maps=`{source.get('structure_map_count', 0)}` "
            f"real_resolution=`{source.get('real_free_resolution_certified')}` status=`{source.get('real_free_resolution_status', 'unavailable')}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    two_parameter_bifiltration = artifacts.get("two_parameter_bifiltration_visual_evidence", {}) if isinstance(artifacts.get("two_parameter_bifiltration_visual_evidence"), dict) else {}
    lines.extend(
        [
            "## Two-Parameter Bifiltration Visual Evidence",
            "",
            f"- Available: `{two_parameter_bifiltration.get('available', False)}`",
            f"- Sources: `{two_parameter_bifiltration.get('source_count', 0)}`",
            f"- Staircase cards / primary cards: `{two_parameter_bifiltration.get('total_staircase_card_count', 0)}` / `{two_parameter_bifiltration.get('total_primary_staircase_card_count', 0)}`",
            f"- Actual generators / minimal antichain / labels: `{two_parameter_bifiltration.get('total_actual_generator_bidegree_count', 0)}` / `{two_parameter_bifiltration.get('total_minimal_antichain_count', 0)}` / `{two_parameter_bifiltration.get('total_generator_label_count', 0)}`",
            f"- Upward regions / quotient basis points: `{two_parameter_bifiltration.get('total_upward_closed_region_count', 0)}` / `{two_parameter_bifiltration.get('total_quotient_basis_lattice_count', 0)}`",
            f"- Hilbert numerator terms / adjacent LCM syzygies: `{two_parameter_bifiltration.get('total_hilbert_numerator_term_count', 0)}` / `{two_parameter_bifiltration.get('total_adjacent_lcm_syzygy_count', 0)}`",
            f"- Structure maps / rank samples / grid fibers: `{two_parameter_bifiltration.get('total_structure_map_count', 0)}` / `{two_parameter_bifiltration.get('total_rank_invariant_sample_count', 0)}` / `{two_parameter_bifiltration.get('total_grid_fiber_row_count', 0)}`",
            "",
            "```json",
            json.dumps(two_parameter_bifiltration.get("status_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in two_parameter_bifiltration.get("sources", []) if isinstance(two_parameter_bifiltration.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` view=`{source.get('primary_view', 'unavailable')}` "
            f"ring=`{source.get('coefficient_ring', 'unavailable')}` cards=`{source.get('staircase_card_count', 0)}` "
            f"quotient_basis=`{source.get('quotient_basis_lattice_count', 0)}` hilbert=`{source.get('hilbert_numerator_term_count', 0)}` "
            f"lcm_syzygies=`{source.get('adjacent_lcm_syzygy_count', 0)}` maps=`{source.get('structure_map_count', 0)}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    persistence_landscape = artifacts.get("persistence_landscape_evidence", {}) if isinstance(artifacts.get("persistence_landscape_evidence"), dict) else {}
    lines.extend(
        [
            "## Persistence Landscape Evidence",
            "",
            f"- Available: `{persistence_landscape.get('available', False)}`",
            f"- Sources: `{persistence_landscape.get('source_count', 0)}`",
            f"- Verified unavailable sources: `{persistence_landscape.get('verified_unavailable_source_count', 0)}`",
            f"- Total landscape rows: `{persistence_landscape.get('total_landscape_row_count', 0)}`",
            f"- Total curve traces: `{persistence_landscape.get('total_curve_trace_count', 0)}`",
            "",
            "```json",
            json.dumps(persistence_landscape.get("backend_counts", {}) or persistence_landscape.get("unavailable_reason_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in persistence_landscape.get("sources", []) if isinstance(persistence_landscape.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` verified_unavailable=`{source.get('verified_unavailable')}` "
            f"backend=`{source.get('landscape_backend', 'unavailable')}` rows=`{source.get('landscape_row_count', 0)}` "
            f"curves=`{source.get('curve_trace_count', 0)}` finite_intervals=`{source.get('finite_persistence_interval_count', 0)}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    chart_bundle_transport = artifacts.get("chart_bundle_transport_evidence", {}) if isinstance(artifacts.get("chart_bundle_transport_evidence"), dict) else {}
    lines.extend(
        [
            "## Chart/Vector-Bundle Evidence",
            "",
            f"- Available: `{chart_bundle_transport.get('available', False)}`",
            f"- Sources: `{chart_bundle_transport.get('source_count', 0)}`",
            f"- Paper-ready telemetry sources: `{chart_bundle_transport.get('paper_ready_source_count', 0)}`",
            f"- Total charts: `{chart_bundle_transport.get('total_chart_count', 0)}`",
            f"- Total transports: `{chart_bundle_transport.get('total_transport_count', 0)}`",
            "",
            "```json",
            json.dumps(chart_bundle_transport.get("completeness_tier_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in chart_bundle_transport.get("sources", []) if isinstance(chart_bundle_transport.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` tier=`{source.get('completeness_tier', 'unavailable')}` "
            f"paper_ready=`{source.get('paper_ready')}` charts=`{source.get('chart_count')}` transports=`{source.get('transport_count')}` "
            f"missing=`{source.get('missing_required_groups', [])}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    toric_tropical_cas = artifacts.get("toric_tropical_cas_evidence", {}) if isinstance(artifacts.get("toric_tropical_cas_evidence"), dict) else {}
    lines.extend(
        [
            "## Toric/Tropical CAS Evidence",
            "",
            f"- Available: `{toric_tropical_cas.get('available', False)}`",
            f"- Sources: `{toric_tropical_cas.get('source_count', 0)}`",
            f"- Certified finite toric ideals: `{toric_tropical_cas.get('certified_finite_toric_ideal_count', 0)}`",
            f"- Certified tropical fans: `{toric_tropical_cas.get('certified_tropical_fan_count', 0)}`",
            f"- Forbidden global/tropical/normal-fan claim count: `{toric_tropical_cas.get('forbidden_global_claim_count', 0)}`",
            "",
            "```json",
            json.dumps(toric_tropical_cas.get("status_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in toric_tropical_cas.get("sources", []) if isinstance(toric_tropical_cas.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` kind=`{source.get('kind', 'unavailable')}` available=`{source.get('available')}` "
            f"status=`{source.get('status', 'unavailable')}` backend=`{source.get('backend', 'unavailable')}` "
            f"toric_ideal=`{source.get('toric_ideal_certified', False)}` fan=`{source.get('fan_diagnostics_certified', False)}` "
            f"rays=`{source.get('ray_count', 0)}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    analogical_query_context = artifacts.get("analogical_query_context_evidence", {}) if isinstance(artifacts.get("analogical_query_context_evidence"), dict) else {}
    lines.extend(
        [
            "## Analogical Query Context Evidence",
            "",
            f"- Available: `{analogical_query_context.get('available', False)}`",
            f"- Sources: `{analogical_query_context.get('source_count', 0)}`",
            f"- Required contract: `{analogical_query_context.get('required_contract_schema', 'unavailable')}`",
            "",
        ]
    )
    for source in analogical_query_context.get("sources", []) if isinstance(analogical_query_context.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` available=`{source.get('available')}` selected=`{source.get('selected_query_complex_source', 'unavailable')}` "
            f"status=`{source.get('conversion_status', 'unavailable')}` prob_vertices=`{source.get('selected_query_probability_vertex_count', 0)}` rejected=`{source.get('rejected_query_context_keys', [])}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    analogical_memory = artifacts.get("analogical_memory_evidence", {}) if isinstance(artifacts.get("analogical_memory_evidence"), dict) else {}
    lines.extend(
        [
            "## Analogical Memory Evidence",
            "",
            f"- Available: `{analogical_memory.get('available', False)}`",
            f"- Sources: `{analogical_memory.get('source_count', 0)}`",
            f"- Verified insufficient-memory sources: `{analogical_memory.get('verified_insufficient_memory_source_count', 0)}`",
            f"- Retrieved memories: `{analogical_memory.get('total_retrieved_count', 0)}`",
            f"- Rendered top-k maps: `{analogical_memory.get('total_top_k_rendered', 0)}`",
            f"- Simplex-tree analogy pairs: `{analogical_memory.get('total_simplex_tree_pair_count', 0)}`",
            f"- Required assignment metric: `{analogical_memory.get('required_assignment_metric', 'unavailable')}`",
            "",
            "```json",
            json.dumps(analogical_memory.get("status_counts", {}) or analogical_memory.get("quality_gate_reason_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in analogical_memory.get("sources", []) if isinstance(analogical_memory.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` kind=`{source.get('kind', 'unavailable')}` available=`{source.get('available')}` "
            f"status=`{source.get('status', 'unavailable')}` retrieved=`{source.get('retrieved_count', source.get('raw_retrieved_count', 0))}` "
            f"topk_rendered=`{source.get('top_k_rendered', 0)}` pairs=`{source.get('pair_count', 0)}` "
            f"verified_insufficient=`{source.get('verified_insufficient_memory')}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    simplicial_complex = artifacts.get("simplicial_complex_evidence", {}) if isinstance(artifacts.get("simplicial_complex_evidence"), dict) else {}
    lines.extend(
        [
            "## Simplicial Complex And Simplex-Tree Evidence",
            "",
            f"- Available: `{simplicial_complex.get('available', False)}`",
            f"- Sources: `{simplicial_complex.get('source_count', 0)}`",
            f"- Radius-slider contracts: `{simplicial_complex.get('total_radius_slider_contracts', 0)}`",
            f"- Simplex-tree poset contracts: `{simplicial_complex.get('total_simplex_tree_poset_contracts', 0)}`",
            f"- Reasoning steps: `{simplicial_complex.get('total_step_count', 0)}`",
            f"- Vertices/edges/faces counted: `{simplicial_complex.get('total_vertices', 0)}` / `{simplicial_complex.get('total_edges', 0)}` / `{simplicial_complex.get('total_faces', 0)}`",
            f"- Source/displayed simplices: `{simplicial_complex.get('total_source_simplices', 0)}` / `{simplicial_complex.get('total_displayed_simplices', 0)}`",
            "",
            "```json",
            json.dumps(simplicial_complex.get("status_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in simplicial_complex.get("sources", []) if isinstance(simplicial_complex.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` kind=`{source.get('kind', 'unavailable')}` available=`{source.get('available')}` "
            f"status=`{source.get('status', 'unavailable')}` steps=`{source.get('step_count', 0)}` "
            f"thresholds=`{source.get('threshold_count', 0)}` simplices=`{source.get('source_simplex_count', source.get('displayed_simplex_count', 0))}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    topological_algebra = artifacts.get("topological_algebra_evidence", {}) if isinstance(artifacts.get("topological_algebra_evidence"), dict) else {}
    lines.extend(
        [
            "## Topological Algebra Evidence",
            "",
            f"- Available: `{topological_algebra.get('available', False)}`",
            f"- Sources: `{topological_algebra.get('source_count', 0)}`",
            f"- Growth rows: `{topological_algebra.get('total_growth_rows', 0)}`",
            f"- Topology reports: `{topological_algebra.get('total_topology_reports', 0)}`",
            f"- Probability-topology reports: `{topological_algebra.get('total_probability_topology_reports', 0)}`",
            f"- Persistence intervals finite/total: `{topological_algebra.get('total_finite_intervals', 0)}` / `{topological_algebra.get('total_persistence_intervals', 0)}`",
            f"- Chain rank entries / boundary maps: `{topological_algebra.get('total_chain_group_rank_entries', 0)}` / `{topological_algebra.get('total_boundary_maps', 0)}`",
            f"- Multiparameter fibers / rank samples / generators: `{topological_algebra.get('total_multiparameter_fiber_rows', 0)}` / `{topological_algebra.get('total_rank_invariant_samples', 0)}` / `{topological_algebra.get('total_chain_module_generators', 0)}`",
            "",
            "```json",
            json.dumps(topological_algebra.get("status_counts", {}), indent=2),
            "```",
            "",
        ]
    )
    for source in topological_algebra.get("sources", []) if isinstance(topological_algebra.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(
            f"- `{source.get('path', '')}` kind=`{source.get('kind', 'unavailable')}` available=`{source.get('available')}` "
            f"status=`{source.get('status', 'unavailable')}` reports=`{source.get('available_topological_algebra_reports', int(bool(source.get('available'))))}` "
            f"probability_reports=`{source.get('available_probability_topological_algebra_reports', 0)}` intervals=`{source.get('persistence_interval_count', 0)}` "
            f"fibers=`{source.get('multiparameter_fiber_rank_profile_count', 0)}`"
        )
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    ranked_validator_categories = validator_gaps.get("ranked_categories", []) if isinstance(validator_gaps.get("ranked_categories"), list) else []
    if ranked_validator_categories:
        lines.extend(["### Required Actions", ""])
        for row in ranked_validator_categories[:12]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- `{row.get('category')}` count=`{row.get('count')}` action={row.get('required_action') or 'unavailable'}")
            for example_row in row.get("examples", [])[:3] if isinstance(row.get("examples"), list) else []:
                if not isinstance(example_row, dict):
                    continue
                lines.append(f"  - example: `{example_row.get('example')}`")
    for source in validator_gaps.get("sources", []) if isinstance(validator_gaps.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        lines.append(f"- `{source.get('name', 'validator')}` path=`{source.get('path', '')}` available=`{source.get('available')}` gaps=`{source.get('gap_count')}`")
        if source.get("reason"):
            lines.append(f"  - reason: `{source.get('reason')}`")
    lines.extend(
        [
            "",
            "## Restart Decision",
        "",
        f"- Action: `{restart.get('action')}`",
        f"- Step-0 restart allowed: `{restart.get('step0_restart_allowed')}`",
        f"- Blocked: `{restart.get('blocked')}`",
        "",
        "```mermaid",
        "flowchart TD",
        "  A[5K review bundle] --> B{BPB target missed?}",
        f"  B -->|{metrics.get('target_missed')}| C{{Checkpoint restart-safe?}}",
        f"  C -->|{checkpoint.get('restart_safe')}| D{{Execution evidence ready?}}",
        f"  D -->|{execution.get('ready')}| E{{Advanced BPB contract safe?}}",
        f"  E -->|{advanced.get('safe_for_restart')}| F[{restart.get('action')}]",
        "```",
        "",
        "## Blockers And Warnings",
        "",
        ]
    )
    for label, values in (
        ("checkpoint warnings", checkpoint.get("warnings", [])),
        ("execution issues", execution.get("issues", [])),
        ("advanced BPB failed gates", advanced.get("failed_gates", [])),
        ("restart blockers", restart.get("blockers", [])),
    ):
        lines.append(f"### {label.title()}")
        if values:
            lines.extend(f"- `{value}`" for value in values)
        else:
            lines.append("- `none`")
        lines.append("")
    lines.extend(
        [
            "## Advanced Sidecars Tail",
            "",
            *[f"- `{path}`" for path in artifacts.get("advanced_sidecars_tail", [])[:80]],
            "",
            "## Policy",
            "",
            summary.get("policy", ""),
            "",
        ]
    )
    return chr(10).join(lines)




def _bar_chart_svg(values: dict[str, Any], *, title: str, chart_id: str) -> str:
    numeric: list[tuple[str, int]] = []
    for key, value in values.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            numeric.append((str(key), count))
    if not numeric:
        return f"<section class='panel' data-chart='{html.escape(chart_id)}'><h2>{html.escape(title)}</h2><p class='muted'>No recorded counts.</p></section>"
    numeric.sort(key=lambda item: (-item[1], item[0]))
    max_count = max(count for _, count in numeric) or 1
    width = 880
    left = 250
    bar_max = width - left - 96
    row_h = 34
    height = 44 + row_h * len(numeric)
    rows = [f"<svg role='img' aria-label='{html.escape(title)}' viewBox='0 0 {width} {height}'>"]
    rows.append(f"<title>{html.escape(title)}</title>")
    rows.append(f"<text x='0' y='20' class='svg-title'>{html.escape(title)}</text>")
    for index, (label, count) in enumerate(numeric):
        y = 40 + index * row_h
        bar_w = max(2, int(bar_max * (count / max_count)))
        rows.append(f"<text x='0' y='{y + 18}' class='axis-label'>{html.escape(label)}</text>")
        rows.append(f"<rect x='{left}' y='{y}' width='{bar_w}' height='22' rx='3'><title>{html.escape(label)}: {count}</title></rect>")
        rows.append(f"<text x='{left + bar_w + 10}' y='{y + 17}' class='value-label'>{count}</text>")
    rows.append("</svg>")
    return f"<section class='panel' data-chart='{html.escape(chart_id)}'><h2>{html.escape(title)}</h2>{''.join(rows)}</section>"


def _html_list(values: list[Any]) -> str:
    if not values:
        return "<li class='muted'>none</li>"
    return "".join(f"<li>{html.escape(str(value))}</li>" for value in values)


def render_html(summary: dict[str, Any]) -> str:
    run = summary["run_identity"]
    metrics = summary["primary_metrics"]
    checkpoint = summary["checkpoint_evidence"]
    execution = summary["execution_evidence"]
    advanced = summary["advanced_bpb_contract"]
    artifacts = summary["artifact_evidence"]
    restart = summary["restart_decision"]
    validator_gaps = artifacts.get("validator_gap_evidence", {}) if isinstance(artifacts.get("validator_gap_evidence"), dict) else {}
    sidecar_groups = artifacts.get("sidecar_groups", {}) if isinstance(artifacts.get("sidecar_groups"), dict) else {}
    validator_counts = validator_gaps.get("combined_category_counts", {}) if isinstance(validator_gaps.get("combined_category_counts"), dict) else {}
    trajectory_embedding = artifacts.get("trajectory_embedding_visual_evidence", {}) if isinstance(artifacts.get("trajectory_embedding_visual_evidence"), dict) else {}
    inference_audit_backfill = artifacts.get("inference_audit_backfill_evidence", {}) if isinstance(artifacts.get("inference_audit_backfill_evidence"), dict) else {}
    gflownet_branch = artifacts.get("gflownet_branch_selection_evidence", {}) if isinstance(artifacts.get("gflownet_branch_selection_evidence"), dict) else {}
    tropical_support = artifacts.get("tropical_support_evidence", {}) if isinstance(artifacts.get("tropical_support_evidence"), dict) else {}
    graphcg_direction = artifacts.get("graphcg_direction_evidence", {}) if isinstance(artifacts.get("graphcg_direction_evidence"), dict) else {}
    nll_density = artifacts.get("nll_density_evidence", {}) if isinstance(artifacts.get("nll_density_evidence"), dict) else {}
    chart_bundle_transport = artifacts.get("chart_bundle_transport_evidence", {}) if isinstance(artifacts.get("chart_bundle_transport_evidence"), dict) else {}
    bivariate_module = artifacts.get("bivariate_module_evidence", {}) if isinstance(artifacts.get("bivariate_module_evidence"), dict) else {}
    two_parameter_bifiltration = artifacts.get("two_parameter_bifiltration_visual_evidence", {}) if isinstance(artifacts.get("two_parameter_bifiltration_visual_evidence"), dict) else {}
    persistence_landscape = artifacts.get("persistence_landscape_evidence", {}) if isinstance(artifacts.get("persistence_landscape_evidence"), dict) else {}
    toric_tropical_cas = artifacts.get("toric_tropical_cas_evidence", {}) if isinstance(artifacts.get("toric_tropical_cas_evidence"), dict) else {}
    analogical_query_context = artifacts.get("analogical_query_context_evidence", {}) if isinstance(artifacts.get("analogical_query_context_evidence"), dict) else {}
    analogical_memory = artifacts.get("analogical_memory_evidence", {}) if isinstance(artifacts.get("analogical_memory_evidence"), dict) else {}
    simplicial_complex = artifacts.get("simplicial_complex_evidence", {}) if isinstance(artifacts.get("simplicial_complex_evidence"), dict) else {}
    topological_algebra = artifacts.get("topological_algebra_evidence", {}) if isinstance(artifacts.get("topological_algebra_evidence"), dict) else {}
    validator_sources = validator_gaps.get("sources", []) if isinstance(validator_gaps.get("sources"), list) else []
    sidecars = [str(path) for path in artifacts.get("advanced_sidecars_tail", [])]
    source_rows = []
    for source in validator_sources:
        if not isinstance(source, dict):
            continue
        source_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('name', 'validator')))}</td>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('gap_count')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not source_rows:
        source_rows.append("<tr><td colspan='5' class='muted'>No validator JSON sources recorded.</td></tr>")
    validator_ranked = validator_gaps.get("ranked_categories", []) if isinstance(validator_gaps.get("ranked_categories"), list) else []
    action_rows = []
    for row in validator_ranked[:20]:
        if not isinstance(row, dict):
            continue
        examples = row.get("examples", [])
        example_text = "; ".join(
            str(example.get("example", "")) for example in examples[:3] if isinstance(example, dict)
        ) if isinstance(examples, list) else ""
        action_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('category', 'other')))}</td>"
            f"<td>{html.escape(str(row.get('count', '')))}</td>"
            f"<td>{html.escape(str(row.get('required_action', '')))}</td>"
            f"<td><code>{html.escape(example_text)}</code></td>"
            "</tr>"
        )
    if not action_rows:
        action_rows.append("<tr><td colspan='4' class='muted'>No concrete validator gap examples recorded.</td></tr>")
    trajectory_embedding_rows = []
    for source in trajectory_embedding.get("sources", []) if isinstance(trajectory_embedding.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        trajectory_embedding_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('kind', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('node_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('edge_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('raw_model_embedding_count', 0)))}</td>"
            f"<td>{html.escape(_fmt(source.get('pca_pairwise_distance_correlation')))}</td>"
            f"<td>{html.escape(_fmt(source.get('pca_normalized_stress')))}</td>"
            f"<td>{html.escape(str(source.get('pca_quality_warning')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not trajectory_embedding_rows:
        trajectory_embedding_rows.append("<tr><td colspan='10' class='muted'>No GoT trajectory or embedding-map payload sidecar paths recorded.</td></tr>")
    inference_audit_backfill_rows = []
    for source in inference_audit_backfill.get("sources", []) if isinstance(inference_audit_backfill.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        inference_audit_backfill_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('kind', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('candidate_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('action_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('policy_no_fabrication', 'n/a')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not inference_audit_backfill_rows:
        inference_audit_backfill_rows.append("<tr><td colspan='8' class='muted'>No inference-audit or backfill-report sidecar paths recorded.</td></tr>")
    gflownet_branch_rows = []
    for source in gflownet_branch.get("sources", []) if isinstance(gflownet_branch.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        gflownet_branch_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('valid_branch_selection_row_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('selected_action_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('policy_counts', {})))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not gflownet_branch_rows:
        gflownet_branch_rows.append("<tr><td colspan='6' class='muted'>No inference-scaling branch-selection sidecar paths recorded.</td></tr>")
    tropical_support_rows = []
    for source in tropical_support.get("sources", []) if isinstance(tropical_support.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        tropical_support_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('support_probability_source', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('token_count')))}</td>"
            f"<td>{html.escape(_fmt(source.get('strict_wall_hit_rate')))}</td>"
            f"<td>{html.escape(_fmt(source.get('near_wall_hit_rate')))}</td>"
            f"<td>{html.escape(str(source.get('low_strict_wall_interpretation_status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not tropical_support_rows:
        tropical_support_rows.append("<tr><td colspan='8' class='muted'>No tropical support payload sidecar paths recorded.</td></tr>")
    graphcg_direction_rows = []
    for source in graphcg_direction.get("sources", []) if isinstance(graphcg_direction.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        graphcg_direction_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('projection_basis', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('direction_count')))}</td>"
            f"<td>{html.escape(str(source.get('candidate_count')))}</td>"
            f"<td>{html.escape(str(source.get('active_rank_nonzero_mean_abs')))}</td>"
            f"<td>{html.escape(_fmt(source.get('mean_abs_p90')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not graphcg_direction_rows:
        graphcg_direction_rows.append("<tr><td colspan='8' class='muted'>No GraphCG direction-cosine payload sidecar paths recorded.</td></tr>")
    nll_density_rows = []
    for source in nll_density.get("sources", []) if isinstance(nll_density.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        nll_density_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('actual_model_anchor_count')))}</td>"
            f"<td>{html.escape(str(source.get('support_sample_count')))}</td>"
            f"<td>{html.escape(_fmt(source.get('kernel_bandwidth')))}</td>"
            f"<td>{html.escape(_fmt(source.get('nll_span')))}</td>"
            f"<td>{html.escape(str(source.get('support_sample_trace_visibility', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not nll_density_rows:
        nll_density_rows.append("<tr><td colspan='8' class='muted'>No NLL density payload sidecar paths recorded.</td></tr>")
    bivariate_module_rows = []
    for source in bivariate_module.get("sources", []) if isinstance(bivariate_module.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        bivariate_module_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('coefficient_ring', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('fiber_rank_profile_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('structure_map_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('chain_presentation_not_a_free_resolution')))}</td>"
            f"<td>{html.escape(str(source.get('real_free_resolution_certified')))}</td>"
            f"<td>{html.escape(str(source.get('safe_unavailable_real_free_resolution')))}</td>"
            f"<td>{html.escape(str(source.get('real_free_resolution_status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not bivariate_module_rows:
        bivariate_module_rows.append("<tr><td colspan='10' class='muted'>No bivariate module sidecar paths recorded.</td></tr>")
    two_parameter_bifiltration_rows = []
    for source in two_parameter_bifiltration.get("sources", []) if isinstance(two_parameter_bifiltration.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        two_parameter_bifiltration_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('primary_view', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('axes_horizontal', 'unavailable')))} / {html.escape(str(source.get('axes_vertical', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('staircase_card_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('quotient_basis_lattice_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('hilbert_numerator_term_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('adjacent_lcm_syzygy_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('structure_map_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not two_parameter_bifiltration_rows:
        two_parameter_bifiltration_rows.append("<tr><td colspan='10' class='muted'>No two-parameter bifiltration visual sidecar paths recorded.</td></tr>")
    persistence_landscape_rows = []
    for source in persistence_landscape.get("sources", []) if isinstance(persistence_landscape.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        persistence_landscape_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('verified_unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('landscape_backend', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('landscape_row_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('curve_trace_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('finite_persistence_interval_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('unavailable_reasons', [])))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not persistence_landscape_rows:
        persistence_landscape_rows.append("<tr><td colspan='9' class='muted'>No persistence landscape sidecar paths recorded.</td></tr>")
    chart_bundle_rows = []
    for source in chart_bundle_transport.get("sources", []) if isinstance(chart_bundle_transport.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        chart_bundle_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('completeness_tier', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('paper_ready')))}</td>"
            f"<td>{html.escape(str(source.get('chart_count')))}</td>"
            f"<td>{html.escape(str(source.get('transport_count')))}</td>"
            f"<td>{html.escape(str(source.get('missing_required_groups', [])))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not chart_bundle_rows:
        chart_bundle_rows.append("<tr><td colspan='8' class='muted'>No chart-bundle transport sidecar paths recorded.</td></tr>")
    toric_tropical_rows = []
    for source in toric_tropical_cas.get("sources", []) if isinstance(toric_tropical_cas.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        toric_tropical_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('kind', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('backend', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('toric_ideal_certified', False)))}</td>"
            f"<td>{html.escape(str(source.get('fan_diagnostics_certified', False)))}</td>"
            f"<td>{html.escape(str(source.get('ray_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not toric_tropical_rows:
        toric_tropical_rows.append("<tr><td colspan='9' class='muted'>No toric/tropical CAS sidecar paths recorded.</td></tr>")
    analogical_query_rows = []
    for source in analogical_query_context.get("sources", []) if isinstance(analogical_query_context.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        analogical_query_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('selected_query_complex_source', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('conversion_status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('selected_query_probability_vertex_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('rejected_query_context_keys', [])))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not analogical_query_rows:
        analogical_query_rows.append("<tr><td colspan='7' class='muted'>No analogical query-context sidecar paths recorded.</td></tr>")
    analogical_memory_rows = []
    for source in analogical_memory.get("sources", []) if isinstance(analogical_memory.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        analogical_memory_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('kind', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('retrieved_count', source.get('raw_retrieved_count', 0))))}</td>"
            f"<td>{html.escape(str(source.get('qualified_model_probability_memory_count', source.get('eligible_count', 0))))}</td>"
            f"<td>{html.escape(str(source.get('top_k_rendered', 0)))}</td>"
            f"<td>{html.escape(str(source.get('pair_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('verified_insufficient_memory')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not analogical_memory_rows:
        analogical_memory_rows.append("<tr><td colspan='10' class='muted'>No analogical memory retrieval or simplex-tree analogy sidecar paths recorded.</td></tr>")
    simplicial_complex_rows = []
    for source in simplicial_complex.get("sources", []) if isinstance(simplicial_complex.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        simplicial_complex_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('kind', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('available_view_count', source.get('step_count', 0))))}</td>"
            f"<td>{html.escape(str(source.get('threshold_count', source.get('rendered_slider_contracts', 0))))}</td>"
            f"<td>{html.escape(str(source.get('source_simplex_count', source.get('displayed_simplex_count', 0))))}</td>"
            f"<td>{html.escape(str(source.get('no_proxy_or_fallback')))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not simplicial_complex_rows:
        simplicial_complex_rows.append("<tr><td colspan='9' class='muted'>No simplicial complex, radius-slider, or simplex-tree sidecar paths recorded.</td></tr>")
    topological_algebra_rows = []
    for source in topological_algebra.get("sources", []) if isinstance(topological_algebra.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        report_or_growth_count = source.get(
            "growth_row_count",
            source.get("available_topological_algebra_reports", int(bool(source.get("available")))),
        )
        topological_algebra_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source.get('path', '')))}</td>"
            f"<td>{html.escape(str(source.get('kind', 'unavailable')))}</td>"
            f"<td>{html.escape(str(source.get('available')))}</td>"
            f"<td>{html.escape(str(source.get('status', 'unavailable')))}</td>"
            f"<td>{html.escape(str(report_or_growth_count))}</td>"
            f"<td>{html.escape(str(source.get('available_probability_topological_algebra_reports', 0)))}</td>"
            f"<td>{html.escape(str(source.get('persistence_interval_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('multiparameter_fiber_rank_profile_count', 0)))}</td>"
            f"<td>{html.escape(str(source.get('reason', '')))}</td>"
            "</tr>"
        )
    if not topological_algebra_rows:
        topological_algebra_rows.append("<tr><td colspan='9' class='muted'>No topological algebra, growth topology, inference topology, or inference algebra sidecar paths recorded.</td></tr>")
    sidecar_items = "".join(f"<li data-path='{html.escape(path.lower())}'>{html.escape(path)}</li>" for path in sidecars[:160]) or "<li class='muted'>No sidecar paths recorded.</li>"
    restart_safe = checkpoint.get("restart_safe") and execution.get("ready") and advanced.get("safe_for_restart") and restart.get("step0_restart_allowed")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Herschel 5K Visual Evidence Report</title>
<style>
:root {{ color-scheme: light; --ink:#151923; --muted:#5b6472; --line:#d8dee8; --panel:#ffffff; --bg:#f5f7fb; --accent:#2457d6; --warn:#b42318; --ok:#087443; }}
body {{ margin:0; font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
header, main {{ max-width:1180px; margin:0 auto; padding:24px; }}
header {{ padding-bottom:10px; }}
h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
h2 {{ margin:0 0 12px; font-size:18px; letter-spacing:0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
.card, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 1px 2px rgba(10,20,40,.04); }}
.card b {{ display:block; font-size:22px; margin-top:5px; }}
.muted {{ color:var(--muted); }}
.badge {{ display:inline-block; padding:3px 8px; border-radius:999px; border:1px solid var(--line); background:#f8fafc; margin-right:6px; }}
.badge.ok {{ color:var(--ok); border-color:#9bd3b7; background:#effaf4; }} .badge.warn {{ color:var(--warn); border-color:#f2aaa4; background:#fff3f1; }}
section {{ margin:14px 0; }}
svg {{ width:100%; height:auto; }}
rect {{ fill:var(--accent); }}
.svg-title {{ font-weight:700; font-size:16px; fill:var(--ink); }} .axis-label,.value-label {{ font-size:12px; fill:var(--ink); }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }} th {{ color:var(--muted); font-weight:600; }}
.flow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px; align-items:stretch; }} .flow div {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcff; }}
input[type=search] {{ width:100%; padding:10px; border:1px solid var(--line); border-radius:6px; margin-bottom:8px; }}
code {{ white-space:break-spaces; }}
</style>
</head>
<body>
<header>
<h1>Herschel 5K Visual Evidence Report</h1>
<p class="muted">Generated {html.escape(str(summary.get('generated_at')))} from <code>{html.escape(str(summary.get('bundle_path')))}</code>. Evidence-only rendering: no training, validation, checkpoint loading, browser control, or GPU command is executed by this report.</p>
<span class="badge {'ok' if restart_safe else 'warn'}">restart_safe={html.escape(str(bool(restart_safe)))}</span>
<span class="badge {'ok' if validator_gaps.get('available') else 'warn'}">validator_gap_evidence={html.escape(str(bool(validator_gaps.get('available'))))}</span>
<span class="badge {'ok' if trajectory_embedding.get('available') else 'warn'}">trajectory_embedding_visual_evidence={html.escape(str(bool(trajectory_embedding.get('available'))))}</span>
<span class="badge {'ok' if inference_audit_backfill.get('available') else 'warn'}">inference_audit_backfill_evidence={html.escape(str(bool(inference_audit_backfill.get('available'))))}</span>
<span class="badge {'ok' if tropical_support.get('available') else 'warn'}">tropical_support_evidence={html.escape(str(bool(tropical_support.get('available'))))}</span>
<span class="badge {'ok' if graphcg_direction.get('available') else 'warn'}">graphcg_direction_evidence={html.escape(str(bool(graphcg_direction.get('available'))))}</span>
<span class="badge {'ok' if nll_density.get('available') else 'warn'}">nll_density_evidence={html.escape(str(bool(nll_density.get('available'))))}</span>
<span class="badge {'ok' if bivariate_module.get('available') else 'warn'}">bivariate_module_evidence={html.escape(str(bool(bivariate_module.get('available'))))}</span>
<span class="badge {'ok' if two_parameter_bifiltration.get('available') else 'warn'}">two_parameter_bifiltration_visual_evidence={html.escape(str(bool(two_parameter_bifiltration.get('available'))))}</span>
<span class="badge {'ok' if persistence_landscape.get('available') else 'warn'}">persistence_landscape_evidence={html.escape(str(bool(persistence_landscape.get('available'))))}</span>
<span class="badge {'ok' if chart_bundle_transport.get('available') else 'warn'}">chart_bundle_evidence={html.escape(str(bool(chart_bundle_transport.get('available'))))}</span>
<span class="badge {'ok' if toric_tropical_cas.get('available') else 'warn'}">toric_tropical_cas_evidence={html.escape(str(bool(toric_tropical_cas.get('available'))))}</span>
<span class="badge {'ok' if analogical_memory.get('available') else 'warn'}">analogical_memory_evidence={html.escape(str(bool(analogical_memory.get('available'))))}</span>
<span class="badge {'ok' if simplicial_complex.get('available') else 'warn'}">simplicial_complex_evidence={html.escape(str(bool(simplicial_complex.get('available'))))}</span>
<span class="badge {'ok' if topological_algebra.get('available') else 'warn'}">topological_algebra_evidence={html.escape(str(bool(topological_algebra.get('available'))))}</span>
</header>
<main>
<section class="grid">
<div class="card">Observed BPB<b>{html.escape(_fmt(metrics.get('bpb')))}</b><span class="muted">target &lt; {html.escape(_fmt(run.get('target_bpb')))}</span></div>
<div class="card">Observed graph-BPB<b>{html.escape(_fmt(metrics.get('graph_bpb')))}</b><span class="muted">graph-conditioned gate</span></div>
<div class="card">Checkpoint restart-safe<b>{html.escape(str(checkpoint.get('restart_safe')))}</b><span class="muted">{html.escape(str(checkpoint.get('unavailable_reason') or 'n/a'))}</span></div>
<div class="card">Restart action<b>{html.escape(str(restart.get('action')))}</b><span class="muted">step-0 allowed={html.escape(str(restart.get('step0_restart_allowed')))}</span></div>
</section>
<section class="panel"><h2>Restart Decision Flow</h2><div class="flow"><div>5K bundle<br><b>step {html.escape(str(run.get('boundary_step')))}</b></div><div>Target missed<br><b>{html.escape(str(metrics.get('target_missed')))}</b></div><div>Checkpoint safe<br><b>{html.escape(str(checkpoint.get('restart_safe')))}</b></div><div>Execution ready<br><b>{html.escape(str(execution.get('ready')))}</b></div><div>Advanced gate<br><b>{html.escape(str(advanced.get('safe_for_restart')))}</b></div><div>Action<br><b>{html.escape(str(restart.get('action')))}</b></div></div></section>
{_bar_chart_svg(sidecar_groups, title='Advanced Sidecar Groups', chart_id='sidecar-groups')}
{_bar_chart_svg(validator_counts, title='Strict Validator Evidence Gaps', chart_id='validator-gap-counts')}
{_bar_chart_svg(trajectory_embedding.get('status_counts', {}) if isinstance(trajectory_embedding, dict) else {}, title='Trajectory Embedding Visual Statuses', chart_id='trajectory-embedding-visual-statuses')}
{_bar_chart_svg(inference_audit_backfill.get('status_counts', {}) if isinstance(inference_audit_backfill, dict) else {}, title='Inference Audit And Backfill Statuses', chart_id='inference-audit-backfill-statuses')}
{_bar_chart_svg(inference_audit_backfill.get('action_kind_counts', {}) if isinstance(inference_audit_backfill, dict) else {}, title='Interactive Backfill Action Kinds', chart_id='interactive-backfill-action-kinds')}
{_bar_chart_svg(gflownet_branch.get('policy_counts', {}) if isinstance(gflownet_branch, dict) else {}, title='GFlowNet Branch Selection Policies', chart_id='gflownet-branch-selection-policies')}
{_bar_chart_svg(tropical_support.get('support_probability_source_counts', {}) if isinstance(tropical_support, dict) else {}, title='Tropical Support Probability Sources', chart_id='tropical-support-probability-sources')}
{_bar_chart_svg(graphcg_direction.get('basis_source_counts', {}) if isinstance(graphcg_direction, dict) else {}, title='GraphCG Projection Basis Sources', chart_id='graphcg-projection-basis-sources')}
{_bar_chart_svg(nll_density.get('visible_density_layer_counts', {}) if isinstance(nll_density, dict) else {}, title='NLL Density Visible Layers', chart_id='nll-density-visible-layers')}
{_bar_chart_svg(bivariate_module.get('resolution_status_counts', {}) if isinstance(bivariate_module, dict) else {}, title='Bivariate Module Real-Resolution Statuses', chart_id='bivariate-module-resolution-statuses')}
{_bar_chart_svg(two_parameter_bifiltration.get('status_counts', {}) if isinstance(two_parameter_bifiltration, dict) else {}, title='Two-Parameter Bifiltration Visual Statuses', chart_id='two-parameter-bifiltration-visual-statuses')}
{_bar_chart_svg(persistence_landscape.get('backend_counts', {}) or persistence_landscape.get('unavailable_reason_counts', {}) if isinstance(persistence_landscape, dict) else {}, title='Persistence Landscape Backends Or Unavailable Reasons', chart_id='persistence-landscape-backends')}
{_bar_chart_svg(chart_bundle_transport.get('completeness_tier_counts', {}) if isinstance(chart_bundle_transport, dict) else {}, title='Chart/Vector-Bundle Completeness Tiers', chart_id='chart-vector-bundle-completeness-tiers')}
{_bar_chart_svg(toric_tropical_cas.get('status_counts', {}) if isinstance(toric_tropical_cas, dict) else {}, title='Toric/Tropical CAS Statuses', chart_id='toric-tropical-cas-statuses')}
{_bar_chart_svg(analogical_memory.get('status_counts', {}) or analogical_memory.get('quality_gate_reason_counts', {}) if isinstance(analogical_memory, dict) else {}, title='Analogical Memory Statuses', chart_id='analogical-memory-statuses')}
{_bar_chart_svg(simplicial_complex.get('status_counts', {}) if isinstance(simplicial_complex, dict) else {}, title='Simplicial Complex And Simplex-Tree Statuses', chart_id='simplicial-complex-statuses')}
{_bar_chart_svg(topological_algebra.get('status_counts', {}) if isinstance(topological_algebra, dict) else {}, title='Topological Algebra Statuses', chart_id='topological-algebra-statuses')}
<section class="panel"><h2>Validator Sources</h2><table><thead><tr><th>Name</th><th>JSON path</th><th>Available</th><th>Gaps</th><th>Reason</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></section>
<section class="panel"><h2>Validator Gap Actions</h2><table><thead><tr><th>Category</th><th>Count</th><th>Required action</th><th>Examples</th></tr></thead><tbody>{''.join(action_rows)}</tbody></table></section>
<section class="panel"><h2>Trajectory Embedding Visual Evidence</h2><table><thead><tr><th>Sidecar</th><th>Kind</th><th>Available</th><th>Nodes</th><th>Edges</th><th>Raw embeddings</th><th>PCA corr</th><th>PCA stress</th><th>PCA warning</th><th>Reason</th></tr></thead><tbody>{''.join(trajectory_embedding_rows)}</tbody></table></section>
<section class="panel"><h2>Inference Audit And Backfill Evidence</h2><table><thead><tr><th>Sidecar</th><th>Kind</th><th>Available</th><th>Status</th><th>Candidates</th><th>Actions</th><th>No fabrication policy</th><th>Reason</th></tr></thead><tbody>{''.join(inference_audit_backfill_rows)}</tbody></table></section>
<section class="panel"><h2>GFlowNet Branch Selection Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Valid rows</th><th>Selected actions</th><th>Policies</th><th>Reason</th></tr></thead><tbody>{''.join(gflownet_branch_rows)}</tbody></table></section>
<section class="panel"><h2>Tropical Support Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Probability source</th><th>Tokens</th><th>Strict wall rate</th><th>Near wall rate</th><th>Status</th><th>Reason</th></tr></thead><tbody>{''.join(tropical_support_rows)}</tbody></table></section>
<section class="panel"><h2>GraphCG Direction Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Basis</th><th>Directions</th><th>Candidates</th><th>Active nonzero</th><th>Mean |cos| p90</th><th>Reason</th></tr></thead><tbody>{''.join(graphcg_direction_rows)}</tbody></table></section>
<section class="panel"><h2>NLL Density Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Anchors</th><th>Support samples</th><th>Kernel bandwidth</th><th>NLL span</th><th>Support visibility</th><th>Reason</th></tr></thead><tbody>{''.join(nll_density_rows)}</tbody></table></section>
<section class="panel"><h2>Bivariate Module Evidence</h2><table><thead><tr><th>Sidecar</th><th>Module available</th><th>Ring</th><th>Fibers</th><th>Structure maps</th><th>Chain only</th><th>Real free resolution</th><th>Safe unavailable</th><th>Status</th><th>Reason</th></tr></thead><tbody>{''.join(bivariate_module_rows)}</tbody></table></section>
<section class="panel"><h2>Two-Parameter Bifiltration Visual Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Primary view</th><th>Axes</th><th>Cards</th><th>Quotient basis</th><th>Hilbert terms</th><th>LCM syzygies</th><th>Structure maps</th><th>Reason</th></tr></thead><tbody>{''.join(two_parameter_bifiltration_rows)}</tbody></table></section>
<section class="panel"><h2>Persistence Landscape Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Verified unavailable</th><th>Backend</th><th>Rows</th><th>Curves</th><th>Finite intervals</th><th>Unavailable reasons</th><th>Reason</th></tr></thead><tbody>{''.join(persistence_landscape_rows)}</tbody></table></section>
<section class="panel"><h2>Chart/Vector-Bundle Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Completeness tier</th><th>Paper-ready telemetry</th><th>Charts</th><th>Transports</th><th>Missing groups</th><th>Reason</th></tr></thead><tbody>{''.join(chart_bundle_rows)}</tbody></table></section>
<section class="panel"><h2>Toric/Tropical CAS Evidence</h2><table><thead><tr><th>Sidecar</th><th>Kind</th><th>Available</th><th>Status</th><th>Backend</th><th>Finite toric ideal</th><th>Tropical fan</th><th>Rays</th><th>Reason</th></tr></thead><tbody>{''.join(toric_tropical_rows)}</tbody></table></section>
<section class="panel"><h2>Analogical Query Context Evidence</h2><table><thead><tr><th>Sidecar</th><th>Available</th><th>Selected source</th><th>Status</th><th>Probability vertices</th><th>Rejected keys</th><th>Reason</th></tr></thead><tbody>{''.join(analogical_query_rows)}</tbody></table></section>
<section class="panel"><h2>Analogical Memory Evidence</h2><table><thead><tr><th>Sidecar</th><th>Kind</th><th>Available</th><th>Status</th><th>Retrieved</th><th>Qualified</th><th>Top-k rendered</th><th>Pairs</th><th>Insufficient memory</th><th>Reason</th></tr></thead><tbody>{''.join(analogical_memory_rows)}</tbody></table></section>
<section class="panel"><h2>Simplicial Complex And Simplex-Tree Evidence</h2><table><thead><tr><th>Sidecar</th><th>Kind</th><th>Available</th><th>Status</th><th>Views/steps</th><th>Thresholds/sliders</th><th>Simplices</th><th>No proxy</th><th>Reason</th></tr></thead><tbody>{''.join(simplicial_complex_rows)}</tbody></table></section>
<section class="panel"><h2>Topological Algebra Evidence</h2><table><thead><tr><th>Sidecar</th><th>Kind</th><th>Available</th><th>Status</th><th>Reports/growth rows</th><th>Probability reports</th><th>Intervals</th><th>Fibers</th><th>Reason</th></tr></thead><tbody>{''.join(topological_algebra_rows)}</tbody></table></section>
<section class="panel"><h2>Blockers And Warnings</h2><div class="grid"><div><h3>Checkpoint</h3><ul>{_html_list(checkpoint.get('warnings', []))}</ul></div><div><h3>Execution</h3><ul>{_html_list(execution.get('issues', []))}</ul></div><div><h3>Advanced BPB</h3><ul>{_html_list(advanced.get('failed_gates', []))}</ul></div><div><h3>Restart</h3><ul>{_html_list(restart.get('blockers', []))}</ul></div></div></section>
<section class="panel"><h2>Advanced Sidecars Tail</h2><input id="sidecar-filter" type="search" placeholder="Filter sidecar paths"><ul id="sidecar-list">{sidecar_items}</ul></section>
</main>
<script>
const input = document.getElementById('sidecar-filter');
const items = Array.from(document.querySelectorAll('#sidecar-list li[data-path]'));
if (input) {{ input.addEventListener('input', () => {{ const q = input.value.toLowerCase(); items.forEach(li => li.style.display = li.dataset.path.includes(q) ? '' : 'none'); }}); }}
</script>
</body>
</html>
"""


def write_herschel_report(
    bundle_path: Path,
    output_path: Path,
    json_output: Path | None = None,
    html_output: Path | None = None,
) -> dict[str, Any]:
    bundle = _read_json(bundle_path)
    summary = summarize_bundle(bundle, bundle_path=bundle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(summary), encoding="utf-8")
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if html_output is not None:
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(render_html(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a CPU-only Herschel 5K evidence report from an existing review bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args(argv)
    summary = write_herschel_report(args.bundle, args.output, args.json_output, args.html_output)
    print(
        json.dumps(
            {
                "output": _project_path(args.output),
                "json_output": _project_path(args.json_output) if args.json_output else "",
                "html_output": _project_path(args.html_output) if args.html_output else "",
                "restart_action": summary["restart_decision"]["action"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
