from __future__ import annotations

import json
from typing import Any


def advanced_bpb_contract_report(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit the explicit config contract for BPB-focused step-0 runs.

    This is intentionally config-only: it does not infer missing advanced methods from
    model defaults or substitute generated evidence. Non-BPB smoke/fixture configs are
    reported as not requiring the contract so local readiness tests stay small.
    """
    gates: list[dict[str, Any]] = []
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    tokengt_cfg = cfg.get("tokengt", {}) if isinstance(cfg.get("tokengt", {}), dict) else {}
    hybrid_cfg = cfg.get("hybrid_data", {}) if isinstance(cfg.get("hybrid_data", {}), dict) else {}
    sources = hybrid_cfg.get("sources", []) if isinstance(hybrid_cfg.get("sources", []), list) else []
    mim_cfg = cfg.get("meet_in_middle", {}) if isinstance(cfg.get("meet_in_middle", {}), dict) else {}
    wandb_cfg = cfg.get("wandb", {}) if isinstance(cfg.get("wandb", {}), dict) else {}
    run_name = str(cfg.get("run_name", ""))
    target_bpb = _optional_float(cfg.get("target_bpb"))
    required = bool(
        cfg.get("parameter_golf_bpb_focus")
        or target_bpb is not None
        or "bpb_5k_gate" in run_name
        or "bpb112" in run_name
    )

    def float_cfg(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = _optional_float(mapping.get(key))
        return default if value is None else value

    def int_cfg(mapping: dict[str, Any], key: str, default: int = 0) -> int:
        try:
            return int(mapping.get(key, default))
        except Exception:
            return default

    def has_required_source(predicate) -> bool:
        return any(bool(source.get("required", False)) and predicate(source) for source in sources if isinstance(source, dict))

    parameter_source = has_required_source(
        lambda source: str(source.get("kind", "")) == "parameter_golf_bin"
        or "parameter_golf" in str(source.get("name", "")).lower()
    )

    def is_required_hf_reasoning_source(source: dict[str, Any]) -> bool:
        source_name = str(source.get("name", "")).lower()
        return str(source.get("kind", "")) == "parquet" and "hf" in source_name and "reasoning" in source_name

    hf_reasoning_source = has_required_source(is_required_hf_reasoning_source)
    dim = int_cfg(model_cfg, "dim")
    graphcg_num_directions = int_cfg(model_cfg, "graphcg_num_directions")
    graphcg_active_directions = int_cfg(model_cfg, "graphcg_active_directions")
    sequence_tropical_enabled = bool(model_cfg.get("use_sequence_tropical", model_cfg.get("sequence_tropical_enabled", False)))
    max_visual_audit_interval = int_cfg(cfg, "advanced_bpb_max_visual_audit_interval", 250)
    require_graphcg_active_full_rank = bool(cfg.get("require_graphcg_active_full_rank", False))
    require_nontrivial_bundle_toric_losses = bool(cfg.get("require_nontrivial_bundle_toric_losses", False))
    bundle_toric_weights = {
        "bundle_transport_weight": float_cfg(model_cfg, "bundle_transport_weight"),
        "bundle_cocycle_weight": float_cfg(model_cfg, "bundle_cocycle_weight"),
        "bundle_flat_rank_weight": float_cfg(model_cfg, "bundle_flat_rank_weight"),
        "toric_normal_fan_weight": float_cfg(model_cfg, "toric_normal_fan_weight"),
        "graphcg_toric_cell_agreement_weight": float_cfg(model_cfg, "graphcg_toric_cell_agreement_weight"),
        "chart_bpb_consistency_weight": float_cfg(model_cfg, "chart_bpb_consistency_weight"),
        "bundle_atom_stability_weight": float_cfg(model_cfg, "bundle_atom_stability_weight"),
    }
    memory_scalar_thresholds = {
        key: _optional_float(cfg.get(key))
        for key in (
            "memory_quality_min_score",
            "memory_min_score",
            "memory_quality_min_quality_score",
            "memory_quality_max_nll",
            "memory_quality_max_bpb",
            "memory_quality_min_nll_improvement",
            "memory_quality_min_margin_mean",
        )
    }
    memory_has_scalar_threshold = any(value is not None and value > 0.0 for value in memory_scalar_thresholds.values())
    memory_min_vertices = int_cfg(cfg, "memory_quality_min_probability_vertices")
    memory_min_simplices = int_cfg(cfg, "memory_quality_min_probability_simplices")
    periodic_top_k = int_cfg(cfg, "periodic_memory_retrieve_top_k")
    inference_top_k = int_cfg(cfg, "inference_memory_retrieve_top_k")
    wandb_run_name = str(cfg.get("wandb_run_name") or cfg.get("wandb_name") or "")
    wandb_run_name_configured = bool(wandb_run_name)
    wandb_run_name_matches_config = bool(run_name) and wandb_run_name == run_name
    section = {
        "required": required,
        "policy": "BPB-focused 5K gate configs must explicitly enable the requested advanced methods; missing entries fail instead of inheriting defaults.",
        "run_name": run_name,
        "target_bpb": target_bpb,
        "tokengt_graph_token": bool(tokengt_cfg.get("graph_token", False)),
        "graph_autoregressive_decoding": bool(cfg.get("graph_autoregressive_decoding", False)),
        "required_parameter_golf_source": parameter_source,
        "required_hf_reasoning_source": hf_reasoning_source,
        "real_data_required": bool(cfg.get("require_data", False)) and bool(cfg.get("data_root")),
        "seq_len": int_cfg(cfg, "seq_len"),
        "batch_size": int_cfg(cfg, "batch_size"),
        "gflownet_weight": float_cfg(model_cfg, "gflownet_weight"),
        "graphcg_weight": float_cfg(model_cfg, "graphcg_weight"),
        "certificate_weight": float_cfg(model_cfg, "certificate_weight"),
        "sequence_tropical_enabled": sequence_tropical_enabled,
        "sequence_tropical_weight": float_cfg(model_cfg, "sequence_tropical_weight"),
        "graph_tropical_block_size": int_cfg(model_cfg, "graph_tropical_block_size"),
        "graphcg_dim": dim,
        "graphcg_num_directions": graphcg_num_directions,
        "graphcg_active_directions": graphcg_active_directions,
        "require_graphcg_active_full_rank": require_graphcg_active_full_rank,
        "require_nontrivial_bundle_toric_losses": require_nontrivial_bundle_toric_losses,
        "bundle_toric_weights": bundle_toric_weights,
        "memory_quality_require_probability_complex": bool(cfg.get("memory_quality_require_probability_complex", False)),
        "memory_quality_require_topological_algebra": bool(cfg.get("memory_quality_require_topological_algebra", False)),
        "memory_quality_min_probability_vertices": memory_min_vertices,
        "memory_quality_min_probability_simplices": memory_min_simplices,
        "memory_quality_scalar_thresholds": memory_scalar_thresholds,
        "periodic_memory_retrieve_top_k": periodic_top_k,
        "inference_memory_retrieve_top_k": inference_top_k,
        "graph_bpb_side_weight": float_cfg(cfg, "graph_bpb_side_weight"),
        "validation_every_steps": int_cfg(cfg, "validation_every_steps"),
        "visualization_every_steps": int_cfg(cfg, "visualization_every_steps"),
        "advanced_bpb_max_visual_audit_interval": max_visual_audit_interval,
        "periodic_interactive_artifacts_enabled": bool(cfg.get("periodic_interactive_artifacts_enabled", False)),
        "periodic_browser_publish": bool(cfg.get("periodic_browser_publish", False)),
        "meet_in_middle": {
            "enabled": bool(mim_cfg.get("enabled", False)),
            "causal_dag_use_forward_reverse": bool(mim_cfg.get("causal_dag_use_forward_reverse", False)),
            "noncausal_use_random_order_autoregression": bool(mim_cfg.get("noncausal_use_random_order_autoregression", False)),
            "agreement_weight": float_cfg(mim_cfg, "agreement_weight"),
            "reverse_nll_weight": float_cfg(mim_cfg, "reverse_nll_weight"),
        },
        "wandb": {
            "enabled": bool(wandb_cfg.get("enabled", False)),
            "mode": str(wandb_cfg.get("mode", "")),
            "project": str(wandb_cfg.get("project", "")),
            "run_name": wandb_run_name,
            "run_name_configured": wandb_run_name_configured,
            "run_name_matches_config": wandb_run_name_matches_config,
            "entity": str(wandb_cfg.get("entity", "")),
            "entity_configured": bool(wandb_cfg.get("entity")),
        },
    }
    add_gate(
        gates,
        "advanced_bpb_contract_required",
        True,
        "required" if required else "not required for non-BPB readiness config",
    )
    if not required:
        return section, gates

    add_gate(gates, "advanced_bpb_target_bpb_112_or_better", target_bpb is not None and target_bpb <= 1.12, str(target_bpb))
    add_gate(gates, "advanced_bpb_real_data_required", section["real_data_required"], str(cfg.get("data_root", "")))
    add_gate(gates, "advanced_bpb_tokengt_graph_token_enabled", section["tokengt_graph_token"], str(tokengt_cfg.get("graph_token")))
    add_gate(gates, "advanced_bpb_graph_autoregressive_enabled", section["graph_autoregressive_decoding"], str(cfg.get("graph_autoregressive_decoding")))
    add_gate(gates, "advanced_bpb_parameter_golf_required_source", parameter_source, json.dumps(sources)[:500])
    add_gate(gates, "advanced_bpb_hf_reasoning_required_source", hf_reasoning_source, json.dumps(sources)[:500])
    add_gate(gates, "advanced_bpb_long_context_seq_len_1024", section["seq_len"] >= 1024, str(section["seq_len"]))
    add_gate(gates, "advanced_bpb_multi_record_batch", section["batch_size"] > 1, str(section["batch_size"]))
    add_gate(gates, "advanced_bpb_gflownet_weight_positive", section["gflownet_weight"] > 0.0, str(section["gflownet_weight"]))
    add_gate(gates, "advanced_bpb_graphcg_weight_positive", section["graphcg_weight"] > 0.0, str(section["graphcg_weight"]))
    add_gate(gates, "advanced_bpb_certificate_weight_positive", section["certificate_weight"] > 0.0, str(section["certificate_weight"]))
    add_gate(
        gates,
        "advanced_bpb_sequence_tropical_enabled",
        sequence_tropical_enabled and section["sequence_tropical_weight"] > 0.0 and section["graph_tropical_block_size"] > 0,
        f"enabled={sequence_tropical_enabled} weight={section['sequence_tropical_weight']} block={section['graph_tropical_block_size']}",
    )
    add_gate(
        gates,
        "advanced_bpb_graphcg_full_rank_directions",
        dim > 0 and graphcg_num_directions >= dim,
        f"directions={graphcg_num_directions} dim={dim}",
    )
    add_gate(gates, "advanced_bpb_graphcg_active_directions_positive", graphcg_active_directions > 0, str(graphcg_active_directions))
    if require_graphcg_active_full_rank:
        add_gate(
            gates,
            "advanced_bpb_graphcg_active_directions_full_rank",
            dim > 0 and graphcg_active_directions >= dim,
            f"active={graphcg_active_directions} dim={dim}",
        )
    if require_nontrivial_bundle_toric_losses:
        add_gate(
            gates,
            "advanced_bpb_chart_bundle_auxiliary_enabled",
            bool(model_cfg.get("enable_chart_bundle_auxiliary", False)),
            str(model_cfg.get("enable_chart_bundle_auxiliary", False)),
        )
        add_gate(
            gates,
            "advanced_bpb_bundle_toric_losses_nonzero",
            all(weight > 0.0 for weight in bundle_toric_weights.values()),
            json.dumps(bundle_toric_weights, sort_keys=True),
        )
    add_gate(
        gates,
        "advanced_bpb_memory_quality_probability_complex",
        section["memory_quality_require_probability_complex"] and memory_min_vertices > 0 and memory_min_simplices > 0,
        f"require={section['memory_quality_require_probability_complex']} vertices={memory_min_vertices} simplices={memory_min_simplices}",
    )
    add_gate(
        gates,
        "advanced_bpb_memory_quality_topological_algebra",
        section["memory_quality_require_topological_algebra"],
        str(section["memory_quality_require_topological_algebra"]),
    )
    add_gate(gates, "advanced_bpb_memory_quality_scalar_threshold", memory_has_scalar_threshold, json.dumps(memory_scalar_thresholds, sort_keys=True))
    add_gate(gates, "advanced_bpb_memory_retrieve_many_top_k", periodic_top_k >= 5 and inference_top_k >= 5, f"periodic={periodic_top_k} inference={inference_top_k}")
    add_gate(gates, "advanced_bpb_graph_bpb_side_weight_positive", section["graph_bpb_side_weight"] > 0.0, str(section["graph_bpb_side_weight"]))
    add_gate(
        gates,
        f"advanced_bpb_visual_audit_cadence_{max_visual_audit_interval}",
        max_visual_audit_interval > 0
        and 0 < section["validation_every_steps"] <= max_visual_audit_interval
        and 0 < section["visualization_every_steps"] <= max_visual_audit_interval,
        (
            f"validation={section['validation_every_steps']} "
            f"visualization={section['visualization_every_steps']} "
            f"max={max_visual_audit_interval}"
        ),
    )
    add_gate(gates, "advanced_bpb_periodic_interactive_artifacts", section["periodic_interactive_artifacts_enabled"], str(section["periodic_interactive_artifacts_enabled"]))
    add_gate(gates, "advanced_bpb_periodic_browser_publish", section["periodic_browser_publish"], str(section["periodic_browser_publish"]))
    add_gate(gates, "advanced_bpb_meet_in_middle_enabled", section["meet_in_middle"]["enabled"], json.dumps(section["meet_in_middle"], sort_keys=True))
    add_gate(gates, "advanced_bpb_meet_in_middle_forward_reverse", section["meet_in_middle"]["causal_dag_use_forward_reverse"], json.dumps(section["meet_in_middle"], sort_keys=True))
    add_gate(gates, "advanced_bpb_meet_in_middle_roar_random_order", section["meet_in_middle"]["noncausal_use_random_order_autoregression"], json.dumps(section["meet_in_middle"], sort_keys=True))
    add_gate(gates, "advanced_bpb_meet_in_middle_loss_weights", section["meet_in_middle"]["agreement_weight"] > 0.0 and section["meet_in_middle"]["reverse_nll_weight"] > 0.0, json.dumps(section["meet_in_middle"], sort_keys=True))
    add_gate(
        gates,
        "advanced_bpb_wandb_online_project",
        section["wandb"]["enabled"]
        and section["wandb"]["mode"] == "online"
        and bool(section["wandb"]["project"])
        and section["wandb"]["run_name_configured"],
        json.dumps(section["wandb"], sort_keys=True),
    )
    add_gate(
        gates,
        "advanced_bpb_wandb_entity_configured",
        section["wandb"]["entity_configured"],
        json.dumps({"project": section["wandb"]["project"], "entity": section["wandb"]["entity"]}, sort_keys=True),
    )
    add_gate(
        gates,
        "advanced_bpb_wandb_run_name_matches_config",
        section["wandb"]["run_name_matches_config"],
        json.dumps({"run_name": run_name, "wandb_run_name": section["wandb"]["run_name"]}, sort_keys=True),
    )
    return section, gates



def enforce_advanced_bpb_contract(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Raise if a BPB-focused training config violates the advanced contract."""
    section, gates = advanced_bpb_contract_report(cfg)
    failed = [gate for gate in gates if gate.get("status") == "fail"]
    if failed:
        detail = "\n".join(f"- {gate.get('name')}: {gate.get('detail', '')}" for gate in failed)
        raise RuntimeError(f"Advanced BPB training contract failed:\n{detail}")
    return section, gates


def add_gate(gates: list[dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    gates.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if _finite_number(parsed) else None
