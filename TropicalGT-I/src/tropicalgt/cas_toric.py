from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cas_free_resolution import (
    _backend_version,
    _candidate_executable,
    _parse_bool,
    _run_tagged_cas_script,
    canonical_json,
    sha256_json,
)

TORIC_EMBEDDING_SCHEMA_VERSION = "tropicalgt.cas_toric_embedding.v1"
TORIC_EMBEDDING_CACHE_SCHEMA_VERSION = "tropicalgt.cas_toric_embedding.cache.v1"
TORIC_EMBEDDING_CACHE_VERSION = "2026-06-16.macaulay2-toric-ideal-v1"


def toric_embedding_certificate_contract() -> dict[str, Any]:
    return {
        "certificate_source": "Macaulay2 Quasidegrees toricIdeal(A,R) on an explicit integer exponent matrix",
        "tool_backed_embedding_scope": (
            "The certified object is the affine toric ideal/kernel of the finite monomial map determined by "
            "the columns of A. It is a finite monomial-map sidecar certificate, not a tropical-variety "
            "embedding into a toric variety and not a global toric model of the neural network."
        ),
        "required_macaulay2_methods": ["needsPackage Quasidegrees", "toricIdeal"],
        "normal_fan_scope": (
            "Normal-fan, fan-refinement, toric-variety, or tropical-variety embedding claims require an additional "
            "certified fan or tropical certificate; the toricIdeal certificate alone certifies only the monomial map kernel."
        ),
        "sage_scope": (
            "Sage ToricIdeal or toric-variety APIs may be added as separate explicit backends only if they emit the same "
            "certificate fields; they are not a silent fallback for Macaulay2."
        ),
        "no_proxy_policy": (
            "No chart-bundle logits, toric-row activations, GraphCG cells, support tokens, embeddings, or visualization rows may "
            "substitute for this CAS certificate."
        ),
    }


def canonicalize_toric_exponent_matrix(spec: dict[str, Any]) -> dict[str, Any]:
    coefficient_field = str(spec.get("coefficient_field", "QQ") or "QQ").strip()
    if coefficient_field != "QQ":
        raise ValueError("toric embedding certificates currently support coefficient_field='QQ' only")
    matrix = spec.get("exponent_matrix")
    if matrix is None and spec.get("columns") is not None:
        columns = [_int_row(row, "exponent column") for row in spec.get("columns") or []]
        if not columns:
            raise ValueError("toric embedding certificates require at least one exponent column")
        dim = len(columns[0])
        if any(len(col) != dim for col in columns):
            raise ValueError("all exponent columns must have the same dimension")
        matrix = [[col[row_idx] for col in columns] for row_idx in range(dim)]
    rows = [_int_row(row, "exponent matrix row") for row in matrix or []]
    if not rows:
        raise ValueError("toric embedding certificates require a nonempty exponent_matrix")
    column_count = len(rows[0])
    if column_count < 2:
        raise ValueError("toric embedding certificates require at least two monomial coordinates/columns")
    if any(len(row) != column_count for row in rows):
        raise ValueError("all exponent_matrix rows must have the same number of columns")
    variable_names = [str(value).strip() for value in spec.get("variable_names", []) if str(value).strip()]
    if not variable_names:
        variable_names = [f"z_{idx}" for idx in range(column_count)]
    if len(variable_names) != column_count:
        raise ValueError("variable_names length must equal the exponent_matrix column count")
    for name in variable_names:
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError(f"unsupported Macaulay2 toric coordinate variable name: {name!r}")
    schema = {
        "schema_version": TORIC_EMBEDDING_SCHEMA_VERSION,
        "coefficient_field": coefficient_field,
        "exponent_matrix": rows,
        "variable_names": variable_names,
        "lattice_dimension": len(rows),
        "coordinate_count": column_count,
        "source": spec.get("source", "model_derived_toric_exponent_matrix"),
    }
    schema["input_sha256"] = sha256_json(schema)
    return schema


def _int_row(value: Any, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty list of integers")
    out: list[int] = []
    for raw in value:
        if isinstance(raw, bool):
            raise ValueError(f"{label} contains a boolean where an integer exponent is required")
        try:
            out.append(int(raw))
        except Exception as exc:
            raise ValueError(f"{label} contains a non-integer exponent: {raw!r}") from exc
    return out


def probe_toric_backends() -> dict[str, Any]:
    m2 = _candidate_executable("M2")
    return {
        "backends": [
            {
                "name": "Macaulay2",
                "executable": m2,
                "available": m2 is not None,
                "version": _backend_version(m2) if m2 else None,
                "required_package": "Quasidegrees",
                "required_method": "toricIdeal",
            }
        ],
        "preferred_order": ["Macaulay2 Quasidegrees toricIdeal"],
    }


def unavailable_toric_embedding_certificate(
    spec: dict[str, Any],
    *,
    status: str,
    reason: str,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        schema = canonicalize_toric_exponent_matrix(spec)
        input_hash = schema.get("input_sha256")
        command_template = build_macaulay2_toric_embedding_script(schema)
    except Exception as exc:
        schema = {}
        input_hash = None
        command_template = ""
        status = "invalid_input"
        reason = f"{reason}: {exc}"
    return {
        "schema_version": TORIC_EMBEDDING_SCHEMA_VERSION,
        "available": False,
        "status": status,
        "reason": reason,
        "input_sha256": input_hash,
        "exponent_matrix_schema": schema,
        "backend": "Macaulay2",
        "backend_probe": probe_toric_backends(),
        "backend_attempts": attempts or [],
        "certificate_contract": toric_embedding_certificate_contract(),
        "command_template": command_template,
        "certificate_attached": False,
        "toric_embedding_certified": False,
        "toric_ideal_certified": False,
        "tropical_variety_embedding_certified": False,
        "global_toric_variety_embedding_certified": False,
        "embedding_scope": "unavailable_finite_monomial_map_toric_ideal_certificate",
        "safe_to_render_as_toric_embedding": False,
        "safe_to_render_as_tropical_variety_embedding": False,
        "safe_to_render_as_global_toric_variety_embedding": False,
        "safe_to_use_as_normal_fan_certificate": False,
        "cas_artifacts": {},
    }


def try_compute_toric_embedding_certificate(
    spec: dict[str, Any],
    *,
    timeout_s: float = 20.0,
    use_cache: bool | None = None,
) -> dict[str, Any]:
    try:
        schema = canonicalize_toric_exponent_matrix(spec)
    except Exception as exc:
        return unavailable_toric_embedding_certificate(spec, status="invalid_input", reason=str(exc))
    cache_context = _toric_cache_context(schema, use_cache=use_cache)
    cached = _load_cached_toric(cache_context)
    if cached is not None:
        return cached
    executable = _candidate_executable("M2")
    if not executable:
        report = unavailable_toric_embedding_certificate(schema, status="backend_not_installed", reason="Macaulay2 executable not found.")
        return _cache_and_annotate_toric(report, cache_context)
    result = _run_tagged_cas_script(
        name="M2",
        executable=executable,
        script=build_macaulay2_toric_embedding_script(schema),
        suffix=".m2",
        timeout_s=timeout_s,
    )
    attempt = result.get("attempt", {})
    if not result.get("available"):
        status = str(attempt.get("status") or "backend_error")
        reason = str(attempt.get("reason") or attempt.get("stderr_tail") or "Macaulay2 toricIdeal did not emit a tagged certificate.")
        report = unavailable_toric_embedding_certificate(schema, status=status, reason=reason, attempts=[attempt])
        return _cache_and_annotate_toric(report, cache_context)
    report = _certified_toric_embedding_result(schema, result.get("parsed", {}), result.get("tagged_output", ""), [attempt])
    return _cache_and_annotate_toric(report, cache_context)


def build_macaulay2_toric_embedding_script(schema: dict[str, Any]) -> str:
    variables = [str(value) for value in schema.get("variable_names", [])]
    rows = [[int(value) for value in row] for row in schema.get("exponent_matrix", [])]
    ring = f"{schema.get('coefficient_field', 'QQ')}[" + ",".join(variables) + "]"
    matrix = "matrix {" + ",".join("{" + ",".join(str(value) for value in row) + "}" for row in rows) + "}"
    return "\n".join([
        "-- TropicalGT Macaulay2 finite monomial-map toric ideal sidecar certificate",
        'toricOk = try (needsPackage "Quasidegrees"; true) else false',
        'print "TROPICALGT_RESOLUTION_BEGIN"',
        'print "backend=Macaulay2"',
        'print concatenate("quasidegrees_package_available=", toString toricOk)',
        'if not toricOk then (print "certificate_error=Quasidegrees package is not loadable"; print "TROPICALGT_RESOLUTION_END"; exit 2)',
        f"R = {ring}",
        f"A = {matrix}",
        "I = toricIdeal(A,R)",
        'print "certificate_type=Macaulay2 Quasidegrees toricIdeal finite monomial-map certificate"',
        'print "embedding_scope=finite_monomial_map_toric_ideal_certificate_only"',
        'print "toric_embedding_certified=true"',
        'print "toric_ideal_certified=true"',
        'print "tropical_variety_embedding_certified=false"',
        'print "global_toric_variety_embedding_certified=false"',
        'print concatenate("ring=", toString R)',
        'print concatenate("exponent_matrix=", replace("\n", " || ", toString A))',
        'print concatenate("toric_ideal_text=", replace("\n", " || ", toString I))',
        'print concatenate("toric_ideal_generators=", replace("\n", " || ", toString flatten entries gens I))',
        'print concatenate("generator_count=", toString numgens I)',
        'try print concatenate("codimension=", toString codim I) else print "codimension_error=unavailable"',
        'try print concatenate("dimension=", toString dim I) else print "dimension_error=unavailable"',
        'print "TROPICALGT_RESOLUTION_END"',
        "exit 0",
    ]) + "\n"


def _certified_toric_embedding_result(
    schema: dict[str, Any],
    parsed: dict[str, Any],
    tagged_output: str,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    package_available = _parse_bool(parsed.get("quasidegrees_package_available"))
    embedding_certified = _parse_bool(parsed.get("toric_embedding_certified"))
    ideal_certified = _parse_bool(parsed.get("toric_ideal_certified"))
    ideal_text = str(parsed.get("toric_ideal_text", "") or "")
    generators_text = str(parsed.get("toric_ideal_generators", "") or "")
    if not (package_available and embedding_certified and ideal_certified and ideal_text):
        return unavailable_toric_embedding_certificate(
            schema,
            status="certificate_failed",
            reason="Macaulay2 toricIdeal ran but did not emit the required toric ideal certificate tags.",
            attempts=attempts,
        )
    generator_count = _parse_int(parsed.get("generator_count"))
    return {
        "schema_version": TORIC_EMBEDDING_SCHEMA_VERSION,
        "available": True,
        "status": "certified",
        "input_sha256": schema["input_sha256"],
        "exponent_matrix_schema": schema,
        "backend": "Macaulay2",
        "backend_probe": probe_toric_backends(),
        "backend_attempts": attempts,
        "command_template": build_macaulay2_toric_embedding_script(schema),
        "certificate_contract": toric_embedding_certificate_contract(),
        "certificate_attached": True,
        "certificate_type": parsed.get("certificate_type", "Macaulay2 Quasidegrees toricIdeal finite monomial-map certificate"),
        "toric_embedding_certified": True,
        "toric_ideal_certified": True,
        "tropical_variety_embedding_certified": _parse_bool(parsed.get("tropical_variety_embedding_certified")),
        "global_toric_variety_embedding_certified": _parse_bool(parsed.get("global_toric_variety_embedding_certified")),
        "embedding_scope": str(parsed.get("embedding_scope") or "finite_monomial_map_toric_ideal_certificate_only"),
        "safe_to_render_as_toric_embedding": True,
        "safe_to_render_as_tropical_variety_embedding": False,
        "safe_to_render_as_global_toric_variety_embedding": False,
        "safe_to_use_as_normal_fan_certificate": False,
        "monomial_map_summary": {
            "lattice_dimension": schema["lattice_dimension"],
            "coordinate_count": schema["coordinate_count"],
            "coordinate_variables": list(schema["variable_names"]),
            "exponent_matrix": [list(row) for row in schema["exponent_matrix"]],
            "interpretation": (
                "Columns of the exponent matrix define the finite monomial coordinate map; Macaulay2 toricIdeal certifies "
                "the kernel ideal of that map. This is not a normal-fan or tropical-variety certificate by itself."
            ),
        },
        "toric_ideal_summary": {
            "ring": str(parsed.get("ring", "") or ""),
            "ideal_text": ideal_text,
            "generators_text": generators_text,
            "generator_count": generator_count,
            "codimension": _parse_int(parsed.get("codimension")),
            "dimension": _parse_int(parsed.get("dimension")),
        },
        "cas_artifacts": {
            "raw_tagged_output": tagged_output,
            "ring": str(parsed.get("ring", "") or ""),
            "exponent_matrix_text": str(parsed.get("exponent_matrix", "") or ""),
            "toric_ideal_text": ideal_text,
            "toric_ideal_generators": generators_text,
            "codimension_error": str(parsed.get("codimension_error", "") or ""),
            "dimension_error": str(parsed.get("dimension_error", "") or ""),
        },
        "render_warning": (
            "Certified finite toric-ideal sidecar only. This is not a certified normal fan, tropical variety, "
            "or global toric model of TropicalGT-I."
        ),
    }


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _toric_cache_context(schema: dict[str, Any], *, use_cache: bool | None) -> dict[str, Any]:
    enabled = bool(use_cache) if use_cache is not None else True
    cache_root = Path(__file__).resolve().parents[3] / ".cache" / "tropicalgt" / "cas_toric"
    key_payload = {
        "cache_schema_version": TORIC_EMBEDDING_CACHE_SCHEMA_VERSION,
        "adapter_cache_version": TORIC_EMBEDDING_CACHE_VERSION,
        "exponent_matrix": schema,
    }
    key = sha256_json(key_payload)
    return {"enabled": enabled, "root": cache_root, "key": key, "path": cache_root / f"{key}.json"}


def _load_cached_toric(context: dict[str, Any]) -> dict[str, Any] | None:
    if not context.get("enabled"):
        return None
    path = Path(context["path"])
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if data.get("cache", {}).get("adapter_cache_version") != TORIC_EMBEDDING_CACHE_VERSION:
        return None
    data["cache"]["hit"] = True
    return data


def _cache_and_annotate_toric(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(canonical_json(report))
    report["cache"] = {
        "enabled": bool(context.get("enabled")),
        "hit": False,
        "cache_schema_version": TORIC_EMBEDDING_CACHE_SCHEMA_VERSION,
        "adapter_cache_version": TORIC_EMBEDDING_CACHE_VERSION,
        "key": context.get("key"),
    }
    if context.get("enabled"):
        path = Path(context["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
