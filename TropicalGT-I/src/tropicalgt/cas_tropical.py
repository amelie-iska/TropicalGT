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

TROPICAL_SCHEMA_VERSION = "tropicalgt.cas_tropical_fan.v1"
TROPICAL_CACHE_SCHEMA_VERSION = "tropicalgt.cas_tropical_fan.cache.v1"
TROPICAL_CACHE_VERSION = "2026-06-16.macaulay2-tropical-contract-v2"
_ALLOWED_M2_POLYNOMIAL_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_+-*/^(),[] 	"
)

TROPICAL_FAN_CERTIFICATE_CONTRACT = {
    "certificate_source": "Macaulay2 Tropical tropicalVariety on an explicit QQ polynomial ideal",
    "ordinary_to_laurent_torus_scope": (
        "The Macaulay2 input ideal is written in QQ[x_i]; the Tropical package certificate is interpreted "
        "as the tropical variety/cycle of the associated torus-side Laurent extension, not as a global "
        "neural toric variety."
    ),
    "required_macaulay2_methods": [
        "needsPackage Tropical",
        "tropicalVariety",
        "rays",
        "maxCones",
        "linealitySpace",
        "multiplicities",
        "isBalanced",
        "isPure",
        "isSimplicial",
        "fan",
    ],
    "side_diagnostic_methods": ["isTropicalBasis", "tropicalPrevariety"],
    "sage_scope": (
        "Sage tropical polynomial/variety APIs may support polynomial, curve, hypersurface, and plotting "
        "checks, but are not accepted here as a replacement for the Macaulay2 ideal-to-tropical-cycle fan certificate."
    ),
    "maclagan_toric_scheme_scope": (
        "Maclagan-Rincon tropical-ideal and toric-scheme language is used only as research scope unless a "
        "backend certifies the exported fan, ideal, grading, or sheaf/module object."
    ),
    "no_proxy_policy": (
        "No support-token, chain-presentation, rank-sample, embedding-only, or visualization diagnostic may substitute "
        "for the real Macaulay2 Tropical certificate."
    ),
}


def tropical_fan_certificate_contract() -> dict[str, Any]:
    return json.loads(json.dumps(TROPICAL_FAN_CERTIFICATE_CONTRACT))


def _validate_macaulay2_polynomial_text(text: str) -> None:
    if not any(ch.isalnum() for ch in text):
        raise ValueError("Macaulay2 tropical ideal generators must contain an alphanumeric polynomial term")
    invalid = sorted({ch for ch in text if ch not in _ALLOWED_M2_POLYNOMIAL_CHARS})
    if invalid:
        display = "".join(repr(ch) for ch in invalid)
        raise ValueError(f"unsupported Macaulay2 polynomial generator text contains {display}")


def canonicalize_tropical_ideal(spec: dict[str, Any]) -> dict[str, Any]:
    variables = [str(value).strip() for value in spec.get("variables", []) if str(value).strip()]
    generators = [str(value).strip() for value in spec.get("generators", []) if str(value).strip()]
    coefficient_field = str(spec.get("coefficient_field", "QQ") or "QQ").strip()
    if coefficient_field not in {"QQ"}:
        raise ValueError("Macaulay2 Tropical fan diagnostics currently support coefficient_field='QQ' only")
    if not variables:
        raise ValueError("tropical ideal diagnostics require at least one variable")
    if not generators:
        raise ValueError("tropical ideal diagnostics require at least one ideal generator")
    for name in variables:
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError(f"unsupported Macaulay2 variable name: {name!r}")
    for generator in generators:
        _validate_macaulay2_polynomial_text(generator)
    schema = {
        "schema_version": TROPICAL_SCHEMA_VERSION,
        "coefficient_field": coefficient_field,
        "variables": variables,
        "generators": generators,
        "source": spec.get("source", "model_derived_tropical_ideal"),
    }
    schema["input_sha256"] = sha256_json(schema)
    return schema


def unavailable_tropical_fan_diagnostics(
    spec: dict[str, Any],
    *,
    status: str,
    reason: str,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        schema = canonicalize_tropical_ideal(spec)
        input_hash = schema.get("input_sha256")
        command_template = build_macaulay2_tropical_script(schema)
    except Exception as exc:
        schema = {}
        input_hash = None
        command_template = ""
        reason = f"{reason}: {exc}"
        status = "invalid_input"
    return {
        "schema_version": TROPICAL_SCHEMA_VERSION,
        "available": False,
        "status": status,
        "reason": reason,
        "input_sha256": input_hash,
        "ideal_schema": schema,
        "backend": "Macaulay2",
        "backend_probe": probe_tropical_backends(),
        "backend_attempts": attempts or [],
        "certificate_contract": tropical_fan_certificate_contract(),
        "command_template": command_template,
        "certificate_attached": False,
        "tropical_cycle_certified": False,
        "fan_diagnostics_certified": False,
        "safe_to_render_as_tropical_fan": False,
        "cas_artifacts": {},
    }


def probe_tropical_backends() -> dict[str, Any]:
    m2 = _candidate_executable("M2")
    return {
        "backends": [
            {
                "name": "Macaulay2",
                "executable": m2,
                "available": m2 is not None,
                "version": _backend_version(m2) if m2 else None,
                "required_package": "Tropical",
            }
        ],
        "preferred_order": ["Macaulay2 Tropical"],
    }


def try_compute_tropical_fan_diagnostics(
    spec: dict[str, Any],
    *,
    timeout_s: float = 20.0,
    use_cache: bool | None = None,
) -> dict[str, Any]:
    try:
        schema = canonicalize_tropical_ideal(spec)
    except Exception as exc:
        return unavailable_tropical_fan_diagnostics(spec, status="invalid_input", reason=str(exc))
    cache_context = _tropical_cache_context(schema, use_cache=use_cache)
    cached = _load_cached_tropical(cache_context)
    if cached is not None:
        return cached
    executable = _candidate_executable("M2")
    if not executable:
        report = unavailable_tropical_fan_diagnostics(schema, status="backend_not_installed", reason="Macaulay2 executable not found.")
        return _cache_and_annotate_tropical(report, cache_context)
    result = _run_tagged_cas_script(
        name="Macaulay2Tropical",
        executable=executable,
        script=build_macaulay2_tropical_script(schema),
        suffix=".m2",
        timeout_s=timeout_s,
    )
    attempt = result.get("attempt", {})
    if not result.get("available"):
        status = str(attempt.get("status") or "backend_error")
        reason = str(attempt.get("reason") or attempt.get("stderr_tail") or "Macaulay2 Tropical did not emit a tagged certificate.")
        report = unavailable_tropical_fan_diagnostics(schema, status=status, reason=reason, attempts=[attempt])
        return _cache_and_annotate_tropical(report, cache_context)
    parsed = result.get("parsed", {})
    report = _certified_tropical_result(schema, parsed, result.get("tagged_output", ""), [attempt])
    return _cache_and_annotate_tropical(report, cache_context)


def build_macaulay2_tropical_script(schema: dict[str, Any]) -> str:
    variables = [str(value) for value in schema.get("variables", [])]
    generators = [str(value) for value in schema.get("generators", [])]
    ring = f"{schema.get('coefficient_field', 'QQ')}[" + ",".join(variables) + "]"
    ideal_expr = "ideal(" + ",".join(generators) + ")"
    lines = [
        "-- TropicalGT Macaulay2 Tropical fan diagnostic probe",
        'tropicalOk = try (needsPackage "Tropical"; true) else false',
        'print "TROPICALGT_RESOLUTION_BEGIN"',
        'print "backend=Macaulay2"',
        'print concatenate("tropical_package_available=", toString tropicalOk)',
        'if not tropicalOk then (print "certificate_error=Tropical package is not loadable"; print "TROPICALGT_RESOLUTION_END"; exit 2)',
        f"R = {ring}",
        f"I = {ideal_expr}",
        "G = flatten entries gens I",
        "basisOk = try (basisResult = isTropicalBasis G; true) else false",
        'print concatenate("tropical_basis_check_available=", toString basisOk)',
        'if basisOk then print concatenate("is_tropical_basis=", toString basisResult) else print "is_tropical_basis_error=not computed by Macaulay2 Tropical for this generator list"',
        "preOk = try (P = tropicalPrevariety G; true) else false",
        'print concatenate("tropical_prevariety_available=", toString preOk)',
        'if preOk then print concatenate("prevariety_class=", toString class P)',
        'if preOk then print concatenate("prevariety_rays=", replace("\\n", " || ", toString rays P))',
        'if preOk then print concatenate("prevariety_max_cones=", replace("\\n", " || ", toString maxCones P))',
        'if preOk then print concatenate("prevariety_lineality_space=", replace("\\n", " || ", toString linealitySpace P))',
        'if preOk then (try print concatenate("prevariety_multiplicities=", replace("\\n", " || ", toString multiplicities P)) else print "prevariety_multiplicities_error=unavailable")',
        'if preOk then (try print concatenate("prevariety_is_balanced=", toString isBalanced P) else print "prevariety_is_balanced_error=unavailable")',
        'if preOk then (try print concatenate("prevariety_is_pure=", toString isPure P) else print "prevariety_is_pure_error=unavailable")',
        'if preOk then (try print concatenate("prevariety_is_simplicial=", toString isSimplicial P) else print "prevariety_is_simplicial_error=unavailable")',
        "T = tropicalVariety I",
        'print "certificate_type=Macaulay2 Tropical tropicalVariety fan diagnostics"',
        'print "tropical_cycle_certified=true"',
        'print concatenate("class=", toString class T)',
        'print concatenate("rays=", replace("\\n", " || ", toString rays T))',
        'print concatenate("max_cones=", replace("\\n", " || ", toString maxCones T))',
        'print concatenate("lineality_space=", replace("\\n", " || ", toString linealitySpace T))',
        'print concatenate("multiplicities=", replace("\\n", " || ", toString multiplicities T))',
        'print concatenate("is_balanced=", toString isBalanced T)',
        'print concatenate("is_pure=", toString isPure T)',
        'print concatenate("is_simplicial=", toString isSimplicial T)',
        'print concatenate("fan_text=", replace("\\n", " || ", toString fan T))',
        'print "TROPICALGT_RESOLUTION_END"',
        "exit 0",
    ]
    return "\n".join(lines) + "\n"

def _certified_tropical_result(schema: dict[str, Any], parsed: dict[str, Any], tagged_output: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    package_available = _parse_bool(parsed.get("tropical_package_available"))
    cycle_certified = _parse_bool(parsed.get("tropical_cycle_certified"))
    rays_text = str(parsed.get("rays", "") or "")
    max_cones_text = str(parsed.get("max_cones", "") or "")
    if not (package_available and cycle_certified and rays_text and max_cones_text):
        return unavailable_tropical_fan_diagnostics(
            schema,
            status="certificate_failed",
            reason="Macaulay2 Tropical ran but did not emit rays/max cones/cycle certificate tags.",
            attempts=attempts,
        )
    rays = _parse_m2_matrix_rows(rays_text)
    max_cones = _parse_m2_index_sets(max_cones_text)
    multiplicities = _parse_m2_int_list(parsed.get("multiplicities", ""))
    lineality = _parse_m2_matrix_rows(parsed.get("lineality_space", ""))
    basis_available = _parse_bool(parsed.get("tropical_basis_check_available"))
    prevariety_available = _parse_bool(parsed.get("tropical_prevariety_available"))
    prevariety_rays = _parse_m2_matrix_rows(parsed.get("prevariety_rays", "")) if prevariety_available else []
    prevariety_max_cones = _parse_m2_index_sets(parsed.get("prevariety_max_cones", "")) if prevariety_available else []
    prevariety_lineality = _parse_m2_matrix_rows(parsed.get("prevariety_lineality_space", "")) if prevariety_available else []
    prevariety_multiplicities = _parse_m2_int_list(parsed.get("prevariety_multiplicities", "")) if prevariety_available else []
    return {
        "schema_version": TROPICAL_SCHEMA_VERSION,
        "available": True,
        "status": "certified",
        "input_sha256": schema["input_sha256"],
        "ideal_schema": schema,
        "backend": "Macaulay2",
        "backend_probe": probe_tropical_backends(),
        "backend_attempts": attempts,
        "command_template": build_macaulay2_tropical_script(schema),
        "certificate_contract": tropical_fan_certificate_contract(),
        "certificate_attached": True,
        "certificate_type": parsed.get("certificate_type", "Macaulay2 Tropical tropicalVariety fan diagnostics"),
        "tropical_cycle_certified": True,
        "fan_diagnostics_certified": True,
        "safe_to_render_as_tropical_fan": True,
        "cas_artifacts": {
            "raw_tagged_output": tagged_output,
            "class": parsed.get("class", ""),
            "rays_text": rays_text,
            "max_cones_text": max_cones_text,
            "lineality_space_text": str(parsed.get("lineality_space", "") or ""),
            "multiplicities_text": str(parsed.get("multiplicities", "") or ""),
            "fan_text": str(parsed.get("fan_text", "") or ""),
            "is_tropical_basis": str(parsed.get("is_tropical_basis", "") or ""),
            "is_tropical_basis_error": str(parsed.get("is_tropical_basis_error", "") or ""),
            "prevariety_rays_text": str(parsed.get("prevariety_rays", "") or ""),
            "prevariety_max_cones_text": str(parsed.get("prevariety_max_cones", "") or ""),
            "prevariety_lineality_space_text": str(parsed.get("prevariety_lineality_space", "") or ""),
            "prevariety_multiplicities_text": str(parsed.get("prevariety_multiplicities", "") or ""),
        },
        "tropical_basis_check": {
            "available": bool(basis_available),
            "is_tropical_basis": _parse_bool(parsed.get("is_tropical_basis")) if basis_available else None,
            "error": None if basis_available else str(parsed.get("is_tropical_basis_error", "not computed") or "not computed"),
            "method": "Macaulay2 Tropical isTropicalBasis on flatten entries gens I",
        },
        "tropical_prevariety_summary": {
            "available": bool(prevariety_available),
            "class": str(parsed.get("prevariety_class", "") or "") if prevariety_available else "",
            "rays": prevariety_rays,
            "max_cones": prevariety_max_cones,
            "lineality_space": prevariety_lineality,
            "multiplicities": prevariety_multiplicities,
            "ray_count": len(prevariety_rays[0]) if prevariety_rays else 0,
            "ambient_dimension": len(prevariety_rays),
            "max_cone_count": len(prevariety_max_cones),
            "is_balanced": _parse_bool(parsed.get("prevariety_is_balanced")),
            "is_pure": _parse_bool(parsed.get("prevariety_is_pure")),
            "is_simplicial": _parse_bool(parsed.get("prevariety_is_simplicial")),
            "errors": {
                "multiplicities": str(parsed.get("prevariety_multiplicities_error", "") or ""),
                "is_balanced": str(parsed.get("prevariety_is_balanced_error", "") or ""),
                "is_pure": str(parsed.get("prevariety_is_pure_error", "") or ""),
                "is_simplicial": str(parsed.get("prevariety_is_simplicial_error", "") or ""),
            },
        },
        "fan_summary": {
            "rays": rays,
            "max_cones": max_cones,
            "lineality_space": lineality,
            "multiplicities": multiplicities,
            "ray_count": len(rays[0]) if rays else 0,
            "ambient_dimension": len(rays),
            "max_cone_count": len(max_cones),
            "is_balanced": _parse_bool(parsed.get("is_balanced")),
            "is_pure": _parse_bool(parsed.get("is_pure")),
            "is_simplicial": _parse_bool(parsed.get("is_simplicial")),
            "one_dimensional_cone_language": "rays are one dimensional cones of the certified Macaulay2 tropical fan/cycle",
        },
        "render_warning": "Certified tropical fan diagnostics only. This is not a multigraded free-resolution or derived-equivalence certificate.",
    }


def _parse_m2_matrix_rows(text: Any) -> list[list[int]]:
    raw = str(text or "")
    if "matrix" in raw:
        raw = raw.split("matrix", 1)[1]
    return _parse_nested_int_lists(raw)


def _parse_m2_index_sets(text: Any) -> list[list[int]]:
    return _parse_nested_int_lists(str(text or ""))


def _parse_m2_int_list(text: Any) -> list[int]:
    nested = _parse_nested_int_lists(str(text or ""))
    if len(nested) == 1:
        return nested[0]
    if nested:
        return [value for row in nested for value in row]
    return []


def _parse_nested_int_lists(text: str) -> list[list[int]]:
    cleaned = str(text).strip().replace("{", "[").replace("}", "]")
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except Exception:
        return []
    if isinstance(parsed, list) and all(isinstance(item, list) for item in parsed):
        out: list[list[int]] = []
        for item in parsed:
            try:
                out.append([int(value) for value in item])
            except Exception:
                continue
        return out
    if isinstance(parsed, list):
        try:
            return [[int(value) for value in parsed]]
        except Exception:
            return []
    return []


def _tropical_cache_context(schema: dict[str, Any], *, use_cache: bool | None) -> dict[str, Any]:
    enabled = bool(use_cache) if use_cache is not None else True
    cache_root = Path(__file__).resolve().parents[3] / ".cache" / "tropicalgt" / "cas_tropical"
    key_payload = {
        "cache_schema_version": TROPICAL_CACHE_SCHEMA_VERSION,
        "adapter_cache_version": TROPICAL_CACHE_VERSION,
        "ideal": schema,
    }
    key = sha256_json(key_payload)
    return {"enabled": enabled, "root": cache_root, "key": key, "path": cache_root / f"{key}.json"}


def _load_cached_tropical(context: dict[str, Any]) -> dict[str, Any] | None:
    if not context.get("enabled"):
        return None
    path = Path(context["path"])
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if data.get("cache", {}).get("adapter_cache_version") != TROPICAL_CACHE_VERSION:
        return None
    data["cache"]["hit"] = True
    return data


def _cache_and_annotate_tropical(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    cache = {
        "enabled": bool(context.get("enabled")),
        "hit": False,
        "written": False,
        "key": context.get("key"),
        "adapter_cache_version": TROPICAL_CACHE_VERSION,
    }
    report["cache"] = cache
    if not context.get("enabled"):
        return report
    path = Path(context["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report), encoding="utf-8")
    report["cache"]["written"] = True
    return report
