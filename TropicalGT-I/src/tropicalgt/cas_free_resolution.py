from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


SCHEMA_VERSION = "tropicalgt.real_free_resolution.v1"
MODULE_SCHEMA_VERSION = "tropicalgt.level_radius_module.v1"
CACHE_SCHEMA_VERSION = "tropicalgt.real_free_resolution.cache.v1"
ADAPTER_CACHE_VERSION = "2026-06-15.cas-free-resolution-cache-v2"
SUPPORTED_RINGS = {
    "F2[x_level,x_radius]": ["x_level", "x_radius"],
    "F2[x_filtration,x_dimension]": ["x_filtration", "x_dimension"],
    "F2[x_filtration,x_dimension,x_position]": ["x_filtration", "x_dimension", "x_position"],
}
UNAVAILABLE_STATUSES = {
    "unavailable_no_certificate",
    "backend_not_installed",
    "backend_error",
    "timeout",
    "unsupported_ring",
    "invalid_grading",
    "parse_error",
    "certificate_failed",
    "disabled_by_environment",
    "complexity_guard",
}

DEFAULT_CAS_MAX_PRESENTATION_CELLS = 4096
DEFAULT_CAS_MAX_DETERMINANT_ORDER = 8


@dataclass(frozen=True)
class CASBackend:
    name: str
    executable: str | None
    available: bool
    version: str | None = None


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(data: Any) -> str:
    return sha256(canonical_json(data).encode("utf-8")).hexdigest()


def ring_variable_names(ring: str) -> list[str]:
    if ring not in SUPPORTED_RINGS:
        raise ValueError(f"unsupported coefficient ring for certified CAS resolution: {ring!r}")
    return list(SUPPORTED_RINGS[ring])


def module_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def probe_cas_backends() -> dict[str, Any]:
    backends = []
    for name in ("M2", "Singular", "sage"):
        executable = _candidate_executable(name)
        backends.append(
            {
                "name": name,
                "executable": executable,
                "available": executable is not None,
                "version": _backend_version(executable) if executable else None,
            }
        )
    return {
        "backends": backends,
        "preferred_order": ["M2", "sage", "Singular"],
        "python_modules": {
            "sageall": module_available("sageall"),
            "sage": module_available("sage"),
            "BEMultipliers": module_available("BEMultipliers"),
            "bemultipliers": module_available("bemultipliers"),
        },
    }


def probe_bemultipliers() -> dict[str, Any]:
    package_path = _local_bemultipliers_package_path()
    m2_executable = _candidate_executable("M2")
    loader = f'load "{package_path}"' if package_path else 'needsPackage "BuchsbaumEisenbudMultipliers"'
    return {
        "role": "optional_buchsbaum_eisenbud_diagnostics_after_certified_resolution",
        "available_python_modules": {
            "BEMultipliers": module_available("BEMultipliers"),
            "bemultipliers": module_available("bemultipliers"),
        },
        "local_macaulay2_package": str(package_path) if package_path else None,
        "macaulay2_executable": m2_executable,
        "macaulay2_loader": loader,
        "macaulay2_loadable": bool(package_path and m2_executable),
        "execution_policy": "load only after a CAS-certified free resolution and bounded multiplier request; never substitute multiplier output for a free-resolution certificate",
        "repository": "https://github.com/amelie-iska/BEMultipliers.git",
        "is_resolution_backend": False,
    }


def _local_bemultipliers_package_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "external" / "BEMultipliers" / "BuchsbaumEisenbudMultipliers.m2",
        Path.cwd() / "external" / "BEMultipliers" / "BuchsbaumEisenbudMultipliers.m2",
    ]
    for raw in os.environ.get("TROPICALGT_BEMULTIPLIERS_PATH", "").split(os.pathsep):
        if raw:
            candidates.insert(0, Path(raw))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def canonicalize_module(module: dict[str, Any]) -> dict[str, Any]:
    ring = str(module.get("coefficient_ring", ""))
    variables = ring_variable_names(ring)
    generators = []
    id_by_simplex: dict[tuple[str, ...], str] = {}
    for idx, generator in enumerate(module.get("chain_module_generators", []) or []):
        simplex = tuple(str(v) for v in generator.get("simplex", []))
        degree = int(generator.get("homological_degree", len(simplex) - 1 if simplex else 0))
        multidegree = _nonnegative_tuple(generator.get("multidegree", []), len(variables), "generator multidegree")
        generator_id = generator.get("generator_id") or _safe_id("g", [idx, degree, *simplex, *multidegree])
        row = {
            "generator_id": str(generator_id),
            "simplex": list(simplex),
            "homological_degree": degree,
            "multidegree": multidegree,
            "filtration": float(generator.get("filtration", generator.get("radius", 0.0)) or 0.0),
            "first_level": int(generator.get("first_level", multidegree[0] if multidegree else 0) or 0),
            "radius_grade": int(generator.get("radius_grade", multidegree[1] if len(multidegree) > 1 else 0) or 0),
        }
        generators.append(row)
        id_by_simplex[simplex] = str(generator_id)

    generator_ids = {str(row["generator_id"]) for row in generators}
    boundaries = []
    for idx, entry in enumerate(_iter_boundary_entries(module.get("boundary_monomials", []) or [])):
        source_simplex = tuple(str(v) for v in entry.get("source_simplex", []))
        target_face = tuple(str(v) for v in entry.get("target_face", []))
        source_generator_id = entry.get("source_generator_id")
        target_generator_id = entry.get("target_generator_id")
        if source_generator_id is None:
            source_generator_id = id_by_simplex.get(source_simplex)
        if target_generator_id is None:
            target_generator_id = id_by_simplex.get(target_face)
        source_generator_id = str(source_generator_id) if source_generator_id is not None else None
        target_generator_id = str(target_generator_id) if target_generator_id is not None else None
        exponent_value = entry.get("monomial_exponent", entry.get("exponent", []))
        exponent = _nonnegative_tuple(exponent_value, len(variables), "boundary monomial exponent")
        boundaries.append(
            {
                "boundary_id": str(entry.get("boundary_id") or _safe_id("d", [idx, source_generator_id, "to", target_generator_id, *source_simplex, *target_face, *exponent])),
                "source_simplex": list(source_simplex),
                "target_face": list(target_face),
                "source_generator_id": source_generator_id if source_generator_id in generator_ids else None,
                "target_generator_id": target_generator_id if target_generator_id in generator_ids else None,
                "monomial_exponent": exponent,
                "monomial": _monomial_from_exponent(exponent, variables),
                "sign": int(entry.get("sign", 1) or 1) % 2,
            }
        )

    schema = {
        "schema_version": MODULE_SCHEMA_VERSION,
        "coefficient_ring": ring,
        "variables": variables,
        "generators": generators,
        "boundary_monomials": boundaries,
        "presentation_matrix": _presentation_from_d1(generators, boundaries, variables),
        "source": module.get("source", "level_radius_bifiltration"),
    }
    schema["input_sha256"] = sha256_json(schema)
    return schema


def unavailable_real_resolution(
    module: dict[str, Any],
    *,
    status: str = "unavailable_no_certificate",
    reason: str | None = None,
    error: str | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if status not in UNAVAILABLE_STATUSES:
        raise ValueError(f"unsupported unavailable status: {status}")
    try:
        if module.get("schema_version") == MODULE_SCHEMA_VERSION and module.get("input_sha256"):
            module_schema = dict(module)
        else:
            module_schema = canonicalize_module(module)
        module_summary = _module_summary(module_schema)
        input_hash = module_schema["input_sha256"]
        templates = cas_command_templates(module_schema)
    except Exception as exc:
        status = "invalid_grading" if status == "unavailable_no_certificate" else status
        module_schema = {}
        module_summary = {}
        input_hash = None
        templates = {}
        error = error or str(exc)
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "status": status,
        "reason": reason or error or "No CAS backend returned a certified real free resolution.",
        "coefficient_ring": module.get("coefficient_ring", ""),
        "module_schema_version": module_schema.get("schema_version"),
        "input_sha256": input_hash,
        "module_summary": module_summary,
        "backend_attempts": attempts or [],
        "backend_probe": probe_cas_backends(),
        "bemultipliers_probe": probe_bemultipliers(),
        "cas_artifacts": {},
        "command_templates": templates,
        "certificate_attached": False,
        "real_free_resolution_certified": False,
        "total_graded_resolution_certified": False,
        "ungraded_resolution_certified": False,
        "multigraded_free_resolution_certified": False,
        "exactness_certified": False,
        "minimality_certified": False,
        "safe_to_render_as_real_free_resolution": False,
        "safe_to_render_as_total_graded_resolution": False,
        "safe_to_render_as_multigraded_free_resolution": False,
        "render_warning": "No certified CAS free resolution is available for this module.",
    }


def try_compute_real_free_resolution(module: dict[str, Any], *, timeout_s: float = 20.0, use_cache: bool | None = None) -> dict[str, Any]:
    try:
        module_schema = canonicalize_module(module)
    except Exception as exc:
        return unavailable_real_resolution(module, status="invalid_grading", error=str(exc))

    disabled_reason = _cas_resolution_disabled_reason()
    if disabled_reason:
        report = unavailable_real_resolution(
            module,
            status="disabled_by_environment",
            reason=disabled_reason,
            attempts=[],
        )
        report["cache"] = {
            "enabled": False,
            "hit": False,
            "reason": "CAS execution disabled by environment",
        }
        return report

    cache_context = _cache_context(module_schema, use_cache=use_cache)
    cached = _load_cached_result(cache_context)
    if cached is not None:
        return cached

    attempts: list[dict[str, Any]] = []
    m2 = _candidate_executable("M2")
    if m2:
        guard = _cas_backend_complexity_guard("M2", module_schema)
        if guard is not None:
            attempts.append(guard)
        else:
            result = _run_tagged_cas_script(
                name="M2",
                executable=m2,
                script=build_macaulay2_script(module_schema),
                suffix=".m2",
                timeout_s=timeout_s,
            )
            attempts.append(result.get("attempt", {}))
            if result.get("available"):
                return _cache_and_annotate(_certified_result(module_schema, result, attempts), cache_context)
    sage = _candidate_executable("sage")
    if sage:
        guard = _cas_backend_complexity_guard("sage", module_schema)
        if guard is not None:
            attempts.append(guard)
        else:
            result = _run_tagged_cas_script(
                name="sage",
                executable=sage,
                script=build_sage_python_script(module_schema),
                suffix=".sage.py",
                timeout_s=timeout_s,
            )
            attempts.append(result.get("attempt", {}))
            if result.get("available"):
                return _cache_and_annotate(_certified_result(module_schema, result, attempts), cache_context)
    singular = _candidate_executable("Singular")
    if singular:
        guard = _cas_backend_complexity_guard("Singular", module_schema)
        if guard is not None:
            attempts.append(guard)
        else:
            result = _run_tagged_cas_script(
                name="Singular",
                executable=singular,
                script=build_singular_script(module_schema),
                suffix=".sing",
                timeout_s=timeout_s,
            )
            attempts.append(result.get("attempt", {}))
            if result.get("available"):
                return _cache_and_annotate(_certified_result(module_schema, result, attempts), cache_context)
    status, reason = _final_unavailable_status_and_reason(attempts)
    return _cache_and_annotate(unavailable_real_resolution(module, status=status, reason=reason, attempts=attempts), cache_context)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _cas_backend_complexity_guard(name: str, module_schema: dict[str, Any]) -> dict[str, Any] | None:
    pmat = module_schema.get("presentation_matrix", {})
    rows = int(pmat.get("rows", 0) or 0)
    cols = int(pmat.get("cols", 0) or 0)
    if rows <= 0 or cols <= 0:
        return None
    cells = rows * cols
    determinant_order = min(rows, cols)
    max_cells = _env_int("TROPICALGT_CAS_MAX_PRESENTATION_CELLS", DEFAULT_CAS_MAX_PRESENTATION_CELLS)
    max_order = _env_int("TROPICALGT_CAS_MAX_DETERMINANT_ORDER", DEFAULT_CAS_MAX_DETERMINANT_ORDER)
    reasons: list[str] = []
    if max_cells > 0 and cells > max_cells:
        reasons.append(f"presentation has {cells} matrix cells, above limit {max_cells}")
    if name in {"M2", "Singular"} and max_order > 0 and determinant_order > max_order:
        reasons.append(f"determinantal order {determinant_order} is above exact-minor limit {max_order}")
    if not reasons:
        return None
    return {
        "backend": name,
        "status": "skipped_complexity_guard",
        "presentation_shape": [rows, cols],
        "presentation_cells": cells,
        "determinantal_order": determinant_order,
        "limits": {
            "max_presentation_cells": max_cells,
            "max_determinant_order": max_order,
        },
        "reason": "; ".join(reasons),
    }


def _final_unavailable_status_and_reason(attempts: list[dict[str, Any]]) -> tuple[str, str | None]:
    if not attempts:
        return "backend_not_installed", None
    statuses = {str(attempt.get("status", "")) for attempt in attempts if attempt}
    if statuses and statuses <= {"skipped_complexity_guard"}:
        reasons = sorted({str(attempt.get("reason", "")) for attempt in attempts if attempt.get("reason")})
        reason = "CAS presentation exceeded configured complexity limits before a certified backend run was safe."
        if reasons:
            reason += " " + " | ".join(reasons)
        return "complexity_guard", reason
    if "timeout" in statuses and statuses <= {"timeout", "skipped_complexity_guard"}:
        return "timeout", "CAS backend timed out before producing a certificate; guarded attempts were not substituted as resolutions."
    return "certificate_failed", None


def cas_command_templates(module_schema: dict[str, Any]) -> dict[str, str]:
    return {
        "macaulay2": build_macaulay2_script(module_schema),
        "singular": build_singular_script(module_schema),
        "sage": build_sage_python_script(module_schema),
    }


def build_macaulay2_script(module_schema: dict[str, Any]) -> str:
    variables = module_schema.get("variables", ["x_level", "x_radius"])
    pmat = module_schema.get("presentation_matrix", {})
    rows = int(pmat.get("rows", 0))
    cols = int(pmat.get("cols", 0))
    entries = pmat.get("entries", [])
    degree_spec = "{" + ",".join("{" + ",".join("1" if i == j else "0" for i in range(len(variables))) + "}" for j in range(len(variables))) + "}"
    row_degrees = _presentation_generator_degrees(module_schema, pmat.get("row_generators", []))
    col_degrees = _presentation_generator_degrees(module_schema, pmat.get("col_generators", []))
    if rows <= 0:
        return "\n".join(
            [
                "-- TropicalGT Macaulay2 multigraded resolution probe",
                f"R = ZZ/2[{','.join(variables)}, Degrees=>{degree_spec}]",
                "print \"TROPICALGT_RESOLUTION_BEGIN\"",
                "print \"backend=Macaulay2\"",
                "print \"exactness_certified=false\"",
                "print \"minimality_certified=false\"",
                "print \"certificate_error=no degree-zero free module rows in the presentation\"",
                f"print \"presentation_shape={rows}x{cols}\"",
                "print \"TROPICALGT_RESOLUTION_END\"",
                "exit 2",
            ]
        )
    if cols <= 0:
        row_shift_spec = _macaulay2_free_module_shift_spec(row_degrees)
        return "\n".join(
            [
                "-- TropicalGT Macaulay2 multigraded resolution probe",
                f"R = ZZ/2[{','.join(variables)}, Degrees=>{degree_spec}]",
                f"F0 = R^{row_shift_spec}",
                "print \"TROPICALGT_RESOLUTION_BEGIN\"",
                "print \"backend=Macaulay2\"",
                "print \"exactness_certified=true\"",
                "print \"minimality_certified=true\"",
                "print \"certificate_type=Macaulay2 trivial free cokernel with no relations\"",
                f"print \"presentation_shape={rows}x{cols}\"",
                "print \"macaulay2_free_modules_begin\"",
                "print concatenate(\"F0_degrees=\", toString degrees F0)",
                "print \"macaulay2_free_modules_end\"",
                "print \"fitting_ideals_begin\"",
                "print \"Fitt0=ideal 0_R\"",
                "print \"fitting_ideals_end\"",
                "print \"minors_begin\"",
                "print \"minors_0=ideal 1_R\"",
                "print \"minors_end\"",
                *_macaulay2_be_diagnostics_lines(
                    exactness_expression="true",
                    minimality_expression="true",
                    source="Macaulay2 trivial free cokernel certificate",
                    run_multiplier=False,
                    skip_reason="trivial free cokernel has no nonzero certified ChainComplex multiplier target",
                ),
                "print \"TROPICALGT_RESOLUTION_END\"",
                "exit 0",
            ]
        )
    matrix_rows: list[list[str]] = [["0_R" for _ in range(cols)] for _ in range(rows)]
    for entry in entries:
        matrix_rows[int(entry["row"])][int(entry["col"])] = _m2_monomial(entry["exponent"], variables)
    matrix_literal = "{" + ",".join("{" + ",".join(row) + "}" for row in matrix_rows) + "}"
    row_shift_spec = _macaulay2_free_module_shift_spec(row_degrees)
    col_shift_spec = _macaulay2_free_module_shift_spec(col_degrees)
    max_minors = min(rows, cols)
    differential_lines: list[str] = []
    max_resolution_len = min(cols + len(variables) + 3, 24)
    for i in range(1, max_resolution_len + 1):
        differential_lines.extend(
            [
                f"if length C >= {i} then (",
                f"  print concatenate(\"d{i}_shape=\", toString numRows C.dd_{i}, \"x\", toString numColumns C.dd_{i});",
                f"  print concatenate(\"d{i}_source_degrees=\", toString degrees source C.dd_{i});",
                f"  print concatenate(\"d{i}_target_degrees=\", toString degrees target C.dd_{i});",
                f"  print concatenate(\"d{i}_matrix=\", replace(\"\\n\", \" || \", toString C.dd_{i}));",
                ")",
            ]
        )
    minor_lines = []
    for k in range(1, max_minors + 1):
        minor_lines.append(f"print concatenate(\"minors_{k}=\", replace(\"\\n\", \" \", toString minors({k}, M)))")
    fitting_lines = [
        "print concatenate(\"Fitt0=\", replace(\"\\n\", \" \", toString fittingIdeal(0, N)))",
        "print concatenate(\"Fitt1=\", replace(\"\\n\", \" \", toString fittingIdeal(1, N)))",
    ]
    return "\n".join(
        [
            "-- TropicalGT certified multigraded free resolution probe generated from model audit data",
            f"R = ZZ/2[{','.join(variables)}, Degrees=>{degree_spec}]",
            f"F0 = R^{row_shift_spec}",
            f"F1 = R^{col_shift_spec}",
            f"M = map(F0, F1, matrix {matrix_literal})",
            "homogeneousPresentation = isHomogeneous M",
            "if not homogeneousPresentation then (",
            "  print \"TROPICALGT_RESOLUTION_BEGIN\";",
            "  print \"backend=Macaulay2\";",
            "  print \"exactness_certified=false\";",
            "  print \"minimality_certified=false\";",
            "  print \"certificate_error=presentation matrix is not homogeneous for stored multidegrees\";",
            f"  print \"presentation_shape={rows}x{cols}\";",
            "  print concatenate(\"row_generator_degrees=\", toString degrees F0);",
            "  print concatenate(\"col_generator_degrees=\", toString degrees F1);",
            "  print \"TROPICALGT_RESOLUTION_END\";",
            "  exit 0;",
            ")",
            "N = coker M",
            "C = res N",
            "higherHomologyGeneratorCount = 0",
            "for i from 1 to length C do (",
            "  H = prune HH_i C;",
            "  higherHomologyGeneratorCount = higherHomologyGeneratorCount + numgens source presentation H;",
            ")",
            "okExact = (higherHomologyGeneratorCount == 0)",
            "B = betti C",
            "print \"TROPICALGT_RESOLUTION_BEGIN\"",
            "print \"backend=Macaulay2\"",
            "print concatenate(\"exactness_certified=\", toString okExact)",
            "print \"minimality_certified=true\"",
            "print concatenate(\"homogeneous_presentation=\", toString homogeneousPresentation)",
            "print \"certificate_type=Macaulay2 res coker presentation over multigraded F2 polynomial ring\"",
            f"print \"presentation_shape={rows}x{cols}\"",
            "print \"betti_table_begin\"",
            "print B",
            "print \"betti_table_end\"",
            "print \"macaulay2_free_modules_begin\"",
            "if length C >= 1 then print concatenate(\"F0_degrees=\", toString degrees target C.dd_1)",
            *[f"if length C >= {i} then print concatenate(\"F{i}_degrees=\", toString degrees source C.dd_{i})" for i in range(1, max_resolution_len + 1)],
            "print \"macaulay2_free_modules_end\"",
            "print \"macaulay2_differentials_begin\"",
            *differential_lines,
            "print \"macaulay2_differentials_end\"",
            "print \"fitting_ideals_begin\"",
            *fitting_lines,
            "print \"fitting_ideals_end\"",
            "print \"minors_begin\"",
            *minor_lines,
            "print \"minors_end\"",
            *_macaulay2_be_diagnostics_lines(
                exactness_expression="okExact",
                minimality_expression="true",
                source="Macaulay2 res/HH exactness certificate for the displayed cokernel presentation",
                run_multiplier=_macaulay2_bemultipliers_allowed(rows, cols, max_minors),
                skip_reason=_macaulay2_bemultipliers_skip_reason(rows, cols, max_minors),
                chain_complex_name="C",
            ),
            "print \"TROPICALGT_RESOLUTION_END\"",
            "exit 0",
        ]
    )



def _macaulay2_bemultipliers_allowed(rows: int, cols: int, determinant_order: int) -> bool:
    if _local_bemultipliers_package_path() is None:
        return False
    if rows <= 0 or cols <= 0:
        return False
    max_cells = _env_int("TROPICALGT_CAS_MAX_BEM_CELLS", 64)
    max_order = _env_int("TROPICALGT_CAS_MAX_BEM_ORDER", 4)
    if max_cells > 0 and rows * cols > max_cells:
        return False
    if max_order > 0 and determinant_order > max_order:
        return False
    return True


def _macaulay2_bemultipliers_skip_reason(rows: int, cols: int, determinant_order: int) -> str:
    if _local_bemultipliers_package_path() is None:
        return "unavailable_no_package_path"
    if rows <= 0 or cols <= 0:
        return "no_nonzero_presentation_matrix"
    max_cells = _env_int("TROPICALGT_CAS_MAX_BEM_CELLS", 64)
    max_order = _env_int("TROPICALGT_CAS_MAX_BEM_ORDER", 4)
    reasons: list[str] = []
    if max_cells > 0 and rows * cols > max_cells:
        reasons.append(f"presentation_cells_{rows * cols}_above_bem_limit_{max_cells}")
    if max_order > 0 and determinant_order > max_order:
        reasons.append(f"determinantal_order_{determinant_order}_above_bem_limit_{max_order}")
    return ";".join(reasons) if reasons else ""


def _macaulay2_be_diagnostics_lines(
    *,
    exactness_expression: str,
    minimality_expression: str,
    source: str,
    run_multiplier: bool = False,
    skip_reason: str = "",
    chain_complex_name: str = "C",
) -> list[str]:
    package_path = _local_bemultipliers_package_path()
    safe_source = _cas_text_literal(source)
    if package_path:
        bem_status = "ready_to_run" if run_multiplier else (skip_reason or "package_path_detected_not_loaded_without_explicit_multiplier_run")
        bem_path = str(package_path)
    else:
        bem_status = "unavailable_no_package_path"
        bem_path = ""
    safe_status = _cas_text_literal(bem_status)
    safe_path = _cas_text_literal(bem_path)
    lines = [
        "print \"buchsbaum_eisenbud_diagnostics_begin\"",
        "print \"backend_diagnostics_available=true\"",
        f"print concatenate(\"exactness_certified=\", toString {exactness_expression})",
        f"print concatenate(\"minimality_certified=\", toString {minimality_expression})",
        f"print \"be_exactness_source={safe_source}\"",
    ]
    if package_path and run_multiplier:
        loader = f'load "{_cas_text_literal(str(package_path))}"'
        safe_complex = _cas_text_literal(chain_complex_name)
        lines.extend(
            [
                "bemComputed = false",
                "bemStatus = \"not_run\"",
                "bemShape = \"\"",
                "bemMatrix = \"\"",
                f"bemLoad = try ({loader}; \"ok\") else \"load_error\"",
                f"if bemLoad == \"ok\" and {exactness_expression} then (",
                "  bemOutcome = try (",
                f"    ABE = aMultiplier(1,{safe_complex},ComputeRanks=>true);",
                "    bemComputed = true;",
                "    bemStatus = \"computed_aMultiplier_1\";",
                "    bemShape = concatenate(toString numRows ABE, \"x\", toString numColumns ABE);",
                "    bemMatrix = replace(\"\\n\", \" || \", toString ABE);",
                "    \"ok\"",
                "  ) else \"backend_error\";",
                "  if bemOutcome != \"ok\" then bemStatus = \"backend_error\";",
                ")",
                "if bemLoad != \"ok\" then bemStatus = bemLoad",
                "print concatenate(\"multiplier_output_available=\", toString bemComputed)",
                "print concatenate(\"bemultipliers_status=\", bemStatus)",
                "print concatenate(\"aMultiplier_1_shape=\", bemShape)",
                "print concatenate(\"aMultiplier_1_matrix=\", bemMatrix)",
            ]
        )
    else:
        lines.extend(
            [
                "print \"multiplier_output_available=false\"",
                f"print \"bemultipliers_status={safe_status}\"",
            ]
        )
    lines.extend(
        [
            f"print \"bemultipliers_package_path={safe_path}\"",
            "print \"bemultipliers_repository=https://github.com/amelie-iska/BEMultipliers.git\"",
            "print \"reason=Buchsbaum-Eisenbud multiplier output is rendered only after an explicit BEMultipliers run on a CAS-certified Macaulay2 ChainComplex; no multiplier data is substituted from rank tables or chain diagnostics\"",
            "print \"buchsbaum_eisenbud_diagnostics_end\"",
        ]
    )
    return lines


def _cas_text_literal(value: str) -> str:
    return str(value).replace("\\", "/").replace('"', "'").replace("\n", " ")


def build_singular_script(module_schema: dict[str, Any]) -> str:
    variables = module_schema.get("variables", ["x_level", "x_radius"])
    pmat = module_schema.get("presentation_matrix", {})
    entries = pmat.get("entries", [])
    rows = int(pmat.get("rows", 0))
    cols = int(pmat.get("cols", 0))
    unit_entries = sum(1 for entry in entries if all(int(power) == 0 for power in entry.get("exponent", [])))
    module_literal = _singular_module_literal(entries, rows=rows, cols=cols, variables=variables)
    matrix_literal = _singular_matrix_literal(entries, rows=rows, cols=cols, variables=variables) if rows > 0 and cols > 0 else ""
    if rows <= 0:
        body = [
            "print(\"TROPICALGT_RESOLUTION_BEGIN\");",
            "print(\"backend=Singular\");",
            "print(\"exactness_certified=false\");",
            "print(\"minimality_certified=false\");",
            "print(\"certificate_error=no degree-zero free module rows in the presentation\");",
            f"print(\"presentation_shape={rows}x{cols}\");",
            "print(\"TROPICALGT_RESOLUTION_END\");",
        ]
    elif cols <= 0:
        body = [
            "print(\"TROPICALGT_RESOLUTION_BEGIN\");",
            "print(\"backend=Singular\");",
            "print(\"exactness_certified=true\");",
            "print(\"minimality_certified=true\");",
            "print(\"certificate_type=trivial free module cokernel with no relations\");",
            f"print(\"presentation_shape={rows}x{cols}\");",
            "print(\"betti_table_begin\");",
            f"print(\"{rows}\");",
            "print(\"betti_table_end\");",
            "print(\"fitting_ideals_begin\");",
            "print(\"Fitt0=0\");",
            "print(\"fitting_ideals_end\");",
            "print(\"minors_begin\");",
            "print(\"minors_0=1\");",
            "print(\"minors_end\");",
            *_singular_be_diagnostics_lines(
                exactness=True,
                minimality=True,
                source="Singular trivial free cokernel certificate",
            ),
            "print(\"TROPICALGT_RESOLUTION_END\");",
        ]
    else:
        max_minors = min(rows, cols)
        minor_lines = ["print(\"minors_begin\");", "print(\"minors_0=1\");"]
        for k in range(1, max_minors + 1):
            minor_lines.extend([f"ideal Iminor{k}=minor(PM,{k});", f"print(\"minors_{k}=\"+string(Iminor{k}));"])
        minor_lines.append("print(\"minors_end\");")
        fitting_lines = ["print(\"fitting_ideals_begin\");"]
        for j in range(0, rows + 1):
            determinantal_order = rows - j
            if determinantal_order <= 0:
                fitting_lines.append(f"print(\"Fitt{j}=1\");")
            elif determinantal_order > max_minors:
                fitting_lines.append(f"print(\"Fitt{j}=0\");")
            else:
                fitting_lines.extend([
                    f"ideal IFitt{j}=minor(PM,{determinantal_order});",
                    f"print(\"Fitt{j}=\"+string(IFitt{j}));",
                ])
        fitting_lines.append("print(\"fitting_ideals_end\");")
        body = [
            f"module M = {module_literal};",
            f"matrix PM[{rows}][{cols}] = {matrix_literal};",
            "resolution R = mres(M,0);",
            "print(\"TROPICALGT_RESOLUTION_BEGIN\");",
            "print(\"backend=Singular\");",
            "print(\"exactness_certified=true\");",
            f"print(\"minimality_certified={'false' if unit_entries else 'true'}\");",
            "print(\"certificate_type=Singular mres image-submodule resolution prepended to the displayed cokernel presentation\");",
            f"print(\"presentation_shape={rows}x{cols}\");",
            f"print(\"unit_entries={unit_entries}\");",
            "print(\"betti_table_begin\");",
            "print(betti(R));",
            "print(\"betti_table_end\");",
            "print(\"singular_resolution_text_begin\");",
            "print(R);",
            "print(\"singular_resolution_text_end\");",
            *fitting_lines,
            *minor_lines,
            *_singular_be_diagnostics_lines(
                exactness=True,
                minimality=(unit_entries == 0),
                source="Singular mres plus exact determinantal/Fitting ideals for the displayed presentation matrix",
            ),
            "print(\"TROPICALGT_RESOLUTION_END\");",
        ]
    return "\n".join(
        [
            "// TropicalGT certified resolution probe generated from model audit data.",
            "// Singular mres resolves the image submodule M; TropicalGT prepends",
            "// the displayed free module F0 -> coker(M) to obtain the cokernel resolution.",
            "// The same presentation matrix PM is used for exact determinantal minors",
            "// and Fitting ideals Fitt_j(coker PM)=I_{rows-j}(PM).",
            f"ring r = 2,({','.join(variables)}),dp;",
            *body,
        ]
    )



def _singular_be_diagnostics_lines(*, exactness: bool, minimality: bool, source: str) -> list[str]:
    exact_text = "true" if exactness else "false"
    minimal_text = "true" if minimality else "false"
    safe_source = _cas_text_literal(source)
    return [
        "print(\"buchsbaum_eisenbud_diagnostics_begin\");",
        "print(\"backend_diagnostics_available=true\");",
        f"print(\"exactness_certified={exact_text}\");",
        f"print(\"minimality_certified={minimal_text}\");",
        f"print(\"be_exactness_source={safe_source}\");",
        "print(\"multiplier_output_available=false\");",
        "print(\"bemultipliers_status=unsupported_in_singular_adapter\");",
        "print(\"bemultipliers_repository=https://github.com/amelie-iska/BEMultipliers.git\");",
        "print(\"reason=Singular can provide exact determinantal and Fitting ideals here, but Buchsbaum-Eisenbud multiplier output requires a Macaulay2 BEMultipliers run and is not inferred\");",
        "print(\"buchsbaum_eisenbud_diagnostics_end\");",
    ]


def build_sage_python_script(module_schema: dict[str, Any]) -> str:
    payload = canonical_json(module_schema)
    script = r"""
# TropicalGT Sage bridge for certified total-graded free resolutions.
import json
from sage.all import GF, PolynomialRing

module_schema = json.loads(__PAYLOAD__)
pmat = module_schema.get("presentation_matrix", {})
rows = int(pmat.get("rows", 0))
cols = int(pmat.get("cols", 0))
entries = pmat.get("entries", [])
variables = list(module_schema.get("variables", ["x_level", "x_radius"]))


def emit_failure(message):
    print("TROPICALGT_RESOLUTION_BEGIN")
    print("backend=sage")
    print("exactness_certified=false")
    print("minimality_certified=false")
    print("certificate_error=" + str(message).replace("\n", " "))
    print("presentation_shape=" + str(rows) + "x" + str(cols))
    print("TROPICALGT_RESOLUTION_END")


def poly_from_exponent(R, gens, exponent):
    term = R(1)
    for gen, power in zip(gens, exponent):
        term *= gen ** int(power)
    return term


try:
    R = PolynomialRing(GF(2), tuple(variables), order="degrevlex")
    gens = R.gens()
    if rows <= 0:
        emit_failure("no degree-zero free module rows in the presentation")
        raise SystemExit(2)
    if cols <= 0:
        summary = {
            "available": True,
            "backend": "sage",
            "grading": "total_graded_betti_ranks",
            "betti_by_homological_and_total_degree": {"0": {"0": int(rows)}},
            "free_modules": [{"homological_degree": 0, "total_degree": 0, "rank": int(rows), "display": "F_0 contains S^%d" % int(rows)}],
            "not_multigraded": True,
            "safe_for_multigraded_claims": False,
            "interpretation": "Sage certified a trivial free cokernel with no relations. This is real, but not multigraded data.",
        }
        print("TROPICALGT_RESOLUTION_BEGIN")
        print("backend=sage")
        print("exactness_certified=true")
        print("minimality_certified=true")
        print("certificate_type=Sage trivial free module cokernel with no relations")
        print("presentation_shape=" + str(rows) + "x" + str(cols))
        print("total_graded_betti_json=" + json.dumps(summary, sort_keys=True))
        print("differentials_json=[]")
        print("sage_resolution_text=S^%d <-- 0" % rows)
        print("buchsbaum_eisenbud_diagnostics_begin")
        print("backend_diagnostics_available=true")
        print("exactness_certified=true")
        print("minimality_certified=true")
        print("be_exactness_source=Sage trivial total-graded free cokernel certificate")
        print("multiplier_output_available=false")
        print("bemultipliers_status=unsupported_in_sage_total_graded_adapter")
        print("reason=Sage adapter output is total-graded only; Buchsbaum-Eisenbud multiplier output requires Macaulay2/BEMultipliers and is not inferred")
        print("buchsbaum_eisenbud_diagnostics_end")
        print("TROPICALGT_RESOLUTION_END")
        raise SystemExit(0)
    if rows != 1:
        emit_failure("Sage adapter certifies only one-row cokernel presentations S/I; multi-row module cokernels require Macaulay2 or Singular")
        raise SystemExit(2)

    polys = [R(0) for _ in range(cols)]
    for entry in entries:
        row = int(entry.get("row", 0))
        col = int(entry.get("col", 0))
        if row == 0 and 0 <= col < cols:
            polys[col] += poly_from_exponent(R, gens, entry.get("exponent", []))
    ideal_polys = [poly for poly in polys if poly != 0]
    if not ideal_polys:
        emit_failure("one-row presentation has no nonzero ideal generators")
        raise SystemExit(2)

    ideal = R.ideal(ideal_polys)
    resolution = ideal.graded_free_resolution()
    max_i = max(4, cols + len(variables) + 4)
    betti = {}
    for i in range(max_i):
        row = resolution.betti(i)
        if row:
            betti[str(i)] = {str(int(deg)): int(rank) for deg, rank in sorted(row.items()) if int(rank) != 0}
    free_modules = []
    for i_text, row in sorted(betti.items(), key=lambda item: int(item[0])):
        i = int(i_text)
        for deg_text, rank in sorted(row.items(), key=lambda item: int(item[0])):
            deg = int(deg_text)
            free_modules.append({
                "homological_degree": i,
                "total_degree": deg,
                "rank": int(rank),
                "display": "F_%d contains S(-%d)^%d" % (i, deg, int(rank)),
            })

    differentials = []
    for i in range(1, max_i):
        try:
            matrix = resolution.matrix(i)
        except Exception:
            continue
        nrows = int(matrix.nrows())
        ncols = int(matrix.ncols())
        if nrows == 0 and ncols == 0:
            continue
        matrix_entries = []
        for r in range(nrows):
            for c in range(ncols):
                value = matrix[r, c]
                if value != 0:
                    matrix_entries.append({"row": int(r), "column": int(c), "entry": str(value)})
        differentials.append({"homological_degree": int(i), "rows": nrows, "cols": ncols, "entries": matrix_entries})

    summary = {
        "available": True,
        "backend": "sage",
        "grading": "total_graded_betti_ranks",
        "betti_by_homological_and_total_degree": betti,
        "free_modules": free_modules,
        "not_multigraded": True,
        "safe_for_multigraded_claims": False,
        "interpretation": "Sage certified a minimal total-graded free resolution of S/I for the one-row cokernel presentation. This is a real resolution, but it is not a multigraded F2[x,y] resolution.",
    }
    print("TROPICALGT_RESOLUTION_BEGIN")
    print("backend=sage")
    print("exactness_certified=true")
    print("minimality_certified=true")
    print("certificate_type=Sage graded_free_resolution of the one-row cokernel ideal quotient")
    print("presentation_shape=" + str(rows) + "x" + str(cols))
    print("total_graded_betti_json=" + json.dumps(summary, sort_keys=True))
    print("differentials_json=" + json.dumps(differentials, sort_keys=True))
    print("sage_resolution_text=" + str(resolution).replace("\n", "; "))
    print("buchsbaum_eisenbud_diagnostics_begin")
    print("backend_diagnostics_available=true")
    print("exactness_certified=true")
    print("minimality_certified=true")
    print("be_exactness_source=Sage total-graded S/I resolution certificate")
    print("multiplier_output_available=false")
    print("bemultipliers_status=unsupported_in_sage_total_graded_adapter")
    print("reason=Sage adapter output is total-graded only; Buchsbaum-Eisenbud multiplier output requires Macaulay2/BEMultipliers and is not inferred")
    print("buchsbaum_eisenbud_diagnostics_end")
    print("TROPICALGT_RESOLUTION_END")
except SystemExit:
    raise
except Exception as exc:
    emit_failure(type(exc).__name__ + ": " + str(exc))
    raise SystemExit(2)
"""
    return script.replace("__PAYLOAD__", repr(payload)).strip() + "\n"

def _certified_result(module_schema: dict[str, Any], backend_result: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = backend_result.get("parsed", {})
    exact = str(parsed.get("exactness_certified", "false")).lower() == "true"
    minimal = str(parsed.get("minimality_certified", "false")).lower() == "true"
    safe = bool(exact and backend_result.get("certificate_attached"))
    if not safe:
        return unavailable_real_resolution(
            module_schema,
            status="certificate_failed",
            reason="CAS backend ran, but did not return an exactness certificate.",
            attempts=attempts,
        )
    backend = str(parsed.get("backend", backend_result.get("backend")))
    betti_text = str(parsed.get("betti_table", "") or "")
    singular_resolution_text = str(parsed.get("singular_resolution_text", "") or "")
    sage_total_graded = _parse_json_dict(parsed.get("total_graded_betti_json"))
    sage_differentials = _parse_json_list(parsed.get("differentials_json"))
    sage_resolution_text = str(parsed.get("sage_resolution_text", "") or "")
    macaulay2_multigraded = _parse_macaulay2_multigraded_artifacts(parsed, backend=backend) if backend == "Macaulay2" else {}
    singular_determinantal = _parse_singular_determinantal_artifacts(parsed, backend=backend) if backend == "Singular" else {}
    be_diagnostics = _parse_buchsbaum_eisenbud_diagnostics(parsed, backend=backend)
    structured_betti = _parse_ungraded_betti_table(betti_text, backend=backend)
    if sage_total_graded.get("available"):
        sage_total_graded.setdefault("safe_for_multigraded_claims", False)
        sage_total_graded.setdefault("not_multigraded", True)
        structured_betti = _ungraded_from_total_graded_betti(sage_total_graded, backend=backend)
    presentation_shape = _parse_presentation_shape(parsed.get("presentation_shape"))
    unit_entries = _parse_int_or_none(parsed.get("unit_entries"))
    free_resolution_summary = macaulay2_multigraded if macaulay2_multigraded.get("available") else (sage_total_graded if sage_total_graded.get("available") else structured_betti)
    variables = list(module_schema.get("variables", []))
    expects_multigrading = len(variables) > 1
    total_graded_certified = bool(exact and sage_total_graded.get("available"))
    ungraded_certified = bool(exact and structured_betti.get("available") and not total_graded_certified)
    multigraded_certified = bool(
        exact
        and free_resolution_summary.get("available")
        and free_resolution_summary.get("safe_for_multigraded_claims") is True
        and free_resolution_summary.get("not_multigraded") is not True
    )
    safe_total = bool(total_graded_certified)
    safe_multigraded = bool(multigraded_certified)
    safe_real = bool(
        exact
        and free_resolution_summary.get("available")
        and (safe_multigraded or (not expects_multigrading and (safe_total or ungraded_certified)))
    )
    if safe_multigraded:
        render_warning = "CAS returned a certified multigraded free resolution for the requested persistence-module grading."
    elif safe_total:
        render_warning = (
            "CAS returned a certified total-graded free resolution. It is real CAS output, "
            "but it is not a multigraded F2[x_level,x_radius] persistence-module resolution."
        )
    elif ungraded_certified:
        render_warning = (
            "CAS returned a certified ungraded free-resolution summary. It is real CAS output, "
            "but it is not a multigraded persistence-module resolution."
        )
    else:
        render_warning = "CAS exactness was certified, but no renderable free-resolution summary was parsed."
    fitting_ideals = macaulay2_multigraded.get("fitting_ideals", {}) if isinstance(macaulay2_multigraded, dict) else {}
    minors = macaulay2_multigraded.get("minors", {}) if isinstance(macaulay2_multigraded, dict) else {}
    if not fitting_ideals and isinstance(singular_determinantal, dict):
        fitting_ideals = singular_determinantal.get("fitting_ideals", {})
    if not minors and isinstance(singular_determinantal, dict):
        minors = singular_determinantal.get("minors", {})
    ideal_diagnostics = _structured_ideal_diagnostics(
        fitting_ideals,
        minors,
        presentation_shape=presentation_shape,
        backend=backend,
    )
    be_rank_conditions = _buchsbaum_eisenbud_rank_condition_diagnostics(
        free_resolution_summary,
        differentials=(macaulay2_multigraded.get("differentials", sage_differentials) if isinstance(macaulay2_multigraded, dict) else sage_differentials),
        backend=backend,
        exactness_certified=exact,
        minimality_certified=minimal,
    )
    if isinstance(free_resolution_summary, dict) and free_resolution_summary.get("available"):
        free_resolution_summary = dict(free_resolution_summary)
        if ideal_diagnostics.get("available"):
            free_resolution_summary["ideal_diagnostics"] = ideal_diagnostics
        if be_rank_conditions.get("available"):
            free_resolution_summary["buchsbaum_eisenbud_rank_conditions"] = be_rank_conditions
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "status": "certified",
        "coefficient_ring": module_schema["coefficient_ring"],
        "module_schema_version": module_schema["schema_version"],
        "input_sha256": module_schema["input_sha256"],
        "module_summary": _module_summary(module_schema),
        "backend_attempts": attempts,
        "backend_probe": probe_cas_backends(),
        "bemultipliers_probe": probe_bemultipliers(),
        "command_templates": cas_command_templates(module_schema),
        "backend": backend,
        "presentation_shape": presentation_shape,
        "unit_entries": unit_entries,
        "cas_artifacts": {
            "betti_table_text": betti_text,
            "betti_table_ungraded": structured_betti,
            "betti_table_total_graded": sage_total_graded,
            "differentials": macaulay2_multigraded.get("differentials", sage_differentials) if isinstance(macaulay2_multigraded, dict) else sage_differentials,
            "macaulay2_multigraded": macaulay2_multigraded,
            "singular_determinantal": singular_determinantal,
            "fitting_ideals": fitting_ideals,
            "minors": minors,
            "ideal_diagnostics": ideal_diagnostics,
            "buchsbaum_eisenbud_diagnostics": be_diagnostics,
            "buchsbaum_eisenbud_rank_conditions": be_rank_conditions,
            "singular_resolution_text": singular_resolution_text,
            "sage_resolution_text": sage_resolution_text,
            "raw_tagged_output": backend_result.get("tagged_output", ""),
        },
        "free_resolution_summary": free_resolution_summary,
        "certificate_attached": True,
        "real_free_resolution_certified": bool(exact and free_resolution_summary.get("available")),
        "total_graded_resolution_certified": total_graded_certified,
        "ungraded_resolution_certified": ungraded_certified,
        "multigraded_free_resolution_certified": multigraded_certified,
        "exactness_certified": exact,
        "minimality_certified": minimal,
        "safe_to_render_as_real_free_resolution": safe_real,
        "safe_to_render_as_total_graded_resolution": safe_total,
        "safe_to_render_as_multigraded_free_resolution": safe_multigraded,
        "render_warning": render_warning,
    }




def _parse_macaulay2_multigraded_artifacts(parsed: dict[str, Any], *, backend: str) -> dict[str, Any]:
    free_text = str(parsed.get("macaulay2_free_modules", "") or "")
    differentials_text = str(parsed.get("macaulay2_differentials", "") or "")
    fitting_text = str(parsed.get("fitting_ideals", "") or "")
    minors_text = str(parsed.get("minors", "") or "")
    if not free_text:
        return {"available": False, "backend": backend, "reason": "Macaulay2 output did not contain multigraded free-module degree blocks."}
    free_modules: list[dict[str, Any]] = []
    betti_table_rows: list[dict[str, Any]] = []
    betti_by_multidegree: dict[str, dict[str, int]] = {}
    for line in free_text.splitlines():
        line = line.strip()
        if not line or "_degrees=" not in line or not line.startswith("F"):
            continue
        left, right = line.split("_degrees=", 1)
        try:
            homological_degree = int(left[1:])
        except ValueError:
            continue
        degrees = _parse_m2_degree_list(right)
        counts: Counter[tuple[int, ...]] = Counter(tuple(int(v) for v in degree) for degree in degrees)
        row: dict[str, int] = {}
        for degree, rank in sorted(counts.items()):
            key = ",".join(str(v) for v in degree)
            row[key] = int(rank)
            shift = ",".join(str(v) for v in degree)
            display = f"F_{homological_degree} contains S(-{shift})^{int(rank)}"
            row_payload = {
                "homological_degree": homological_degree,
                "multidegree": list(degree),
                "rank": int(rank),
                "display": display,
                "grading": "multigraded_bidegree_shift",
                "multidegree_shifts_available": True,
            }
            free_modules.append(row_payload)
            betti_table_rows.append(
                {
                    "homological_degree": homological_degree,
                    "multidegree": list(degree),
                    "shift_display": f"({shift})",
                    "rank": int(rank),
                    "multiplicity": int(rank),
                    "grading": "multigraded_bidegree_shift",
                    "source": "macaulay2_free_module_degree_block",
                    "not_multigraded": False,
                    "multidegree_shifts_available": True,
                    "safe_for_multigraded_claims": True,
                }
            )
        if row:
            betti_by_multidegree[str(homological_degree)] = row
    differentials = _parse_macaulay2_differentials(differentials_text)
    fitting_ideals = _parse_key_value_lines(fitting_text)
    minors = _parse_key_value_lines(minors_text)
    available = bool(free_modules)
    return {
        "available": available,
        "backend": backend,
        "grading": "multigraded_bidegree_shifts_over_F2_polynomial_ring",
        "free_modules": free_modules,
        "betti_table_rows": betti_table_rows,
        "betti_by_homological_and_multidegree": betti_by_multidegree,
        "differentials": differentials,
        "fitting_ideals": fitting_ideals,
        "minors": minors,
        "not_multigraded": False,
        "safe_for_multigraded_claims": available,
        "interpretation": "Macaulay2 certified a real multigraded free resolution of the displayed cokernel presentation over the requested F2 polynomial ring.",
    }


def _parse_singular_determinantal_artifacts(parsed: dict[str, Any], *, backend: str) -> dict[str, Any]:
    fitting_ideals = _parse_key_value_lines(str(parsed.get("fitting_ideals", "") or ""))
    minors = _parse_key_value_lines(str(parsed.get("minors", "") or ""))
    return {
        "available": bool(fitting_ideals or minors),
        "backend": backend,
        "grading": "determinantal_ideals_from_presentation_matrix_over_F2_polynomial_ring",
        "fitting_ideals": fitting_ideals,
        "minors": minors,
        "interpretation": (
            "Singular computed exact determinantal ideals from the displayed presentation matrix. "
            "For an r-row presentation matrix PM, Fitt_j(coker PM) is I_{r-j}(PM)."
        ),
    }



def _structured_ideal_diagnostics(
    fitting_ideals: dict[str, str],
    minors: dict[str, str],
    *,
    presentation_shape: list[int] | None,
    backend: str,
) -> dict[str, Any]:
    shape = list(presentation_shape) if presentation_shape else None
    row_count = int(shape[0]) if shape and len(shape) >= 1 else None
    fitting_rows: list[dict[str, Any]] = []
    for key, value in sorted((fitting_ideals or {}).items(), key=lambda item: _ideal_sort_key(item[0])):
        if not key.startswith("Fitt"):
            continue
        index = _parse_int_or_none(key[4:])
        determinantal_order = None
        if index is not None and row_count is not None:
            determinantal_order = max(row_count - index, 0)
        fitting_rows.append(
            {
                "name": key,
                "fitting_index": index,
                "determinantal_order": determinantal_order,
                "ideal_text": str(value),
                "source": f"{backend}_fitting_ideal_block",
                "method": "Fitt_j(coker(PM)) = I_{rows-j}(PM) for the displayed presentation matrix PM",
            }
        )
    minor_rows: list[dict[str, Any]] = []
    for key, value in sorted((minors or {}).items(), key=lambda item: _ideal_sort_key(item[0])):
        if not key.startswith("minors_"):
            continue
        order = _parse_int_or_none(key.split("_", 1)[1])
        minor_rows.append(
            {
                "name": key,
                "minor_order": order,
                "ideal_text": str(value),
                "source": f"{backend}_minors_block",
                "method": "determinantal ideal generated by order-k minors of the certified presentation matrix",
            }
        )
    return {
        "available": bool(fitting_rows or minor_rows),
        "backend": backend,
        "presentation_shape": shape,
        "fitting_invariants": fitting_rows,
        "determinantal_minors": minor_rows,
        "paper_method_note": (
            "Fitting invariants and rank strata are represented by determinantal ideals/minors; "
            "Buchsbaum-Eisenbud multiplier and exactness claims still require explicit CAS certificates."
        ),
        "not_a_resolution_certificate_by_itself": True,
    }


def _ideal_sort_key(key: str) -> tuple[str, int, str]:
    digits = "".join(ch for ch in str(key) if ch.isdigit())
    return ("".join(ch for ch in str(key) if not ch.isdigit()), int(digits) if digits else -1, str(key))


def _buchsbaum_eisenbud_rank_condition_diagnostics(
    summary: dict[str, Any],
    *,
    differentials: list[Any],
    backend: str,
    exactness_certified: bool,
    minimality_certified: bool,
) -> dict[str, Any]:
    modules = summary.get("free_modules") if isinstance(summary, dict) and isinstance(summary.get("free_modules"), list) else []
    ranks: dict[int, int] = {}
    for row in modules:
        if not isinstance(row, dict):
            continue
        degree = _parse_int_or_none(row.get("homological_degree"))
        if degree is None:
            continue
        ranks[degree] = ranks.get(degree, 0) + int(row.get("rank", 0) or 0)
    if not ranks:
        return {
            "available": False,
            "backend": backend,
            "reason": "No certified free-module ranks were parsed, so Buchsbaum-Eisenbud rank diagnostics are unavailable.",
        }
    image_rank_estimates: dict[str, int] = {}
    next_image_rank = 0
    for degree in sorted(ranks, reverse=True):
        if degree <= 0:
            continue
        estimate = int(ranks[degree]) - int(next_image_rank)
        image_rank_estimates[f"d{degree}"] = estimate
        next_image_rank = estimate
    differential_shapes: dict[str, list[int]] = {}
    shape_bounds: dict[str, bool] = {}
    for row in differentials or []:
        if not isinstance(row, dict):
            continue
        degree = _parse_int_or_none(row.get("homological_degree"))
        rows = _parse_int_or_none(row.get("rows"))
        cols = _parse_int_or_none(row.get("cols"))
        if degree is None or rows is None or cols is None:
            continue
        key = f"d{degree}"
        differential_shapes[key] = [rows, cols]
        if key in image_rank_estimates:
            rank_est = image_rank_estimates[key]
            shape_bounds[key] = 0 <= rank_est <= min(rows, cols)
    nonnegative = all(value >= 0 for value in image_rank_estimates.values())
    shape_bounds_hold = all(shape_bounds.values()) if shape_bounds else None
    return {
        "available": True,
        "backend": backend,
        "free_module_ranks_by_homological_degree": {str(k): int(v) for k, v in sorted(ranks.items())},
        "image_rank_estimates_by_differential": image_rank_estimates,
        "differential_shapes": differential_shapes,
        "nonnegative_rank_conditions": bool(nonnegative),
        "shape_bounds_hold": shape_bounds_hold,
        "exactness_certified_by_backend": bool(exactness_certified),
        "minimality_certified_by_backend": bool(minimality_certified),
        "is_independent_certificate": False,
        "paper_method_note": (
            "For an exact finite free complex, Buchsbaum-Eisenbud rank conditions require "
            "rank(F_i)=rank(d_i)+rank(d_{i+1}); regular-element/grade conditions and multipliers "
            "are not inferred from these ranks and must come from the CAS diagnostic block."
        ),
    }


def _parse_buchsbaum_eisenbud_diagnostics(parsed: dict[str, Any], *, backend: str) -> dict[str, Any]:
    raw = str(parsed.get("buchsbaum_eisenbud_diagnostics", "") or "")
    values = _parse_key_value_lines(raw)
    if not values:
        return {
            "available": False,
            "backend": backend,
            "multiplier_output_available": False,
            "reason": "CAS output did not include a Buchsbaum-Eisenbud diagnostic block; no multiplier or grade/depth evidence is inferred.",
        }
    multiplier_available = _parse_bool(values.get("multiplier_output_available"))
    return {
        "available": True,
        "backend": backend,
        "exactness_certified": _parse_bool(values.get("exactness_certified")),
        "minimality_certified": _parse_bool(values.get("minimality_certified")),
        "backend_diagnostics_available": _parse_bool(values.get("backend_diagnostics_available"), default=True),
        "multiplier_output_available": multiplier_available,
        "bemultipliers_status": values.get("bemultipliers_status", "unreported"),
        "bemultipliers_repository": values.get("bemultipliers_repository", "https://github.com/amelie-iska/BEMultipliers.git"),
        "bemultipliers_package_path": values.get("bemultipliers_package_path", ""),
        "be_exactness_source": values.get("be_exactness_source", ""),
        "a_multiplier_1_shape": values.get("aMultiplier_1_shape", ""),
        "a_multiplier_1_matrix": values.get("aMultiplier_1_matrix", ""),
        "raw_key_values": values,
        "interpretation": (
            "Buchsbaum-Eisenbud multiplier output is present as explicit CAS output."
            if multiplier_available
            else "No Buchsbaum-Eisenbud multiplier output is present; exactness/Fitting/minor data is not substituted for multiplier evidence."
        ),
    }


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_m2_degree_list(text: str) -> list[list[int]]:
    cleaned = str(text).strip().replace("{", "[").replace("}", "]")
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return []
    if isinstance(parsed, list) and all(isinstance(item, list) for item in parsed):
        out = []
        for item in parsed:
            try:
                out.append([int(v) for v in item])
            except Exception:
                continue
        return out
    return []


def _parse_key_value_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_macaulay2_differentials(text: str) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("d") or "_" not in line or "=" not in line:
            continue
        left, value = line.split("=", 1)
        prefix, field = left.split("_", 1)
        try:
            degree = int(prefix[1:])
        except ValueError:
            continue
        row = grouped.setdefault(degree, {"homological_degree": degree})
        if field == "shape":
            shape = _parse_presentation_shape(value)
            if shape:
                row["rows"], row["cols"] = shape
        elif field in {"source_degrees", "target_degrees"}:
            row[field] = _parse_m2_degree_list(value)
        elif field == "matrix":
            row["matrix_text"] = value.strip()
    return [grouped[key] for key in sorted(grouped)]

def _parse_json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _ungraded_from_total_graded_betti(total_graded: dict[str, Any], *, backend: str) -> dict[str, Any]:
    rows = total_graded.get("betti_by_homological_and_total_degree", {})
    if not isinstance(rows, dict):
        rows = {}
    column_ranks: dict[int, int] = {}
    betti_table_rows: list[dict[str, Any]] = []
    for homological_degree, row in rows.items():
        if not isinstance(row, dict):
            continue
        degree = int(homological_degree)
        column_ranks[degree] = sum(int(rank) for rank in row.values())
        for total_degree, rank in sorted(row.items(), key=lambda item: int(item[0])):
            rank_int = int(rank)
            if rank_int == 0:
                continue
            total_degree_int = int(total_degree)
            betti_table_rows.append(
                {
                    "homological_degree": degree,
                    "total_degree": total_degree_int,
                    "multidegree": [],
                    "shift_display": f"total degree {total_degree_int}",
                    "rank": rank_int,
                    "multiplicity": rank_int,
                    "grading": "total_graded_rank",
                    "source": f"{backend}_total_graded_betti_json",
                    "not_multigraded": True,
                    "multidegree_shifts_available": False,
                    "safe_for_multigraded_claims": False,
                }
            )
    free_modules = [
        {
            "homological_degree": degree,
            "rank": rank,
            "display": f"F_{degree} = S^{rank}" if rank != 1 else f"F_{degree} = S",
            "grading": "total_graded_rank_aggregated",
            "multidegree_shifts_available": False,
        }
        for degree, rank in sorted(column_ranks.items())
        if rank != 0
    ]
    return {
        "available": bool(free_modules),
        "backend": backend,
        "grading": "total_graded_betti_ranks_aggregated_by_homological_degree",
        "homological_column_ranks": [rank for _, rank in sorted(column_ranks.items())],
        "free_modules": free_modules,
        "betti_table_rows": betti_table_rows,
        "total_rank": int(sum(column_ranks.values())),
        "not_multigraded": True,
        "safe_for_multigraded_claims": False,
        "interpretation": "Aggregated from a certified Sage total-graded free resolution. It is real, but it is not a multigraded Betti table.",
    }

def _parse_ungraded_betti_table(text: str, *, backend: str) -> dict[str, Any]:
    """Parse CAS Betti text as ungraded homological ranks only.

    Singular's ``betti(resolution)`` output here certifies ranks in each
    homological degree, but it does not include multidegree shifts. Those shifts
    must come from a graded Macaulay2/Sage computation before the result can be
    rendered as a multigraded minimal free resolution over F2[x,y].
    """
    if backend == "Macaulay2":
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("total:"):
                continue
            values = [int(piece) for piece in stripped.split(":", 1)[1].split() if _is_int_literal(piece)]
            free_modules = [
                {
                    "homological_degree": degree,
                    "rank": int(rank),
                    "display": f"F_{degree} = S^{int(rank)}" if int(rank) != 1 else f"F_{degree} = S",
                    "grading": "ungraded_total_rank_from_macaulay2_total_row",
                    "multidegree_shifts_available": False,
                }
                for degree, rank in enumerate(values)
                if int(rank) != 0
            ]
            return {
                "available": bool(free_modules),
                "backend": backend,
                "grading": "ungraded_total_betti_ranks_from_macaulay2_total_row",
                "homological_column_ranks": values,
                "free_modules": free_modules,
                "betti_table_rows": _ungraded_betti_rows_from_column_ranks(values, backend=backend, source="macaulay2_total_row"),
                "total_rank": int(sum(values)),
                "not_multigraded": True,
                "safe_for_multigraded_claims": False,
                "interpretation": "Macaulay2 total Betti row parsed as ungraded homological ranks. Multigraded claims use the separate Macaulay2 degree-shift blocks.",
            }
    rows: list[list[int]] = []
    for line in str(text or "").splitlines():
        values = [int(piece) for piece in line.replace("|", " ").split() if _is_int_literal(piece)]
        if values:
            rows.append(values)
    if not rows:
        return {
            "available": False,
            "backend": backend,
            "grading": "ungraded",
            "reason": "CAS Betti text contained no integer rank table.",
            "not_multigraded": True,
            "safe_for_multigraded_claims": False,
        }
    width = max(len(row) for row in rows)
    matrix = [row + [0] * (width - len(row)) for row in rows]
    column_ranks = [sum(row[col] for row in matrix) for col in range(width)]
    free_modules = [
        {
            "homological_degree": degree,
            "rank": int(rank),
            "display": f"F_{degree} = S^{int(rank)}" if int(rank) != 1 else f"F_{degree} = S",
            "grading": "ungraded_total_rank",
            "multidegree_shifts_available": False,
        }
        for degree, rank in enumerate(column_ranks)
        if int(rank) != 0
    ]
    betti_table_rows = []
    for matrix_row, row in enumerate(matrix):
        for degree, rank in enumerate(row):
            rank_int = int(rank)
            if rank_int == 0:
                continue
            betti_table_rows.append(
                {
                    "homological_degree": degree,
                    "matrix_row": matrix_row,
                    "multidegree": [],
                    "shift_display": f"ungraded row {matrix_row}",
                    "rank": rank_int,
                    "multiplicity": rank_int,
                    "grading": "ungraded_total_rank",
                    "source": f"{backend}_betti_matrix",
                    "not_multigraded": True,
                    "multidegree_shifts_available": False,
                    "safe_for_multigraded_claims": False,
                }
            )
    return {
        "available": bool(free_modules),
        "backend": backend,
        "grading": "ungraded_total_betti_ranks",
        "matrix": matrix,
        "homological_column_ranks": column_ranks,
        "free_modules": free_modules,
        "betti_table_rows": betti_table_rows,
        "reason": "CAS Betti table had no nonzero free-module ranks after parsing." if not free_modules else "",
        "total_rank": int(sum(column_ranks)),
        "not_multigraded": True,
        "safe_for_multigraded_claims": False,
        "interpretation": (
            "Certified CAS output parsed as ungraded Betti ranks by homological degree. "
            "It is figure-ready as an ungraded rank table, but it does not certify multidegree shifts, "
            "Fitting ideals, minors, or Buchsbaum-Eisenbud diagnostics."
        ),
    }


def _ungraded_betti_rows_from_column_ranks(values: list[int], *, backend: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for degree, rank in enumerate(values):
        rank_int = int(rank)
        if rank_int == 0:
            continue
        rows.append(
            {
                "homological_degree": degree,
                "multidegree": [],
                "shift_display": "ungraded",
                "rank": rank_int,
                "multiplicity": rank_int,
                "grading": "ungraded_total_rank",
                "source": f"{backend}_{source}",
                "not_multigraded": True,
                "multidegree_shifts_available": False,
                "safe_for_multigraded_claims": False,
            }
        )
    return rows


def _is_int_literal(value: str) -> bool:
    value = value.strip()
    return bool(value) and (value.isdigit() or (value.startswith("-") and value[1:].isdigit()))


def _parse_presentation_shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if "x" not in text:
        return None
    left, right = text.split("x", 1)
    try:
        return [int(left), int(right)]
    except ValueError:
        return None


def _parse_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None



def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _falsey_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off", "disabled", "disable"}


def _cas_resolution_disabled_reason() -> str | None:
    explicit_disable = os.environ.get("TROPICALGT_DISABLE_CAS_FREE_RESOLUTION")
    if _truthy_env(explicit_disable):
        return "CAS free-resolution execution disabled by TROPICALGT_DISABLE_CAS_FREE_RESOLUTION"
    mode = os.environ.get("TROPICALGT_CAS_FREE_RESOLUTION")
    if _falsey_env(mode):
        return "CAS free-resolution execution disabled by TROPICALGT_CAS_FREE_RESOLUTION"
    return None


def _cache_context(module_schema: dict[str, Any], *, use_cache: bool | None) -> dict[str, Any]:
    cache_env = os.environ.get("TROPICALGT_CAS_FREE_RESOLUTION_CACHE")
    enabled = use_cache is not False and not _falsey_env(cache_env)
    backend_probe = probe_cas_backends()
    key_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "adapter_cache_version": ADAPTER_CACHE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "module_schema_version": module_schema.get("schema_version"),
        "module_input_sha256": module_schema.get("input_sha256"),
        "module_schema": module_schema,
        "backend_probe": backend_probe,
    }
    key = sha256_json(key_payload)
    cache_dir = _cache_dir()
    path = cache_dir / key[:2] / f"{key}.json"
    return {
        "enabled": enabled,
        "key": key,
        "path": path,
        "backend_probe": backend_probe,
        "module_input_sha256": module_schema.get("input_sha256"),
    }


def _cache_dir() -> Path:
    override = os.environ.get("TROPICALGT_CAS_FREE_RESOLUTION_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return root / "tropicalgt" / "cas_free_resolution"


def _load_cached_result(cache_context: dict[str, Any]) -> dict[str, Any] | None:
    if not cache_context.get("enabled"):
        return None
    path = cache_context.get("path")
    if not isinstance(path, Path) or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if payload.get("key") != cache_context.get("key"):
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
    except Exception:
        return None
    result = dict(result)
    result["cache"] = {
        "enabled": True,
        "hit": True,
        "written": False,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "adapter_cache_version": ADAPTER_CACHE_VERSION,
        "key": cache_context.get("key"),
        "path": str(path),
    }
    return result


def _cache_and_annotate(result: dict[str, Any], cache_context: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    enabled = bool(cache_context.get("enabled"))
    path = cache_context.get("path")
    written = False
    write_error = None
    if enabled and isinstance(path, Path) and _result_cacheable(result):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "adapter_cache_version": ADAPTER_CACHE_VERSION,
                "key": cache_context.get("key"),
                "module_input_sha256": cache_context.get("module_input_sha256"),
                "backend_probe": cache_context.get("backend_probe"),
                "result": {key: value for key, value in result.items() if key != "cache"},
            }
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(cache_payload, sort_keys=True, indent=2), encoding="utf-8")
            tmp_path.replace(path)
            written = True
        except Exception as exc:
            write_error = f"{type(exc).__name__}: {exc}"
    result["cache"] = {
        "enabled": enabled,
        "hit": False,
        "written": written,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "adapter_cache_version": ADAPTER_CACHE_VERSION,
        "key": cache_context.get("key"),
        "path": str(path) if isinstance(path, Path) else None,
        **({"write_error": write_error} if write_error else {}),
    }
    return result


def _result_cacheable(result: dict[str, Any]) -> bool:
    status = str(result.get("status", ""))
    if status == "certified":
        return bool(result.get("certificate_attached") and result.get("exactness_certified"))
    return status in {"backend_not_installed", "certificate_failed", "unsupported_ring", "invalid_grading", "complexity_guard"}

def _run_tagged_cas_script(
    *,
    name: str,
    executable: str,
    script: str,
    suffix: str,
    timeout_s: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tropicalgt_cas_") as tmpdir:
        path = os.path.join(tmpdir, "probe" + suffix)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(script)
        command = [executable, path]
        if name == "M2":
            command = [executable, "--script", path]
        elif name == "Singular":
            command = [executable, "-q", path]
        try:
            proc = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as exc:
            return {"available": False, "attempt": {"backend": name, "status": "backend_error", "error": str(exc), "executable": executable}}
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stdout, stderr = _terminate_cas_process_group(proc)
            return {
                "available": False,
                "attempt": {
                    "backend": name,
                    "status": "timeout",
                    "executable": executable,
                    "timeout_s": timeout_s,
                    "pid": proc.pid,
                    "stdout_tail": stdout[-800:],
                    "stderr_tail": stderr[-800:],
                },
            }
    parsed = _parse_tagged_output(stdout)
    ok = proc.returncode == 0 and parsed is not None
    return {
        "available": ok,
        "backend": name,
        "parsed": parsed or {},
        "tagged_output": parsed.get("_raw", "") if parsed else "",
        "certificate_attached": ok,
        "attempt": {
            "backend": name,
            "status": "ran" if ok else "parse_error",
            "returncode": proc.returncode,
            "executable": executable,
            "stderr_tail": stderr[-800:],
        },
    }


def _terminate_cas_process_group(proc: subprocess.Popen[str]) -> tuple[str, str]:
    stdout = ""
    stderr = ""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            proc.kill()
        stdout, stderr = proc.communicate()
    return stdout or "", stderr or ""

def _parse_tagged_output(stdout: str) -> dict[str, str] | None:
    start = "TROPICALGT_RESOLUTION_BEGIN"
    end = "TROPICALGT_RESOLUTION_END"
    if start not in stdout or end not in stdout:
        return None
    block = stdout.split(start, 1)[1].split(end, 1)[0]
    parsed: dict[str, str] = {"_raw": block.strip()}
    for line in block.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
    for key in ("betti_table", "singular_resolution_text", "macaulay2_free_modules", "macaulay2_differentials", "fitting_ideals", "minors", "buchsbaum_eisenbud_diagnostics"):
        nested = _extract_tagged_block(block, f"{key}_begin", f"{key}_end")
        if nested is not None:
            parsed[key] = nested
    return parsed


def _extract_tagged_block(text: str, begin: str, end: str) -> str | None:
    if begin not in text or end not in text:
        return None
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


def _candidate_executable(name: str) -> str | None:
    env_name = f"TROPICALGT_{name.upper()}_BIN"
    if os.environ.get(env_name):
        candidate = os.environ[env_name]
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    candidates = {
        "Singular": [home / "miniconda3" / "envs" / "tropicalgt-cas" / "bin" / "Singular"],
        "M2": [home / "macaulay2" / "bin" / "M2"],
        "sage": [
            home / "miniconda3" / "envs" / "tropicalgt-sage" / "bin" / "python",
            home / "miniconda3" / "envs" / "tropicalgt-sage" / "bin" / "sage",
            home / "miniconda3" / "envs" / "tropicalgt-cas" / "bin" / "python",
            home / "miniconda3" / "envs" / "tropicalgt-cas" / "bin" / "sage",
        ],
    }
    for path in candidates.get(name, []):
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def _iter_boundary_entries(boundary_monomials: Any) -> list[dict[str, Any]]:
    if isinstance(boundary_monomials, dict):
        out: list[dict[str, Any]] = []
        for map_name, rows in sorted(boundary_monomials.items()):
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    out.append({**row, "boundary_map": str(map_name)})
        return out
    if isinstance(boundary_monomials, list):
        return [row for row in boundary_monomials if isinstance(row, dict)]
    return []

def _presentation_from_d1(generators: list[dict[str, Any]], boundaries: list[dict[str, Any]], variables: list[str]) -> dict[str, Any]:
    rows = [g for g in generators if int(g["homological_degree"]) == 0]
    cols = [g for g in generators if int(g["homological_degree"]) == 1]
    row_index = {g["generator_id"]: idx for idx, g in enumerate(rows)}
    col_index = {g["generator_id"]: idx for idx, g in enumerate(cols)}
    entries = []
    for entry in boundaries:
        row = row_index.get(entry.get("target_generator_id"))
        col = col_index.get(entry.get("source_generator_id"))
        if row is None or col is None:
            continue
        entries.append({"row": row, "col": col, "exponent": entry["monomial_exponent"], "monomial": entry["monomial"]})
    return {
        "description": "degree-one boundary matrix used as a finite presentation over the coefficient ring",
        "variables": variables,
        "rows": len(rows),
        "cols": len(cols),
        "row_generators": [g["generator_id"] for g in rows],
        "col_generators": [g["generator_id"] for g in cols],
        "entries": entries,
    }


def _module_summary(module_schema: dict[str, Any]) -> dict[str, Any]:
    by_degree: dict[str, int] = {}
    for generator in module_schema.get("generators", []):
        degree = str(generator.get("homological_degree", 0))
        by_degree[degree] = by_degree.get(degree, 0) + 1
    return {
        "coefficient_ring": module_schema.get("coefficient_ring"),
        "variables": list(module_schema.get("variables", [])),
        "generators": len(module_schema.get("generators", [])),
        "boundary_monomials": len(module_schema.get("boundary_monomials", [])),
        "generators_by_homological_degree": by_degree,
        "presentation_shape": [
            module_schema.get("presentation_matrix", {}).get("rows", 0),
            module_schema.get("presentation_matrix", {}).get("cols", 0),
        ],
    }


def _nonnegative_tuple(value: Any, length: int, label: str) -> list[int]:
    items = list(value or [])
    if len(items) < length:
        items.extend([0] * (length - len(items)))
    items = items[:length]
    out = [int(v) for v in items]
    if any(v < 0 for v in out):
        raise ValueError(f"{label} must be nonnegative: {out}")
    return out


def _safe_id(prefix: str, parts: list[Any]) -> str:
    digest = sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _monomial_from_exponent(exponent: list[int], variables: list[str]) -> str:
    factors = []
    for variable, power in zip(variables, exponent):
        if power == 0:
            continue
        factors.append(variable if power == 1 else f"{variable}^{power}")
    return "*".join(factors) if factors else "1"


def _m2_monomial(exponent: list[int], variables: list[str]) -> str:
    factors = []
    for variable, power in zip(variables, exponent):
        if power == 0:
            continue
        factors.append(variable if power == 1 else f"{variable}^{power}")
    return "*".join(factors) if factors else "1_R"


def _presentation_generator_degrees(module_schema: dict[str, Any], generator_ids: list[Any]) -> list[list[int]]:
    variables = list(module_schema.get("variables", []))
    degree_by_id = {
        str(generator.get("generator_id")): [int(v) for v in generator.get("multidegree", [])]
        for generator in module_schema.get("generators", [])
    }
    degrees: list[list[int]] = []
    for generator_id in generator_ids:
        key = str(generator_id)
        if key not in degree_by_id:
            raise ValueError(f"presentation generator {key!r} is missing a multidegree")
        degree = list(degree_by_id[key])[: len(variables)]
        if len(degree) < len(variables):
            degree.extend([0] * (len(variables) - len(degree)))
        degrees.append(degree)
    return degrees


def _macaulay2_free_module_shift_spec(degrees: list[list[int]]) -> str:
    return "{" + ",".join("{" + ",".join(str(-int(value)) for value in degree) + "}" for degree in degrees) + "}"


def _singular_monomial(exponent: list[int], variables: list[str]) -> str:
    return _monomial_from_exponent(exponent, variables)


def _singular_module_literal(entries: list[dict[str, Any]], *, rows: int, cols: int, variables: list[str]) -> str:
    terms: dict[tuple[int, int], Counter[str]] = {}
    for entry in entries:
        row = int(entry["row"])
        col = int(entry["col"])
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        monomial = _singular_monomial(entry["exponent"], variables)
        terms.setdefault((row, col), Counter())[monomial] += 1
    column_literals: list[str] = []
    for col in range(cols):
        row_literals: list[str] = []
        for row in range(rows):
            cell_terms = [term for term, count in sorted(terms.get((row, col), {}).items()) if count % 2]
            row_literals.append("+".join(cell_terms) if cell_terms else "0")
        column_literals.append("[" + ",".join(row_literals) + "]")
    return ",".join(column_literals) if column_literals else "0"


def _singular_matrix_literal(entries: list[dict[str, Any]], *, rows: int, cols: int, variables: list[str]) -> str:
    terms: dict[tuple[int, int], Counter[str]] = {}
    for entry in entries:
        row = int(entry["row"])
        col = int(entry["col"])
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        monomial = _singular_monomial(entry["exponent"], variables)
        terms.setdefault((row, col), Counter())[monomial] += 1
    cells: list[str] = []
    for row in range(rows):
        for col in range(cols):
            cell_terms = [term for term, count in sorted(terms.get((row, col), {}).items()) if count % 2]
            cells.append("+".join(cell_terms) if cell_terms else "0")
    return ",".join(cells) if cells else "0"


def _backend_version(executable: str | None) -> str | None:
    if not executable:
        return None
    try:
        proc = subprocess.run([executable, "--version"], text=True, capture_output=True, timeout=5, check=False)
    except Exception:
        return None
    text = (proc.stdout or proc.stderr or "").strip().splitlines()
    return text[0][:200] if text else None
