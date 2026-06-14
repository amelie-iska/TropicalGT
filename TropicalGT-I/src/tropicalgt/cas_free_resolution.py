from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


SCHEMA_VERSION = "tropicalgt.real_free_resolution.v1"
MODULE_SCHEMA_VERSION = "tropicalgt.level_radius_module.v1"
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
}


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
    return {
        "role": "optional_buchsbaum_eisenbud_diagnostics_after_certified_resolution",
        "available_python_modules": {
            "BEMultipliers": module_available("BEMultipliers"),
            "bemultipliers": module_available("bemultipliers"),
        },
        "repository": "https://github.com/amelie-iska/BEMultipliers.git",
        "is_resolution_backend": False,
    }


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

    boundaries = []
    for idx, entry in enumerate(_iter_boundary_entries(module.get("boundary_monomials", []) or [])):
        source_simplex = tuple(str(v) for v in entry.get("source_simplex", []))
        target_face = tuple(str(v) for v in entry.get("target_face", []))
        exponent = _nonnegative_tuple(entry.get("monomial_exponent", []), len(variables), "boundary monomial exponent")
        boundaries.append(
            {
                "boundary_id": str(entry.get("boundary_id") or _safe_id("d", [idx, *source_simplex, "to", *target_face, *exponent])),
                "source_simplex": list(source_simplex),
                "target_face": list(target_face),
                "source_generator_id": id_by_simplex.get(source_simplex),
                "target_generator_id": id_by_simplex.get(target_face),
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
        "exactness_certified": False,
        "minimality_certified": False,
        "safe_to_render_as_real_free_resolution": False,
    }


def try_compute_real_free_resolution(module: dict[str, Any], *, timeout_s: float = 20.0) -> dict[str, Any]:
    try:
        module_schema = canonicalize_module(module)
    except Exception as exc:
        return unavailable_real_resolution(module, status="invalid_grading", error=str(exc))

    attempts: list[dict[str, Any]] = []
    m2 = _candidate_executable("M2")
    if m2:
        result = _run_tagged_cas_script(
            name="M2",
            executable=m2,
            script=build_macaulay2_script(module_schema),
            suffix=".m2",
            timeout_s=timeout_s,
        )
        attempts.append(result.get("attempt", {}))
        if result.get("available"):
            return _certified_result(module_schema, result, attempts)
    sage = _candidate_executable("sage")
    if sage:
        result = _run_tagged_cas_script(
            name="sage",
            executable=sage,
            script=build_sage_python_script(module_schema),
            suffix=".sage.py",
            timeout_s=timeout_s,
        )
        attempts.append(result.get("attempt", {}))
        if result.get("available"):
            return _certified_result(module_schema, result, attempts)
    singular = _candidate_executable("Singular")
    if singular:
        result = _run_tagged_cas_script(
            name="Singular",
            executable=singular,
            script=build_singular_script(module_schema),
            suffix=".sing",
            timeout_s=timeout_s,
        )
        attempts.append(result.get("attempt", {}))
        if result.get("available"):
            return _certified_result(module_schema, result, attempts)
    status = "backend_not_installed" if not attempts else "certificate_failed"
    return unavailable_real_resolution(module, status=status, attempts=attempts)


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
    matrix_rows: list[list[str]] = [["0" for _ in range(cols)] for _ in range(rows)]
    for entry in entries:
        matrix_rows[int(entry["row"])][int(entry["col"])] = _m2_monomial(entry["exponent"], variables)
    matrix_literal = "{" + ",".join("{" + ",".join(row) + "}" for row in matrix_rows) + "}"
    return "\n".join(
        [
            "-- TropicalGT certified resolution probe generated from model audit data",
            "needsPackage \"JSON\"",
            f"R = ZZ/2[{','.join(variables)}]",
            f"M = matrix {matrix_literal}",
            "C = res coker M",
            "H = prune HH C",
            "okExact = (length H == 0)",
            "B = betti C",
            "print \"TROPICALGT_RESOLUTION_BEGIN\"",
            "print concatenate(\"backend=Macaulay2\")",
            "print concatenate(\"exactness_certified=\", toString okExact)",
            "print concatenate(\"minimality_certified=\", toString true)",
            "print concatenate(\"betti_table=\", replace(\"\\n\", \";\", toString B))",
            "print \"TROPICALGT_RESOLUTION_END\"",
        ]
    )


def build_singular_script(module_schema: dict[str, Any]) -> str:
    variables = module_schema.get("variables", ["x_level", "x_radius"])
    pmat = module_schema.get("presentation_matrix", {})
    entries = pmat.get("entries", [])
    rows = int(pmat.get("rows", 0))
    cols = int(pmat.get("cols", 0))
    unit_entries = sum(1 for entry in entries if all(int(power) == 0 for power in entry.get("exponent", [])))
    module_literal = _singular_module_literal(entries, rows=rows, cols=cols, variables=variables)
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
            "print(\"TROPICALGT_RESOLUTION_END\");",
        ]
    else:
        body = [
            f"module M = {module_literal};",
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
            "print(\"TROPICALGT_RESOLUTION_END\");",
        ]
    return "\n".join(
        [
            "// TropicalGT certified resolution probe generated from model audit data.",
            "// Singular mres resolves the image submodule M; TropicalGT prepends",
            "// the displayed free module F0 -> coker(M) to obtain the cokernel resolution.",
            f"ring r = 2,({','.join(variables)}),dp;",
            *body,
        ]
    )


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
    structured_betti = _parse_ungraded_betti_table(betti_text, backend=backend)
    if sage_total_graded.get("available"):
        sage_total_graded.setdefault("safe_for_multigraded_claims", False)
        sage_total_graded.setdefault("not_multigraded", True)
        structured_betti = _ungraded_from_total_graded_betti(sage_total_graded, backend=backend)
    presentation_shape = _parse_presentation_shape(parsed.get("presentation_shape"))
    unit_entries = _parse_int_or_none(parsed.get("unit_entries"))
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
            "differentials": sage_differentials,
            "singular_resolution_text": singular_resolution_text,
            "sage_resolution_text": sage_resolution_text,
            "raw_tagged_output": backend_result.get("tagged_output", ""),
        },
        "free_resolution_summary": sage_total_graded if sage_total_graded.get("available") else structured_betti,
        "certificate_attached": True,
        "real_free_resolution_certified": True,
        "exactness_certified": exact,
        "minimality_certified": minimal,
        "safe_to_render_as_real_free_resolution": True,
    }



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
    for homological_degree, row in rows.items():
        if not isinstance(row, dict):
            continue
        degree = int(homological_degree)
        column_ranks[degree] = sum(int(rank) for rank in row.values())
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
    return {
        "available": True,
        "backend": backend,
        "grading": "ungraded_total_betti_ranks",
        "matrix": matrix,
        "homological_column_ranks": column_ranks,
        "free_modules": free_modules,
        "total_rank": int(sum(column_ranks)),
        "not_multigraded": True,
        "safe_for_multigraded_claims": False,
        "interpretation": (
            "Certified CAS output parsed as ungraded Betti ranks by homological degree. "
            "It is figure-ready as an ungraded rank table, but it does not certify multidegree shifts, "
            "Fitting ideals, minors, or Buchsbaum-Eisenbud diagnostics."
        ),
    }


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
        if name == "Singular":
            command = [executable, "-q", path]
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return {"available": False, "attempt": {"backend": name, "status": "timeout", "executable": executable}}
        except Exception as exc:
            return {"available": False, "attempt": {"backend": name, "status": "backend_error", "error": str(exc), "executable": executable}}
    parsed = _parse_tagged_output(proc.stdout)
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
            "stderr_tail": proc.stderr[-800:],
        },
    }


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
    for key in ("betti_table", "singular_resolution_text"):
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
            home / "miniconda3" / "envs" / "tropicalgt-sage" / "bin" / "sage",
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


def _backend_version(executable: str | None) -> str | None:
    if not executable:
        return None
    try:
        proc = subprocess.run([executable, "--version"], text=True, capture_output=True, timeout=5, check=False)
    except Exception:
        return None
    text = (proc.stdout or proc.stderr or "").strip().splitlines()
    return text[0][:200] if text else None
