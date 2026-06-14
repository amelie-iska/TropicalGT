from __future__ import annotations

import json
import os
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
        executable = shutil.which(name)
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
        "preferred_order": ["M2", "Singular", "sage"],
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
    m2 = shutil.which("M2")
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
    singular = shutil.which("Singular")
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
    sage = shutil.which("sage")
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
    assignments = ["module M;"]
    for entry in entries:
        row = int(entry["row"]) + 1
        col = int(entry["col"]) + 1
        monomial = _singular_monomial(entry["exponent"], variables)
        assignments.append(f"M[{row},{col}]={monomial};")
    return "\n".join(
        [
            "// TropicalGT certified resolution probe generated from model audit data",
            f"ring r = 2,({','.join(variables)}),dp;",
            *assignments,
            "resolution R = mres(M,0);",
            "print(\"TROPICALGT_RESOLUTION_BEGIN\");",
            "print(\"backend=Singular\");",
            "print(\"exactness_certified=false\");",
            "print(\"minimality_certified=false\");",
            f"print(\"presentation_shape={rows}x{cols}\");",
            "print(\"TROPICALGT_RESOLUTION_END\");",
        ]
    )


def build_sage_python_script(module_schema: dict[str, Any]) -> str:
    payload = canonical_json(module_schema)
    return "\n".join(
        [
            "# TropicalGT Sage bridge stub. It returns no certificate until a Sage free-resolution backend is wired.",
            "import json",
            f"module_schema = json.loads({payload!r})",
            "print('TROPICALGT_RESOLUTION_BEGIN')",
            "print('backend=Sage')",
            "print('exactness_certified=false')",
            "print('minimality_certified=false')",
            "print('TROPICALGT_RESOLUTION_END')",
        ]
    )


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
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "status": "certified",
        "coefficient_ring": module_schema["coefficient_ring"],
        "module_schema_version": module_schema["schema_version"],
        "input_sha256": module_schema["input_sha256"],
        "module_summary": _module_summary(module_schema),
        "backend_attempts": attempts,
        "backend": parsed.get("backend", backend_result.get("backend")),
        "cas_artifacts": {
            "betti_table_text": parsed.get("betti_table", ""),
            "raw_tagged_output": backend_result.get("tagged_output", ""),
        },
        "certificate_attached": True,
        "real_free_resolution_certified": True,
        "exactness_certified": exact,
        "minimality_certified": minimal,
        "safe_to_render_as_real_free_resolution": True,
    }


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
    return parsed



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


def _backend_version(executable: str | None) -> str | None:
    if not executable:
        return None
    try:
        proc = subprocess.run([executable, "--version"], text=True, capture_output=True, timeout=5, check=False)
    except Exception:
        return None
    text = (proc.stdout or proc.stderr or "").strip().splitlines()
    return text[0][:200] if text else None
