from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import combinations, permutations, product
from math import comb, isinf
from typing import Any, Iterable
import importlib.util
import shutil
import warnings

import numpy as np


def compute_topological_algebra_report(
    filtered_object: dict[str, Any],
    audit_level: str = "full",
    ph_backend: str = "auto",
    max_simplices: int = 1024,
    max_hochster_vertices: int = 8,
    max_nonfaces: int = 256,
    max_syzygy_basis: int = 16,
    max_multiparameter_grid: int = 6,
    max_rank_invariant_pairs: int = 64,
) -> dict[str, Any]:
    """Compute finite topology and commutative-algebra diagnostics.

    The report is deliberately exact for the bounded finite complex it receives:
    boundary ranks and syzygies are over F2, persistence is computed from a
    filtration-preserving simplex tree when Gudhi is available, and Hochster
    Betti data is evaluated on a bounded induced subcomplex.  Taylor data is
    reported as the standard nonminimal upper-bound resolution of the displayed
    Stanley-Reisner generators; it is not labeled as a minimal free resolution.
    """

    level = (audit_level or "none").lower()
    if level == "none":
        return {"enabled": False, "audit_level": "none"}

    closed = _closed_complex(filtered_object, max_simplices=max_simplices)
    by_dim = _simplices_by_dim(closed["simplices"])
    boundaries = _boundary_reports(by_dim, max_syzygy_basis=max_syzygy_basis)
    homology = _homology_report(by_dim, boundaries)
    persistence_module = _persistence_module_report(closed["simplices"])
    two_parameter = _two_parameter_module_report(
        closed["simplices"],
        max_grid=max_multiparameter_grid,
        max_pairs=max_rank_invariant_pairs,
    )
    multiparameter = _multiparameter_module_report(
        closed["simplices"],
        max_grid=max_multiparameter_grid,
        max_pairs=max_rank_invariant_pairs,
    )
    graph_metrics = _graph_metrics(filtered_object)
    persistence = _persistent_homology_report(closed["simplices"], ph_backend)
    persistence_representations = _persistence_representations_report(persistence)

    report: dict[str, Any] = {
        "enabled": True,
        "audit_level": level,
        "input_summary": filtered_object.get("summary", {}),
        "closure": closed["summary"],
        "chain_complex": {
            "field": "F2",
            "chain_group_ranks": {str(dim): len(rows) for dim, rows in sorted(by_dim.items())},
            "boundary_maps": boundaries["maps"],
            "homology": homology,
            "euler_characteristic": _euler_characteristic(by_dim),
        },
        "persistence": persistence,
        "persistence_representations": persistence_representations,
        "persistence_module": persistence_module,
        "two_parameter_persistence": two_parameter,
        "multiparameter_persistence": multiparameter,
        "graph_metrics": graph_metrics,
        "external_backends": {"multipers": _multipers_backend_status()},
        "derived_equivalence_signature": _derived_signature(homology, persistence, persistence_module, multiparameter),
    }

    if level in {"algebra", "full"}:
        sr = _stanley_reisner_report(closed["simplices"], max_nonfaces=max_nonfaces)
        hochster = _hochster_report(
            closed["simplices"],
            max_vertices=max_hochster_vertices,
            max_syzygy_basis=max_syzygy_basis,
        )
        dgca = _dgca_report(by_dim, boundaries)
        two_parameter_chain = _multigraded_free_resolution_report(two_parameter)
        multiparameter_chain = _multigraded_free_resolution_report(multiparameter)
        report["commutative_algebra"] = {
            "multigraded_hilbert_poincare": _hilbert_poincare_report(closed["simplices"]),
            "stanley_reisner": sr,
            "hochster_betti": hochster,
            "taylor_resolution_upper_bound": _taylor_resolution_report(sr),
            "boundary_syzygies": boundaries["syzygies"],
            "dg_commutative_algebra": dgca,
            "two_parameter_chain_presentation_diagnostics": two_parameter_chain,
            "multiparameter_chain_presentation_diagnostics": multiparameter_chain,
            "two_parameter_free_resolution": {**two_parameter_chain, "deprecated_alias_for": "two_parameter_chain_presentation_diagnostics"},
            "multiparameter_free_resolution_proxy": {**multiparameter_chain, "deprecated_alias_for": "multiparameter_chain_presentation_diagnostics"},
            "notes": [
                "Boundary syzygies are F2 nullspaces of displayed boundary maps.",
                "Taylor ranks are nonminimal upper bounds for the displayed Stanley-Reisner generators.",
                "Hochster Betti data is exact on the bounded vertex subset named in the report.",
                "Two-parameter chain modules and monomial boundary labels are exact for the displayed finite bifiltered complex over F2[x_filtration,x_dimension].",
                "Three-parameter chain modules and monomial boundary labels are exact for the displayed finite multi-filtered complex over F2[x_filtration,x_dimension,x_position].",
            ],
        }
    return report


def summarize_algebra_reports(reports: list[dict[str, Any]], prefix: str = "algebra_") -> dict[str, float]:
    enabled = [report for report in reports if report.get("enabled")]
    if not enabled:
        return {}
    out: dict[str, float] = {f"{prefix}reports": float(len(enabled))}
    for dim in range(4):
        vals = [float(report["chain_complex"]["homology"]["betti"].get(str(dim), 0.0)) for report in enabled]
        out[f"{prefix}betti_{dim}_mean"] = float(np.mean(vals)) if vals else 0.0
    out[f"{prefix}euler_mean"] = float(np.mean([report["chain_complex"]["euler_characteristic"] for report in enabled]))
    out[f"{prefix}persistence_intervals_mean"] = float(np.mean([len(report.get("persistence", {}).get("intervals", [])) for report in enabled]))
    out[f"{prefix}filtration_thresholds_mean"] = float(np.mean([len(report.get("persistence_module", {}).get("thresholds", [])) for report in enabled]))
    landscape_norms = [
        float(report.get("persistence_representations", {}).get("summary", {}).get("landscape_l2_norm", 0.0))
        for report in enabled
    ]
    out[f"{prefix}persistence_landscape_l2_mean"] = float(np.mean(landscape_norms)) if landscape_norms else 0.0
    entropy_vals = [
        float(report.get("persistence_representations", {}).get("summary", {}).get("entropy_scalar_sum", 0.0))
        for report in enabled
    ]
    out[f"{prefix}persistence_entropy_sum_mean"] = float(np.mean(entropy_vals)) if entropy_vals else 0.0
    topvec_norms = [
        float(report.get("persistence_representations", {}).get("summary", {}).get("topological_vector_l2_norm", 0.0))
        for report in enabled
    ]
    out[f"{prefix}topological_vector_l2_mean"] = float(np.mean(topvec_norms)) if topvec_norms else 0.0
    out[f"{prefix}two_parameter_grid_points_mean"] = float(
        np.mean([len(report.get("two_parameter_persistence", {}).get("fiber_rank_profile", [])) for report in enabled])
    )
    out[f"{prefix}two_parameter_h0_rank_samples_mean"] = float(
        np.mean([len(report.get("two_parameter_persistence", {}).get("rank_invariant_samples", [])) for report in enabled])
    )
    out[f"{prefix}multiparameter_grid_points_mean"] = float(
        np.mean([len(report.get("multiparameter_persistence", {}).get("fiber_rank_profile", [])) for report in enabled])
    )
    sr_counts = [
        report.get("commutative_algebra", {}).get("stanley_reisner", {}).get("degree_two_generator_count", 0)
        for report in enabled
    ]
    out[f"{prefix}sr_generators_mean"] = float(np.mean(sr_counts)) if sr_counts else 0.0
    hochster_counts = [
        report.get("commutative_algebra", {}).get("hochster_betti", {}).get("nonzero_multigrades", 0)
        for report in enabled
    ]
    out[f"{prefix}hochster_nonzero_mean"] = float(np.mean(hochster_counts)) if hochster_counts else 0.0
    return out


def compute_persistence_representations_from_intervals(
    intervals: list[dict[str, Any]],
    *,
    max_dimension: int = 2,
    landscape_layers: int = 4,
    resolution: int = 96,
    image_resolution: int = 24,
    topological_vector_threshold: int = 24,
) -> dict[str, Any]:
    """Vectorize persistence intervals with GUDHI representation methods.

    This is the fast train/eval/inference path for topology features.  It
    accepts the interval JSON emitted by the persistence backends and returns
    cached NumPy/scikit-learn representation vectors suitable for metrics,
    memory retrieval, W&B logging, and optional visualization.
    """

    return _persistence_representations_report(
        {"backend": "intervals", "available": True, "intervals": intervals},
        max_dimension=max_dimension,
        landscape_layers=landscape_layers,
        resolution=resolution,
        image_resolution=image_resolution,
        topological_vector_threshold=topological_vector_threshold,
    )


def compute_level_radius_bifiltration_report(
    growth: list[dict[str, Any]] | None,
    *,
    object_key: str = "filtered_simplicial_object",
    max_simplices: int = 1024,
    max_rank_invariant_pairs: int = 64,
) -> dict[str, Any]:
    """Compute the explicit trajectory bifiltration as a finite k[x,y]-module.

    The variables are x_level and x_radius.  The grade (i, j) contains the
    subcomplex K_{level <= i, radius <= r_j}; all chain groups and boundary
    maps are over F2 for the bounded finite complex.  This is the object used
    by the interactive two-parameter persistence plots.
    """

    rows = [row for row in (growth or []) if isinstance(row, dict)]
    if not rows:
        return {
            "available": False,
            "num_parameters": 2,
            "parameters": _level_radius_parameters(),
            "coefficient_ring": "F2[x_level,x_radius]",
            "fiber_rank_profile": [],
            "rank_invariant_samples": [],
            "reason": "trajectory growth rows unavailable",
        }

    closed_by_level: dict[int, list[dict[str, Any]]] = {}
    all_thresholds: set[float] = {0.0}
    for idx, item in enumerate(rows):
        level = int(item.get("level", idx) or 0)
        obj = item.get(object_key) if isinstance(item.get(object_key), dict) else {}
        closed = _closed_complex(obj, max_simplices=max_simplices)["simplices"]
        closed_by_level[level] = closed
        all_thresholds.update(float(row.get("filtration", 0.0) or 0.0) for row in closed)

    levels = sorted(closed_by_level)
    thresholds = sorted(all_thresholds)
    if len(thresholds) > 64:
        take = np.linspace(0, len(thresholds) - 1, 64).round().astype(int)
        thresholds = [thresholds[int(idx)] for idx in take]
    thresholds = sorted(dict.fromkeys(float(value) for value in thresholds))
    radius_grade_values = {idx: float(value) for idx, value in enumerate(thresholds)}

    def radius_grade(value: float) -> int:
        # Use the exact sampled radius-grid index, not a rounded bucket. This
        # keeps the finite F2[x_level,x_radius] fibers injective over the
        # displayed grid and prevents distinct radii from overwriting one
        # another in the persistence-module dictionary.
        if not thresholds:
            return 0
        idx = bisect_left(thresholds, float(value) - 1e-12)
        return int(min(max(idx, 0), len(thresholds) - 1))

    subsets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    fiber_profile: list[dict[str, Any]] = []
    generator_by_simplex: dict[tuple[str, ...], dict[str, Any]] = {}

    for level in levels:
        level_simplices: dict[tuple[str, ...], dict[str, Any]] = {}
        for src_level in levels:
            if src_level > level:
                continue
            for simplex in closed_by_level.get(src_level, []):
                key = tuple(simplex["simplex"])
                existing = level_simplices.get(key)
                if existing is None or float(simplex.get("filtration", 0.0) or 0.0) < float(existing.get("filtration", 0.0) or 0.0):
                    level_simplices[key] = {**simplex, "first_level": src_level}
                    prior = generator_by_simplex.get(key)
                    if prior is None or (src_level, float(simplex.get("filtration", 0.0) or 0.0)) < (
                        int(prior["multidegree"][0]),
                        float(prior.get("filtration", 0.0) or 0.0),
                    ):
                        generator_by_simplex[key] = {
                            **simplex,
                            "multidegree": [src_level, radius_grade(float(simplex.get("filtration", 0.0) or 0.0))],
                            "radius_value": float(simplex.get("filtration", 0.0) or 0.0),
                            "first_level": src_level,
                        }
        for radius_idx, threshold in enumerate(thresholds):
            subset = [row for row in level_simplices.values() if float(row.get("filtration", 0.0) or 0.0) <= threshold + 1e-12]
            grade = (level, radius_idx)
            subsets[grade] = subset
            by_dim = _simplices_by_dim(subset)
            boundaries = _boundary_reports(by_dim, max_syzygy_basis=0)
            homology = _homology_report(by_dim, boundaries)
            fiber_profile.append(
                {
                    "grade": [level, radius_idx],
                    "level": level,
                    "radius_grade": radius_idx,
                    "radius": float(threshold),
                    "chain_group_ranks": {str(dim): len(rows_) for dim, rows_ in sorted(by_dim.items())},
                    "betti": homology["betti"],
                    "boundary_ranks": {key: value.get("rank", 0) for key, value in boundaries.get("maps", {}).items()},
                    "euler_characteristic": _euler_characteristic(by_dim),
                }
            )

    grid_points = sorted(subsets)
    rank_samples: list[dict[str, Any]] = []
    for i, u in enumerate(grid_points):
        for v in grid_points[i + 1 :]:
            if _leq(u, v):
                rank_samples.append({"source_grade": list(u), "target_grade": list(v), "h0_rank": _h0_inclusion_rank(subsets[u], subsets[v])})
                if len(rank_samples) >= max_rank_invariant_pairs:
                    break
        if len(rank_samples) >= max_rank_invariant_pairs:
            break

    generators = sorted(generator_by_simplex.values(), key=lambda row: (row["multidegree"], row["dimension"], row["simplex"]))
    boundary = _multigraded_boundary_monomials_named(generators, ["x_level", "x_radius"])
    report = {
        "available": True,
        "num_parameters": 2,
        "parameters": _level_radius_parameters(),
        "coefficient_ring": "F2[x_level,x_radius]",
        "levels": levels,
        "radii": thresholds,
        "chain_module_generators": [
            {
                "simplex": row["simplex"],
                "homological_degree": int(row["dimension"]),
                "multidegree": row["multidegree"],
                "filtration": float(row.get("filtration", 0.0) or 0.0),
                "radius_value": float(row.get("radius_value", row.get("filtration", 0.0)) or 0.0),
                "source": row.get("source", "observed"),
            }
            for row in generators
        ],
        "boundary_monomials": boundary,
        "grid_axes": [levels, list(range(len(thresholds)))],
        "radius_grade_values": radius_grade_values,
        "radius_grade_policy": "exact_sorted_radius_grid_index_no_bucket_collision",
        "fiber_rank_profile": fiber_profile,
        "rank_invariant_samples": rank_samples,
        "notes": [
            "A finite 2-parameter persistence module is represented as a multigraded F2[x_level,x_radius]-module.",
            "Each fiber is computed by F2 chain complexes on K_{level,radius}; H0 rank samples are exact inclusion-induced ranks on the sampled radius grid.",
            "The x_radius grade is the exact sorted radius-grid index; radius_grade_values records the real radius for each index.",
            "Real free resolutions are unavailable unless a CAS backend certifies them; the emitted object is only the exact multigraded free chain presentation plus labeled boundary maps.",
        ],
    }
    chain_report = _multigraded_free_resolution_report(report)
    report["chain_presentation_diagnostics"] = chain_report
    report["free_resolution"] = {**chain_report, "deprecated_alias_for": "chain_presentation_diagnostics"}
    return report


def _two_parameter_module_report(
    simplices: list[dict[str, Any]],
    max_grid: int,
    max_pairs: int,
) -> dict[str, Any]:
    if not simplices:
        return {"available": False, "num_parameters": 2, "parameters": _filtration_dimension_parameters(), "fiber_rank_profile": [], "rank_invariant_samples": []}
    max_dim = max(int(row["dimension"]) for row in simplices) or 1
    enriched = []
    for row in simplices:
        multidegree = [_bucket(float(row.get("filtration", 0.0) or 0.0)), _bucket(int(row["dimension"]) / max_dim)]
        enriched.append({**row, "multidegree": multidegree})
    grid_axes = [_axis_values(enriched, idx, max_grid=max_grid) for idx in range(2)]
    grid_points = list(product(*grid_axes))
    fiber_profile = []
    for point in grid_points:
        subset = [row for row in enriched if _leq(row["multidegree"], point)]
        by_dim = _simplices_by_dim(subset)
        boundaries = _boundary_reports(by_dim, max_syzygy_basis=0)
        homology = _homology_report(by_dim, boundaries)
        fiber_profile.append(
            {
                "grade": list(point),
                "chain_group_ranks": {str(dim): len(rows) for dim, rows in sorted(by_dim.items())},
                "betti": homology["betti"],
                "boundary_ranks": {key: value.get("rank", 0) for key, value in boundaries.get("maps", {}).items()},
                "euler_characteristic": _euler_characteristic(by_dim),
            }
        )
    report = {
        "available": True,
        "num_parameters": 2,
        "parameters": _filtration_dimension_parameters(),
        "coefficient_ring": "F2[x_filtration,x_dimension]",
        "chain_module_generators": [
            {"simplex": row["simplex"], "homological_degree": int(row["dimension"]), "multidegree": row["multidegree"], "source": row.get("source", "observed")}
            for row in enriched
        ],
        "boundary_monomials": _multigraded_boundary_monomials_named(enriched, ["x_filtration", "x_dimension"]),
        "grid_axes": [list(axis) for axis in grid_axes],
        "fiber_rank_profile": fiber_profile,
        "rank_invariant_samples": _h0_rank_invariant_samples_generic(enriched, grid_points, max_pairs=max_pairs),
        "notes": [
            "This is a genuine two-parameter finite module over scalar filtration and simplex dimension.",
            "Trajectory growth uses the separate explicit F2[x_level,x_radius] bifiltration report.",
        ],
    }
    chain_report = _multigraded_free_resolution_report(report)
    report["chain_presentation_diagnostics"] = chain_report
    report["free_resolution"] = {**chain_report, "deprecated_alias_for": "chain_presentation_diagnostics"}
    return report


def _level_radius_parameters() -> list[dict[str, str]]:
    return [
        {"name": "trajectory_level", "meaning": "reasoning growth level in the sampled graph-of-thought"},
        {"name": "radius", "meaning": "scalar radius/filtration threshold"},
    ]


def _filtration_dimension_parameters() -> list[dict[str, str]]:
    return [
        {"name": "filtration", "meaning": "original scalar filtration threshold"},
        {"name": "simplex_dimension", "meaning": "normalized homological/simplex dimension"},
    ]


def _multiparameter_module_report(
    simplices: list[dict[str, Any]],
    max_grid: int,
    max_pairs: int,
) -> dict[str, Any]:
    if not simplices:
        return {"num_parameters": 3, "parameters": [], "fiber_rank_profile": [], "rank_invariant_samples": []}
    labels = sorted({v for row in simplices for v in row["simplex"]})
    vertex_order = {label: idx for idx, label in enumerate(labels)}
    max_dim = max(int(row["dimension"]) for row in simplices) or 1
    max_pos = max(len(labels) - 1, 1)
    enriched = []
    for row in simplices:
        max_vertex_position = max(vertex_order[v] for v in row["simplex"]) if row["simplex"] else 0
        multidegree = [
            _bucket(float(row.get("filtration", 0.0))),
            _bucket(int(row["dimension"]) / max_dim),
            _bucket(max_vertex_position / max_pos),
        ]
        enriched.append({**row, "multidegree": multidegree})

    grid_axes = [_axis_values(enriched, idx, max_grid=max_grid) for idx in range(3)]
    grid_points = list(product(*grid_axes))
    fiber_profile = []
    for point in grid_points:
        subset = [row for row in enriched if _leq(row["multidegree"], point)]
        by_dim = _simplices_by_dim(subset)
        boundaries = _boundary_reports(by_dim, max_syzygy_basis=0)
        homology = _homology_report(by_dim, boundaries)
        fiber_profile.append(
            {
                "grade": list(point),
                "chain_group_ranks": {str(dim): len(rows) for dim, rows in sorted(by_dim.items())},
                "betti": homology["betti"],
                "euler_characteristic": _euler_characteristic(by_dim),
            }
        )

    boundary_monomials = _multigraded_boundary_monomials(enriched)
    h0_pairs = _h0_rank_invariant_samples(enriched, grid_points, max_pairs=max_pairs)
    return {
        "num_parameters": 3,
        "parameters": [
            {"name": "filtration", "meaning": "original scalar filtration threshold"},
            {"name": "simplex_dimension", "meaning": "normalized homological/simplex dimension"},
            {"name": "max_vertex_position", "meaning": "maximum ordered vertex position inside the simplex"},
        ],
        "coefficient_ring": "F2[x_filtration,x_dimension,x_position]",
        "chain_module_generators": [
            {
                "simplex": row["simplex"],
                "homological_degree": int(row["dimension"]),
                "multidegree": row["multidegree"],
                "source": row.get("source", "observed"),
            }
            for row in enriched
        ],
        "boundary_monomials": boundary_monomials,
        "grid_axes": [list(axis) for axis in grid_axes],
        "fiber_rank_profile": fiber_profile,
        "rank_invariant_samples": h0_pairs,
        "notes": [
            "This is a genuine three-parameter finite grid module, not a scalar barcode.",
            "The displayed H0 rank-invariant samples are exact ranks of inclusion-induced maps for comparable multigrades.",
            "Higher homology map ranks require heavier matrix-chain reduction and are left to optional backends such as multipers/RIVET/Macaulay2.",
        ],
    }


def _multigraded_free_resolution_report(module: dict[str, Any]) -> dict[str, Any]:
    generators = module.get("chain_module_generators", [])
    by_degree: Counter[int] = Counter(int(row.get("homological_degree", 0)) for row in generators)
    boundary_counts = {key: len(value) for key, value in module.get("boundary_monomials", {}).items() if isinstance(value, list)}
    matrices = _monomial_boundary_matrices(module)
    determinantal = _determinantal_ideal_report(module, matrices)
    fitting = _fitting_ideal_report(matrices, determinantal)
    be = _buchsbaum_eisenbud_complex_report(module, matrices, determinantal)
    m2 = _macaulay2_style_resolution_report(module, matrices, determinantal, fitting, be)
    real_resolution = _real_free_resolution_backend_report(module)
    return {
        "ring": module.get("coefficient_ring"),
        "method": "finite_multigraded_chain_presentation_diagnostics",
        "not_a_free_resolution": True,
        "resolution_status": "chain_presentation_only",
        "certificate_attached": False,
        "field": "F2",
        "free_chain_modules": [{"homological_degree": degree, "rank": count} for degree, count in sorted(by_degree.items())],
        "monomial_labeled_boundary_entry_counts": boundary_counts,
        "determinantal_ideals": determinantal,
        "fitting_ideals": fitting,
        "buchsbaum_eisenbud": be,
        "chain_presentation_diagnostics": m2,
        "macaulay2_style": m2,
        "real_free_resolution": real_resolution,
        "minimal_free_resolution": {
            "available": False,
            "reason": "minimal multigraded free resolutions, Groebner bases, ideal grade/depth, and certified Buchsbaum-Eisenbud multipliers require a real CAS backend such as Macaulay2/Singular/Sage plus optional BEMultipliers; no Python diagnostic is substituted",
        },
        "interpretation": "Exact multigraded free chain presentation over the reported polynomial ring. Determinantal/Fitting data is computed from bounded monomial boundary matrices. This object is not a free resolution; real resolutions are unavailable unless a CAS certificate is attached.",
    }



def _real_free_resolution_backend_report(module: dict[str, Any]) -> dict[str, Any]:
    executable_backends = {
        "Macaulay2_M2": shutil.which("M2"),
        "SageMath_sage": shutil.which("sage"),
        "Singular": shutil.which("Singular"),
    }
    python_backends = {
        name: bool(importlib.util.find_spec(name))
        for name in ("sageall", "sage", "sageconf")
    }
    multiparameter_backends = {"multipers": bool(importlib.util.find_spec("multipers"))}
    candidate_backend_available = any(bool(path) for path in executable_backends.values()) or any(python_backends.values())
    certificate_attached = False
    return {
        "available": bool(certificate_attached),
        "candidate_backend_available": bool(candidate_backend_available),
        "ring": module.get("coefficient_ring"),
        "allowed_backends": [
            "Macaulay2: res, syz, KustinMiller::resBE, MultiplierIdeals",
            "SageMath: syzygy_matrix, free_resolution, macaulay2 bridge",
            "Singular: std/syz/res via Sage or CLI",
            "BEMultipliers: https://github.com/amelie-iska/BEMultipliers.git for Buchsbaum-Eisenbud multiplier analyses",
        ],
        "executable_backends": executable_backends,
        "python_backends": python_backends,
        "multiparameter_backends": multiparameter_backends,
        "certificate_attached": certificate_attached,
        "reason": "No Macaulay2/SageMath/Singular/BEMultipliers certificate is attached in the active tokengt environment; minimal free resolutions are therefore unavailable and must not be rendered or scored as real resolutions.",
        "multipers_note": "multipers is useful for multiparameter persistence invariants and signed measures, but it is not treated here as a certified minimal-free-resolution backend.",
    }


def _macaulay2_style_resolution_report(
    module: dict[str, Any],
    matrices: dict[str, dict[str, Any]],
    determinantal: dict[str, Any],
    fitting: dict[str, Any],
    be: dict[str, Any],
) -> dict[str, Any]:
    ring = str(module.get("coefficient_ring", "F2[x,y]"))
    variables = _ring_variable_names(ring)
    generators = [row for row in module.get("chain_module_generators", []) if isinstance(row, dict)]
    by_degree: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generators:
        by_degree[int(row.get("homological_degree", 0) or 0)].append(row)
    betti_rows: list[dict[str, Any]] = []
    free_modules: list[dict[str, Any]] = []
    for degree in sorted(by_degree):
        shift_counts: Counter[tuple[int, ...]] = Counter()
        for row in by_degree[degree]:
            multidegree = row.get("multidegree", [])
            if isinstance(multidegree, list):
                shift_counts[tuple(int(v) for v in multidegree)] += 1
        shifts = [
            {
                "multidegree": list(shift),
                "multiplicity": int(count),
                "summand": f"{_ring_module_symbol(ring)}(-{','.join(str(v) for v in shift)})" + (f"^{int(count)}" if int(count) > 1 else ""),
            }
            for shift, count in sorted(shift_counts.items())
        ]
        display = f"F_{degree} = " + (" + ".join(row["summand"] for row in shifts[:12]) if shifts else "0") + (" + ..." if len(shifts) > 12 else "")
        free_modules.append({"name": f"F_{degree}", "rank": len(by_degree[degree]), "display": display, "shifts": shifts})
        for shift, count in sorted(shift_counts.items()):
            betti_rows.append({"homological_degree": degree, "multidegree": list(shift), "rank": int(count)})
    differentials: list[dict[str, Any]] = []
    for key, matrix in sorted(matrices.items(), key=lambda item: item[0]):
        degree = int(matrix.get("homological_degree", str(key)[1:] if str(key).startswith("d") else 0) or 0)
        entries = matrix.get("entries", {}) if isinstance(matrix.get("entries"), dict) else {}
        samples = []
        for (r, c), exp in sorted(entries.items())[:24]:
            samples.append({"row": int(r), "column": int(c), "entry": _multi_monomial_named(exp, variables), "exponent": list(exp)})
        differentials.append(
            {
                "name": str(key),
                "display": f"{key}: F_{degree} -> F_{degree - 1}",
                "shape": [int(matrix.get("target_rank", 0)), int(matrix.get("source_rank", 0))],
                "rank_over_F2_incidence": int(matrix.get("rank_over_F2_incidence", 0)),
                "nonzero_entries": int(matrix.get("nonzero_entries", 0)),
                "entry_samples": samples,
                "matrix_preview": _macaulay2_matrix_preview(samples, int(matrix.get("target_rank", 0)), int(matrix.get("source_rank", 0))),
            }
        )
    staircase = _two_variable_staircase_report(generators, variables)
    be_checks = be.get("composition_zero_checks", []) if isinstance(be.get("composition_zero_checks"), list) else []
    exact_checks = be.get("rank_exactness_checks", []) if isinstance(be.get("rank_exactness_checks"), list) else []
    return {
        "available": True,
        "ring": ring,
        "field": "F2",
        "style": "Macaulay2-inspired finite multigraded chain-presentation diagnostic (not a real free resolution)",
        "betti_table_rows": betti_rows,
        "free_modules": free_modules,
        "differentials": differentials,
        "staircase": staircase,
        "chain_complex_certificate": {
            "d_squared_zero_checks": be_checks,
            "rank_exactness_shadow": exact_checks,
            "minimality_certified": False,
            "derived_equivalence_certified": False,
            "reason": "This is a finite multigraded free-chain presentation with determinantal/Fitting/BE diagnostics. It is not a free resolution; minimality and derived equivalence require CAS certificates and are not asserted.",
        },
        "fitting_ideals": fitting,
        "determinantal_ideals": determinantal,
    }


def _ring_module_symbol(ring: str) -> str:
    return "S" if "[" in str(ring) else "F"


def _macaulay2_matrix_preview(samples: list[dict[str, Any]], rows: int, cols: int, max_rows: int = 8, max_cols: int = 8) -> list[list[str]]:
    grid = [["0" for _ in range(min(cols, max_cols))] for _ in range(min(rows, max_rows))]
    for item in samples:
        r = int(item.get("row", -1))
        c = int(item.get("column", -1))
        if 0 <= r < len(grid) and 0 <= c < (len(grid[r]) if grid else 0):
            grid[r][c] = str(item.get("entry", "1"))
    return grid


def _two_variable_staircase_report(generators: list[dict[str, Any]], variables: list[str]) -> dict[str, Any]:
    if len(variables) != 2:
        return {"available": False, "reason": "staircase diagram is specific to two-variable multigrades"}
    points = []
    for row in generators:
        multidegree = row.get("multidegree", [])
        if isinstance(multidegree, list) and len(multidegree) >= 2:
            points.append((int(multidegree[0]), int(multidegree[1])))
    counts = Counter(points)
    unique = sorted(counts)
    antichain = []
    for pnt in unique:
        dominated = any(q != pnt and q[0] <= pnt[0] and q[1] <= pnt[1] for q in unique)
        if not dominated:
            antichain.append(pnt)
    antichain = sorted(antichain, key=lambda item: (item[0], item[1]))
    lcms = []
    for left, right in zip(antichain, antichain[1:]):
        lcm = (max(left[0], right[0]), max(left[1], right[1]))
        lcms.append({"from": [list(left), list(right)], "lcm_bidegree": list(lcm), "syzygy_candidate": f"lcm({variables[0]}^{left[0]}{variables[1]}^{left[1]}, {variables[0]}^{right[0]}{variables[1]}^{right[1]})"})
    return {
        "available": True,
        "variables": variables,
        "generator_bidegrees": [{"bidegree": list(pnt), "multiplicity": int(counts[pnt])} for pnt in unique],
        "minimal_antichain_candidates": [list(pnt) for pnt in antichain],
        "adjacent_lcm_syzygy_candidates": lcms,
        "interpretation": "Miller-Sturmfels two-variable staircase candidate: antichain generators and adjacent lcm syzygies are computed from displayed bidegrees; minimal ideal/resolution status is not asserted without CAS.",
    }

def _monomial_boundary_matrices(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    generators = module.get("chain_module_generators", [])
    by_degree: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for row in generators if isinstance(generators, list) else []:
        try:
            degree = int(row.get("homological_degree", 0))
            simplex = tuple(str(part) for part in row.get("simplex", []))
        except Exception:
            continue
        by_degree[degree].append(simplex)
    by_degree = {degree: sorted(set(rows), key=lambda key: (len(key), key)) for degree, rows in by_degree.items()}
    out: dict[str, dict[str, Any]] = {}
    boundary_monomials = module.get("boundary_monomials", {})
    if not isinstance(boundary_monomials, dict):
        return out
    variable_count = len(_ring_variable_names(str(module.get("coefficient_ring", ""))))
    for key, entries in boundary_monomials.items():
        if not str(key).startswith("d") or not isinstance(entries, list):
            continue
        try:
            degree = int(str(key)[1:])
        except ValueError:
            continue
        row_labels = by_degree.get(degree - 1, [])
        col_labels = by_degree.get(degree, [])
        row_index = {label: idx for idx, label in enumerate(row_labels)}
        col_index = {label: idx for idx, label in enumerate(col_labels)}
        matrix_entries: dict[tuple[int, int], tuple[int, ...]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source = tuple(str(part) for part in entry.get("source_simplex", []))
            target = tuple(str(part) for part in entry.get("target_face", []))
            if source not in col_index or target not in row_index:
                continue
            exponent_raw = entry.get("monomial_exponent", [])
            exponent = tuple(int(v) for v in exponent_raw) if isinstance(exponent_raw, list) else tuple()
            if variable_count and len(exponent) < variable_count:
                exponent = tuple(list(exponent) + [0] * (variable_count - len(exponent)))
            matrix_entries[(row_index[target], col_index[source])] = exponent
        incidence = np.zeros((len(row_labels), len(col_labels)), dtype=np.uint8)
        for rc in matrix_entries:
            incidence[rc] = 1
        out[str(key)] = {
            "homological_degree": degree,
            "source_rank": len(col_labels),
            "target_rank": len(row_labels),
            "row_labels": [list(label) for label in row_labels],
            "column_labels": [list(label) for label in col_labels],
            "entries": matrix_entries,
            "rank_over_F2_incidence": int(_rank_mod2(incidence)),
            "nonzero_entries": int(incidence.sum()),
        }
    return out


def _ring_variable_names(ring: str) -> list[str]:
    if "[" not in ring or "]" not in ring:
        return []
    inside = ring.split("[", 1)[1].split("]", 1)[0]
    return [part.strip() for part in inside.split(",") if part.strip()]


def _determinantal_ideal_report(
    module: dict[str, Any],
    matrices: dict[str, dict[str, Any]],
    *,
    max_minor_size: int = 3,
    max_minors_per_size: int = 96,
) -> dict[str, Any]:
    variables = _ring_variable_names(str(module.get("coefficient_ring", "")))
    maps: dict[str, Any] = {}
    for key, matrix in sorted(matrices.items(), key=lambda item: item[0]):
        m = int(matrix.get("target_rank", 0))
        n = int(matrix.get("source_rank", 0))
        rank = int(matrix.get("rank_over_F2_incidence", 0))
        entries = matrix.get("entries", {}) if isinstance(matrix.get("entries"), dict) else {}
        by_size: list[dict[str, Any]] = []
        for size in range(1, min(m, n, max_minor_size) + 1):
            total_minors = comb(m, size) * comb(n, size)
            sampled = total_minors > max_minors_per_size
            nonzero: list[dict[str, Any]] = []
            checked = 0
            for row_choice in combinations(range(m), size):
                for col_choice in combinations(range(n), size):
                    poly = _minor_polynomial_mod2(entries, row_choice, col_choice, len(variables))
                    checked += 1
                    if poly:
                        nonzero.append(
                            {
                                "rows": list(row_choice),
                                "columns": list(col_choice),
                                "polynomial": _polynomial_to_string(poly, variables),
                                "term_exponents": [list(exp) for exp in sorted(poly)],
                            }
                        )
                    if checked >= max_minors_per_size:
                        break
                if checked >= max_minors_per_size:
                    break
            by_size.append(
                {
                    "ideal": f"I_{size}({key})",
                    "minor_size": size,
                    "total_minors": int(total_minors),
                    "checked_minors": int(checked),
                    "sampled": bool(sampled),
                    "nonzero_generator_count_in_checked_minors": int(len(nonzero)),
                    "nonzero_generator_samples": nonzero[:16],
                    "numeric_rank_implication": (
                        "all minors of this size vanish" if size > rank else "at least one minor of this size is nonzero over the F2 incidence specialization"
                    ),
                }
            )
        maps[key] = {
            "shape": [m, n],
            "rank_over_F2_incidence": rank,
            "nonzero_entries": int(matrix.get("nonzero_entries", 0)),
            "determinantal_ideals_by_minor_size": by_size,
            "rank_certificate": {
                "rank": rank,
                "all_(rank+1)_minors_vanish": True,
                "some_rank_minor_nonzero": bool(rank == 0 or rank <= min(m, n)),
                "certificate_kind": "F2 row-reduction plus bounded monomial minor samples",
            },
        }
    return {
        "available": True,
        "ring": module.get("coefficient_ring"),
        "field": "F2",
        "maps": maps,
        "sample_policy": {"max_minor_size": max_minor_size, "max_minors_per_size": max_minors_per_size},
        "notes": [
            "For numeric F2 boundary matrices, determinantal/Fitting ideals collapse to rank facts; the nontrivial entries here are monomial-labeled boundary matrices over the displayed polynomial ring.",
            "Large determinantal ideals are sampled for browser/runtime safety; use the emitted CAS payload to certify full ideals and Groebner bases when needed.",
        ],
    }


def _minor_polynomial_mod2(
    entries: dict[tuple[int, int], tuple[int, ...]],
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    variable_count: int,
) -> set[tuple[int, ...]]:
    size = len(rows)
    terms: Counter[tuple[int, ...]] = Counter()
    for perm in permutations(range(size)):
        exponent = [0] * variable_count
        ok = True
        for ridx, cpos in enumerate(perm):
            entry = entries.get((rows[ridx], cols[cpos]))
            if entry is None:
                ok = False
                break
            if variable_count == 0:
                exponent = []
            else:
                padded = list(entry) + [0] * max(0, variable_count - len(entry))
                exponent = [a + int(b) for a, b in zip(exponent, padded[:variable_count])]
        if ok:
            terms[tuple(exponent)] += 1
    return {exp for exp, count in terms.items() if count % 2 == 1}


def _polynomial_to_string(poly: set[tuple[int, ...]], variables: list[str]) -> str:
    if not poly:
        return "0"
    if not variables:
        return "1"
    return " + ".join(_multi_monomial_named(exp, variables) for exp in sorted(poly))


def _fitting_ideal_report(matrices: dict[str, dict[str, Any]], determinantal: dict[str, Any]) -> dict[str, Any]:
    maps = determinantal.get("maps", {}) if isinstance(determinantal, dict) else {}
    out: dict[str, Any] = {}
    for key, matrix in sorted(matrices.items(), key=lambda item: item[0]):
        target_rank = int(matrix.get("target_rank", 0))
        incidence_rank = int(matrix.get("rank_over_F2_incidence", 0))
        det_by_size = {
            int(row.get("minor_size", -1)): row
            for row in maps.get(key, {}).get("determinantal_ideals_by_minor_size", [])
            if isinstance(row, dict)
        }
        interesting_j = sorted({0, max(target_rank - incidence_rank - 1, 0), max(target_rank - incidence_rank, 0), min(target_rank, max(target_rank - incidence_rank + 1, 0)), target_rank})
        rows = []
        for j in interesting_j:
            minor_size = target_rank - j
            if minor_size <= 0:
                rows.append({"fitting_index": int(j), "ideal": f"Fitt_{j}(coker {key})", "minor_size": 0, "status": "unit_ideal", "generators": ["1"]})
                continue
            det = det_by_size.get(minor_size)
            if det is None:
                rows.append(
                    {
                        "fitting_index": int(j),
                        "ideal": f"Fitt_{j}(coker {key})",
                        "minor_size": int(minor_size),
                        "status": "not_enumerated_in_browser_report",
                        "reason": "minor size exceeds bounded determinantal enumeration; certify with CAS for the full ideal",
                        "numeric_rank_implication": "zero under F2 incidence rank" if minor_size > incidence_rank else "nonzero possible under F2 incidence rank",
                    }
                )
            else:
                rows.append(
                    {
                        "fitting_index": int(j),
                        "ideal": f"Fitt_{j}(coker {key})",
                        "minor_size": int(minor_size),
                        "status": "bounded_generators_reported",
                        "sampled": bool(det.get("sampled", False)),
                        "checked_minors": int(det.get("checked_minors", 0)),
                        "nonzero_generator_count_in_checked_minors": int(det.get("nonzero_generator_count_in_checked_minors", 0)),
                        "generator_samples": det.get("nonzero_generator_samples", []),
                    }
                )
        out[key] = {"presentation": f"C_{key[1:]} -> C_{int(key[1:]) - 1} -> coker({key}) -> 0", "target_free_rank": target_rank, "fitting_invariants": rows}
    return {
        "available": True,
        "definition": "Fitt_j(coker(phi)) = I_{rank(target)-j}(phi), generated by the corresponding minors of the presentation matrix.",
        "maps": out,
    }


def _buchsbaum_eisenbud_complex_report(
    module: dict[str, Any],
    matrices: dict[str, dict[str, Any]],
    determinantal: dict[str, Any],
) -> dict[str, Any]:
    generators = module.get("chain_module_generators", [])
    chain_ranks: Counter[int] = Counter(int(row.get("homological_degree", 0)) for row in generators if isinstance(row, dict))
    degrees = sorted(chain_ranks)
    composition_checks = []
    variable_count = len(_ring_variable_names(str(module.get("coefficient_ring", ""))))
    variables = _ring_variable_names(str(module.get("coefficient_ring", "")))
    for degree in degrees:
        left = matrices.get(f"d{degree}")
        right = matrices.get(f"d{degree + 1}")
        if not left or not right:
            continue
        comp = _composition_zero_report(left, right, variables)
        composition_checks.append({"at": f"d{degree} o d{degree + 1}", **comp})
    exactness = []
    for degree in degrees:
        b = int(chain_ranks[degree])
        rank_out = int(matrices.get(f"d{degree}", {}).get("rank_over_F2_incidence", 0))
        rank_in = int(matrices.get(f"d{degree + 1}", {}).get("rank_over_F2_incidence", 0))
        homology_rank = max(b - rank_out - rank_in, 0)
        exactness.append(
            {
                "chain_module": f"C_{degree}",
                "free_rank": b,
                "rank_outgoing_boundary": rank_out,
                "rank_incoming_boundary": rank_in,
                "finite_F2_homology_rank": homology_rank,
                "exact_at_chain_module_over_F2_incidence": bool(homology_rank == 0),
            }
        )
    maximal_minor_inputs = []
    det_maps = determinantal.get("maps", {}) if isinstance(determinantal, dict) else {}
    for key, matrix in sorted(matrices.items(), key=lambda item: item[0]):
        rank = int(matrix.get("rank_over_F2_incidence", 0))
        candidates = []
        for row in det_maps.get(key, {}).get("determinantal_ideals_by_minor_size", []):
            if isinstance(row, dict) and int(row.get("minor_size", -1)) == rank:
                candidates = row.get("nonzero_generator_samples", [])[:8]
                break
        maximal_minor_inputs.append({"map": key, "rank": rank, "maximal_minor_samples": candidates})
    return {
        "available": True,
        "criterion_used_for_finite_display": "Buchsbaum-Eisenbud rank/exactness shadow: d_i d_{i+1}=0 and rank(d_i)+rank(d_{i+1})=rank(C_i) over the finite F2 incidence specialization.",
        "composition_zero_checks": composition_checks,
        "rank_exactness_checks": exactness,
        "maximal_minor_inputs_for_multipliers": maximal_minor_inputs,
        "buchsbaum_eisenbud_multipliers": {
            "available": False,
            "reason": "BE multipliers are symbolic complementary-minor relations in the polynomial coordinate ring; this report emits the maximal-minor inputs but requires a CAS backend to certify multiplier equations and grade/depth exactness conditions.",
        },
        "grade_depth_conditions": {
            "available": False,
            "reason": "Ideal grade/depth and regular-element checks for determinantal ideals require Macaulay2/Singular/Sage; no proxy is substituted.",
        },
    }


def _composition_zero_report(left: dict[str, Any], right: dict[str, Any], variables: list[str]) -> dict[str, Any]:
    variable_count = len(variables)
    left_entries = left.get("entries", {}) if isinstance(left.get("entries"), dict) else {}
    right_entries = right.get("entries", {}) if isinstance(right.get("entries"), dict) else {}
    left_rows = int(left.get("target_rank", 0))
    middle = int(left.get("source_rank", 0))
    right_cols = int(right.get("source_rank", 0))
    nonzero_entries = []
    checked = 0
    for i in range(left_rows):
        for j in range(right_cols):
            terms: Counter[tuple[int, ...]] = Counter()
            for k in range(middle):
                a = left_entries.get((i, k))
                b = right_entries.get((k, j))
                if a is None or b is None:
                    continue
                aa = list(a) + [0] * max(0, variable_count - len(a))
                bb = list(b) + [0] * max(0, variable_count - len(b))
                terms[tuple(int(x) + int(y) for x, y in zip(aa[:variable_count], bb[:variable_count]))] += 1
            poly = {exp for exp, count in terms.items() if count % 2 == 1}
            checked += 1
            if poly:
                nonzero_entries.append({"row": i, "column": j, "polynomial": _polynomial_to_string(poly, variables)})
    return {"checked_entries": checked, "is_zero_over_F2_monomial_matrix": not nonzero_entries, "nonzero_entry_samples": nonzero_entries[:8]}


def _axis_values(enriched: list[dict[str, Any]], index: int, max_grid: int) -> tuple[int, ...]:
    values = sorted({int(row["multidegree"][index]) for row in enriched})
    if len(values) <= max_grid:
        return tuple(values)
    take = np.linspace(0, len(values) - 1, max_grid).round().astype(int)
    return tuple(values[int(idx)] for idx in take)


def _bucket(value: float, scale: int = 10) -> int:
    return int(max(0, min(scale, round(float(value) * scale))))


def _leq(left: Iterable[int], right: Iterable[int]) -> bool:
    return all(int(a) <= int(b) for a, b in zip(left, right))


def _multi_monomial(exponent: Iterable[int]) -> str:
    return _multi_monomial_named(exponent, ["x_filtration", "x_dimension", "x_position"])


def _multi_monomial_named(exponent: Iterable[int], names: list[str]) -> str:
    pieces = []
    for name, power in zip(names, exponent):
        power = int(power)
        if power == 0:
            continue
        if power == 1:
            pieces.append(name)
        else:
            pieces.append(f"{name}^{power}")
    return "1" if not pieces else "*".join(pieces)


def _multipers_backend_status() -> dict[str, Any]:
    try:
        import multipers  # type: ignore

        public = [name for name in dir(multipers) if not name.startswith("_")][:48]
        return {"available": True, "version": getattr(multipers, "__version__", ""), "public_api_sample": public}
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "recommendation": "Install multipers to add module approximation/signed-measure backends for high-scale multiparameter graph descriptors.",
        }


def _multigraded_boundary_monomials(enriched: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return _multigraded_boundary_monomials_named(enriched, ["x_filtration", "x_dimension", "x_position"])


def _multigraded_boundary_monomials_named(enriched: list[dict[str, Any]], names: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_key = {tuple(row["simplex"]): row for row in enriched}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        simplex = tuple(row["simplex"])
        dim = int(row["dimension"])
        if dim <= 0:
            continue
        for face in combinations(simplex, len(simplex) - 1):
            face_key = tuple(sorted(face))
            face_row = by_key.get(face_key)
            if face_row is None:
                continue
            exponent = [max(int(a) - int(b), 0) for a, b in zip(row["multidegree"], face_row["multidegree"])]
            out[f"d{dim}"].append(
                {
                    "source_simplex": list(simplex),
                    "target_face": list(face_key),
                    "source_multidegree": row["multidegree"],
                    "target_multidegree": face_row["multidegree"],
                    "monomial_exponent": exponent,
                    "monomial": _multi_monomial_named(exponent, names),
                }
            )
    return dict(out)


def _h0_rank_invariant_samples(
    enriched: list[dict[str, Any]],
    grid_points: list[tuple[int, int, int]],
    max_pairs: int,
) -> list[dict[str, Any]]:
    return _h0_rank_invariant_samples_generic(enriched, grid_points, max_pairs=max_pairs)


def _h0_rank_invariant_samples_generic(
    enriched: list[dict[str, Any]],
    grid_points: list[tuple[int, ...]],
    max_pairs: int,
) -> list[dict[str, Any]]:
    pairs = []
    for i, u in enumerate(grid_points):
        for v in grid_points[i + 1 :]:
            if _leq(u, v):
                pairs.append((u, v))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    samples = []
    for u, v in pairs:
        ku = [row for row in enriched if _leq(row["multidegree"], u)]
        kv = [row for row in enriched if _leq(row["multidegree"], v)]
        samples.append({"source_grade": list(u), "target_grade": list(v), "h0_rank": _h0_inclusion_rank(ku, kv)})
    return samples


def _h0_inclusion_rank(source: list[dict[str, Any]], target: list[dict[str, Any]]) -> int:
    source_vertices = [tuple(row["simplex"])[0] for row in source if int(row["dimension"]) == 0 and len(row["simplex"]) == 1]
    target_vertices = [tuple(row["simplex"])[0] for row in target if int(row["dimension"]) == 0 and len(row["simplex"]) == 1]
    target_edges = [tuple(row["simplex"]) for row in target if int(row["dimension"]) == 1 and len(row["simplex"]) == 2]
    if not source_vertices:
        return 0
    parent = {v: v for v in target_vertices}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in target_edges:
        union(a, b)
    return len({find(v) for v in source_vertices if v in parent})


def _closed_complex(filtered_object: dict[str, Any], max_simplices: int) -> dict[str, Any]:
    raw_simplices = list(filtered_object.get("simplices", []))
    vertices_seen: dict[str, float] = {}
    closed: dict[tuple[str, ...], dict[str, Any]] = {}

    for simplex in raw_simplices:
        labels = _canonical(simplex.get("simplex", []))
        if len(labels) == 1:
            vertices_seen[labels[0]] = min(vertices_seen.get(labels[0], float("inf")), float(simplex.get("filtration", 0.0)))

    for simplex in raw_simplices:
        labels = _canonical(simplex.get("simplex", []))
        if not labels:
            continue
        filtration = float(simplex.get("filtration", 0.0))
        for size in range(1, len(labels) + 1):
            for face in combinations(labels, size):
                face_key = tuple(sorted(face))
                vertex_floor = max(vertices_seen.get(v, 0.0) for v in face_key)
                face_filtration = min(filtration, vertex_floor if size < len(labels) else filtration)
                observed = size == len(labels)
                entry = closed.get(face_key)
                if entry is None:
                    closed[face_key] = {
                        "simplex": list(face_key),
                        "dimension": len(face_key) - 1,
                        "filtration": round(face_filtration, 6),
                        "source": "observed" if observed else "closure",
                        "types": [str(simplex.get("type", "simplex"))] if observed else ["closure_face"],
                    }
                else:
                    entry["filtration"] = round(min(float(entry["filtration"]), face_filtration), 6)
                    if observed:
                        entry["source"] = "observed"
                        entry.setdefault("types", []).append(str(simplex.get("type", "simplex")))

    ordered = sorted(closed.values(), key=lambda row: (row["filtration"], row["dimension"], row["simplex"]))
    truncated = len(ordered) > max_simplices
    if truncated:
        ordered = ordered[:max_simplices]
        keep = {tuple(row["simplex"]) for row in ordered}
        # Re-close the retained cofaces so the reported object is still a complex.
        for row in list(ordered):
            labels = tuple(row["simplex"])
            for size in range(1, len(labels)):
                for face in combinations(labels, size):
                    key = tuple(sorted(face))
                    if key not in keep:
                        ordered.append(
                            {
                                "simplex": list(key),
                                "dimension": len(key) - 1,
                                "filtration": row["filtration"],
                                "source": "closure_after_truncation",
                                "types": ["closure_face"],
                            }
                        )
                        keep.add(key)
        ordered = sorted({tuple(row["simplex"]): row for row in ordered}.values(), key=lambda row: (row["filtration"], row["dimension"], row["simplex"]))

    counts = Counter(int(row["dimension"]) for row in ordered)
    return {
        "simplices": ordered,
        "summary": {
            "simplices": len(ordered),
            "truncated": truncated,
            "max_simplices": max_simplices,
            "added_closure_faces": sum(1 for row in ordered if str(row.get("source", "")).startswith("closure")),
            "dimension_counts": {str(dim): int(count) for dim, count in sorted(counts.items())},
            "max_dimension": max(counts) if counts else -1,
        },
    }


def _simplices_by_dim(simplices: list[dict[str, Any]]) -> dict[int, list[tuple[str, ...]]]:
    by_dim: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for simplex in simplices:
        key = tuple(simplex["simplex"])
        by_dim[int(simplex["dimension"])].append(key)
    return {dim: sorted(rows, key=lambda key: (len(key), key)) for dim, rows in by_dim.items()}


def _boundary_reports(by_dim: dict[int, list[tuple[str, ...]]], max_syzygy_basis: int) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    syzygies: dict[str, Any] = {}
    max_dim = max(by_dim) if by_dim else 0
    for dim in range(1, max_dim + 1):
        rows = by_dim.get(dim - 1, [])
        cols = by_dim.get(dim, [])
        matrix = _boundary_matrix(rows, cols)
        rank = _rank_mod2(matrix)
        nullspace = _nullspace_mod2(matrix)
        key = f"d{dim}"
        maps[key] = {
            "source": f"C_{dim}",
            "target": f"C_{dim - 1}",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "rank": int(rank),
            "nullity": int(matrix.shape[1] - rank),
            "nonzeros": int(matrix.sum()),
        }
        syzygies[f"ker_{key}"] = {
            "rank": int(len(nullspace)),
            "basis_truncated": len(nullspace) > max_syzygy_basis,
            "basis": _basis_supports(nullspace[:max_syzygy_basis], cols),
        }
    return {"maps": maps, "syzygies": syzygies}


def _boundary_matrix(rows: list[tuple[str, ...]], cols: list[tuple[str, ...]]) -> np.ndarray:
    row_index = {row: idx for idx, row in enumerate(rows)}
    matrix = np.zeros((len(rows), len(cols)), dtype=np.uint8)
    for col_idx, simplex in enumerate(cols):
        for face in combinations(simplex, len(simplex) - 1):
            key = tuple(sorted(face))
            row_idx = row_index.get(key)
            if row_idx is not None:
                matrix[row_idx, col_idx] ^= 1
    return matrix


def _homology_report(by_dim: dict[int, list[tuple[str, ...]]], boundaries: dict[str, Any]) -> dict[str, Any]:
    max_dim = max(by_dim) if by_dim else 0
    betti: dict[str, int] = {}
    cycles: dict[str, int] = {}
    boundaries_rank: dict[str, int] = {}
    for dim in range(0, max_dim + 1):
        dim_ck = len(by_dim.get(dim, []))
        rank_dk = boundaries["maps"].get(f"d{dim}", {}).get("rank", 0) if dim > 0 else 0
        rank_next = boundaries["maps"].get(f"d{dim + 1}", {}).get("rank", 0)
        z_dim = dim_ck - int(rank_dk)
        beta = max(z_dim - int(rank_next), 0)
        cycles[str(dim)] = int(z_dim)
        boundaries_rank[str(dim)] = int(rank_next)
        betti[str(dim)] = int(beta)
    return {"betti": betti, "cycle_ranks": cycles, "boundary_ranks": boundaries_rank}


def _persistence_module_report(simplices: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = sorted({float(row["filtration"]) for row in simplices})
    states = []
    for threshold in thresholds:
        subset = [row for row in simplices if float(row["filtration"]) <= threshold + 1e-12]
        by_dim = _simplices_by_dim(subset)
        boundaries = _boundary_reports(by_dim, max_syzygy_basis=0)
        homology = _homology_report(by_dim, boundaries)
        states.append(
            {
                "threshold": threshold,
                "chain_group_ranks": {str(dim): len(rows) for dim, rows in sorted(by_dim.items())},
                "betti": homology["betti"],
                "euler_characteristic": _euler_characteristic(by_dim),
            }
        )
    return {"thresholds": thresholds, "states": states, "structure_maps": "inclusions along increasing filtration thresholds"}


def _persistent_homology_report(simplices: list[dict[str, Any]], ph_backend: str) -> dict[str, Any]:
    backend = (ph_backend or "auto").lower()
    if backend == "none":
        return {"backend": "none", "intervals": [], "available": False}
    if backend == "ripser":
        return _ripser_persistent_homology_report(simplices)
    if backend not in {"auto", "gudhi"}:
        return {"backend": backend, "available": False, "error": f"Unsupported persistent homology backend: {backend}", "intervals": []}
    try:
        import gudhi  # type: ignore

        vertex_labels = sorted({v for row in simplices for v in row["simplex"]})
        vertex_index = {label: idx for idx, label in enumerate(vertex_labels)}
        tree = gudhi.SimplexTree()
        for row in sorted(simplices, key=lambda item: (len(item["simplex"]), item["filtration"])):
            tree.insert([vertex_index[v] for v in row["simplex"]], filtration=float(row["filtration"]))
        tree.make_filtration_non_decreasing()
        raw = tree.persistence(homology_coeff_field=2, min_persistence=0.0)
        intervals = []
        for dim, (birth, death) in raw:
            infinite = bool(isinf(float(death)))
            intervals.append(
                {
                    "dimension": int(dim),
                    "birth": float(birth),
                    "death": None if infinite else float(death),
                    "infinite": infinite,
                }
            )
        return {"backend": "gudhi", "available": True, "intervals": intervals}
    except Exception as exc:
        return {"backend": "gudhi", "available": False, "error": f"{type(exc).__name__}: {exc}", "intervals": []}


def _ripser_persistent_homology_report(simplices: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from ripser import ripser  # type: ignore
    except Exception as exc:
        return {"backend": "ripser", "available": False, "error": f"{type(exc).__name__}: {exc}", "intervals": []}
    vertices = sorted({v for row in simplices for v in row["simplex"]})
    if not vertices:
        return {"backend": "ripser", "available": True, "intervals": []}
    index = {label: idx for idx, label in enumerate(vertices)}
    finite_filtrations = [float(row.get("filtration", 0.0)) for row in simplices if not isinf(float(row.get("filtration", 0.0)))]
    far = (max(finite_filtrations) if finite_filtrations else 1.0) + 1.0
    distance = np.full((len(vertices), len(vertices)), far, dtype=float)
    np.fill_diagonal(distance, 0.0)
    for row in simplices:
        simplex = list(row.get("simplex", []))
        if len(simplex) == 2:
            i, j = index[simplex[0]], index[simplex[1]]
            value = max(float(row.get("filtration", 0.0)), 0.0)
            distance[i, j] = min(distance[i, j], value)
            distance[j, i] = min(distance[j, i], value)
    maxdim = min(2, max((int(row.get("dimension", 0)) for row in simplices), default=0))
    try:
        raw = ripser(distance, distance_matrix=True, maxdim=maxdim)
    except Exception as exc:
        return {"backend": "ripser", "available": False, "error": f"{type(exc).__name__}: {exc}", "intervals": []}
    intervals = []
    for dim, diagram in enumerate(raw.get("dgms", [])):
        for birth, death in np.asarray(diagram):
            infinite = bool(isinf(float(death)))
            intervals.append(
                {
                    "dimension": int(dim),
                    "birth": float(birth),
                    "death": None if infinite else float(death),
                    "infinite": infinite,
                }
            )
    return {"backend": "ripser", "available": True, "intervals": intervals}


def _persistence_representations_report(
    persistence: dict[str, Any],
    *,
    max_dimension: int = 2,
    landscape_layers: int = 4,
    resolution: int = 96,
    image_resolution: int = 24,
    topological_vector_threshold: int = 24,
) -> dict[str, Any]:
    intervals = persistence.get("intervals", []) if isinstance(persistence, dict) else []
    diagrams = _finite_diagrams_by_dimension(intervals, max_dimension=max_dimension)
    finite_count = int(sum(diagram.shape[0] for diagram in diagrams.values()))
    if finite_count == 0:
        return {
            "available": False,
            "backend": "gudhi.representations",
            "reason": "no finite persistence intervals",
            "decision_policy": _persistence_vectorization_decision_policy(),
            "diagrams_by_dimension": {str(dim): [] for dim in range(max_dimension + 1)},
        }
    try:
        from gudhi.representations import (  # type: ignore
            BettiCurve,
            Entropy,
            Landscape,
            PersistenceImage,
            PersistenceLengths,
            Silhouette,
            TopologicalVector,
        )
    except Exception as exc:
        return {
            "available": False,
            "backend": "gudhi.representations",
            "error": f"{type(exc).__name__}: {exc}",
            "decision_policy": _persistence_vectorization_decision_policy(),
            "diagrams_by_dimension": {str(dim): _diagram_to_rows(diagrams.get(dim, np.zeros((0, 2)))) for dim in range(max_dimension + 1)},
        }

    method_reports: dict[str, Any] = {}
    summary: dict[str, float] = {}
    for dim in range(max_dimension + 1):
        diagram = diagrams.get(dim, np.zeros((0, 2), dtype=float))
        if diagram.size == 0:
            method_reports[str(dim)] = {"available": False, "reason": "no finite intervals in dimension"}
            continue
        try:
            landscape = Landscape(num_landscapes=landscape_layers, resolution=resolution, keep_endpoints=True)
            landscape_vec = np.asarray(landscape.fit_transform([diagram])[0], dtype=float)
            landscape_grid = np.asarray(getattr(landscape, "grid_", np.linspace(float(np.min(diagram[:, 0])), float(np.max(diagram[:, 1])), resolution)), dtype=float)
            betti_curve = BettiCurve(resolution=resolution, keep_endpoints=True)
            betti_vec = np.asarray(betti_curve.fit_transform([diagram])[0], dtype=float)
            betti_grid = np.asarray(getattr(betti_curve, "grid_", landscape_grid), dtype=float)
            silhouette = Silhouette(resolution=resolution, keep_endpoints=True)
            silhouette_vec = np.asarray(silhouette.fit_transform([diagram])[0], dtype=float)
            silhouette_grid = np.asarray(getattr(silhouette, "grid_", landscape_grid), dtype=float)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
                entropy_scalar = np.asarray(Entropy(mode="scalar").fit_transform([diagram])[0], dtype=float)
                entropy_vector = np.asarray(Entropy(mode="vector", resolution=resolution, keep_endpoints=True).fit_transform([diagram])[0], dtype=float)
            lengths_vec = np.asarray(PersistenceLengths(num_lengths=topological_vector_threshold).fit_transform([diagram])[0], dtype=float)
            topological_vec = np.asarray(TopologicalVector(threshold=topological_vector_threshold).fit_transform([diagram])[0], dtype=float)
            image = PersistenceImage(
                bandwidth=max(_diagram_scale(diagram) / 18.0, 1e-3),
                resolution=[image_resolution, image_resolution],
            )
            image_vec = np.asarray(image.fit_transform([diagram])[0], dtype=float)
            image_grid = image_vec.reshape(image_resolution, image_resolution)
        except Exception as exc:
            method_reports[str(dim)] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "finite_interval_count": int(diagram.shape[0]),
            }
            continue
        landscape_vec = np.nan_to_num(landscape_vec, nan=0.0, posinf=0.0, neginf=0.0)
        betti_vec = np.nan_to_num(betti_vec, nan=0.0, posinf=0.0, neginf=0.0)
        silhouette_vec = np.nan_to_num(silhouette_vec, nan=0.0, posinf=0.0, neginf=0.0)
        entropy_scalar = np.nan_to_num(entropy_scalar, nan=0.0, posinf=0.0, neginf=0.0)
        entropy_vector = np.nan_to_num(entropy_vector, nan=0.0, posinf=0.0, neginf=0.0)
        lengths_vec = np.nan_to_num(lengths_vec, nan=0.0, posinf=0.0, neginf=0.0)
        topological_vec = np.nan_to_num(topological_vec, nan=0.0, posinf=0.0, neginf=0.0)
        image_vec = np.nan_to_num(image_vec, nan=0.0, posinf=0.0, neginf=0.0)
        image_grid = np.nan_to_num(image_grid, nan=0.0, posinf=0.0, neginf=0.0)
        method_reports[str(dim)] = {
            "available": True,
            "finite_interval_count": int(diagram.shape[0]),
            "diagram": _diagram_to_rows(diagram),
            "landscape": {
                "num_landscapes": landscape_layers,
                "resolution": resolution,
                "grid": _round_list(landscape_grid),
                "values": _matrix_rows(landscape_vec.reshape(landscape_layers, resolution)),
                "vector": _round_list(landscape_vec),
                "l2_norm": float(np.linalg.norm(landscape_vec)),
                "l1_norm": float(np.linalg.norm(landscape_vec, ord=1)),
            },
            "betti_curve": {
                "resolution": resolution,
                "grid": _round_list(betti_grid),
                "values": _round_list(betti_vec),
                "area": float(np.trapezoid(betti_vec, betti_grid)) if len(betti_grid) == len(betti_vec) else float(np.sum(betti_vec)),
            },
            "silhouette": {
                "resolution": resolution,
                "grid": _round_list(silhouette_grid),
                "values": _round_list(silhouette_vec),
                "l2_norm": float(np.linalg.norm(silhouette_vec)),
            },
            "entropy": {
                "scalar": _round_list(entropy_scalar),
                "vector": _round_list(entropy_vector),
                "vector_l2_norm": float(np.linalg.norm(entropy_vector)),
            },
            "persistence_lengths": {
                "threshold": topological_vector_threshold,
                "values": _round_list(lengths_vec),
                "sum": float(np.sum(lengths_vec)),
                "max": float(np.max(lengths_vec)) if lengths_vec.size else 0.0,
            },
            "topological_vector": {
                "threshold": topological_vector_threshold,
                "values": _round_list(topological_vec),
                "l2_norm": float(np.linalg.norm(topological_vec)),
            },
            "persistence_image": {
                "resolution": [image_resolution, image_resolution],
                "values": _matrix_rows(image_grid),
                "l2_norm": float(np.linalg.norm(image_vec)),
                "mass": float(np.sum(image_vec)),
            },
        }
        summary[f"dim{dim}_landscape_l2_norm"] = float(np.linalg.norm(landscape_vec))
        summary[f"dim{dim}_betti_area"] = method_reports[str(dim)]["betti_curve"]["area"]
        summary[f"dim{dim}_entropy_scalar_sum"] = float(np.sum(entropy_scalar))
        summary[f"dim{dim}_persistence_length_sum"] = float(np.sum(lengths_vec))
        summary[f"dim{dim}_topological_vector_l2_norm"] = float(np.linalg.norm(topological_vec))
        summary[f"dim{dim}_persistence_image_mass"] = float(np.sum(image_vec))
    summary["landscape_l2_norm"] = float(
        sum(float(row.get("landscape", {}).get("l2_norm", 0.0)) for row in method_reports.values() if isinstance(row, dict))
    )
    summary["entropy_scalar_sum"] = float(
        sum(float(np.sum(row.get("entropy", {}).get("scalar", [0.0]))) for row in method_reports.values() if isinstance(row, dict) and row.get("available"))
    )
    summary["topological_vector_l2_norm"] = float(
        sum(float(row.get("topological_vector", {}).get("l2_norm", 0.0)) for row in method_reports.values() if isinstance(row, dict))
    )
    return {
        "available": any(isinstance(row, dict) and row.get("available") for row in method_reports.values()),
        "backend": "gudhi.representations",
        "finite_interval_count": finite_count,
        "max_dimension": max_dimension,
        "decision_policy": _persistence_vectorization_decision_policy(),
        "diagrams_by_dimension": {str(dim): _diagram_to_rows(diagrams.get(dim, np.zeros((0, 2)))) for dim in range(max_dimension + 1)},
        "methods": method_reports,
        "summary": summary,
        "train_time_use": {
            "fast_features": ["landscape.vector", "betti_curve.values", "entropy.scalar", "persistence_lengths.values", "topological_vector.values"],
            "loss_candidates": [
                "landscape_l2 or cosine distance to successful memory trajectories",
                "persistence_length sparsity/stability penalty",
                "topological_vector contrastive retrieval loss",
                "Betti-curve drift penalty across invalid reasoning branches",
            ],
            "autograd_note": "GUDHI vectorizers are NumPy/scikit-learn transforms in this implementation; use as cached features, rewards, retrieval keys, and diagnostics unless replaced by a torch-native differentiable surrogate.",
        },
    }


def _persistence_vectorization_decision_policy() -> list[dict[str, str]]:
    return [
        {"method": "Landscape", "priority": "primary", "use": "stable vector-space summary for losses, retrieval, averages, and dark-mode curve plots"},
        {"method": "BettiCurve", "priority": "primary", "use": "interpretable rank trace by filtration; cheap train/eval metric"},
        {"method": "PersistenceImage", "priority": "auxiliary", "use": "compact heatmap artifact and convolution-friendly topology feature"},
        {"method": "Silhouette", "priority": "auxiliary", "use": "single weighted landscape summary curve for dashboards"},
        {"method": "PersistenceLengths", "priority": "fast scalar/vector", "use": "cheap persistence mass, max-length, and regularizer features"},
        {"method": "TopologicalVector", "priority": "retrieval", "use": "fixed-length memory-retrieval key and contrastive feature"},
        {"method": "Entropy", "priority": "diagnostic", "use": "topological complexity scalar/vector for W&B and branch pruning"},
        {"method": "ComplexPolynomial", "priority": "defer", "use": "optional experimental signature; lower interpretability for current GoT audit"},
    ]


def _finite_diagrams_by_dimension(intervals: list[Any], max_dimension: int) -> dict[int, np.ndarray]:
    grouped: dict[int, list[list[float]]] = {dim: [] for dim in range(max_dimension + 1)}
    for row in intervals:
        if not isinstance(row, dict):
            continue
        dim = int(row.get("dimension", 0) or 0)
        if dim < 0 or dim > max_dimension:
            continue
        birth = row.get("birth", 0.0)
        death = row.get("death")
        if death is None or row.get("infinite"):
            continue
        try:
            b = float(birth)
            d = float(death)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(b) and np.isfinite(d)) or d <= b:
            continue
        grouped.setdefault(dim, []).append([b, d])
    return {
        dim: np.asarray(rows, dtype=float).reshape((-1, 2)) if rows else np.zeros((0, 2), dtype=float)
        for dim, rows in grouped.items()
    }


def _diagram_to_rows(diagram: np.ndarray) -> list[dict[str, float]]:
    arr = np.asarray(diagram, dtype=float).reshape((-1, 2)) if np.asarray(diagram).size else np.zeros((0, 2), dtype=float)
    return [{"birth": float(row[0]), "death": float(row[1]), "persistence": float(row[1] - row[0])} for row in arr]


def _diagram_scale(diagram: np.ndarray) -> float:
    arr = np.asarray(diagram, dtype=float)
    if arr.size == 0:
        return 1.0
    return float(max(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 0]), np.nanmax(arr[:, 1] - arr[:, 0]), 1e-6))


def _round_list(values: np.ndarray | list[float], decimals: int = 6, max_items: int | None = None) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if max_items is not None:
        arr = arr[:max_items]
    return [_finite_round(value, decimals) for value in arr]


def _matrix_rows(values: np.ndarray, decimals: int = 6) -> list[list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [[_finite_round(value, decimals) for value in row] for row in arr]


def _finite_round(value: float, decimals: int = 6) -> float:
    val = float(value)
    if not np.isfinite(val):
        return 0.0
    return float(round(val, decimals))


def _graph_metrics(filtered_object: dict[str, Any]) -> dict[str, Any]:
    vertices = [row["simplex"][0] for row in filtered_object.get("simplices", []) if int(row.get("dimension", -1)) == 0]
    edges = [row["simplex"] for row in filtered_object.get("simplices", []) if int(row.get("dimension", -1)) == 1]
    try:
        import networkx as nx  # type: ignore

        graph = nx.DiGraph()
        graph.add_nodes_from(vertices)
        graph.add_edges_from((edge[0], edge[1]) for edge in edges if len(edge) == 2)
        undirected = graph.to_undirected()
        weak_components = nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0
        strong_components = nx.number_strongly_connected_components(graph) if graph.number_of_nodes() else 0
        undirected_components = nx.number_connected_components(undirected) if undirected.number_of_nodes() else 0
        cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + undirected_components
        is_dag = nx.is_directed_acyclic_graph(graph)
        longest_path = nx.dag_longest_path_length(graph) if is_dag and graph.number_of_nodes() else None
        return {
            "backend": "networkx",
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "weak_components": weak_components,
            "strong_components": strong_components,
            "is_dag": bool(is_dag),
            "dag_longest_path_length": longest_path,
            "undirected_cycle_rank": int(max(cycle_rank, 0)),
            "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        }
    except Exception as exc:
        return {"backend": "none", "error": f"{type(exc).__name__}: {exc}", "nodes": len(vertices), "edges": len(edges)}


def _stanley_reisner_report(simplices: list[dict[str, Any]], max_nonfaces: int) -> dict[str, Any]:
    vertices = sorted(row["simplex"][0] for row in simplices if int(row["dimension"]) == 0)
    edge_keys = {tuple(row["simplex"]) for row in simplices if int(row["dimension"]) == 1}
    nonfaces = []
    for u, v in combinations(vertices, 2):
        key = tuple(sorted((u, v)))
        if key not in edge_keys:
            nonfaces.append(key)
    samples = nonfaces[:max_nonfaces]
    return {
        "ring": "F2[x_v : v in vertices]",
        "ideal": "Stanley-Reisner ideal of missing faces; degree-2 flag generators displayed",
        "degree_two_generator_count": len(nonfaces),
        "displayed_generator_count": len(samples),
        "truncated": len(nonfaces) > len(samples),
        "generators": [{"support": list(pair), "monomial": _monomial(pair)} for pair in samples],
    }


def _hochster_report(simplices: list[dict[str, Any]], max_vertices: int, max_syzygy_basis: int) -> dict[str, Any]:
    vertices = sorted(row["simplex"][0] for row in simplices if int(row["dimension"]) == 0)
    selected = vertices[:max_vertices]
    simplex_keys = {tuple(row["simplex"]) for row in simplices}
    nonzero = []
    table: Counter[tuple[int, int]] = Counter()
    for size in range(1, len(selected) + 1):
        for subset in combinations(selected, size):
            induced = [
                {"simplex": list(key), "dimension": len(key) - 1, "filtration": 0.0}
                for key in simplex_keys
                if set(key).issubset(subset)
            ]
            by_dim = _simplices_by_dim(induced)
            homology = _homology_report(by_dim, _boundary_reports(by_dim, max_syzygy_basis=max_syzygy_basis))
            for dim_key, value in homology["betti"].items():
                dim = int(dim_key)
                reduced = int(value) - 1 if dim == 0 and induced else int(value)
                if reduced <= 0:
                    continue
                homological_degree = size - dim - 1
                if homological_degree < 0:
                    continue
                table[(homological_degree, size)] += reduced
                nonzero.append(
                    {
                        "homological_degree": homological_degree,
                        "internal_degree": size,
                        "subset": list(subset),
                        "reduced_homology_degree": dim,
                        "value": reduced,
                    }
                )
    return {
        "formula": "beta_{i,W}(F2[K]) = dim_F2 reduced H_{|W|-i-1}(K_W; F2)",
        "vertices_considered": selected,
        "truncated_vertices": len(vertices) > len(selected),
        "nonzero_multigrades": len(nonzero),
        "multigraded_betti": nonzero,
        "betti_table": [
            {"homological_degree": i, "internal_degree": j, "value": value}
            for (i, j), value in sorted(table.items())
        ],
    }


def _taylor_resolution_report(sr: dict[str, Any], max_degree: int = 5) -> dict[str, Any]:
    count = int(sr.get("degree_two_generator_count", 0))
    degrees = []
    for homological_degree in range(0, min(count, max_degree) + 1):
        degrees.append(
            {
                "homological_degree": homological_degree,
                "rank_upper_bound": int(comb(count, homological_degree)) if homological_degree <= count else 0,
            }
        )
    return {
        "kind": "Taylor resolution upper bound",
        "minimal": False,
        "generator_count": count,
        "ranks": degrees,
    }


def _hilbert_poincare_report(simplices: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[tuple[int, int, str]] = Counter()
    for row in simplices:
        dim = int(row["dimension"])
        bucket = int(round(float(row.get("filtration", 0.0)) * 10))
        source = str(row.get("source", "observed"))
        counts[(dim, bucket, source)] += 1
    return {
        "variables": ["simplex_dimension", "filtration_decile", "source_grade"],
        "coefficients": [
            {"dimension": dim, "filtration_decile": bucket, "source": source, "coefficient": count}
            for (dim, bucket, source), count in sorted(counts.items())
        ],
    }


def _dgca_report(by_dim: dict[int, list[tuple[str, ...]]], boundaries: dict[str, Any]) -> dict[str, Any]:
    cup_counts = {"vertex_edge_incidence_products": 0, "edge_edge_triangle_products": 0}
    vertices = by_dim.get(0, [])
    edges = by_dim.get(1, [])
    triangles = by_dim.get(2, [])
    for vertex in vertices:
        v = vertex[0]
        cup_counts["vertex_edge_incidence_products"] += sum(1 for edge in edges if v in edge)
    edge_sets = [set(edge) for edge in edges]
    for triangle in triangles:
        tri = set(triangle)
        cup_counts["edge_edge_triangle_products"] += sum(1 for a, b in combinations(edge_sets, 2) if a | b == tri)
    return {
        "model": "C^*(K; F2) with transpose differential and finite cup-product support counts",
        "cochain_group_ranks": {str(dim): len(rows) for dim, rows in sorted(by_dim.items())},
        "differential_ranks": {
            f"d^{dim - 1}": info["rank"]
            for key, info in boundaries["maps"].items()
            for dim in [int(key[1:])]
        },
        "cohomology_betti": _homology_report(by_dim, boundaries)["betti"],
        "cup_product_support_counts": cup_counts,
    }


def _derived_signature(
    homology: dict[str, Any],
    persistence: dict[str, Any],
    persistence_module: dict[str, Any],
    multiparameter: dict[str, Any],
) -> dict[str, Any]:
    finite_lengths = []
    infinite = 0
    for interval in persistence.get("intervals", []):
        if interval.get("infinite"):
            infinite += 1
        elif interval.get("death") is not None:
            finite_lengths.append(float(interval["death"]) - float(interval["birth"]))
    return {
        "betti_vector": [int(homology["betti"].get(str(dim), 0)) for dim in range(4)],
        "persistence_finite_interval_count": len(finite_lengths),
        "persistence_infinite_interval_count": infinite,
        "persistence_total_finite_length": float(sum(finite_lengths)),
        "rank_profile": [
            {"threshold": state["threshold"], "betti": state["betti"]}
            for state in persistence_module.get("states", [])
        ],
        "multiparameter_h0_rank_sample": [
            {
                "source_grade": row["source_grade"],
                "target_grade": row["target_grade"],
                "h0_rank": row["h0_rank"],
            }
            for row in multiparameter.get("rank_invariant_samples", [])[:16]
        ],
        "multiparameter_grid_points": len(multiparameter.get("fiber_rank_profile", [])),
        "interpretation": "Comparable invariant signature only; equality is not a proof of derived equivalence.",
    }


def _euler_characteristic(by_dim: dict[int, list[tuple[str, ...]]]) -> int:
    return int(sum(((-1) ** dim) * len(rows) for dim, rows in by_dim.items()))


def _rank_mod2(matrix: np.ndarray) -> int:
    _, pivots = _rref_mod2(matrix)
    return len(pivots)


def _nullspace_mod2(matrix: np.ndarray) -> list[np.ndarray]:
    if matrix.size == 0:
        n_cols = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        return [np.eye(n_cols, dtype=np.uint8)[idx] for idx in range(n_cols)]
    rref, pivots = _rref_mod2(matrix)
    n_cols = matrix.shape[1]
    pivot_set = set(pivots)
    basis = []
    for free_col in [idx for idx in range(n_cols) if idx not in pivot_set]:
        vec = np.zeros(n_cols, dtype=np.uint8)
        vec[free_col] = 1
        for row_idx, pivot_col in enumerate(pivots):
            if rref[row_idx, free_col]:
                vec[pivot_col] = 1
        basis.append(vec)
    return basis


def _rref_mod2(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    a = np.array(matrix, dtype=np.uint8, copy=True) % 2
    if a.ndim != 2:
        return np.zeros((0, 0), dtype=np.uint8), []
    rows, cols = a.shape
    pivots: list[int] = []
    pivot_row = 0
    for col in range(cols):
        candidates = np.flatnonzero(a[pivot_row:, col])
        if candidates.size == 0:
            continue
        row = int(candidates[0] + pivot_row)
        if row != pivot_row:
            a[[pivot_row, row]] = a[[row, pivot_row]]
        for other in range(rows):
            if other != pivot_row and a[other, col]:
                a[other] ^= a[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    return a, pivots


def _basis_supports(basis: list[np.ndarray], labels: list[tuple[str, ...]]) -> list[dict[str, Any]]:
    out = []
    for vec in basis:
        support = [list(labels[idx]) for idx, value in enumerate(vec.tolist()) if value]
        out.append({"support_size": len(support), "support": support})
    return out


def _canonical(simplex: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in simplex if str(value) != "")


def _monomial(labels: Iterable[str]) -> str:
    return "".join(f"x_{{{label}}}" for label in labels)
