from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, product
from math import comb, isinf
from typing import Any, Iterable

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
    multiparameter = _multiparameter_module_report(
        closed["simplices"],
        max_grid=max_multiparameter_grid,
        max_pairs=max_rank_invariant_pairs,
    )
    graph_metrics = _graph_metrics(filtered_object)
    persistence = _persistent_homology_report(closed["simplices"], ph_backend)

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
        "persistence_module": persistence_module,
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
        report["commutative_algebra"] = {
            "multigraded_hilbert_poincare": _hilbert_poincare_report(closed["simplices"]),
            "stanley_reisner": sr,
            "hochster_betti": hochster,
            "taylor_resolution_upper_bound": _taylor_resolution_report(sr),
            "boundary_syzygies": boundaries["syzygies"],
            "dg_commutative_algebra": dgca,
            "multiparameter_free_resolution_proxy": _multiparameter_free_resolution_proxy(multiparameter),
            "notes": [
                "Boundary syzygies are F2 nullspaces of displayed boundary maps.",
                "Taylor ranks are nonminimal upper bounds for the displayed Stanley-Reisner generators.",
                "Hochster Betti data is exact on the bounded vertex subset named in the report.",
                "Multigraded chain modules and monomial boundary labels are exact for the displayed finite multi-filtered complex over F2[x1,x2,x3].",
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


def _multiparameter_free_resolution_proxy(multiparameter: dict[str, Any]) -> dict[str, Any]:
    generators = multiparameter.get("chain_module_generators", [])
    by_degree: Counter[int] = Counter(int(row.get("homological_degree", 0)) for row in generators)
    boundary_counts = {
        key: len(value)
        for key, value in multiparameter.get("boundary_monomials", {}).items()
        if isinstance(value, list)
    }


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
    names = ["x_filtration", "x_dimension", "x_position"]
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
    return {
        "ring": multiparameter.get("coefficient_ring"),
        "free_chain_modules": [
            {"homological_degree": degree, "rank": count}
            for degree, count in sorted(by_degree.items())
        ],
        "monomial_labeled_boundary_entry_counts": boundary_counts,
        "interpretation": "A multigraded free chain complex over the multiparameter polynomial ring; minimal free resolutions of homology modules require a CAS/backend.",
    }


def _multigraded_boundary_monomials(enriched: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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
                    "monomial": _multi_monomial(exponent),
                }
            )
    return dict(out)


def _h0_rank_invariant_samples(
    enriched: list[dict[str, Any]],
    grid_points: list[tuple[int, int, int]],
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
