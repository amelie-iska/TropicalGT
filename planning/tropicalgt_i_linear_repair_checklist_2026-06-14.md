# TropicalGT-I Linear Repair Checklist

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` or a stricter equivalent. Work item-by-item in the order below. Do not jump to a later subsystem while the active item is incomplete. Do not implement broad shotgun edits.

**Goal:** Repair TropicalGT-I so algebraic, topological, tropical, toric, analogical-memory, visualization, training, and paper artifacts are implemented as real computations, displayed truthfully in the browser, documented in the paper as they become concrete, and pushed on the non-main branch.

**Branch:** `tropicalgt-i-real-cas-no-proxy-20260614`

**Remote workspace:** `/home/iska/Documents/amelie/bio/TropicalGT`

**Current active training run:** `tropicalgt_i_pg_bpb_step0_full24b_b55_v11_bpb_5k_gate`; keep it alive until the 5K-step policy or explicit user request requires a restart.

**Execution behavior:** exactly one active repair item at a time. Update this checklist and the browser status before and after each item. Visual QA in `http://127.0.0.1:8990/` is required for every artifact-facing change. No secrets, datasets, checkpoints, W&B run directories, caches, or bulky generated artifacts are staged.

## Non-Negotiable Mathematical Policy

1. No proxy computations may be presented as the requested object.
2. If an object is required and difficult, implement or integrate the real backend for it; do not omit it because it is hard.
3. When a proxy path is encountered during the active item, replace it immediately with a real implementation in that item lane, or render the object unavailable with an explicit implementation target. Do not leave proxy terminology, proxy metrics, or proxy plots silently active.
4. Total-graded, ungraded, or finite-chain diagnostics may be rendered only under those exact names.
5. Multigraded persistence-module free resolutions must be actual multigraded objects over the intended polynomial ring, usually `F2[x_level,x_radius]` for bifiltrations.
6. Analogical maps must be derived from model-predicted probability data and checked against the filtered simplicial complexes and their algebraic invariants.
7. Toric geometry is an auxiliary ambient geometry when correct: tropical schemes/fans may be embedded into toric varieties, following the Maclagan-style viewpoint where the toric variety controls ambient combinatorics. This does not claim the neural network itself is toric. It supplies correct tooling and geometry for tropical schemes, one dimensional cone data, fan strata, vector-bundle-style losses, objectives, regularizers, metrics, diagnostics, sheaves, and toric embeddings when SageMath or Macaulay2 gives a real computation.
8. Use “one dimensional cone” / “one dimensional cones” terminology where the object is a ray or cone of a fan.
9. Update the TropicalGT-I paper as implementation details become real: definitions, theorems, proofs, pseudocode, examples, losses, objectives, regularizers, and metrics must track the implemented algebraic/topological/tropical/toric machinery.

## Sequential Implementation Items

### Item 1: Planning and Browser Status Contract

**Status:** done.

**Files:**
- Modify: `planning/tropicalgt_i_linear_repair_checklist_2026-06-14.md`
- Modify/generated browser status: `TropicalGT-I/outputs/multi_sample_browser/latest/progress_status.json`
- Modify/generated browser entry copy: `TropicalGT-I/outputs/multi_sample_browser/latest/index.html`

**Definition of done:** this file exists, records the current ordered checklist, records the toric-embedding and paper-update requirements accurately, and the browser progress status points to the current active implementation target.

**Latest behavior update:** continue linearly through the active item; when a proxy or proxy-labeled artifact is encountered inside that item, replace it with a real implementation path immediately or render it as unavailable with the exact missing backend named. Do not switch subsystems to chase adjacent issues until the current item has tests and browser evidence.

### Item 2: Real CAS Multigraded Free-Resolution Contract

**Status:** done.

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/cas_free_resolution.py`
- Modify: `TropicalGT-I/src/tropicalgt/algebra.py`
- Test: `TropicalGT-I/tests/test_algebraic_persistence.py`

**Required implementation:**
- Add explicit certificate fields distinguishing real CAS output from multigraded free-resolution output:
  - `real_free_resolution_certified`
  - `total_graded_resolution_certified`
  - `ungraded_resolution_certified`
  - `multigraded_free_resolution_certified`
  - `safe_to_render_as_total_graded_resolution`
  - `safe_to_render_as_multigraded_free_resolution`
- Multi-parameter persistence modules over `F2[x_level,x_radius]` must require `multigraded_free_resolution_certified=True` before rendering as a multigraded free resolution.
- If the current backend can only return total-graded or ungraded information, preserve and render that real output under its actual grading and add a clear implementation-needed status for the multigraded resolution backend.
- Add or preserve tests proving total-graded CAS output cannot be mislabeled as multigraded persistence-module output.

**Definition of done:** focused algebra tests pass, and browser status explicitly names the CAS gate result.

### Item 3: Real CAS Backend Expansion

**Status:** active.

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/cas_free_resolution.py`
- Possibly create: `TropicalGT-I/src/tropicalgt/cas_backends/`
- Planning update: `planning/tropicalgt_i_cas_free_resolution_implementation_plan_2026-06-14.md`

**Required implementation:**
- Probe and integrate available real backends in this order: Macaulay2, SageMath, Singular.
- Evaluate `amelie-iska/BEMultipliers.git` for Buchsbaum-Eisenbud multiplier diagnostics.
- For Macaulay2, support real commands for multigraded modules, Betti tables, resolutions, Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, and the `Tropical` package where relevant.
- For SageMath, support polynomial-ring, ideal, syzygy/free-resolution, tropical polynomial, and tropical variety interfaces where real and available.
- If a backend is unavailable, emit an actionable unavailable status, not a proxy.

**Definition of done:** a backend capability report is generated and real backend code paths are used wherever available.

### Item 4: Analogical Memory and Derived/Module Maps

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/memory.py`
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: relevant memory/topology tests

**Required implementation:**
- Retrieval uses model-predicted probability vectors and Jensen-Shannon assignment between probability-vector filtered complexes.
- Validate simplicial-map preservation on vertices, edges, and faces.
- Add simplex-tree-level maps between query and retrieved memory trajectories.
- Derived/free-resolution similarity must be computed from real algebraic evidence. If no real resolution is available, render “unavailable pending CAS backend,” not a numeric fake.
- Memories enter the bank only above the configured quality threshold. Early training should show insufficient-memory states rather than fabricated analogies.

**Definition of done:** analogical top-k and map pages render many valid analogies when memory is populated, or a truthful unavailable state when it is not.

### Item 5: Simplicial Complex and Simplex Tree Visuals

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Modify tests for visualization payloads if present

**Required implementation:**
- Radius filtration starts as disjoint vertices and grows min-to-max by adding edges/faces.
- Solid simplex edges/faces are controlled by radius sliders.
- Dotted directed edges show causal structure and/or decoding order and are controlled by reasoning/decoding sliders.
- Every reasoning step has its own filtered complex and actual simplex tree.
- Simplex-tree pages show a genuine trie or face-coface poset, not disconnected columns.

**Definition of done:** browser inspection confirms sliders, dotted decoding edges, and simplex tree layout are interpretable.

### Item 6: Two-Parameter Persistence and Lattice Rendering

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/algebra.py`
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`

**Required implementation:**
- Every nonempty trajectory emits a bifiltration over `F2[x_level,x_radius]`.
- Compute grid fibers `K_(level,radius)`, structure maps, rank invariants, Betti surfaces, and JSON payloads.
- Render a 3D `(level, radius, rank/Betti)` lattice view and a separate algebraic module view with multidegrees.

**Definition of done:** browser plots show a correct lattice/grid rather than flat or misleading strips.

### Item 7: NLL Density and Energy/Fitness Landscape

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`

**Required implementation:**
- Add a 3D PCA Gaussian density cloud around actual model embedding vectors; generated cloud samples are visualization support only and must not be rendered as model states.
- Color local density by NLL and keep actual GoT vertices/edges visible.
- Keep 2D/3D NLL surfaces only when they cover the trajectory neighborhood and explicitly anchor observed vertices.

**Definition of done:** browser plot reads as a continuous density/energy object rather than a flat simplex.

### Item 8: Tropical/Toric Embedding Tooling and Paper Integration

**Files:**
- Modify: vector-bundle and toric/tropical planning files
- Modify: `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex`
- Implement after CAS foundations are stable.

**Required implementation:**
- Review and use Macaulay2 `Tropical` commands where available: `tropicalVariety`, `tropicalPrevariety`, `tropicalCycle`, `BergmanFan`, `fan`, `rays`, `cones`, `maxCones`, `linealitySpace`, `multiplicities`, `isBalanced`, `isPure`, `isSimplicial`, `stableIntersection`, `isTropicalBasis`, and `visualizeHypersurface`.
- Review and use SageMath tropical polynomial and tropical variety APIs where available.
- Correctly embed tropical schemes/fans attached to model embedding data into toric varieties when that improves real computation, losses, objectives, regularizers, metrics, or diagnostics. Track one dimensional cone data, fan strata, lineality, multiplicity, balancing, stable intersections, sheaves, vector bundles, and derived/scheme-theoretic terminology where it is mathematically doing work.
- Update the paper with the actual construction, not aspirational filler.

**Definition of done:** paper/planning/code explain and compute toric ambient data and show how it enters losses, objectives, regularizers, metrics, and diagnostics without claiming the network is intrinsically toric.

### Item 9: Training and Metric Stabilization

**Files:**
- Modify config after current item sequence or 5K policy indicates.
- Do not interrupt active training unless explicitly requested or policy triggers.

**Required implementation:**
- Optimize BPB and graph BPB with advanced techniques already implemented.
- Keep periodic visual audits every 250 steps.
- Monitor tropical margin sign/scale, certificate loss, GraphCG full-rank directions, wall-hit rate, memory quality, and BPB.

**Definition of done:** restart only per policy/request, record command/log/W&B run, and keep training alive.

### Item 10: Docs, Browser QA, Commit, Push

**Files:**
- Modify: `README.md` and relevant planning files after code changes.

**Required implementation:**
- After each item, run focused tests, update browser status, inspect `http://127.0.0.1:8990/`, and commit/push safe files to the non-main branch.

**Definition of done:** pushed non-main branch contains only safe source/planning/docs/reference changes.
