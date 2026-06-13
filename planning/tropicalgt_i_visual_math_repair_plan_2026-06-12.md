# TropicalGT-I Visualization and Algebra Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace misleading or duplicated TropicalGT-I audit artifacts with model-backed, mathematically faithful interactive plots and metrics for graph-of-thought reasoning, radius filtrations, two-parameter persistence, simplex trees, analogical retrieval, GraphCG, and tropical support.

**Architecture:** The repair is evidence-first: inspect the generated JSON payloads and model telemetry before changing renderers, then patch computation contracts and visualizations so every displayed object is traceable to model outputs, embeddings, probabilities, graph token traces, or GUDHI/multipersistence computations. Artifacts that cannot be computed from real payloads must render as unavailable with a precise reason rather than falling back to synthetic geometry.

**Tech Stack:** Python, PyTorch, Plotly, GUDHI SimplexTree/persistence APIs, optional multipers when installed, W&B, local Codex browser QA, Tailscale SSH remote execution only.

---

## Current Live-Run Reset

- [ ] Stop the existing `train_full_dataset_pg_bpb_step0_full24b_b44` run because it was launched before the latest memory-quality and visualization contract fixes.
- [ ] Start a fresh step-0 run from a new config with a BPB-first objective, lower auxiliary/certificate pressure, stochastic/deeper GoT audit sampling, and enough batch size to keep RTX 4090 VRAM above 18GB without OOM.
- [ ] Keep W&B and periodic artifacts active, but do not stage checkpoints, data, W&B run state, or browser export directories.

## Defect Inventory From Browser and Photo Annotations

- [ ] **Identical reasoning-step complexes:** Audit whether `reasoning_step_complex_maps/reasoning_step_*.html` reuse a trajectory-level complex, a static probability complex, or duplicated graph-token embeddings. Each reasoning step must own a distinct vertex set from that step's model output, hidden graph-state embeddings, token probabilities, and graph-token trace.
- [ ] **Extraneous duplicate selected complex panels:** Remove static duplicate bottom panels. A selected-complex panel may appear only as a click/tap inspector tied to a specific node/simplex and must say which reasoning step or token generated it.
- [ ] **Radius filtration direction:** Sliders must move min-to-max left-to-right and begin as a disjoint cloud of 0-simplices, then add edges and faces as radius grows. Every edge/face must come from GUDHI SimplexTree filtration values or the recorded model-probability Jensen-Shannon filtration.
- [ ] **Directed causal/order overlays:** Full trajectory complexes and probability complexes must render dotted directed edges for causal DAG order, decoding order, and forward/reverse meet-in-the-middle directions when present. Non-causal or cyclic graphs must use ROAR/random-order metadata instead.
- [ ] **NLL/fitness landscape:** The GoT NLL page must not call a sparse triangulation an energy landscape. A valid landscape needs either model-evaluated local anchors around the trajectory or a clearly labeled observed-anchor interpolation with uncertainty. Trajectory points must lie on the surface when projected in `(PC1, PC2, NLL)` coordinates.
- [ ] **Simplex tree plot is not a connected trie/Hasse diagram:** Replace the current disconnected vertical stripes with a connected graph whose nodes are simplices and whose directed edges are immediate face-to-coface inclusions from the GUDHI simplex tree. Coordinates must encode simplex dimension, filtration value, and trie sibling order without inventing disconnected components.
- [ ] **2-parameter persistence over `F2[x_level,x_radius]`:** Replace the surface-only view with a lattice/staircase module view. Use bidegrees `(level, radius_bin)`; display vector-space ranks on grid cells, horizontal/vertical multiplication maps, minimal generator candidates, adjacent lcm syzygy candidates, and Hilbert/rank invariant summaries.
- [ ] **Macaulay2-style free resolutions and derived objects:** Add research-figure renderings for finite graded chain/free-resolution data: Betti tables with homological degree columns and multidegree rows, free modules such as `F_0 = ⊕ S(-a_i,-b_i)`, differentials as sparse monomial matrices over `F2[x_level,x_radius]`, staircase/lcm syzygy diagrams, and chain-map/derived-morphism diagrams between query and memory objects. If exact minimality or derived equivalence is not certified, label the figure as a computed candidate/witness rather than a proof.
- [ ] **Miller-Sturmfels staircase model:** The `k[x,y]` view should follow Chapter 3.1's staircase diagram for monomial ideals in two variables: minimal generators form an antichain, the staircase separates occupied/unoccupied monomial regions, adjacent lcms identify first syzygies, and Hilbert-series/rank summaries come from inclusion-exclusion over the staircase. Do not present non-minimal or heuristic signatures as exact minimal free resolutions.
- [ ] **Free resolutions and derived similarity overclaiming:** If PH similarity or free-resolution similarity is zero, derived/algebraic similarity must not be high. Rename any coarse vector cosine to `signature cosine`; reserve `derived/algebraic similarity` for conservative checks over Betti tables, multigraded generators/syzygies, rank invariants, and chain-map/simplicial-map preservation.
- [ ] **Analogical top-k:** Retrieval must list multiple memories above the memory-quality gate, show unavailable state early in training, and visualize probability-matched correspondences only when model probabilities exist. The top-k index needs a readable table plus one selected map view, not a jammed full graph.
- [ ] **Analogical simplicial maps:** A displayed map must check vertex assignment, edge preservation, face preservation, filtration monotonicity, and probability/JS assignment cost. If preservation fails, render it as a correspondence rather than a simplicial map.
- [ ] **Simplex-tree analogies:** Add a comparison of query and memory simplex-tree Hasse/trie diagrams, with preserved face/coface chains highlighted. Treat this separately from geometric PCA views.
- [ ] **Persistence landscapes:** Render actual GUDHI landscape vectors with legible small multiples or tabs. Avoid overlapping titles, unreadable heatmaps, and hover-only explanations.
- [ ] **Tropical support heatmap:** Improve compact labels, split support strip from margin/profile panels, show token categories and support collapse diagnostics without overlapping axes or legends.
- [ ] **GraphCG directions:** Re-layout full-rank direction spectra, top active directions, candidate activity, and signed bias into separate coordinated panels. Labels must be compact and hover must contain full path/action text.

## Implementation Tasks

### Task 1: Evidence Audit of Generated Payloads

**Files:**
- Inspect: `TropicalGT-I/outputs/**/sample_*/**/*.json`
- Inspect: `TropicalGT-I/src/tropicalgt/visualization.py`
- Inspect: `TropicalGT-I/src/tropicalgt/scaling.py`
- Inspect: `TropicalGT-I/src/tropicalgt/simplicial.py`

- [ ] Locate the active browser artifact source served on port 8990.
- [ ] Hash every reasoning-step complex payload for `sample_001` and `sample_002` and report whether they are identical.
- [ ] Confirm whether per-step complexes use per-candidate embeddings/probabilities or a reused trajectory/global object.
- [ ] Confirm whether analogical top-k rows have real probability vectors and memory-quality gate metadata.
- [ ] Confirm which plots still contain `fallback`, `synthetic`, `proxy`, or `surrogate` strings and either replace them with real computation or render unavailable.

### Task 2: Training Config Reset

**Files:**
- Create: `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b48_v2.json`
- Update: `planning/tropicalgt_i_takeover_repair_log_2026-06-12.md`

- [ ] Copy the prior full-dataset config.
- [ ] Set `run_name` and `output_dir` to `tropicalgt_i_pg_bpb_step0_full24b_b48_v2` / `TropicalGT-I/outputs/train_full_dataset_pg_bpb_step0_full24b_b48_v2`.
- [ ] Set `batch_size=48`, `lr=0.00022`, `weight_decay=0.03`, `grad_clip=0.75`.
- [ ] Reduce `certificate_weight` to `0.0001`; keep certificate metrics logged.
- [ ] Keep `gflownet_weight=0.01`; set `graphcg_weight=0.006` to reduce non-BPB pressure while retaining full-rank diagnostics.
- [ ] Set `periodic_viz_scale_depth=8`, `periodic_viz_scale_width=12`, `periodic_viz_scale_branch_factor=5`, `periodic_viz_scale_stochastic_actions=true`, `periodic_viz_scale_sampling_temperature=1.25`, `periodic_viz_scale_sampling_exploration=0.35`.
- [ ] Set `inference_scaling.stochastic_actions=true`, `sampling_temperature=1.20`, `sampling_exploration=0.30`.
- [ ] Leave `meet_in_middle.enabled=false` by default but keep the config toggle present.
- [ ] Launch with `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/train_tropicalgt_i.py --config TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b48_v2.json`.

### Task 3: Correct Radius and Probability Complex Rendering

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Modify: `TropicalGT-I/src/tropicalgt/simplicial.py`
- Test: `TropicalGT-I/tests/test_visualization_artifacts.py`

- [ ] Add tests proving the first slider frame contains only vertices for a Vietoris-Rips radius filtration.
- [ ] Add tests proving later slider frames monotonically add edges/faces without deleting prior simplices.
- [ ] Add dotted directed overlays from decoding/causal metadata when present.
- [ ] Remove static duplicate selected-complex panels; replace them with click-linked inspector metadata.

### Task 4: Correct Simplex Tree Rendering

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_visualization_artifacts.py`

- [ ] Build a simplex-tree graph where each node is a simplex tuple and each edge is an immediate face-to-coface inclusion.
- [ ] Layout coordinates: x=simplex dimension, y=filtration, z=trie sibling/order or connected-component index.
- [ ] Ensure every non-vertex simplex has at least one parent edge to a codimension-one face.
- [ ] Render connected Hasse/trie edges as lines and do not show disconnected vertical stripes unless the complex itself has isolated vertices at radius zero.

### Task 5: Correct `F2[x_level,x_radius]` 2-Parameter Module View

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/algebra.py`
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_algebraic_metrics.py`

- [ ] Compute a bidegree grid over `(trajectory_level, radius_bin)`.
- [ ] Compute vector-space ranks `beta_0`, `beta_1`, and rank-invariant samples at each grid point from the filtered complexes.
- [ ] Derive generator candidates as bidegrees where rank appears relative to west/south predecessors.
- [ ] Derive adjacent lcm/syzygy candidates from incomparable adjacent generator bidegrees, following the two-variable monomial ideal staircase pattern.
- [ ] Render a 3D lattice/staircase plot: x=level, y=radius, z=rank or homological degree, with east/north module maps and generator/syzygy markers.
- [ ] Label the algebraic output conservatively as computed invariants/candidates unless minimality is verified.

### Task 6: Correct Analogical Retrieval and Derived/Resolution Similarity

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/memory.py`
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_metrics_and_memory.py`

- [ ] Enforce memory-quality gates before retrieval display.
- [ ] For each retrieved memory, compute probability-JS vertex assignment from model probabilities.
- [ ] Check edge, face, and filtration preservation before calling it a simplicial map.
- [ ] Compute conservative `derived_algebraic_similarity = min(ph_similarity, free_resolution_similarity, rank_invariant_similarity, chain_map_score)`.
- [ ] Move any embedding/probability cosine into a separate `signature_cosine` column.
- [ ] Add a simplex-tree analogy view with preserved face/coface paths.

### Task 7: Browser QA and Push

**Files:**
- Update: `README.md`
- Update: `TropicalGT-I/README.md`
- Update: `planning/tropicalgt_i_takeover_repair_log_2026-06-12.md`

- [ ] Regenerate a model-backed audit bundle from real payloads.
- [ ] Serve it in the browser on port 8990 or update the current server root.
- [ ] Inspect `sample_001` and `sample_002` pages in the Codex browser and capture screenshot evidence for NLL landscape, full radius complex, probability complex, simplex tree, two-parameter module lattice, analogical top-k, GraphCG, and tropical support.
- [ ] Run focused tests with `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_visualization_artifacts.py TropicalGT-I/tests/test_algebraic_metrics.py TropicalGT-I/tests/test_metrics_and_memory.py -q`.
- [ ] Commit and push to `origin/tropicalgt-i-implementation`.

## Acceptance Criteria

- Reasoning-step complexes are not identical unless their underlying model payloads are byte-identical and the page says so explicitly.
- All topology/algebra plots are computed from model outputs, embeddings, probabilities, graph traces, or GUDHI/multipersistence data. No synthetic fallback is rendered as data.
- Radius filtrations begin with disjoint vertices and grow monotonically.
- The simplex tree visualization is a connected inclusion graph/trie/Hasse diagram over actual simplices.
- The two-parameter plot presents an `F2[x_level,x_radius]` grid/lattice module with maps, ranks, generator candidates, and syzygy candidates.
- Analogical maps are only called simplicial maps when preservation checks pass; otherwise they are correspondences with explicit failure diagnostics.
- Browser artifacts are visibly more readable: no jammed labels, no overlapping colorbars, no duplicate static panels.
- A fresh step-0 training run is alive with tuned BPB-first hyperparameters and >18GB VRAM usage without OOM.
