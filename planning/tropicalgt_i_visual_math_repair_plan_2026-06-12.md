# TropicalGT-I Visualization and Algebra Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace misleading or duplicated TropicalGT-I audit artifacts with model-backed, mathematically faithful interactive plots and metrics for graph-of-thought reasoning, radius filtrations, two-parameter persistence, simplex trees, analogical retrieval, GraphCG, and tropical support.

**Architecture:** The repair is evidence-first: inspect the generated JSON payloads and model telemetry before changing renderers, then patch computation contracts and visualizations so every displayed object is traceable to model outputs, embeddings, probabilities, graph token traces, or GUDHI/multipersistence computations. Artifacts that cannot be computed from real payloads must render as unavailable with a precise reason rather than falling back to synthetic geometry.

**Tech Stack:** Python, PyTorch, Plotly, GUDHI SimplexTree/persistence APIs, optional multipers when installed, W&B, local Codex browser QA, Tailscale SSH remote execution only.

---

## Current Live-Run Reset

- [ ] Stop the existing active run only after the replacement command and config are ready. Current target to replace: `train_full_dataset_pg_bpb_step0_full24b_b48_v3_250viz`, PID observed as `1625054` on 2026-06-13.
- [ ] Start a fresh step-0 run from a new BPB-first config with OpenAI Parameter-Golf BPB and graph-BPB as the controlling metrics, while keeping tropical ring attention, long-context graph-token training, GFlowNet GoT search, GraphCG, persistence/vector topology, memory retrieval, meet-in-the-middle toggles, and ROAR available as auxiliary mechanisms.
- [ ] Retune auxiliary weights so they support BPB rather than dominate it: reduce or gate certificate/tropical penalties when they rise, keep GraphCG full-rank diagnostics but low-pressure, keep memory retrieval quality-gated, and log all advanced objectives as ablation signals.
- [ ] Use hybrid graph-structured training data: OAI Parameter-Golf rows encoded as graphs plus HuggingFace/GoT/CoT/ToT graph-structured rows, with causal DAG decoding when available and ROAR/random-order decoding for cyclic or noncausal graphs.
- [ ] Keep W&B and periodic artifacts active every 250 steps, keep RTX 4090 VRAM utilization high without OOM, and do not stage checkpoints, data, W&B run state, or browser export directories.

## Defect Inventory From Browser and Photo Annotations

- [ ] **Identical reasoning-step complexes:** Audit whether `reasoning_step_complex_maps/reasoning_step_*.html` reuse a trajectory-level complex, a static probability complex, or duplicated graph-token embeddings. Each reasoning step must own a distinct vertex set from that step's model output, hidden graph-state embeddings, token probabilities, and graph-token trace.
- [ ] **Extraneous duplicate selected complex panels:** Remove static duplicate bottom panels. A selected-complex panel may appear only as a click/tap inspector tied to a specific node/simplex and must say which reasoning step or token generated it.
- [ ] **Radius filtration direction:** Sliders must move min-to-max left-to-right and begin as a disjoint cloud of 0-simplices, then add edges and faces as radius grows. Every edge/face must come from GUDHI SimplexTree filtration values or the recorded model-probability Jensen-Shannon filtration.
- [ ] **Directed causal/order overlays:** Full trajectory complexes and probability complexes must render dotted directed edges for causal DAG order, decoding order, and forward/reverse meet-in-the-middle directions when present. Non-causal or cyclic graphs must use ROAR/random-order metadata instead.
- [ ] **Line semantics across all simplicial objects:** Dotted lines are reserved for decoding/causal/order metadata. Solid lines and filled faces are reserved for radius-filtered simplices and must appear/disappear only through the radius/reasoning/decoding-step slider contract.
- [ ] **NLL/fitness landscape:** The GoT NLL page must not call a sparse triangulation an energy landscape. A valid landscape needs either model-evaluated local anchors around the trajectory or a clearly labeled observed-anchor interpolation with uncertainty. Trajectory points must lie on the surface when projected in `(PC1, PC2, NLL)` coordinates.
- [ ] **NLL density cloud continuity:** Repair `got_nll_density_cloud_pca_3d.html` so Gaussian neighborhoods around actual 3D PCA graph-state/token embeddings form a continuous local NLL density/fitness field. Generated Gaussian support vectors are interpolation samples only and must not be rendered as extra model states; show actual model vertices and trajectory/dotted decoding edges separately.
- [ ] **Simplex tree plot is not a connected trie/Hasse diagram:** Replace the current disconnected vertical stripes with a connected graph whose nodes are simplices and whose directed edges are immediate face-to-coface inclusions from the GUDHI simplex tree. Coordinates must encode simplex dimension, filtration value, and trie sibling order without inventing disconnected components.
- [ ] **2-parameter persistence over `F2[x_level,x_radius]`:** Replace the surface-only view with a lattice/staircase module view. Use bidegrees `(level, radius_bin)`; display vector-space ranks on grid cells, horizontal/vertical multiplication maps, minimal generator candidates, adjacent lcm syzygy candidates, and Hilbert/rank invariant summaries.
- [ ] **2-parameter module-fiber plot defect:** Repair `trajectory_persistence/persistence_module_betti.html` / module-fiber views that currently render as flat or degenerate yellow/blue sheets. The figure must show actual `F2[x_level,x_radius]` grid fibers, ranks/Betti values, east/north structure maps, generator/syzygy markers, and a Miller-Sturmfels staircase or monomial-lattice interpretation when applicable; no decorative rank plane is acceptable.
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
- [ ] Commit and push to `origin/tropicalgt-i-real-cas-no-proxy-20260614` or its newer non-main successor.

## Acceptance Criteria

- Reasoning-step complexes are not identical unless their underlying model payloads are byte-identical and the page says so explicitly.
- All topology/algebra plots are computed from model outputs, embeddings, probabilities, graph traces, or GUDHI/multipersistence data. No synthetic fallback is rendered as data.
- Radius filtrations begin with disjoint vertices and grow monotonically.
- The simplex tree visualization is a connected inclusion graph/trie/Hasse diagram over actual simplices.
- The two-parameter plot presents an `F2[x_level,x_radius]` grid/lattice module with maps, ranks, generator candidates, and syzygy candidates.
- Analogical maps are only called simplicial maps when preservation checks pass; otherwise they are correspondences with explicit failure diagnostics.
- Browser artifacts are visibly more readable: no jammed labels, no overlapping colorbars, no duplicate static panels.
- A fresh step-0 training run is alive with tuned BPB-first hyperparameters and >18GB VRAM usage without OOM.

## Paper-Theory Workstream

- [x] A dedicated subagent reviewed `references/2405.03505v1.pdf` and `references/2009.03030v2.pdf` in full.
- [x] Updated `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` with tropical vector-bundle and toric-embedding background, definitions, theorem/proposition material, examples, and pseudocode connecting those papers to BPB-oriented TropicalGT-I training.
- [x] Developed implementable training techniques from that theory: monomial chart transports, cocycle defects, matroid flat-rank defects, tropical toric max-linear embedding regularization, chart BPB consistency, and transported persistence-landscape memory metrics.
- [x] Created `planning/tropicalgt_i_vector_bundle_toric_embedding_training_plan.md` describing implementation hooks, losses, metrics, visualizations, tests, W&B telemetry, inference outputs, and research risks.
- [ ] Fold the subagent findings into code/configs after review: chart ids, monomial transport heads, toric active rows, BPB promotion gates, and persistence-landscape memory metrics are still implementation tasks.



### Step-0 BPB-Focused Restart Evidence (2026-06-14)

- Previous run reviewed: `train_full_dataset_pg_bpb_step0_full24b_b48_v3_250viz`, W&B `8wi645zu`. Recent validation remained above the `<1.18` BPB target around steps 21000--22750 (`bpb` roughly 1.202--1.216), and periodic interactive artifact generation was disabled in the v3 config.
- New run config: `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b52_v4_bpb_restart.json`.
- New run name: `tropicalgt_i_pg_bpb_step0_full24b_b52_v4_bpb_restart`; W&B run id `ml3j81za`.
- Restart policy: step 0, batch size 52, sequence length 1024, max steps 454455, configured token slots 24,198,819,840, meeting the full-dataset token-slot guard with a 23,552-token ceiling margin.
- Active-training intent: BPB and graph-BPB are primary; advanced methods remain active or observable but have lower auxiliary weights. Periodic interactive artifacts are enabled every 250 steps with one model-output sample bundle. Meet-in-the-middle decoding is enabled with tiny symmetric-KL/reverse-NLL auxiliary weights and `max_records=2` to avoid destabilizing BPB.
- VRAM target: the initial run occupies about 21.6GB/24.6GB, satisfying the requested >18GB utilization without targeting OOM.


### Fresh b59 BPB Gate and Restart Worker (2026-06-16)

- b58 was stopped at user request after reaching validation step `8750` with BPB `1.331143465780304`, graph-BPB `19.24880762755732`, NLL `0.9226783402264118`, and invalid graph rate `0.0`; BPB `<1.12` remained open.
- b59 was launched fresh from step 0 after source commit `70c2e93` added certified-CAS retrieval telemetry to the training loop. Run name: `tropicalgt_i_pg_bpb_step0_full24b_b59_20260616T001122Z_fresh_bpb112_5k_gate`; PID `73189`; W&B id `itxgxj40`; output dir `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b59_20260616T001122Z_fresh_bpb112_5k_gate`.
- b59 readiness passed with CUDA dry-run BPB `7.969144958556688`, graph-BPB `33.58283549642874`, and no failed gates. Early live log review reached step `112` with latest train loss/NLL around `1.748/1.724`, no traceback, and no OOM.
- Galileo (`019ecdc9-a3b8-7761-b0e5-36e6adad8e15`) now owns the 5K evidence loop: monitor to at least step `5000`, run analyses and visualization sidecars, review advanced metrics plus topological/geometric/algebraic artifacts, then create and launch an evidence-backed step-0 restart with adjusted BPB-first hyperparameters/configs.



### Integration Update: v5 Restart, Paper Workstream, and Remaining CAS/Visualization Repairs (2026-06-14)

- Paper/theory sidecar status: Avicenna completed the requested review of `references/2405.03505v1.pdf` and `references/2009.03030v2.pdf`, updated `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex`, and wrote `planning/tropicalgt_i_vector_bundle_toric_embedding_training_plan.md`. The paper now contains a TropicalGT chart-bundle definition, scoped max-linear exponent-chart sidecar definition, monomial transport principle, matroid-filtration compatibility theorem, BPB chart-consistency proposition, proof sketches, examples, and pseudocode. Remaining paper risk: the remote has no LaTeX engine on PATH, so full compilation is still pending.
- Training status: v4 `tropicalgt_i_pg_bpb_step0_full24b_b52_v4_bpb_restart` crashed at the first periodic audit because full reasoning audit required `graph_token_trace_complete` for every generated reasoning step. That crash is a visualization/artifact completeness failure, not a model OOM. The replacement v5 config is `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b52_v5_bpb_restart.json`, W&B run `pah6uswd`, PID `1982070`, using about 21.6GB VRAM. The v5 config keeps strict model-output artifacts but changes missing trace behavior to render unavailable/diagnostic instead of crashing the BPB run.
- Periodic-artifact status: v5 emitted `periodic/step_00000250` with reasoning, GraphCG, metrics, and validation report artifacts. It currently emits the lighter periodic training audit, not the full sample-browser catalog with all richer pages. Required repair: periodic evaluation should produce or link the same sample-based catalog at every 250 steps while keeping the training loop alive.
- Browser QA status: `127.0.0.1:8990/browser_index.html` is visible and points at the richer sample-based browser catalog. The catalog distinguishes `GoT observed NLL PCA anchors`, `GoT NLL density cloud (3D PCA)`, `Persistence landscape lambda_k(t)`, `2-parameter F2[x_level,x_radius] bifiltration`, `Full radius SimplexTree poset`, `Probability SimplexTree poset`, GraphCG, and tropical support. Current code repairs make the simplex-tree view connected through a root/prefix backbone and add a Gaussian NLL density cloud, but further visual polish and validation remain required.
- NLL visualization repairs still required: the 3D PCA density cloud should look more like a continuous electron-density/fitness field around actual model embedding vertices, with support samples rendered only as interpolation density and never as extra reasoning states. The sparse triangulated NLL surface must remain diagnostic/legend-only unless model-evaluated local anchors make a true surface.
- Simplicial repairs still required: all radius-filtered views need dotted causal/decoding-order overlays and solid radius-gated simplices. Every reasoning step must own a distinct model-derived complex. Sliders must begin at the disjoint vertex cloud and grow min-to-max.
- CAS methodology required for real resolutions: use Macaulay2 `res`, `syz`, `betti`, `mingens`, `fittingIdeal`, `minors`, `KustinMiller::resBE`, and `MultiplierIdeals` where applicable; use SageMath/Singular for polynomial ideals and syzygies when M2 is unavailable; evaluate `amelie-iska/BEMultipliers.git` only as an optional Buchsbaum-Eisenbud multiplier backend, not as a substitute for a certified minimal free resolution. If no backend emits a certificate, render `unavailable_no_certificate` and show chain-presentation diagnostics only.
- 2-parameter module repair still required: compute actual `F2[x_level,x_radius]` grid fibers `K_(level,radius)`, east/north structure maps, rank invariant samples, Betti surfaces, staircase generator candidates, adjacent-lcm syzygy candidates, and downloadable JSON. The rendered figure should resemble a multigraded module/staircase research figure rather than a decorative sheet.
- Derived/analogical repair still required: analogical retrieval must use model-predicted probability vectors and Jensen-Shannon assignments, then validate vertex, edge, face, and filtration preservation before calling anything a simplicial map. Derived/algebraic similarity must be conservative and consistent with PH, rank-invariant, chain-map, and real-resolution evidence; high derived similarity with zero resolution similarity is disallowed unless explicitly labeled as a different coarse signature metric.
- Vector-bundle/toric implementation tasks now queued: add chart ids and overlaps, `MonomialTransportHead`, `BundleMatroidHead`, `ToricEmbeddingHead`, zero-default coefficients, W&B metrics, chart-overlap browser panels, and BPB/graph-BPB promotion gates. These should remain auxiliary and must not be promoted if validation BPB or graph-BPB regresses.

### Paper-Theory Repair Addendum

- [ ] Add the vector-bundle ablation matrix to the active repair goal: zero auxiliary, telemetry-only, transport-only, matroid/one dimensional cone-only, toric/GraphCG-only, memory-landscape-only, chart-BPB-only, and full-stack variants, each gated by validation BPB/graph-BPB and read-only artifact checks.
- [ ] Fold the vector-bundle paper sidecar into the active repair checklist: chart ids, monomial transport ids, toric active rows, one dimensional cone-filtration flat defects, GraphCG-toric agreement, and transported persistence-landscape metrics must be emitted as model-backed audit payload fields.
- [ ] Add BPB/graph-BPB promotion gates for every new vector-bundle or toric auxiliary. The browser may display the diagnostics before promotion, but decoding/training acceptance must remain BPB-first.
- [ ] Add tests that distinguish real GUDHI persistence landscape `lambda_k(t)` vectors from GoT NLL/fitness/density fields and prove unavailable landscapes are not converted to zero vectors.
- [ ] Keep paper claims conservative in code/docs: monomial transports and matroid/one dimensional cone filtrations are mathematically motivated regularizers unless the implementation constructs an actual tropical toric variety or tropical scheme.

## 2026-06-14 Main-Agent Status Addendum: v6 Restart, Paper Sidecar, and Density Repair

### Training restart and crash repair
- Fixed the step-250 periodic validation crash in `TropicalGT-I/src/tropicalgt/run.py` by constructing `AnalogicalMemoryQualityGate` inside `_run_periodic_validation_round` before memory storage/retrieval diagnostics are computed.
- Launched fresh BPB-oriented run `tropicalgt_i_pg_bpb_step0_full24b_b54_v6_bpb_restart` from step 0 with config `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b54_v6_bpb_restart.json`.
- v6 changes: batch size 54, lower BPB-competing auxiliary weights, stricter analogical-memory quality thresholds, fresh memory bank under the v6 output tree, active periodic visual audits every 250 steps.
- Training must remain alive; at each periodic boundary verify that interactive artifacts are produced without synthetic replacement.

### Paper sidecar integration
- Avicenna reviewed `references/2405.03505v1.pdf` and `references/2009.03030v2.pdf` and updated `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` plus `planning/tropicalgt_i_vector_bundle_toric_embedding_training_plan.md`.
- Implementation hooks now required by the paper sidecar: tropical vector-bundle chart ids on graph tokens, scoped exponent-chart features, one dimensional cone-filtration defects, chart-transition penalties, GraphCG/exponent-chart agreement, and BPB/graph-BPB ablations for these terms.
- These additions are theory-backed training ideas only until code paths expose exact metrics and tests; do not log them as active losses before implementation.

### NLL / energy-density visualization repair
- The NLL density page must render only actual model GoT state anchors as model states.
- Gaussian support samples are hidden by default and must never be labeled as reasoning states.
- The visible density layer is a Gaussian support field plus translucent NLL-colored neighborhoods centered at actual 3D PCA GoT embeddings.
- Remaining visual work: improve camera defaults, expose optional 2D surface only when it covers the local trajectory neighborhood, and show dotted decoding/causal edges over the same PCA coordinate frame.

### CAS / real free-resolution policy
- Real free resolutions remain unavailable unless a certified CAS backend (Macaulay2, Sage, Singular, or verified equivalent) returns differentials, multidegree shifts, exactness/minimality checks, and Betti table data.
- `BEMultipliers` may be used for Buchsbaum-Eisenbud diagnostics but is not by itself a generic free-resolution backend.
- Until CAS certification exists, figures must render unavailable/certificate diagnostics rather than proxy free resolutions.

### Immediate remaining visual defects
- Simplex-tree pages still need a genuine connected trie/face-coface-poset view instead of disconnected columns.
- 2-parameter module pages must show an honest `(level, radius)` lattice/grid with fiber ranks, structure-map diagnostics, and downloadable JSON.
- Analogical maps must compare probability-vector complexes with Jensen-Shannon assignment and must explain any non-preserved edges/faces instead of reporting high derived similarity with zero free-resolution support.
- Tropical support heatmaps need grouped labels, top-support summaries, margin profile readability, and collapse diagnostics without jammed axes.

## 2026-06-16 Main-Agent Status Addendum: b62 Always-On Restart And Cleanup

### Training continuity
- b61 `tropicalgt_i_pg_bpb_step0_full24b_b61_20260616T155103Z_fresh_bpb112_alwayson_5k_gate` crashed at periodic step 500 while writing `trajectory_growth.json` because the root filesystem was full.
- Safe cleanup completed before restart: removed `~/.cache/tropicalgt/cas_free_resolution`, `/tmp/torchinductor_iska`, pytest scratch, and toric/tropical temporary smoke directories. Datasets, checkpoints, W&B folders, generated training outputs, and secrets were preserved.
- Fresh step-0 b62 run is active: `tropicalgt_i_pg_bpb_step0_full24b_b62_20260616T162800Z_fresh_bpb112_alwayson_5k_gate`.
- b62 trainer PID: `960097`; 5K gate monitor PID: `960152`; W&B run id: `pi8weh6v`.
- b62 config: `TropicalGT-I/outputs/launch_configs/tropicalgt_i_pg_bpb_step0_full24b_b62_20260616T162800Z_fresh_bpb112_alwayson_5k_gate.json`.
- b62 train log: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b62_20260616T162800Z_fresh_bpb112_alwayson_5k_gate/logs/train_nohup_20260616T162800Z.log`.
- b62 stop record: `TropicalGT-I/outputs/training_stop_records/b62_fresh_step0_alwayson_step5000_gate.json`.
- Initial b62 verification: W&B initialized, GPU memory rose to about 22.4GB, and first training steps logged normally.

### Sequential implementation progress
- Completed the current CAS-cache hardening item for toric/tropical sidecars: caches now use XDG/home cache paths or explicit env overrides, validated wrapper payloads, versioned keys including backend probes and input hashes, atomic writes, explicit `written` flags, and deterministic unavailable-state cache hits.
- Focused verification passed with `CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` returning `30 passed`.
- Next linear item remains the visual/math repair queue: continue no-proxy simplex-tree/radius/analogical/tropical-support repairs while b62 trains to the 5K gate.

## 2026-06-16 Radius Slider Contract Repair

Sequential visual/math item completed after the toric/tropical CAS cache hardening pass:

- Added a machine-readable `radius_filtration_slider_contract` to 3D complex slider Plotly metadata. The contract records per-frame thresholds, visible 0-simplices, solid radius edges, filled faces, dotted trajectory overlays, dotted graph-token overlays, and dotted causal/decoding overlays.
- Enforced the initial-frame rule for true radius filtrations: the first slider frame is a disjoint 0-simplex cloud. Positive-dimensional solid simplices and dotted overlays are hidden on that initial frame even when an input edge reports filtration `0.0`.
- Preserved later-frame semantics: solid edges and filled 2-simplices remain radius-gated; dotted overlays remain order/causal metadata and do not become simplices.
- Added regression coverage proving first-frame disjointness, monotone visible counts, and later-frame recovery of radius edges, filled faces, and dotted overlays.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q
# 37 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py TropicalGT-I/tests/test_backfill_interactive_audit_artifacts.py -q
# 11 passed
```

Operational note: b62 remained alive during verification and had advanced past step 181/5000 with train loss/NLL around 1.641/1.617 when checked.

Next linear visual/math item: continue from slider-contract verification into remaining no-proxy simplex-tree, analogical-map, NLL-density, and tropical-support readability repairs.



## 2026-06-16 Analogical Map Claim Contract Repair

Sequential visual/math item completed after the radius-slider contract repair:

- Added explicit no-proxy render-claim fields to probability-vector analogical map diagnostics: `safe_to_render_as_simplicial_map`, `safe_to_render_as_chain_map`, `safe_to_render_as_persistence_module_morphism`, `map_render_claim`, `map_claim_failure_reason`, and `no_proxy_or_fallback`.
- Memory retrieval sidecars now distinguish certified filtered simplicial maps from vertex correspondences that fail simplex-tree or filtration preservation. Failed correspondences receive zero probability-map score contribution and are labeled `probability_correspondence_not_a_simplicial_map`.
- Analogical visualization JSON, pair pages, quality table, hover text, and top-k index now expose the same certified/not-certified status. The top-k table body now fills the advertised probability-map contribution, similarity, preservation, source, and map-claim columns.
- Simplex-tree map reports now have conditional interpretations: certified reports may mention induced F2 chain maps and persistence-module morphisms; failed or incomplete reports explicitly state that no simplicial map, chain map, or persistence-module morphism is asserted.
- The algebraic realization certificate now uses a conditional chain-map note so non-certified probability correspondences cannot be read as persistence-module morphisms.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/memory.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_metrics_and_memory.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py TropicalGT-I/tests/test_simplicial_visualization.py -q
# 49 passed
```

Operational cleanup note: after b62 was verified active, the failed generated b61 output directory was removed to free space, with small logs/JSON metadata preserved under `TropicalGT-I/outputs/training_stop_records/preserved_b61_failure_20260616T155103Z`. Datasets, checkpoints, W&B folders, current b62 outputs, b60 evidence outputs, secrets, and source history were not removed.

Operational note: b62 remained alive during verification and had advanced past step 372/5000 with train loss/NLL around 1.487/1.465 when checked.

Next linear visual/math item: continue from the analogical-map claim contract into remaining no-proxy simplex-tree trie/face-coface-poset, NLL-density, and tropical-support readability repairs.


## 2026-06-16 SimplexTree Poset Contract Repair

Sequential visual/math item completed after the analogical-map claim contract repair:

- The 3D SimplexTree renderer now emits a machine-readable `simplex_tree_poset_contract` in Plotly metadata.
- The contract records the backend, safe-render state, empty-simplex root presence, displayed/source simplex counts, truncation, dimension counts, actual face-to-coface Hasse cover edge count, empty-root-to-vertex cover count, optional sorted-label trie prefix edge count, and the primary/secondary edge policy.
- The renderer keeps actual face-to-coface covers as the primary graph and keeps sorted-label trie prefix links legend-only; the contract records `not_disconnected_simplex_columns=true` and `primary_edges=actual_face_to_coface_covers`.
- Fixed the analogical top-k empty-table colspan to match the 21-column table introduced by the map-claim contract repair.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 49 passed
```

Next linear visual/math item: continue into remaining no-proxy NLL-density and tropical-support readability repairs while b62 trains to the 5K gate.

## 2026-06-16 NLL Density Render Contract Repair

Sequential visual/math item completed after the SimplexTree poset contract repair:

- The main trajectory density overlay and standalone NLL-density cloud now carry a machine-readable `nll_density_render_contract` / `visual_layer_contract` with schema `tropicalgt.nll_density_render.v1`.
- The contract states that plotted coordinates are actual 3D PCA coordinates of model graph-state embeddings, with z fixed to PC3; raw NLL is encoded through color, hover, local density metadata, and explicit summaries rather than by moving model-state markers onto a fake NLL z-axis.
- Gaussian support samples are now audit-only density support, hidden by default as a legend-only trace named `audit samples from the Gaussian NLL field (hidden by default)`. The contract records `support_samples_are_model_states=false`, `support_samples_hidden_as_model_states=true`, and `support_sample_trace_visibility=legendonly`.
- The visible default layers are explicitly enumerated as density volume, anchor Gaussian neighborhoods, and actual model anchor markers. Actual model anchors remain visible by default and carry the counted model-state layer provenance.
- The interactive artifact validator now requires the top-level portable NLL-density payload to include this visual-layer contract, verifies the PC3 z-axis policy, legend-only support samples, support-vs-anchor provenance, visible layer list, counts, bandwidth, and no-proxy flag. Nested or textual render notes alone are no longer accepted as sufficient evidence.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 49 passed
```

Operational note: b62 remained alive during verification and had advanced past step 566/5000 with train loss/NLL around 1.395/1.375 when checked.

Next linear visual/math item: continue into the remaining no-proxy tropical-support readability repair while b62 trains to the 5K gate.

## 2026-06-16 Tropical Support No-Proxy Render Contract Repair

Sequential visual/math item completed after the NLL-density render contract repair:

- Added a structured `tropical_support_render_contract` with schema `tropicalgt.tropical_support_render.v1` to the tropical active-support payload and Plotly metadata.
- The contract records that support columns are only observed valid `graph_token_trace.tokens[*].active_support_index` values emitted by the model, that the assignment matrix is a binary argmax support-selection mask, and that zero assignment cells are unselected cells rather than zero tropical margins.
- `selected_margin_matrix` is now documented as the margin-evidence matrix: finite model tropical margins appear only on selected observed support cells, with unselected cells null. The legacy zero-filled `margin_matrix` remains for backward compatibility and is explicitly marked display-only in the contract.
- Mixed invalid `active_support_index` rows are now preserved as invalid evidence. They receive `invalid_active_support_index` status in `support_assignment_status_by_token` and `support_flow_edges`, are not rendered as assignment cells, and do not create fabricated support columns.
- The support artifact now records valid/invalid support-assignment counts, invalid row details, no-proxy flags, and the non-certified wall-margin scope `margin_threshold_audit_not_certified_normal_fan_wall_crossing`.
- The interactive artifact validator now requires the no-proxy support render contract, support-assignment status rows, consistent invalid-support counts, assignment-mask semantics, observed-support column policy, and explicit invalid status before accepting out-of-range support indices.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 50 passed
```

Operational note: b62 remained alive during verification and had advanced past step 702/5000 with train loss/NLL around 1.369/1.347 when checked.

Next linear visual/math item: continue the remaining visual/math repair queue after a cleanup/diff/push cycle, keeping b62 active until the 5K gate.

## 2026-06-16 Toric Embedding Sidecar Audit Surface Pass

Sequential visual/math item completed after the tropical-support no-proxy render contract repair:

- Added `toric_embedding_sidecar.html` and `toric_embedding_sidecar.json` as new audit artifacts written by `write_inference_audit_artifacts` for new bundles.
- Certified rendering is allowed only for an explicit integer exponent matrix (or columns) plus a Macaulay2 `Quasidegrees` `toricIdeal(A,R)` certificate emitted through the real CAS toric bridge.
- The sidecar renders the exponent matrix and toric-ideal certificate as a finite monomial-map toric ideal sidecar only. It explicitly does not claim a normal fan, tropical-variety embedding, global toric-variety embedding, sheaf, Cox-module, or global neural toric model.
- If no model-derived exponent matrix exists, the page and payload render an unavailable state with the exact missing input and no-proxy contract; chart-bundle logits, toric activation rows, support tokens, GraphCG cells, embeddings, and visualization rows cannot substitute for the CAS certificate.
- The artifact validator now checks the toric sidecar when present: schema, CAS schema, no-proxy render contract, finite-sidecar safety, certificate attachment when available, exponent-matrix evidence, and false normal-fan/tropical-variety/global toric-variety safety flags.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 52 passed
```

Operational note: b62 remained alive during verification and had advanced past step 750/5000 with train loss/NLL around 1.345/1.322 when checked.

Next linear item: continue source-side audit hardening and browser validation while b62 trains to the 5K gate; newly generated bundles will include the toric sidecar, while pre-existing b62 artifacts may need backfill before strict sidecar review.
