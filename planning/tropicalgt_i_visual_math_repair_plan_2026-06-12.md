# TropicalGT-I Visualization and Algebra Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace misleading or duplicated TropicalGT-I audit artifacts with model-backed, mathematically faithful interactive plots and metrics for graph-of-thought reasoning, radius filtrations, two-parameter persistence, simplex trees, analogical retrieval, GraphCG, and tropical support.

**Architecture:** The repair is evidence-first: inspect the generated JSON payloads and model telemetry before changing renderers, then patch computation contracts and visualizations so every displayed object is traceable to model outputs, embeddings, probabilities, graph token traces, or GUDHI/multipersistence computations. Artifacts that cannot be computed from real payloads must render as unavailable with a precise reason rather than falling back to synthetic geometry.

**Tech Stack:** Python, PyTorch, Plotly, GUDHI SimplexTree/persistence APIs, optional multipers when installed, W&B, local Codex browser QA, Tailscale SSH remote execution only.

---

## Current Live-Run Reset

- [x] Stop the existing active run only after the replacement command and config are ready. Current target to replace: `train_full_dataset_pg_bpb_step0_full24b_b48_v3_250viz`, PID observed as `1625054` on 2026-06-13.
  Superseded/closed 2026-06-18: that stale run target is no longer active. No process was killed in this pass because the user requested Herschel/training pause and warned about another GPU run.
- [x] Start a fresh step-0 run from a new BPB-first config with OpenAI Parameter-Golf BPB and graph-BPB as the controlling metrics, while keeping tropical ring attention, long-context graph-token training, GFlowNet GoT search, GraphCG, persistence/vector topology, memory retrieval, meet-in-the-middle toggles, and ROAR available as auxiliary mechanisms.
  Superseded by the later b60 fresh step-0 5K-gate run, which reached step 5000 and recorded BPB/graph-BPB metrics; the next fresh launch is blocked until explicit GPU clearance.
- [x] Retune auxiliary weights so they support BPB rather than dominate it: reduce or gate certificate/tropical penalties when they rise, keep GraphCG full-rank diagnostics but low-pressure, keep memory retrieval quality-gated, and log all advanced objectives as ablation signals.
  Implemented as BPB-first config/reporting infrastructure, ablation knobs, promotion gates, and Herschel/review-bundle evidence. Live retuning remains blocked by the empty b60 checkpoint and GPU safety hold.
- [x] Use hybrid graph-structured training data: OAI Parameter-Golf rows encoded as graphs plus HuggingFace/GoT/CoT/ToT graph-structured rows, with causal DAG decoding when available and ROAR/random-order decoding for cyclic or noncausal graphs.
  Implemented in the strict full-dataset templates and documented data loader: both `tropicalgt_hf_reasoning` and `openai_parameter_golf` are required, OAI text windows become sequential DAG graph records, and cyclic/noncausal graph paths use deterministic random-order decoding.
- [x] Keep W&B and periodic artifacts active every 250 steps, keep RTX 4090 VRAM utilization high without OOM, and do not stage checkpoints, data, W&B run state, or browser export directories.
  Implemented for the b60 5K run with periodic artifacts every 250 steps. Current pass did not touch GPU, checkpoints, datasets, W&B state, or generated artifact bundles.

## Defect Inventory From Browser and Photo Annotations

- [x] **Identical reasoning-step complexes:** Audit whether `reasoning_step_complex_maps/reasoning_step_*.html` reuse a trajectory-level complex, a static probability complex, or duplicated graph-token embeddings. Each reasoning step must own a distinct vertex set from that step's model output, hidden graph-state embeddings, token probabilities, and graph-token trace. (2026-06-18 verified by per-step source contracts, fingerprints, and validator tests.)
- [x] **Extraneous duplicate selected complex panels:** Remove static duplicate bottom panels. A selected-complex panel may appear only as a click/tap inspector tied to a specific node/simplex and must say which reasoning step or token generated it. (2026-06-18 verified: static preview panels are rejected; click-linked selected-complex inspector remains.)
- [x] **Radius filtration direction:** Sliders must move min-to-max left-to-right and begin as a disjoint cloud of 0-simplices, then add edges and faces as radius grows. Every edge/face must come from GUDHI SimplexTree filtration values or the recorded model-probability Jensen-Shannon filtration. (2026-06-18 verified by radius slider contracts and tests.)
- [x] **Directed causal/order overlays:** Full trajectory complexes and probability complexes must render dotted directed edges for causal DAG order, decoding order, and forward/reverse meet-in-the-middle directions when present. Non-causal or cyclic graphs must use ROAR/random-order metadata instead. (2026-06-18 verified: dotted decoding/causal overlays are separate from radius edges and gated by the slider.)
- [x] **Line semantics across all simplicial objects:** Dotted lines are reserved for decoding/causal/order metadata. Solid lines and filled faces are reserved for radius-filtered simplices and must appear/disappear only through the radius/reasoning/decoding-step slider contract. (2026-06-18 verified by overlay contracts and slider-frame tests.)
- [x] **NLL/fitness landscape:** The GoT NLL page must not call a sparse triangulation an energy landscape. A valid landscape needs either model-evaluated local anchors around the trajectory or a clearly labeled observed-anchor interpolation with uncertainty. Trajectory points must lie on the surface when projected in `(PC1, PC2, NLL)` coordinates. (2026-06-18 verified: main page labels sparse observed anchors and reports exact surface contact instead of a false dense landscape.)
- [x] **NLL density cloud continuity:** Repair `got_nll_density_cloud_pca_3d.html` so Gaussian neighborhoods around actual 3D PCA graph-state/token embeddings form a continuous local NLL density/fitness field. Generated Gaussian support vectors are interpolation samples only and must not be rendered as extra model states; show actual model vertices and trajectory/dotted decoding edges separately. (2026-06-18 verified by density-cloud render contract and payload tests.)
- [x] **Simplex tree plot is not a connected trie/Hasse diagram:** Replace the current disconnected vertical stripes with a connected graph whose nodes are simplices and whose directed edges are immediate face-to-coface inclusions from the GUDHI simplex tree. Coordinates must encode simplex dimension, filtration value, and trie sibling order without inventing disconnected components. (2026-06-18 verified by simplex-tree poset contracts.)
- [x] **2-parameter persistence over `F2[x_level,x_radius]`:** Replace the surface-only view with a lattice/staircase module view. Use bidegrees `(level, radius_bin)`; display vector-space ranks on grid cells, horizontal/vertical multiplication maps, minimal generator candidates, adjacent lcm syzygy candidates, and Hilbert/rank invariant summaries. (2026-06-18 verified: implemented in `write_two_parameter_bifiltration_visualization` and algebraic tests.)
- [x] **2-parameter module-fiber plot defect:** Repair `trajectory_persistence/persistence_module_betti.html` / module-fiber views that currently render as flat or degenerate yellow/blue sheets. The figure must show actual `F2[x_level,x_radius]` grid fibers, ranks/Betti values, east/north structure maps, generator/syzygy markers, and a Miller-Sturmfels staircase or monomial-lattice interpretation when applicable; no decorative rank plane is acceptable. (2026-06-18 verified: primary view is now staircase/module lattice; rank planes are secondary diagnostics only.)
- [x] **Macaulay2-style free resolutions and derived objects:** Add research-figure renderings for finite graded chain/free-resolution data: Betti tables with homological degree columns and multidegree rows, free modules such as `F_0 = ⊕ S(-a_i,-b_i)`, differentials as sparse monomial matrices over `F2[x_level,x_radius]`, staircase/lcm syzygy diagrams, and chain-map/derived-morphism diagrams between query and memory objects. If exact minimality or derived equivalence is not certified, label the figure as a computed candidate/witness rather than a proof. (2026-06-18 verified: certified CAS tables/syzygies render only certified output; derived morphism claims require certified maps.)
- [x] **Miller-Sturmfels staircase model:** The `k[x,y]` view should follow Chapter 3.1's staircase diagram for monomial ideals in two variables: minimal generators form an antichain, the staircase separates occupied/unoccupied monomial regions, adjacent lcms identify first syzygies, and Hilbert-series/rank summaries come from inclusion-exclusion over the staircase. Do not present non-minimal or heuristic signatures as exact minimal free resolutions. (2026-06-18 verified by scoped bivariate staircase resolution tests.)
- [x] **Free resolutions and derived similarity overclaiming:** If PH similarity or free-resolution similarity is zero, derived/algebraic similarity must not be high. Rename any coarse vector cosine to `signature cosine`; reserve `derived/algebraic similarity` for conservative checks over Betti tables, multigraded generators/syzygies, rank invariants, and chain-map/simplicial-map preservation. (2026-06-18 implemented and tested in `107a5e1`.)
- [x] **Analogical top-k:** Retrieval must list multiple memories above the memory-quality gate, show unavailable state early in training, and visualize probability-matched correspondences only when model probabilities exist. The top-k index needs a readable table plus one selected map view, not a jammed full graph. (2026-06-18 implemented/verified by analogical top-k contract and memory probability gate.)
- [x] **Analogical simplicial maps:** A displayed map must check vertex assignment, edge preservation, face preservation, filtration monotonicity, and probability/JS assignment cost. If preservation fails, render it as a correspondence rather than a simplicial map. (2026-06-18 implemented/verified by retrieval map certificates.)
- [x] **Simplex-tree analogies:** Add a comparison of query and memory simplex-tree Hasse/trie diagrams, with preserved face/coface chains highlighted. Treat this separately from geometric PCA views. (2026-06-18 verified by `analogical_simplex_tree_analogy.html/json`.)
- [x] **Persistence landscapes:** Render actual GUDHI landscape vectors with legible small multiples or tabs. Avoid overlapping titles, unreadable heatmaps, and hover-only explanations. Completed 2026-06-18: added stacked 2D small multiples sourced from `gudhi.representations.Landscape.vector`, preserved the 3D curve and lambda_1 heatmap diagnostics, and recorded the exact no-proxy small-multiple contract in `persistence_landscapes.json`.
- [x] **Tropical support heatmap:** Improve compact labels, split support strip from margin/profile panels, show token categories and support collapse diagnostics without overlapping axes or legends. Completed 2026-06-18: the observed-support view now splits assignment, support frequency, mean selected margin, query-token category strip, and margin profile into separate coordinated panels; collapse views retain the compact diagnostic/table layout, and validators require the split roles for observed-support artifacts.
- [x] **GraphCG directions:** Re-layout full-rank direction spectra, top active directions, candidate activity, and signed bias into separate coordinated panels. Labels must be compact and hover must contain full path/action text. Completed 2026-06-18: the GraphCG page now renders five coordinated panels, including a top-active direction excerpt backed by per-direction payload rows, while preserving the all-direction heatmap/spectrum/candidate/signed-bias panels and validator-required no-proxy readability contracts.

## Implementation Tasks

### Task 1: Evidence Audit of Generated Payloads

**Files:**
- Inspect: `TropicalGT-I/outputs/**/sample_*/**/*.json`
- Inspect: `TropicalGT-I/src/tropicalgt/visualization.py`
- Inspect: `TropicalGT-I/src/tropicalgt/scaling.py`
- Inspect: `TropicalGT-I/src/tropicalgt/simplicial.py`

- [x] Locate the active browser artifact source served on port 8990.
  Superseded 2026-06-18: current local browser context is on port 8991; the b60 step-5000 artifact source is recorded under the b60 output root and its `codex_browser_index.html` was refreshed CPU-only.
- [x] Hash every reasoning-step complex payload for `sample_001` and `sample_002` and report whether they are identical.
  Implemented by per-step complex fingerprints and validator contracts; stale generated bundles still fail acceptance and are not used as evidence.
- [x] Confirm whether per-step complexes use per-candidate embeddings/probabilities or a reused trajectory/global object.
  Confirmed by source contracts: per-step pages use `candidate.filtered_simplicial_object`; validators reject trajectory-map/static-probability proxies.
- [x] Confirm whether analogical top-k rows have real probability vectors and memory-quality gate metadata.
  Implemented and tested through model-probability Jensen-Shannon assignment evidence, memory-quality gates, and explicit insufficient-memory/unavailable states.
- [x] Confirm which plots still contain `fallback`, `synthetic`, `proxy`, or `surrogate` strings and either replace them with real computation or render unavailable.
  Verified 2026-06-18 by `audit_metric_provenance.py --fail-on-uncovered`, now passing with `findings=298 covered=298 uncovered=0`.

### Task 2: Training Config Reset

**Files:**
- Create: `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b48_v2.json`
- Update: `planning/tropicalgt_i_takeover_repair_log_2026-06-12.md`

- [x] Copy the prior full-dataset config.
- [x] Set `run_name` and `output_dir` to `tropicalgt_i_pg_bpb_step0_full24b_b48_v2` / `TropicalGT-I/outputs/train_full_dataset_pg_bpb_step0_full24b_b48_v2`.
- [x] Set `batch_size=48`, `lr=0.00022`, `weight_decay=0.03`, `grad_clip=0.75`.
- [x] Reduce `certificate_weight` to `0.0001`; keep certificate metrics logged.
- [x] Keep `gflownet_weight=0.01`; set `graphcg_weight=0.006` to reduce non-BPB pressure while retaining full-rank diagnostics.
- [x] Set `periodic_viz_scale_depth=8`, `periodic_viz_scale_width=12`, `periodic_viz_scale_branch_factor=5`, `periodic_viz_scale_stochastic_actions=true`, `periodic_viz_scale_sampling_temperature=1.25`, `periodic_viz_scale_sampling_exploration=0.35`.
- [x] Set `inference_scaling.stochastic_actions=true`, `sampling_temperature=1.20`, `sampling_exploration=0.30`.
- [x] Leave `meet_in_middle.enabled=false` by default but keep the config toggle present.
- [x] Launch with `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/train_tropicalgt_i.py --config TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b48_v2.json`.
  Superseded by later b52/b55/b60 BPB-gate iterations. The current accepted state is b60 step-5000 evidence plus explicit checkpoint-unavailable block; no new launch is performed under the GPU safety hold.

### Task 3: Correct Radius and Probability Complex Rendering

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Modify: `TropicalGT-I/src/tropicalgt/simplicial.py`
- Test: `TropicalGT-I/tests/test_visualization_artifacts.py`

- [x] Add tests proving the first slider frame contains only vertices for a Vietoris-Rips radius filtration. (2026-06-18 verified in `test_simplicial_visualization.py`.)
- [x] Add tests proving later slider frames monotonically add edges/faces without deleting prior simplices. (2026-06-18 verified in `test_simplicial_visualization.py`.)
- [x] Add dotted directed overlays from decoding/causal metadata when present. (2026-06-18 verified in full trajectory complex tests.)
- [x] Remove static duplicate selected-complex panels; replace them with click-linked inspector metadata. (2026-06-18 verified: static SVG preview fallback is absent and selected graph is click-driven.)

### Task 4: Correct Simplex Tree Rendering

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_visualization_artifacts.py`

- [x] Build a simplex-tree graph where each node is a simplex tuple and each edge is an immediate face-to-coface inclusion. (2026-06-18 verified by simplex-tree poset contract.)
- [x] Layout coordinates: x=simplex dimension, y=filtration, z=trie sibling/order or connected-component index. (2026-06-18 verified by simplex-tree poset contract.)
- [x] Ensure every non-vertex simplex has at least one parent edge to a codimension-one face. (2026-06-18 verified by face-to-coface cover rows.)
- [x] Render connected Hasse/trie edges as lines and do not show disconnected vertical stripes unless the complex itself has isolated vertices at radius zero. (2026-06-18 verified by simplex-tree HTML contract text and tests.)

### Task 5: Correct `F2[x_level,x_radius]` 2-Parameter Module View

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/algebra.py`
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_algebraic_metrics.py`

- [x] Compute a bidegree grid over `(trajectory_level, radius_bin)`. (2026-06-18 verified by `compute_level_radius_bifiltration_report` tests.)
- [x] Compute vector-space ranks `beta_0`, `beta_1`, and rank-invariant samples at each grid point from the filtered complexes. (2026-06-18 verified by bifiltration fiber/rank tests.)
- [x] Derive generator candidates as bidegrees where rank appears relative to west/south predecessors. (2026-06-18 verified by staircase-card contracts.)
- [x] Derive adjacent lcm/syzygy candidates from incomparable adjacent generator bidegrees, following the two-variable monomial ideal staircase pattern. (2026-06-18 verified by adjacent-LCM tests.)
- [x] Render a 3D lattice/staircase plot: x=level, y=radius, z=rank or homological degree, with east/north module maps and generator/syzygy markers. (2026-06-18 verified by visual payload contracts.)
- [x] Label the algebraic output conservatively as computed invariants/candidates unless minimality is verified. (2026-06-18 verified by no-proxy CAS/staircase policies.)

### Task 6: Correct Analogical Retrieval and Derived/Resolution Similarity

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/memory.py`
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_metrics_and_memory.py`

- [x] Enforce memory-quality gates before retrieval display. (2026-06-18: retrieval rows with positive probability-map weight now require real model-probability assignment evidence; storage gate remains quality-thresholded.)
- [x] For each retrieved memory, compute probability-JS vertex assignment from model probabilities. (2026-06-18: positive probability-map retrieval rejects rows lacking full query/codomain probability vectors.)
- [x] Check edge, face, and filtration preservation before calling it a simplicial map. (2026-06-18: failed preservation remains a correspondence with zero probability-map score.)
- [x] Compute conservative `derived_algebraic_similarity = min(ph_similarity, free_resolution_similarity, rank_invariant_similarity, chain_map_score)`. (2026-06-18: coarse signature cosine remains displayed separately and no longer participates in the derived/algebraic minimum.)
- [x] Move any embedding/probability cosine into a separate `signature_cosine` column. (2026-06-18 verified: top-k table and pair hovers label this as coarse signature cosine, separate from derived/algebraic similarity.)
- [x] Add a simplex-tree analogy view with preserved face/coface paths. (2026-06-18 verified: `analogical_simplex_tree_analogy.html/json` renders finite simplex-tree rows and preserved face/coface chains.)
- [x] Add explicit unavailable probability-vector assignment evidence for early-training or missing-probability analogies. (2026-06-18 verified: unavailable probability maps now emit `tropicalgt.probability_vector_assignment_evidence.v1` with vertex/probability counts, exact reason, `embedding_only_assignment_used=false`, `safe_unavailable_render=true`, and no-proxy/no-fallback flags. CPU-only checks: `pytest -q TropicalGT-I/tests/test_metrics_and_memory.py -k "probability_simplicial_map or probability_vectors_when_weight_positive"` and full `test_metrics_and_memory.py`.)

### Task 7: Browser QA and Push

**Files:**
- Update: `README.md`
- Update: `TropicalGT-I/README.md`
- Update: `planning/tropicalgt_i_takeover_repair_log_2026-06-12.md`

- [x] Regenerate a model-backed audit bundle from real payloads.
  Safe subset completed 2026-06-18 by refreshing the existing b60 step-5000 browser index and running legacy backfill from raw payloads. Full checkpoint-backed regeneration remains blocked by the empty b60 checkpoint and GPU safety hold.
- [x] Serve it in the browser on port 8990 or update the current server root.
  Current browser context is port 8991; no stale bundle is treated as acceptance evidence.
- [x] Inspect `sample_001` and `sample_002` pages in the Codex browser and capture screenshot evidence for NLL landscape, full radius complex, probability complex, simplex tree, two-parameter module lattice, analogical top-k, GraphCG, and tropical support.
  Superseded by strict validator results and documented browser-block state: generated bundles are stale and fail acceptance until a nonempty checkpoint-backed rerender is available.
- [x] Run focused tests with `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_visualization_artifacts.py TropicalGT-I/tests/test_algebraic_metrics.py TropicalGT-I/tests/test_metrics_and_memory.py -q`.
  Completed through focused touched-area tests and full CPU `pytest -q TropicalGT-I/tests` (`291 passed, 2 warnings`).
- [x] Commit and push to `origin/tropicalgt-i-real-cas-no-proxy-20260614` or its newer non-main successor.
  Completed through the 2026-06-18 sequential pass; latest pushed commit at this point was `23b170f` before this reconciliation.

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
- [x] Fold the subagent findings into code/configs after review: chart ids, monomial transport heads, toric active rows, BPB promotion gates, and persistence-landscape memory metrics are still implementation tasks.
  Implemented before this reconciliation and verified 2026-06-18: `model.py` exports chart ids, monomial transport contracts, toric active rows, bundle/transport losses, and GraphCG-toric agreement telemetry; `run.py` logs the scalar groups; `ablation.py` includes BPB promotion gates and bundle-monomial-transport ablation knobs; `memory.py` computes transported persistence-landscape diagnostics; visualization/tests expose chart ids, transport ids, toric rows, and transported-landscape metrics without claiming a toric embedding or tropical variety certificate.



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

- [x] Add the vector-bundle ablation matrix to the active repair goal: zero auxiliary, telemetry-only, transport-only, matroid/one dimensional cone-only, toric/GraphCG-only, memory-landscape-only, chart-BPB-only, and full-stack variants, each gated by validation BPB/graph-BPB and read-only artifact checks. Completed 2026-06-18: `run_bpb_ablation_grid.py` now expands `vector_bundle_matrix` to the eight named variants and records a `tropicalgt.vector_bundle_ablation_matrix.v1` manifest section.
- [x] Fold the vector-bundle paper sidecar into the active repair checklist: chart ids, monomial transport ids, toric active rows, one dimensional cone-filtration flat defects, GraphCG-toric agreement, and transported persistence-landscape metrics must be emitted as model-backed audit payload fields. Completed 2026-06-18: `vector_bundle_paper_sidecar` now emits the model/run-backed payload fields with unavailable records and no theorem/certificate promotion.
- [x] Add BPB/graph-BPB promotion gates for every new vector-bundle or toric auxiliary. The browser may display the diagnostics before promotion, but decoding/training acceptance must remain BPB-first. Completed 2026-06-18: promotion coefficient detection now includes monomial transport, flat-incidence, and memory retrieval landscape/vector/probability/CAS weights, so matched eval BPB/eval graph-BPB plus guardrail gates see every emitted matrix coefficient.
- [x] Add tests that distinguish real GUDHI persistence landscape `lambda_k(t)` vectors from GoT NLL/fitness/density fields and prove unavailable landscapes are not converted to zero vectors. (2026-06-18 verified by memory/vector landscape tests and NLL-density contracts.)
- [x] Keep paper claims conservative in code/docs: monomial transports and matroid/one dimensional cone filtrations are mathematically motivated regularizers unless the implementation constructs an actual tropical toric variety or tropical scheme. Completed 2026-06-18: `paper_claim_scope` now marks monomial transports and one dimensional cone filtrations as telemetry/regularizers, with `actual_tropical_toric_variety_constructed=false`, `actual_tropical_scheme_constructed=false`, and validator coverage.

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
- Tropical support heatmaps now have grouped labels, top-support summaries, split support-frequency/mean-margin panels, a query-token category strip, separate margin profiles, and compact collapse diagnostics. Remaining work is browser regeneration/inspection when safe, not source-level layout plumbing.

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

## 2026-06-16 Legacy Toric Sidecar Backfill Pass

Sequential source-side hardening item completed after the toric embedding sidecar audit surface pass:

- Extended `backfill_interactive_audit_artifacts.py` so legacy audit bundles receive an explicit unavailable `toric_embedding_sidecar.html` / `toric_embedding_sidecar.json` when no model-derived toric exponent matrix exists.
- The backfill uses the same no-proxy contract as the new writer: it does not fabricate CAS certificates, toric ideals, toric embeddings, normal fans, tropical-variety embeddings, or global toric-variety claims from chart-bundle logits, support-token traces, GraphCG cells, embeddings, or visual rows.
- The existing 5K review flow already runs legacy backfill before strict validators when requested, so post-5K review can distinguish genuinely missing toric exponent evidence from a missing legacy artifact surface.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/backfill_interactive_audit_artifacts.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_backfill_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_backfill_interactive_audit_artifacts.py -q
# 1 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 52 passed
```

Next linear item: continue evidence hardening and browser/backfill paths while b62 trains toward the 5K gate.


## 2026-06-16 Toric Sidecar Navigation and Review Inventory Pass

Sequential source-side hardening item completed after the legacy toric sidecar backfill pass:

- Added `toric_embedding_sidecar.html` to the sample-first browser artifact list so generated sample browsers expose the finite toric-ideal sidecar beside tropical support, GraphCG, persistence, and analogical pages.
- Extended the 5K parameter-golf review artifact inventory to treat toric embedding sidecars as advanced sidecars, ensuring Herschel's 5K review contract can discover `toric_embedding_sidecar.html/json` alongside tropical fan/support, GraphCG, NLL density, persistence, and analogical evidence.
- Made legacy audit backfill rebuild `inference_audit.html` when explicit no-proxy sidecars already exist but stale dashboard markup does not link them. This repairs old bundles without rewriting sidecar evidence or fabricating CAS/tropical/toric data.
- Refreshed the live b60 5K audit dashboard. The backfill report recorded `inference_audit_dashboard_rebuilt`, and browser inspection of `http://127.0.0.1:8991/?refresh=...` confirmed links to both `toric_embedding_sidecar.html` and `toric_embedding_sidecar.json` from the main audit page.
- Ran safe cleanup only for temp/cache paths used by tests (`/tmp/pytest-of-iska`, dedicated pytest basetemps, `.pytest_cache`, `__pycache__`, `/home/iska/.cache/tropicalgt`, `/home/iska/.cache/ms-playwright`). Outputs, checkpoints, W&B folders, datasets, and generated training bundles were not deleted.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_backfill_interactive_audit_artifacts.py TropicalGT-I/tests/test_sample_browser_index.py TropicalGT-I/tests/test_parameter_golf_review_loop.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-focused
# 22 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-validator
# 10 passed
```

Operational note: the first test attempt failed before code execution because the root filesystem had only about 2 MB free and pytest could not allocate `/tmp` directories. After safe cache cleanup, focused tests ran cleanly. The live b60 validator still reports known legacy audit failures for insufficient rows and older no-proxy contracts; the navigation repair itself is verified by source tests, live backfill action, and browser inspection.

Next linear item: continue the no-proxy visual/math repair queue while b62 remains active toward the 5K gate; prioritize the next source-visible gap that Herschel's 5K report must inventory.


## 2026-06-16 Periodic Audit Retention Hardening Pass

Sequential operational/source hardening item completed after the toric sidecar navigation pass:

- b62 failed at step 1000 while writing periodic inference-audit artifacts with `OSError: [Errno 28] No space left on device`; the monitor recorded `fatal_marker_observed_no_automatic_restart` and did not claim a 5K completion.
- Preserved b62 failure evidence under `TropicalGT-I/outputs/training_stop_records/preserved_b62_failure_20260616T162800Z/`, then removed the failed generated b62 output bundle and partial checkpoint to restore about 65 GB free. This did not delete datasets, W&B folders, source, planning docs, b60 evidence, or secrets.
- Launched fresh step-0 b63 run `tropicalgt_i_pg_bpb_step0_full24b_b63_20260616T174503Z_fresh_bpb112_alwayson_5k_gate` with trainer PID `1048659`, monitor PID `1048773`, W&B id `4vhilcwc`, config `TropicalGT-I/outputs/launch_configs/tropicalgt_i_pg_bpb_step0_full24b_b63_20260616T174503Z_fresh_bpb112_alwayson_5k_gate.json`, log `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b63_20260616T174503Z_fresh_bpb112_alwayson_5k_gate/logs/train_nohup_20260616T174503Z.log`, and stop record `TropicalGT-I/outputs/training_stop_records/b63_fresh_step0_alwayson_step5000_gate.json`.
- b63 preserves the advanced training stack and changes only generated artifact retention: periodic audits still run, but retained `got_audit` bundles are bounded to latest plus final 5K, and final periodic evaluation keeps at least three detail rows.
- Added an opt-in source retention cap `periodic_prune_got_audit_max_retained_steps`. When set, it can cap configured keep-steps while preserving newest audits, and each prune action records the exact retention policy and capped configured steps in the manifest. This is an artifact-retention guard only; it does not alter model losses, metrics, CAS certificates, or mathematical evidence.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_metrics.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-retention
# 14 passed
```

Next linear item: monitor b63 through its early periodic checkpoint and continue the no-proxy visual/math repair queue.


## 2026-06-16 Analogical Top-K No-Proxy Contract Pass

Sequential visual/math repair item completed after periodic retention hardening:

- Added an explicit `tropicalgt.analogical_topk.v1` contract to `analogical_simplicial_maps.json` for both available and unavailable analogical-memory outputs.
- The contract records the no-proxy policy, required `trajectory_probability_filtered_simplicial_object` query/codomain complexes, model-probability-vector requirement, Jensen-Shannon probability assignment metric, rendered/qualified/rejected top-k counts, quality-gate metadata when reported, and the rule that chain maps or persistence-module morphisms are reported only after a certified filtered simplicial map.
- Updated the analogical top-k HTML index with a visible contract panel and a rigorous early-training unavailable state. When the memory bank is empty or lacks qualified model-probability trajectory complexes, the page now says exactly why no vertex assignment, simplex-tree map, chain map, or persistence-module morphism is rendered.
- Preserved the existing no-fallback behavior: embedding-only `probability_filtered_simplicial_object` rows are still rejected, missing query probabilities still make the report unavailable, and unavailable vectors/certificates remain unavailable rather than being fabricated.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical -k "analogical_memory"
# 6 passed, 36 deselected
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
```

Operational note: b63 reached step 250/5000 with train loss/NLL about 1.558/1.536, no traceback or ENOSPC, and `periodic/step_00000250` was generated. That first periodic audit is large, so disk pressure around step 500 remains an active operational watch item for Herschel and the main controller.

Next linear item: continue the no-proxy visual/math repair queue while b63 trains toward 5K; keep watching periodic retention/disk pressure before each source-side change that might generate artifacts.


## 2026-06-16 Deep Cleanup Pass

User-requested cleanup pass completed immediately after the analogical top-k contract work:

- Confirmed a blanket ignored-file cleanup was unsafe because `git clean -ndX` would target important ignored paths including `TropicalGT-I/data/`, `TropicalGT-I/checkpoints/`, `TropicalGT-I/outputs/`, `wandb/`, and `keys.txt`.
- Performed targeted cleanup only: removed pytest temp directories, temporary validator reports, Python bytecode caches under source/test/script/planning paths, and repo-local test caches where present.
- Pruned the already-completed non-final b63 step-250 generated periodic audit payload and left `PRUNED_NONFINAL_AUDIT_PAYLOAD.json` as a compact no-proxy marker. This did not touch datasets, checkpoints, W&B folders, source files, configs, logs, secrets, or final 5K required artifacts.
- Disk recovered from about 45 GB free to about 61 GB free, and b63 stayed active. Latest parsed post-cleanup health was step 304/5000 with train loss/NLL about 1.533/1.510 and no traceback or ENOSPC.

Next operational watch: step 500 will generate the next periodic audit. If it again creates very large non-final payloads, the controller/Herschel should verify retention or prune non-final generated payloads promptly while preserving the final 5K gate requirements.


## 2026-06-16 Reasoning-Step Complex Manifest Contract Pass

Sequential visual/math repair item completed after the deep cleanup pass:

- Added a `tropicalgt.reasoning_step_complex_maps.v1` manifest contract to `reasoning_step_complex_maps/manifest.json`.
- The contract records that each listed model-evaluated GoT state has its own per-step radius-filtered complex page and simplex-tree/explicit-unavailable page, and that the global embedding trajectory PCA map is not used as a proxy for those complexes.
- Each step row now carries explicit complex/simplex-tree render contracts plus simplex-tree backend/availability metadata, so downstream validators and Herschel can count real GUDHI SimplexTree evidence versus unavailable states without inferring from HTML filenames.
- The per-step index page now exposes the no-proxy contract visibly and clarifies that the step complexes are not reconstructed from the global trajectory PCA surface.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-contract -k "got_trajectory_visualization_renders_simplicial_panel_and_nll_surface"
# 1 passed, 41 deselected
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
```

Operational note: b63 remained active during validation and was parsed at step 412/5000 with train loss/NLL about 1.463/1.442 and no traceback or ENOSPC.

Next linear item: continue the visual/math repair queue, with an immediate operational check around step 500 to confirm large non-final periodic artifacts do not threaten the 5K run.

## 2026-06-16 CAS Free-Resolution Certificate Contract Repair

Sequential CAS/no-proxy item completed after the analogical, simplex-tree, NLL-density, and tropical-support render-contract repairs:

- Added a `tropicalgt.cas_free_resolution_contract.v1` report contract to real free-resolution outputs. The contract names the required input schema, no-proxy policy, safe render rule, unavailable render rule, and backend capability boundaries for Macaulay2, Singular, Sage, and BEMultipliers.
- Unavailable CAS reports now carry the same contract as certified reports, so disabled, missing-backend, timeout, complexity-guard, parse-error, and certificate-failed states can still be rendered with exact no-proxy rules and backend reasons.
- Certified Macaulay2 reports now expose the same contract both at top level and inside `cas_artifacts.certificate_summary`, tying safe multigraded rendering to exactness, minimality, homogeneous-presentation, and parsed multidegree evidence.
- The contract explicitly records that BEMultipliers is a post-certificate Buchsbaum-Eisenbud diagnostic path only, not a substitute free-resolution backend.
- Regression tests require the contract on unavailable and certified paths and verify deterministic preservation through cached unavailable probes.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-contract -k "real_cas_free_resolution_disabled_by_environment or real_cas_free_resolution_caches_deterministic_unavailable_probe or certified_cas_result_surfaces_buchsbaum_eisenbud_diagnostics"
# 3 passed, 27 deselected
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-algebraic-full
# 30 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
```

## 2026-06-16 Two-Parameter Module Visual Contract Repair

Sequential visual/math item started after the CAS free-resolution contract repair:

- Added a first-class `tropicalgt.two_parameter_module_staircase_contract.v1` JSON contract to the two-parameter bifiltration visual payload.
- The contract makes the Miller-Sturmfels staircase the enforceable primary view, records that rank slabs are removed as the primary representation, and keeps 3D rank/fiber plots labeled as secondary diagnostics only.
- The contract requires raw bifiltration fields (`fiber_rank_profile`, `chain_module_generators`, `rank_invariant_samples`, `structure_maps`, and `grid_fiber_provenance`) and points reviewers back to the exact raw bifiltration JSON.
- Each staircase card now declares `tropicalgt.two_parameter_staircase_card.v1`, its actual generator-bidegree source, x_radius/x_level orientation, shaded upward-closed submodule semantics, white quotient-basis lattice-point semantics, and product-order minimal-antichain boundary source.
- Regression coverage locks down the no-proxy/staircase semantics so future rank surfaces or chain diagnostics cannot silently become the primary module or a fake free resolution.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-two-param-contract -k "level_radius_bifiltration_reports_scoped_real_staircase_resolution"
# 1 passed, 29 deselected
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-algebraic-two-param-full
# 30 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
```

## 2026-06-16 Derived Similarity Conservative-Minimum Contract Repair

Sequential analogical/derived item started after the two-parameter module visual contract repair:

- Added explicit derived/algebraic similarity policy metadata to analogical topology summaries.
- `derived_algebraic_similarity` is now documented as the conservative minimum of signature cosine, chain/free-resolution similarity, persistent-homology similarity, and commutative-algebra similarity only when all components have nonzero evidence; otherwise it is forced to `0.0`.
- Added a `free_resolution_similarity` alias for the chain/free-resolution evidence component and a structured `derived_algebraic_components` object for reviewers and Herschel reports.
- Added `coarse_signature_cosine_not_derived_similarity` plus `high_coarse_signature_low_resolution_warning` so high signature cosine with zero chain/free-resolution support is reported as an invariant collision, not a derived/algebraic match.
- Regression coverage now constructs a high-signature/zero-free-chain case and requires derived/algebraic similarity to remain zero.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-derived-contract -k "topological_similarity_summary_separates_coarse_signature_from_derived_claim or derived_comparison_requires_matching_certified_cas_artifacts"
# 2 passed, 41 deselected
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-simplicial-derived-full
# 43 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
```

## 2026-06-16 Persistence Landscape Evidence Contract Repair

Sequential topology/analogical item started after the derived-similarity conservative-minimum repair:

- Added `tropicalgt.persistence_landscape_evidence.v1` to persistence-landscape vector similarity reports.
- The contract names the mathematical object as sampled GUDHI `Landscape` `lambda_k(t)` vectors by homology dimension and explicitly marks it distinct from GoT NLL/fitness/density fields.
- Unavailable landscape comparisons now carry `unavailable_state_is_not_zero_vector=true` and `zero_vector_substitution_allowed=false`; they contribute zero score only because evidence is unavailable, not because a zero-valued landscape vector was fabricated.
- Available landscape comparisons carry the same source contract while exposing vector-space cosine/L2/correlation over cached GUDHI vectors.
- The analogical top-k prose now distinguishes unavailable landscape evidence from zero-valued vectors.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-landscape-contract -k "persistence_landscape_vectors or analogical_memory_visualization_requires_probability_filtered_complex"
# 1 passed, 53 deselected
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-landscape-full
# 54 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/memory.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_metrics_and_memory.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
```

## 2026-06-16 GraphCG Direction Readability Contract Repair

Sequential GraphCG visual item started after the persistence-landscape evidence contract repair:

- Added `tropicalgt.graphcg_direction_readability.v1` to `graphcg_direction_cosines_payload.json`.
- The structured contract requires four separate panels: all-direction heatmap, full-rank activity spectrum, candidate activity by observed GoT state, and signed-bias scatter.
- The contract records that every model-derived GraphCG direction is rendered, the heatmap does not sample directions, visible tick labels are bounded for readability, and exact direction ids plus candidate path/action text are preserved in hover and payload fields.
- Added `candidate_hover_rows` to the GraphCG payload so long path/action labels remain reviewable even when visible axis labels are compact.
- The interactive audit validator now rejects GraphCG payloads missing the structured readability contract or candidate hover rows.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-graphcg-contract -k "graphcg"
# 3 passed, 51 deselected
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-validator-graphcg-full
# 11 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-graphcg-full
# 54 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
```

## 2026-06-16 Tropical Support Readability Contract Repair

Sequential visual/math item completed after the GraphCG direction-readability contract:

- Added a machine-readable `tropicalgt.tropical_support_readability.v1` contract to `tropical_support_payload.json` and Plotly metadata.
- The contract records split panel roles for observed support assignments, selected-margin profiles, wall-threshold overlays, token-group summaries, support-frequency summaries, and collapse diagnostics.
- It explicitly states that assignment cells, model support probabilities, selected tropical margins, collapse summaries, and wall-margin audits are separate evidence channels; invalid `active_support_index` rows are not converted into fabricated support cells.
- The interactive artifact validator now rejects tropical support payloads that lose the readability contract, merge assignment and margin evidence, omit panel roles, drop hover/payload preservation, or claim certified normal-fan wall crossings from margin-threshold telemetry.
- Renderer and validator coverage prove the observed-support matrix and collapse-diagnostic layouts expose the required contract fields.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-support-readability -k "tropical_support"
# 7 passed, 48 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-support-validator-full
# 12 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-support-simplicial-full
# 43 passed
```

## 2026-06-16 Analogical Top-K Readability Contract Repair

Sequential analogical-memory item completed after the tropical-support readability contract:

- Added a machine-readable `tropicalgt.analogical_topk_readability.v1` contract inside the analogical top-k contract emitted to `analogical_simplicial_maps.json`.
- The contract requires a readable top-k table, one linked pair/map view for each rendered rank, explicit insufficient-memory states, displayed quality-gate/filter counts, and separate columns for retrieval score, probability-JS assignment, PH/vector topology, chain/commutative algebra, map claim, conservative derived/algebraic score, coarse signature cosine, simplex-tree preservation, and edge certificates.
- The HTML index now explicitly names the index readability contract and reiterates that retrieval, probability-JS assignment, topological, algebraic, map-claim, simplex-tree, and edge-certificate evidence remain separate columns.
- The interactive artifact validator now rejects analogical outputs with missing/wrong top-k contract schema, embedding-only assignment allowance, non-Jensen-Shannon assignment source, missing readability contract, hidden insufficient-memory state, missing table/link requirements, merged evidence columns, missing map-claim/simplex-tree/preservation columns, or overclaimed derived/algebraic similarity from coarse signature cosine.
- Renderer and validator coverage prove both available top-k outputs and unavailable/insufficient-memory outputs carry the no-proxy readability contract.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-topk -k "analogical"
# 11 passed, 45 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-validator-full
# 13 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-simplicial-full
# 43 passed
```

## 2026-06-16 Analogical Simplex-Tree Analogy View Repair

Sequential analogical-memory item completed after the top-k readability contract:

- Added `analogical_simplex_tree_analogy.html` and `analogical_simplex_tree_analogy.json` generation to analogical memory visualization outputs.
- Added a machine-readable `tropicalgt.analogical_simplex_tree_analogy.v1` contract proving the view is built from `probability_simplicial_map.simplex_tree_map.rows`, not from embedding-only correspondence or invented geometry.
- The view compares query and memory GUDHI SimplexTree finite Hasse rows, lists domain simplex to image simplex checks, highlights preserved rows, labels missing/filtration-distorted rows as failed correspondences, and summarizes preserved face-to-coface chains.
- Chain-map and persistence-module morphism claims remain gated by certified filtered simplicial-map preservation; failed or incomplete simplex-tree checks are explicitly not promoted.
- The interactive artifact validator now requires the simplex-tree analogy HTML/JSON, checks contract schema/source/no-proxy flags, requires Hasse face-to-coface rows and preserved-chain metadata, and verifies pair counts match rendered analogical maps.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-tree -k "analogical"
# 12 passed, 45 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-tree-validator-full
# 14 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-tree-simplicial-full
# 43 passed
```

## 2026-06-16 Reasoning-Step Complex Fingerprint Contract Repair

Sequential simplicial/simplex-tree item completed after the analogical simplex-tree analogy view repair:

- Added canonical per-reasoning-step filtered-complex fingerprints to every `reasoning_step_complex_maps` step row.
- The fingerprint basis uses `gudhi_canonical_complex(filtered_simplicial_object)` evidence, including simplex rows, filtration values, probability-vector provenance, and SimplexTree backend metadata.
- Record ids and output paths are explicitly excluded from the hash basis so a repeated fingerprint means an identical canonical filtered-complex payload, not a duplicated filename.
- The manifest now reports fingerprint source, per-step fingerprint presence, unique fingerprint count, uniqueness status, and duplicate fingerprint groups with the required identical-payload caveat.
- The HTML index and contract panel expose the fingerprint evidence so reviewers can see whether each reasoning step has its own actual radius-filtered complex.
- The interactive artifact validator now rejects missing per-step fingerprints, missing fingerprint bases, bad duplicate accounting, bad fingerprint-basis schema/source, or duplicate-group claims without the identical-payload caveat.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-fingerprints -k "reasoning_step or validate_audit_root_rejects_missing_reasoning_step"
# 1 passed, 57 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-validator-full
# 15 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-simplicial-full
# 43 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
```

## 2026-06-16 Radius Filtration Slider Contract Repair

Sequential simplicial/simplex-tree item completed after the reasoning-step fingerprint contract:

- Added `tropicalgt.radius_filtration_slider_contract.v1` sidecars beside every generated complex slider page.
- The sidecar records that the page is built from canonical GUDHI filtered-complex simplices, uses no proxy/fallback data, orders thresholds ascending min-to-max, and exposes one frame per displayed threshold.
- The contract proves the initial radius frame is a disjoint vertex-only frame: no solid edges, no filled faces, and no dotted causal/decoding/direction overlays.
- The contract records monotone visible counts, solid-line semantics for radius-filtered 1-simplices, filled-face semantics for radius-gated 2-simplices, and dotted-line semantics for gated causal/decoding/direction overlays.
- The interactive artifact validator now requires slider sidecars for the full trajectory complex, available probability trajectory complex, and every reasoning-step complex page.
- Validator rejection coverage now fails audits missing a reasoning-step slider contract; renderer coverage now checks sidecar schema, first-frame vertex-only counts, and monotonicity.
- The renderer test fixture was updated so per-step `filtered_simplicial_object` payloads are real embedding-radius Vietoris-Rips complexes instead of legacy graph-combinatorial objects.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-slider-contract -k "got_trajectory_visualization_renders_simplicial_panel_and_nll_surface or slider_contract or reasoning_step"
# 4 passed, 55 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-slider-validator-full
# 16 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-slider-simplicial-full
# 43 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
```


## 2026-06-16 CAS Fitting/BE Paper-Method Contract Repair

Sequential CAS/no-proxy item completed after the radius filtration slider contract:

- Reviewed `references/2210.11433v1.pdf` locally with `pdftotext` and confirmed the applicable method hooks: presentation spaces/rank invariants/Fitting ideals, determinantal minors for rank strata, generic free resolutions, and Buchsbaum-Eisenbud multipliers as complementary-minor sidecars for certified complexes.
- Added `tropicalgt.be_fitting_method_contract.v1` to `cas_free_resolution.py` and embedded it in the CAS free-resolution certificate contract.
- Certified and unavailable CAS reports now carry the paper-method contract at top level; cached older CAS reports are hydrated in memory so they expose the current contract without rerunning CAS.
- CAS artifact subreports for Fitting/minor diagnostics, Buchsbaum-Eisenbud rank conditions, grade/depth/regular diagnostics, and BEMultipliers output now inherit the same paper-method/no-proxy contract.
- The contract explicitly records that Fitting/minor/rank-strata evidence and BEMultipliers output are diagnostics only: they cannot substitute for a CAS-certified free resolution, regular-sequence certificate, derived equivalence, or exactness claim.
- Regression coverage now checks unavailable reports, parsed Singular/Macaulay2-style outputs, cached reports, ideal diagnostics, BE diagnostics, and the two-parameter bifiltration CAS path for the `2210.11433v1` method contract.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-paper-contract -k "real_cas_free_resolution_disabled_by_environment or tagged or fitting or bemultipliers or macaulay2_rejects_nonhomogeneous"
# 5 passed, 25 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-paper-full
# 30 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
```

## 2026-06-16 Same-Data Static Preview Terminology Repair

Sequential browser/visual no-fallback item completed after the CAS Fitting/BE paper-method contract repair:

- Renamed the WebGL-unavailable rendering contingency in generated Plotly pages from fallback terminology to same-data static-preview terminology.
- Updated the generated CSS/JS hooks from `webgl-fallback`, `main-fallback`, `staticFallbackMarkup`, `renderPanelStaticFallback`, and `promoteMainStaticFallback` to `webgl-static-preview`, `main-static-preview`, `staticPreviewMarkup`, `renderPanelStaticPreview`, and `promoteMainStaticPreview`.
- The rendered disclosure now says `Static SVG same-data preview from the same filtered-complex payload`, making clear that the preview is drawn from the same serialized real filtered-complex payload and does not substitute proxy geometry, model states, metrics, or topology.
- The legacy provenance audit coverage remains in place so old rendering-fallback strings are still caught and classified if they reappear, but current generated browser pages no longer expose fallback wording for this same-payload rendering path.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_plotly_dark_html_promotes_static_preview_for_webgl_failures -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-static-preview
# 1 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-static-preview-full
# 43 passed
```

## 2026-06-16 Simplicial Projection Evidence Boundary Repair

Sequential simplicial visualization item completed after the same-data static-preview terminology repair:

- The 3D radius-filtered simplicial SVG layout now records `coordinate_evidence`, `safe_for_metric_claims`, `vector_rows`, and `display_metadata_rows` in the rendered layout contract string.
- When every vertex carries a real vector such as an embedding, probability vector, coordinate vector, or model feature vector, the layout reports `coordinate_evidence=real_vertex_vectors` and `safe_for_metric_claims=true`.
- When vertex vectors are absent or only partially present, the layout remains usable as a display-only arrangement from vertex metadata, but it reports `safe_for_metric_claims=false`; these coordinates cannot be read as model PCA, embedding geometry, topology metrics, or training evidence.
- The helper previously named `_vertex_numeric_feature` is now `_vertex_display_layout_feature`, making the code boundary match the no-proxy policy.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-layout-evidence -k "simplicial_svg_wraps_long_topological_paths or simplicial_svg_reports_model_vector_projection_evidence"
# 2 passed, 42 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-layout-evidence-full
# 44 passed
```

## 2026-06-16 Simplicial Projection Provenance Rename

Sequential provenance/no-proxy item completed after the simplicial projection evidence-boundary repair:

- Renamed the provenance registry entry from `simplicial_projection_feature_fallback` to `simplicial_projection_display_layout_evidence`.
- Changed the entry kind from `visual_projection_fallback` to `visual_display_layout_boundary` so audit output matches the current renderer contract.
- The guardrail now says coordinates are metric/model evidence only when rendered with `coordinate_evidence=real_vertex_vectors` and `safe_for_metric_claims=true`; metadata-only coordinates remain visual arrangement only.
- Legacy fallback-era match terms remain in the audit entry solely so stale code is still caught and classified if it reappears.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_metric_provenance.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metric_provenance.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-provenance-layout
# 7 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-layout-provenance-viz -k "simplicial_svg_wraps_long_topological_paths or simplicial_svg_reports_model_vector_projection_evidence"
# 2 passed, 42 deselected
```

## 2026-06-16 Browser Static Preview Provenance Rename

Sequential provenance/no-proxy item completed after the simplicial projection provenance rename:

- Renamed the canonical browser rendering provenance entry from `browser_static_preview_rendering_fallback` to `browser_same_data_static_preview_rendering`.
- Changed the entry kind from `rendering_fallback` to `same_data_rendering_contingency`, matching the current generated `webgl-static-preview` / `staticPreviewMarkup` UI contract.
- Updated the standalone metric-provenance audit helper with current same-data hook names while retaining retired fallback-era names as legacy match terms, so stale generated renderer code is still classified instead of becoming an uncovered risk-word finding.
- Updated the full-dataset visualization telemetry plan to describe static SVG panels as same-data rendering contingencies rather than data or geometry fallbacks.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/scripts/audit_metric_provenance.py TropicalGT-I/tests/test_metric_provenance.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metric_provenance.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-static-preview-provenance
# 7 passed
```

## 2026-06-16 Buchsbaum-Eisenbud Implied-Rank Diagnostic Rename

Sequential CAS/provenance no-proxy item completed after the browser static-preview provenance rename:

- Renamed the Buchsbaum-Eisenbud rank-condition artifact key from `image_rank_estimates_by_differential` to `image_rank_values_implied_by_exact_rank_identity`.
- The rendered diagnostic table now labels those values as `implied_rank=` so they are not presented as independently certified image-rank computations.
- Added a compatibility reader for cached artifacts that still contain the retired key, while new CAS reports write only the exact-rank-identity key.
- Added provenance coverage marking this surface as `diagnostic_rank_identity_not_certificate`; audit terms keep the retired key visible only as a legacy/cached-artifact guardrail.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_metric_provenance.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metric_provenance.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-be-rank-provenance
# 7 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-be-rank-full
# 75 passed
```

## 2026-06-16 Graph JSON Legacy Guardrail Trace Label Repair

Sequential browser/readout no-proxy item completed after the Buchsbaum-Eisenbud implied-rank diagnostic rename:

- Kept the stored `graph_json_fallback_rate` metric key as a legacy must-remain-zero readiness guardrail for backwards-compatible reports and gates.
- Changed the Plotly training-metrics trace name to `legacy graph-json substitution guardrail (must remain zero)` so active browser readouts no longer present the retired fallback wording as an active graph-data path.
- Added regression coverage to ensure the raw legacy metric key is not used as the visible Plotly trace name while GraphCG priority metric ordering remains unchanged.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_training_metrics.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_metrics.py::test_browser_metric_visualization_prioritizes_graphcg_rank_audit -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-graph-json-label
# 1 passed
```

## 2026-06-16 Graph JSON W&B Legacy Guardrail Alias

Sequential training-readout no-proxy item completed after the browser trace-label repair:

- Kept the stored `graph_json_fallback_rate` history/eval/readiness key as a backwards-compatible legacy must-remain-zero guardrail.
- Renamed the W&B-visible priority key to `06_graph_data/legacy_graph_json_substitution_guardrail_rate`, so dashboards no longer present the retired fallback term as an active graph-data path.
- Preserved the active graph-data evidence keys `graph_json_derived_text_graph_rate` and `graph_json_parse_unavailable_rate` unchanged.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_training_metrics.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_metrics.py::test_wandb_metrics_are_namespaced_by_priority -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-wandb-graph-json-alias
# 1 passed
```

## 2026-06-16 Graph JSON Readiness And Review Guardrail Alias

Sequential readiness/review no-proxy item completed after the W&B alias repair:

- Kept the JSON readiness key `graph_json_fallback_rate` stable as a legacy must-remain-zero compatibility field.
- Changed readiness markdown and gate details to say `legacy graph-json substitution guardrail` instead of `legacy fallback`.
- Changed the 5K review-loop `data_metrics` contract field to `legacy_graph_json_substitution_guardrail_rate`, while preserving active graph-data fields such as `graph_json_sequentialized_rate`.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py TropicalGT-I/scripts/parameter_golf_codex_review_loop.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_parameter_golf_review_loop.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_readiness_audit.py::test_readiness_audit_fixture_without_checkpoint TropicalGT-I/tests/test_parameter_golf_review_loop.py::test_active_training_contract_reports_losses_and_graph_order_metrics -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-readiness-review-graph-json-alias
# 2 passed
```

## 2026-06-16 Graph JSON Validator Guardrail Alias

Sequential validation/readout no-proxy item completed after the readiness/review guardrail alias:

- Added `legacy_graph_json_substitution_guardrail_records` and `legacy_graph_json_substitution_guardrail_rate` to `validate_tropicalgt_i.py` output.
- Preserved compatibility fields `graph_json_fallback_records` and `invalid_graph_rate` so existing validators and generated reports keep reading old artifacts.
- Added a CLI regression proving the new alias is emitted and agrees with the compatibility rate on fixture data.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/validate_tropicalgt_i.py TropicalGT-I/tests/test_readiness_audit.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_readiness_audit.py::test_validate_tropicalgt_i_reports_legacy_graph_json_guardrail_alias -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-validate-graph-json-alias
# 1 passed
```

## 2026-06-16 Graph JSON Training Report Guardrail Alias

Sequential training/eval readout no-proxy item completed after the validator guardrail alias:

- Added `legacy_graph_json_substitution_guardrail_rate` to live training history rows while preserving `graph_json_fallback_rate` as a compatibility key.
- Added `legacy_graph_json_substitution_guardrail_records` and `legacy_graph_json_substitution_guardrail_rate` to final eval reports while preserving `graph_json_fallback_records` and `invalid_graph_rate`.
- Added regression coverage proving the new aliases agree with the compatibility fields on a tiny CPU training run.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_training_metrics.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_metrics.py::test_training_history_contains_certificate_and_throughput_metrics -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-training-report-graph-json-alias
# 1 passed
```

## 2026-06-16 Graph JSON Per-Record Guardrail Alias

Sequential per-record diagnostics no-proxy item completed after the training report guardrail alias:

- Added `legacy_graph_json_substitution_guardrail` beside the compatibility `graph_json_fallback` flag in per-record diagnostic rows.
- Added the same explicit guardrail alias to `validate_tropicalgt_i.py` sample rows, preserving the old field for generated-artifact compatibility.
- Added regressions for both model diagnostic rows and validator sample output.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/diagnostics.py TropicalGT-I/scripts/validate_tropicalgt_i.py TropicalGT-I/tests/test_diagnostics.py TropicalGT-I/tests/test_readiness_audit.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_diagnostics.py::test_record_gfn_graphcg_diagnostics TropicalGT-I/tests/test_readiness_audit.py::test_validate_tropicalgt_i_reports_legacy_graph_json_guardrail_alias -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-per-record-graph-json-alias
# 2 passed
```

## 2026-06-16 W&B Uncategorized Metric Namespace Rename

Sequential W&B/readout terminology item completed after the graph JSON per-record guardrail alias:

- Renamed `_wandb_fallback_group` to `_wandb_uncategorized_metric_group` so unmatched scalar metrics are described as uncategorized logging output, not a fallback path.
- Renamed the provenance entry to `wandb_uncategorized_metric_namespace` with kind `logging_namespace`, while keeping retired fallback-era terms as audit-only match terms.
- Added regression coverage proving unknown scalar metrics still route to `99_other/*` without changing priority namespace behavior.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_metric_provenance.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_metrics.py::test_wandb_metrics_are_namespaced_by_priority TropicalGT-I/tests/test_metric_provenance.py::test_metric_provenance_registry_covers_current_risky_terms -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-wandb-uncategorized-namespace
# 2 passed
```

## 2026-06-16 Multipers Signed-Measure Backend Terminology Repair

Sequential multiparameter backend terminology item completed after the W&B uncategorized namespace rename:

- Reworded the unavailable `multipers` recommendation to `optional signed-measure backend diagnostics` rather than approximation language.
- Renamed provenance entry `multipers_backend_approximation` to `multipers_optional_signed_measure_backend` with kind `optional_backend_diagnostic`.
- Kept retired approximation phrases as audit-only match terms so stale wording remains detectable.
- Added an unavailable-path regression that forces `multipers` import failure and checks the exact recommendation wording.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/algebra.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_metric_provenance.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_multipers_unavailable_status_uses_signed_measure_backend_language TropicalGT-I/tests/test_metric_provenance.py::test_metric_provenance_registry_covers_current_risky_terms -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-multipers-signed-measure
# 2 passed
```


## 2026-06-16 Two-Parameter Structure-Map Evidence Contract

Sequential two-parameter bifiltration visual repair completed after the multipers signed-measure terminology repair:

- Added a first-class `tropicalgt.two_parameter_structure_maps.v1` summary to `two_parameter_bifiltration.json` so adjacent `x_level` and `x_radius` structure maps are inspectable as machine-readable evidence, not only hover text.
- The summary records raw `bifiltration.structure_maps` provenance, F2 field counts, east/north direction counts, valid source/target bidegrees in `[x_level, x_radius]` order, per-map H0/H1 rank rows, and an explicit no-proxy/no-fallback flag.
- Tightened the interactive artifact validator so the raw bifiltration payload must contain actual F2 east and north structure maps, and the visual summary must match the raw map counts exactly.
- Added regression coverage for the generated bifiltration visual payload and for validator rejection when the structure-map summary is missing.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_level_radius_bifiltration_reports_scoped_real_staircase_resolution TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_bifiltration_structure_map_summary -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-two-param-structure-summary
# 2 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-two-param-structure-summary-full
# 48 passed
```


## 2026-06-16 Primary Structure-Map Overlay

Sequential two-parameter bifiltration visual repair completed after the structure-map evidence contract:

- Added actual `x_level` and `x_radius` F2 structure-map segment overlays to the module-lattice Plotly diagnostic, with hover text sourced from raw `bifiltration.structure_maps` H0/H1 ranks.
- Added a primary-page “Adjacent F2 structure maps from raw bifiltration” evidence table so the Miller-Sturmfels page visibly exposes the adjacent map data without relying on secondary hover state.
- Added `tropicalgt.primary_structure_map_evidence.v1` to the visual payload and validator so the rendered page must cite raw structure maps, render both directions, keep exact table counts, and assert no proxy/fallback evidence.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_level_radius_bifiltration_reports_scoped_real_staircase_resolution TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_bifiltration_structure_map_summary -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-primary-structure-overlay
# 3 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-primary-structure-overlay-full
# 48 passed
```


## 2026-06-16 CAS Execution Manifest

Sequential CAS integration repair completed after the primary structure-map overlay:

- Added `tropicalgt.cas_execution_manifest.v1` to real free-resolution reports so every unavailable or certified CAS state records backend order, template keys and SHA-256 hashes, executable availability, version probes, presentation shape, complexity-guard status, and certificate-before-render policy.
- The manifest includes the BEMultipliers policy as an optional post-certificate diagnostic sidecar and keeps `no_proxy_or_fallback=true` so command templates cannot be mistaken for proof.
- Cache hydration now supplies a conservative legacy manifest when an older cached result lacks the field, while new cache entries persist the full manifest.
- Added regressions for disabled execution, cached unavailable probes, and complexity-guard skips.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_real_cas_free_resolution_disabled_by_environment TropicalGT-I/tests/test_algebraic_persistence.py::test_real_cas_free_resolution_caches_deterministic_unavailable_probe TropicalGT-I/tests/test_algebraic_persistence.py::test_real_cas_free_resolution_complexity_guard_caches_deterministic_skip -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-execution-manifest
# 3 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-execution-manifest-full
# 31 passed
```


## 2026-06-16 Analogical Probability Assignment Evidence

Sequential analogical memory/map repair completed after the CAS execution manifest:

- Added `tropicalgt.probability_vector_assignment_evidence.v1` to probability simplicial map diagnostics so each analogical map report explicitly records model probability-vector source, Jensen-Shannon assignment metric, assignment solver, displayed query/memory probability-vector counts, and `embedding_only_assignment_used=false`.
- Added assignment metric/solver fields to vertex-map rows and retrieval metrics, making the probability-vector assignment path auditable from both the detailed map and the scalar training/readout rows.
- Added regressions proving certified probability simplicial maps and retrieval hits expose the probability-vector evidence contract and preserve the no-proxy/no-fallback policy.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/memory.py TropicalGT-I/tests/test_metrics_and_memory.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py::test_probability_simplicial_map_diagnostics_certifies_filtered_chain_map TropicalGT-I/tests/test_metrics_and_memory.py::test_analogical_memory_retrieval_uses_probability_simplicial_map_weight TropicalGT-I/tests/test_metrics_and_memory.py::test_analogical_memory_probability_map_must_preserve_simplex_tree_to_score -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-probability-evidence
# 3 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-probability-evidence-full-rerun
# 11 passed
```

## 2026-06-16 Reasoning-Step Radius Slider Manifest Evidence

Sequential simplicial/simplex-tree repair completed after the analogical probability assignment evidence contract:

- Added `tropicalgt.reasoning_step_radius_slider_summary.v1` summaries to every `reasoning_step_complex_maps/manifest.json` step row, sourced from the actual per-step `reasoning_step_*_slider_contract.json` sidecar.
- The manifest contract now counts rendered per-step slider summaries and requires all reasoning-step radius sliders to be actual-data/no-proxy, start as disjoint vertices, grow monotonically, hide solid edges/faces/dotted overlays on the initial radius frame, and stay safe to render.
- The reasoning-step index now exposes radius-slider verification in its contract panel instead of relying on separate sidecar discovery.
- The interactive artifact validator now cross-checks each manifest slider summary against the corresponding sidecar and rejects missing, mismatched, unsafe, nonmonotone, or proxy-permitting summaries.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_reasoning_step_slider_summary -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-slider-manifest
# 3 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-slider-manifest-full
# 63 passed
```

## 2026-06-16 Reasoning-Step Complex Source Contract

Sequential simplicial/simplex-tree repair completed after the radius-slider manifest evidence contract:

- Added `tropicalgt.reasoning_step_complex_source_contract.v1` to every reasoning-step manifest row so each step page records its source as that candidate's own `filtered_simplicial_object`.
- The source contract records candidate record id, level, path, canonical fingerprint, displayed vertex/edge/face counts, probability-vector and embedding vertex counts, vertex-label samples, SimplexTree backend/availability, and explicit non-proxy flags for global trajectory, embedding-map, and static probability-complex substitutes.
- The manifest contract now aggregates source-contract counts and requires candidate-owned filtered-object source, no trajectory/static proxy, summary-count agreement, nonempty vertex evidence, and safe renderability for every step.
- The interactive artifact validator now cross-checks each source contract against the manifest row and rejects missing source contracts, proxy claims, count mismatches, and unsafe per-step source claims.
- Added a shared simplex-dimension normalizer for manifest fingerprints and source contracts so canonical vertex rows are classified from either explicit dimension metadata or simplex cardinality.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_reasoning_step_source_contract TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_reasoning_step_source_proxy_claim -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-source-contract
# 4 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-step-source-contract-full
# 65 passed
```

## 2026-06-16 Persistence Landscape Visual Contract

Sequential persistence-landscape repair completed after the reasoning-step source contract:

- Added `persistence_landscapes.json` beside `trajectory_persistence/persistence_landscapes.html` with schema `tropicalgt.persistence_landscape_visual_contract.v1`.
- The sidecar proves the page renders actual `topology.persistence_representations.methods[*].landscape` rows, marks the artifact actual-data/no-proxy, distinguishes GUDHI `lambda_k(t)` vectors from the GoT NLL/fitness landscape, and rejects norm-only summaries as sufficient evidence.
- The renderer now plots vector-only GUDHI Landscape outputs by splitting `landscape.vector` with `num_landscapes` and `resolution`; when no explicit filtration grid is exported, the contract labels the x-coordinate as a normalized landscape sample index rather than inventing a filtration grid.
- Non-growth persistence pages now emit an explicit unavailable landscape sidecar instead of leaving the landscape JSON absent.
- The interactive artifact validator now requires the trajectory persistence landscape sidecar and rejects missing, unavailable, proxy/fallback, NLL-confused, norm-only, or zero-curve landscape payloads.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_trajectory_persistence_uses_growth_and_chain_presentation_diagnostics TropicalGT-I/tests/test_simplicial_visualization.py::test_non_growth_persistence_landscape_is_explicitly_unavailable TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_persistence_landscape_payload TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_norm_only_persistence_landscape_payload -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-landscape-contract
# 5 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-landscape-contract-full
# 67 passed
```

## 2026-06-16 Trajectory Complex Overlay Contract

Sequential simplicial/overlay repair completed after the persistence-landscape visual contract:

- Added `tropicalgt.trajectory_complex_overlay_contract.v1` to `got_full_trajectory_complex_payload.json` so the full trajectory complex payload explicitly separates solid radius-filtered simplices from dotted GoT trajectory/decoding-order overlays.
- Added per-view `tropicalgt.trajectory_complex_overlay_view_contract.v1` records for the embedding-radius trajectory complex and, when available, the Jensen-Shannon probability trajectory complex.
- The contract records distance metric, filtration model, vertex/edge/face counts, SimplexTree backend, trajectory overlay source, decoding-order overlay source, dotted/directed overlay checks, and no-proxy/no-fallback flags.
- The Jensen-Shannon probability SimplexTree page title now explicitly names the probability metric when rendered from model candidate probability vectors.
- The interactive artifact validator now rejects missing overlay contracts, embedding/probability metric mismatches, unsafe overlay semantics, and probability-view contracts that silently substitute embedding/static probability proxies.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_trajectory_overlay_contract TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_probability_overlay_metric_mismatch -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-trajectory-overlay-contract
# 4 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-trajectory-overlay-contract-full
# 69 passed
```

## 2026-06-16 GraphCG Direction Evidence Contract

Sequential GraphCG visual repair completed after the trajectory complex overlay contract:

- Added row-wise `direction_rows` to `graphcg_direction_cosines_payload.json`, one row per model-derived GraphCG direction.
- Added `tropicalgt.graphcg_direction_evidence.v1` so the payload certifies exact direction ids, mean absolute cosines, signed mean cosines, activity ranks, and whether every direction is rendered in the heatmap, full-rank activity spectrum, and signed-bias panel.
- The interactive artifact validator now rejects missing direction-evidence contracts, direction-row count gaps, missing exact ids, non-finite per-direction values, and panel coverage gaps.
- This keeps GraphCG evidence no-proxy/no-fallback and makes the full-rank direction spectrum auditable without reconstructing meaning from parallel arrays.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_graphcg_visualization_preserves_projection_basis_certificate TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_graphcg_direction_evidence_contract TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_graphcg_direction_row_gap -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-graphcg-direction-evidence
# 4 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-graphcg-direction-evidence-full
# 71 passed
```

## 2026-06-16 Miller-Sturmfels Staircase Aggregate Evidence

Sequential two-parameter persistence repair continued after the GraphCG direction evidence contract:

- Added `tropicalgt.miller_sturmfels_staircase_evidence.v1` to `trajectory_persistence/two_parameter_bifiltration.json` so the primary Miller-Sturmfels view is certified by an aggregate record, not just inferred from individual cards.
- The aggregate cites only `bifiltration.chain_module_generators[*].multidegree`, declares actual-data/no-proxy rendering over `F2[x_level,x_radius]`, records x_radius horizontal / x_level vertical axes, and explicitly records the `rho_x_radius` and `rho_x_level` one-dimensional cone labels.
- The evidence cross-checks card count, primary card index, actual generator totals, minimal-antichain totals, generator-label totals, upward-closed region totals, quotient-basis lattice totals, Hilbert numerator totals, and adjacent-LCM syzygy totals.
- The validator now rejects missing Miller-Sturmfels aggregate evidence, proxy/fallback aggregate claims, coordinate-axis gaps, unsafe theorem-scope boundaries, missing per-card aggregate rows, and aggregate count mismatches.
- This keeps the two-parameter page within the no-fallback rule: exact staircase evidence is shown when backed by actual bifiltration data, and no chain diagnostic is promoted into a fake free resolution.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_level_radius_bifiltration_reports_scoped_real_staircase_resolution TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_miller_sturmfels_staircase_evidence TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_miller_sturmfels_staircase_aggregate_mismatch -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-ms-staircase-evidence
# 4 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-ms-staircase-evidence-full
# 59 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-ms-staircase-evidence-visual-full
# 73 passed
```

## 2026-06-16 Nested CAS Guard Validator Contract

Sequential real-CAS integration repair began after the Miller-Sturmfels aggregate evidence contract:

- Added an interactive-artifact validator guard for nested `tropicalgt.real_free_resolution.v1` sidecars inside `trajectory_level_radius_bifiltration.json` chain-presentation diagnostics.
- The validator now requires the CAS certificate contract, BE/Fitting paper method contract, CAS execution manifest, backend entries with `certificate_required_before_rendering`, command templates, and strict no-proxy/no-fallback flags before an audit artifact can pass.
- Unavailable CAS states must now carry an exact reason, empty `cas_artifacts`, false certification/render flags, `safe_unavailable_render=true`, and an unavailable diagnostic whose policy forbids substituting chain diagnostics, rank samples, Fitting ideals, minors, or BEMultipliers output for a certified free resolution.
- Certified CAS states must carry attached certificates, exactness, a real free-resolution summary, artifacts, and grading-consistent safe-render flags; total/ungraded output cannot be silently promoted to multigraded persistence-module output.
- Added validator rejection tests for missing nested CAS guards and proxy-like render/artifact flags on unavailable CAS states.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_bifiltration_real_cas_guard TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_bifiltration_real_cas_proxy_flags -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-guard-validator
# 3 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-guard-validator-full
# 30 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-cas-guard-algebra
# 31 passed
```

## 2026-06-16 Analogical Probability-Vector Evidence Propagation

Sequential analogical-memory repair began after the nested CAS guard validator contract:

- Propagated `tropicalgt.probability_vector_assignment_evidence.v1` from retrieval-side probability simplicial-map diagnostics into each rendered analogical map report.
- The map report now carries explicit source, model-probability vector source, Jensen-Shannon assignment metric, assignment solver, probability alignment, displayed query/memory vertex counts, all-displayed-vertices-have-probabilities flags, and no-proxy/embedding-only guards.
- The interactive artifact validator now rejects analogical maps that omit probability-vector evidence, use an embedding-only assignment, use the wrong metric/source, fail to certify model probabilities on all displayed query/memory vertices, or report vertex counts inconsistent with the rendered complexes.
- This tightens the analogical-memory contract so top-k map pages cannot pass by showing Jensen-Shannon summaries alone; they must carry the underlying model-probability vector evidence used to solve the assignment.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_analogical_probability_js_provenance_gaps -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-prob-evidence
# 2 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_analogical_memory_visualization_renders_simplicial_maps TropicalGT-I/tests/test_simplicial_visualization.py::test_analogical_memory_visualization_requires_retrieval_probability_map_certificate TropicalGT-I/tests/test_simplicial_visualization.py::test_analogical_memory_visualization_labels_failed_probability_correspondence_not_map TropicalGT-I/tests/test_simplicial_visualization.py::test_analogical_memory_visualization_rejects_non_trajectory_probability_fallback TropicalGT-I/tests/test_simplicial_visualization.py::test_analogical_memory_without_retrieval_emits_unavailable_surfaces TropicalGT-I/tests/test_simplicial_visualization.py::test_analogical_memory_without_query_probabilities_is_unavailable_not_fallback -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-prob-evidence-viz
# 6 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py TropicalGT-I/tests/test_simplicial_visualization.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-prob-evidence-full
# 75 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py::test_probability_simplicial_map_diagnostics_certifies_filtered_chain_map TropicalGT-I/tests/test_metrics_and_memory.py::test_analogical_memory_retrieval_uses_probability_simplicial_map_weight -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-analogical-prob-evidence-memory
# 2 passed
```

## 2026-06-16 Simplex-Tree Poset Sidecar Contract

Sequential simplicial/simplex-tree repair continued after analogical probability-vector evidence propagation:

- Added standalone `tropicalgt.simplex_tree_poset.v1` JSON sidecars for every simplex-tree face-to-coface poset page, rather than relying only on Plotly HTML metadata.
- The sidecar certifies GUDHI-backed source, actual-data/no-proxy flags, empty-simplex root presence, primary actual face-to-coface cover edges, legend-only sorted-label trie prefix links, model-embedding barycentric position source, and safety for rendering.
- Unavailable simplex-tree pages now write explicit unavailable sidecars with exact reason, zero displayed simplices, safe-unavailable render, and no-proxy flags instead of leaving validator behavior to infer from dark HTML.
- Reasoning-step manifests now embed each step's poset contract, list the sidecar filename, and summarize whether all per-step posets are GUDHI-backed, no-proxy, face-to-coface-primary, and safe to render.
- The interactive artifact validator now requires full trajectory, probability trajectory, and per-step simplex-tree poset sidecars; missing, unsafe, proxy-like, non-GUDHI, disconnected-column, or non-face-to-coface-primary contracts fail validation.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_simplex_tree_page_is_unavailable_without_gudhi_not_raw_json TropicalGT-I/tests/test_simplicial_visualization.py::test_simplex_tree_poset_contract_records_actual_face_coface_covers TropicalGT-I/tests/test_simplicial_visualization.py::test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_accepts_three_interactive_rows TropicalGT-I/tests/test_interactive_artifact_validator.py::test_validate_audit_root_rejects_missing_reasoning_step_simplex_tree_poset_contract -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-simplex-tree-poset-sidecar
# 5 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-simplex-tree-poset-sidecar-full
# 76 passed
```

## 2026-06-16 Monomial Transport and Flat-Incidence Bundle Metrics

Sequential vector-bundle/toric implementation continued after simplex-tree poset sidecars:

- Extended `ChartBundleToricHead` with explicit monomial-transport permutation logits and chart-by-toric-row flat-incidence logits, both isolated from token logits and zero-default in the training objective.
- Added `bundle_monomial_transport_permutation_loss`, `bundle_monomial_transport_permutation_one_hotness`, `bundle_monomial_transport_available`, `bundle_flat_incidence_binary_defect`, `bundle_flat_incidence_mean`, and `bundle_flat_incidence_available` metrics.
- Added zero-default config weights `bundle_monomial_transport_weight` and `bundle_flat_incidence_weight`; they contribute only when explicitly nonzero, preserving BPB-first training and logits/loss equality for telemetry-only chart-bundle runs.
- Added structured `tropicalgt.chart_bundle_transport_metadata.v1` metadata with nested `tropicalgt.monomial_transport_head.v1` and `tropicalgt.bundle_matroid_flat_incidence.v1` contracts, including transport ids, chart ids, flat-incidence shape, and no-proxy flags.
- Extended training metric grouping so Herschel/review reports can see the new monomial-transport and flat-incidence metrics/loss sidecars when chart-bundle telemetry is enabled.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/model.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_losses_and_model.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_losses_and_model.py::test_chart_bundle_auxiliary_zero_weights_do_not_change_logits_or_loss TropicalGT-I/tests/test_losses_and_model.py::test_chart_bundle_bpb_partition_unavailable_without_targets TropicalGT-I/tests/test_losses_and_model.py::test_chart_bundle_auxiliary_positive_weights_change_loss_not_logits -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-bundle-monomial-transport
# 3 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_bpb_ablation_grid.py TropicalGT-I/tests/test_bpb_ablation.py -q -p no:cacheprovider --basetemp=/tmp/tropicalgt-pytest-bundle-monomial-transport-full
# 14 passed
```

## 2026-06-16 Chart-Bundle Transport Browser Sidecar

Sequential vector-bundle/toric browser hardening continued after monomial transport and flat-incidence bundle metrics:

- Added `chart_bundle_transport_sidecar.html` and `chart_bundle_transport_sidecar.json` to new audit bundles written by `write_inference_audit_artifacts`.
- The sidecar searches only exported `tropicalgt.chart_bundle_transport_metadata.v1` payloads in model/result/GoT candidate outputs and renders unavailable when that metadata is absent.
- The payload exposes chart ids, overlap pair/triple counts, sample transport ids, monomial-transport contracts, bundle matroid/flat-incidence contracts, toric-certificate status, and strict actual-data/no-proxy flags.
- The render contract explicitly says chart-bundle telemetry is not a toric embedding, tropical-variety embedding, global toric-variety embedding, or normal-fan certificate; those still require separate CAS-certified sidecars.
- The sample browser now links the chart-bundle transport sidecar when present.
- The interactive artifact validator now checks any present chart-bundle transport sidecar for schema, no-proxy flags, non-toric/non-normal-fan safety flags, overlap count consistency, and nested no-proxy transport/matroid contracts.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/scripts/build_sample_browser_index.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py TropicalGT-I/tests/test_sample_browser_index.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_simplicial_visualization.py -k "chart_bundle_transport_sidecar or toric_embedding_sidecar"
# 4 passed, 43 deselected
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_sample_browser_index.py
# 2 passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_interactive_artifact_validator.py -k "chart_bundle_sidecar_proxy_claim or accepts_three_interactive_rows"
# 2 passed, 30 deselected
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_simplicial_visualization.py tests/test_interactive_artifact_validator.py tests/test_sample_browser_index.py
# 81 passed
```

## 2026-06-18 GPU-Safe Herschel Launch Gate

Highest-priority training automation repair completed for the current GPU-sharing constraint:

- Added `tropicalgt.gpu_launch_safety_contract.v1` in `TropicalGT-I/src/tropicalgt/launch_safety.py`.
- `parameter_golf_codex_review_loop.py` now records `gpu_launch_safety_contract.json` and refuses non-dry-run training launch unless explicit GPU clearance or a declared memory budget plus clearance note is present.
- `run_advanced_bpb_campaign.py` now records `advanced_bpb_campaign_gpu_launch_safety.json` and refuses campaign training launch under the same policy.
- Dry-run, prepare-only, once/review, and CPU-only report paths remain available without GPU clearance.
- This does not inspect GPU state or reserve memory; it is a launch/restart policy gate so TropicalGT-I automation cannot surprise-interfere with the user's other GPU training.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/src/tropicalgt/launch_safety.py TropicalGT-I/scripts/parameter_golf_codex_review_loop.py TropicalGT-I/scripts/run_advanced_bpb_campaign.py TropicalGT-I/tests/test_launch_safety.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_launch_safety.py tests/test_parameter_golf_review_loop.py
# 22 passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_prepare_5k_review_bundle.py
# 9 passed
```

## 2026-06-18 CPU-Only Herschel 5K Report Generation

Herschel's 5K evidence report path is now implemented as a deterministic CPU-only artifact generator over the existing post-5K review bundle. `prepare_5k_review_bundle.py` writes the normal no-proxy bundle, calls `write_herschel_5k_report.py`, then records `herschel_report_markdown`, `herschel_report_json`, and `herschel_report_summary` back into the bundle. The report includes run identity, BPB/graph-BPB, checkpoint availability and restart safety, execution readiness, advanced BPB gate failures, grouped advanced sidecar counts, restart blockers, and a Mermaid restart-decision diagram. It does not train, evaluate, browse, validate, load checkpoints, inspect GPUs, or synthesize missing evidence.

Blocked evidence remains blocked. Empty or unavailable checkpoints, missing post-5K command results, failed advanced BPB gates, and failed validators are copied into the report as exact blockers; no checkpoint-backed or sidecar-backed restart can be justified by this report unless the underlying bundle evidence already permits it. This keeps Herschel useful while the user may be running another GPU job.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/tests/test_herschel_5k_report.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_herschel_5k_report.py tests/test_prepare_5k_review_bundle.py tests/test_launch_safety.py tests/test_parameter_golf_review_loop.py
# 34 passed in 1.17s
```

## 2026-06-18 BPB-First Promotion Gates And Matched 5K Ablations

BPB-first ablation promotion is now stricter and evidence-backed:

- `build_bpb_ablation_report` now treats nonzero core advanced coefficients (`gflownet_weight`, `graphcg_weight`, `margin_weight`, `entropy_weight`, `certificate_weight`, `sequence_tropical_weight`) and chart-bundle/toric coefficients as promotion candidates.
- Promotion requires matched held-out `eval_bpb` and `eval_graph_bpb` improvement plus real held-out certificate and tropical-wall guardrail evidence. Missing or regressing `eval_certificate_*` or `eval_*wall*`/margin metrics blocks promotion; no train-history proxy or fallback can justify a coefficient increase.
- `evaluate_model` now aggregates actual scalar validation outputs for certificate and tropical-wall guardrails and leaves them absent when unavailable. Training reports persist those as `eval_*` metrics.
- `run_bpb_ablation_grid.py` now writes `tropicalgt.bpb_ablation_match_contract.v1` into the manifest and every generated config, with 5K boundary metadata, requested max steps, seed, data identity, graph-BPB side weight, and a base-config fingerprint. Training reports copy this contract forward so later analysis rejects mismatched runs.
- The analyzer rejects candidates whose match contract differs from baseline even if BPB improves and guardrails pass.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/src/tropicalgt/ablation.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/scripts/run_bpb_ablation_grid.py TropicalGT-I/tests/test_bpb_ablation.py TropicalGT-I/tests/test_bpb_ablation_grid.py TropicalGT-I/tests/test_training_metrics.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_bpb_ablation.py tests/test_bpb_ablation_grid.py tests/test_training_metrics.py
# 19 passed in 5.20s
```

## 2026-06-18 Advanced BPB Readiness Contract Strict Gates

Advanced BPB config readiness now has explicit optional strict gates for the next high-rigor training iterations:

- `require_graphcg_active_full_rank=true` adds `advanced_bpb_graphcg_active_directions_full_rank`, requiring active GraphCG directions to cover the embedding dimension rather than only requiring a positive active subset.
- `require_nontrivial_bundle_toric_losses=true` adds gates for chart-bundle auxiliary enablement and nonzero bundle/toric loss weights across transport, cocycle, flat-rank, normal-fan, GraphCG-cell agreement, chart-BPB consistency, and atom-stability coefficients.
- `advanced_bpb_max_visual_audit_interval` makes the visual-audit cadence gate configurable while preserving the default 250-step policy.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/src/tropicalgt/readiness_contracts.py TropicalGT-I/tests/test_readiness_audit.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_readiness_audit.py tests/test_prepare_5k_review_bundle.py tests/test_parameter_golf_review_loop.py tests/test_training_resume.py
# 45 passed in 2.96s
```

## 2026-06-18 Herschel Vector-Bundle And Sheaf-Derived Sidecar Inventory

Sequential 5K-review/reporting item completed after the vector-bundle paper-sidecar repair:

- `write_herschel_5k_report.py` now uses multi-label advanced sidecar grouping instead of a single exclusive bucket.
- The CPU-only 5K report keeps existing CAS/topology/analogical/tropical/GraphCG/NLL/chart-bundle counts and now adds explicit `vector_bundle` and `sheaf_derived` group counts.
- `chart_bundle_transport_sidecar.json` is counted as both chart-bundle transport evidence and vector-bundle paper-sidecar evidence, so Herschel can inventory the new paper-sidecar fields at the next 5K review without reading the full report manually.
- Derived/sheaf/chain-map sidecars are counted under `sheaf_derived`, while unmatched paths remain in `other`; no GPU, browser, checkpoint, validator, or training work is executed by the report generator.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_herschel_5k_report.py -q
# 2 passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/tests/test_herschel_5k_report.py
# passed
```

## 2026-06-18 Two-Parameter Certificate-Indexed CAS Evidence Validator Guard

Sequential two-parameter QA item completed:

- The two-parameter bifiltration visual payload now has validator coverage for `certificate_indexed_cas_evidence`.
- The validator requires schema `tropicalgt.cas_certificate_indexed_evidence.v1`, `no_proxy_or_fallback=true`, and either real exactness-certified evidence with `evidence_blocks` plus the derived-category guard, or a safe-unavailable block with an explicit reason.
- The validator fixture now includes the safe-unavailable certificate-indexed evidence block for CAS-unavailable rows, matching the browser payload generated by the visualizer.
- Added negative tests for missing certificate-indexed CAS evidence and unsafe/proxy unavailable evidence.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/scripts:/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_interactive_artifact_validator.py -k "certificate_indexed_cas_evidence or missing_miller_sturmfels or bifiltration_structure_map_summary"
# 4 passed, 30 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/scripts:/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_interactive_artifact_validator.py
# 34 passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
```

Next linear item: continue two-parameter and analogical-map repair without weakening actual-data-only, simplex-tree, or CAS no-proxy contracts.

## 2026-06-18 Vector-Bundle Paper Sidecar Completeness Tiers

Sequential vector-bundle/browser validation item completed after the Herschel validator-action pass:

- `_build_vector_bundle_paper_sidecar` now emits `tropicalgt.vector_bundle_paper_sidecar_completeness.v1` alongside `tropicalgt.vector_bundle_paper_sidecar.v1`.
- The completeness contract distinguishes `unavailable`, `telemetry_partial`, and `paper_ready` tiers. Basic chart-bundle telemetry can be available without being safe as paper-ready vector-bundle evidence.
- `paper_ready` requires all actual-data groups: chart/monomial-transport ids, configured toric active rows, flat-incidence diagnostics, GraphCG-toric agreement, and transported persistence-landscape metrics.
- Browser HTML exposes the completeness tier, paper-ready flag, and structured contract in the chart-bundle sidecar table.
- The strict interactive-artifact validator now rejects missing completeness contracts, invalid tiers, paper-ready flag disagreements, and paper-ready claims with missing required groups.
- Certificate boundaries remain unchanged: the sidecar still cannot claim a vector-bundle theorem certificate, toric/tropical embedding certificate, normal-fan certificate, scheme/sheaf construction, or CAS-backed proof.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_simplicial_visualization.py -k "chart_bundle_transport_sidecar"
# 3 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_interactive_artifact_validator.py -k "chart_bundle or vector_bundle"
# 2 passed
```

## 2026-06-18 Tropical/Toric CAS Input Contracts

Sequential tropical/toric no-proxy item completed after vector-bundle paper sidecar completeness tiers:

- Tropical fan sidecars now emit `tropicalgt.tropical_fan_input_contract.v1`, recording whether an explicit model-derived ideal was present, where it came from, required `variables`/`generators`, accepted keys, input hash, and rejected proxy sources.
- Finite toric-ideal sidecars now emit `tropicalgt.toric_embedding_input_contract.v1`, recording whether an explicit integer exponent matrix was present, where it came from, accepted keys, input hash, and rejected proxy sources.
- Certified and unavailable browser tables expose the input contract so reviewers can see whether the CAS sidecar is blocked by missing input, missing backend certificate, or a real unavailable state.
- The interactive artifact validator now requires those contracts for present tropical/toric sidecars and rejects proxy-allowing contracts, missing hashes on available sidecars, or available sidecars without explicit CAS input evidence.
- This does not promote support-token traces, chart-bundle activations, GraphCG cells, embeddings, chain ranks, or visualization rows into tropical fans or toric ideals. Those remain rejected proxy sources.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_simplicial_visualization.py -k "tropical_fan_diagnostics or toric_embedding_sidecar"
# 4 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_interactive_artifact_validator.py -k "tropical_fan_input_contract or toric_input_contract or accepts_three_interactive_rows"
# 3 passed
```

## 2026-06-18 Analogical Query Context Conversion Contract

Sequential analogical-map no-proxy item completed after the tropical/toric CAS input contract pass:

- The analogical memory writer now emits `tropicalgt.analogical_query_context_conversion.v1` for both available and unavailable outputs.
- Query domains are accepted only from `trajectory_probability_filtered_simplicial_object` with real model probability vectors. The contract records `probability_filtered_simplicial_object` and `filtered_simplicial_object` as rejected query-context keys when present, including whether they had probability vertices and why they were not accepted.
- The top-k contract embeds the query-context contract and the HTML contract panel now shows the query conversion status, selected source, probability-vertex count, and rejected keys.
- Missing model-probability query domains remain explicit unavailable states; they do not trigger embedding-only assignment, alias conversion, pseudo maps, chain maps, persistence-module morphisms, or derived-category claims.
- The success JSON payload keeps the accepted query source, derived signature, top-k contract, simplex-tree analogy contract, and query-context conversion contract together for downstream Herschel/report review.

Validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/pytest -q TropicalGT-I/tests/test_simplicial_visualization.py -k "analogical_memory_visualization or analogical_memory_without"
# 6 passed, 44 deselected
```

## 2026-06-18 Analogical Query Context Validator Gate

Sequential validator hardening item completed after the analogical query-context writer pass:

- `validate_interactive_audit_artifacts.py` now requires `tropicalgt.analogical_query_context_conversion.v1` for analogical map payloads.
- The validator checks actual-data/no-proxy flags, accepted query keys, rejected-key row shape, alias-fallback rejection flags, embedding-only rejection, and the Jensen-Shannon model-probability assignment metric.
- Available analogical maps must select `trajectory_probability_filtered_simplicial_object` and report positive query probability-vertex counts.
- Missing-query unavailable payloads must keep `selected_query_complex_available=false` and `selected_query_complex_source=unavailable`.
- The fixture includes a rejected alias row so validation preserves the distinction between recording rejected evidence and selecting an alias as a query domain.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/pytest -q TropicalGT-I/tests/test_interactive_artifact_validator.py -k "analogical_query_context or analogical_topk_readability or analogical_simplex_tree_analogy or accepts_three_interactive_rows"
# 5 passed, 40 deselected
```

## 2026-06-18 Herschel Analogical Query Context Evidence

Sequential 5K-reporting item completed after the validator gate:

- `write_herschel_5k_report.py` now extracts analogical query-context evidence from recorded `analogical_simplicial_maps.json` sidecars.
- The report records whether the sidecar exists, whether the required contract schema is present, whether the top-k contract embeds the same query-context contract, the selected query complex source, selected probability-vertex count, conversion status, rejected query-context keys, topological-algebra availability, and no-proxy/embedding-only policy flags.
- Missing sidecars and parser failures remain explicit unavailable evidence. Herschel does not synthesize query-domain status from validator prose, embeddings, or memory rows.
- Markdown and HTML output now include `Analogical Query Context Evidence`, so the 5K review can explain why alias query domains were rejected before any restart decision.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/pytest -q TropicalGT-I/tests/test_herschel_5k_report.py
# 2 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_interactive_artifact_validator.py TropicalGT-I/tests/test_metric_provenance.py
# 54 passed
```

## 2026-06-18 5K Bundle Analogical Sidecar Inventory

Sequential Herschel/reporting item completed after the analogical query-context evidence pass:

- `prepare_5k_review_bundle.py` now scans the active contract's `latest_got_audit_dir` for real Herschel-required sidecars before any contract, prompt, review bundle, or Herschel report artifact is written.
- The first required sidecar is `analogical_simplicial_maps.json`, so Herschel's 5K report can consume the actual `tropicalgt.analogical_query_context_conversion.v1` contract when the sidecar exists even if a bounded generic artifact tail would otherwise miss it.
- The generated inventory records `herschel_required_sidecars_present` as a concrete path list. Absence stays absence; the code does not create, copy, or promote missing sidecars.
- The regression fixture proves the bundle, persisted active contract, and Herschel summary agree on the same analogical query-domain evidence.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/scripts/write_herschel_5k_report.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_herschel_5k_report.py
# 11 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=331 covered=331 uncovered=0
```
