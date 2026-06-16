# TropicalGT-I Current Repair Inventory - 2026-06-14

## Operating Rules

- Work only on `iska` in `/home/iska/Documents/amelie/bio/TropicalGT`.
- Stay on the non-main branch `tropicalgt-i-real-cas-no-proxy-20260614` or a newer non-main branch created from it.
- Proceed linearly: finish the current repair, test it, inspect it in the browser, then move to the next item.
- Replace proxy or synthetic claims when encountered. If a required computation is not yet implemented, the next task is the real implementation, not a fake figure.
- Generated browser artifacts may be regenerated for QA, but source, planning, tests, and docs are the commit targets.

## Current Objective

Repair `trajectory_persistence/two_parameter_bifiltration.html` so it is mathematically faithful and readable:

1. The z-axis remains the actual fiber rank `beta_i = dim_F2 H_i(K_(level,radius))`.
2. H0 and H1 occupy distinguishable, low, close visual layers without changing the reported rank.
3. The displayed radius coordinate is normalized only for readability; the exact radius grade stays in hover and JSON.
4. The `F2[x_level,x_radius]` module grid shows actual fibers and structure-map edges.
5. The Miller-Sturmfels panel is an exponent-lattice/staircase diagram for the scoped monomial ideal, not a full toric embedding claim.
6. Coordinate one dimensional cone(s) terminology is used only where it accurately denotes the coordinate generators of the exponent semigroup.

## Implementation List

### 1. Bifiltration and 2-Parameter Persistence

- Guarantee every nonempty reasoning trajectory emits an `F2[x_level,x_radius]` bifiltration.
- Compute grid fibers `K_(level,radius)` from the actual filtered complexes.
- Compute structure maps on adjacent grid fibers in the `x_level` and `x_radius` directions.
- Render a 3D lattice view with `x_level`, normalized/display `x_radius`, and actual fiber rank on z.
- Keep exact radius grades, rank-invariant samples, Betti surfaces, and fiber provenance in downloadable JSON.
- Add tests proving every nonempty trajectory has level grades, radius grades, fibers, and module ring metadata.

### 2. Review `references/2210.11433v1.pdf`

- Extract and read the full paper.
- Write `planning/tropicalgt_i_2210_11433_cas_methodology_review.md`.
- Identify transferable methods: fitting ideals, determinantal minors, Buchsbaum-Eisenbud multipliers, exactness criteria, multigraded resolutions, and module invariants.
- Separate what applies to two-parameter persistence modules over `F2[x_level,x_radius]` from what applies only to auxiliary ideals or unrelated rings.

### 3. CAS Computations and Certified Free Resolutions

- Probe and integrate Macaulay2, SageMath, Singular, and optional `amelie-iska/BEMultipliers` in that order.
- Compute real minimal multigraded free resolutions over `F2[x_level,x_radius]` only when a backend emits exactness, minimality, multidegree-shift, and differential-matrix evidence.
- Implement Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, Betti tables, differential matrices, multidegree shifts, and certificate payloads.
- Render Macaulay2/Sage-style research figures for certified algebraic objects.
- Never render chain-presentation diagnostics as certified free resolutions.

### 4. Derived, Analogical, and Memory Maps

- Compute analogical retrieval from model-predicted probability vectors.
- Use Jensen-Shannon assignments between probability-vector filtered complexes.
- Validate vertex, edge, face, simplex-tree, and bifiltration preservation before calling a correspondence a simplicial map.
- Compare derived objects only through certified free-resolution or chain-map evidence.
- Retrieve many top-k memories only when stored memories pass the quality threshold; otherwise render an explicit insufficient-memory state.

### 5. Simplicial Complexes and Simplex Trees

- Radius filtrations must start as disjoint vertices and grow min-to-max by adding edges and faces.
- Dotted directed edges show causal direction, decoding order, or both; solid simplicial edges/faces are radius-gated.
- Every reasoning step gets its own filtered complex and simplex tree.
- Simplex tree plots must show a connected trie or face-coface poset, not disconnected columns.
- Hover payloads must include model input/output, simplex, filtration value, dimension, and provenance.

### 6. NLL/Fitness Visuals

- Replace misleading sparse triangular surfaces with a 3D PCA Gaussian density cloud around actual model embeddings.
- Color density by local NLL/fitness while keeping actual trajectory vertices and edges separate.
- Keep 2D/3D surfaces only when they interpolate the neighborhood around the trajectory and visibly anchor the model vertices.
- Show edgewise NLL improvements and terminal improvement.

### 7. Tropical Support, GraphCG, and Decoding

- Make tropical support heatmaps interpretable with grouped labels, collapse diagnostics, wall-hit context, and margin summaries.
- Fix tropical margin sign/scale naming so negative values are not mislabeled as a loss unless that is the intended maximization convention.
- Investigate rising certificate loss and distinguish real certificate failures from auxiliary penalties.
- Ensure GraphCG directions are full rank relative to embedding dimension and render readable spectra/activity/bias plots.
- Keep meet-in-the-middle decoding behind a config toggle.
- Decode causal DAGs with forward and reverse causal directions; use ROAR/random-order autoregression for cyclic or noncausal graphs.

### 8. Tropical/Toric Embedding and Paper Updates

- Research Macaulay2 `Tropical`, Sage tropical polynomial and tropical variety APIs, `references/1710.10651v2.pdf`, and Maclagan-style embeddings of tropical schemes into toric varieties.
- Implement toric embeddings only when fan, one dimensional cone(s), semigroup, and embedding maps are mathematically correct and tool-backed.
- Add vector-bundle, tropical toric embedding, scheme, and sheaf-theoretic material to the TropicalGT-I paper as implementations mature.

### 9. Training, Browser QA, Docs, and Push

- Reconcile training state and restart/resume BPB training only under the user-requested 5K gate policy or explicit instruction.
- Generate periodic browser audit artifacts every 250 steps.
- Keep `127.0.0.1:8991` or the active forwarded port pointed at the latest repaired bundle.
- Inspect every repaired visualization in the browser.
- Update `README.md`, planning docs, and tests.
- Push source/planning/docs changes to the non-main branch.

## Present Status

- Current branch: `tropicalgt-i-real-cas-no-proxy-20260614`.
- Active training: fresh b59 step-0 run `tropicalgt_i_pg_bpb_step0_full24b_b59_20260616T001122Z_fresh_bpb112_5k_gate`, PID `73189`, W&B `itxgxj40`, config under `TropicalGT-I/outputs/launch_configs/`, output dir under `TropicalGT-I/outputs/`.
- b59 is under the 5K evidence gate: do not restart before step `5000` unless it crashes, OOMs, or logs nonfinite/invalid losses. Galileo (`019ecdc9-a3b8-7761-b0e5-36e6adad8e15`) owns the post-5K analysis/visualization review and evidence-backed step-0 restart.
- Focused tests currently pass for certified CAS/analogical memory/training metrics paths after commit `70c2e93`.
- Tropical support readability now includes strict/near-wall margin diagnostics, per-token wall buckets, and threshold guide traces after the 2026-06-16 pass; remaining visual/math items are NLL/fitness density polish, simplex-tree/browser inspection, and CAS-rendered algebra panels as real backend evidence becomes available.

## 2026-06-14 Bifiltration Figure Repair Pass

The current renderer for `trajectory_persistence/two_parameter_bifiltration.html` now separates three objects that had been visually conflated:

1. The top 3D panel is the actual finite `F2[x_level,x_radius]` fiber-rank lattice.  The z coordinate is always the computed Betti rank `beta_i = dim_F2 H_i(K_(level,radius))`.  H0 and H1 are separated only by a small visual offset in the displayed radius coordinate; the exact radius grade remains in hover and JSON.
2. The middle module panel is a Miller-Sturmfels-style exponent-lattice diagram built from the same actual grid fibers.  It renders zero H1 fibers, nonzero H1 fibers, actual H1 rank-change staircase curves, and actual multigraded chain-generator bidegrees from the chain-presentation diagnostics.  The displayed `x_radius` coordinate is normalized for readability, with exact grades in hover.
3. The algebra tables remain diagnostic unless a CAS backend attaches a certificate for an actual minimal multigraded free resolution.  Chain-presentation objects are not labeled as certified free resolutions.

Focused validation after this pass:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 17 passed
```

The next linear item is the full review of `references/2210.11433v1.pdf`, followed by exact CAS integration for Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, and certified minimal multigraded free resolutions over `F2[x_level,x_radius]` where the backend certifies exactness and minimality.

## 2026-06-14 Bifiltration Figure Repair Pass 2

Status: implemented and regenerated for `TropicalGT-I/outputs/multi_sample_browser/latest/sample_000/trajectory_persistence/two_parameter_bifiltration.html`.

Changes made:
- Replaced the cramped two-panel 2-parameter module plot with a stacked dark-mode research figure.
- First panel now follows the Miller-Sturmfels staircase convention for a bivariate monomial ideal: lattice coordinates are bidegrees over `F2[x_level,x_radius]`, the support staircase is drawn from actual nonzero H1 grid fibers, and coordinate axes are labeled as one dimensional cone(s).
- Second panel keeps the actual fiber rank as the third dimension, with only a small visual y-offset separating H0 and H1 layers when they occupy the same lattice coordinates.
- Hover payloads report the actual level, radius grade, H0/H1 rank, and chain-generator bidegrees from the generated bifiltration payload.
- CAS/free-resolution tables remain explicitly diagnostic unless populated by certified CAS output; no page text claims certification without CAS evidence.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/src/tropicalgt/algebra.py` passed.
- `PYTHONPATH=TropicalGT-I/src python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` passed: 17 tests.
- Browser QA opened the regenerated page at `http://127.0.0.1:8991/sample_000/trajectory_persistence/two_parameter_bifiltration.html`. DOM checks confirmed the visible strings `2-parameter module fibers`, `H0 fiber rank`, `Miller-Sturmfels staircase`, and `one dimensional cone(s)`.

Next sequential item:
- Review `references/2210.11433v1.pdf` in detail, then implement the CAS-backed computations for minimal multigraded free resolutions, Fitting ideals, Buchsbaum-Eisenbud diagnostics, and derived/analogical maps.

## 2026-06-14 Bifiltration Figure Repair Pass 3

Status: implemented and browser-reviewed at `http://127.0.0.1:8991/sample_000/trajectory_persistence/two_parameter_bifiltration.html`.

Changes made:
- Replaced the compressed split/flat-looking module panel with a Miller-Sturmfels-style exponent-lattice diagram over `F2[x_level,x_radius]`.
- The main panel now renders actual grid fibers / monomial lattice points, actual `H0` and `H1` fiber-rank samples, actual `C_0`, `C_1`, and `C_2` shifted free-chain-module generator bidegrees, minimal multidegree antichain / staircase, and dotted coordinate one dimensional cone(s).
- Shaded orthants now come only from actual shifted free-chain-module bidegrees. The page no longer treats arbitrary `H_i` fiber-rank support as a monomial ideal support.
- The companion 3D panel keeps fiber rank as the z-axis and uses only a small visual y-offset to distinguish `H0` from `H1` layers.

Browser QA evidence:
- DOM text contains `Miller-Sturmfels-style multigraded`, `shifted free-chain-module supports`, `minimal multidegree antichain`, and `one dimensional cone`.
- Rendered legend contains: actual grid fibers / monomial lattice; H0 fiber-rank samples; H1 fiber-rank samples; C_0/C_1/C_2 shifted free-module generators; minimal multidegree antichain / staircase; coordinate one dimensional cone(s); H0/H1 fiber-rank lattice; H0/H1 module structure maps.
- Focused test command passed: `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `17 passed`.

Next sequential item:
- CAS integration for real multigraded free resolutions, Fitting ideals, minors, and Buchsbaum-Eisenbud diagnostics. Update the CAS plan with the completed `2210.11433v1.pdf` review and then implement the Macaulay2/Singular/BEMultipliers bridge item-by-item.



## 2026-06-14 Bifiltration Figure Repair Pass 4

Status: implemented, regenerated, and opened in the in-app browser at `http://127.0.0.1:8991/sample_000/trajectory_persistence/two_parameter_bifiltration.html`.

Correction made from browser review:
- The Miller-Sturmfels module panel now uses the same visual convention as the monomial-ideal staircase reference: horizontal coordinate is the `x_radius` exponent/radius grade, vertical coordinate is the `x_level` exponent/reasoning growth level.
- This replaces the visually compressed `x_level`-horizontal, `x_radius`-vertical layout that made the module appear as a tall block rather than a staircase.
- The 3D companion panel still uses `z = beta_i = dim_F2 H_i(K_(level,radius))`; the small H0/H1 layer offset is only visual and is reported as such in hover text.
- The page text now explicitly states horizontal `x_radius`, vertical `x_level`, persisted adjacent structure-map ranks, and that diagnostic chain data is not substituted for a free resolution.

Regression added:
- `test_level_radius_bifiltration_reports_scoped_real_staircase_resolution` now checks real adjacent `F2` structure maps in both `x_level` and `x_radius` directions and validates the generated HTML orientation/language.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` -> `9 passed`.
- Browser DOM check confirmed: `horizontal lattice coordinates are x_radius`, `Columns are radius grades`, two Plotly panels, and `Adjacent structure maps persisted=`.

Next sequential item remains CAS integration: review `references/2210.11433v1.pdf` and implement certified multigraded free-resolution, Fitting-ideal, minor, Buchsbaum-Eisenbud, and derived-map computations without proxy substitutions.


## 2026-06-16 Certified CAS Retrieval Scoring Pass

Status: implemented and focused-tested.

Changes made:
- `AnalogicalMemoryBank.retrieve` now adds a `certified_cas_evidence` score component for query/memory pairs with matching certified CAS real-free-resolution artifacts.
- The score is exact-match gated and contributes zero for unavailable or mismatched CAS evidence.
- Retrieval rows preserve the audit trail: evidence availability, exact-match flag, similarity fraction, mismatch list, score contribution, and no-derived-claim policy.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py -q` -> `9 passed`.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `33 passed`.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` -> `15 passed`.

Next sequential item: continue simplex-tree/NLL/tropical-support repairs.


## 2026-06-16 SimplexTree Face/Coface Contract Pass

Status: implemented and focused-tested.

Changes made:
- The full graph-of-thought SimplexTree page now visibly states that it is a model-embedding barycentric face/coface poset and not disconnected simplex columns.
- Actual face-to-coface cover edges are the primary SimplexTree structure; optional sorted-label trie-prefix links are present but legend-only.
- Tests now enforce the face/coface cover strings, trie-prefix strings, empty-simplex root, and probability-SimplexTree unavailable state when model probability vectors are absent.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `33 passed`.

Next sequential item: continue NLL density and tropical-support repairs.


## 2026-06-16 NLL Density Contract Pass

Status: implemented and focused-tested.

Changes made:
- Added regression coverage for `got_nll_density_cloud_pca_3d.html` and `got_nll_density_cloud_payload.json`.
- The payload is now tested for actual model graph-state PCA anchors, measured raw NLL values, visualization-only Gaussian support samples, exact anchor layers, kernel-weighted local NLL metadata, density-volume provenance, and edge-wise NLL deltas.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `33 passed`.

Next sequential item: continue tropical-support heatmap and wall-crossing repairs.


## 2026-06-16 NLL Density Sidecar Audit Contract Pass

Status: implemented and focused-tested.

Changes made:
- `got_nll_density_cloud_payload.json` now has reviewer-friendly top-level fields for render contract, density contract, exact model-anchor counts, hidden Gaussian support-sample counts, kernel bandwidth, NLL range, local-NLL summaries, density summaries, edge delta summaries, terminal NLL progress, density-volume provenance, actual anchors, and support-sample visibility policy.
- The validator now loads and validates the NLL density payload while retaining compatibility with b59's already-running older writer schema and its explicit legacy unavailable tropical support-probability trace.
- Regression tests cover the enriched payload and reject provenance gaps that would treat density samples as model states or lose actual-anchor evidence.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `35 passed`.
- Generated fixture: `TropicalGT-I/outputs/visualization_nll_density_contract_fixture_20260616T/1781570302` with enriched JSON sidecar.
- Local headless Chrome DOM check confirmed Plotly markers and density/anchor provenance strings. Screenshot rasterization remains blocked by headless Chrome WebGL support for this 3D Plotly page. The upgraded validator passes b59's complete step-250 and step-500 audits under the compatibility path; step-500 BPB is `2.011564489777277` with graph-conditioned BPB without side cost `1.7685317056430416`.

Next sequential item: continue the remaining topological/geometric/algebraic audit repairs while b59 trains toward the 5K evidence gate.

## 2026-06-16 Chart-Bundle/Toric Training Hook Pass

Status: implemented and focused-tested as a zero-default auxiliary hook for future restarts.

Changes made:
- `TropicalGTModel` now has an opt-in chart-bundle/toric auxiliary head that computes chart confidence, monomial projection one-hotness, overlap transport L1, cocycle defect, flat-rank defect, toric normal-fan margin, GraphCG/toric active-cell agreement, chart-BPB availability, and atom-stability gap.
- The disabled default path emits explicit zero metrics. The enabled zero-weight path is tested to preserve logits and loss exactly, so current training behavior is not changed unless a restart config turns coefficients on.
- W&B priority groups now expose the new bundle/toric metrics under `10_bundle_toric`; default train configs declare all related coefficients as `0.0`.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/model.py TropicalGT-I/src/tropicalgt/run.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py -q` -> `19 passed`.

Training gate note:
- b59 continues running toward the required step `5000` minimum review gate. The new hook is for the next evidence-backed restart or ablation, not a mid-run mutation.

Next sequential item: continue source-side audit repairs that do not interfere with b59, then use the step-5000 metrics, sidecars, topology/geometric/algebraic visualizations, and Galileo review to choose restart coefficients and hyperparameters.

## 2026-06-16 Chart-Local BPB Partition Pass

Status: implemented and focused-tested.

Changes made:
- Chart-bundle diagnostics now include a real per-record BPB partition when targets are available: global BPB, chart-local min/max, chart BPB spread, active chart count, availability, and a mass-weighted chart-BPB consistency loss.
- Per-record BPB is detached before reaching the chart head, so positive chart-consistency coefficients train the chart partition side without adding a second gradient path through token logits.
- When targets are absent, the chart-BPB fields remain explicit unavailable zeros.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_losses_and_model.py -q` -> `9 passed`.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py -q` -> `20 passed`.

Next sequential item: implement the remaining transport-side chart-bundle metrics, especially explicit overlap-pair/triple transport ids and memory transported-landscape L2/cosine diagnostics, while b59 continues toward step `5000`.
