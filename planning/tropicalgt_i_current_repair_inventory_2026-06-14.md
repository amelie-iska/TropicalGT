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
- Add vector-bundle, scoped exponent-chart sidecar, certified finite toric-ideal sidecar, scheme, and sheaf-theoretic material to the TropicalGT-I paper as implementations mature.

### 9. Training, Browser QA, Docs, and Push

- Reconcile training state and restart/resume BPB training only under the user-requested 5K gate policy or explicit instruction.
- Generate periodic browser audit artifacts every 250 steps.
- Keep `127.0.0.1:8991` or the active forwarded port pointed at the latest repaired bundle.
- Inspect every repaired visualization in the browser.
- Update `README.md`, planning docs, and tests.
- Push source/planning/docs changes to the non-main branch.

## Present Status

- Current branch: `tropicalgt-i-real-cas-no-proxy-20260614`.
- Active training: fresh b60 step-0 run `tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate`, PID `378962`, watcher PID `379304`, W&B `ld5u55p5`, config under `TropicalGT-I/outputs/launch_configs/`, output dir under `TropicalGT-I/outputs/`.
- b60 is under the 5K evidence gate: do not restart before step `5000` unless it crashes, OOMs, or logs nonfinite/invalid losses. The post-5K analysis/visualization review must use the Codex evidence-review/subagent workflow and the evidence-bound restart schema before any step-0 restart.
- Focused tests currently pass for certified CAS, analogical memory, training metrics, post-5K review bundle, legacy audit backfill, and evidence-bound restart-schema paths after commit `db3a5d2`.
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
- `TropicalGTModel` now has an opt-in chart-bundle/toric auxiliary head that computes chart confidence, monomial projection one-hotness, overlap transport L1, cocycle defect, flat-rank defect, toric activation-cell margin (normal-fan only with certified fan sidecar), GraphCG/toric active-cell agreement, chart-BPB availability, and atom-stability gap.
- The disabled default path emits explicit zero metrics. The enabled zero-weight path is tested to preserve logits and loss exactly, so current training behavior is not changed unless a restart config turns coefficients on.
- W&B priority groups now expose the new bundle/toric metrics under `10_bundle_toric`; default train configs declare all related coefficients as `0.0`.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/model.py TropicalGT-I/src/tropicalgt/run.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py -q` -> `19 passed`.

Training gate note:
- b60 continues running toward the required step `5000` minimum review gate. The new hook is for the next evidence-backed restart or ablation, not a mid-run mutation.

Next sequential item: continue source-side audit repairs that do not interfere with b60, then use the step-5000 metrics, sidecars, topology/geometric/algebraic visualizations, Codex subagent review, and evidence-bound restart schema to choose restart coefficients and hyperparameters.

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

## 2026-06-16 Transport-Gated Landscape Memory Pass

Status: implemented and focused-tested.

Changes made:
- Analogical memory retrieval now reports transported persistence-landscape diagnostics only when the probability simplicial map is available and fully preserved.
- The diagnostic reuses real cached GUDHI landscape vectors and records L2 distance, L2 similarity, cosine, correlation, overlap dimension, chain-map certification, and persistence-module morphism certification.
- Missing probability transport or missing landscape vectors produce explicit unavailable reasons and zero scalar fields.
- Periodic training metrics now aggregate transported-landscape availability, L2, and cosine for W&B/post-5K review.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/memory.py TropicalGT-I/src/tropicalgt/run.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py -q` -> `10 passed`.

Next sequential item: add explicit overlap-pair/triple transport identifiers in the chart-bundle head and visual/audit payloads, then use matched ablations after the b59 5K review before any nonzero transport coefficient is promoted.

## 2026-06-16 Explicit Chart Overlap Transport Id Pass

Status: implemented and focused-tested.

Changes made:
- Enabled chart-bundle model outputs now carry exact chart ids, directed overlap pair ids, directed overlap triple ids, pair-id references for cocycle checks, and transport-logit indices.
- W&B scalar telemetry includes overlap pair/triple counts, while the full metadata remains nested under `chart_bundle_transport_metadata` for audit consumers.
- Disabled chart-bundle mode reports unavailable metadata and zero overlap counts.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/model.py TropicalGT-I/src/tropicalgt/run.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_losses_and_model.py -q` -> `9 passed`.

Next sequential item: run matched BPB/graph-BPB ablations after the b59 5K evidence review before promoting nonzero bundle/toric coefficients.

## 2026-06-16 Macaulay2 Tropical Fan Diagnostic Pass

Status: implemented and focused-tested.

Changes made:
- Added a certificate-gated Macaulay2 `Tropical` wrapper for model/audit ideals over `QQ[x_i]`.
- The wrapper emits real tropical fan/cycle diagnostics from `tropicalVariety`: rays, max cones, lineality space, multiplicities, balance, purity, and simpliciality.
- Missing or failed backend output returns explicit unavailable states and never substitutes these tropical diagnostics for free resolutions.
- The report uses one dimensional cone language only for fan rays and carries the warning that the artifact is not a multigraded free-resolution/derived-equivalence certificate.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/cas_tropical.py TropicalGT-I/src/tropicalgt/__init__.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` -> `18 passed`.

Next sequential item: connect the certified tropical fan diagnostic to a bounded visualization/audit surface when a model-derived tropical ideal is actually exported; unavailable state remains mandatory otherwise.

## 2026-06-16 Tropical Fan Audit Surface Pass

Status: implemented and focused-tested.

Changes made:
- Added always-written tropical fan audit artifacts for inference/audit bundles.
- Certified rendering is gated on an explicit model-derived tropical ideal plus the Macaulay2 `Tropical` certificate; otherwise the page and payload are explicitly unavailable.
- The audit page uses fan-theoretic one dimensional cone language only for certified rays and repeats that the diagnostic is not a free-resolution or derived-equivalence certificate.
- The interactive artifact validator now checks the fan payload schema and no-proxy render contract.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py` passed.
- Tropical fan writer focused tests -> `2 passed`.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `10 passed`.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` -> `27 passed`.

Next sequential item: expose/export a genuinely model-derived tropical ideal spec when the model has enough implemented algebraic state to justify it; until then the audit surface remains unavailable by design.

## 2026-06-16 Tropical Basis And Prevariety Diagnostic Pass

Status: implemented and focused-tested.

Changes made:
- Added optional Macaulay2 `isTropicalBasis` and `tropicalPrevariety` diagnostics to the tropical fan wrapper.
- Side diagnostics are availability-gated and do not weaken the primary safe-render condition, which still requires certified `tropicalVariety` fan/cycle output.
- The audit HTML exposes tropical-basis and prevariety rows so post-run reviewers can compare fan and prevariety evidence without treating either as a free-resolution certificate.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/cas_tropical.py TropicalGT-I/src/tropicalgt/visualization.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` -> `18 passed`.
- Tropical fan focused visualization tests -> `2 passed`; `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` -> `27 passed`.

Next sequential item: expand certified tropical package coverage only where Macaulay2 exposes stable, testable methods, and keep all unsupported checks as explicit unavailable states.

## 2026-06-16 5K Review Contract Inventory Pass

Status: implemented and focused-tested.

Changes made:
- The review-loop target now defaults to BPB `< 1.12`.
- Active training contracts include a bounded artifact inventory with latest periodic audit directories, advanced JSON/HTML sidecars, and validator commands for the post-5K reviewer.
- This supports the requested workflow: train to 5000 steps, analyze metrics/sidecars/visualizations, then restart from step 0 with evidence-backed hyperparameter/config changes.

Verification:
- `python -m py_compile TropicalGT-I/scripts/parameter_golf_codex_review_loop.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_parameter_golf_review_loop.py -q` -> `3 passed`.

Next sequential item: after b59 reaches 5000, use this inventory to guide the subagent review and step-0 restart; until then continue non-interfering source/audit hardening.

## 2026-06-16 5K Step Gate Monitor Pass

Status: implemented and focused-tested.

Changes made:
- Added a reusable training step-gate monitor that watches the b59 log and writes a bounded JSON status/stop record.
- The monitor terminates the configured PID only after the parsed training step reaches `5000`; when required artifact paths are configured, it waits for those step-5000 validation/audit files before terminating, up to a bounded grace window. It otherwise exits early only for fatal markers or a process that dies before the gate.
- This keeps the user-requested 5K evidence policy operational without changing the already-running training config or fabricating a review boundary.

Verification:
- `python -m py_compile TropicalGT-I/scripts/monitor_training_step_gate.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_step_gate_monitor.py -q` -> `8 passed`.

Next sequential item: keep the active b60 watcher running with required step-5000 artifact paths, then keep hardening non-interfering audit paths while b60 trains.

## 2026-06-16 Post-5K Review Bundle Helper Pass

Status: implemented and focused-tested.

Changes made:
- Added a post-5K review-bundle helper that prepares the Codex evidence-review handoff from an already-trained run without launching training.
- The bundle includes the active training contract, Codex review prompt, stop-record payload, bounded artifact inventory, eval/visualization command, and interactive-audit validator commands.
- The bundle is path-only and keeps generated artifacts/checkpoints out of git.

Verification:
- `python -m py_compile TropicalGT-I/scripts/prepare_5k_review_bundle.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py -q` -> `4 passed`.

Next sequential item: when the b60 watcher writes a target-reached stop record, run the helper against the b60 config/report/checkpoint/stop-record and hand the resulting bundle to the Codex evidence-review/subagent workflow before any evidence-backed restart.

## 2026-06-16 Bivariate Staircase Visual Contract Pass

Status: implemented and focused-tested.

Changes made:
- The two-parameter trajectory persistence page now emits a dedicated visual sidecar proving the primary view is the Miller-Sturmfels bivariate staircase over `F2[x_level,x_radius]`.
- Removed unreachable legacy code for the old rank-surface-first Plotly implementation and removed the retired `trajectory_level_radius_bifiltration_3d` path key from source/tests.
- The validator checks the visual sidecar when present and remains compatible with b59 artifacts emitted before this sidecar existed.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `55 passed`.
- `git diff --check` passed.

Legacy b59 note:
- The already-running b59 process predates tropical fan and bifiltration visual sidecar writers. Its old step-500 validation probe still fails on missing `tropical_fan_diagnostics.json/html`; the post-5K review should either rerender missing audits with current code or record explicit unavailable states rather than treating absent old sidecars as completed evidence.

Next sequential item: continue the visual/math repair queue by adding a small post-hoc legacy audit repair/rerender path or move into the remaining CAS/free-resolution backend coverage, depending on the b59 5K timing.

## 2026-06-16 Legacy Audit Backfill Pass

Status: implemented, focused-tested, and live-smoked on b59 step 500.

Changes made:
- Added a dedicated legacy audit backfill helper for `got_audit` directories produced before the current tropical fan and bivariate staircase visual-contract writers existed.
- Missing tropical fan diagnostics are repaired only as explicit unavailable/no-proxy artifacts unless an actual model-derived ideal and CAS certificate are present.
- Missing two-parameter bifiltration visual sidecars are regenerated only from the raw `trajectory_level_radius_bifiltration.json` payload, preserving the Miller-Sturmfels bivariate staircase contract.
- The helper writes `backfill_report.json` so post-5K reviewers can distinguish generated unavailable states from certified mathematical evidence.

Verification:
- `python -m py_compile TropicalGT-I/scripts/backfill_interactive_audit_artifacts.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_backfill_interactive_audit_artifacts.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` -> `11 passed`.
- Live b59 step-500 audit backfill plus strict validator -> PASS, rows checked `1`, `bpb=2.011564489777277`, `graph_bpb=19.847021684446936`, and `graph_conditioned_bpb_no_side_cost=1.7685317056430416`.

Next sequential item: post-5K review should run this helper on legacy b59 audit folders before strict validation, then proceed with metrics, advanced sidecars, topological/geometric/algebraic visual review, and evidence-backed restart planning.

## 2026-06-16 Periodic GoT Trace Cap Pass

Status: implemented and focused-tested after b59 stopped on disk exhaustion at step 2500.

Changes made:
- Root cause: b59 wrote very large periodic `got_audit` payloads every 250 steps and exhausted disk while writing the step-2500 audit bundle.
- The validation curve through step 2500 remains intact and improving: step 2500 `bpb=1.542353032164675`, `graph_bpb=19.43449932006042`, `nll=1.069077655673027`, `invalid_graph_rate=0.0`.
- Periodic GoT scaling budgets now include requested/effective trace limits, and the training loop uses the effective bounded trace limit for in-training audits.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/run.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_metrics.py -q` -> `11 passed`.
- `git diff --check` passed.

Operational follow-up: prune bulky generated intermediate `got_audit` directories, preserve compact validation reports/checkpoints, resume b59 from the step-2500 checkpoint, and keep the 5K artifact gate active.

## 2026-06-16 Periodic Audit Retention Policy Pass

Status: implemented and focused-tested.

Changes made:
- Added a config-gated retention policy for generated periodic `got_audit` bundles so future runs can avoid another disk-exhaustion stop.
- Retention keeps the latest N audit bundles plus configured protected milestone steps, and removes only generated `got_audit` directories.
- Retained validation reports and non-audit periodic artifacts preserve the compact metric curve even when bulky generated trace payloads are pruned.

Verification:
- `python -m py_compile TropicalGT-I/src/tropicalgt/run.py` passed.
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_metrics.py -q` -> `12 passed`.
- `git diff --check` passed.

Operational note: the live b59 resume is already running from the step-2250 checkpoint under PID `334581` with watcher PID `334852`; this source change is for subsequent launches/restarts.

## 2026-06-18 GFlowNet Action Selection Contract Pass

Status: implemented and focused-tested.

Changes made:
- Deterministic and stochastic GFlowNet/GoT branch selections now carry an explicit action-selection contract sourced from real `gflownet_action_probs`.
- The contract makes the audit-score and sampling policy visible while stating that selected branches are not policy-quality certificates and are not proxy/fallback evidence.
- Provenance tracking now keys this surface as `gflownet_action_selection_contract`.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_scaling.py TropicalGT-I/tests/test_metric_provenance.py -q` -> `12 passed`.
- `PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered` -> `333 covered, 0 uncovered`.

Next sequential item: continue source-side no-proxy contract expansion and browser/report QA without launching or interrupting GPU training.

## 2026-06-18 GFlowNet Branch-Selection Report Surfacing Pass

Status: implemented and focused-tested.

Changes made:
- Inference-scaling reports now include compact branch-selection audits on expansion levels.
- Selected branch actions expose the same no-proxy action-selection contract in user/reviewer-visible report payloads.
- The branch audit keeps probabilities, audit scores, and stochastic sampling metadata separate from model-quality claims.

Verification:
- `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_scaling.py TropicalGT-I/tests/test_metric_provenance.py -q` -> `12 passed`.
- `PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered` -> `334 covered, 0 uncovered`.

Next sequential item: continue source-side no-proxy/report surfacing and browser QA without starting or interrupting GPU work.

## 2026-06-18 Herschel GFlowNet Branch-Selection Evidence Pass

Status: implemented and focused-tested.

Changes made:
- Herschel review bundles now inventory `inference_scaling_tree.json` from the latest GoT audit directory when present.
- Herschel reports real GFlowNet branch-selection contracts, selected-action counts, and policy counts from recorded sidecars only.
- Missing or unsafe branch-selection sidecars remain explicit unavailable evidence.

Verification:
- `PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py -q` -> `11 passed`.
- Broader Herschel/bundle/review-loop/provenance slice -> `36 passed`.
- `audit_metric_provenance.py --fail-on-uncovered` -> `339 covered, 0 uncovered`.

Next sequential item: continue post-5K evidence/report surfacing and validator/backfill coverage without launching or interrupting GPU work.

### Current Objective Update - Herschel Tropical Support Evidence Pass

- [x] Added `tropical_support_payload.json` to the Herschel-required audit sidecar inventory when it exists in the latest recorded `got_audit` directory.
- [x] Added `tropicalgt.herschel_tropical_support_evidence.v1` to the Herschel 5K summary, sourced only from recorded tropical support payload sidecars.
- [x] Availability now requires the real tropical support render/readability contracts, `model_tropical_support_probabilities` provenance, observed support assignments, strict/near wall-margin rates, and no-proxy/no-fallback flags.
- [x] Markdown and HTML reports now expose a `Tropical Support Evidence` section and a probability-source chart while preserving the distinction between model margin-threshold audits and certified normal-fan wall-crossing counts.
- [x] Missing, unparsable, stale, or unsafe tropical support sidecars remain unavailable and cannot justify a restart, artifact promotion, or hyperparameter change.
- [x] Verification passed: py-compile; focused Herschel/bundle tests (`11 passed`); broader Herschel/bundle/review-loop/provenance tests (`36 passed`); metric provenance audit (`348 covered, 0 uncovered`); metric provenance tests included in the broader slice.

### Current Objective Update - Herschel GraphCG Direction Evidence Pass

- [x] Added `graphcg_direction_cosines_payload.json` to the Herschel-required audit sidecar inventory when it exists in the latest recorded `got_audit` directory.
- [x] Added `tropicalgt.herschel_graphcg_direction_evidence.v1` to the Herschel 5K summary, sourced only from recorded GraphCG direction-cosine payload sidecars.
- [x] Availability now requires `tropicalgt.graphcg_direction_evidence.v1`, `tropicalgt.graphcg_direction_readability.v1`, all model-derived direction rows, exact direction-id preservation, all-direction panels without heatmap sampling, a projection-basis certificate, and no-proxy/no-fallback flags.
- [x] Markdown and HTML reports now expose `GraphCG Direction Evidence` plus a projection-basis source chart for post-5K review.
- [x] Missing, unparsable, unsafe, sampled, or certificate-lacking GraphCG payloads remain unavailable and cannot justify restart, promotion, semantic-identifiability, or toric-fan claims.
- [x] Verification passed: py-compile; focused Herschel/bundle tests (`11 passed`); broader Herschel/bundle/review-loop/provenance tests (`36 passed`); metric provenance audit (`354 covered, 0 uncovered`).

### Current Objective Update - Herschel NLL Density Evidence Pass

- [x] Added `got_nll_density_cloud_payload.json` to the Herschel-required audit sidecar inventory when it exists in the latest recorded `got_audit` directory.
- [x] Added `tropicalgt.herschel_nll_density_evidence.v1` to the Herschel 5K summary, sourced only from recorded NLL density cloud payload sidecars.
- [x] Availability now requires `tropicalgt.nll_density_render.v1`, visible actual model anchors, legend-only non-model Gaussian support samples, hidden support samples, positive kernel bandwidth, PC3 z-axis policy, NLL range, density-volume non-model-state provenance, and no-proxy/no-fallback flags.
- [x] Markdown and HTML reports now expose `NLL Density Evidence` plus a visible-density-layer chart for post-5K review.
- [x] Missing, unparsable, unsafe, anchor-missing, or support-sample-confused NLL density payloads remain unavailable and cannot justify restart, BPB claims, or hyperparameter promotion.
- [x] Verification passed: py-compile; focused Herschel/bundle tests (`11 passed`); broader Herschel/bundle/review-loop/provenance tests (`36 passed`); metric provenance audit (`360 covered, 0 uncovered`).

### Current Objective Update - Herschel Chart/Vector-Bundle Evidence Pass

- [x] Added `chart_bundle_transport_sidecar.json` to the Herschel-required audit sidecar inventory when it exists in the latest recorded `got_audit` directory.
- [x] Added `tropicalgt.herschel_chart_bundle_transport_evidence.v1` to the Herschel 5K summary, sourced only from recorded chart/vector-bundle transport sidecars.
- [x] Availability now requires the chart-bundle transport schema, exported chart metadata schema, monomial transport no-proxy contract, bundle matroid flat-incidence no-proxy contract, vector-bundle paper sidecar, completeness contract, chart ids, and explicit safety flags that prevent theorem/toric/tropical/normal-fan promotion.
- [x] Markdown and HTML reports now expose `Chart/Vector-Bundle Evidence` plus a completeness-tier chart for post-5K review.
- [x] Missing, unparsable, unsafe, schema-mismatched, metadata-lacking, or contract-lacking chart/vector-bundle sidecars remain unavailable and cannot justify restart, BPB claims, paper-ready claims, toric embedding claims, or hyperparameter promotion.
- [x] Verification passed: py-compile; broader Herschel/bundle/review-loop/provenance tests (`36 passed`); metric provenance audit (`372 covered, 0 uncovered`).

### Current Objective Update - Herschel Toric/Tropical CAS Evidence Pass

- [x] Added `toric_embedding_sidecar.json` and `tropical_fan_diagnostics.json` to the Herschel-required audit sidecar inventory when they exist in the latest recorded `got_audit` directory.
- [x] Added `tropicalgt.herschel_toric_tropical_cas_evidence.v1` to the Herschel 5K summary, sourced only from recorded toric embedding and tropical fan CAS sidecars.
- [x] Finite toric-ideal sidecar evidence now requires the toric visual-audit schema, `tropicalgt.cas_toric_embedding.v1`, explicit toric exponent-matrix input contract, certificate attachment, certified toric ideal, no-proxy flags, and false global toric-variety/tropical-variety/normal-fan safety flags.
- [x] Tropical fan evidence now requires the tropical fan visual-audit schema, `tropicalgt.cas_tropical_fan.v1`, explicit model-derived tropical ideal input contract, certificate attachment, certified fan diagnostics, one dimensional cone/ray count, and no-proxy flags.
- [x] Markdown and HTML reports now expose `Toric/Tropical CAS Evidence` plus a CAS status chart for post-5K review.
- [x] Missing, unavailable, schema-mismatched, input-lacking, certificate-lacking, no-proxy-lacking, or globally unsafe toric/tropical sidecars remain unavailable and cannot justify restart, BPB claims, theorem claims, toric embedding claims, tropical-variety claims, or normal-fan claims.
- [x] Verification passed: py-compile; focused Herschel/bundle tests (`11 passed`); broader Herschel/bundle/review-loop/provenance tests (`36 passed`); metric provenance audit (`382 covered, 0 uncovered`).

### Current Objective Update - Herschel Persistence-Landscape PH Evidence Pass

- [x] Added `trajectory_persistence/persistence_landscapes.json` to the Herschel-required audit sidecar inventory when it exists in the latest recorded `got_audit` directory.
- [x] Added `tropicalgt.herschel_persistence_landscape_evidence.v1` to the Herschel 5K summary, sourced only from recorded persistence-landscape visual-contract sidecars.
- [x] Available landscape evidence now requires `tropicalgt.persistence_landscape_visual_contract.v1`, actual GUDHI `lambda_k(t)` landscape rows, backend provenance, curve traces, finite interval evidence, no-proxy flags, `not_nll_fitness_landscape`, and `not_norm_only_summary`.
- [x] Verified-unavailable no-finite-interval states are reported explicitly and cannot become zero landscape evidence, NLL/fitness landscapes, norm-only summaries, BPB evidence, or restart justification.
- [x] Markdown and HTML reports now expose `Persistence Landscape Evidence` plus a backend/unavailable-reason chart for post-5K review.
- [x] Verification passed: py-compile; focused Herschel/bundle tests (`11 passed`); broader Herschel/bundle/review-loop/provenance tests (`36 passed`); metric provenance audit (`388 covered, 0 uncovered`).

### Current Objective Update - Herschel Bivariate Module/Free-Resolution Guard Evidence Pass

- [x] Added `trajectory_level_radius_bifiltration.json` to the Herschel-required audit sidecar inventory when it exists in the latest recorded `got_audit` directory.
- [x] Added `tropicalgt.herschel_bivariate_module_evidence.v1` to the Herschel 5K summary, sourced only from the recorded raw `F2[x_level,x_radius]` bivariate module sidecar.
- [x] Herschel now reports module-grid availability, fiber-rank profile count, structure-map count, rank-invariant sample count, chain-generator count, selected object key, and coefficient ring.
- [x] Herschel separately reports the nested `tropicalgt.real_free_resolution.v1` guard: certified real free-resolution count, safe-unavailable count, status counts, input hash, and render-safety flags.
- [x] Finite multigraded chain-presentation diagnostics remain explicitly `not_a_free_resolution`; missing, failed, or safe-unavailable CAS guards cannot justify free-resolution, derived-equivalence, BPB, or restart claims.
- [x] Markdown and HTML reports now expose `Bivariate Module Evidence` plus a real-resolution status chart for post-5K review.
- [x] Verification passed: py-compile; focused Herschel/bundle tests (`11 passed`); broader Herschel/bundle/review-loop/provenance tests (`36 passed`); metric provenance audit (`393 covered, 0 uncovered`).

## 2026-06-18 Herschel Analogical Memory Evidence Pass

Status: implemented and pushed-ready for the current sequential item.

Changes made:
- The 5K review bundle now discovers `analogical_memory_retrieval.json` and `analogical_simplex_tree_analogy.json` from the latest recorded GoT audit directory when present.
- Herschel's report summary now includes `tropicalgt.herschel_analogical_memory_evidence.v1` with real retrieval, top-k, simplex-tree analogy, quality-gate, and insufficient-memory fields.
- Report availability remains no-proxy: model-probability Jensen-Shannon assignments are required, embedding-only assignments are rejected, and chain-map/module-morphism claims require certified filtered simplicial maps.
- The provenance registry now classifies the new analogical memory report contract and no-proxy guard strings.

Verification:
- `CUDA_VISIBLE_DEVICES="" ... py_compile ...` passed for touched source and tests.
- Focused Herschel/bundle tests passed: `11 passed`.
- Broader Herschel/bundle/review-loop/provenance tests passed: `36 passed`.
- Metric provenance audit passed: `findings=412 covered=412 uncovered=0`.

Next sequential item: continue the Herschel evidence family queue by selecting the next missing implemented sidecar or move to the next source-side repair item if all current Herschel 5K evidence slices are surfaced.
