# TropicalGT-I Current Required Updates and Repair List

This checklist merges the browser/photo review, the current active training state, the CAS/free-resolution policy, and the new tropical vector-bundle / tropical toric-embedding paper workstream. All work is remote-only on `iska@iska` in `/home/iska/Documents/amelie/bio/TropicalGT`, branch `tropicalgt-i-real-cas-no-proxy-20260614` or a newer non-main successor branch.

## 1. Active Training and BPB Priority

- Keep `tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate` alive until at least step 5K unless it crashes, OOMs, or produces nonfinite/invalid losses. Current detached launch PID at last check: `378962`; watcher PID: `379304`; W&B run id: `ld5u55p5`; run URL: `https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/ld5u55p5`; config: `TropicalGT-I/outputs/launch_configs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.json`; output dir: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate`; tested 5K monitor record target: `TropicalGT-I/outputs/training_stop_records/b60_fresh_step0_casrows_step5000_gate.json`.
- Do not restart before 5K merely because early metrics are noisy. The active `monitor_training_step_gate.py` watcher should stop b60 at or after the parsed step `5000` after required step-5000 validation/audit artifacts are present after a one-poll settle window or a bounded grace window expires, write the stop record, and then the 5K review worker should run the analysis/eval/visualization sidecars, inspect BPB, graph-BPB, NLL, advanced topology/geometric/algebraic diagnostics, and restart from step 0 only with evidence-backed hyperparameter/config adjustments.
- Preserve BPB and graph-BPB as primary optimization and promotion gates.
- Use advanced auxiliaries only when they are zero-default or BPB-gated and ablated: tropical support, GFlowNet GoT rewards, GraphCG, persistence/landscape diagnostics, chart-bundle transports, tropical toric active-cell agreement, and memory retrieval.
- Track step, VRAM, wall time, train/eval NLL, BPB, graph-BPB, certificate loss, tropical wall-hit rate, support entropy, GraphCG rank, meet-in-the-middle agreement, ROAR/causal decoding path mix, and artifact-generation status.
- While b60 trains to the 5K gate, the main implementation focus is the remaining visual/math repair queue. Historical b59 notes below are retained only for provenance and are not active run instructions.

## 2. Browser QA and Visual Evidence

- Keep the served audit bundle at `127.0.0.1:8990` pointed to the latest real model-output artifacts.
- Every visual change should be regenerated from `inference_audit.json` or live periodic audit payloads, never from synthetic examples.
- Review pages visually after edits when browser access is available; keep screenshots/evidence for before/after when possible.
- If browser navigation is blocked by the app policy, continue regenerating the served artifact files and record the blocked browser step in status rather than using a workaround.

## 3. Real Data Only Policy

- No fake fallback data in plots, metrics, losses, or algebraic objects.
- If a real model output, embedding, probability vector, GUDHI object, CAS certificate, or memory row is unavailable, render an explicit unavailable diagnostic and do not substitute zeros or synthetic samples.
- Gaussian density samples may be used only as visualization support around actual model embedding vertices; they must never be rendered or labeled as reasoning states.

## 4. Bifiltration and 2-Parameter Persistence

- Every nonempty reasoning trajectory must emit an `F2[x_level,x_radius]` bifiltration.
- Compute actual grid fibers `K_(level,radius)` and east/north structure-map metadata.
- Render a readable 3D `(level, radius, rank/Betti)` lattice/grid view with downloadable JSON payloads.
- Add rank invariant samples, Betti surfaces, generator candidates, adjacent-lcm syzygy candidates, and staircase-style views over the `(level,radius)` lattice.
- Add tests proving level grades, radius grades, fibers, structure maps, and module-ring metadata are present for all nonempty trajectories.

## 5. Radius-Filtered Simplicial Complexes

- Radius filtrations must start as disjoint vertices and grow min-to-max by adding edges and faces.
- Solid simplex edges/faces must be gated by the radius slider.
- Dotted directed overlays must show causal structure, decoding order, or both when they are present, and should be separately gated by radius, reasoning step, or decoding step.
- Remove duplicate/extraneous lower panels unless they show a different object with a clear label.
- Every reasoning step must have its own model-derived filtered complex and must not share identical payloads unless the page explicitly reports identical model data.

## 6. Simplex Tree / Face-Coface Poset

- Render an actual simplex tree, trie, or Hasse/face-coface poset from the GUDHI `SimplexTree` or equivalent exact simplex list.
- The visualization must be connected by real inclusion edges or an explicitly labeled artificial root used only for layout.
- Hover must show simplex vertices, dimension, filtration value, parent/coface relations, associated reasoning step, and model input/output payload.
- Current disconnected-column simplex-tree pages are defects to repair.

## 7. Free Resolutions and CAS Methodology

- Only real free resolutions are allowed.
- Preferred backends: Macaulay2 (`res`, `syz`, `betti`, `mingens`, `fittingIdeal`, `minors`, `KustinMiller::resBE`, `MultiplierIdeals`), SageMath/Singular bridges, and optional `amelie-iska/BEMultipliers.git` for Buchsbaum-Eisenbud multiplier diagnostics.
- If no backend returns certified differentials, multidegree shifts, Betti table, exactness/minimality diagnostics, and module presentation, render `unavailable_no_certificate`.
- Chain-presentation ranks, adjacent-lcm staircases, or rank samples may be displayed only as diagnostics, never called free resolutions.
- Research-style figures should show Betti tables, differential matrices, multidegree shifts, Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, and module/category notes.

## 8. Derived / Analogical Maps

- Retrieval must be derived from model-predicted probability vectors, not embeddings alone.
- Use Jensen-Shannon assignment between probability-vector complexes.
- Validate vertex, edge, face, and filtration preservation before calling a correspondence a simplicial map.
- Add simplex-tree-level maps between query trajectory and memory trajectory.
- Derived similarity must be conservative and consistent with PH, rank invariant, chain-map, and real-resolution evidence. High derived similarity with zero resolution similarity is invalid unless explicitly labeled as a different coarse signature metric.
- Use persistence landscapes as vectorized real GUDHI features for analogical comparison only when both query and memory have available landscape vectors.

## 9. Memory Retrieval

- Store trajectory memories only above a quality threshold.
- Early training may show insufficient-memory/unavailable states; do not fabricate maps.
- When memory is active, retrieve many top-k analogies and show quality, retrieval score, JS assignment cost, landscape availability, transported landscape L2/cosine, PH/rank/resolution evidence, and map-preservation diagnostics.

## 10. NLL / Energy / Density Visualizations

- The main GoT trajectory plot should use actual 3D PCA coordinates of model graph-state or token embeddings.
- Raw NLL should be color, hover metadata, and/or a local density/field value, not a fake z-coordinate unless explicitly labeled.
- Add an electron-density-like 3D NLL cloud around actual PCA embedding vertices using Gaussian support samples hidden as support, with visible translucent local neighborhoods and actual model states as anchors.
- Optional 2D/3D NLL surfaces must cover the trajectory neighborhood and explicitly anchor or interpolate the actual vertices.
- Show edgewise NLL deltas and terminal improvement so reasoning progress is inspectable.

## 11. Tropical Support

- Make support heatmaps interpretable with grouped labels, compact token labels, top-support summary, margin profile, support frequency, entropy, collapse diagnostics, and readable tables.
- Audit why wall-hit rate is low: distinguish strict chamber-wall crossing from near-wall margin events.
- Rename or fix negative tropical-margin metrics so their sign and interpretation are clear.

## 12. GraphCG

- GraphCG directions must remain full rank relative to embedding dimension.
- Improve direction heatmaps, spectra, candidate activity, signed bias plots, and labels so they are not jammed.
- Add GraphCG-toric agreement metrics once toric active-row hooks exist.

## 13. Decoding and Dataset Graph Structure

- Add/maintain config toggle for meet-in-the-middle two-sided decoding compatible with ROAR.
- Causal DAGs use forward plus reverse causal directions.
- Cyclic or noncausal graphs use ROAR/random-order autoregressive decoding.
- Dataset graphs that should have causal structure must be annotated; cyclic/noncausal cases must remain supported.

## 14. Tropical Vector Bundles and Tropical Toric Embeddings

- Add zero-default config flags for monomial chart transports, tropical toric embeddings, chart BPB consistency, landscape transport metrics, and atom stability.
- Model hooks must emit chart ids, overlap pairs/triples, toric active rows, active tropical support cells, GraphCG active directions, chart-local NLL, and transport ids without changing logits when coefficients are zero.
- Implement `MonomialTransportHead`, `BundleMatroidHead`, and `ToricEmbeddingHead` only behind config gates.
- Add losses/metrics: `bundle/transport_l1`, `bundle/cocycle_defect`, `bundle/flat_rank_defect`, `toric/normal_fan_loss`, `graphcg/toric_cell_agreement`, `chart/bpb_consistency`, `bundle/atom_stability_gap`, `memory/transported_landscape_l2`, and `memory/transported_landscape_cosine`.
- Treat all of these as auxiliary until matched ablations show BPB/graph-BPB benefit.


Implementation checkpoint, 2026-06-16:
- Zero-default config flags and model hooks are now implemented for chart-bundle/toric telemetry and losses: bundle transport L1, cocycle defect, flat-rank defect, toric normal-fan loss, GraphCG-toric cell agreement, chart-BPB consistency availability, and atom-stability gap.
- The implementation is deliberately auxiliary and gated. Disabled defaults emit zero metrics; enabled zero-weight hooks preserve logits and loss; positive coefficients affect only the auxiliary regularizer.
- Chart-local NLL/BPB partitions, explicit chart overlap pair/triple ids, and transport-gated memory persistence-landscape L2/cosine diagnostics are now implemented where their evidence exists. Remaining item in this section: matched BPB/graph-BPB ablations before promoting any nonzero bundle/toric coefficient.


### Paper Workstream Status: Vector Bundles and Tropical Toric Embeddings

- Subagent Avicenna reviewed `references/2405.03505v1.pdf` and `references/2009.03030v2.pdf` in full from extracted text and updated `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex`.
- The new paper material introduces TropicalGT chart bundles, TokenGT tropical atlases, monomial tropical chart transports, one dimension cone-filtration atomization, tropical toric embeddings, matroid flat-defect objectives, GraphCG-toric agreement, chart-BPB consistency, and transported persistence-landscape memory metrics.
- Implementation planning is in `planning/tropicalgt_i_vector_bundle_toric_embedding_training_plan.md`; the key rule is zero-default auxiliary coefficients with telemetry-only and active-loss ablations before any BPB promotion.
- Remote LaTeX compilation remains blocked by missing `latexmk`, `pdflatex`, and `tectonic`; the current lightweight TeX environment-balance check passed.

## 15. Tests, Docs, and Paper

- Add tests for zero-coefficient no-op logits, unavailable landscape masking, monomial projection one-hotness, cocycle identity, decreasing one dimension cone filtrations, finite atom-stability diagnostics, real-only free-resolution reporting, bifiltration availability, and browser label separation.
- Update README with current training, eval, visualization, inference, W&B, and artifact commands.
- Remove AI-generated filler language from `references/main.pdf` source and the TropicalGT-I paper source.
- Recompile papers when a LaTeX engine is available.
- Push only to non-main branch `tropicalgt-i-real-cas-no-proxy-20260614` or its newer non-main successor, excluding secrets, datasets, W&B runs, caches, checkpoints, and bulky artifacts.

## 2026-06-14 v9 5K-Gated Restart Addendum

- Deleted old post-v4 output runs and W&B local run directories, preserving `tropicalgt_i_pg_bpb_step0_full24b_b52_v4_bpb_restart` as the requested baseline lineage.
- Removed old post-v4 restart configs and created `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b54_v9_bpb_5k_gate.json` as the fresh run config.
- v9 uses batch 54, learning rate `1.9e-4`, weight decay `0.022`, low-weight tropical/GFlowNet/GraphCG/certificate auxiliaries, memory quality threshold `0.05`, top-k memory retrieval `12`, and periodic real topology audits every 250 steps.
- v9 restart policy: keep the run alive until step 5K unless the process crashes or emits nonfinite/invalid loss. The review gate at 5K should inspect BPB, graph-BPB, NLL, certificate loss, tropical wall/near-wall rates, GraphCG rank, GFlowNet reward, memory quality, meet-in-the-middle agreement, and artifact generation before any restart.
- Current next-focus order: (1) CAS integration and real free-resolution reporting, (2) derived/analogical map consistency using probability-vector complexes and real resolution evidence, (3) browser rendering of CAS-certified Betti tables, differential matrices, Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, and derived-category notes.

## 2026-06-14 v10 Training and CAS-Guard Addendum

- v9 (`tropicalgt_i_pg_bpb_step0_full24b_b54_v9_bpb_5k_gate`, W&B `h4emzqgh`) reached the first periodic audit boundary and then failed because the full reasoning audit required complete per-step `graph_token_trace_complete` payloads for all sampled reasoning states. The run artifacts and v9 config were removed as part of the requested post-v4 cleanup.
- Created and launched `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate.json` with run name `tropicalgt_i_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate`, PID `2154851`, and W&B id `nw0bo45u`.
- v10 keeps the 5K minimum review gate, keeps BPB/graph-BPB as the primary objective, and changes periodic visualization failure policy from hard-crash-on-incomplete-trace to `record_incomplete_without_fabrication`. Missing model trace fields must render as unavailable/incomplete diagnostics, never fabricated graph states.
- v10 config sets `periodic_viz_require_complete_reasoning_steps=false`, `viz_require_complete_reasoning_steps=false`, `interactive_artifacts_require_complete_reasoning_steps=false`, and `require_complete_reasoning_steps=false` only to prevent audit infrastructure from killing training. It does not authorize synthetic plot inputs.
- CAS integration checkpoint: added `TropicalGT-I/src/tropicalgt/cas_free_resolution.py` as a strict real-only adapter boundary for modules over `F2[x_level,x_radius]`, `F2[x_filtration,x_dimension]`, and `F2[x_filtration,x_dimension,x_position]`. The adapter currently returns unavailable/probe states unless Macaulay2, Singular, Sage, or a certified backend is present.
- Visualization guard checkpoint: `TropicalGT-I/src/tropicalgt/visualization.py` no longer silently falls back from missing certified resolution rows to chain-presentation rows in Macaulay2-style displays. Chain presentations can be rendered as diagnostics, but not as real free resolutions.

## 2026-06-14 Sequential CAS Update: Singular Backend Installed And Bridged

Status: complete for the first real CAS bridge checkpoint.

- Installed Singular 4.4.1 in the isolated conda environment `/home/iska/miniconda3/envs/tropicalgt-cas` so the active `tokengt` training environment remains untouched.
- Added deterministic executable discovery for `TROPICALGT_SINGULAR_BIN`, `PATH`, and `/home/iska/miniconda3/envs/tropicalgt-cas/bin/Singular`.
- Replaced the invalid Singular module assignment syntax with a true Singular module literal such as `[x_level,0],[0,x_radius]` built from the finite presentation matrix over `F2[x_level,x_radius]` or the supported multigraded rings.
- The certified Singular path computes `mres(M,0)` for the image submodule `M` of the presentation matrix and records the cokernel-resolution interpretation explicitly: TropicalGT prepends the displayed free module `F0 -> coker(M)` to the Singular image-submodule resolution.
- Exactness is marked certified only when Singular runs successfully and emits tagged output. Minimality is marked separately and is false if the presentation contains unit entries; this prevents a nonminimal presentation from being mislabeled as a minimal free resolution.
- Certified outputs now include `backend_probe`, `bemultipliers_probe`, command templates, tagged raw output, Singular Betti text, and Singular resolution text. Uncertified paths still render with empty `cas_artifacts`.
- Added a focused smoke test for a real `F2[x_level,x_radius]` CAS free-resolution computation. The algebra suite now accepts both unavailable environments and certified Singular/Macaulay2/Sage environments.

Verification:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 7 passed in 1.48s
```

Remaining CAS items after the 2026-06-16 structured-row checkpoint:

1. Expand Macaulay2 as the preferred backend for minimal graded free resolutions, richer Fitting/minor extraction, and Buchsbaum-Eisenbud diagnostics when its packages are available. The bridge already parses real Macaulay2 multigraded free-module degree blocks into `betti_table_rows`; keep using this as the only source for multigraded shift claims.
2. Keep Singular as a certified fallback for exactness, determinantal ideals, Fitting ideals, minors, and figure-ready ungraded Betti rows. Singular `betti(R)` rows are now structured but explicitly `not_multigraded`; do not promote them to `F2[x_level,x_radius]` multidegree shifts.
3. Add optional Sage hardening only if it records the underlying backend and returns the same certificate fields. Current total-graded Sage rows must remain labeled total-graded/non-multigraded.
4. Continue wiring `amelie-iska/BEMultipliers.git` only as a Buchsbaum-Eisenbud diagnostic layer after a certified Macaulay2 resolution exists; it is not a substitute for a resolution backend.
5. Browser-verify the rendered CAS tables after the next 5K audit bundle exists, especially Betti rows, differential previews, Fitting ideals, minors, and BE diagnostic cells.

## 2026-06-16 Sequential CAS Update: Structured Betti Rows For Certified CAS Output

Status: complete for the row-schema checkpoint; incomplete for the broader Macaulay2/BEMultipliers expansion.

- Added `betti_table_rows` to the certified CAS parser for Macaulay2 multigraded degree blocks, Sage total-graded summaries, Macaulay2 total rows, and Singular Betti matrices.
- Singular rows now carry `source=Singular_betti_matrix`, `shift_display=ungraded row <i>`, `multidegree=[]`, `not_multigraded=true`, and `safe_for_multigraded_claims=false`, so research-style figures can display the real certified rows without implying unavailable multidegree shifts.
- The two-parameter persistence visualization adapter now prefers explicit `betti_table_rows` from certified CAS artifacts and falls back to module ranks only for legacy artifacts.
- Added a deterministic tagged-output test for the Singular certified path, plus Macaulay2 structured-row assertions and live/unavailable CAS smoke coverage.

Verification:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_algebraic_persistence.py
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 19 passed in 8.02s
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 37 passed in 3.61s
git diff --check
# clean
```


## 2026-06-16 Sequential CAS Update: Structured Fitting/Minor And BE Rank Diagnostics

Status: complete for the structured diagnostic checkpoint; still incomplete for full grade/depth/regular-element certification beyond what the CAS backend explicitly emits.

- Added structured `ideal_diagnostics` for real CAS Fitting ideals and determinantal minor ideals. Rows record fitting index, inferred determinantal order for the displayed presentation matrix, ideal text, source backend, and the method note `Fitt_j(coker(PM)) = I_{rows-j}(PM)`.
- Added `buchsbaum_eisenbud_rank_conditions` computed from certified free-module ranks and differential shapes. These rows expose BE-style nonnegative rank-condition sanity checks while explicitly marking `is_independent_certificate=false`; exactness/minimality still come only from the CAS certificate.
- The visualization CAS certificate table now includes ideal diagnostics and BE rank-condition summaries alongside Fitting ideals, minors, BEMultipliers status, and aMultiplier output.
- Method scan from `references/2210.11433v1.pdf` reinforced the guardrail: rank strata are determinantal, Fitting invariants are minors of presentation maps, and Buchsbaum-Eisenbud multiplier/exactness claims require explicit CAS-backed evidence, not inferred chain diagnostics.

Verification:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_algebraic_persistence.py
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 19 passed in 6.01s
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 37 passed in 3.54s
git diff --check
# clean
```


## 2026-06-14 Sequential CAS Update: BEMultipliers Repository Inspection

- Active replacement run snapshot after detached relaunch: `tropicalgt_i_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate`, PID `2195134`, W&B id `e3qevo1u`, URL `https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/e3qevo1u`. The run reached step 250 after relaunch and remained alive at the previous check; continue the 5K gate unless it crashes or emits nonfinite/invalid losses.
- `external/BEMultipliers` was cloned and inspected from `amelie-iska/BEMultipliers.git` at commit `d0b55d7c2cb879acc27df533d0117c98a98e463d`; `BuchsbaumEisenbudMultipliers.m2` hash is `562d3c2879e6a306a2b39ff1c9ad7b23689d953757f498abe294161a3eac00b7`.
- BEMultipliers exports `aMultiplier`, `cMultiplier`, `ComputeRanks`, and `exteriorDuality`, and operates on Macaulay2 `ChainComplex` values. It currently cannot run because Macaulay2 is unavailable on the remote machine.
- Important guardrail: BEMultipliers has no built-in safety checks and must never be treated as a resolution backend. It can only annotate a separately certified Macaulay2 free resolution with Buchsbaum-Eisenbud multiplier diagnostics.
- Immediate next CAS work item: provision/bridge Macaulay2 or an equivalent certified backend, then add a smoke probe and renderer for real Betti tables, differential matrices, multidegree shifts, Fitting ideals, minors, and BE diagnostics.

## CAS And Tropical Geometry Research Addenda

- Macaulay2 `Tropical` package: first certified wrapper implemented for `tropicalVariety`, `fan`, `rays`, `maxCones`, `linealitySpace`, `multiplicities`, `isBalanced`, `isPure`, and `isSimplicial`; optional side diagnostics now cover `isTropicalBasis` and `tropicalPrevariety` with independent availability states; inference/audit bundles now always write a tropical fan diagnostic JSON/HTML surface that renders one dimensional cones only from an explicit model-derived ideal plus the Macaulay2 certificate. It returns unavailable states rather than proxies and explicitly does not substitute for multigraded `F2[x_level,x_radius]` persistence-module free resolutions. Remaining package audit items include `tropicalPrevariety`, `tropicalCycle`, `BergmanFan`, cones beyond max-cone summaries, `stableIntersection`, `isTropicalBasis`, and `visualizeHypersurface`.
- Sage `tropical_variety` documentation: review Newton polytope, tropical hypersurface/variety, and polyhedral plotting/construction hooks for exact tropical geometry checks associated with embedding-space probes and TokenGT graph-state coordinates.
- Sage `tropical_mpolynomial` documentation: review tropical multivariate polynomial construction, monomial support, coefficient arithmetic, Newton-polytope data, and exact tropical semiring operations for model-derived tropical polynomial certificates.
- Reference `1710.10651v2.pdf`: review fully before adding any training theorem, paper claim, or implementation hook tied to tropical/toric embeddings.
- CAS package survey: look for Sage, Macaulay2, Singular, RIVET/multipers, polymake, Normaliz, and related packages that compute real multigraded modules, minimal resolutions, Betti tables, Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, tropical fans, toric varieties, and stable intersections.
- Maclagan-style toric embedding research direction: investigate whether the transformer graph-state and tropical-attention coordinate system can be embedded into a toric variety via model-derived monomial coordinates, Newton polytopes, fan data, and one dimensional cone data. Any resulting implementation must distinguish theorem-level certified constructions from diagnostic embeddings or visual probes.

_Last updated: 2026-06-16T07:38:00Z_


## 2026-06-16 Fresh b59 5K Review Gate and Worker Handoff

- The active b58 run was stopped cleanly at user request. Its latest checked validation was step `8750` with BPB `1.331143465780304`, graph-BPB `19.24880762755732`, NLL `0.9226783402264118`, and invalid graph rate `0.0`; target BPB `<1.12` remained open.
- Source commit `70c2e93` added certified-CAS retrieval metrics to periodic training logs before the new launch: certified-CAS availability, match rate, score contribution, similarity, and configured retrieval weight are now exposed for W&B/audit review.
- Fresh b59 step-0 launch uses source commit `70c2e93`, batch size `54`, sequence length `1024`, learning rate `2.1e-4`, checkpoint/validation/visualization cadence `250`, `periodic_viz_got_scaling=true`, `memory_retrieval_probability_map_weight=0.20`, `memory_retrieval_certified_cas_weight=0.12`, and `minimum_steps_before_restart=5000`.
- Readiness audit for b59 returned `status=ready` with no failed gates. The train dry-run was finite on CUDA with BPB `7.969144958556688`, graph-BPB `33.58283549642874`, and `0.0` graph JSON fallback rate.
- Fresh run launch details: PID `73189`, W&B id `itxgxj40`, run URL `https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/itxgxj40`, log `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b59_20260616T001122Z_fresh_bpb112_5k_gate/logs/train_nohup_20260616T001158Z.log`. Early log review reached step `112` with latest train loss/NLL around `1.748/1.724`, no traceback, no OOM, and GPU allocation around `17.8` GiB on the RTX 4090.
- Galileo (`019ecdc9-a3b8-7761-b0e5-36e6adad8e15`) owns the post-5K sidecar: monitor b59 to at least step `5000`, run project analyses/evaluations/visualizations, review metrics and topological/geometric/algebraic artifacts, write `planning/tropicalgt_i_b59_5k_worker_review_2026-06-16.md`, then stop b59 and launch an evidence-backed step-0 restart with adjusted hyperparameters/configs for lower BPB.


## 2026-06-16 5K Step Gate Monitor Addendum

- Added and focused-tested `TropicalGT-I/scripts/monitor_training_step_gate.py` plus `TropicalGT-I/tests/test_training_step_gate_monitor.py`.
- The b59 operational plan is now: keep training alive, run the detached watcher against PID `73189` and the b59 log, terminate only when the parsed step reaches `5000` and required step-5000 artifacts are present after a one-poll settle window or the bounded grace window expires, then hand the resulting stop record and artifact inventory to Galileo for post-5K review and evidence-backed restart.
- The monitor writes generated records under `TropicalGT-I/outputs/training_stop_records/`; those records are operational artifacts and should remain untracked.


## 2026-06-16 Post-5K Review Bundle Helper Addendum

- Added `TropicalGT-I/scripts/prepare_5k_review_bundle.py` so the post-5K worker can turn the b59 config/report/checkpoint/stop-record into a bounded Codex review bundle, active training contract, prompt, eval/visualization command, and interactive-audit validator commands.
- Use this helper after the step gate records target reached; do not stage its generated bundle outputs unless they are deliberately curated planning notes.


## 2026-06-16 Bivariate Staircase Visual Contract Addendum

- `write_two_parameter_bifiltration_visualization` now writes `two_parameter_bifiltration.html` plus `two_parameter_bifiltration.json`. The JSON sidecar records `primary_view=miller_sturmfels_bivariate_staircase`, x_radius horizontal, x_level vertical, coordinate one dimensional cone records, actual-data-only rendering, and no-proxy-resolution policy.
- The old unreachable rank-surface-first implementation block and the retired `trajectory_level_radius_bifiltration_3d` source path key were removed. Keep 3D fiber-rank displays only as secondary diagnostics.
- Already-running b59 periodic artifacts may lack this new sidecar and the tropical fan diagnostic pair; post-5K review must rerender or record unavailable states rather than inferring those missing artifacts.

## 2026-06-16 Legacy Audit Backfill Addendum

- Added `TropicalGT-I/scripts/backfill_interactive_audit_artifacts.py` and `TropicalGT-I/tests/test_backfill_interactive_audit_artifacts.py` for legacy audit directories produced before the current tropical fan diagnostics and bivariate staircase visual sidecar existed.
- Post-5K b59 workflow: before running strict `validate_interactive_audit_artifacts.py` on an older periodic `got_audit`, run the backfill helper. It writes explicit unavailable tropical fan diagnostics when no certified model-derived ideal exists and regenerates the bifiltration visual contract only from the raw bifiltration payload.
- Live step-500 b59 smoke after backfill validated successfully. Treat the helper as a repair/rerender step for audit completeness, not as mathematical evidence of a CAS certificate, tropical fan, or free resolution.


## 2026-06-16 Periodic Trace Cap Addendum

- b59 reached step 2500 before failing on `OSError: [Errno 28] No space left on device` during periodic audit JSON writing. This is an artifact-retention failure, not a loss/NaN/OOM failure.
- The training loop now records and applies an effective periodic GoT trace limit from the training-safe budget. In-training periodic audits should remain bounded; full-depth/high-trace audits belong in the post-5K offline analysis phase after disk pruning.
- Resume b59 from `TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b59_20260616T001122Z_fresh_bpb112_5k_gate.latest.pt` and continue to the 5000-step gate after pruning generated intermediate audit bundles.


## 2026-06-16 Periodic Audit Retention Addendum

- Future long runs may set `periodic_prune_got_audit_keep_latest` and `periodic_prune_got_audit_keep_steps` to keep generated audit evidence bounded without losing compact validation reports.
- The retention policy removes only generated `periodic/step_*/got_audit` folders and records actions in `periodic/got_audit_retention_manifest.jsonl`.
- Use this for post-5K restarts that retain frequent validation but do not need every intermediate multi-gigabyte interactive audit bundle.


## 2026-06-16 b60 Analogical Memory No-Proxy Addendum

- Fresh b60 (`tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate`) is running under the 5K step gate with trainer PID `378962`, watcher PID `379304`, and W&B id `ld5u55p5`. Latest checked watcher state reached step `250` with train loss/NLL `1.543/1.521`, no fatal markers, and the target step-5000 validation/audit paths still pending.
- Analogical retrieval rows now carry explicit probability simplex-tree certificate fields from the model-probability Jensen-Shannon assignment: vertex assignment count, checked/preserved simplex counts, edge and 2-simplex preservation rates, JS/assignment summaries, filtration distortion, chain-map certification, and persistence-module morphism certification.
- Probability-map retrieval score contribution is positive only when the stored model-probability assignment extends to a filtration-preserving simplex-tree map. If edge/face/simplex preservation fails, the contribution is exactly zero and the failed counts are exposed.
- The analogical memory visualization now consumes the retrieval-side `probability_simplicial_map` certificate as the source of truth. It does not recompute a map from raw query/codomain complexes and does not use embedding-only, heuristic, proxy, or compatibility fallback maps.
- If a retrieval row lacks the probability simplex-tree certificate, the artifact renders `missing_retrieval_probability_simplicial_map_certificate` with unavailable chain-map and persistence-module morphism diagnostics. Raw probability complexes alone are not enough to render an analogical map in this page.
- Focused and full verification completed: `py_compile` on memory/visualization/tests, `pytest TropicalGT-I/tests/test_metrics_and_memory.py -q` (`11 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`28 passed`), and `git diff --check`.


## 2026-06-16 Simplex-Tree Radius Start-State Addendum

- Reasoning-trajectory radius complexes now treat a single model-evaluated GoT state as a valid radius filtration when every displayed vertex has the required real metric vector. The complex starts as disjoint zero-simplices at filtration `0.0`; edges and 2-simplices appear only when actual embedding/probability distances exist.
- `build_reasoning_trajectory_complex` records `radius_filtration`, `metric_vertex_count`, `metric_vertices_complete`, and `single_vertex_radius_filtration` in the summary. Missing required metric/probability vectors still render unavailable with `unavailable_missing_<metric>_radius_vertices`.
- The scaling and visualization probability-complex gates now accept real vertex-only Jensen-Shannon radius complexes and require every 0-simplex to carry a valid model probability vector. They no longer require an edge before recognizing a real probability filtration.
- Fixed a zero-simplex gate bug where `dimension=0` was treated as missing by an `or -1` expression during probability-complex validation.
- The visualization-side GUDHI canonicalizer now emits the same `simplex_tree.available=true`, `simplex_tree_backend`, and `simplex_tree_available` metadata as the source builder for raw/legacy simplex JSON payloads.
- GUDHI closure-inserted faces are now explicitly disclosed with `gudhi_closure_inserted=true`, `filtration_source=gudhi_simplex_tree_closure`, and `simplex_tree_closure_inserted_simplices` counts instead of silently appearing as raw model simplices.
- Verification completed: `py_compile` on simplicial/scaling/visualization/tests, `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`31 passed`), `pytest TropicalGT-I/tests/test_metrics_and_memory.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`21 passed`), and `git diff --check`.


## 2026-06-16 Certified CAS Diagnostic Table Addendum

- The two-parameter bifiltration page now separates certified algebra details into distinct secondary tables for Betti-style diagnostics, free modules, differentials, certified Fitting ideals/determinantal minors, Buchsbaum-Eisenbud rank and multiplier diagnostics, and the CAS certificate summary.
- Fitting ideals and determinantal minors are rendered only from explicit structured `ideal_diagnostics` certificates attached to the certified CAS free-resolution artifact. Legacy/raw `fitting_ideals` and `minors` keys are not rendered as a substitute when the structured certificate is absent.
- Buchsbaum-Eisenbud rows are rendered only from explicit certified resolution fields, BEMultipliers output fields, or structured `buchsbaum_eisenbud_rank_conditions`. Missing structured data renders an unavailable diagnostic with the exact reason; no default `False` exactness/minimality rows are fabricated from absent evidence.
- The bifiltration visual payload now advertises `certified_fitting_minor_tables` and `buchsbaum_eisenbud_diagnostic_tables` as secondary views while preserving the Miller-Sturmfels bivariate staircase as the primary view. The page continues to state that diagnostic chain data is not substituted for a free resolution.
- b60 latest checked training state reached step `500` with train loss/NLL `1.428/1.406`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_certified_cas_diagnostic_tables_require_explicit_structured_certificates -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_level_radius_bifiltration_reports_scoped_real_staircase_resolution -q` (`1 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`32 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`19 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Bifiltration Vertex-Only Probability Start-State Addendum

- The inference-scaling bifiltration selection policy now explicitly accepts real Jensen-Shannon probability radius complexes whose start state is a single vertex with a valid model probability vector and no edges yet. This keeps the `F2[x_level,x_radius]` report aligned with radius filtrations that start as disjoint vertices and grow only when real radius distances add edges/faces.
- The selected bifiltration object key remains `probability_filtered_simplicial_object` whenever every growth row has real probability-vector evidence on all displayed vertices. Embedding radius complexes are selected only when that probability-vector evidence is unavailable; no proxy probability or synthetic edge is introduced.
- b60 latest checked training state reached step `515` with train loss/NLL `1.382/1.360`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/scaling.py TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_inference_scaling_bifiltration_uses_vertex_only_probability_start_state -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`20 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`32 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Missing-Bifiltration Unavailable-State Addendum

- Legacy or partial inference-scaling payloads with nonempty candidates but no `trajectory_growth` now materialize `trajectory_level_radius_bifiltration.json` as an explicit unavailable `F2[x_level,x_radius]` report instead of silently omitting the two-parameter sidecar.
- The unavailable report preserves the coefficient ring, two parameters, empty fiber/rank/structure-map arrays, selected object key `unavailable`, and the exact reason `trajectory_level_radius_bifiltration_missing_for_nonempty_scaling_report_without_trajectory_growth`. Empty or invalid trajectories continue to receive the separate empty/invalid reason.
- No substitute fibers, ranks, generators, maps, or synthetic radius edges are produced for missing growth rows.
- b60 latest checked training state reached step `570` with train loss/NLL `1.352/1.330`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py`, `pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_inference_audit_materializes_unavailable_bifiltration_for_legacy_nonempty_scaling -q` (`1 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`20 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Rank-Invariant Table And 2-Parameter Persistence Display Addendum

- The two-parameter bifiltration page now renders a secondary `Rank-invariant samples over F2[x_level,x_radius]` table sourced only from computed `rank_invariant_samples`. Rows show source grade, target grade, H0 inclusion rank, and the corresponding source/target monomials.
- The visual payload now advertises `rank_invariant_samples_table` and records `rank_invariant_sample_count`, so browser/audit validators can distinguish the primary Miller-Sturmfels staircase from secondary rank-invariant diagnostics.
- This completes the current actual 2-parameter persistence display checklist: grid fibers come from closed GUDHI simplex-tree simplices and stored grades, adjacent `x_level`/`x_radius` structure maps are emitted, rank/Betti fiber displays and rank-invariant samples are rendered, downloadable JSON carries fiber bases/ranks/grades/provenance, and the 3D lattice remains a secondary view with small H0/H1 offsets while z stays actual fiber rank.
- b60 latest checked training state reached step `614` with train loss/NLL `1.346/1.324`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_level_radius_bifiltration_reports_scoped_real_staircase_resolution -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`20 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 2210 Methodology And Known Monomial Test Addendum

- The `references/2210.11433v1.pdf` methodology review now reflects the live backend state: Macaulay2 `/usr/bin/M2` and Singular `/usr/bin/Singular` are detected, Sage is not on the active shell PATH, and BEMultipliers is cloned at `external/BEMultipliers` commit `d0b55d7` as a diagnostic layer only.
- Added exact regression coverage for singleton, two-generator, and three-generator bivariate monomial staircase ideals, including adjacent-LCM first syzygies, Hilbert-Burch/Miller-Sturmfels-style free modules, Betti rows, and one dimensional cone language.
- The sequential queue now marks the 2210 review/implementation checklist complete for the paper-derived rank-invariant, Fitting/minor, Buchsbaum-Eisenbud diagnostic, and research-figure rendering objects already implemented. Remaining CAS hardening continues in the real-resolution section.
- b60 latest checked training state reached step `667` with train loss/NLL `1.362/1.340`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `python -m py_compile TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_bivariate_staircase_resolution_known_monomial_ideals -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`21 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## Real Implementations Only Policy

No TropicalGT-I metric, loss, visualization, analogical map, persistence module, free resolution, derived comparison, tropical-cycle diagnostic, or CAS artifact should be presented as a mathematical object unless it is computed from the actual model outputs, graph states, embeddings, probabilities, simplex trees, bifiltrations, or certified CAS/backend output that define that object. Temporary placeholders, synthetic fallback objects, mock charts, fabricated simplices, and convenience stand-ins are not acceptable. When a requested object cannot yet be computed, the artifact must render an explicit unavailable/uncertified state and the training metric must either be disabled or logged under an audit-only unavailable flag. Finite chain-presentation diagnostics may be shown only as chain diagnostics, never as free resolutions. Total-graded or ungraded CAS output may be shown as real CAS output only under its actual grading; it must not be advertised as a multigraded `F2[x_level,x_radius]` free resolution unless the backend certifies that multigraded structure.

Use "one dimensional cone" or "one dimensional cones" as the preferred fan-theoretic language whenever the intended object is a cone of a fan or a cone-indexed filtration datum. Use singular or plural according to ordinary grammar.

_Last updated: 2026-06-16T07:38:00Z_
