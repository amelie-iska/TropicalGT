# TropicalGT-I Current Required Updates and Repair List

This checklist merges the browser/photo review, the current active training state, the CAS/free-resolution policy, and the new tropical vector-bundle / tropical toric-embedding paper workstream. All work is remote-only on `iska@iska` in `/home/iska/Documents/amelie/bio/TropicalGT`, branch `tropicalgt-i-real-cas-no-proxy-20260614` or a newer non-main successor branch.

## 1. Active Training and BPB Priority

- `tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate` reached the step-5000 gate. Trainer PID `378962` and watcher PID `379304` are no longer running; W&B run id: `ld5u55p5`; run URL: `https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/ld5u55p5`; config: `TropicalGT-I/outputs/launch_configs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.json`; output dir: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate`; stop record: `TropicalGT-I/outputs/training_stop_records/b60_fresh_step0_casrows_step5000_gate.json`.
- The required step-5000 validation and audit artifacts exist, but the required checkpoint `TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt` is a zero-byte file and is therefore unavailable. Under the no-proxy/no-fallback policy, post-5K eval/visualization/backfill/validator commands must not run from this checkpoint; `prepare_5k_review_bundle.py --run-*` fails closed with `empty_checkpoint`, and any restart decision must record this missing evidence explicitly.
- Preserve BPB and graph-BPB as primary optimization and promotion gates.
- Use advanced auxiliaries only when they are zero-default or BPB-gated and ablated: tropical support, GFlowNet GoT rewards, GraphCG, persistence/landscape diagnostics, chart-bundle transports, tropical toric active-cell agreement, and memory retrieval.
- Track step, VRAM, wall time, train/eval NLL, BPB, graph-BPB, certificate loss, tropical wall-hit rate, support entropy, GraphCG rank, meet-in-the-middle agreement, ROAR/causal decoding path mix, and artifact-generation status.
- With b60 stopped at the 5K gate, the immediate focus is the real-evidence review state: preserve the path-only 5K bundle, dispatch the Codex evidence-review subagent, and do not launch a step-0 restart unless the restart schema can cite real metrics/sidecars and explicitly account for the unavailable checkpoint. Historical b59 notes below are retained only for provenance and are not active run instructions.

## 2. Browser QA and Visual Evidence

- Keep the served audit bundle at `127.0.0.1:8991` pointed to the latest completed real b60 periodic audit; current browser serving is step `5000` without copying generated artifacts. The step-5000 validation report records BPB/exact BPB `1.4304583543547733`, graph-BPB `20.122144813809587`, graph-conditioned BPB without side cost `1.2576335315794374`, NLL `0.9915181752294302`, and invalid graph rate `0.0`. During step-3250 through step-5000 audit generation, generated `step_00002750/got_audit`, `step_00003000/got_audit`, `step_00003250/got_audit`, `step_00003500/got_audit`, `step_00003750/got_audit`, `step_00004000/got_audit`, `step_00004250/got_audit`, and `step_00004500/got_audit` were pruned under the active retention policy after the next audit step existed; milestone audits, step 4750, step 5000, compact validation reports, and non-audit artifacts remain.
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

- Add zero-default config flags for monomial chart transports, scoped exponent-chart sidecars, certified finite toric-ideal sidecars, chart BPB consistency, landscape transport metrics, and atom stability.
- Model hooks must emit chart ids, overlap pairs/triples, toric active rows, active tropical support cells, GraphCG active directions, chart-local NLL, and transport ids without changing logits when coefficients are zero.
- Implement `MonomialTransportHead`, `BundleMatroidHead`, and `ToricEmbeddingHead` only behind config gates.
- Add losses/metrics: `bundle/transport_l1`, `bundle/cocycle_defect`, `bundle/flat_rank_defect`, `toric/normal_fan_loss`, `graphcg/toric_cell_agreement`, `chart/bpb_consistency`, `bundle/atom_stability_gap`, `memory/transported_landscape_l2`, and `memory/transported_landscape_cosine`.
- Treat all of these as auxiliary until matched ablations show BPB/graph-BPB benefit.


Implementation checkpoint, 2026-06-16:
- Zero-default config flags and model hooks are now implemented for chart-bundle/toric telemetry and losses: bundle transport L1, cocycle defect, flat-rank defect, toric normal-fan loss, GraphCG-toric cell agreement, chart-BPB consistency availability, and atom-stability gap.
- The implementation is deliberately auxiliary and gated. Disabled defaults emit zero metrics; enabled zero-weight hooks preserve logits and loss; positive coefficients affect only the auxiliary regularizer.
- Chart-local NLL/BPB partitions, explicit chart overlap pair/triple ids, and transport-gated memory persistence-landscape L2/cosine diagnostics are now implemented where their evidence exists. Source-side matched-ablation promotion gates are now implemented for chart-bundle/toric coefficients; runtime promotion still requires real matched-seed BPB and graph-BPB ablation reports before any nonzero coefficient is accepted.


### Paper Workstream Status: Vector Bundles and Tropical Toric Embeddings

- Subagent Avicenna reviewed `references/2405.03505v1.pdf` and `references/2009.03030v2.pdf` in full from extracted text and updated `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex`.
- The new paper material introduces TropicalGT chart bundles, TokenGT tropical atlases, monomial tropical chart transports, one dimension cone-filtration atomization, scoped exponent-chart sidecars, certified finite toric-ideal sidecars, matroid flat-defect objectives, GraphCG/exponent-chart agreement, chart-BPB consistency, and transported persistence-landscape memory metrics.
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

Status: complete for structured Fitting/minor, Buchsbaum-Eisenbud rank, and CAS-emitted grade/depth diagnostic checkpoints. Independent regular-element certification remains unavailable unless a backend explicitly emits such a certificate.

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
- Maclagan-style toric embedding research direction: investigate whether finite model-derived monomial coordinates, Newton polytopes, certified fan data, and one dimensional cone data define a tool-backed toric sidecar. Any resulting implementation must distinguish theorem-level certified tropical-variety embeddings into toric varieties from scoped exponent-chart diagnostics or visual probes.


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


## 2026-06-16 CAS Backend Probe Contract Addendum

- The CAS backend probe is now covered by a regression test that compares `probe_cas_backends()` against the adapter's deterministic executable discovery, not only the active shell `PATH`. This records the live remote contract for Macaulay2 `/usr/bin/M2`, Singular `/usr/bin/Singular`, and any unavailable Sage executable without inventing a backend state.
- The BEMultipliers probe is explicitly asserted as `is_resolution_backend=false`, with the execution policy requiring use only after a CAS-certified free resolution and never as a substitute certificate.
- The sequential queue marks backend detection complete and leaves the remaining CAS hardening items open until each has direct evidence: bridge provenance, minimal multigraded resolution content, Betti/differential/Fitting/minor/BE certificate rows, and exact unavailable rendering.
- b60 latest checked training state reached step `750` with train loss/NLL `1.327/1.305`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `python -m py_compile TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_cas_backend_probe_reports_detected_executable_paths -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`22 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Strict CAS Bridge Provenance Addendum

- Fixed `unavailable_real_resolution` so failed certification paths that already hold a canonical module schema keep that schema and its `input_sha256` instead of trying to recanonicalize it as raw input. This prevents failed backend runs from losing generator/boundary provenance while still refusing to render uncertified CAS artifacts.
- Added a failed-certificate regression where a Macaulay2-tagged backend response with `exactness_certified=false` and `certificate_attached=false` returns `certificate_failed`, preserves the original module hash, generator count, and boundary count, leaves `cas_artifacts={}`, and keeps `certificate_attached=false`.
- The sequential queue marks the strict CAS bridge/provenance item complete. The next open CAS item is certified minimal multigraded free-resolution content over `F2[x_level,x_radius]` when Macaulay2 can provide it.
- b60 latest checked training state remained at step `750` with train loss/NLL `1.327/1.305`; trainer PID `378962` and watcher PID `379304` were alive under the step-5000 gate, with the GPU pulse showing `22381/24564` MiB allocated and transient `0%` utilization during the check.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_failed_cas_certificate_preserves_module_provenance_without_artifacts -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`23 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Explicit Macaulay2 Multigraded Resolution Addendum

- The Macaulay2 bridge now builds the presentation map as `M = map(F0, F1, matrix ...)`, where `F0` and `F1` are shifted free modules derived from the canonical row and column generator multidegrees. This ties `res coker M` to the actual stored `F2[x_level,x_radius]` bifiltration grades instead of allowing Macaulay2 to infer shifts from an ungraded matrix shell.
- The generated script checks `homogeneousPresentation = isHomogeneous M` before resolving. Nonhomogeneous stored presentations emit an uncertified tagged response with no multigraded-resolution claim.
- The live homogeneous smoke fixture now asserts the exact shifted module contract (`F0 = R^{{0,-1},{-1,0}}`, `F1 = R^{{-1,-1}}`) and the certified multigraded Betti rows `(0,(0,1),1)`, `(0,(1,0),1)`, and `(1,(1,1),1)` when Macaulay2 is available. A paired nonhomogeneous fixture asserts `certificate_failed` with empty `cas_artifacts`.
- b60 latest checked training state reached step `853` with train loss/NLL `1.306/1.285`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate, with GPU allocation `22381/24564` MiB during the pulse.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_real_cas_free_resolution_smoke_when_backend_available -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_macaulay2_rejects_nonhomogeneous_stored_multigrading -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`24 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Structured CAS Certificate Content Addendum

- Certified real-free-resolution outputs now include `cas_artifacts.certificate_summary`, a structured certificate row with backend, certificate type, attached/exact/minimal booleans, homogeneity status, presentation shape, coefficient ring, module schema version, input hash, render-safe grading flags, backend-attempt count, and the no-proxy policy.
- The existing CAS artifacts remain structured and certificate-backed: Macaulay2 multigraded Betti rows/free modules, differential matrices with source/target degrees, Fitting ideals, determinantal minors, Buchsbaum-Eisenbud rank conditions, and explicit BEMultipliers diagnostics when available.
- The two-parameter visualization adapter now passes `certificate_summary` into the selected resolution payload, and the certificate table displays certificate type, homogeneity, input hash, and no-proxy policy alongside exactness/minimality and BE diagnostics.
- b60 latest checked training state reached step `954` with train loss/NLL `1.294/1.273`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate, with GPU allocation `22381/24564` MiB during the pulse.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_simplicial_visualization.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_certified_cas_result_surfaces_buchsbaum_eisenbud_diagnostics TropicalGT-I/tests/test_algebraic_persistence.py::test_real_cas_free_resolution_smoke_when_backend_available -q` (`2 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py::test_certified_cas_diagnostic_tables_require_explicit_structured_certificates -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`24 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Explicit CAS Unavailable Rendering Addendum

- Uncertified real-free-resolution reports now include `unavailable_diagnostic` and `unavailable_dependency_action`. These fields give a status-specific action for missing backend executables, disabled CAS execution, complexity guards, timeouts, invalid grading, unsupported rings, and failed certificates.
- The unavailable diagnostic also records available backend names, backend attempt statuses, BEMultipliers' non-backend role, a safe-unavailable-render flag, and the no-proxy policy: chain diagnostics, rank samples, Fitting ideals, minors, and BEMultipliers output must not substitute for a certified free resolution.
- The deterministic no-backend regression monkeypatches all CAS executable probes unavailable and verifies that `backend_not_installed` renders only as an unavailable diagnostic with an install/activation action and empty `cas_artifacts`.
- Section 4 CAS integration is now complete in the sequential queue. The next open implementation section is derived/analogical maps using probability-vector assignments and certified algebra evidence.
- b60 latest checked training state reached step `996` with train loss/NLL `1.251/1.229`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate, with GPU allocation `22381/24564` MiB during the pulse.
- Verification completed: `python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_real_cas_free_resolution_caches_deterministic_unavailable_probe -q` (`1 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`24 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Derived Analogical Maps Completion Addendum

- Section 5 is complete by audit of current source and tests. Retrieval uses model-predicted probability-vector complexes and Jensen-Shannon assignment certificates; embedding/signature similarity remains a score component but is not allowed to certify a simplicial map.
- Retrieval rows include probability-map certificate fields for vertex assignments, JS/assignment summaries, edge and 2-simplex preservation, simplex-tree preservation, chain-map certification, persistence-module morphism certification, and transported persistence-landscape diagnostics.
- Certified CAS retrieval evidence is exact-match only over certified real-free-resolution artifacts. Mismatches and unavailable evidence are reported with reasons, produce zero CAS score contribution, and do not assert derived-category equivalence.
- The analogical browser views render top-k probability correspondences and explicit unavailable states for no memory, missing retrieval-side probability certificates, missing query probability complexes, and non-trajectory probability fallbacks.
- b60 latest checked training state reached step `1000` with train loss/NLL `1.230/1.208`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate, with GPU allocation `22381/24564` MiB during the pulse.
- Verification completed: `pytest TropicalGT-I/tests/test_metrics_and_memory.py -q` (`11 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`33 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 SimplexTree No-Fallback And Radius Slider Addendum

- SimplexTree rendering now refuses to use raw JSON simplex rows as a substitute when GUDHI cannot build a real `SimplexTree`. The page renders `unavailable_gudhi_simplex_tree` with the exact import/build error and states that no trie or face-coface poset is rendered without GUDHI evidence.
- The visualization and core simplicial serializers attach `simplex_tree_no_proxy_or_fallback=true`, `safe_to_render_simplex_tree=false`, and `simplex_tree_unavailable_reason` on GUDHI failures.
- The provenance registry now treats this as an explicit certified-unavailable state rather than a serialization fallback.
- Section 6 remains open only for the duplicate/extraneous-panel cleanup audit. The per-step complex/simplex-tree manifest, dotted causal/decoding overlays, solid radius edges, filled radius-gated 2-simplices, and min-to-max radius slider contracts are now covered by regression tests.
- b60 latest checked training state reached step `1100` with train loss/NLL `1.176/1.155`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate, with GPU allocation `22381/24564` MiB during the pulse.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/src/tropicalgt/simplicial.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_simplicial_visualization.py`, focused no-fallback/trajectory visualization tests (`2 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`34 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_reasoning_trajectory_complex_grows_by_level -q` (`1 passed`), and `git diff --check`.


## 2026-06-16 Duplicate Reasoning-Step Panel Removal Addendum

- Pages that already render the interactive selected filtered-complex panel no longer include the adjacent static preview block for the same selected reasoning-step object. This removes the duplicate side-panel surface while leaving the interactive panel, hover card, and per-step linked complex pages intact.
- Static same-payload preview remains available only on pages that do not enable the interactive selected-complex panel, so it is not a duplicate rendering of the same reasoning step.
- Section 6 is complete in the sequential repair queue. The next open implementation section is tropical support and wall-crossing diagnostics.
- b60 latest checked training state reached step `1198` with train loss/NLL `1.278/1.254`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate, with GPU allocation `22381/24564` MiB and transient utilization `46%` during the pulse.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py`, focused duplicate-panel tests (`3 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`34 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), and `git diff --check`.


## 2026-06-16 Tropical Support Grouped Labels Addendum

- Tropical support heatmaps now summarize model graph tokens by grouped trace labels (`kind:semantic_type`) from real `graph_token_trace` metadata only. The payload records query/support group summaries and a no-proxy grouping policy so downstream audits can verify labels are not fabricated.
- The support payload now records a top-support summary with selected-query count, capture rate, mean selected margin, support group, and model support-probability mean when `model_tropical_support_probabilities` are present.
- The observed-support matrix and high-collapse diagnostic views now display grouped token labels, top-support group, margin profiles, strict wall-hit counts, near-wall hit counts, and collapse metrics while keeping the matrix contract explicit: selected cells are assignment events, not confidence scores.
- b60 latest checked training state reached step `1250` with train loss/NLL `1.234/1.211`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py`, focused tropical support tests (`2 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`34 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`).


## 2026-06-16 Wall-Hit Metric Scope Addendum

- Tropical wall-hit reporting is now explicitly scoped as a model tropical-margin threshold audit, not a certified normal-fan wall-crossing count. Low strict wall-hit rates are interpreted through the actual margin bands: strict events observed, near-wall-only ambiguity, interior chamber margins, or metric issue when finite margins are unavailable.
- `tropical_support_payload.json` now records `metric_scope`, strict and near-wall definitions, `low_strict_wall_interpretation_status`, explanatory text, and a boolean `metric_issue` flag inside `wall_margin_audit`.
- The visualization preserves explicit zero thresholds and displays wall-audit scope plus low-strict interpretation in both observed-support and collapse-diagnostic views.
- The interactive artifact validator now requires the no-proxy wall-audit scope and interpretation fields for current payloads with wall-margin audit data.
- b60 latest checked training state reached step `1287` with train loss/NLL `1.235/1.212`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py`, focused tropical support wall tests (`3 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`35 passed`), and `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`).


## 2026-06-16 Tropical Margin Loss Sign Addendum

- `tropical_margin_loss` now logs the nonnegative margin shortfall loss. The signed reward-maximizing objective is explicit under `tropical_margin_signed_loss` and `tropical_margin_signed_objective`, with `tropical_margin_reward` recording the positive margin reward.
- Weighted telemetry now separates the training objective from the diagnostic shortfall: `loss_tropical_margin_signed_weighted` is the signed regularizer used in `loss_regularizer_total`, while `loss_tropical_margin_shortfall_weighted` is a nonnegative diagnostic. The legacy `loss_margin_weighted` remains as the signed-objective alias for continuity.
- W&B priority grouping and regression tests were updated so BPB reviewers do not confuse a negative signed objective with a bad nonnegative loss.
- b60 latest checked training state reached step `1357` with train loss/NLL `1.196/1.173`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/model.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py`, focused model/W&B checks, `pytest TropicalGT-I/tests/test_losses_and_model.py -q` (`9 passed`), and `pytest TropicalGT-I/tests/test_training_metrics.py -q` (`12 passed`).


## 2026-06-16 Certificate Loss Decomposition Addendum

- `certificate_loss` remains the real negative-log allowed-support objective. `certificate_objective_loss` records the same value explicitly, while `certificate_diagnostic_penalty` is a separate zero-valued channel and `certificate_loss_reconstruction_error` verifies no hidden diagnostic penalty is mixed into the objective.
- Certificate telemetry now includes valid-token count and allowed-target count summaries, giving BPB reviewers direct context for rising certificate loss from real graph-token target structure.
- Weighted telemetry now separates `loss_certificate_objective_weighted` from `loss_certificate_diagnostic_penalty_weighted`; `loss_certificate_weighted` remains the objective-weight alias for continuity.
- Section 8 of the sequential repair queue is now complete: support heatmap readability, wall-hit metric scope, tropical margin sign, and certificate-loss decomposition are all covered by tests.
- b60 latest checked training state reached step `1407` with train loss/NLL `1.178/1.156`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/model.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py`, focused certificate/W&B checks, `pytest TropicalGT-I/tests/test_losses_and_model.py -q` (`9 passed`), and `pytest TropicalGT-I/tests/test_training_metrics.py -q` (`12 passed`).


## 2026-06-16 GraphCG Embedding-Rank Direction Bank Addendum

- GraphCG model construction now enforces an effective direction bank of size `dim`, clamping any explicit `graphcg_num_directions` request to the embedding dimension while preserving the requested count as telemetry.
- The exact QR/Stiefel projection cap now covers the active `1760 x 1760` bank, so downstream GraphCG projections use the full-rank effective basis instead of normalized raw rows.
- New metrics include `graphcg_requested_num_directions`, `graphcg_effective_num_directions`, `graphcg_embedding_span_rank_target`, `graphcg_embedding_span_full_rank`, and `graphcg_direction_bank_clamped_to_embedding_dim`; W&B priority grouping includes these under GraphCG.
- b60 latest checked training state reached step `1479` with train loss/NLL `1.246/1.223`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/losses.py TropicalGT-I/src/tropicalgt/model.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py`, focused GraphCG rank tests (`4 passed`), focused training priority tests (`2 passed`), `pytest TropicalGT-I/tests/test_losses_and_model.py -q` (`10 passed`), and `pytest TropicalGT-I/tests/test_training_metrics.py -q` (`12 passed`).


## 2026-06-16 GraphCG Readability Contract Addendum

- The GraphCG trajectory browser payload now certifies the readable four-panel layout: all-direction heatmap, full-rank activity spectrum, candidate activity by observed GoT state, and direction signed-bias scatter.
- The payload records panel names/count, panel availability flags, and `directions_sampled_for_heatmap=false`, keeping the no-sampling all-direction heatmap contract explicit while hover/payload retain exact ids.
- Regression tests now require the four panel titles and the candidate effective-direction plus signed-mean arrays.
- b60 latest checked training state reached step `1500` with train loss/NLL `1.214/1.191`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py`, focused GraphCG visualization test (`1 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`35 passed`), and `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`).


## 2026-06-16 GraphCG Metric Priority Addendum

- W&B GraphCG metrics now surface the BPB-review-critical rank facts first: `graphcg_embedding_span_full_rank`, `graphcg_direction_bank_clamped_to_embedding_dim`, requested/effective direction counts, and embedding-span rank target before lower-level active-rank and singular-value diagnostics.
- `training_metrics.html` now uses an explicit browser GraphCG priority tuple, so the local/browser audit exposes the same real rank-audit metrics plus active-direction and singular-value diagnostics in stable order.
- Regression coverage now verifies both the W&B GraphCG payload order and the generated browser Plotly trace order. The prior direction-bank pass already covers the direction-rank configuration tests, so Section 9 is complete in the sequential queue.
- b60 latest checked training state reached step `1542` with train loss/NLL `1.225/1.202`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_training_metrics.py`, focused GraphCG metric-order tests (`2 passed`), `pytest TropicalGT-I/tests/test_training_metrics.py -q` (`13 passed`), and `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`35 passed`).



## 2026-06-16 Meet-In-The-Middle Toggle Addendum

- Meet-in-the-middle decoding is now unambiguously behind the shared config toggle: dictionary configs keep their explicit parameters, boolean `true` enables default zero-weight diagnostics, and boolean `false` disables the path.
- Disabled MIM paths now report `disabled_by_config` instead of silently omitting the structured report. The batch helper returns before any reverse model call when disabled, satisfying the no-proxy/no-fallback requirement.
- `evaluate_model`, `eval_tropicalgt_i.py`, and `infer_tropicalgt_i.py` now all use the shared parser, so boolean launch artifacts do not crash CLIs or silently become default-off empty dictionaries.
- b60 latest checked training state reached step `1732` with train loss/NLL `1.239/1.215`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/decoding.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/scripts/eval_tropicalgt_i.py TropicalGT-I/scripts/infer_tropicalgt_i.py TropicalGT-I/tests/test_meet_in_middle_decoding.py`, `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`10 passed`), and `pytest TropicalGT-I/tests/test_training_metrics.py -q` (`13 passed`).


## 2026-06-16 Causal Forward/Reverse Decoding Addendum

- Causal DAG graph records already compute a forward topological decoding order plus a reverse causal order. The new regression drives an explicit causal DAG through the full meet-in-the-middle batch path and verifies the report uses `causal_dag` with `reverse_causal_dag` under graph autoregressive decoding.
- This keeps forward/reverse causal decoding tied to actual graph metadata and model logits; no byte-reversal proxy or synthetic reverse graph is substituted for causal DAGs.
- b60 latest checked training state reached step `1750` with train loss/NLL `1.151/1.128`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/tests/test_meet_in_middle_decoding.py` and `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`11 passed`).


## 2026-06-16 ROAR Random-Order Decoding Addendum

- Cyclic and explicitly noncausal graphs remain on ROAR/random-order autoregressive semantics. The new regression drives a cyclic graph through the full meet-in-the-middle batch path and verifies the report uses `random_autoregressive` with `reverse_random_autoregressive` under graph autoregressive decoding.
- This keeps cyclic/noncausal decoding tied to real graph metadata and model logits; no causal-DAG proxy is fabricated when the graph contains a directed cycle or explicit noncausal edge.
- b60 latest checked training state remained at the latest parsed step `1750` with train loss/NLL `1.151/1.128`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/tests/test_meet_in_middle_decoding.py` and `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`12 passed`).


## 2026-06-16 Dataset Causal Annotation Preservation Addendum

- Dataset graph normalization keeps causal annotations real and conservative: known causal/temporal edge types are directed causal edges, explicit noncausal or undirected edges remain noncausal, and mixed graphs do not get promoted to causal DAGs.
- The new regression covers a graph with causal reasoning edges plus an explicit noncausal similarity edge. It verifies the noncausal edge is preserved, the graph is decoded as `random_autoregressive`, and reverse decoding is `reverse_random_autoregressive`.
- b60 latest checked training state remained at the latest parsed step `1750` with train loss/NLL `1.151/1.128`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/tests/test_data_loader.py`, `pytest TropicalGT-I/tests/test_data_loader.py -q` (`13 passed`, `2 warnings`), and `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`12 passed`).


## 2026-06-16 Dotted Decoding/Causal Visualization Overlay Addendum

- The relevant visualization payloads keep causal graph edges plus forward and reverse decoding order edges as dotted directed overlays. They remain semantically separate from solid radius edges and filled radius-gated 2-simplices.
- The new cyclic-graph regression verifies ROAR forward/reverse decoding edges are present and dotted in both the overlay object and the browser plot payload, so cyclic/noncausal graph overlays are not silently hidden or rendered as simplicial edges.
- Section 10 of the sequential queue is complete: MIM toggle semantics, causal forward/reverse decoding, ROAR random-order decoding, dataset causal/noncausal preservation, and dotted visualization overlays are implemented or regression-covered.
- b60 latest checked training state reached step `1769` with train loss/NLL `1.174/1.150`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: `/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/tests/test_simplicial_visualization.py`, focused decoding overlay tests (`2 passed`), and `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`36 passed`).

## 2026-06-16 Tropical/Toric Research Contract Addendum

- Section 11 research checkpoint complete for Macaulay2 `Tropical`, Sage tropical polynomial/variety APIs, Maclagan-Rincon tropical-ideal/toric-scheme scope, and `references/1710.10651v2.pdf`.
- The implemented fan certificate path is Macaulay2 `Tropical` on an explicit `QQ[x_i]` ideal: `needsPackage "Tropical"`, `tropicalVariety`, `rays`, `maxCones`, `linealitySpace`, `multiplicities`, `isBalanced`, `isPure`, `isSimplicial`, and `fan` are the required source methods; `isTropicalBasis` and `tropicalPrevariety` remain side diagnostics with their own availability states.
- Sage tropical polynomial/variety APIs are documented as useful for exact tropical polynomial, curve, hypersurface, and plotting checks, but they are not accepted as replacements for the Macaulay2 ideal-to-tropical-cycle fan certificate in the current browser view.
- Maclagan-Rincon tropical-ideal and tropical toric-scheme language remains research scope unless a backend certifies the exported fan, ideal, grading, or sheaf/module object. The paper may use the language for scope, but implementation claims must stay certificate-backed.
- Tropical fan diagnostic payloads now carry a machine-readable `certificate_contract` with source method, ordinary-to-Laurent torus scope, Sage scope, Maclagan toric-scheme scope, and no-proxy policy. The browser table renders those fields for both certified and unavailable states.
- No support-token, chain-presentation, rank-sample, embedding-only, or visualization diagnostic can be treated as a tropical fan/cycle certificate. Unavailable remains unavailable.
- b60 latest checked training state reached step `1947` with train loss/NLL `1.129/1.105`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- Verification completed: py-compile for touched Python/test files, focused tropical-fan tests (`3 passed` + `2 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`24 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`36 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), generated Plotly HTML smoke, and `git diff --check`.

## 2026-06-16 Tool-Backed Toric Embedding Sidecar Addendum

- Added `TropicalGT-I/src/tropicalgt/cas_toric.py` as the real-only finite toric embedding certificate boundary. It accepts explicit integer exponent matrices, validates coordinate variable names, builds a Macaulay2 script using `needsPackage "Quasidegrees"` and `toricIdeal(A,R)`, and returns a tagged certificate or an explicit unavailable state.
- Certified toric reports are scoped to the affine toric ideal/kernel of the finite monomial map determined by the columns of the exponent matrix. They are safe as finite monomial-map toric-ideal sidecars but not as normal-fan, tropical-variety embedding, sheaf, or global neural toric-variety certificates.
- Chart-bundle toric telemetry now records an explicit unavailable/uncertified certificate object. The live metrics `toric_normal_fan_loss`, `toric_active_row_count`, and `graphcg_toric_cell_agreement` are activation diagnostics only until a real `cas_toric` or `cas_tropical` certificate is attached.
- Live smoke on exponent matrix `[[1,1,1],[0,1,2]]` returned `ideal(z_1^2-z_0*z_2)` from Macaulay2 and kept `safe_to_use_as_normal_fan_certificate=false`.
- Verification completed: py-compile for touched source/tests, focused toric embedding tests (`3 passed`), focused chart-bundle tests (`3 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`27 passed`), and `pytest TropicalGT-I/tests/test_losses_and_model.py -q` (`10 passed`).
- b60 latest checked training state remained at step `2000` with train loss/NLL `1.139/1.116`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.

## 2026-06-16 One Dimensional Cone Terminology Addendum

- Visible paper prose now uses "one dimensional cone" / "one dimensional cones" for fan-theoretic cone-indexed filtrations, probes, incidence, and vector-bundle compatibility statements. Machine labels and literal backend APIs remain unchanged where appropriate.
- Older planning bullets in `planning/tropicalgt_i_visual_math_repair_plan_2026-06-12.md` now use one dimensional cone terminology instead of `matroid/ray`, `ray-only`, or `ray-filtration` shorthand.
- Verification completed: terminology audit passed, lightweight TeX environment-balance check passed, and remote `pdflatex` compile was attempted in `/tmp/tropicalgt_paper_compile` but stopped because `mathtools.sty` is missing from the remote TeX installation.
- b60 latest checked training state reached step `2005` with train loss/NLL `1.191/1.167`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.

## Real Implementations Only Policy

No TropicalGT-I metric, loss, visualization, analogical map, persistence module, free resolution, derived comparison, tropical-cycle diagnostic, or CAS artifact should be presented as a mathematical object unless it is computed from the actual model outputs, graph states, embeddings, probabilities, simplex trees, bifiltrations, or certified CAS/backend output that define that object. Temporary placeholders, synthetic fallback objects, mock charts, fabricated simplices, and convenience stand-ins are not acceptable. When a requested object cannot yet be computed, the artifact must render an explicit unavailable/uncertified state and the training metric must either be disabled or logged under an audit-only unavailable flag. Finite chain-presentation diagnostics may be shown only as chain diagnostics, never as free resolutions. Total-graded or ungraded CAS output may be shown as real CAS output only under its actual grading; it must not be advertised as a multigraded `F2[x_level,x_radius]` free resolution unless the backend certifies that multigraded structure.

Use "one dimensional cone" or "one dimensional cones" as the preferred fan-theoretic language whenever the intended object is a cone of a fan or a cone-indexed filtration datum. Use singular or plural according to ordinary grammar.


## 2026-06-16 Step-5000 Gate And Empty-Checkpoint Readiness Addendum

- b60 reached the required step-5000 gate. The stop record `TropicalGT-I/outputs/training_stop_records/b60_fresh_step0_casrows_step5000_gate.json` reports `latest_step=5000`, `fatal=false`, and no missing required step-5000 validation/audit paths.
- Real step-5000 metrics from `periodic/step_00005000/validation_report.json`: BPB/exact BPB `1.4304583543547733`, graph-BPB `20.122144813809587`, graph-conditioned BPB without side cost `1.2576335315794374`, NLL `0.9915181752294302`, and invalid graph rate `0.0`. The BPB target `< 1.12` is not met.
- The local browser endpoint `http://127.0.0.1:8991/` is forwarded to the remote step-5000 `got_audit` directory and returns the real `TropicalGT-I Inference Audit` HTML.
- The only discovered b60 checkpoint path `TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt` is zero bytes. `parameter_golf_codex_review_loop.py` now reports that checkpoint as unavailable instead of attempting to load it, and `prepare_5k_review_bundle.py --run-*` refuses command execution with `empty_checkpoint:TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt`.
- The path-only evidence bundle exists at `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/post_5k_review_bundle/review_bundle_step_00005000.json` with `execution_readiness.ready=false`, issue `empty_checkpoint:TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt`, `command_results=[]`, and `restart_policy=beginning`. It may be reviewed by a Codex subagent, but no restart should be launched from unsupported checkpoint-sidecar claims.


## 2026-06-16 Codex Step-5000 Evidence Review Result

- Codex subagent Lagrange reviewed the real b60 step-5000 bundle and wrote `planning/tropicalgt_i_b60_step5000_codex_evidence_review_2026-06-16.md`.
- The restart-schema result is `blocked_missing_evidence_no_restart`: BPB `1.4304583543547733` missed target `< 1.12`, but checkpoint-dependent executable analyses remain unavailable because `TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt` is zero bytes.
- No restart was launched and no hyperparameter/config patch was proposed. Under the no-proxy/no-fallback rule, the next source-side task is checkpoint-write hardening for future gates, not a BPB restart based on incomplete checkpoint-sidecar evidence.


## 2026-06-16 Atomic Checkpoint Write Hardening Result

- Future trainer checkpoint writes now use a temporary file, nonempty/loadable verification by default, fsync, atomic replace, parent-directory fsync, and post-replace verification. This prevents a failed save from replacing a prior checkpoint with an empty `.latest.pt`.
- The guard is source-side only and does not repair or mutate the already-empty b60 checkpoint; the b60 restart decision remains blocked on missing real checkpoint-dependent evidence.
- Focused verification passed: `pytest TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_training_metrics.py -q` (`16 passed`).


## 2026-06-16 Generated Artifact Cleanup Audit Result

- Removed only safe cache directories (`.pytest_cache` and Python `__pycache__` directories) after the atomic-checkpoint verification run. The git tree remained clean after cache deletion.
- Preserved b60 5K evidence, generated review bundles, stop records, launch configs, checkpoint paths, and old b55-b59 provenance. Large generated cleanup candidates were recorded in the sequential queue but not deleted without explicit approval.
- Disk remains tight, with about `32G` available on `/`, dominated by the b60 generated evidence directory and older generated run outputs.


## 2026-06-16 BEMultipliers Diagnostic Contract Result

- Macaulay2 BEMultipliers output now carries explicit non-resolution contract fields in generated CAS scripts and parsed certificate summaries. A computed `aMultiplier(1,C,ComputeRanks=>true)` matrix may render only after a certified Macaulay2 ChainComplex exists, and it is never a substitute for a free-resolution backend.
- Parsed Buchsbaum-Eisenbud diagnostics now expose `safe_to_render_multiplier_output`, `is_resolution_backend=false`, `safe_to_substitute_for_resolution=false`, and a no-proxy diagnostic contract.
- Focused verification passed: `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`28 passed`).


## 2026-06-16 BEMultipliers Browser Contract Rendering Result

- Browser-facing CAS payloads and tables now render BEMultipliers contract fields: multiplier output is post-resolution diagnostic-only, requires a certified Macaulay2 ChainComplex, is not a resolution backend, and cannot substitute for missing free-resolution evidence.
- Certified-CAS signatures used for derived/analogical comparison include these flags, so BEMultipliers status/matrix hashes cannot masquerade as resolution evidence.
- Focused verification passed: `pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_algebraic_persistence.py -q` (`64 passed`).


## 2026-06-16 Sequential Derived/CAS Update: No-Proxy Explanation Pass

Status: complete for the analogical comparison explanation checkpoint.

- Certified real free-resolution comparisons now carry machine-readable `component_explanations`, `diagnostic_only_components`, `mismatch_explanations`, and the policy tag `no_proxy_no_fallback_exact_cas_components_only`.
- Missing query or memory CAS evidence now reports per-side unavailable reasons and explicitly states that missing certified resolutions are unavailable, not estimated and not replaced by fallback data.
- Buchsbaum-Eisenbud/BEMultipliers entries are explicitly diagnostic-only sidecars in the analogical/derived comparison payload. They are never a resolution backend and cannot substitute for matching Betti shifts, differentials, Fitting ideals, minors, or a certified chain-map/isomorphism.
- Browser-visible derived/free-resolution interpretation text now explains both exact matches and mismatches under the same no-proxy/no-fallback policy.

Verification:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_simplicial_visualization.py
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py -q
# 36 passed in 3.07s
git diff --check
# clean
```

## 2026-06-16 Sequential Training/Ablation Update: Advanced Auxiliary Promotion Gate

Status: complete for source-side ablation gating; incomplete for actual runtime promotion because no matched chart-bundle/toric BPB ablation reports have been run under the no-proxy evidence policy.

- `run_bpb_ablation_grid.py` now includes chart-bundle/toric variants for telemetry-only zero-weight checks and a small nonzero `chart_bundle_toric_0p1x` candidate.
- Training reports now persist `ablation_variant` and `ablation_overrides`, allowing downstream analysis to identify exact coefficient changes from real runs.
- `build_bpb_ablation_report` now emits `advanced_auxiliary_promotion_gate` with policy `no_proxy_no_fallback_matched_seed_eval_bpb_and_eval_graph_bpb_required_before_promoting_advanced_auxiliary_coefficients`.
- The gate requires matched seed, matched final step, finite held-out eval BPB and eval graph-BPB deltas, and improvement on both held-out eval BPB and eval graph-BPB before any nonzero chart-bundle/toric coefficient is marked promotable.
- Telemetry-only rows, unrelated variants, missing target deltas, unmatched seeds/steps, and dataset-manifest mismatches remain blocked/unavailable rather than treated as evidence.

Verification:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/ablation.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/scripts/run_bpb_ablation_grid.py TropicalGT-I/tests/test_bpb_ablation.py TropicalGT-I/tests/test_bpb_ablation_grid.py
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_bpb_ablation.py TropicalGT-I/tests/test_bpb_ablation_grid.py TropicalGT-I/tests/test_training_resume.py -q
# 6 passed in 2.89s
git diff --check
# clean
```

## 2026-06-16 Sequential CAS Update: Grade/Depth Diagnostic Certificate Surface

Status: complete for CAS-emitted grade/depth diagnostics; independent regular-element certification remains unavailable unless emitted by the backend.

- Macaulay2 script generation now emits `grade_depth_regular_diagnostics_begin/end` around rank-ideal diagnostics for certified resolution differentials.
- Parsed diagnostics include ambient ring dimension, differential rank, rank ideal, rank-ideal codimension, rank-ideal depth, and grade lower-bound status where Macaulay2 emits those values.
- Certificate summaries now expose `grade_depth_regular_diagnostics_available` and `regular_element_certificate_available`.
- Visualization payloads and Macaulay2-style tables now render grade/depth diagnostics and a regular-element note. These diagnostics are explicit CAS output and never replace the free-resolution certificate or a regular-sequence certificate.
- Backends or certificates without the block remain explicitly unavailable with reason text; no rank-only inference is used.

Verification:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_simplicial_visualization.py
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_simplicial_visualization.py -q
# 64 passed in 5.68s
git diff --check
# clean
```

## 2026-06-16 Sequential Training/Readiness Update: Advanced BPB Contract Gate

Status: complete for source-side BPB contract readiness enforcement; incomplete for an actual restart because b60 still lacks a nonempty loadable step-5000 checkpoint.

- `audit_tropicalgt_i_readiness.py` now emits an `advanced_bpb_contract` section for BPB-focused step-0/5K-gate configs. The section is config-only and intentionally strict: missing advanced-method switches fail rather than being inferred from model defaults.
- The contract gates TokenGT graph tokens, graph autoregressive decoding, required Parameter-Golf and HF reasoning data sources, real-data requirement, long-context/multi-record shape, positive GFlowNet/GraphCG/certificate/sequence-tropical weights, full-rank GraphCG direction bank, memory probability/topological quality thresholds, many top-k retrieval, graph-BPB side weighting, 250-step validation/visual cadence, periodic interactive/browser artifacts, meet-in-the-middle forward+reverse and ROAR/random-order settings, and online W&B project/run-name configuration.
- Fixture/smoke readiness configs are reported as `advanced_bpb_contract.required=false`, so local non-BPB tests remain small and are not silently promoted into BPB training evidence.
- Tests now assert the current b54 5K gate config passes the full advanced BPB contract, and that disabled GraphCG, weak/non-explicit HF reasoning source names, missing explicit W&B run-name fields, meet-in-middle, memory-quality, periodic-artifact, and cadence settings are blocked with named failed gates.
- No W&B entity/organization value was guessed or hardcoded; the report surfaces the configured project/run-name and whether an explicit entity is present. The W&B gate requires an explicit `wandb_run_name` or `wandb_name`, while a stricter entity gate should be added only when the intended non-secret W&B entity is known.
- No training restart was launched. Under the no-proxy/no-fallback policy, the b60 restart remains blocked by the zero-byte checkpoint; this pass hardens the next preflight rather than manufacturing missing checkpoint-sidecar evidence.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py TropicalGT-I/tests/test_readiness_audit.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_readiness_audit.py -q
# 7 passed in 1.50s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 35 passed, 2 warnings in 1.53s
```


## 2026-06-16 Sequential Training/Readiness Update: W&B Run-Name Alignment Gate

Status: complete for source/config readiness hardening.

- BPB-focused advanced readiness now requires the explicit W&B run-name field (`wandb_run_name` or `wandb_name`) to match the config `run_name`, preventing stale W&B names from passing preflight.
- Fixed `train_full_dataset_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate.json` so `wandb_name` and `wandb_run_name` both match `tropicalgt_i_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate`.
- Added regression coverage proving mismatched W&B run names fail with `advanced_bpb_wandb_run_name_matches_config`.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py TropicalGT-I/tests/test_readiness_audit.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 36 passed, 2 warnings in 1.56s
```


## 2026-06-16 Sequential Training/Readiness Update: Trainer BPB Contract Enforcement

Status: complete for source-side launch hardening; actual BPB restart remains blocked by the missing loadable b60 checkpoint evidence.

- Moved the advanced BPB config contract into source module `tropicalgt.readiness_contracts`, so readiness audits and training launches share the same no-proxy/no-fallback gates.
- `train()` now calls `enforce_advanced_bpb_contract(cfg)` immediately after config load. BPB-focused configs with missing or disabled advanced-method requirements fail before dataset loading, model construction, W&B initialization, checkpoint writes, or generated report output.
- Successful non-BPB training reports now persist `advanced_bpb_contract` and `advanced_bpb_contract_gates`, making fixture/smoke runs explicitly report that the BPB contract was not required.
- Added regression coverage proving an under-specified BPB 5K config raises `Advanced BPB training contract failed` and does not write `train_report.json`.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/readiness_contracts.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_readiness_audit.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 40 passed, 2 warnings in 1.77s
```


## 2026-06-16 Sequential Training/Readiness Update: W&B Organization Gate

Status: complete for source/config readiness hardening.

- BPB-focused advanced readiness now requires an explicit non-secret W&B entity/organization in `wandb.entity`, closing the remaining gap in the original W&B organization requirement.
- Tracked b54 and b55 5K-gate configs now set `wandb.entity` to the observed project organization `amelie-iska-math`, matching the real historical W&B URLs under `https://wandb.ai/amelie-iska-math/TropicalGT-I/...`.
- Added regression coverage proving missing W&B entity fails with `advanced_bpb_wandb_entity_configured`.
- Direct source-contract probes confirm b54 and b55 still pass all advanced BPB gates after adding the entity field.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/readiness_contracts.py TropicalGT-I/tests/test_readiness_audit.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_resume.py -q
# 12 passed in 1.49s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 40 passed, 2 warnings in 1.66s
```

## 2026-06-16 Sequential Training/Readiness Update: BPB Wrapper And Preflight Bypass Hardening

Status: complete for the wrapper/preflight bypass-hardening checkpoint; actual BPB restart remains blocked by the zero-byte b60 checkpoint.

- `parameter_golf_codex_review_loop.py` now enforces the shared advanced BPB contract before launching any training subprocess. The default train script was already safe through `train()`, and this closes the user-supplied `--train-script` wrapper bypass.
- `audit_tropicalgt_i_readiness.py --train-dry-run` and CUDA-required preflight now short-circuit before dataset loading or model/optimizer construction when a BPB-focused config fails the advanced contract. The report stays truthful and blocked, with `advanced_bpb_contract_failed_before_train_preflight` rather than synthetic dry-run evidence.
- `run_bpb_ablation_grid.py` now audits each generated variant in memory before writing configs. BPB-focused variants that fail the contract are blocked before files are written unless `--allow-contract-breaking-ablation-configs` is explicitly passed for analysis-only config emission, and such manifest rows carry `contract_safe_to_run=false` plus named failed gates.
- Strengthened trainer regression coverage to assert data, model, W&B, and checkpoint helpers are not called after a failed advanced BPB contract.
- No generated audit bundles, datasets, checkpoints, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_bpb_ablation_grid.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_resume.py -q
# 24 passed in 5.35s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_bpb_ablation_grid.py TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 52 passed, 2 warnings in 5.53s
```

## 2026-06-16 Sequential Training/Readiness Update: 5K Bundle Advanced Contract Evidence

Status: complete for path-only post-5K review-bundle contract visibility; actual BPB restart remains blocked by the zero-byte b60 checkpoint.

- `prepare_5k_review_bundle.py` now records the shared config-only advanced BPB contract in every review bundle under `advanced_bpb_contract` and writes `advanced_bpb_contract_step_<step>.json` beside the active training contract and Codex review prompt.
- The bundle markdown now includes an `Advanced BPB Contract` section, so reviewers and subagents can see whether the config is safe to use for a step-0 BPB restart before proposing hyperparameter changes.
- Failed contract gates are preserved by name, and `safe_to_use_for_step0_bpb_restart=false` is emitted when any required advanced BPB gate fails. This is evidence only; it does not fabricate checkpoint, CAS, topology, geometry, algebra, W&B, or visualization evidence.
- No generated review bundles, checkpoints, datasets, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py -q
# 7 passed in 1.05s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_training_step_gate_monitor.py TropicalGT-I/tests/test_readiness_audit.py -q
# 33 passed in 1.60s
```

## 2026-06-16 Sequential Training/Readiness Update: 5K Restart Evidence Gate

Status: complete for source-side post-5K restart permission hardening; actual BPB restart remains blocked by the zero-byte b60 checkpoint.

- `prepare_5k_review_bundle.py` now emits `restart_evidence_gate` in every review bundle, separating a missed BPB target from permission to restart.
- The gate returns `blocked_missing_required_evidence_no_restart` and `step0_restart_allowed=false` when the checkpoint is missing, empty, unloadable, execution evidence is not ready, the primary BPB metric is missing, post-5K command results are absent or failed, or the advanced BPB contract has failed gates.
- Bundle markdown now renders a `Restart Evidence Gate` section so Codex/subagent reviewers see the no-proxy block before proposing step-0 hyperparameter/config changes.
- Regression tests cover missing checkpoint, empty checkpoint without command execution, loadable checkpoint plus missing post-5K command results, and failed advanced BPB contract cases. The existing command-execution path still raises before running eval/backfill/validators when checkpoint evidence is missing or empty.
- A real b60 path-only probe written to `/tmp/tropicalgt_b60_restart_gate_probe` records BPB `1.4304583543547733`, graph-BPB `20.122144813809587`, `restart_action=blocked_missing_required_evidence_no_restart`, `step0_restart_allowed=false`, the empty checkpoint blocker, failed advanced BPB gates for the stale b60 launch config, and missing eval/backfill/validator command-result blockers.
- No generated review bundles, checkpoints, datasets, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py -q
# 9 passed in 0.99s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_training_step_gate_monitor.py TropicalGT-I/tests/test_readiness_audit.py -q
# 35 passed in 1.43s
git diff --check
# clean
```

## 2026-06-16 Sequential Training/Readiness Update: Review Loop Same-Config Restart Halt

Status: complete for loop-level no-proxy restart hardening; actual BPB restart remains blocked by the zero-byte b60 checkpoint and missing post-5K command evidence.

- `parameter_golf_codex_review_loop.py` now halts by default after a triggered BPB review boundary instead of automatically scheduling another training run with the same config.
- The loop records and prints `triggered_restart_block.v1` with `restart_action=blocked_pending_evidence_backed_config_patch`, forcing post-5K review evidence and a reviewed config patch before another step-0 launch.
- Legacy same-config continuation is still available only through the explicit `--allow-same-config-restart-after-triggered-review` opt-in flag, making that behavior visible rather than accidental.
- The restart decision schema now lists `reviewed_config_patch_before_any_same_config_restart` as required evidence and includes `blocked_pending_evidence_backed_config_patch` as an allowed blocked action.
- A no-training dry-run probe written to `/tmp/tropicalgt_review_loop_halt_probe` with tracked b55 config prints the restart block directly and records it in `review_loop_state.json`; no training process was launched.
- No generated review-loop state, checkpoints, datasets, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/parameter_golf_codex_review_loop.py TropicalGT-I/tests/test_parameter_golf_review_loop.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_parameter_golf_review_loop.py -q
# 11 passed in 1.01s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/parameter_golf_codex_review_loop.py --config TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b55_v11_bpb_5k_gate.json --dry-run --max-reviews 1 --max-total-steps 5000 --output-dir /tmp/tropicalgt_review_loop_halt_probe
# printed restart_block.restart_action=blocked_pending_evidence_backed_config_patch
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_training_step_gate_monitor.py TropicalGT-I/tests/test_readiness_audit.py -q
# 37 passed in 1.53s
git diff --check
# clean
```

## 2026-06-16 Sequential Training/Readiness Update: Explicit Checkpoint Load Evidence

Status: complete for shared checkpoint-load evidence hardening; actual BPB restart remains blocked by the zero-byte b60 checkpoint.

- `load_checkpoint()` now checks file size and payload schema before building a model, and raises explicit `checkpoint_file_empty`, `checkpoint_load_failed`, or `checkpoint_invalid_payload` errors instead of leaking raw `torch.load` failures to inference/eval/readiness callers.
- `_verify_checkpoint_file()` now supports `expected_step=None` for load-time structural validation while preserving exact-step verification for checkpoint writes.
- Regression tests cover empty checkpoint load and malformed payload load, alongside the existing atomic-save test that prevents empty temporary files from replacing an existing target.
- A real b60 load probe now reports `checkpoint_file_empty:TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt`, matching the no-proxy restart blocker.
- No checkpoints, generated bundles, datasets, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py -q
# 6 passed in 1.49s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 43 passed, 2 warnings in 1.68s
git diff --check
# clean
```

## 2026-06-16 Sequential Training/Readiness Update: Boundary Checkpoint Snapshot Evidence

Status: complete for review-loop checkpoint snapshot hardening; actual BPB restart remains blocked by the zero-byte b60 checkpoint.

- `parameter_golf_codex_review_loop.py` now records `checkpoint_snapshot_status` for each review event instead of copying any existing checkpoint path unconditionally.
- Boundary checkpoint snapshots require a nonempty, torch-loadable full training checkpoint payload with `model`, `config`, and `step`; missing, empty, unloadable, or malformed checkpoints remain explicit unavailable evidence.
- `_checkpoint_step()` now reads from the validated checkpoint summary and returns `0` for unavailable checkpoints instead of raw-loading arbitrary files.
- If a target is met or a legacy latest-checkpoint restart path would need an unavailable checkpoint, the loop records `checkpoint_snapshot_block.v1` with `blocked_missing_loadable_boundary_checkpoint` rather than reusing a bad file.
- Regression tests cover empty checkpoint snapshot refusal, malformed checkpoint summaries, valid snapshot copying, checkpoint-step handling for invalid files, and checkpoint snapshot block content.
- A real b60 snapshot probe written only under `/tmp/tropicalgt_b60_snapshot_probe` reports `available=false`, `unavailable_reason=checkpoint_file_is_empty`, and creates no snapshot file.
- No generated snapshots, checkpoints, review-loop state, datasets, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/parameter_golf_codex_review_loop.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_parameter_golf_review_loop.py -q
# 16 passed in 0.95s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_training_step_gate_monitor.py TropicalGT-I/tests/test_readiness_audit.py -q
# 42 passed in 1.45s
git diff --check
# clean
```

## 2026-06-16 Sequential Training/Readiness Update: Training Report Checkpoint Integrity Metadata

Status: complete for source-side checkpoint evidence metadata in future training reports; actual BPB restart remains blocked by the existing zero-byte b60 checkpoint.

- `_save_training_checkpoint()` now returns a checkpoint integrity record after atomic replace and load verification, including path, availability, size, expected step, observed step, and whether load verification was enabled.
- Training reports now include `checkpoint_integrity` for the final checkpoint and `latest_checkpoint_integrity` for the latest periodic checkpoint path when present, so post-5K review can inspect checkpoint evidence directly from the report.
- `_checkpoint_integrity_report()` records explicit unavailable reasons instead of letting missing or invalid checkpoint paths masquerade as usable evidence.
- Regression tests cover report integrity metadata on a real tiny training run and unavailable integrity reporting for missing checkpoint paths.
- No checkpoints, generated reports, datasets, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_training_resume.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py -q
# 7 passed in 1.54s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 44 passed, 2 warnings in 1.63s
git diff --check
# clean
```

## 2026-06-16 Sequential Training/Readiness Update: Final Eval Metrics In Checkpoints

Status: complete for future final/latest checkpoint metric evidence; actual BPB restart remains blocked by the existing zero-byte b60 checkpoint.

- Final and latest checkpoints are now written after final validation evaluation, so their payload metrics include final `eval_bpb`, `eval_graph_bpb`, and other numeric eval metrics rather than only pre-eval training metrics.
- The final history row is updated with final eval metrics before checkpoint save, keeping report history and checkpoint history aligned for post-5K review.
- Regression tests now load both the final and latest checkpoint from a tiny training run and assert their payload metrics match the final report eval BPB and graph-BPB values.
- No checkpoints, generated reports, datasets, W&B folders, caches, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/tests/test_training_resume.py
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py -q
# 7 passed in 1.53s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_readiness_audit.py TropicalGT-I/tests/test_training_metrics.py TropicalGT-I/tests/test_data_loader.py -q
# 44 passed, 2 warnings in 1.73s
git diff --check
# clean
```

_Last updated: 2026-06-16T15:31:09Z_


## 2026-06-16 Always-On b61 Training And Checkpoint-Evidence Contract Update

- Active training is restored under the new always-on rule. Fresh step-0 b61 run `tropicalgt_i_pg_bpb_step0_full24b_b61_20260616T155103Z_fresh_bpb112_alwayson_5k_gate` is running on trainer PID `927496`, with 5K monitor PID `931644`, W&B run id `mhxrmoxl`, and log `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b61_20260616T155103Z_fresh_bpb112_alwayson_5k_gate/logs/train_nohup_20260616T155425Z.log`.
- b61 was launched from step 0 because b60 checkpoint evidence is unavailable: the b60 latest checkpoint is an empty file. The b61 config cites real b60 validation/audit evidence and records that no checkpoint-derived evidence was used.
- A dedicated subagent, Herschel (`019ed124-c3b0-7b00-81a9-1a39c76e5583`), owns only the 5K training-iteration loop. At each 5K boundary it must inspect all metrics, advanced readouts, algebraic/topological/geometric sidecars, newly implemented toric/tropical/vector-bundle/sheaf/derived/PH/memory diagnostics, and then restart from step 0 with evidence-backed changes.
- Cleanup completed for old `/tmp/tropicalgt*` scratch files, repo `__pycache__` directories, and `.pytest_cache`; provenance/state directories were preserved.
- Source-side hardening added a `tropicalgt.checkpoint_evidence.v1` block to active training contracts and markdown. Future reviewers now see checkpoint availability, integrity, step/metric mismatch warnings, and whether a checkpoint-backed restart is safe. The 5K monitor now treats zero-byte required files as unavailable, so empty checkpoints cannot satisfy required-path gates.
- Remaining immediate updates: keep b61 alive to 5K, continue source-side preflight hardening, and then proceed through CAS, analogical-memory, simplex-tree, bifiltration, NLL, tropical-support, GraphCG, toric/vector-bundle, paper, docs, and browser-QA items sequentially.

## 2026-06-16 Sequential Training/Readiness Update: Review Bundle Checkpoint Evidence Propagation

Status: complete for post-5K restart bundle checkpoint-evidence propagation; active b61 training remains running under the always-on rule.

- `prepare_5k_review_bundle.py` now passes the resolved checkpoint path into the active training contract and copies the resulting `tropicalgt.checkpoint_evidence.v1` block into the generated review bundle.
- `restart_evidence_gate` now records `checkpoint_evidence_safe` and `checkpoint_evidence_warnings`; missed BPB targets are blocked when checkpoint evidence is unavailable, stale, empty, mismatched, or otherwise unsafe for checkpoint-backed restart.
- Bundle markdown now includes a dedicated `Checkpoint Evidence` section so the 5K training subagent can review checkpoint proof beside execution readiness, advanced BPB contract gates, and command results.
- Regression tests now cover missing checkpoint, empty checkpoint, and valid loadable checkpoint evidence in the bundle/gate path.
- No datasets, checkpoints, outputs, caches, W&B folders, generated review bundles, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py -q
# 9 passed in 2.44s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_training_step_gate_monitor.py TropicalGT-I/tests/test_readiness_audit.py -q
# 45 passed in 1.73s
```

_Last updated: 2026-06-16T16:19:00Z_

## 2026-06-16 Sequential Training/Readiness Update: Data Path Fallback Resolver Removal

Status: complete for source-side dataset path fallback hardening; active b61 training remains running under the always-on rule.

- Removed the dead alternate-candidate `fallbacks` parameter from `_resolve_existing_config_path`; dataset source resolution now accepts exactly one explicit primary path and fails if it is absent or missing.
- Renamed the `dataset_manifest` optional root argument from `fallback_root` to `data_root`, eliminating misleading fallback terminology for the primary dataset root.
- Replaced `_strict_config_path_candidates` with `_reject_config_path_fallbacks`, which explicitly rejects legacy `fallback_roots` and `tokenizer_fallback_paths` before any path resolution occurs.
- Updated provenance registry terms so the audit tracks the explicit rejection guard rather than obsolete fallback-candidate plumbing.
- Corrected stale planning text that implied an opt-in path fallback resolver was still allowed.
- No datasets, checkpoints, generated outputs, caches, W&B folders, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_data_loader.py TropicalGT-I/tests/test_metric_provenance.py TropicalGT-I/tests/test_readiness_audit.py -q
# 31 passed, 2 warnings in 4.05s
```

_Last updated: 2026-06-16T16:39:00Z_

## 2026-06-16 Sequential Visual/Math Update: Bivariate Staircase Sidecar Contract

Status: complete for the two-parameter bifiltration sidecar contract and validator hardening; active b61 training remains running under the always-on rule.

- `write_two_parameter_bifiltration_visualization` now emits explicit per-staircase JSON contracts for actual generator labels, minimal antichains, upward-closed generated regions, displayed quotient-basis lattice points, Hilbert numerator terms, adjacent LCM syzygies, and theorem scope.
- Primary staircase selection now chooses the first nonprincipal Miller-Sturmfels staircase when available, instead of leaving every card secondary when a principal card sorts first.
- The interactive audit validator now requires these staircase-card contracts and checks that quotient-basis counts match the listed lattice points and that theorem scope preserves the no-proxy boundary against full persistence-module free-resolution claims.
- Tests pin the rendered JSON sidecar, the validator fixture, and the scoped two-variable monomial-ideal resolution behavior.
- No generated audit bundles, browser artifacts, datasets, checkpoints, caches, W&B folders, or secrets were staged.

Verification:

```bash
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py::test_level_radius_bifiltration_reports_scoped_real_staircase_resolution TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 11 passed in 1.94s
PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_backfill_interactive_audit_artifacts.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 39 passed in 6.36s
```

_Last updated: 2026-06-16T16:55:00Z_

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
