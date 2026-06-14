# TropicalGT-I Current Required Updates and Repair List

This checklist merges the browser/photo review, the current active training state, the CAS/free-resolution policy, and the new tropical vector-bundle / tropical toric-embedding paper workstream. All work is remote-only on `iska@iska` in `/home/iska/Documents/amelie/bio/TropicalGT`, branch `tropicalgt-i-implementation`.

## 1. Active Training and BPB Priority

- Keep `tropicalgt_i_pg_bpb_step0_full24b_b54_v10_bpb_5k_gate` alive until at least step 5K unless it crashes or produces nonfinite/invalid losses. Current tracked PID at launch: `2154851`; W&B run id at launch: `nw0bo45u`; run URL: `https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/nw0bo45u`. The run is configured for >10B token slots, batch 54, BPB target `<1.12` by 5K, meet-in-the-middle decoding, causal forward/reverse decoding for DAGs, ROAR for cyclic/noncausal graphs, and low-weight BPB-gated advanced objectives.
- Do not restart before 5K merely because early metrics are noisy. Review at 5K, then restart/resume only if BPB and diagnostics justify a new hyperparameter setting.
- Preserve BPB and graph-BPB as primary optimization and promotion gates.
- Use advanced auxiliaries only when they are zero-default or BPB-gated and ablated: tropical support, GFlowNet GoT rewards, GraphCG, persistence/landscape diagnostics, chart-bundle transports, tropical toric active-cell agreement, and memory retrieval.
- Track step, VRAM, wall time, train/eval NLL, BPB, graph-BPB, certificate loss, tropical wall-hit rate, support entropy, GraphCG rank, meet-in-the-middle agreement, ROAR/causal decoding path mix, and artifact-generation status.
- After this v10 restart, the main implementation focus is CAS integration and utilization only until real free-resolution and derived-map reporting are wired, tested, and rendered.

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
- Push only to non-main branch `tropicalgt-i-implementation`, excluding secrets, datasets, W&B runs, caches, checkpoints, and bulky artifacts.

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
