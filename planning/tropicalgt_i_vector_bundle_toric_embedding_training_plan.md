# TropicalGT-I Vector-Bundle and Toric-Embedding Training Plan

**Terminology: one dimension cone.** In the fan-theoretic language used by the reviewed papers, the relevant one-dimensional cones are called one dimension cones in this plan. The implementation vocabulary is therefore one dimension cone-indexed filtrations, one dimension cone pairings, and one dimension cone probes.

## Summary of the Two Papers

- Khan and Maclagan, `2405.03505v1.pdf`, define tropical toric vector bundles by a simple valuated matroid, a ground set of atoms, and one dimension cone-indexed decreasing filtrations by flats. The vector-bundle condition says that on each maximal cone the filtrations are generated from a matroid basis by thresholding one dimension cone pairings and taking joins of selected flats. They also describe Cox semimodule presentations, fibers as tropical linear spaces, realizability limits, direct sums, tensoring by line bundles, global sections via parliaments of polytopes, and stability/Harder-Narasimhan-style structures under extra hypotheses.
- Jun, Mincheva, and Tolliver, `2009.03030v2.pdf`, develop vector bundles on semiring and tropical schemes as locally free sheaves. For zero-sum-free semirings with only trivial idempotent pairs, bases of finite free modules are unique up to rescaling and permutation; invertible matrices are therefore monomial. They classify vector bundles by nonabelian `H^1(X, GL_n(O_X))`, prove splitting results in locally constrained irreducible cases, compare with topological and monoid-scheme vector bundles, and study lifting of line bundles.

## What Transfers to TropicalGT-I

- Monomial chart transports transfer as a mathematically supported local form for invertible tropical-linear changes of hidden feature coordinates.
- Valuated-matroid atoms and flat filtrations transfer as a finite compatibility language for chart-local feature supports, GraphCG directions, persistence bins, and memory atoms.
- Toric one dimension cone filtrations transfer as a design pattern for small integer max-linear probes over TokenGT graph tokens and tropical ring attention supports.
- Fibers as tropical linear spaces transfer as an interpretation of chart-local feature packets, not as a claim that the neural hidden space is an algebraic vector bundle.
- Stability and filtration language transfer as diagnostics for whether feature atoms split into independent pieces or concentrate in unstable subfamilies.

## What Does Not Transfer Directly

- A TokenGT batch is not literally a tropical toric variety, and graph-of-thought state charts are not maximal cones of a fan unless an explicit fan construction is supplied.
- A neural hidden state is not automatically a locally free sheaf over a tropical scheme; the bundle language is an imposed regularization and audit structure.
- The Khan-Maclagan compatibility theorem does not prove BPB improvement. It only justifies the form of the matroid-filtration regularizer.
- The Jun-Mincheva-Tolliver splitting theorem warns that some semiring-scheme vector bundle notions split into line bundles; it does not by itself produce expressive neural feature bundles.
- Persistence landscapes, NLL fields, and GoT fitness/density fields are distinct objects. Landscape-vector losses apply only to real GUDHI-produced `lambda_k(t)` vectors.

## Goal

Use the theory of tropical vector bundles and vector bundles on tropical schemes to add trainable chart transports, toric embedding regularizers, and BPB-oriented chart consistency objectives to TropicalGT-I. The guiding rule is that invertible tropical chart changes should be monomial transports, while chart filtrations should be checked against small valuated-matroid flat systems.

## Model Hooks

- Add optional `return_charts=True` outputs from the graph encoder: chart ids, node membership, chart feature tables, overlap pairs, and overlap triples.
- Add a `MonomialTransportHead` that predicts a permutation and tropical shifts for each overlap pair.
- Add a small `BundleMatroidHead` with feature atoms, basis candidates, rank-defect scoring, and flat-incidence masks.
- Add a `ToricEmbeddingHead` with a small integer exponent/probe matrix and max-linear scores.
- Keep GUDHI persistence-landscape vectors optional. Missing vectors should remain masked/unavailable, not zero-filled or fabricated.

## Losses

- `bundle/transport_l1`: L1 difference between transported chart features and target chart features on overlaps.
- `bundle/cocycle_defect`: discrepancy between `T_bc T_ab` and `T_ac` on sampled triple overlaps.
- `bundle/flat_rank_defect`: rank or incidence defect between predicted filtrations and basis-generated flats.
- `toric/normal_fan_loss`: disagreement between active toric embedding rows and chart filtration cells.
- `chart/bpb_consistency`: robust NLL difference across transported overlapping charts.
- `bundle/atom_stability_gap`: sampled flat/atom load imbalance diagnostic inspired by tropical vector-bundle stability; use only as a BPB-gated auxiliary.
- `memory/transported_landscape_l2` and `memory/transported_landscape_cosine`: analogical retrieval comparisons only when real GUDHI landscape vectors are present.


## Concrete Training Objectives and Regularizers

- `bundle/transport_l1`: feature agreement under monomial transports on sampled TokenGT chart overlaps.
- `bundle/cocycle_defect`: consistency of `T_ab`, `T_bc`, and `T_ac` on graph-token, GoT-prefix, or memory-overlap triples.
- `bundle/flat_rank_defect`: matroid flat reconstruction error for active tropical attention supports, GraphCG one dimension cone probes, and persistence atoms.
- `toric/normal_fan_loss`: agreement between active max-linear toric rows and chart filtration cells.
- `graphcg/toric_cell_agreement`: agreement between active GraphCG direction cones and toric active cells, with the existing full-rank barrier retained.
- `gfn/bundle_reward`: optional GFlowNet reward shaping by transport, cocycle, and flat defects; promotion still depends on BPB and graph-BPB gates.
- `memory/transported_landscape_l2` and `memory/transported_landscape_cosine`: analogical retrieval comparisons only when both memory rows have real GUDHI persistence-landscape vectors.
- `chart/bpb_consistency`: robust NLL difference between overlapping charts after transport.
- `bundle/atom_stability_gap`: optional sampled flat-load penalty to discourage collapse of all useful atoms into one overloaded flat.

## Training Schedule

1. Train the baseline byte and graph BPB objectives with all new coefficients set to zero.
2. Enable soft monomial transports at a very small coefficient after BPB stabilizes.
3. Add chart BPB consistency only if validation BPB and graph BPB do not regress.
4. Add matroid flat compatibility on tiny ground sets and audit basis coverage.
5. Add toric embedding regularization after chart transports are stable.
6. Project soft permutations to hard monomial maps for periodic audits, not necessarily every optimizer step.

## Metrics and W&B Keys

- `train/bpb`, `val/bpb`, `train/graph_bpb`, `val/graph_bpb`
- `bundle/transport_l1`, `bundle/cocycle_defect`, `bundle/flat_rank_defect`, `bundle/basis_coverage`
- `toric/normal_fan_loss`, `toric/active_row_entropy`, `toric/tropical_margin`
- `chart/bpb_consistency`, `chart/overlap_count`, `chart/transport_projection_error`
- `memory/landscape_available_rate`, `memory/transported_landscape_l2`, `memory/transported_landscape_cosine`
- `audit/browser_landscape_label_ok`, `audit/topk_landscape_columns_present`


## Ablations, Telemetry, and Promotion Gates

Ablations should use matched seeds and identical data windows whenever possible.

| Variant | Active coefficients | Required telemetry | Promotion rule |
|---|---|---|---|
| zero auxiliary | all new coefficients `0` | baseline BPB, graph-BPB, certificate loss, wall-hit rate | reference run |
| telemetry only | coefficients `0`, artifact emission on | chart ids, toric rows, flat defects, landscape availability | logits and BPB must match zero auxiliary within tolerance |
| transport only | `lambda_transport`, `lambda_cocycle` | `bundle/transport_l1`, `bundle/cocycle_defect`, `chart/overlap_count` | no BPB/graph-BPB regression |
| matroid/one dimension cone only | `lambda_flat`, optional atom stability | `bundle/flat_rank_defect`, `bundle/basis_coverage`, `bundle/atom_stability_gap` | no BPB/graph-BPB regression and no certificate-loss spike |
| toric/GraphCG only | `lambda_toric`, `lambda_graphcg_toric` | `toric/normal_fan_loss`, `graphcg/toric_cell_agreement`, full-rank spectra | no graph-BPB regression and GraphCG full-rank remains finite |
| memory landscape only | `lambda_landscape` | landscape availability, transported L2/cosine, top-k columns | unavailable vectors remain unavailable; retrieval quality gate passes |
| chart BPB consistency | `lambda_chart_bpb` | chart-local NLL gap, overlap count, validation BPB | text BPB improves or is unchanged |
| full bundle stack | all selected coefficients | all above plus artifact-byte manifest | promote only if BPB, graph-BPB, certificate loss, and wall-hit gates pass |

W&B keys should be namespaced as `bundle/*`, `toric/*`, `graphcg/*`, `chart/*`, `memory/*`, and `audit/*`. Inference artifacts must be read-only: enabling artifact emission should not alter logits, sampled actions, retrieval gates, or greedy decoding.


## Paper Coverage Verification

- Background theory: covered in `tropicalgt_neurips_research_paper.tex` under `Reference-theory background`, using Khan-Maclagan valuated-matroid/one dimension cone-filtration tropical toric bundles and Jun-Mincheva-Tolliver semiring-scheme monomial transition results.
- Definitions: covered by `TropicalGT chart bundle`, `Tropical toric embedding of a reasoning state`, `One dimension cone-filtration atomization for graph tokens`, and `TokenGT tropical atlas`.
- Theorem/proposition material: covered by monomial transport, matroid-filtration compatibility, BPB-oriented chart consistency, finite one dimension cone-filtration auditability, and metric-gated auxiliary geometry.
- Examples: covered by the chart-consistency batch and transported persistence-landscape memory examples.
- Pseudocode: covered by `vector_bundle_toric_step` and `promote_bundle_geometry`.
- BPB/Parameter-Golf compatibility: covered by metric-facing implementation contract, Parameter-Golf accounting for bundle geometry, and the ablation/promotion matrix below.
- Implementation readiness: remaining work is code-level only: chart metadata, transport heads, toric/one dimension cone heads, memory masks, telemetry, browser labels, and zero-coefficient no-op tests.

## Browser and Visualization Artifacts

- Label GUDHI persistence landscapes explicitly as `persistence landscape lambda_k(t)`.
- Label GoT loss, fitness, heat, or density fields as NLL, fitness, or density, never as persistence landscapes.
- Add top-k analogical retrieval columns for landscape availability, transported L2, transported cosine, source chart, target chart, and transport id.
- Add chart-overlap audit panels showing transport L1, cocycle defect, flat rank defect, and BPB consistency.
- Add toric embedding panels showing active exponent rows and normal-fan cell assignments.

## Tests

- Unit test that missing persistence-landscape vectors remain unavailable and are not replaced by zeros.
- Unit test monomial transport projection: one active entry per row and column, with shifts preserved.
- Unit test cocycle defect is zero for consistent synthetic transports.
- Unit test flat rank-defect scoring on a small uniform matroid and one nontrivial flat family.
- Unit test browser labels distinguish GUDHI `lambda_k(t)` landscapes from GoT NLL/fitness/density fields.
- Integration smoke test that enabling all new losses with zero coefficients leaves baseline outputs unchanged.


## Implementation Checklist

- Likely model files: add optional chart ids, overlap metadata, toric active rows, and chart-local NLL outputs in `TropicalGT-I/src/tropicalgt/model.py` or the current model module that owns graph-token/tropical attention outputs.
- Likely training files: add zero-default coefficients, metric logging, and promotion gates in `TropicalGT-I/src/tropicalgt/run.py` and any config schemas that own loss weights.
- Likely memory files: add landscape availability masks and transported L2/cosine comparisons in `TropicalGT-I/src/tropicalgt/memory.py` without fabricating zero vectors.
- Likely inference files: add read-only optional artifact emission in `TropicalGT-I/src/tropicalgt/infer_tropicalgt_i.py`; tests should verify logits are unchanged when artifacts are enabled.
- Likely visualization files: add distinct browser labels for GUDHI `persistence landscape lambda_k(t)` versus GoT NLL/fitness/density fields in `TropicalGT-I/src/tropicalgt/visualization.py`.
- Likely tests: extend metrics/memory/visualization tests for unavailable landscapes, top-k landscape columns, monomial projection, cocycle defect, and zero-coefficient no-op behavior.
- Add config flags for `enable_bundle_transports`, `enable_toric_embedding`, `enable_chart_bpb_consistency`, and `enable_landscape_transport_metrics`; default all new losses to coefficient zero.
- Surface chart ids, overlap pairs, overlap triples, and chart-local NLL tensors from the model only when the flags are enabled.
- Implement monomial transport projection with a hard-audit path and a soft training path; log projection error separately from transport L1.
- Keep byte BPB and graph-BPB as promotion gates: reject or anneal down any auxiliary coefficient that worsens validation BPB, graph-BPB, certificate loss, or tropical wall-hit rate.
- Add persistence-landscape availability masks to memory rows; compute transported L2 and cosine only when both source and target rows have real GUDHI `lambda_k(t)` vectors.
- Add browser artifacts for chart ids, transport ids, toric active rows, normal-fan cell labels, landscape availability, transported L2, and transported cosine.
- Add inference toggles so optional artifacts can be emitted for audits without altering the greedy/sample decoding path.
- Add tests for zero-coefficient no-op behavior, unavailable landscape masking, monomial projection shape, cocycle defect on synthetic charts, and browser label separation between GUDHI landscapes and GoT NLL/fitness/density fields.

## Risks

- Dense or unconstrained transports can violate the tropical vector-bundle interpretation and overfit chart noise.
- Large matroid ground sets can make basis search too expensive; begin with tiny sampled atoms.
- Toric regularization can collapse useful non-toric hidden directions if annealed too quickly.
- Chart consistency can hide real ambiguity if overlap sampling is biased toward easy contexts.
- Persistence-landscape vectors are optional; retrieval metrics must separate unavailable vectors from true zero distances.
- BPB remains the primary acceptance criterion. Auxiliary geometry should be rolled back or gated if validation BPB, graph BPB, certificate loss, or wall-hit metrics degrade.
- Optional inference artifacts can accidentally become training signals if callbacks mutate decoding state; keep artifact emission read-only and test zero-diff logits with artifacts enabled.


## Main-Agent Repair Checklist Addendum

- Add paper-backed config flags and zero-default coefficients for monomial chart transports, one dimension cone-filtration flat defects, toric active-cell agreement, GraphCG-toric agreement, chart BPB consistency, and transported persistence-landscape metrics.
- Model hook: emit chart ids, active tropical support cells, toric active rows, GraphCG active directions, chart-local NLL, and overlap metadata from the TokenGT/tropical attention path without changing logits when all new coefficients are zero.
- Memory hook: carry a real/unavailable flag for each GUDHI persistence-landscape vector and compute transported L2/cosine only when both query and memory vectors are present.
- Visualization hook: expose chart id, transport id, one dimension cone-filtration flat defect, toric active cell, GraphCG-toric agreement, landscape availability, transported L2, and transported cosine in the browser; keep GoT NLL/fitness/density labels distinct from persistence landscape `lambda_k(t)`.
- Gate promotion through validation BPB, graph-BPB, certificate loss, and tropical wall-hit rate. Treat all vector-bundle and toric objectives as auxiliary until matched ablations show benefit.
- Tests to fold into the repair goal: zero-coefficient no-op logits, monomial projection one-hotness, cocycle identity on synthetic charts, decreasing one dimension cone filtrations, finite atom-stability diagnostics, unavailable landscape masking, top-k landscape columns, and read-only inference artifact emission.
