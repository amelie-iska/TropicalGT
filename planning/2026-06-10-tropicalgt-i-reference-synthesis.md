# TropicalGT-I Reference Synthesis and Porting Crosswalk

This note is the working literature audit for TropicalGT-I.  It records what was reviewed from the local reference files, which ideas were ported into the TropicalGT-I paper/code, which ideas were intentionally omitted, and which ideas remain future work.  The source PDFs were inspected locally with `pdfinfo` and `pdftotext`; the extracted text cache was temporary and is not tracked.

## Source Manifest

| File | Pages/lines | Role in TropicalGT-I |
|---|---:|---|
| `references/1311.2360v3.pdf` | 27 pages | Didactic tropical geometry: tropical polynomials, curves, balancing, stable intersections, and patchworking. |
| `references/2301.12594v2-1.pdf` | 32 pages | Continuous GFlowNets: measurable state spaces, flow matching, detailed balance, trajectory balance, and reward matching. |
| `references/2308.09687v4.pdf` | 63 pages | Graph of Thoughts: arbitrary graph-structured reasoning, transformations, scoring, aggregation, cost/quality tradeoffs. |
| `references/2401.17123v1-1.pdf` | 33 pages | GraphCG: unsupervised steerable latent directions for graph generative models via contrastive/EBM/NCE objectives. |
| `references/2505.17190v2-1.pdf` | 20 pages | Tropical Attention: tropical projective attention, Hilbert metric, MHTA, tropical circuit and dynamic-program simulation. |
| `references/2602.01992v5-1.pdf` | 30 pages | Emergent analogical reasoning: relational structure transport and tests for in-context analogy. |
| `references/2604.14727v1-1.pdf` | 25 pages | Transformer expressivity through tropical geometry: power diagrams, Newton polytopes, MHSA Minkowski sums, region bounds, finite-temperature stability. |
| `references/1001.1554v4.pdf` | 49 pages | Tropical correspondence and toric degeneration machinery: parameterized tropical curves, obstruction/deformation spaces, correspondence theorems. |
| `references/MaclaganSturmfels.pdf` | 202 pages | Foundational tropical algebraic geometry: tropical varieties, Groebner complexes, tropical bases, Kapranov theorem, normal fans. |
| `references/main.pdf` | 30 pages | TokenGT expressivity extension: graph-to-graph universality, Boolean circuit adaptation, tropical TokenGT region bounds, toric boundary, GraphCG fan slices. |
| `references/toricgt_paper_pg_softmoe_final.tex` | 3112 lines | ToricGT methods source: graph-token reasoning, ring attention, GFlowNet branch training, Soft-MoE/Parameter-Golf discipline, visualization contracts, dataset schema. |
| `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` | 1844 lines before iteration 9 | Active TropicalGT-I manuscript to keep aligned with code. |

## Paper-by-Paper Synthesis

### Brugalle-Shaw, `1311.2360v3.pdf`

The paper is useful for exposition rather than architecture.  It introduces tropical polynomials as max-plus objects, tropical plane curves as corner loci, weighted edges, and the balancing condition.  The extracted theorem structure includes algebraic closure of the tropical semiring, tropical curves as balanced graphs, stable intersections, tropical Bezout, Viro patchworking, and the Mikhalkin-Rullgard limit picture.

Ported into TropicalGT-I:

- Use tropical polynomials as the language for attention heads whose outputs are maxima of affine candidates.
- Interpret active supports as exposed alternatives, not as probabilistic soft assignments.
- Use balancing/stability language carefully as geometric intuition for active-face and margin diagnostics.

Not ported:

- Enumerative curve-counting and patchworking algorithms are not training objectives in v1.
- Tropical curve multiplicities are not currently used as losses; v1 only logs support/margin/fallback/certificate diagnostics.

### Continuous GFlowNets, `2301.12594v2-1.pdf`

The paper extends GFlowNets from discrete DAGs to measurable spaces.  The key technical pieces are terminating states, transition kernels, forward/backward kernels, flow matching, detailed balance, trajectory balance, reward matching, and conditions under which the terminal law is proportional to reward.

Ported into TropicalGT-I:

- A trajectory-balance auxiliary over embedding-space graph-of-thought states.
- A scalar log-partition parameter `log_z`, per-trajectory log-probability, reward log, residual, and reward diagnostics.
- A design rule that the terminal reward can later be swapped from smoke-stage likelihood/margin reward to verifier, proof-validity, or certificate reward without changing the report schema.

Not ported:

- The v1 policy is discrete over eight reasoning actions; it does not yet implement a full continuous transition kernel with learned density over the embedding space.
- Detailed-balance and subtrajectory-balance training are logged as future diagnostics, not full objectives.

### Graph of Thoughts, `2308.09687v4.pdf`

GoT models reasoning as a directed graph whose vertices are thoughts and whose edges encode dependencies.  It generalizes chain/tree reasoning with graph operations such as generation, aggregation, scoring, refinement, merging, and deletion.  The extracted text emphasizes graph state, operation modules, controller, sorting/set/document tasks, quality improvements, and cost-volume tradeoffs.

Ported into TropicalGT-I:

- Prompt-level thoughts become graph tokens and filtered simplicial objects rather than only external prompt strings.
- The inference-scaling controller expands candidate graph states by GFlowNet-preferred actions: expand, merge, refine, stop, retrieve, verify, compress, reject.
- Every candidate records path, score, NLL proxy, token budget, tropical support trace, and filtered simplicial object.

Not ported:

- The implementation does not call an external LLM for every thought transformation.
- GoT's prompt-engineering API is not reproduced; TropicalGT internalizes the graph in embeddings and graph tokens.

### GraphCG, `2401.17123v1-1.pdf`

GraphCG studies pretrained graph DGMs whose latent spaces are empirically entangled.  It learns semantic direction vectors by maximizing mutual information between same-direction graph edits.  The paper formulates EBM and NCE variants, uses positive pairs generated by same direction/step families, negatives from other directions/steps, and adds similarity and sparsity terms.

Ported into TropicalGT-I:

- `GraphCGLoss` learns a small bank of latent directions over pooled graph-token states.
- The v1 objective uses same-condition contrastive consistency plus direction orthogonality/covariance and sparsity penalties.
- Diagnostics report direction norms and off-diagonal cosine/covariance proxies.
- The paper interprets GraphCG directions as coordinates in which tropical fan crossings and graph-of-thought moves become more identifiable.

Not ported:

- TropicalGT-I does not include a graph DGM decoder or molecule/point-cloud generation loop.
- The full GraphCG-NCE estimator over decoded graph edit sequences is not implemented in v1.
- GraphCG is treated as a light chart regularizer, not as proof that semantic factors are identifiable without task/verifier data.

### Tropical Attention, `2505.17190v2-1.pdf`

This is the immediate attention mechanism source.  It replaces softmax dot-product attention with tropical projective attention, uses tropical Hilbert distance, proves 1-Lipschitz/projective behavior, and shows MHTA can simulate max-plus dynamic programs and approximate tropical circuits/transitive closure.

Ported into TropicalGT-I:

- A max-plus `TropicalRingAttention` layer over graph-token features.
- Active support ids, top-two margins, support entropy, and blockwise exactness tests.
- The dynamic-programming interpretation: edge-local max-plus/min-plus updates can be viewed as graph-token attention over candidate predecessors/evidence.

Modified for v1:

- The implementation uses a compact max-plus score/reduction with finite support diagnostics, not the complete Hilbert-projective MHTA kernel from the paper.
- The v1 graph encoder conditions a byte-level model; it is a functional training path, not a full neural algorithmic benchmark suite.

### Emergent Analogical Reasoning, `2602.01992v5-1.pdf`

This paper motivates evaluating whether transformers preserve relational mappings rather than isolated facts.  TropicalGT-I uses it as a diagnostic target: graph-of-thought trajectories should be checked for relation-preserving transport in embedding space.

Ported into TropicalGT-I:

- The visualization payloads keep full filtered simplicial objects so future analogical audits can compare local reasoning complexes.
- Inference-scaling reports preserve action paths and graph edits, making relational transfer errors inspectable.

Not ported:

- The synthetic analogy benchmark and full experimental protocol are not part of v1 smoke training.
- No claim is made that current smoke checkpoints exhibit emergent analogy.

### Su-Liu Transformer Expressivity, `2604.14727v1-1.pdf`

This paper is the tropical expressivity backbone.  It models self-attention as a vector-valued tropical rational map.  In the zero-temperature limit, attention partitions query space as a power Voronoi diagram.  Single-head attention has an O(N) Newton-polytope bottleneck, while multi-head aggregation corresponds to Minkowski sums of head Newton polytopes and can grow like O(N^H) under general-position assumptions.  The paper also derives depth/width/sequence-length linear-region bounds and finite-temperature stability via log-sum-exp approximation.

Ported into TropicalGT-I:

- Replace the sequence token count N by the graph-token count N_G = 1 + |V| + |E| (or sparse variants) in expressivity reasoning.
- Treat active attention supports as cells in a normal fan/power-diagram-like partition.
- Use top-two margins as finite-temperature stability proxies.
- Explain why graph tokenization changes expressivity before learning: edge tokens make incidence predicates directly available to attention.

Not ported or qualified:

- V1 does not compute exact power Voronoi cells or fan volumes during training; it logs support/margin proxies.
- General-position O(N^H) region growth is a worst-case geometric capacity statement, not an empirical claim about the smoke checkpoint.

### Nishinou/Siebert-style correspondence source, `1001.1554v4.pdf`

The extracted theorem structure covers parameterized tropical curves, stabilization, deformation/obstruction spaces, tropical degenerations in toric varieties, stacky reductions, and correspondence theorems.  It is useful for discipline when moving between toric and tropical language.

Ported into TropicalGT-I:

- The paper's cautionary lesson: a tropical object often corresponds to a classical/toric object only with extra hypotheses, obstruction control, and compatibility constraints.
- TropicalGT-I uses this to avoid claiming an unconditional toric geometry theorem for transformers.

Not ported:

- Stable maps, stacky reductions, and Gromov-Witten correspondence machinery are outside v1 training.

### Maclagan-Sturmfels, `MaclaganSturmfels.pdf`

This book supplies the foundation for tropical varieties, Groebner complexes, tropical bases, Kapranov's theorem, the Fundamental Theorem of Tropical Algebraic Geometry, polyhedral complexes, and normal fans.  It is the reference for precise use of Newton polytopes and normal fans.

Ported into TropicalGT-I:

- Newton polytopes and normal fans are used as the correct language for active affine candidates.
- The implementation distinguishes structural certificate diagnostics from genuine algebraic certificates.
- The paper's toric/tropical boundary is phrased as conditional on rational polyhedral skeletons.

Not ported:

- No Groebner basis computation is in the v1 code path.
- No claim is made that generic finite-temperature transformers define tropical varieties.

### TokenGT expressivity extension, `references/main.pdf`

This is the closest mathematical template for TropicalGT-I.  It defines graph-to-graph maps, proves TokenGT universality for continuous equivariant graph-to-graph maps on fixed compact graph domains, gives exact finite interpolation on finite graph domains, adapts transformer circuit results to graph tokens, derives TokenGT tropical region bounds, analyzes GraphCG fan slices, and gives counterexamples to a broad toric-transformer theorem.

Ported into TropicalGT-I:

- TokenGT-style graph tokenization with graph/vertex/edge tokens, type ids, deterministic identifiers, endpoint ids, masks, and graph-level slot.
- Universal approximation and finite-domain interpolation motivate the paper's theoretical claims.
- Boolean/tropical reachability examples are framed as graph-token circuit simulations.
- Counterexample discipline is adopted: finite-temperature softmax, moving keys, affine biases, normalization, and signed values block an unconditional toric theorem.

Not ported:

- The v1 code does not attempt to instantiate every theorem's constructive network; it provides the first trainable scaffold and diagnostics.

### ToricGT, `references/toricgt_paper_pg_softmoe_final.tex`

ToricGT is much broader than TropicalGT-I.  Its portable layers are graph-token reasoning, exact blockwise tropical ring attention, embedding-space GFlowNet fine-tuning, GraphCG charts, W&B/reproducibility discipline, dataset schema, Parameter-Golf compression discipline, and interactive trajectory visualization.  Its toric-only layers include noncommutative tori, finite Weyl pairs, toric ideals, Hebrew morphology sections, Soft-MoE expert-bank design, PolarQuant toric phase caches, and behavior-corridor algebraic control.

Ported into TropicalGT-I:

- Dataset schema and moved ToricGT shards.
- TokenGT-style graph tokens and endpoint ids.
- GFlowNet branch/scaling reports.
- GraphCG latent-direction auxiliary.
- W&B metrics, checkpoints, resume support, visual artifacts, and readiness audits.
- Filtered simplicial objects in hover payloads.

Not ported:

- Noncommutative torus phase memory, toric ideals, Weyl-pair traces, Hebrew morphology losses, full Soft-MoE expert banks, PolarQuant caches, and behavior corridor algebra are deliberately omitted from v1 unless later translated into a tropical fan/support/margin analogue with measurable benefit.

## Crosswalk From Toric To Tropical

| ToricGT item | TropicalGT-I treatment | Reason |
|---|---|---|
| TokenGT graph tokens | Ported | Required for vertex/edge expressivity and graph-to-graph universality. |
| Tropical ring attention | Ported in compact max-plus form | Directly tropical, useful for active-support diagnostics. |
| Noncommutative tori/phase memory | Omitted | Toric phase channels have no necessary max-plus analogue. |
| Toric ideals and BGG/Koszul probes | Omitted | Algebraic certificates are not available from v1 support logs. |
| GFlowNet trajectory training | Ported as TB auxiliary and inference-scaling controller | Matches graph-of-thought search and continuous GFlowNet motivation. |
| GraphCG steering | Ported as light latent chart regularizer | Useful as coordinate chart for tropical fan slices. |
| Soft-MoE typed experts | Not in v1 | May be useful later, but first pass prioritizes small reproducible graph-token model. |
| Parameter-Golf compression discipline | Ported as optional external graph adapter and BPB-first reporting | Keeps baseline path intact while enabling graph conditioning. |
| Interactive toric trajectory overlays | Translated to PCA/NLL/filtered-simplicial Plotly audits | TropicalGT needs support/margin/simplex evidence rather than torus phase plots. |

## Implementation Alignment After Iteration 8

Implemented evidence:

- `TropicalGT-I/src/tropicalgt/tokenizer.py`: graph token, vertex tokens, edge tokens, type ids, endpoint ids, deterministic identifiers.
- `TropicalGT-I/src/tropicalgt/attention.py`: max-plus tropical attention, support ids, margins, masked/soft support entropy, blockwise exactness tests.
- `TropicalGT-I/src/tropicalgt/model.py`: graph-conditioned byte model with GFlowNet, GraphCG, margin/entropy/certificate objectives and metrics.
- `TropicalGT-I/src/tropicalgt/scaling.py`: bounded graph-of-thought inference-time scaling.
- `TropicalGT-I/src/tropicalgt/simplicial.py` and `visualization.py`: filtered simplicial objects and interactive Plotly PCA/NLL audits.
- `TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py`: data/package/checkpoint/eval/inference/visual readiness gates.
- `external/parameter-golf`: optional TokenGT-style graph adapter that preserves the default baseline path.

Important limitations:

- The current TropicalGT-I model is a first functional scaffold, not a competitive compression model.
- The GFlowNet reward remains smoke-stage likelihood/margin/token-budget based.
- GraphCG is a light auxiliary; no semantic identifiability claim is made.
- Current "certificates" are structural TokenGT skeleton diagnostics unless teacher supports or external proof/verifier labels are supplied.
- Exact normal-fan wall distances and ReLU toric divisors are not computed in v1.

## Paper Update Requirements Derived From The Review

- Keep the paper explicit that TropicalGT-I extends the Su-Liu tropical expressivity picture by changing token count and incidence structure through TokenGT tokenization.
- Keep graph-of-thought, GFlowNet, GraphCG, and inference-scaling claims tied to the implemented lightweight objectives and reports.
- Keep toric claims conditional and counterexample-aware: restricted rational zero-temperature/ReLU skeletons admit toric shadows; general finite-temperature transformers do not automatically lie at a toric/tropical intersection.
- Use "certificate diagnostics" for v1 support/margin logs unless exact teacher supports, verifier proofs, or algebraic certificates are present.
