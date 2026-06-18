# TropicalGT-I Paper Review: Tropical Vector Bundles and Tropical Schemes

## Scope

This note reviews the two local references for the TropicalGT-I paper sidecar:

- `references/2405.03505v1.pdf`, Bivas Khan and Diane Maclagan, *Tropical Vector Bundles*.
- `references/2009.03030v2.pdf`, Jaiung Jun, Kalina Mincheva, and Jeffrey Tolliver, *Vector Bundles on Tropical Schemes*.

The review uses the full `pdftotext -layout` extractions of both PDFs. The goal is to identify theorem-level material that can be transferred into TropicalGT-I without overstating what the neural model actually constructs.

**Terminology: one dimensional cone(s).** The plans use one dimensional cone or one dimensional cones as the preferred fan-theoretic language. Singular or plural form follows grammar. These terms index Klyachko-style and Khan-Maclagan-style filtrations, pairings, probes, and TropicalGT-I audit objects.

## Khan-Maclagan: Tropical Vector Bundles

### Core Definitions

- A tropical toric reflexive sheaf on `trop(X_Sigma)` consists of a simple valuated matroid `M` of rank `r` on a ground set `G`, plus for each one dimensional cone `rho_i` a decreasing family of flats `E^i(j)` of the underlying matroid. The filtrations are eventually `G` for low thresholds and empty for high thresholds.
- A tropical toric vector bundle is a reflexive sheaf satisfying a Klyachko-style compatibility condition: for every maximal cone `sigma`, there is a multiset of characters and a matroid basis `B_sigma={w_u}` such that each one dimensional cone filtration is recovered as the join of basis atoms whose one dimensional cone pairing exceeds the threshold.
- The Cox-module construction presents a toric vector bundle as a graded module over the Cox ring; tropicalization uses bend congruences and produces Cox semimodule analogues.
- Fibers of the tropical total space are tropical linear spaces associated to shifted valuated-matroid circuits.
- Realizability is a genuine restriction: a tropical toric vector bundle may fail to come from a classical toric vector bundle either because the valuated matroid is not realizable or because the flat filtrations cannot come from vector-space filtrations.

### Theorem-Level Takeaways

- The main compatibility theorem is finite and combinatorial: a vector-bundle condition can be checked cone-by-cone using matroid bases and flat joins.
- Direct sums correspond to direct sums of valuated matroids; indecomposability is related to connectedness of the underlying matroid.
- Tensoring by a line bundle shifts filtrations, suggesting that threshold shifts can be separated from atom permutations.
- Global sections and `h^0` admit a tropical definition using parliaments of polytopes; this suggests countable finite diagnostics, but it does not directly yield a neural loss without a chosen polytope model.
- Stability and Harder-Narasimhan-style filtrations require modularity or partial modularity hypotheses. For TropicalGT-I this is best treated as a finite diagnostic over feature atoms, not as a theorem about hidden states.

### Transfer to TropicalGT-I

- Use valuated-matroid atoms for feature supports: graph-token support, GraphCG directions, persistence bins, and memory-landscape availability can be finite atoms.
- Use one dimensional cone filtrations as chart-local thresholded probes: endpoint incidence, tropical support margin, active GraphCG one dimensional cone probes, persistence-bin activity, and memory-availability flags become one dimensional cone-like scores.
- Use basis-generated flat reconstruction as a regularizer: `bundle/flat_rank_defect` measures how close predicted chart filtrations are to basis-generated joins.
- Use line-bundle shift intuition for transport shifts: monomial transports should contain both a permutation and tropical additive shifts.
- Use stability language only as an audit heuristic unless the implementation verifies the needed modularity hypotheses.

## Jun-Mincheva-Tolliver: Vector Bundles on Tropical Schemes

### Core Definitions

- A vector bundle on a semiring scheme is a locally free sheaf of modules over the structure sheaf.
- A semiring is zero-sum-free when `a+b=0` implies `a=b=0`; it has only trivial idempotent pairs when decompositions of `1` into orthogonal idempotents are trivial.
- Free modules over such semirings have very few bases: bases are unique up to permutation and multiplication by units.
- Invertible matrices over these semirings are monomial: one nonzero unit per row and column.
- Vector bundles are classified by nonabelian `H^1(X, GL_n(O_X))` in the semiring-scheme setting.

### Theorem-Level Takeaways

- Proposition 3.15 and Corollary 3.17 imply the monomial form of invertible transition maps under the zero-sum-free/trivial-idempotent-pair hypotheses.
- Theorem 4.7 proves that on irreducible semiring schemes locally satisfying those hypotheses, rank-`n` vector bundles split uniquely, up to permutation, as coproducts of line bundles.
- Theorem 5.5 relates topological `T`-vector bundles to finite covering spaces, emphasizing that many tropical vector-bundle notions are combinatorial and can be less expressive than classical analogues.
- Later sections on labelled algebras and saturated spectra show how line bundles can lift, but this material is only indirectly relevant to TropicalGT-I.

### Transfer to TropicalGT-I

- Monomial chart transitions are mathematically justified as the correct invertible tropical-linear transition class under the local semiring hypotheses.
- The splitting theorem is a warning: if TropicalGT-I uses only locally free semiring-scheme bundles, the geometry may collapse to line-bundle-like components. The neural method should therefore treat chart bundles as regularizers over feature packets, not as a claim of constructing a rich semiring-scheme vector bundle.
- The `H^1` classification motivates cocycle consistency: transition maps on triple overlaps should compose coherently, leading to `bundle/cocycle_defect`.

## Transferable TropicalGT-I Design

### Objects

- TokenGT tropical atlas: charts over graph-token subsets, active tropical attention support cells, graph-of-thought states, GraphCG neighborhoods, persistence bins, and analogical-memory packets.
- Bundle atoms: finite feature atoms representing graph-token support, one dimensional cone probes, GraphCG directions, persistence summaries, and real GUDHI landscape vectors.
- Monomial transport: a sparse permutation plus tropical shift between overlapping charts.
- One dimension cone filtration: a decreasing family of matroid flats induced by thresholding graph-token or direction probes.
- Toric embedding: a small integer max-linear feature map whose active rows identify local normal-fan-like cells.

### Losses and Metrics

- `bundle/transport_l1`: overlap feature disagreement after monomial transport.
- `bundle/cocycle_defect`: failure of `T_bc T_ab = T_ac` on sampled triples.
- `bundle/flat_rank_defect`: distance between predicted filtrations and basis-generated flat joins.
- `toric/normal_fan_loss`: disagreement between toric active rows and chart filtration cells.
- `graphcg/toric_cell_agreement`: alignment between active GraphCG direction cones and toric active cells.
- `chart/bpb_consistency`: robust chart-local NLL agreement after transport.
- `memory/transported_landscape_l2` and `memory/transported_landscape_cosine`: computed only when both sides have real GUDHI `lambda_k(t)` vectors.
- `bundle/atom_stability_gap`: optional stability-inspired diagnostic comparing atom-subset slopes or loads to the whole bundle load.

## Non-Transfers and Required Caution

- TokenGT graph-token batches are not automatically tropical toric varieties.
- Neural hidden states are not automatically locally free sheaves over a tropical scheme.
- BPB improvement is not a theorem of either paper. It must be validated by held-out BPB and graph-BPB gates.
- Persistence landscapes are not GoT NLL surfaces or fitness-density fields. They must remain separately labeled and separately masked.
- Stability and Harder-Narasimhan language should be diagnostic unless the implementation verifies modularity or a precise finite replacement condition.

## Implementation Handoff

The paper-sidecar implementation now has a read-only audit path for the first required code-level hooks: chart ids, overlap triples, monomial transport ids, configured toric active rows, one dimensional cone-filtration flat defects, GraphCG-toric agreement metrics, and transported persistence-landscape memory metrics. The browser JSON/HTML sidecar uses schema `tropicalgt.vector_bundle_paper_sidecar.v1`, marks all missing fields unavailable with source/reason records, and explicitly refuses theorem/certificate promotion. Promotion to training use still requires no regression in validation BPB, graph-BPB, certificate loss, or tropical wall-hit rate, and browser labels must continue distinguishing real GUDHI `lambda_k(t)` landscapes from NLL, fitness, or density fields.

## Real Implementations Only Policy

No TropicalGT-I metric, loss, visualization, analogical map, persistence module, free resolution, derived comparison, tropical-cycle diagnostic, or CAS artifact should be presented as a mathematical object unless it is computed from the actual model outputs, graph states, embeddings, probabilities, simplex trees, bifiltrations, or certified CAS/backend output that define that object. Temporary placeholders, synthetic fallback objects, mock charts, fabricated simplices, and convenience stand-ins are not acceptable. When a requested object cannot yet be computed, the artifact must render an explicit unavailable/uncertified state and the training metric must either be disabled or logged under an audit-only unavailable flag. Finite chain-presentation diagnostics may be shown only as chain diagnostics, never as free resolutions. Total-graded or ungraded CAS output may be shown as real CAS output only under its actual grading; it must not be advertised as a multigraded `F2[x_level,x_radius]` free resolution unless the backend certifies that multigraded structure.

Use "one dimensional cone" or "one dimensional cones" as the preferred fan-theoretic language whenever the intended object is a cone of a fan or a cone-indexed filtration datum. Use singular or plural according to ordinary grammar.

_Last updated: 2026-06-18T00:00:00+00:00_
