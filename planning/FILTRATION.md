# TropicalGT-I Filtration, Simplex Tree, Persistence, and Analogical Maps

This document is the implementation contract for the simplicial and algebraic
audit pipeline used by TropicalGT-I figures, training diagnostics, inference
artifacts, and memory retrieval.  It is intentionally stricter than a plotting
note: if a visualization or metric cannot be traced back to one of the finite
objects below, it must be marked unavailable or removed.  Approximate algebraic
quantities are allowed only when explicitly named as approximations.

## Objects Being Filtered

TropicalGT-I uses two closely related filtered simplicial objects.

1. **Reasoning-step complex.**  For one model reasoning state, vertices are the
   graph/token vertices carried by the `GraphRecord` or the generated reasoning
   state.  Edges are observed graph edges, causal text edges, or TokenGT-style
   graph edges.  Directed length-two motifs may contribute 2-simplices when all
   faces are present.

2. **Graph-of-thought trajectory complex.**  For a sampled reasoning trajectory,
   vertices are sampled GoT states in model embedding space.  When graph-state
   embeddings are available, the trajectory complex is the bounded
   Vietoris-Rips 2-skeleton on those embedding vectors: a 1-simplex enters at
   the normalized Euclidean radius between two GoT state embeddings, and a
   2-simplex enters at the maximum radius of its three faces.  Parent-child GoT
   transitions are retained as marked edges, but they do not replace the radius
   filtration.

The first filtration frame must show all 0-simplices as a disjoint vertex cloud
and no positive-radius higher simplices.  The only allowed exception is a
genuine zero-distance higher simplex, for example duplicate embedding vectors;
that case must be reported in the summary as `zero_distance_edges`.

## Scalar Radius Filtration

Every displayed scalar filtration is finite and monotone.

- Vertices: filtration `0.0`.
- Graph-token edges with endpoint model embeddings: normalized Euclidean
  endpoint distance.
- Graph-token edges in probability mode: model-predicted graph-token
  probability vectors, normally tropical-support distributions, determine the
  edge radius by Jensen-Shannon distance.  Embedding-softmax substitutes are not
  allowed.
- Graph edges without endpoint embeddings or probability vectors are not a
  radius filtration. They may appear only in the legacy
  `graph_combinatorial_non_radius_legacy` helper and must not be used for
  training topology, inference topology, or any figure labeled radius.
- Trajectory Rips edges: normalized Euclidean distance between GoT state
  embeddings.
- Trajectory probability Rips edges: Jensen-Shannon distance between
  model-predicted GoT action/state probability vectors.
- 2-simplices: maximum filtration of their faces, optionally plus a small
  positive motif offset only in explicitly labeled non-radius graph diagnostics.

Thus a slider at radius `r` displays exactly the subcomplex

```text
K_r = { sigma in K : filtration(sigma) <= r }.
```

The slider direction is always min-to-max: left is the disjoint vertex cloud,
right is the full displayed complex.

## Gudhi Simplex Tree

Before persistence or simplex-tree figures are rendered, the JSON complex is
canonicalized through `gudhi.SimplexTree` when Gudhi is available.

1. Insert each simplex with its scalar filtration.
2. Call `make_filtration_non_decreasing()`.
3. Serialize `tree.get_filtration()` back to JSON, preserving available source
   metadata.
4. Record `backend`, `num_vertices`, `num_simplices`, `dimension`, and
   `filtration_non_decreasing`.

The 3D simplex-tree plot is not a PCA/radius complex.  It is a Hasse-style
inclusion diagram:

- x-axis: scalar filtration value;
- y-axis: simplex dimension;
- z-axis: deterministic sibling order inside a dimension;
- edges: face-to-coface inclusions.

This plot is expected to look structured because a simplex tree is an ordered
combinatorial object.  The geometric/Rips complex plots should look like
embedding-space topology; the simplex-tree plot should reveal inclusion
structure.

## Persistent Homology

Persistent homology is computed from the closed finite filtered complex.

- Preferred backend: `gudhi.SimplexTree.persistence()`.
- Optional backend: ripser for distance-matrix style checks where appropriate.
- Field: `F2`.
- Zero-length intervals are filtered from human barcode views unless explicitly
  requested for debugging.
- Infinite bars are rendered with a finite display endpoint and marked as
  infinite in hover payloads.

The trajectory barcode should evolve as training changes the GoT embedding
geometry.  It may become more regular if training collapses trajectories or if
the same tree policy repeatedly samples similar paths; that is a model behavior
diagnostic, not a plotting success.

## Persistence Representations

Vectorized persistence features are computed for faster training-time and
retrieval-time diagnostics.  When Gudhi representations are available, the
pipeline computes:

- persistence landscapes, including actual lambda-k curves for plotting;
- persistence images;
- Betti curves;
- entropy summaries;
- topological vectors when diagram size permits.

These vectorized summaries are suitable for W&B metrics, auxiliary rewards,
analogical memory retrieval keys, and inference-time optional outputs.  They do
not replace the barcode or simplex tree; they are fast representations of the
same finite persistence diagrams.

## Multiparameter Persistence Module

The multiparameter module currently uses a genuine finite three-parameter grid
over the displayed complex:

```text
(scalar filtration, simplex dimension, max vertex position).
```

For each grid point, the implementation computes the subcomplex fiber,
boundary matrices over `F2`, Betti ranks, and Euler characteristic.  H0
rank-invariant samples are exact inclusion-induced ranks on comparable grid
points.  Higher-dimensional multiparameter rank invariants may require heavier
optional backends such as multipers, RIVET, or Macaulay2 and must be labeled
accordingly when absent.

## Free Resolutions and DG-Commutative Algebra

The commutative algebra diagnostics distinguish exact computations from upper
bounds.

- Boundary maps and boundary syzygies are exact F2 matrix/nullspace
  computations for the displayed finite complex.
- Stanley-Reisner generators are computed from missing faces in the bounded
  displayed vertex set.
- Hochster Betti numbers are exact on the bounded induced subcomplex named in
  the report.
- Taylor resolution ranks are nonminimal upper bounds and must be labeled as
  such.
- The multiparameter free-resolution object is a finite free-chain/free-module
  proxy over `F2[x_filtration,x_dimension,x_position]`; it is not to be called
  a minimal free resolution unless a backend certifies minimality.
- DG-commutative algebra reports use the displayed cochain/boundary data over
  `F2` and are finite audit objects, not a claim of full symbolic resolution
  over an unbounded module.

## Derived Equivalence Signatures

The current derived-equivalence signature is an invariant vector used for
retrieval and comparison.  It includes Betti vectors, finite and infinite
persistence interval counts, multiparameter grid size, and related finite
module summaries.  It is a derived-invariant-inspired signature, not a proof of
derived equivalence.  A figure or training loss may say "derived-equivalence
signature similarity" but must not say two objects are derived equivalent
without an explicit certified functor/equivalence or backend proof.

## Analogical Reasoning as Simplicial Maps

Analogical memory retrieval compares a query trajectory complex with memory
trajectory complexes.

The intended map is a filtration-aware simplicial map:

1. Match vertices using model-predicted probability vectors, normally by a
   minimum-cost assignment under Jensen-Shannon distance. Embedding coordinates
   can be used for layout and visualization, but they are not the source of the
   analogical map when probability vectors are available.
2. Check whether mapped edges exist in the codomain at the displayed
   filtration.
3. Check whether mapped 2-simplices exist in the codomain at the displayed
   filtration.
4. Render preserved simplices distinctly from vertex-only correspondences.
5. Report map quality as counts and fractions of preserved 1- and 2-simplices.

If the displayed skeleton fails the simplicial-map condition, the plot must say
so.  A map with many vertex correspondences but few preserved edges/faces is an
analogical retrieval candidate, not a valid simplicial map on that displayed
subcomplex.

The analogical map slider is required because the truth of "is a simplicial map
on the displayed skeleton" changes with radius.

## Training and Inference Use

Training and inference can use the following outputs.

- Scalar topology metrics: Betti ranks, Euler characteristic, finite interval
  counts, landscape norms, entropy, topological vector norms.
- Graph/trajectory metrics: Rips edge counts, zero-distance duplicates,
  preserved analogical edge/face ratios, simplex-tree size, H0/H1 changes.
- Auxiliary rewards: persistence-landscape similarity, trajectory diversity,
  analogical retrieval quality, and stable tropical support agreement.
- Optional inference artifacts: trajectory complex, simplex tree, barcodes,
  persistence landscapes, multiparameter summaries, free-resolution reports,
  and analogical simplicial maps.

No synthetic fallback may be silently rendered.  If a backend is unavailable,
the output must state that the computation is unavailable.  If a mathematically
bounded approximation is intentionally used, the approximation name must appear
in the payload and plot subtitle.

## Interactive Artifact Contract

The current browser-facing audit emits named artifacts with separate geometric
and probability filtrations:

- `got_trajectory_pca_3d.html`: sampled GoT graph-state PCA with exact NLL
  anchors plus the surface-contact projection contract from model-evaluated states only.
- `got_embedding_map_3d.html`: model `graph_state` embedding PCA, independent
  of tree/layout coordinates.
- `got_full_trajectory_complex.html` and
  `got_full_trajectory_simplex_tree_3d.html`: Euclidean radius complex and its
  GUDHI inclusion poset.
- `got_full_trajectory_complex_jensen_shannon.html` and
  `got_full_trajectory_simplex_tree_3d_jensen_shannon.html`: probability
  complex and simplex tree from model probability vectors under Jensen-Shannon
  distance.
- `reasoning_step_###.html` and `reasoning_step_###_simplex_tree.html`: every
  sampled reasoning state's filtered complex and inclusion poset.
- `analogical_memory_map_##.html`: per-rank analogical maps; preserved
  simplex maps are visible, while vertex-only correspondences are low-emphasis
  or legend-only.

The sample-first browser index must expose per-sample buttons for these files
when present. `validate_interactive_audit_artifacts.py` gates relative links,
per-sample `data-samples` coverage, analogical `pair_page` links, unavailable
reasons, GUDHI simplex-tree provenance, probability-map source, filtration
distortion summaries, and preserved/failed simplex evidence.

## 2026-06-12: Multiparameter Persistence, Fitting Ideals, and Buchsbaum-Eisenbud Diagnostics

TropicalGT-I now separates the fast finite topology layer from the commutative-algebra layer so that the browser artifacts and training metrics do not conflate true computations with unavailable symbolic certificates.

### One-Parameter Radius Persistence

For each displayed reasoning state or full trajectory complex, the radius filtration starts with a disjoint cloud of model-derived vertices. The vertices are graph-token, reasoning-state, or action-probability objects carrying actual model embeddings and, when available, actual model probability vectors. A Vietoris-Rips-style two-skeleton is formed by adding an edge when the selected metric distance is at most the radius threshold and adding a two-simplex when all its faces are present by that radius. The default metric is Euclidean distance on model graph-state embeddings for embedding complexes and Jensen-Shannon distance on model probability vectors for probability complexes. No fallback point cloud is used for model-backed audits; missing embeddings or probabilities render the object unavailable.

The one-parameter persistent homology page is computed from the resulting finite filtered simplicial complex over `F2`. With GUDHI available, the simplex tree stores the exact simplex filtration values and returns persistence intervals. The barcode and persistence-landscape pages are therefore visual summaries of the same finite radius filtration, not separate synthetic traces.

### Two-Parameter Trajectory Module over `F2[x_level,x_radius]`

For a sampled graph-of-thought trajectory, the primary two-parameter module is

`K(i,r) = { simplices born by reasoning growth level <= i and radius <= r }`.

This is a finite bifiltration indexed by `(level, radius)`. Algebraically, it is represented as a finitely generated multigraded persistence module over `F2[x_level,x_radius]`. Each displayed grid fiber `K(i,r)` is reduced as an exact finite `F2` chain complex. The interactive plot `trajectory_persistence/two_parameter_bifiltration.html` renders Betti surfaces over the `(radius, level)` grid, while hover text reports the exact fiber, Betti rank, chain ranks, and Euler characteristic.

The chain module generators are the displayed simplices with multidegrees `(first_reasoning_level, radius_bucket)`. Boundary entries are monomial-labeled maps: if a simplex `sigma` has multidegree `a` and a face `tau` has multidegree `b <= a`, the boundary entry is `x_level^(a_1-b_1) x_radius^(a_2-b_2)` over `F2[x_level,x_radius]`. The implementation checks and reports these monomial boundary maps as the finite multigraded free-chain presentation.

### Fitting Ideals and Minors

For a presentation map `phi: F -> G -> coker(phi) -> 0`, TropicalGT-I records the Fitting invariant convention used in the 2210.11433 paper:

`Fitt_j(coker(phi)) = I_{rank(G)-j}(phi)`,

where `I_s(phi)` is generated by the size-`s` minors of the matrix for `phi`. For numeric `F2` boundary matrices, these ideals collapse to rank facts. For the monomial-labeled multiparameter boundary matrices, bounded determinantal generators are computed directly over `F2[x_level,x_radius]` or the displayed polynomial ring. Large ideals are sampled for browser/runtime safety and marked as sampled; full ideal generation, Groebner bases, and quotient-ring computations require a CAS backend.

### Buchsbaum-Eisenbud Diagnostics and Multipliers

The browser/inference algebra payload includes the finite Buchsbaum-Eisenbud rank/exactness shadow:

- exact monomial composition checks `d_i d_{i+1}=0`,
- finite `F2` rank exactness checks `rank(d_i)+rank(d_{i+1})=rank(C_i)`,
- maximal-minor samples that are the inputs needed for Buchsbaum-Eisenbud multiplier relations.

The actual Buchsbaum-Eisenbud multipliers are symbolic complementary-minor relations in the polynomial coordinate ring. Their certification, together with grade/depth or regular-element conditions on determinantal ideals, requires Macaulay2, Singular, Sage, or an equivalent commutative-algebra backend. TropicalGT-I therefore emits the exact finite and bounded symbolic inputs, but marks multiplier certification and minimal free resolutions as unavailable unless a CAS certificate is attached.

### Derived Analogy Checks

Analogical memory retrieval should compare candidate memories using the model-derived probability or embedding complexes, then augment the similarity score with derived-invariant summaries from the algebra payload: Betti signatures, persistence landscapes, rank-invariant samples, Fitting/minor summaries, and certified resolution data when available. Without a CAS certificate, the system may use the exact finite free-chain presentation and bounded determinantal summaries as invariants, but it must not claim a certified derived equivalence of minimal resolutions.

