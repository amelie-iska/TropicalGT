# TropicalGT-I Vectorized Persistence Memory Notes

Date: 2026-06-13

## Terminology

The codebase uses two different objects that have both been called
`landscape`. They must stay separate.

1. **GoT NLL / fitness landscape.** This is a visualization of model-evaluated
   NLL or fitness values around graph-of-thought states in a low-dimensional
   projection of model embeddings. It is not a persistent-homology invariant.

2. **GUDHI persistence landscape.** This is the vectorized persistence-diagram
   representation `lambda_k(t)` computed from finite persistence intervals by
   `gudhi.representations.Landscape`. It is a topology feature used for
   retrieval, comparison, plotting, and optional inference artifacts.

Browser labels and planning text should say `NLL/fitness landscape` for the
first object and `GUDHI persistence landscape` for the second object.

## Current Retrieval Policy

Analogical memory retrieval now separates three topology channels:

- `persistence_landscape_score_contribution`: standalone GUDHI
  `Landscape.vector` similarity.
- `persistence_vector_score_contribution`: weighted vector-space similarity
  across the broader GUDHI vector-representation family.
- `retrieval_score`: the full weighted score before diversity reranking.

When `memory_retrieval_landscape_weight > 0`, the vector-family comparison
excludes `Landscape` so that the standalone persistence-landscape signal is not
double-counted. When the vector-family channel is used without a standalone
landscape weight, it may include `Landscape` for backward compatibility.

The non-landscape vector family currently includes:

- `BettiCurve`
- `Silhouette`
- `Entropy`
- `PersistenceLengths`
- `TopologicalVector`
- `PersistenceImage`

All values are extracted from actual cached GUDHI representation outputs in the
topological-algebra report. Missing diagrams, missing vectorizers, failed
methods, or absent shared methods render unavailable rather than creating a
synthetic score.

## Differentiability Scope

The retrieval comparison is vectorized and differentiable with respect to the
already-computed feature vectors because it uses L2, cosine, and correlation
style operations. This implementation does not claim autograd through GUDHI's
diagram-to-vector transforms. A future torch-native persistence-vector surrogate
could be added as a training-time differentiable approximation, but it must be
marked separately from the GUDHI certificate path.

## Telemetry

Periodic audits now log memory metrics for:

- retrieved top-k count,
- configured landscape and vector-family weights,
- vector-family availability rate,
- vector method-count mean,
- vector aggregate similarity mean,
- vector score contribution mean,
- landscape score contribution mean.

These are intended to be compared against BPB and graph BPB to decide whether
topological retrieval helps compression or only looks visually plausible.

## Tests Added / Maintained

The focused memory and visualization tests assert that:

- standalone GUDHI persistence-landscape retrieval contributes through the
  landscape channel;
- explicit vector-family retrieval contributes through the vector channel;
- `Landscape` is excluded from the vector-family channel when a separate
  landscape weight is active;
- diversity reranking no longer overwrites the full topology-aware retrieval
  score;
- exported analogical map JSON carries the full vector-family report and its
  `includes_landscape` policy flag.

## Open Follow-Ups

- Add per-method vector-family W&B availability rates if method-level ablations
  become necessary.
- Add a browser table that expands each vector method contribution rather than
  only the compact component summary.
- Keep the `trajectory_persistence/persistence_landscapes.html` page; it is a
  real GUDHI persistence-landscape page, not the GoT NLL/fitness landscape.
