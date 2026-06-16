# Tropical/Toric Embedding Research Review - 2026-06-16

This note completes the first Section 11 research checkpoint for TropicalGT-I. It records what may be implemented now, what must remain unavailable, and which tool certificates are required before the paper or browser can claim a tropical/toric object.

## Sources Checked

- Local reference: `references/1710.10651v2.pdf`, Amendola-Kohn-Lamboglia-Maclagan-Smith-Sommars-Tripoli-Zajaczkowska, *Computing Tropical Varieties in Macaulay2*.
- Official Macaulay2 documentation: `Tropical` package index, `tropicalVariety`, `TropicalCycle`, and method index for `rays`, `maxCones`, `linealitySpace`, `multiplicities`, `isBalanced`, `isPure`, `isSimplicial`, `isTropicalBasis`, and `tropicalPrevariety`.
- Official Sage documentation: tropical semiring multivariate polynomials and `TropicalVariety`/`TropicalCurve` APIs.
- Maclagan-Rincon tropical-ideal work: tropical ideals, varieties of tropical ideals, and tropical subschemes of tropical toric varieties.

## Implemented Certificate Boundary

- `TropicalGT-I/src/tropicalgt/cas_tropical.py` is the current real-only fan/cycle bridge.
- The required certificate path is Macaulay2 `Tropical` on an explicit `QQ[x_i]` ideal. Required tags are produced from `tropicalVariety I`, `rays T`, `maxCones T`, `linealitySpace T`, `multiplicities T`, `isBalanced T`, `isPure T`, `isSimplicial T`, and `fan T`.
- The input ideal is written in an ordinary polynomial ring for Macaulay2. The report records that the certificate is interpreted as the torus-side tropical variety/cycle associated to that finite exported ideal, not as a global toric model of the neural network.
- `isTropicalBasis` and `tropicalPrevariety` are side diagnostics. They can enrich the audit table, but safe fan rendering remains gated by the `tropicalVariety` cycle certificate.

## Sage Scope

- Sage tropical polynomial and variety APIs can support exact tropical semiring polynomial arithmetic, curves, hypersurfaces, plotting, and Newton/polyhedral checks.
- Those APIs do not replace the current Macaulay2 ideal-to-tropical-cycle certificate for the browser fan view. If a future Sage path certifies equivalent fan/cycle fields, it must record its backend and the same schema fields rather than falling back silently.

## Maclagan/Toric-Scheme Scope

- Maclagan-Rincon tropical ideals and tropical toric schemes justify the research direction for tropical subschemes, Hilbert functions, Groebner complexes, finite tropical bases, and toric ambient language.
- TropicalGT-I may use that language only as scoped theory until a backend certifies the exported fan, ideal, grading, sheaf, vector bundle, Cox module, or local cohomology object.
- The existing finite toric sidecar paper language is acceptable because it says the sidecar is a finite certificate for sampled computations rather than a global toric variety claim.

## No-Proxy Contract

- No support-token diagnostic, chain-presentation row, rank sample, embedding-only assignment, interpolated surface, or browser visualization row may substitute for a tropical fan/cycle certificate.
- Unavailable states must stay explicit and include the reason. They may include the command template and backend probe, but they must not render one dimensional cones unless `safe_to_render_as_tropical_fan=true`.
- This contract is now machine-readable as `certificate_contract` on both certified and unavailable tropical fan diagnostic payloads.

## Next Implementation Item

Implement toric embeddings only where the embedding is mathematically correct and tool-backed. In practice this means starting with finite model-derived exponent matrices, toric ideals or Laurent-polynomial sidecars, certified fan/one dimensional cone data, and explicit unavailable states when Macaulay2/Sage/Polymake/Normaliz-level evidence is absent.
