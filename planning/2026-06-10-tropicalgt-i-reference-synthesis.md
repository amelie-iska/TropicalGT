# TropicalGT-I Reference Synthesis and Porting Crosswalk

## Reviewed sources
- `1311.2360v3.pdf`: Brugalle and Shaw gives the didactic tropical geometry basis: max-plus algebra, tropical curves, stable intersections, patchworking, and the guiding principle that tropical objects preserve classical structure through piecewise-linear degeneration.
- `2301.12594v2-1.pdf`: continuous GFlowNets extend flow matching, detailed balance, and trajectory balance to measurable spaces; TropicalGT uses this for mixed discrete graph edits and continuous embedding-space steps.
- `2308.09687v4.pdf`: Graph of Thoughts models thoughts as arbitrary graph vertices with dependency edges and transformations such as aggregate, refine, score, and merge; TropicalGT internalizes these as latent graph-token trajectories rather than only prompt graphs.
- `2401.17123v1-1.pdf`: GraphCG learns steerable latent factors for graph DGMs via same-direction/step contrastive objectives plus orthogonality/sparsity regularization; TropicalGT uses this as a latent reasoning-coordinate auxiliary, not as a standalone generator.
- `2505.17190v2-1.pdf`: Tropical Attention supplies max-plus/min-plus projective attention, Hilbert-metric alignment, tropical circuit approximation, and transitive-closure style algorithmic reasoning.
- `2602.01992v5-1.pdf`: analogical reasoning in transformers motivates measuring whether graph-of-thought trajectory transports preserve relational structure in embedding space.
- `2604.14727v1-1.pdf`: transformer expressivity via tropical geometry motivates active cells, Newton polytopes, normal fans, and region-count diagnostics for attention.
- `1001.1554v4.pdf` and `MaclaganSturmfels.pdf`: correspondence theorems and foundational tropical geometry justify translating toric program language only where a genuine tropical fan/polyhedral complex analogue exists.
- `references/main.pdf`: TokenGT expressivity extension supplies the graph-token universality and GraphCG fan-slice theory that TropicalGT-I should rely on.
- `TropicalGT-I/assets/tropicalgt_neurips_research_paper.*`: current paper already frames TropicalGT as an auditable tropical reasoning program; this implementation pass grounds the claims in runnable code.
- `references/toricgt_paper_pg_softmoe_final.tex`: ToricGT methods to port are graph-of-thought trajectories, GFlowNet branch training, dataset schema, W&B logging discipline, artifact/reproducibility protocol, and trajectory visualization. Toric-only algebraic ideals, torus phase probes, and noncommutative toric memory are omitted unless later mapped to tropical fan or persistence diagnostics.

## Implementation decisions
- Tokenization follows TokenGT: graph token, vertex tokens, edge tokens, type identifiers, deterministic node identifiers, and endpoint ids.
- Tropical attention is a max-plus ring reduction over graph-token values with Hilbert-projective scores, support ids, and top-two margins.
- GFlowNet training is implemented as a lightweight trajectory-balance auxiliary over embedding-space graph-of-thought actions.
- GraphCG is implemented as a lightweight latent-direction contrastive regularizer over graph pooled states.
- Visualizations are Plotly HTML artifacts: 3D PCA trajectories and PCA with NLL height, with hover payloads containing filtered graph object summaries.
