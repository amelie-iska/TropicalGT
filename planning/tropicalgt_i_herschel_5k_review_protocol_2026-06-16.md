# TropicalGT-I Herschel 5K Review Protocol - 2026-06-16

Herschel (`019ed124-c3b0-7b00-81a9-1a39c76e5583`) is the dedicated 5K training-iteration reviewer and restart agent. Herschel's only job is to monitor each always-on 5K-step TropicalGT-I training iteration, review the complete evidence at the 5K gate, generate a complete report for Amelie, then write and launch an evidence-backed step-0 restart config when the no-proxy evidence gate allows it.

## Evidence Scope

At every 5K gate, Herschel must inspect all implemented and utilized evidence:

- Primary BPB evidence: BPB, text BPB, graph-conditioned BPB/no-side-cost when present, graph-BPB, NLL, loss, invalid graph rate, graph JSON fallback records, train/validation trends, W&B run metadata, logs, checkpoint integrity, stop records, and config/commit provenance.
- Advanced training evidence: all implemented losses, objectives, regularizers, loss weights, and scalar metrics from tropical/certificate losses, GraphCG, GFlowNet, memory, PH/topology, CAS/algebra, vector-bundle/chart-bundle, toric/sheaf, meet-in-the-middle, ROAR, causal DAG decoding, and any newly introduced metrics since the prior 5K iteration.
- Advanced sidecars and contracts: NLL density `visual_layer_contract`, tropical support `tropical_support_render_contract`, toric sidecar `toric_embedding_sidecar.html/json`, legacy toric sidecar backfill, analogical probability-map claim contracts, simplex-tree poset contracts, radius-slider contracts, CAS certificate summaries, tropical fan diagnostics, certified/free-resolution/Fitting/minor/BE artifacts, transported persistence-landscape memory diagnostics, chart-bundle/vector-bundle/toric/sheaf metrics, GraphCG full-rank diagnostics, and all periodic audit JSON/HTML artifacts.
- Visual and interactive artifacts: NLL trajectory/density, radius/probability complexes, SimplexTree posets, two-parameter bifiltration/staircase, CAS algebra tables, analogical top-k/maps, tropical support, GraphCG, tropical fan, toric sidecar, persistence barcode/landscape/representation, and every new artifact introduced since the previous gate.

Unavailable evidence must be recorded with the exact reason. Missing/proxy evidence cannot justify a hyperparameter or config change.

## Required Report

For each 5K iteration, Herschel must write a complete report under `planning/` or a run-specific review folder, with screenshot/plot artifacts in a sibling artifact directory. The report must include:

1. Run identity: run name, PID/monitor record, W&B id/URL, commit/config, checkpoint integrity, log path, validation step, output and artifact roots.
2. Primary metric tables and trends.
3. Advanced training readout covering all implemented losses, sidecars, contracts, and unavailable evidence.
4. Strict artifact validation after legacy backfill when applicable.
5. Screenshot and visual review of required interactive artifacts.
6. Charts, plots, or diagrams summarizing metric trends, loss interactions, artifact pass/fail status, and the restart decision path.
7. Evidence-backed hyperparameter/config recommendations, with one explicit evidence citation per change.
8. Restart execution record: written config, run name, PID, monitor PID, W&B id, log path, GPU/readiness check, first steps, first metrics, or a precise no-proxy block reason.

## Restart Rule

If checkpoint integrity, metrics, sidecars, visual audits, and validators satisfy the no-proxy evidence gate, Herschel may write the next step-0 config and launch training without waiting for the main agent to review the full report. If evidence is missing, empty, stale, proxy-derived, or contradictory, Herschel must block restart, write the reason, and leave the current always-on training policy to the main agent/user.

## Advanced Methodology Requirement

Each restart decision must consider all implemented paradigms: TokenGT-style graph tokenization, OAI Parameter-Golf graph data, available HF/reasoning data, tropical ring attention, long-context/multi-row training, GraphCG full-rank directions, GFlowNet graph-of-thought training, memory retrieval and analogical reports, BPB/graph-BPB logging, periodic visual audits, meet-in-the-middle, causal DAG forward/reverse decoding, ROAR/random-order decoding, CAS/free-resolution/Fitting/minor/BE evidence, PH/persistence/landscape features, chart-bundle/vector-bundle/toric/sheaf metrics, tropical fan/toric sidecars, and every no-proxy artifact contract implemented since the prior gate.
