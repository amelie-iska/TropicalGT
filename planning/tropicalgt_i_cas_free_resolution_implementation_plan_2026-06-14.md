# TropicalGT-I CAS Free-Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan.

**Goal:** Add real, certificate-gated multigraded free-resolution support for TropicalGT-I without presenting finite chain presentations, staircase toy ideals, or sampled rank shadows as certified resolutions.

**Architecture:** Keep Python/GUDHI responsible for filtrations, diagrams, vectorized persistence summaries, and finite chain-presentation diagnostics. Delegate real minimal free resolutions and Buchsbaum-Eisenbud grade/depth diagnostics to external CAS backends only. Surface unavailable states explicitly when no CAS certificate is present.

**Tech Stack:** Python orchestration in `TropicalGT-I/src/tropicalgt/algebra.py`, optional new CAS adapter module, Macaulay2 primary backend, SageMath and Singular fallbacks, optional `amelie-iska/BEMultipliers` Macaulay2 package for Buchsbaum-Eisenbud multipliers.

---

## Reviewed Anchors

- `TropicalGT-I/src/tropicalgt/algebra.py`: `compute_topological_algebra_report`, `compute_level_radius_bifiltration_report`, `_multigraded_free_resolution_report`, `_real_free_resolution_backend_report`, `_bemultipliers_backend_status`, `_macaulay2_style_resolution_report`, `_two_variable_staircase_report`, `_determinantal_ideal_report`, `_fitting_ideal_report`, `_buchsbaum_eisenbud_complex_report`, and `_derived_signature`.
- `TropicalGT-I/tests/test_algebraic_persistence.py`: current tests already assert unavailable real-resolution state, backend labels, and the scoped staircase quotient distinction.
- `planning/FILTRATION.md`: existing bifiltration contract for `K(i,r)`, `F2[x_level,x_radius]`, generator bidegrees, boundary monomials, Fitting ideals, BE diagnostics, and derived-category limits.
- `planning/tropicalgt_i_visual_math_repair_plan_2026-06-12.md`: latest checklist requires real CAS methodology and explicitly warns that BEMultipliers is optional BE support, not a certified resolution substitute.

## Current Evidence Snapshot

- The current algebra path emits a finite two-parameter chain presentation over `F2[x_level,x_radius]`; it is useful input data, not a real free resolution.
- `_multigraded_free_resolution_report` correctly keeps `not_a_free_resolution=True`, `resolution_status="chain_presentation_only"`, and `certificate_attached=False`.
- `_real_free_resolution_backend_report` already has the intended safety shell: schema `tropicalgt.real_free_resolution.v1`, `status="unavailable_no_certificate"`, backend probe metadata, empty `cas_artifacts`, and `safe_to_render_as_real_free_resolution=False`.
- Current tests require unavailable-state guardrails and verify that `Macaulay2_M2`, `SageMath_sage`, `Singular`, and `BEMultipliers` appear as allowed/probed backends.

## Current Remote Backend Probe Results

Probe command used from `/home/iska/Documents/amelie/bio/TropicalGT`:

```bash
for c in M2 sage Singular python3 git curl; do command -v "$c" || true; done
/home/iska/miniconda3/envs/tokengt/bin/python -c 'import importlib.util; [print(n, importlib.util.find_spec(n) is not None) for n in ["sageall", "sage", "sageconf", "BEMultipliers", "bemultipliers", "multipers", "gudhi"]]'
find . -maxdepth 5 \( -iname "*BEMultipliers*" -o -iname "*bemultipliers*" \) -print
```

Observed status:

- `M2`: unavailable.
- `sage`: unavailable.
- `Singular`: unavailable.
- `multipers`: unavailable.
- `gudhi`: available in the `tokengt` Python environment.
- Local BEMultipliers package/module: unavailable.

BEMultipliers repository probe:

```bash
git ls-remote --heads https://github.com/amelie-iska/BEMultipliers.git
curl -fsSL https://api.github.com/repos/amelie-iska/BEMultipliers
curl -fsSL https://api.github.com/repos/amelie-iska/BEMultipliers/git/trees/master?recursive=1
curl -fsSL https://raw.githubusercontent.com/amelie-iska/BEMultipliers/master/BuchsbaumEisenbudMultipliers.m2 | sed -n '1,180p'
```

Observed status:

- Default branch: `master`.
- Description: `Buchsbaum-Eisenbud multipliers in Macaulay2`.
- Main files: `BuchsbaumEisenbudMultipliers.m2`, `example.m2`, `tex/multipliers.tex`.
- Package exports include `aMultiplier`, `cMultiplier`, `ComputeRanks`, and `exteriorDuality`.
- The package comments state that no safety checks are implemented, so TropicalGT-I must independently certify the input resolution and validate any BE output before rendering it as evidence.

## Real-Only Resolution Policy

1. A real free resolution is available only when a CAS backend returns a machine-readable certificate with all of these true:
   - `certificate_attached`
   - `real_free_resolution_certified`
   - `exactness_certified`
   - `minimality_certified`, when the UI labels the object as minimal
   - `safe_to_render_as_real_free_resolution`
2. Python-native finite chain complexes, rank shadows, sampled minors, staircase renderings, and Macaulay2-style text are diagnostics only. They must not set `real_free_resolution_certified=True`.
3. A scoped monomial staircase quotient such as `S/I` may have a real exact resolution only for that named ideal. It must not be promoted to a full persistence-module resolution unless the full module presentation was passed to CAS and certified.
4. BEMultipliers is not a resolution backend by itself. It consumes a Macaulay2 `ChainComplex` and may provide BE multiplier diagnostics after Macaulay2 has already built and certified the resolution.
5. When no backend emits a certificate, the status must be `unavailable_no_certificate`, `cas_artifacts` must remain empty, and rendering must show unavailable state rather than fabricated empty Betti tables or fake BE claims.

## CAS Backend Order

1. **Macaulay2 executable `M2` - primary backend.**
   - Best fit for multigraded modules over `GF(2)[x_level,x_radius]`.
   - Required operations: `map`, `coker`, `res`, `syz`, `betti`, `mingens`, `fittingIdeal`, `minors`, `codim`, `depth`, and homogeneity checks.
   - Optional packages: `KustinMiller` where applicable, and `BuchsbaumEisenbudMultipliers` for multiplier diagnostics.
2. **SageMath executable or Python modules - fallback backend.**
   - Acceptable for polynomial rings, ideals, Groebner bases, syzygies, and Singular/Macaulay2 bridges.
   - May certify only if it returns complete free modules, differentials, exactness evidence, minimality evidence, and a backend version.
   - If Sage calls another CAS, record both Sage and the underlying backend in the certificate.
3. **Singular executable - fallback backend.**
   - Acceptable for ideals, modules, syzygies, standard bases, and resolutions when exact commands pass smoke tests.
   - Parser must preserve grading, module ranks, differentials, and ideal generators. If grading cannot be reconstructed, mark `unsupported_ring` or `parse_error`.
4. **BEMultipliers - optional Macaulay2 package only.**
   - Use after a certified Macaulay2 `ChainComplex` exists.
   - Do not clone into the repo automatically. Prefer an explicit path such as `TROPICALGT_BEMULTIPLIERS_M2_PATH=/path/to/BuchsbaumEisenbudMultipliers.m2` or a documented install step outside generated artifact directories.

## Exact Backend Probes And Smoke Commands

All commands should run from a temporary directory, not under checkpoints, caches, W&B, data, or generated artifact trees.

### Macaulay2 Probe

```bash
command -v M2
M2 --version
cat > /tmp/tropicalgt_m2_probe.m2 <<'M2'
kk = GF(2)
S = kk[x_level, x_radius, MonomialOrder=>GRevLex]
phi = map(S^1, S^2, matrix{{x_level^2, x_radius^2}})
M = coker phi
R = res M
print "TROPICALGT_M2_PROBE_OK"
print betti R
print fittingIdeal(0, M)
print minors(1, phi)
exit 0
M2
M2 --script /tmp/tropicalgt_m2_probe.m2
```

Pass condition: command exists, script exits 0, and `res M`, `betti R`, `fittingIdeal`, and `minors` produce parseable output.

### Sage Probe

```bash
command -v sage
sage --version
sage -python - <<'PY'
from sageall import GF, PolynomialRing
R = PolynomialRing(GF(2), ("x_level", "x_radius"), order="degrevlex")
x_level, x_radius = R.gens()
I = R.ideal([x_level**2, x_radius**2])
print("TROPICALGT_SAGE_PROBE_OK")
print(I.groebner_basis())
PY
```

Pass condition: Sage can construct `GF(2)[x_level,x_radius]`, build ideals, compute a Groebner basis, and expose enough module or bridge APIs for certified resolution extraction.

### Singular Probe

```bash
command -v Singular
Singular -v
cat > /tmp/tropicalgt_singular_probe.sing <<'SING'
ring r = 2,(x_level,x_radius),dp;
ideal I = x_level^2, x_radius^2;
resolution R = mres(I,0);
print("TROPICALGT_SINGULAR_PROBE_OK");
print(betti(R));
quit;
SING
Singular -q /tmp/tropicalgt_singular_probe.sing
```

Pass condition: Singular can construct the ring and compute a resolution for a small ideal. If the local Singular build needs a different library or syntax, the backend remains unavailable until the implementation records the working local command in tests.

### BEMultipliers Probe

```bash
git ls-remote --heads https://github.com/amelie-iska/BEMultipliers.git
curl -fsSL https://raw.githubusercontent.com/amelie-iska/BEMultipliers/master/BuchsbaumEisenbudMultipliers.m2 -o /tmp/BuchsbaumEisenbudMultipliers.m2
cat > /tmp/tropicalgt_bem_probe.m2 <<'M2'
needs "/tmp/BuchsbaumEisenbudMultipliers.m2"
A = QQ[x,y,z]
K = koszul vars A
print "TROPICALGT_BEMULTIPLIERS_PROBE_OK"
print aMultiplier(1,K,ComputeRanks=>true)
exit 0
M2
M2 --script /tmp/tropicalgt_bem_probe.m2
```

Pass condition: Macaulay2 exists, the package loads, and `aMultiplier` runs on a known Macaulay2 `ChainComplex`. This does not certify TropicalGT-I persistence modules by itself.

## Module Data Schema For `F2[x_level,x_radius]`

The CAS input should be serialized as a stable, versioned schema before any backend-specific script is generated.

```json
{
  "schema_version": "tropicalgt.level_radius_module.v1",
  "coefficient_field": "F2",
  "polynomial_ring": "F2[x_level,x_radius]",
  "variables": ["x_level", "x_radius"],
  "grading": "N2",
  "levels": [0, 1, 2],
  "radii": [0.0, 0.5, 1.0],
  "radius_grade_values": {"0": 0.0, "1": 0.5, "2": 1.0},
  "radius_grade_policy": "sorted_unique_radius_grid_index",
  "fiber_rank_profile": [],
  "rank_invariant_samples": [],
  "chain_module_generators": [
    {
      "generator_id": "c0_s0",
      "homological_degree": 0,
      "simplex": [0],
      "multidegree": [0, 0],
      "filtration_value": 0.0,
      "radius_value": 0.0,
      "source": "observed"
    }
  ],
  "boundary_monomials": {
    "d1": [
      {
        "source_generator_id": "c1_s0_1",
        "target_generator_id": "c0_s0",
        "source_simplex": [0, 1],
        "target_face": [0],
        "source_multidegree": [1, 1],
        "target_multidegree": [0, 1],
        "monomial_exponent": [1, 0],
        "coefficient": 1,
        "monomial": "x_level"
      }
    ]
  },
  "presentation_matrix": {
    "target_generators": ["c0_s0"],
    "source_generators": ["c1_s0_1"],
    "entries": [
      {"row": 0, "col": 0, "coefficient": 1, "monomial_exponent": [1, 0]}
    ]
  },
  "real_free_resolution": {
    "schema_version": "tropicalgt.real_free_resolution.v1",
    "status": "unavailable_no_certificate",
    "available": false,
    "certificate_attached": false,
    "safe_to_render_as_real_free_resolution": false,
    "cas_artifacts": {}
  }
}
```

Implementation notes:

- Add stable `generator_id` values before CAS serialization. Do not require UI-only simplex labels to serve as backend keys.
- Preserve current display fields for compatibility, but add backend-safe IDs and matrix rows/columns.
- Use nonnegative integer bidegrees only. If a boundary term would require a negative exponent, fail the CAS export with `invalid_grading` and render unavailable.
- Hash the serialized schema and include `input_sha256` in all CAS outputs.

## Real Free Resolution Output Contract

A certified result should extend the existing `tropicalgt.real_free_resolution.v1` shape:

```json
{
  "schema_version": "tropicalgt.real_free_resolution.v1",
  "status": "certified",
  "available": true,
  "backend": "Macaulay2_M2",
  "backend_version": "Macaulay2 1.x",
  "command": ["M2", "--script", "/tmp/tropicalgt_resolution.m2"],
  "input_sha256": "...",
  "certificate_attached": true,
  "real_free_resolution_certified": true,
  "minimality_certified": true,
  "exactness_certified": true,
  "safe_to_render_as_real_free_resolution": true,
  "free_modules": [
    {"index": 0, "rank": 3, "degree_shifts": [[0, 0], [1, 0], [0, 1]]}
  ],
  "differentials": [
    {"index": 1, "source_rank": 2, "target_rank": 3, "entries": []}
  ],
  "betti_table_rows": [],
  "fitting_ideals": {},
  "determinantal_ideals": {},
  "buchsbaum_eisenbud": {},
  "checks": {
    "homogeneous_presentation": true,
    "d_squared_zero": true,
    "cas_resolution_object": true,
    "minimality_test": true,
    "exactness_test": true
  },
  "cas_artifacts": {
    "stdout_excerpt": [],
    "stderr_excerpt": [],
    "script_sha256": "..."
  }
}
```

Allowed unavailable statuses:

- `unavailable_no_certificate`
- `backend_not_installed`
- `backend_error`
- `timeout`
- `unsupported_ring`
- `invalid_grading`
- `parse_error`
- `certificate_failed`

## Fitting Ideals, Minors, And BE Diagnostics

1. For a differential or presentation map `phi: F_source -> F_target`, record full CAS ideals rather than bounded Python samples when certified.
2. Keep the current convention: `Fitt_j(coker(phi)) = I_{rank(F_target)-j}(phi)`.
3. Store determinantal ideals by matrix id and minor size:
   - `matrix_id`: for example `presentation_phi` or `d_2`.
   - `minor_size`: integer.
   - `ideal_generators`: CAS-normalized polynomial strings.
   - `codimension` and `depth` when available.
4. Buchsbaum-Eisenbud diagnostics require symbolic grade/depth or codimension evidence for the determinantal ideals. Finite F2 rank checks are useful but must remain a shadow diagnostic.
5. BEMultipliers output may be stored under `buchsbaum_eisenbud.multipliers` only when:
   - the input is a certified CAS `ChainComplex`,
   - the BEMultipliers package loads from an explicit path or installed M2 package,
   - TropicalGT-I records that BEMultipliers itself has no internal safety checks,
   - independent exactness/minimality checks already passed.

## Derived-Category Comparison Limits

- The full level-radius persistence object may be described as a finite multigraded module presentation only after CAS export succeeds.
- A certified free resolution can support invariants of an object in `D^b(gr-F2[x_level,x_radius])`, but it does not prove derived equivalence between trajectories.
- A derived-equivalence or derived-geometric-realization label requires additional evidence: certified resolutions for both objects, a filtered chain map or module map, compatibility with differentials, and an explicit equivalence criterion.
- Current analogical retrieval should keep using conservative labels such as `finite_filtered_chain_presentation_similarity`, `rank_invariant_similarity`, and `persistence_summary_similarity` unless a certificate is attached.
- Deprecated `free_resolution_similarity` should either remain an alias to chain-presentation similarity with a clear deprecation note or be gated to zero/none when real resolutions are unavailable.
- If any component needed for a derived claim is unavailable, the combined derived-algebraic score must not imply high derived similarity. Use a gated minimum or render `derived_claim_unavailable`.

## Unavailable-State Rendering Rules

1. Page titles and section labels must say `Real free resolution unavailable: no CAS certificate attached` when status is unavailable.
2. Diagnostic sections may render finite chain modules, boundary monomial matrices, rank invariant samples, sampled minors, Fitting ideal conventions, and scoped staircase quotient resolutions.
3. Diagnostic sections must not be titled `minimal free resolution`, `Macaulay2 resolution`, `derived equivalence`, or `Buchsbaum-Eisenbud certified` unless the real-resolution contract says safe.
4. CAS artifacts should be empty on unavailable status. Do not fabricate empty Betti tables as if the backend ran.
5. Show backend probe status and exact commands in collapsed details so a reviewer can reproduce why the resolution is unavailable.
6. Use badges such as `chain-presentation diagnostic`, `scoped monomial ideal resolution`, `CAS unavailable`, and `real free resolution certified`.
7. Validation scripts should fail if `safe_to_render_as_real_free_resolution=True` while `certificate_attached=False`.

## Implementation Steps

### Step 1: Preserve Existing Guardrails

Files:

- `TropicalGT-I/src/tropicalgt/algebra.py`
- `TropicalGT-I/tests/test_algebraic_persistence.py`
- `TropicalGT-I/scripts/validate_interactive_audit_artifacts.py`

Actions:

- Keep current unavailable defaults and tests intact.
- Add tests that every unavailable backend state keeps `cas_artifacts` empty and rendering-safety flags false.
- Add tests that BEMultipliers cannot set `real_free_resolution_certified=True` without Macaulay2 resolution evidence.

### Step 2: Add A CAS Adapter Boundary

Preferred file:

- `TropicalGT-I/src/tropicalgt/cas_free_resolution.py`

Actions:

- Define dataclasses or plain dict builders for the module input schema and real-resolution output schema.
- Implement backend probes as read-only subprocess calls with timeouts.
- Write temporary scripts under `tempfile.TemporaryDirectory()` only.
- Never write under data, checkpoints, W&B, outputs, or caches.

### Step 3: Serialize Level-Radius Modules

Files:

- `TropicalGT-I/src/tropicalgt/algebra.py`
- `TropicalGT-I/tests/test_algebraic_persistence.py`

Actions:

- Add stable generator IDs and matrix row/column IDs to the existing level-radius report.
- Validate all monomial exponents are nonnegative.
- Compute `input_sha256` from canonical JSON.
- Keep old fields for existing visualization/tests.

### Step 4: Implement Macaulay2 Backend First

Files:

- `TropicalGT-I/src/tropicalgt/cas_free_resolution.py`
- `TropicalGT-I/tests/test_cas_free_resolution_contract.py`

Actions:

- Generate a script that constructs `S = GF(2)[x_level,x_radius]`.
- Build `phi = map(S^target_rank, S^source_rank, matrix{...})` from `presentation_matrix`.
- Build `M = coker phi` and `R = res M`.
- Extract Betti table, free-module shifts, differential entries, `fittingIdeal`, `minors`, `codim`, and `depth`.
- Emit a single parseable JSON block or strict line protocol. If Macaulay2 JSON output is brittle, use a custom tagged protocol with schema version and hashes.
- Mark certified only if script succeeds, parser succeeds, homogeneity checks pass, and exactness/minimality checks pass.

### Step 5: Add Sage And Singular Fallbacks Only After M2 Contract Is Stable

Actions:

- Implement fallback probes first, not full certification.
- Promote a fallback to certified only when it can emit the same output contract as M2.
- If a fallback cannot preserve multigrading or exact differentials, return `unsupported_ring` or `parse_error`.

### Step 6: Integrate BEMultipliers As Optional BE Diagnostics

Actions:

- Require Macaulay2 availability and a certified Macaulay2 `ChainComplex`.
- Load from configured path or installed package only.
- Record package source URL, commit if known, and file hash.
- Store multiplier outputs separately from minimal-free-resolution certificate fields.
- Do not use BEMultipliers to replace `res`, `syz`, `betti`, exactness checks, or minimality checks.

### Step 7: Update Visualization Labels And Validators

Files:

- `TropicalGT-I/src/tropicalgt/visualization.py`
- `TropicalGT-I/scripts/validate_interactive_audit_artifacts.py`

Actions:

- Render certified free resolutions only when `safe_to_render_as_real_free_resolution=True`.
- Rename current diagnostic pages that show chain presentations or scoped staircase ideals.
- Add validator checks that reject fake labels in generated HTML payloads.

## Test Plan

Run after implementation, not for this planning-only change:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_cas_free_resolution_contract.py -q
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/validate_interactive_audit_artifacts.py TropicalGT-I/outputs/multi_sample_browser/vector_persistence_latest/sample_000
```

Backend-dependent tests should skip when `M2`, `sage`, or `Singular` are absent, while contract tests should still assert unavailable states.

## Open Risks

- Macaulay2 may not emit convenient JSON, so parser design must be tested on small modules before wiring it into browser artifacts.
- Singular resolution syntax and grading support need a live installation before certification can be promised.
- Sage fallback may silently bridge to Singular or Macaulay2; the output must record both wrapper and underlying backend.
- BEMultipliers has no safety checks, so it is useful as optional evidence but not as the trust root.
- Full derived-category comparisons require more than certified resolutions; chain maps and equivalence criteria remain separate future work.

## Completion Criteria

- No UI or artifact can call a finite chain presentation a real free resolution.
- No real free-resolution artifact renders unless a CAS certificate is attached.
- Macaulay2 is the first certified backend implemented.
- Sage and Singular remain unavailable/fallback until they meet the same schema.
- BEMultipliers is documented and implemented only as optional Macaulay2 BE diagnostics.
- Tests cover both unavailable states and at least one certified smoke fixture in an environment with CAS installed.
