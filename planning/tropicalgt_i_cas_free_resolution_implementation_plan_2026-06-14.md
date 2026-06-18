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

## Implementation Status - 2026-06-14 Sequential CAS Pass

- Added `TropicalGT-I/src/tropicalgt/cas_free_resolution.py` as the real-only CAS adapter boundary.
- The adapter canonicalizes finite `F2[x_level,x_radius]`, `F2[x_filtration,x_dimension]`, and `F2[x_filtration,x_dimension,x_position]` chain-presentation data into stable backend-safe generator IDs, multidegrees, monomial boundary entries, a degree-one presentation matrix, and an `input_sha256` hash.
- The adapter probes Macaulay2 (`M2`), Singular, Sage, and optional BEMultipliers availability, but reports `cas_artifacts: {}` and all certificate flags false unless an external CAS emits a tagged exactness certificate.
- Integrated `_real_free_resolution_backend_report` with the adapter, replacing the old placeholder artifact arrays. This means missing CAS support is now rendered as unavailable, not as an empty Betti table or differential matrix that could be mistaken for a resolution.
- Updated focused algebra tests so unavailable real-resolution states require `cas_artifacts == {}` and `safe_to_render_as_real_free_resolution == False`.
- Verified in the remote `tokengt` environment with:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/src/tropicalgt/algebra.py
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
```

The current remote machine still has no `M2`, `Singular`, or `sage` executable, so real minimal free resolutions remain unavailable until a CAS backend is installed or bridged. This is intentional: only real CAS-certified resolutions may populate `cas_artifacts`.

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
- At least one certified backend is implemented and tested, with Macaulay2 still preferred when available for richer graded-resolution and Buchsbaum-Eisenbud workflows.
- Sage and additional Singular/Macaulay2 paths remain unavailable until they meet the same schema.
- BEMultipliers is documented and implemented only as optional Macaulay2 BE diagnostics.
- Tests cover both unavailable states and at least one certified smoke fixture in an environment with CAS installed.

## 2026-06-14 Implementation Checkpoint

Completed in this checkpoint:

- Added `TropicalGT-I/src/tropicalgt/cas_free_resolution.py` as the real-only CAS adapter boundary. It canonicalizes finite multigraded module presentations, records stable generator and matrix ids, computes a canonical `input_sha256`, and probes Macaulay2 (`M2`), Singular, Sage, and Python Sage/BEMultipliers import availability without inventing algebraic output.
- Wired `TropicalGT-I/src/tropicalgt/algebra.py` so `_real_free_resolution_backend_report()` delegates to the CAS adapter and keeps the hard rendering contract: a page may render a real free resolution only when `available`, `certificate_attached`, `real_free_resolution_certified`, `exactness_certified`, and `safe_to_render_as_real_free_resolution` are all true. Minimality additionally requires `minimality_certified`.
- Tightened `TropicalGT-I/src/tropicalgt/visualization.py` so missing certified/scoped resolution rows no longer fall back to finite chain-presentation rows inside Macaulay2-style Betti, free-module, or differential tables. The rendered state must be explicitly unavailable unless a certified CAS output or scoped monomial-ideal resolution is present.
- Added focused tests in `TropicalGT-I/tests/test_algebraic_persistence.py` covering canonical adapter schema, unavailable backend states, empty CAS artifacts without certificates, and non-certified BEMultipliers import behavior.
- Verified on the remote `tokengt` environment with `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`6 passed`) and `py_compile` for `cas_free_resolution.py`, `algebra.py`, and `visualization.py`.

Checkpoint backend reality before the sequential Singular pass:

- `M2`, `Singular`, and `sage` were not installed on the remote PATH at this earlier checkpoint, so the adapter correctly rendered full CAS-certified minimal free resolutions as unavailable at that time.
- BEMultipliers is allowed only as optional Buchsbaum-Eisenbud diagnostic evidence after a trusted CAS resolution exists. It cannot certify exactness, minimality, or derived equivalence by itself.
- The next CAS item is to install or bridge a real backend, starting with Macaulay2 if available for the platform, and then add one certified smoke fixture whose differential matrices, multidegree shifts, Betti table, Fitting ideals, minors, and exactness/minimality checks come from the backend.

## 2026-06-14 Sequential CAS Update: Singular Backend Installed And Bridged

Status: complete for the first real CAS bridge checkpoint.

- Installed Singular 4.4.1 in the isolated conda environment `/home/iska/miniconda3/envs/tropicalgt-cas` so the active `tokengt` training environment remains untouched.
- Added deterministic executable discovery for `TROPICALGT_SINGULAR_BIN`, `PATH`, and `/home/iska/miniconda3/envs/tropicalgt-cas/bin/Singular`.
- Replaced the invalid Singular module assignment syntax with a true Singular module literal such as `[x_level,0],[0,x_radius]` built from the finite presentation matrix over `F2[x_level,x_radius]` or the supported multigraded rings.
- The certified Singular path computes `mres(M,0)` for the image submodule `M` of the presentation matrix and records the cokernel-resolution interpretation explicitly: TropicalGT prepends the displayed free module `F0 -> coker(M)` to the Singular image-submodule resolution.
- Exactness is marked certified only when Singular runs successfully and emits tagged output. Minimality is marked separately and is false if the presentation contains unit entries; this prevents a nonminimal presentation from being mislabeled as a minimal free resolution.
- Certified outputs now include `backend_probe`, `bemultipliers_probe`, command templates, tagged raw output, Singular Betti text, and Singular resolution text. Uncertified paths still render with empty `cas_artifacts`.
- Added a focused smoke test for a real `F2[x_level,x_radius]` CAS free-resolution computation. The algebra suite now accepts both unavailable environments and certified Singular/Macaulay2/Sage environments.

Verification:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 7 passed in 1.48s
```

Remaining CAS items:

1. Parse Singular Betti matrices into structured multigraded Betti-table rows suitable for the research-style figures.
2. Add Macaulay2 when available, because it remains the preferred backend for minimal graded free resolutions, Fitting ideals, minors, and Buchsbaum-Eisenbud diagnostics.
3. Add optional Sage bridge only if it records the underlying backend and returns the same certificate fields.
4. Clone and wire `amelie-iska/BEMultipliers.git` only as a Buchsbaum-Eisenbud diagnostic layer after a certified resolution exists; it is not a substitute for a resolution backend.


## 2026-06-14 Sequential CAS Update: BEMultipliers Cloned And Inspected

Status: complete for repository inspection; not executable as a backend on this machine until Macaulay2 is installed.

- Cloned `https://github.com/amelie-iska/BEMultipliers.git` into `external/BEMultipliers` for local inspection. The external directory is gitignored, so this checkpoint records the inspected package metadata instead of staging the cloned source tree.
- Inspected commit `d0b55d7c2cb879acc27df533d0117c98a98e463d`. The main package file is `BuchsbaumEisenbudMultipliers.m2` with SHA256 `562d3c2879e6a306a2b39ff1c9ad7b23689d953757f498abe294161a3eac00b7`.
- The package exports `aMultiplier`, `cMultiplier`, `ComputeRanks`, and `exteriorDuality`. Its method signatures consume Macaulay2 `ChainComplex` objects, for example `aMultiplier(ZZ, ChainComplex)` and `cMultiplier(ZZ, ZZ, ChainComplex)`.
- The package source explicitly warns that no safety checks are implemented. TropicalGT-I must therefore perform its own input checks, exactness checks, minimality checks, ring/degree checks, and output validation before rendering Buchsbaum-Eisenbud multiplier information as evidence.
- Since `M2`/Macaulay2 is currently unavailable on the remote machine, BEMultipliers cannot be loaded or smoke-tested yet. It remains an optional diagnostic layer after a certified Macaulay2 resolution exists; it is not a substitute for `res`, `syz`, `betti`, Fitting ideals, minors, or derived-equivalence evidence.

Next CAS item after this checkpoint:

1. Provision or bridge Macaulay2 so the existing BEMultipliers package can be loaded against a known certified `ChainComplex`.
2. Add a guarded Macaulay2 probe that records package source path, commit/hash, Macaulay2 version, package load status, and a small `koszul vars A` smoke result.
3. Keep BE multiplier outputs in a separate `buchsbaum_eisenbud_diagnostics` block and refuse to set `real_free_resolution_certified`, `exactness_certified`, or `minimality_certified` from BEMultipliers alone.

## Added CAS/Tropical Survey Targets

1. Macaulay2 `Tropical`: use tropical cycles/fans for audited tropical geometry diagnostics: `tropicalVariety`, `BergmanFan`, `fan`, `rays`, `cones`, `maxCones`, `isBalanced`, `isPure`, `isSimplicial`, `stableIntersection`, and `isTropicalBasis`.
2. Sage tropical varieties and tropical multivariate polynomials: use Newton-polytope and tropical-polynomial arithmetic to build exact model-derived tropicalization checks and scoped finite toric sidecar diagnostics.
3. `1710.10651v2.pdf`: review before adding any training theorem or implementation hook tied to toric/tropical embeddings.
4. Maclagan-style toric embeddings: investigate certified tropical-variety embeddings into toric varieties only from finite monomial coordinates, Newton polytopes, fan data, and one dimensional cones with tool-backed evidence; distinguish those certified constructions from scoped exponent-chart visualization/probe diagnostics.

_Last updated: 2026-06-14T15:27:54+00:00_

## Real Implementations Only Policy

No TropicalGT-I metric, loss, visualization, analogical map, persistence module, free resolution, derived comparison, tropical-cycle diagnostic, or CAS artifact should be presented as a mathematical object unless it is computed from the actual model outputs, graph states, embeddings, probabilities, simplex trees, bifiltrations, or certified CAS/backend output that define that object. Temporary placeholders, synthetic fallback objects, mock charts, fabricated simplices, and convenience stand-ins are not acceptable. When a requested object cannot yet be computed, the artifact must render an explicit unavailable/uncertified state and the training metric must either be disabled or logged under an audit-only unavailable flag. Finite chain-presentation diagnostics may be shown only as chain diagnostics, never as free resolutions. Total-graded or ungraded CAS output may be shown as real CAS output only under its actual grading; it must not be advertised as a multigraded `F2[x_level,x_radius]` free resolution unless the backend certifies that multigraded structure.

Use "one dimensional cone" or "one dimensional cones" as the preferred fan-theoretic language whenever the intended object is a cone of a fan or a cone-indexed filtration datum. Use singular or plural according to ordinary grammar.

_Last updated: 2026-06-14T15:34:45+00:00_


## 2026-06-14 CAS Backend Update: Generator-ID Boundaries And Existing Sage/Singular Environments

Sequential CAS item completed in the no-proxy lane:

- Confirmed the repository is on `tropicalgt-i-real-cas-no-proxy-20260614`; local and remote branch sets contain `main`, `tropicalgt-i-implementation`, and `tropicalgt-i-real-cas-no-proxy-20260614`.
- Confirmed active training run `tropicalgt_i_pg_bpb_step0_full24b_b55_v11_bpb_5k_gate` is alive under PID `2448854` during this pass.
- Backend inventory on the remote machine:
  - `M2`: not installed in PATH or `/home/iska/miniconda3/envs/tropicalgt-cas/bin`.
  - Sage: usable through `/home/iska/miniconda3/envs/tropicalgt-sage/bin/python` with `sage.all`; the packaged `sage` command exists but does not support the `--python` invocation used by the first probe.
  - Singular: usable through `/home/iska/miniconda3/envs/tropicalgt-cas/bin/Singular` and `/home/iska/miniconda3/envs/tropicalgt-sage/bin/Singular`.
  - GUDHI: available in `tokengt`; `multipers`: unavailable.
- Fixed a real CAS adapter bug: direct `source_generator_id` / `target_generator_id` boundary rows with an `exponent` field were previously canonicalized as empty simplices with exponent `[0,0]`. They now preserve generator IDs and monomial exponents in the presentation matrix.
- Added Sage discovery through `/home/iska/miniconda3/envs/tropicalgt-sage/bin/python`, so real total-graded Sage output can be consumed when applicable.
- Added correct Macaulay2 invocation shape (`M2 --script <file>`) for when `M2` becomes available.
- Tightened Singular parsing: a Singular exactness run with no nonzero parsed free modules is not renderable as a resolution; flags remain false and the report explains that no renderable free-resolution summary was parsed.
- Removed remaining active `free_resolution` / `free_resolution_similarity` aliases from chain-presentation diagnostics and analogical comparison output. Chain diagnostics now stay under chain-presentation names unless a CAS certificate exists.

Validated with:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_metric_provenance.py -q
# 15 passed
```

Live CAS smoke after the fix:

- Input module over `F2[x_level,x_radius]` with one degree-zero generator and two degree-one generators mapped by `x_level` and `x_radius`.
- Sage returned a real total-graded free resolution with ranks `[1,2,1]`.
- The report sets `total_graded_resolution_certified=True` and `safe_to_render_as_total_graded_resolution=True`.
- The report keeps `multigraded_free_resolution_certified=False`, `safe_to_render_as_multigraded_free_resolution=False`, and `safe_to_render_as_real_free_resolution=False` for the requested persistence-module interpretation.
- This is the intended distinction: real total-graded CAS evidence is allowed under its actual grading, but it is not a multigraded `F2[x_level,x_radius]` persistence-module resolution.

Next linear CAS item: implement Macaulay2/Sage/Singular multigraded extraction only when the backend can emit multidegree shifts, differential matrices, and an exactness/minimality certificate. If no backend emits that, the artifact remains unavailable rather than approximated.

## 2026-06-14 Active Metric Name Cleanup

Sequential no-proxy item completed after the CAS generator-boundary fix:

- Renamed the exact BPB compatibility field from `bpb_proxy` to `bpb_exact`; it is now an exact alias/name for the leaderboard BPB computation, not a proxy-labelled metric.
- Renamed GraphCG spectral diagnostics from `graphcg_direction_*_condition_proxy` to `graphcg_direction_*_condition_number`; these are direct condition-number diagnostics computed from Gram/SVD spectra, not proxy objectives.
- Updated diagnostics, training metric emission, visualization metric priority lists, and tests to use the clean names.
- Kept the provenance scanner able to detect stale retired labels, but active training/eval metric outputs no longer emit those proxy-named fields.
- Validation: `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_metric_provenance.py TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py -q` returned `30 passed`.

## 2026-06-14 Cleanup And Exact Bivariate Toric Certificate
- Cleaned generated artifacts without touching source, data, secrets, or the active b55_v11 training run: stale outputs/smoke runs/checkpoints/local W&B runs removed; latest audit payload JSONs compressed and catalog links rewritten.
- Current generated footprint after cleanup: `TropicalGT-I/outputs` 784MB, `TropicalGT-I/checkpoints` 4KB, local `wandb` 4.4MB, no generated files over 100MB.
- Tightened the exact bivariate monomial staircase certificate: the scoped real resolution now records the coordinate exponent semigroup chart `Spec F2[x_level,x_radius]`, the semigroup `N^2`, and the coordinate one dimensional cone(s) that generate the chart.
- The certificate remains deliberately scoped: it is a real Hilbert-Burch/Miller-Sturmfels minimal free resolution for the named two-variable staircase monomial ideal, not a fake full persistence-module free resolution.
- Regression coverage: `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` passed with `9 passed`.

## 2026-06-14 Bifiltration Module Figure Repair

Sequential visualization/CAS artifact item completed for the Miller-Sturmfels staircase requirement:

- Reoriented the primary 2-parameter module figure so the horizontal axis is the `x_radius` exponent/radius grade and the vertical axis is the `x_level` exponent/reasoning growth level. This matches the bivariate monomial-ideal staircase convention from Miller-Sturmfels: lattice columns are radius grades, rows are reasoning levels, and antichain/staircase corners are plotted in the exponent plane.
- Preserved the third dimension in the companion 3D panel as the actual fiber rank `beta_i = dim_F2 H_i(K_(level,radius))`; small H0/H1 display offsets are visual separation only and do not replace the fiber-rank height.
- Kept adjacent structure maps as persisted real homology maps over `F2`, computed by the rank formula `rank(B_target + image(Z_source)) - rank(B_target)` for source-to-target inclusions in the `x_level` and `x_radius` directions.
- The figure now says explicitly that chain-presentation diagnostics are not substituted for free resolutions. Certified multigraded free resolutions remain CAS-gated; exact bivariate staircase certificates remain scoped to the named monomial ideal.
- Browser QA opened the regenerated page at `127.0.0.1:8991/sample_000/trajectory_persistence/two_parameter_bifiltration.html` and confirmed the visible labels and DOM text describe horizontal `x_radius`, vertical `x_level`, adjacent structure maps, and no fake free-resolution substitution.
- Regression coverage: `PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` returned `9 passed`.

Next linear CAS item: review `references/2210.11433v1.pdf` in full and implement the paper-derived CAS objects only when Macaulay2/Sage/Singular can certify the requested multigraded data: minimal free resolutions, differential matrices, Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, and derived/chain-map evidence.


## 2026-06-15 Buchsbaum-Eisenbud Diagnostic Contract Pass

Sequential CAS item completed in the no-proxy lane:

- Macaulay2, Singular, and Sage script templates now emit a tagged `buchsbaum_eisenbud_diagnostics` section whenever they emit a certified successful CAS result.
- The tagged section records whether backend diagnostics are available, the exactness/minimality source, BEMultipliers repository/path/status, and whether multiplier output is actually present.
- BEMultipliers remains optional and is not treated as a free-resolution backend. If no explicit multiplier output is emitted, the result says `multiplier_output_available=false` and explains that no multiplier data is substituted from chain diagnostics, Fitting ideals, minors, or rank tables.
- Certified-result parsing now stores this payload in `cas_artifacts["buchsbaum_eisenbud_diagnostics"]`, so visualization and downstream analogy code can distinguish real CAS output, total/ungraded output, unavailable multiplier output, and actual future BEMultipliers output.
- Added regression coverage using a synthetic tagged Macaulay2 certificate containing multigraded shifts, differential metadata, Fitting ideals, minors, and a BE diagnostic block.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 15 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 32 passed
```

Next linear CAS item: run the backend smoke path against any installed Macaulay2/Sage/Singular executable and add small-known-module tests for real backend Fitting/minor output where the local backend is available; if Macaulay2 remains absent, continue rendering the exact unavailable state with command templates and certificate requirements.


## 2026-06-15 Bounded BEMultipliers Execution Pass

Follow-up CAS item completed after the diagnostic contract pass:

- The remote backend inventory now includes working Macaulay2, Singular, Sage Python, and a local BEMultipliers Macaulay2 package path.
- Macaulay2 script generation now performs a bounded post-certificate BEMultipliers call for small certified resolutions: `aMultiplier(1,C,ComputeRanks=>true)` runs only after `C = res N` has an exactness certificate and only when presentation size/order limits allow it.
- BEM package load and multiplier computation are guarded independently. If either fails, the result records a backend status and keeps `multiplier_output_available=false`; it does not erase the certified resolution, Fitting ideals, minors, or differential data.
- Successful multiplier output is parsed into `cas_artifacts["buchsbaum_eisenbud_diagnostics"]` with `a_multiplier_1_shape` and `a_multiplier_1_matrix`, and the two-parameter certificate table displays BEM availability/status/shape.
- `ADAPTER_CACHE_VERSION` was bumped so cached certificates produced before these fields are not reused.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 15 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 32 passed
```

Next linear CAS item: add more small-known-module fixtures for Macaulay2/Singular Fitting/minor/BEM behavior, then connect certified CAS evidence into derived/analogical comparison only when both sides expose compatible certified artifacts.


## 2026-06-15 Certified CAS Evidence Comparison Pass

Follow-up CAS/analogical item completed:

- Derived/analogical comparison now distinguishes three states: no certified multigraded free-resolution pair, certified pair with mismatched CAS evidence, and certified pair with matching CAS evidence.
- Matching evidence requires agreement of ring, input hash, stable artifact hash, multigraded Betti shifts, differential summaries, Fitting ideals, minors, and Buchsbaum-Eisenbud multiplier output/status.
- Mismatched certified artifacts remain useful evidence but are not safe for derived-category claims without an explicit resolution or chain-map isomorphism. The mismatch list is stored in `real_free_resolution_comparison["mismatched_components"]`.
- Analogical realization certificates now inherit `safe_for_derived_category_claims` from this artifact-level comparison rather than from the weaker condition that both sides merely have a certified resolution.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 33 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 15 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py -q
# 8 passed
```

Next linear item completed below: certified CAS comparisons are now part of retrieval scoring only when compatible query/memory CAS artifacts are available. Continue next with the simplex-tree/NLL/tropical-support repair queue.


## 2026-06-16 Certified CAS Evidence Retrieval Scoring Pass

Follow-up CAS/analogical retrieval item completed:

- `AnalogicalMemoryBank.retrieve` now computes a `certified_cas_evidence` retrieval component from the query topology and each memory topology.
- The component searches the topology payloads for CAS-certified multigraded real free-resolution reports with stored CAS artifacts. Chain-presentation diagnostics without a real CAS certificate remain unavailable for this score.
- Matching requires exact agreement of ring, input hash, stable artifact hash, multigraded Betti shifts, differential summaries, Fitting ideals, minors, and Buchsbaum-Eisenbud multiplier output/status.
- Only exact certified CAS artifact matches contribute to retrieval score. Partial matches, mismatches, and unavailable evidence are audited with explicit states and reasons but contribute zero.
- Retrieval output records `certified_cas_evidence_available`, `certified_cas_evidence_match`, `certified_cas_evidence_similarity`, `certified_cas_score_contribution`, `certified_cas_mismatched_components`, and a policy string stating that no derived-equivalence claim is asserted by retrieval scoring.

Validation:

```text
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_metrics_and_memory.py -q
# 9 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_interactive_artifact_validator.py -q
# 33 passed
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 15 passed
```

Next linear item: continue the simplex-tree/NLL/tropical-support repair queue while keeping active BPB training alive.

## 2026-06-16 Toric And Tropical CAS Cache Hardening Pass

Follow-up CAS infrastructure item completed after the certified-evidence retrieval scoring pass:

- Hardened `cas_tropical.py` and `cas_toric.py` cache handling to mirror the real free-resolution adapter contract: cache keys now include schema version, adapter cache version, canonical input hash, canonical input payload, and backend probe state.
- Moved default cache roots out of the repository into XDG/home cache locations: `~/.cache/tropicalgt/cas_tropical` and `~/.cache/tropicalgt/cas_toric`, with `TROPICALGT_CAS_TROPICAL_CACHE_DIR` and `TROPICALGT_CAS_TORIC_CACHE_DIR` overrides for tests or controlled runs.
- Added `TROPICALGT_CAS_TROPICAL_CACHE` and `TROPICALGT_CAS_TORIC_CACHE` falsey env guards for explicit cache disablement while preserving per-call `use_cache=False` behavior.
- Cache files are now wrapper payloads with `cache_schema_version`, `adapter_cache_version`, `key`, input hash, backend probe, and a `result` object. Loads reject stale schema/version/key mismatches or malformed payloads.
- Writes are atomic via a temporary file followed by replace, and reports record `cache.enabled`, `cache.hit`, `cache.written`, schema/version/key/path, and any write error.
- Added deterministic unavailable-probe regression tests for both tropical fan diagnostics and finite toric embedding certificates. These assert first-call writes, second-call hits, stable keys, explicit unavailable status, empty `cas_artifacts`, and no unsafe render flags.

Validation:

```text
/home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_tropical.py TropicalGT-I/src/tropicalgt/cas_toric.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests/test_algebraic_persistence.py -q
# 30 passed
```

Operational note: b61 failed at step 500 due to full disk while writing a periodic visualization artifact. The safe cache cleanup removed the old 33GB `~/.cache/tropicalgt/cas_free_resolution` tree and temp scratch directories, then b62 was restarted from step 0 under the always-on 5K gate.

Next linear CAS item: continue expanding certified backend evidence and UI surfaces only from real CAS output; do not reintroduce repository-local CAS caches or substitute unavailable toric/tropical certificates with chain or support-token diagnostics.

## 2026-06-18 CAS Bridge Provenance Contract

Sequential CAS hardening item completed:

- Added `tropicalgt.cas_backend_bridge_provenance.v1` as an explicit adapter-identity contract for Macaulay2, Sage, and Singular backend bridges.
- `probe_cas_backends()` now records per-backend bridge provenance with executable path, version, availability, unavailable reason, subprocess bridge type, no-proxy policy, and render-safety flags.
- `cas_execution_manifest()` now carries the same bridge provenance per backend alongside template key/hash/availability and complexity guards. The manifest also records `bridge_provenance_schema` at top level.
- Legacy cached free-resolution results are hydrated with conservative bridge provenance if the cached manifest is missing it. Hydrated bridge rows remain adapter identity only and are explicitly unsafe to render as free resolutions without backend-emitted exactness/minimality evidence.
- Regression tests assert that bridge provenance is never treated as a certificate: `adapter_identity_only=true`, `certificate_required_before_rendering=true`, `safe_to_render_without_certificate=false`, `safe_to_render_as_free_resolution=false`, and `no_proxy_or_fallback=true` for fresh probes, execution manifests, deterministic cache hits, complexity guards, disabled CAS reports, and legacy manifest hydration.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py -k "backend_probe or disabled_by_environment or deterministic_unavailable_probe or complexity_guard or legacy_cached_resolution_manifest"
# 7 passed, 25 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py
# 32 passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
```

Next linear CAS item: expand certified small-module fixtures for Fitting/minor/Buchsbaum-Eisenbud output while keeping BEMultipliers post-certificate only and never substituting unavailable CAS evidence.

## 2026-06-18 Certificate-Indexed CAS Evidence Summary

Sequential CAS evidence item completed after bridge provenance:

- Added `tropicalgt.cas_certificate_indexed_evidence.v1` to certified free-resolution reports.
- Certified result payloads now include `cas_artifacts["certificate_indexed_evidence"]`, which indexes Fitting ideals, determinantal minors, ideal diagnostics, Buchsbaum-Eisenbud rank conditions, grade/depth regular diagnostics, optional BEMultipliers multiplier output, and syzygy diagnostics back to the exact CAS certificate and input hash.
- The summary records render-safety flags from the certificate and marks diagnostic blocks as diagnostic-only where appropriate. It explicitly states that derived-category claims still require a chain map or certified resolution comparison.
- Existing certified cache hits are hydrated with this evidence summary when they already contain a certificate summary. Hydration does not create a certificate; it only indexes already stored certified artifacts and preserves no-proxy policy.
- Regression tests cover Singular ungraded determinantal/Fitting evidence, Macaulay2 multigraded evidence with grade/depth and syzygy diagnostics, BEMultipliers computed-output-as-post-resolution-diagnostic-only, live/cache certified reports, and older cache hydration.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py -k "singular_certified_result or certified_cas_result_surfaces or bemultipliers_computed_output or smoke_when_backend_available"
# 4 passed, 28 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py -k "total_graded_output or topological_algebra_report_has_multiparameter_data or level_radius_bifiltration_reports"
# 3 passed, 29 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py
# 32 passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
```

Next linear CAS/UI item: surface `certificate_indexed_evidence` in the relevant two-parameter/CAS visual and Herschel report paths only when present, otherwise show exact unavailable states.

### 2026-06-18 Live CAS Source-Contract Smoke Coverage

Follow-up verification item completed after adding the certificate-indexed source contract:

- The installed remote CAS path now exercises the same `tropicalgt.cas_certificate_indexed_source_contract.v1` assertions used by the deterministic Singular and Macaulay2 fixtures.
- The live backend smoke asserts the certificate summary source, backend-specific Fitting/minor source block names, exactness-required rendering flag, no-proxy/no-fallback flag, and the explicit rule that Fitting/minor diagnostics do not imply multipliers or derived equivalence.
- This remains a source-contract check, not a new algebraic claim: the backend still has to emit the certified resolution, exactness/minimality flags, and structured artifact blocks before any visual/report path may render them as mathematical evidence.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/tests/test_algebraic_persistence.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py::test_real_cas_free_resolution_smoke_when_backend_available
# 1 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py
# 32 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=332 covered=332 uncovered=0
```

## 2026-06-18 Certificate-Indexed Evidence Visual And Herschel Surfacing

Sequential CAS/UI surfacing item completed:

- `_cas_real_resolution_display()` now passes through `certificate_indexed_evidence` from certified CAS artifacts and emits an explicit unavailable contract when a certified legacy artifact lacks that block.
- The two-parameter bifiltration visual JSON now includes `certificate_indexed_cas_evidence` beside the Miller-Sturmfels staircase and structure-map evidence. Available blocks expose exact certificate-indexed CAS evidence; unavailable blocks render only as safe unavailable states with reasons.
- The CAS certificate table now includes a clipped JSON row for certificate-indexed evidence so reviewers can see the evidence schema, no-proxy policy, and derived-category guard without opening raw artifacts.
- Herschel's sidecar grouping now counts `certificate_indexed` evidence filenames under CAS algebra, so future review bundles that emit standalone certificate-indexed evidence sidecars are inventoried with the rest of the algebraic evidence.
- Regression tests cover direct display extraction, certificate-table rendering, two-parameter visual payload emission, Herschel sidecar grouping, and CPU-only full affected test files.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_simplicial_visualization.py -k "certified_cas_diagnostic_tables"
# 1 passed, 47 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py -k "level_radius_bifiltration_reports_scoped_real_staircase_resolution"
# 1 passed, 31 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/scripts:/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py
# 2 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_algebraic_persistence.py
# 80 passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_algebraic_persistence.py TropicalGT-I/tests/test_herschel_5k_report.py
# passed
```

Next linear item: continue the two-parameter persistence repair queue, preserving actual-data-only Miller-Sturmfels staircase rendering and adding any remaining no-proxy QA/validator coverage.

## 2026-06-18 Certificate-Indexed CAS Evidence Source Contract

Sequential CAS fixture hardening item completed after the two-parameter Betti diagnostic QA pass:

- `certificate_indexed_evidence` now includes `tropicalgt.cas_certificate_indexed_source_contract.v1`.
- The source contract records the exact source blocks for the exactness certificate, Fitting ideals, determinantal minors, ideal diagnostics, Buchsbaum-Eisenbud rank diagnostics, grade/depth diagnostics, BEMultipliers output, and syzygy diagnostics.
- The contract explicitly states that an exactness certificate is required before rendering, Fitting/minor diagnostics do not imply Buchsbaum-Eisenbud multipliers, and diagnostics do not certify a free resolution or derived equivalence by themselves.
- Singular and Macaulay2 fixture tests now assert backend-specific source names and no-proxy flags.

Validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_free_resolution.py TropicalGT-I/tests/test_algebraic_persistence.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py -k "singular_certified_result or certified_cas_result_surfaces"
# 2 passed, 30 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py
# 32 passed
```

## 2026-06-18 Miller-Sturmfels Hilbert/LCM Validator Gate

Sequential two-parameter QA item completed:

- The validator now requires available exact bivariate staircase-resolution cards to carry nonempty exact Hilbert numerator terms.
- Nonprincipal two-variable staircase cards must carry adjacent LCM syzygy rows when the theorem surface reports an available resolution.
- Aggregate Miller-Sturmfels evidence must explicitly certify Hilbert numerator terms and adjacent-LCM syzygy lists across all cards.
- This closes the remaining validator gap for the user-facing requirements: x_radius horizontal, x_level vertical, upward-closed generated regions, quotient-basis lattice points, minimal antichain boundary, generator labels, exact Hilbert terms where available, adjacent-LCM syzygies where the two-variable theorem applies, and no 3D rank slab as the primary view.

Validation: py-compile passed; focused validator regressions passed (`3 passed, 45 deselected`); full interactive artifact validator passed (`48 passed`); metric provenance audit passed (`332 covered, 0 uncovered`).

Next linear item: continue through the simplex-tree/NLL/tropical-support repair queue and preserve actual-data-only browser/validator contracts.

## 2026-06-18 Optional Macaulay2 Tropical Side Diagnostics

Sequential CAS/tropical hardening item completed:

- The Macaulay2 Tropical adapter now probes additional optional methods after the primary `tropicalVariety I` certificate path: `cones T`, `tropicalCycle I`, `BergmanFan I`, `stableIntersection(T, T)`, and `visualizeHypersurface first G`.
- These outputs are recorded under `optional_method_diagnostics` and clipped raw `cas_artifacts` only when Macaulay2 actually emits them.
- Unavailable or unsupported methods render as explicit unavailable side diagnostics with backend reasons; they never replace the `tropicalVariety` fan/cycle certificate gate.
- The certificate contract now lists the additional methods as side diagnostics and preserves the no-proxy rule that support-token, chain, toric, or normal-fan substitutes cannot certify tropical fan evidence.
- Regression coverage checks script generation, tagged-parser behavior, available/unavailable side diagnostics, raw artifact pass-through, and the certificate-gate string for each optional method.

Validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/cas_tropical.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py -k "tropical_fan"
# 4 passed, 28 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_algebraic_persistence.py
# 32 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=436 covered=436 uncovered=0
```

Next linear CAS item: connect optional tropical side diagnostics into visual/report surfaces only as side evidence when present; continue real CAS wrapper expansion for certified resolutions, Fitting/minor evidence, and derived/analogical comparisons without proxy substitutions.

## 2026-06-18 Optional Tropical Diagnostics Visual/Herschel Surfacing

Sequential CAS/UI surfacing item completed:

- `write_tropical_fan_diagnostics()` now emits `tropicalgt.tropical_fan_optional_method_visual_summary.v1` beside the real Macaulay2 Tropical fan diagnostics payload.
- The browser table displays optional Macaulay2 Tropical side diagnostics as side evidence only, with available/unavailable counts and the explicit certificate gate string.
- Herschel now summarizes optional tropical side-method evidence from either the top-level visual summary or the raw CAS `optional_method_diagnostics` block.
- The 5K report records aggregate optional method counts, per-method available/unavailable status counts, and per-source method counts without changing the certified tropical-fan availability predicate.
- The certificate gate remains unchanged: `tropicalVariety` fan/cycle output from an explicit model-derived ideal is still required before one-dimensional cones are rendered as tropical fan evidence.

Validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/src/tropicalgt/visualization.py TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_simplicial_visualization.py -k "tropical_fan_diagnostics"
# 2 passed, 48 deselected
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/scripts:TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# 11 passed
```

Next linear item: continue real CAS wrapper expansion and source contracts for certified resolutions, Fitting/minor evidence, BEMultipliers post-certificate diagnostics, and derived/analogical comparisons without proxy substitutions.
