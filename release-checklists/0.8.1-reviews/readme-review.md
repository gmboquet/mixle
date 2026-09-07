# README stale-content review — 0.8.1

- **Date:** 2026-09-07 (review begun on the pivot tree `c9c5fbbb`; every claim re-checked on the
  repaired 0.8.1 candidate tree that this record is committed with).
- **Reviewer:** release owner's agent, line by line; machine checks run from `/tmp` against the
  candidate venv (`mixle` 0.8.1 from the candidate wheel, never a source checkout on `sys.path`).
- **Method:** every badge, count, version, link, command, and code snippet in `README.md` checked
  against the tree and the live docs-site layout (releases under `v<version>/`, development docs
  under `main/`, the site root a redirect); every fenced Python block executed verbatim, then
  varied (adversarial pass 08 repeated this independently: all seven snippets run, then varied by
  input type, seed, size, and model width).

## Verified true on the candidate

| Claim | Check |
|---|---|
| `python-3.11+` badge; "`requires-python` is `>=3.11,<3.13`" | `pyproject.toml` |
| `tests-15,000+` badge; "15,000+ tests, organized into tiers" | `pytest --collect-only -m ""` on the candidate: 16,538 collected (docs review, 2026-09-07); 16,552 in this owner's earlier count of the pivot tree |
| Every named extra exists and `[all]` installs every extra listed | `pyproject.toml [project.optional-dependencies]` |
| Every documentation link resolves under `v0.8.1/` | ten `v0.8.1/` links; none at the site root or without the prefix (the 0.8.0 candidate's README carried five dead links, which this gate exists to catch) |
| Every GitHub link (`blob/v0.8.1/...`, raw assets) names a path that exists in the tree | `git ls-files` |
| Quickstart, PPL, propose/describe, planner, DOE, distill, enumeration snippets | all execute on the candidate; the enumeration snippet required the library repair recorded below |
| "Any module exposing `log_density(x)` fits with one call", `GradEstimator`, `TorchEngine`, `SymbolicEngine`, `AutoregressiveEnumerable`, `compare`/`waic`/`loo`, `solve(**distill_kw)` | import and signature checks; snippets execute |
| CI matrix statement (Linux x86_64 + macOS arm64; full = 4 shards + coverage floor) | `.github/workflows/tests.yml` |
| Metal ~2% divergence note | checklist GPU appendix; `docs/backend-support.rst` |
| "~90 distribution families" | an E1-graded public claim owned by `manifests/public_claims.json` and guarded by `model_catalog_test`/`invariant_catalog_test`; kept as the ledger's number (the raw count of exported `*Distribution` classes is 165; the ledger's counting rule is the manifest's) |
| "the five named examples need only the base install; no example downloads a dataset" | `docs/example-execution-manifest.rst` |

## Repaired

1. "`[all]` covers ... jax/gmpy2/sympy/sage/mongo/hadoop/arrays install separately" -> every
   named extra is inside `[all]`; the sentence now says so.
2. "40 of its 41 base-distribution families" -> the test holds 40 configurations across 34
   families and generates 41 estimation tests; the sentence carries the real numbers.
3. The "Enumeration & ranking" snippet raised on the candidate venv (`next_logprobs returned a
   non-normalized distribution (kept probabilities sum to 0.998532)`): transformers 5 loads
   SmolLM2 in bfloat16 and the bf16 log-softmax sums 0.15% short of one, which the 1e-4 tolerance
   rejected. Repaired in the library (tables within 2% of one are renormalized;
   `autoregressive_reduced_precision_test.py`), not by pinning a dtype in the README; the snippet's
   printed outputs (`[' located in the', ' the city of', ' the capital of']`, rank 6, cumulative
   probability 0.114) reproduce on the repaired tree (adversarial pass 08, independently).

## Noted, not changed

- The "Compose to any depth" nested-HMM snippet collapses to a one-state HMM at `optimize()`'s
  default `max_its=10` and is correct at 300 (pass 08, P08-F02, real): carried in
  `0.8.2-followups.md`; the snippet illustrates composition, and its output is not printed in the
  README.
- The README comment "(out=None: quiet)" is accurate; `print_iter=` without `out=` prints nothing
  (P08-F04, minor, carried).
