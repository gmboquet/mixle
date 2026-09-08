# README stale-content review — 0.8.2

- **Date:** 2026-09-08, on the 0.8.2 candidate tree.
- **Reviewer:** release owner's agent, line by line; every machine check run from `/tmp` against a
  candidate venv holding the 0.8.2 wheel built from this tree, never a source checkout on
  `sys.path` (the tree is confirmed by `mixle.__path__[0]`, not by `importlib.metadata.version()`,
  which reports the installed distribution and would say the same thing either way).
- **Method:** every badge, count, version, link, and code block in `README.md` re-checked against
  this tree. Each of the seven fenced Python blocks was executed; the ones that are deliberate
  fragments were executed again with the context their prose declares, so that "it runs" is a
  statement about the library rather than about the snippet's completeness.

## Verified true on the candidate

| Claim | Check |
|---|---|
| `python-3.11+` badge | `pyproject.toml` `requires-python = ">=3.11,<3.13"` |
| `tests-15,000+` badge; "15,000+ tests, organized into tiers" | `pytest --collect-only -m ""` on this tree: **16,849** collected |
| "40 base-distribution configurations across 34 families" | `base_dist_test._build_dists()` returns **40** configurations over **34** distinct classes |
| Every documentation link resolves under `v0.8.2/` | five `v0.8.2/` docs-site links, plus two deliberate site-root links (the docs badge and the "Docs:" line, both of which point at the redirecting root by design) and three `v0.8.2` GitHub tree/blob links |
| Every named extra exists and `[all]` is their union | `scripts/check_optional_extras.py`: "all equals the runtime-feature extra union", exit 0 |
| The standalone-release paragraph names 0.8.2 | "Mixle Core 0.8.2 is a standalone release (a patch of 0.8.1; 0.8.0 itself was never published)" |

## The seven fenced Python blocks

Run with the 0.8.2 wheel installed, from `/tmp`.

| Block | Kind | Result |
|---|---|---|
| 1 — `optimize(records)` quickstart | self-contained | exits 0 |
| 2 — `solve(teacher, inputs)` | fragment (`teacher`, `inputs`, `x` declared in the comment above it) | with those supplied: `assistant(x)` answers, `report()` returns its documented keys, `save()` writes |
| 3 — `optimize(x, my_module)` | fragment (`x`, `my_module` are the reader's) | with a two-line `nn.Module`: fits, and `model.module` hands the raw module back |
| 4 — heterogeneous HMM | fragment (`sequences`, `my_module`) | with those supplied: returns a `HiddenMarkovModelDistribution` with the two states written |
| 5 — engine/precision/backend one-liners | fragment (literal `...` for the data) | all three keywords are accepted; `device="cuda"` fails only on "Torch not compiled with CUDA enabled" and `backend="spark"` only on the absent RDD, which is what those lines claim |
| 6 — torch integration | self-contained | exits 0 |
| 7 — PPL one-liners | fragment (the last line carries explicit `...` placeholders) | the four complete calls all return a `RandomVariable`; the regression line does too once its placeholders are filled |

Blocks 2, 3, 4, 5 and 7 do not execute verbatim, and are not meant to: they name values the
surrounding prose defines, or carry an explicit `...`. That is worth stating rather than counting
them as "7 of 7 execute", because the interesting question is whether the LIBRARY still does what
each line says, and that is what was checked.

## Nothing stale found

No claim in `README.md` was found stale on this tree. The version strings, the docs-site prefix,
and the standalone-release paragraph were all moved to 0.8.2 when the release branch was cut; the
two counted claims (16,849 collected against a "15,000+" badge, and 40/34 in `base_dist_test`) both
hold with room. This is a shorter record than 0.8.1's because 0.8.1's review repaired three stale
items and this one had none to repair.
