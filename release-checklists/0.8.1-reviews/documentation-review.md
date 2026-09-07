# mixle 0.8.1 documentation stale-content review

- **Date:** 2026-09-07
- **Candidate branch:** `release/0.8.1`
- **Repository:** `<worktree>`
- **Git HEAD:** `c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02` ("Prepare 0.8.1: the 0.8.0 candidate tree released under a new version (D-0212)")
- **Working tree at review time:** 15 modified, uncommitted files (`CHANGELOG.md`, `README.md`, three `examples/*.py`, ten `mixle/**.py` incl. `mixle/ppl/inference.py`, `mixle/stats/univariate/discrete/logseries.py`). **Nothing under `docs/` is modified**, so every page below is exactly as at HEAD. Line counts quoted for `large-module-audit.rst` are from the on-disk tree; where the uncommitted edits move a count the HEAD figure is given too.
- **Candidate under test:** `<review-root>/candidate-081/venv` — `mixle.__version__ == "0.8.1"`, resolved from `site-packages/mixle` (not the source tree; verified via `importlib.util.find_spec`).
- **Facts from the tree used as the baseline:** `pyproject.toml` version `0.8.1`, `requires-python >=3.11,<3.13`; `mixle/tests`: 1,238 `*_test.py` + 34 `test_*.py` files, **16,538 tests collected** (`pytest --collect-only -q -m ""`), 7 in the `benchmark` tier, 4 in `smoke` (`mixle/tests/smoke-manifest.txt` = one file, `smoke_test.py`, 4 tests); `examples/`: **57** `.py` scripts, 0 notebooks (no `notebooks/` in the tree); `mixle.stats` exports **165** `*Distribution` classes (164 in `__all__`; 176 distinct non-test, non-experimental `*Distribution` classes defined); workflows: `docs.yml extras-matrix.yml post-publish-verify.yml publish.yml security.yml tests.yml` (`tests.yml`/`docs.yml` push-trigger on `[main, release/0.8.1]`); `docs/conf.py` derives `release = "0.8.1"`, `version = "0.8"` from `pyproject.toml`; every `docs/index.rst` toctree entry exists; 33 modules exceed 1,500 lines.
- **Method:** every page listed under the focus set was read in full. Every other toctree page was swept for version strings (`0.8.0`, `0.7.x`, `0.9`, `v0.8`), branch names (`release/…`, `release-prep/…`), docs-site and GitHub URLs, workflow names (`*.yml`), script/checklist/manifest/example/test paths (each checked for existence), `pip install` extras (each checked against `pyproject.toml`), counts, and commands (`--help` run on `mixle-reproduce`, `scripts/reproduce.py`, `run_repro_entry.py`, `run_smoke.py`, `run_test_tier.py`). In addition, every `from mixle… import …` statement (256 distinct) and every dotted `mixle.*` name (214 distinct) in the non-generated pages was resolved against the candidate; all imports resolve, three dotted names do not (listed below). All 696 `docs/api/*.rst` pages name importable modules. The strict Sphinx build was not re-run (already known green).

## Findings by page

Verdict `stale` = at least one claim contradicted by the tree or by the 0.8.1 facts (rules a–e of the brief). `history` in the notes marks 0.8.0/`release/0.8.0` mentions that are dated records and acceptable as written.

| Page | Verdict | What is stale (file:line — stale text — fact from the tree) |
|---|---|---|
| `docs/index.rst` | current | Toctree: all 97 entries exist (incl. `migrations/0.8.1`, `migrations/0.8.0`, `api/modules`). Hero links are relative (`quickstart.html` …). No version claims. |
| `docs/conf.py` | current | `release`/`version` read from `pyproject.toml` → `0.8.1` / `0.8`. `html_context` exposes `mixle_release`; version switcher builds `root + v.name + "/index.html"` from `switcher.json` (no hard-coded URLs). Comment at :140 mentions the 0.8.0 docs gate — history in a comment. `audits/*.md` are not in `exclude_patterns` but carry `orphan: true`. |
| `docs/installation.rst` | **stale** | :4 "``mixle`` 0.8.0 supports Python 3.11 and 3.12" — shipping version is 0.8.1. Everything else current: all 16 named extras exist in `pyproject.toml` (`torch scientist numba spark dask mpi ray lightning jax highprec data umap sympy sage grammar examples all docs test lint`); `make -C docs apidoc` / `html` targets exist; `examples/shared_embedding_example.py` exists. |
| `docs/maturity.rst` | current | The seven named modules + `parameter_packing` return `stable` from `maturity_of`; `beta`, `hidden_markov`, `ppl` → `provisional`; `experimental.cvi` → `experimental`. `:mod:\`mixle.maturity\`` at :17 has no API page (see `api/`). |
| `docs/stable-surface.rst` | **stale** | :4 "what 0.8.0 supports as **stable**" — 0.8.1. :27 `fixtures/quantitative-semantics-v1.json` exists; :38 "digest-bound 0.7.0 load fixtures" is a correct prior-release reference. |
| `docs/what-mixle-is-not.rst` | current | No version/count/path claims. |
| `docs/quickstart.rst` | current | `pip install mixle`, `pip install -e .`; all imports resolve. |
| `docs/concepts.rst` | current | Sweep clean; imports resolve. |
| `docs/quantitative-semantics.rst` | current | Sweep clean. |
| `docs/module-ownership.rst` | current | `manifests/module_ownership.json` exists. |
| `docs/package-map.rst` | current | All dotted names resolve. Observation (not stale): top-level modules `mixle.capability`, `mixle.lifecycle`, `mixle.program`, `mixle.reproduction` are not mentioned on the "human map" page. |
| `docs/lifecycle.rst` | current | Sweep clean; imports resolve. |
| `docs/capability-lifecycle.rst` | current | `mixle.capability_lifecycle`, `mixle.substrate.authorization_decision` resolve (no API page for the former — see `api/`). |
| `docs/tutorials/index.rst` + 9 tutorials | current | All `:doc:` targets exist; all imports resolve; no version/branch/URL/count claims. |
| `docs/neural-boundary.rst` | current | Sweep clean. |
| `docs/neural-llm.rst` | current | Sweep clean. |
| `docs/torch-modules.rst` | current | Sweep clean. |
| `docs/automatic-inference.rst` | current | Sweep clean. |
| `docs/models.rst` | current | Sweep clean. |
| `docs/representation.rst` | current | Sweep clean. |
| `docs/task-distillation.rst` | current | Named example scripts and tests exist. |
| `docs/task-serving.rst` | current | Sweep clean. |
| `docs/bring_your_own_model.rst` | current | Sweep clean; imports resolve. |
| `docs/training-at-scale.rst` | current | Named tests (`training_health_test.py`, `pilot_ladder_test.py`) exist; "PR #171" is a history reference. |
| `docs/agentic-task-distillation.rst` | current | Sweep clean. |
| `docs/uncertainty.rst` | current | Sweep clean. |
| `docs/reasoning-systems.rst` | current | Sweep clean. |
| `docs/reasoning-ecosystem.rst` | current | Sweep clean. |
| `docs/hmms-latent.rst` | current | Sweep clean. |
| `docs/processes.rst` | current | Sweep clean. |
| `docs/automatic-modeling-contract.rst` | current | Worklist IDs only; `model_selection_benchmark_test` exists. |
| `docs/automatic-modeling-internals.rst` | current | Sweep clean. |
| `docs/cookbook.rst` | current | Sweep clean; imports resolve. |
| `docs/release-readiness.rst` | current | :11 "0.8.1 package metadata declares Python 3.11 and 3.12 (>=3.11,<3.13)" matches `pyproject.toml`; :143 `release-checklists/0.8.1.md` exists. Note: :72 requires regenerated API pages when modules are added — see `api/` row. |
| `docs/claim-evidence-ledger.rst` | **stale** | :18 "not made in 0.8.0"; :20 "pending the 0.8.0 re-run"; :42 "remains a 0.8.0 exit criterion"; :74 "a 0.8.0 gate in progress" — the release is 0.8.1 (and stable-surface.rst:38 says the 0.7.0 load fixtures now exist, so :74 is behind on its own evidence). :29 "~90 distribution families" — the candidate exports 165 `*Distribution` classes from `mixle.stats`; no counting rule in the tree yields ~90 (same statement pinned in `manifests/public_claims.json`). :39 "Measured 2026-08-04: 15,121 collected" — not contradicted ("15,000+" holds) but dated; the candidate collects 16,538. :8 "from the 0.8.0 release contract" — history, acceptable. |
| `docs/validation.rst` | current | Commands (`python -m pytest`, `python -m build`, `twine check dist/*`, `make -C docs html SPHINXOPTS="-W --keep-going"`) are generic and the make target exists. |
| `docs/scale-out-economics.rst` | current | `mixle/utils/parallel/mpi.py`, `mixle/tests/reduction_payload_telemetry_test.py` exist. |
| `docs/test-tiers.rst` | current | :28/:65 "4 tests in an explicit one-file manifest" — verified: manifest = `mixle/tests/smoke_test.py`, 4 tests collected with `-m smoke`. :55 "7 tests collected" (benchmark) — verified: 7. `scripts/run_smoke.py`, `scripts/run_test_tier.py` exist and run `--help`. :45 "roughly 1100 … roughly 340" (optional tier) — not re-verified (needs two environments). |
| `docs/performance-crossover.rst` | **stale** | :19 "The 0.8.0 release does not publish a numerical crossover table"; :63 "claimed for 0.8.0" — 0.8.1. `benchmarks/archive/` exists. |
| `docs/benchmark-methodology.rst` | current | `benchmarks/` exists; worklist IDs only. |
| `docs/reproduction.rst` | **stale** | :70 hyperlink `<../release-checklists/0.8.0-decisions.md>` — relative to the built HTML it leaves the site (`https://gmboquet.github.io/mixle/release-checklists/…` → dead; releases are served under `/v0.8.1/`), and the live ledger is `0.8.1-decisions.md`. Current: `mixle-reproduce --wheel/--source-tree/--out`, `scripts/reproduce.py --source-tree`, `scripts/run_repro_entry.py --entry gallery-univariate` all run; the four bundle entries match `release-checklists/0.8.1-repro-bundle.json`; `mixle/tests/reproduce_receipt_test.py` exists. |
| `docs/support-policy.rst` | **stale** | :19 "Core 0.8.0 declares this runtime floor"; :34 "excluded from Core 0.8.0 compatibility"; :50 "For 0.8.0 the extras resolver…"; :126 "(announced in ``0.8.0`` → removable no earlier than ``0.10.0``)" — a deprecation cannot have been announced in an unpublished version; :183 "For 0.8.0, ``requires-python`` is capped" — all should read 0.8.1. `:func:` refs at :131-132 point at `mixle.utils.deprecation`, which has no API page. `mixle/tests/deprecation_test.py` exists. |
| `docs/backend-support.rst` | **stale** | :105 the "Retained local execution evidence…" sentence is duplicated verbatim in the Ray row (rename artefact). :52/:151-152 "Executed once for 0.8.0 … `release-checklists/0.8.0-cuda-receipt.json`" — history; the receipt file exists under that name (D-0212 keeps 0.8.0-named history files). :90/:95/:105/:120/:150 `release-checklists/0.8.1.md` exists and carries the backend-execution-evidence appendix. |
| `docs/security-and-data.rst` | **stale** | :43 "As of 0.8.0, every loader…" — the first published release with the gate is 0.8.1. `from mixle.data import load_encoded` resolves. |
| `docs/stability-and-missing-data.rst` | current | Sweep clean. |
| `docs/family-release.rst` | **stale** | :13 heading "0.8.0 Scope Decision"; :16 "Mixle Core 0.8.0 is deliberately a standalone core release"; :19 "excluded from the 0.8.0 artifact"; :112 "standalone Core 0.8.0 release" — 0.8.1. |
| `docs/release-notes.rst` | **stale** | :94-99 "independent statistical and systems review, external clean-install reproduction, final sign-off" — checklist §11 changed under D-0212 (statistical/systems reviews DONE; external-tester gate replaced by ten AI adversarial reviews; README and documentation stale-content reviews added; version row and sign-off reopened). Current: :4 correctly states 0.8.1 first-published / D-0212; :60 `:doc:\`migrations/0.8.0\`` as the per-surface list agrees with `migrations/0.8.1.md`; :63-65 Python floor matches; :89-92 job set matches `tests.yml`/`docs.yml`/`security.yml`. |
| `docs/example-execution-manifest.rst` | **stale** | :12 "**not** 0.8.0 release evidence" — 0.8.1. :66-67, :82, :140, :147, :185 dated passes name `release/0.8.0` / "current ``0.8.0`` source" / `release-prep/0.8.0` — history; acceptable once :12 says the dated passes ran on the 0.8.0 candidate branch. Current: :31 "57 Python example scripts" = tree; :98-99 23 + 34 = 57; inventory table lists all 57 shipped scripts, none missing, none extra; :359 `mixle.example_execution_manifest/v2` matches `scripts/build_example_execution_manifest.py:235`; :16/:362 `release-checklists/0.8.1-repro-bundle.json` exists; :108 "131-notebook corpus" is external (`mixle-notebooks`), consistent with the checklist, not verifiable from this tree. |
| `docs/changelog.rst` | current | :13-20 0.8.1 section states first-published / D-0212 / `release-checklists/0.8.1-decisions.md` (exists). The 0.8.0 section is a doc-visible summary under a heading the 0.8.1 text explains. Narrower than `CHANGELOG.md` by design (:8-11). |
| `docs/charter.md` | current (observation) | :46 "Version 0.8.1 is a published release … 0.8.0 … never published (D-0212)" — correct on the facts, but it is the only page that speaks in the past tense; `releases.md:7-8` ("remains unreleased until all applicable gates have accepted evidence"), `release-notes.rst:9-11`, `support-policy.rst:12` and `family-release.rst:72` all say pre-publication. Not stale by the brief's rules; worth aligning before the tag. |
| `docs/requirements.md` | current | :31 "Pull requests targeting 0.8.1". |
| `docs/architecture.md` | current | All named modules exist in the tree. |
| `docs/contracts.md` | current | Sweep clean. |
| `docs/development.rst` | current | Commands generic; `make -C docs apidoc`/`html` exist; `.githooks` exists. `path/to/changed.py` / `file_test.py` are placeholders. |
| `docs/testing.md` | current | `.github/workflows/tests.yml`, `mixle/tests/conftest.py` exist; both filename conventions are collected (1,238 + 34 files). |
| `docs/security.md` | **stale** | :27 "opt-in, per call, as of 0.8.0" — 0.8.1. `../SECURITY.md` is plain text, and the file exists. |
| `docs/scientific-validity.md` | current | Sweep clean. |
| `docs/operations.rst` | current | `mixle.ops` reference; every operation named resolves; no release claims. |
| `docs/releases.md` | current | :7 "0.8.1 … 0.8.0 was never published, D-0212"; :12 "targets release/0.8.1" matches `tests.yml`; :16-17 `REL-PRJ-CORE-0.8.1` / milestone 0.8.1. |
| `docs/migrations/0.8.1.md` | current | :7 additions verified: `win_probability` on `BradleyTerryDistribution` and `ThurstoneMostellerDistribution`, `win_probability`+`tie_probability` on `DavidsonDistribution`, `strongly_connected` on both fit-diagnostics dataclasses. |
| `docs/migrations/0.8.0.md` | **stale** | :3 "Status: released 2026-08-22." — 0.8.0 was never published (CHANGELOG.md, D-0212); this is the guide `migrations/0.8.1.md` and `release-notes.rst:60` send readers to. Content otherwise consistent with the candidate (`Verdict.calibration_status` is a dataclass field; `mmd`/`mmd_squared`, `receipt_subject`, `VerificationReceipt` resolve). :120 "101 public flags", :128 "35 record types", :136 "eight remain mutable" — not re-verified (no counting script in the tree). |
| `docs/migrations/README.md` | current | Orphan index; :10-11 describe 0.8.1 as the published 0.8 release and 0.8.0 as development notes. |
| `docs/api-overview.rst` | current | All 214 dotted names / imports resolve. |
| `docs/capabilities-contracts.rst` | current | Sweep clean. |
| `docs/compute-layer.rst` | current | Sweep clean. |
| `docs/distributions.rst` | current | Sweep clean. |
| `docs/stats-univariate.rst` | current | Sweep clean. |
| `docs/stats-structured.rst` | current | Sweep clean. |
| `docs/stats-latent-bayes.rst` | current | Sweep clean. |
| `docs/inference.rst` | current | Sweep clean. |
| `docs/inference-toolkit.rst` | current | Sweep clean. |
| `docs/ppl.rst` | current | Sweep clean. |
| `docs/relations.rst` | current | Sweep clean. |
| `docs/engines.rst` | current | Sweep clean. |
| `docs/enumeration.rst` | current | Sweep clean. |
| `docs/data.rst` | current | `mixle[pandas]`, `mixle[arrow]` extras exist. |
| `docs/doe.rst` | current | Sweep clean. |
| `docs/analysis.rst` | current | Sweep clean. |
| `docs/evolution.rst` | current | Sweep clean. |
| `docs/production.rst` | current | :41 `https://github.com/gmboquet/mixle-mlops` — external repository, not verifiable from this tree (only GitHub URL in the docs). |
| `docs/utilities-and-parallelism.rst` | current | Sweep clean. |
| `docs/experimental-program.rst` | current | Sweep clean. |
| `docs/examples.rst` | **stale** | :178 "Complete Inventory" lists 43 of the 57 shipped scripts; 14 missing (`autoregressive_enumeration_example.py`, `calibrated_report_demo.py`, `capability_layer_example.py`, `copula_vine_example.py`, `frontier_family_showcase.py`, `geoscience_inversion_report.py`, `label_economics_demo.py`, `model_comparison_example.py`, `multimodal_stage1_demo.py`, `peft_lora_grad_leaf.py`, `precedence_scheduling_example.py`, `quickstart_example.py` — recommended at :27 of the same page — `symbolic_export_example.py`, `vlm_trust_receipts_demo.py`). No listed script is missing from the tree; all commands and `literalinclude` targets exist. |
| `docs/examples_gallery.rst` | current | Both scripts and both paired smoke tests exist; `pip install "mixle[torch]" transformers peft` matches the examples' own requirements. Printed receipts (-48.27 → -46.18; 11 frozen tensors; -3.21 → -0.60) not re-run here (torch/peft); pinned by the paired tests. |
| `docs/troubleshooting.rst` | current | Sweep clean. |
| `docs/glossary.rst` | current | `:mod:\`mixle.maturity\`` at :75 has no API page (see `api/`). |
| `docs/extending.rst` | current | Sweep clean. |
| `docs/api/modules.rst` (+ `api/mixle.rst`, 696 generated pages) | **stale** | Last regenerated 2026-08-03 (`f2fa807f`). 99 public modules present in the tree and importable from the candidate have no page — 49 outside `mixle.experimental`, including top-level `mixle.blending`, `mixle.causal`, `mixle.capability_lifecycle`, `mixle.maturity`, `mixle.reproduction`, `mixle.fulfillment`, `mixle.pipeline_twin`, `mixle.precedence_scheduling`, `mixle.stochastic_opt`, and `mixle.utils.deprecation` (full list in `stale.json`); `api/mixle.rst` lists none of them. `release-readiness.rst:72` makes regenerated pages a documentation gate. All 696 existing pages name importable modules. |
| `docs/design-notes.rst` | current | Sweep clean. |
| `docs/frontier-integration-note.rst` | current | Sweep clean. |
| `docs/large-module-audit.rst` | **stale** | :68 "``mixle.stats.lookback_hmm``" — no such module in the tree or candidate. Per-module line counts stale for 30 of 33 modules (e.g. :35 `hidden_markov.py` (4,395) vs 5,308; :56 `structured_hmm.py` (1,822) vs 3,320; :119 `temporal_graph_grammar.py` (2,509) vs 4,424; :227 `ppl/inference.py` (2,843) vs 4,479 on disk / 4,259 at HEAD; :238 `ppl/core.py` (2,235) vs 3,317). :4 "Thirty-three modules … exceed 1,500 lines" is exact (33 on disk); `large_module_audit_test.py` pins only the set and paths. |
| `docs/audits/0.8.0-exhaustive-code-review.md`, `docs/audits/0.8.0-finding-dispositions.md` | current (history, orphan) | `orphan: true`; dated 2026-08-02 records of the 0.8.0 candidate (`REL-PRJ-CORE-0.8.0`, candidate revision `8263c4c8`). Not linked from any page; served under `v0.8.1/audits/` if built. Suggest a one-line note that the candidate shipped as 0.8.1, but not required by the brief's rules. |
| `docs/_templates/sidebar/version-switcher.html`, `docs/_root_templates/*` | current | No hard-coded site URLs; the root `index.html` template links `main/index.html`; `switcher.json` is rendered from the assembly workflow's revision list. |

## Pages confirmed current

`index.rst`, `conf.py`, `maturity.rst`, `what-mixle-is-not.rst`, `quickstart.rst`, `concepts.rst`, `quantitative-semantics.rst`, `module-ownership.rst`, `package-map.rst`, `lifecycle.rst`, `capability-lifecycle.rst`, `tutorials/index.rst`, `tutorials/heterogeneous-records.rst`, `tutorials/ppl-mixture.rst`, `tutorials/enumeration-ranking.rst`, `tutorials/production-artifacts.rst`, `tutorials/llm-distillation-cascade.rst`, `tutorials/llm-uncertainty.rst`, `tutorials/representation-and-models.rst`, `tutorials/relations-and-operations.rst`, `tutorials/evolution-and-analysis.rst`, `neural-boundary.rst`, `neural-llm.rst`, `torch-modules.rst`, `automatic-inference.rst`, `models.rst`, `representation.rst`, `task-distillation.rst`, `task-serving.rst`, `bring_your_own_model.rst`, `training-at-scale.rst`, `agentic-task-distillation.rst`, `uncertainty.rst`, `reasoning-systems.rst`, `reasoning-ecosystem.rst`, `hmms-latent.rst`, `processes.rst`, `automatic-modeling-contract.rst`, `automatic-modeling-internals.rst`, `cookbook.rst`, `release-readiness.rst`, `validation.rst`, `scale-out-economics.rst`, `test-tiers.rst`, `benchmark-methodology.rst`, `stability-and-missing-data.rst`, `changelog.rst`, `charter.md` (with the tense observation above), `requirements.md`, `architecture.md`, `contracts.md`, `development.rst`, `testing.md`, `scientific-validity.md`, `operations.rst`, `releases.md`, `migrations/0.8.1.md`, `migrations/README.md`, `api-overview.rst`, `capabilities-contracts.rst`, `compute-layer.rst`, `distributions.rst`, `stats-univariate.rst`, `stats-structured.rst`, `stats-latent-bayes.rst`, `inference.rst`, `inference-toolkit.rst`, `ppl.rst`, `relations.rst`, `engines.rst`, `enumeration.rst`, `data.rst`, `doe.rst`, `analysis.rst`, `evolution.rst`, `production.rst`, `utilities-and-parallelism.rst`, `experimental-program.rst`, `examples_gallery.rst`, `troubleshooting.rst`, `glossary.rst`, `extending.rst`, `design-notes.rst`, `frontier-integration-note.rst`, `audits/*.md` (orphan history), `_templates/`, `_root_templates/`.

Depth caveat: the focus pages (release notes, migrations, changelog, maturity, release-readiness, validation, test-tiers, support-policy, backend-support, example-execution-manifest, releases, requirements, charter, installation, stable-surface, family-release, claim-evidence-ledger, performance-crossover, reproduction, security, security-and-data, testing, development, architecture, contracts, scientific-validity, operations, examples, examples_gallery, benchmark-methodology, scale-out-economics, what-mixle-is-not, module-ownership, capability-lifecycle, large-module-audit) were read line by line. The remaining narrative pages were checked mechanically (version/branch/URL/workflow/path/count/command sweep plus import resolution against the candidate), not read for prose accuracy.

## Not reviewed

- The 696 generated `docs/api/*.rst` pages were checked only for module existence (all importable) and coverage (99 tree modules lack pages); autodoc'd docstring content was not reviewed.
- Numerical receipts printed by examples (`examples_gallery.rst`), the optional-tier counts in `test-tiers.rst:45`, and the "101 flags / 35 frozen records / eight mutable" counts in `migrations/0.8.0.md` were not re-executed or recounted.
- `README.md` (separate gate) and `release-checklists/*.md` (not part of the Sphinx tree). Out-of-scope observation for the release owner: `release-checklists/0.8.1.md` §11 still says `gh workflow run tests.yml --ref release/0.8.0` and describes the `testpypi` environment's deployment refs as `branch:release/0.8.0`; `tests.yml` itself already triggers on `release/0.8.1`.

## Unresolvable names found by the import/dotted-name probe

- `large-module-audit.rst:68` `mixle.stats.lookback_hmm` — does not exist (stale, above).
- `example-execution-manifest.rst:359` `mixle.example_execution_manifest/v2` — a schema identifier, not a module; matches `scripts/build_example_execution_manifest.py:235` (current).
- `migrations/0.8.0.md:37` `mixle.evolve.verify.Verdict.calibration_status` — a dataclass field, present on instances (current).

## Summary

1. Pages reviewed: 108 (97 toctree pages incl. 9 tutorials and 2 migration guides, `conf.py`, `migrations/README.md`, 2 orphan audit records, the `api/` tree as one unit, and the templates).
2. Pages stale: 16 (`migrations/0.8.0.md`, `installation.rst`, `stable-surface.rst`, `support-policy.rst`, `family-release.rst`, `performance-crossover.rst`, `security.md`, `security-and-data.rst`, `claim-evidence-ledger.rst`, `example-execution-manifest.rst`, `backend-support.rst`, `reproduction.rst`, `release-notes.rst`, `examples.rst`, `large-module-audit.rst`, `api/`), 31 items in `stale.json`; 20 of them are "0.8.0 named as the shipping version" one-line substitutions.
3. Worst item: `docs/migrations/0.8.0.md:3` "Status: released 2026-08-22." — the migration guide that `migrations/0.8.1.md` and `release-notes.rst` designate as the authoritative per-surface list opens by asserting a release that never happened.
4. Structural items that need more than a word swap: the generated API reference is a month behind the tree (99 modules without pages, incl. ten top-level ones, leaving `:mod:`/`:func:` cross-refs without targets); `examples.rst` "Complete Inventory" omits 14 of 57 scripts; `large-module-audit.rst` line counts are stale on 30 of 33 modules and name a non-existent module; `reproduction.rst:70` links off-site to a dead path; `release-notes.rst:94-99` still describes the pre-D-0212 gate set; `claim-evidence-ledger.rst:29` "~90 families" has no counting rule that matches the 165 exported `*Distribution` classes.
5. Verified current by execution: 57 examples, 16,538 collected tests, 4 smoke / 7 benchmark, all extras, all scripts' `--help`, all 256 documented imports, all 696 API-page modules, all toctree entries, `conf.py` version = 0.8.1, `tests.yml`/`docs.yml` on `release/0.8.1`.
