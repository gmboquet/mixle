# Release lessons learned

A cumulative, cross-release record of things that went wrong — or nearly did — while releasing
`mixle`, so the same mistake is caught by process next time instead of by luck.

## The rule that keeps this useful

A lessons-learned log that is only a list of war stories rots into something nobody reads. This one
has one rule: **every lesson must terminate in a checklist gate.** Each entry ends with either

- **→ Gate:** the checklist item (existing or newly added) that now catches this class of failure; or
- **→ No gate:** an explicit, dated decision that a gate isn't worth it, and why.

If a lesson can't be turned into a gate or a deliberate no-gate decision, it isn't done being
understood yet. When you add a lesson here, you add or strengthen the gate in the current release
checklist in the *same* change — that is the whole point of writing it down.

## How to run a release retrospective

After each release (or after a release *attempt* that got blocked), before the memory fades:

1. Walk the release checklist and this session's history. For every gate that was `TODO`/`PARTIAL`
   at the end, every fire drill, and every "we almost shipped X" — write it up below under that
   release's heading.
2. For each, find the gate that would have caught it. If the gate exists, note that it worked (or
   why it didn't fire). If it doesn't exist, add it to the checklist template and reference it here.
3. Keep entries concrete: the actual symptom, the actual root cause, the actual command that now
   catches it. "Be more careful" is not a lesson.

---

## 0.6.3

The 0.6.3 cycle was a large, fast, multi-agent build (~160 commits merged in a few days), which is
exactly the condition under which release discipline earns its keep. Most of these were caught
during the build or a pre-release review, not in the wild — but each is a real thing that happened.

### L-0.6.3-1 — Shipped a release branch with the previous version string

**What happened.** `release/0.6.3-generic-capabilities` carried `version = "0.6.2"` in
`pyproject.toml` the entire time — the branch name said 0.6.3, the metadata said 0.6.2. Left
unchecked, the build would have produced a `mixle-0.6.2` wheel from the 0.6.3 branch.

**Root cause.** The version bump was treated as a "do it at the end" step with nothing enforcing it,
and a branch name is not metadata.

**→ Gate:** §2, *`pyproject.toml` version equals the release version* — verified with
`grep -nE '^version' pyproject.toml` and a `git tag` check that the target tag doesn't already exist,
as a hard blocker before any build.

### L-0.6.3-2 — `PYTHONPATH`/editable runs hid a real packaging failure

**What happened.** A persistence test spawned a subprocess that failed with
`No module named 'mixle.task'` — but *only* under a `PYTHONPATH`/editable-install run. It passed the
moment the actual built wheel was installed. The "failure" was a harness artifact, and only a clean
install could prove that.

**Root cause.** A warm dev tree (editable install, `PYTHONPATH` on the path) resolves imports that a
real installed wheel would not. Tests run against the working tree do not validate the artifact.

**→ Gate:** §3, *fresh, isolated venv install from the built wheel* (never editable, never
`PYTHONPATH`) + import smoke + running the suite against that install.

### L-0.6.3-3 — The default test command silently ran a subset

**What happened.** The repo's default `pytest` is `-m fast`. On this codebase that selected ~4,440
tests and *silently deselected ~849* slower ones. A developer running `pytest` and seeing green had
not run the suite a release needs.

**Root cause.** A fast per-commit gate and a release gate are different things, and pytest deselects
quietly (exit 0, no warning).

**→ Gate:** §5, run the *full* non-optional suite (`pytest -m "not optional and not benchmark"`),
and the `optional` extras suite — the checklist names the exact selection so "I ran the tests" can't
mean the fast subset.

### L-0.6.3-4 — Missing optional-backend guards broke the no-torch CI lane, repeatedly

**What happened.** New test files (`eval_harness_test`, `scaling_laws_test`, `training_health_test`,
`doe_amplify_test`, and others) imported torch — or called a torch-only path like
`GaussianProcessRegressor` — with no skip guard. On the torch-free `fast`/`full` CI lanes these
*errored during collection* instead of skipping, reddening CI. This recurred several times across the
batch merges.

**Root cause.** The base install is deliberately torch-free (torch is an optional extra), but a
test that needs an optional backend must declare that with `pytest.importorskip(...)` or
`@unittest.skipUnless(_HAS_TORCH, ...)`. It's an easy omission and nothing at authoring time forces
it.

**→ Gate:** §4 *optional deps stay truly optional* + §5 running the full suite on the torch-free
install, which is exactly the lane that surfaces an unguarded import. Recurring-pattern note in the
checklist: a new test touching an optional backend must guard.

### L-0.6.3-5 — A public family added without registering it in its enforcing catalog

**What happened.** New copula distributions were exported from `mixle.stats` but not added to the
seed-repeatability catalog that `sampler_seed_test` checks — twice (first Frank/Clayton/StudentT,
then Gumbel/CVine/DVine/RVine). Each broke the release-branch test suite and, through it, every open
PR's CI.

**Root cause.** A public export and its enforcement catalog must move in lockstep, and the coupling
isn't obvious from either file alone. (The good news: the catalog test *is* the enforcement — it
failed loudly rather than letting an unverified sampler ship.)

**→ Gate:** the catalog test itself is the standing gate, exercised by §5's full suite; plus a
`CONTRIBUTING`-level note that adding a public distribution means updating its catalog test in the
same change.

### L-0.6.3-6 — A boundary violation landed because the full suite wasn't run on the merged tip

**What happened.** Copula-selection logic was added *inside* `mixle/inference/estimation.py`,
importing concrete `mixle.stats` distributions (even function-locally) — a violation of a boundary
rule enforced by an AST-scanning test (`estimation.py` is a high-level compute utility that must
never import concrete distributions). Each contributing branch's own CI was green; the violation only
appeared once several branches were merged and the full suite ran against the combined tree.

**Root cause.** Green-on-each-branch is not green-on-the-merge. A rule enforced by a test only helps
if that test actually runs against the state you're shipping.

**→ Gate:** §9 (new), *re-verify the full suite against the post-merge / current tip*, not just
per-branch CI — the merge is where cross-branch interactions surface.

### L-0.6.3-7 — Cross-platform numeric flakiness (passes on dev, reds on CI)

**What happened.** Seeded exact-t-SNE trajectories and a couple of correlation/separation thresholds
landed at different values across numpy/BLAS builds — e.g. separation 1.414 locally (macOS) vs 1.2491
on the Linux CI py3.12 wheels, a hair under a 1.25 pin. Green locally, red on CI.

**Root cause.** A seed does not make floating-point BLAS deterministic across platforms; a threshold
pinned to the dev machine's exact value is a platform-specific assertion in disguise.

**→ Gate:** §5 *no known flaky tests* + the evidence rule that CI (the real target OS), not the dev
machine, is the source of truth. Platform-sensitive thresholds are pinned with margin below every
observed trajectory, asserting the qualitative claim, not a knife-edge number.

### L-0.6.3-8 — A stochastic test passed alone but reddened in the full run

**What happened.** `structure_learning::test_responsibilities_recover_clusters` passed 3/3 in
isolation but went red once in the full suite — a stochastic EM-recovery test sensitive to RNG/order
pollution from other tests.

**Root cause.** A test that depends on global RNG state (rather than its own in-test seed) is order-
dependent, so "passes when I run it" and "passes in the gate" diverge.

**→ Gate:** §5 *no known flaky tests* — detect by looping a suspect test and by varying order; fix by
seeding the test's own RNG. A gate that reds at random destroys the confidence a release exists to
build.

### L-0.6.3-9 — A dependency API used beyond its declared lower bound

**What happened.** The grammar serializer called `networkx.node_link_data(edges=...)`, which exists
only in networkx ≥ 3.4, and raised `TypeError` on an environment's networkx 3.0. The dependency was
declared, but the *bound* didn't reflect the API the code actually used.

**Root cause.** "The dependency is declared" and "the declared range actually works" are different
claims; the second needs a version-bounds review against the real API surface.

**→ Gate:** §4 *version bounds are real, not just present* — spot-check version-sensitive deps
(`networkx`, `numpy`, `scipy`, `torch`) against the API used, and pin the lower bound to it.

### L-0.6.3-10 — CI drifted far behind the tip while everyone assumed "it's green"

**What happened.** The last *confirmed*-green CI run was ~90 commits behind the current release tip.
"CI is green" was true about a commit nobody was about to ship.

**Root cause.** CI status is about a specific SHA, and a fast-moving branch outruns its last full
run. "Green" with no SHA attached is not a fact about the tip.

**→ Gate:** §1 *CI is green on the exact current tip*, plus the README evidence rule: green-on-an-
old-SHA does not satisfy a gate for the current one; a branch that gained commits needs a fresh run.

### L-0.6.3-11 — Reading a stale local branch instead of `origin/*`

**What happened.** During the review a local checkout trailed `origin` by dozens of commits, which
produced wrong conclusions about branch state until the reads were switched to `origin/*` refs.

**Root cause.** A local branch is a cache, not the truth; release decisions made from it can be
decisions about a state that no longer exists.

**→ Gate:** §1 phrasing — every branch/tip/ahead-behind check reads `origin/*` refs (or does a fresh
`git fetch` first), never a possibly-stale local ref.

### L-0.6.3-12 — The release process pointed at a doc nobody releasing could reach

**What happened.** `CONTRIBUTING.md` and `CHANGELOG.md` referenced the release checklist as living in
`notes/mixle-development-agent-rules.md` — a path in a *separate, private, unpublished* sibling repo.
Anyone who cloned `mixle` and followed the process hit a dead reference.

**Root cause.** The process that governs a repo has to live *in* that repo. A pointer to a
private/inaccessible location is the same as no process for an outside contributor. (This is why
`release-checklists/` exists as a tracked, in-repo folder — this file included.)

**→ Gate:** §7 *docs/process references resolve within the repo* — the checklist, and any process doc
it links, live in `mixle` itself, not in a sibling repo.
