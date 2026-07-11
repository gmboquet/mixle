# Release checklists

This folder is the tracked, public record of what has to be true before a `mixle` version ships.

- **`<version>.md`** (`0.7.0.md`, `0.7.1.md`, …) — one checklist per release, created when
  preparation begins and updated in place *with evidence* as gates are verified. It stays in git
  history after the release ships, so anyone can see exactly what was checked, how, and when.
- **`lessons-learned.md`** — a cumulative, cross-release record of what went wrong (or nearly did),
  where each lesson is tied to the checklist gate that now catches it. This is what keeps the
  checklist from being a static wish-list: it grows from real failures.
- **`<version>-decisions.md`** (`0.8.0-decisions.md`, …) — the append-only release decision log for a
  version: every scope change, waiver, API break, benchmark-methodology change, and claim change, with
  date, alternatives, evidence, owner, reviewer, expiration, and worklist ID. No release decision may
  live only in a PR thread or a local note.

This is a checklist of **gates**, not a task list. An unchecked item means the release is not
ready, full stop — there is no "ship now, verify later" for anything marked here.

## Scope

This tracks the `mixle` package itself: the code in this repository, its own CI, its own version.
`mixle` ships alongside sibling packages (`mixle-knowledge`, `mixle-agent`, `mixle-mlops`,
`mixle-pde`, and others) as part of a coordinated release, but cross-package coordination —
lockstep versions, family co-install, publication order — is the release owner's responsibility and
tracked outside this repo. Where a family-level decision constrains something checked here (a
sibling's minimum required `mixle` version, for example), this checklist links to it rather than
duplicating it.

## Status legend

| Status | Meaning |
| --- | --- |
| `TODO` | Required, no evidence yet. |
| `PARTIAL` | Some evidence exists but doesn't cover the exact release commit, or is otherwise incomplete. |
| `DONE` | Verified against the exact commit that will be (or was) released, with evidence recorded inline. |
| `EXCLUDED` | Explicitly out of scope for this release, with a one-line reason. Silence is not the same as excluded. |

## Evidence discipline

A gate is `DONE` only when the entry names, at minimum:

- the command run;
- the commit SHA it was run against;
- the result (pass/fail, with the actual number — "4,890 passed" not "tests passed");
- the date.

"It passed in CI on some earlier commit" does not satisfy a gate for the current tip. A release
branch that gained commits since the last green run needs a fresh run — CI status against an old
SHA is not evidence about the current one.

## What's checked here

The gate categories run pre-flight → verify → publish → post-publish:

- **Branch / CI state** — no open PRs, `main` not ahead, no pre-existing tag, CI green *on the exact
  tip*.
- **Version and metadata** — the version string is actually bumped, semver is right, changelog and
  migration notes are current.
- **Build and install clean** — a fresh venv installing the *built wheel* (not the dev tree, not
  `PYTHONPATH`), an import smoke, and a full public-module import sweep.
- **Dependency correctness** — declared deps match imports both ways, bounds are real and tested,
  optional deps stay optional.
- **Test rigor** — the *full* suite (not the fast default), across the supported Python/OS matrix,
  run against the clean install, with no known flakes.
- **Hygiene** — secrets, debug leftovers, and clean commit/author history.
- **Documentation and examples** — docs build strict, process references resolve *within the repo*,
  every shipped example/notebook actually executes.
- **Post-merge re-verification** — the confidence gates re-run against the exact tag commit, because
  green-per-branch isn't green-on-the-merge.
- **Reproducibility** — the resolved dependency set is captured, and the *previous* release still
  installs.
- **Publication and rollback** — build → TestPyPI dry run → PyPI → tag → post-publish verify, in
  dependency order, with the rollback path (patch, or yank-not-delete) understood first.
- **Sign-off** — a named, dated release decision, only after everything above is `DONE` or
  `EXCLUDED`.

Each release's file spells these out with the exact commands for that release's CI configuration,
since the gates themselves change between releases (a new CI job, a new supported Python version, a
new optional extra) — and each new gate should trace back to a lesson in `lessons-learned.md`.

## Using this for a new release

1. Copy the most recent release's file as a starting template — the shape doesn't change much
   release to release, but re-verify every gate; don't carry over old evidence.
2. Fold in every open lesson from `lessons-learned.md`: each one should already correspond to a gate
   in the template. If a past lesson has no gate yet, add it before you start.
3. Update the version-specific specifics (target version, changelog section, any new gates).
4. Work through it in roughly top-to-bottom order — branch/CI state and version metadata gate
   everything after them.
5. Commit progress as you go, in the open, so the file's git history shows how the release
   actually got verified, not just a final "all green" snapshot.

## Closing the loop after a release

The checklist only gets better if failures feed back into it. After each release — or each release
*attempt* that got blocked — run a short retrospective (`lessons-learned.md` describes the steps):
write up what went wrong or nearly did, and for each item, add or strengthen the gate that would
have caught it, in the same change. A lesson that doesn't change the checklist is just a story.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the day-to-day PR/test/lint conventions this
checklist assumes.
