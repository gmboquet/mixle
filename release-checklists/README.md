# Release checklists

This folder is the tracked, public record of what has to be true before a `mixle` version ships.
One file per release: `0.6.3.md`, `0.6.4.md`, and so on. The file is created when release
preparation begins and updated in place — with evidence — as gates are actually verified. It stays
in git history after the release ships, so anyone can see exactly what was checked, how, and when.

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

The gate categories are: branch/CI state, version and package metadata, a clean build-and-install
(not just "tests pass in the dev tree" — a fresh venv installing the built wheel), dependency
correctness, the full test gate (not just the fast default), a hygiene scan (secrets, debug
leftovers), documentation, and example/notebook execution. Each release's file
spells these out with the exact commands for that release's CI configuration, since the gates
themselves can change between releases (a new CI job, a new supported Python version, a new
optional extra).

## Using this for a new release

1. Copy the most recent release's file as a starting template — the shape doesn't change much
   release to release, but re-verify every gate; don't carry over old evidence.
2. Update the version-specific specifics (target version, changelog section, any new gates).
3. Work through it in roughly top-to-bottom order — branch/CI state and version metadata gate
   everything after them.
4. Commit progress as you go, in the open, so the file's git history shows how the release
   actually got verified, not just a final "all green" snapshot.
5. See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the day-to-day PR/test/lint conventions this
   checklist assumes.
