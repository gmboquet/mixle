<!--
0.8.0 is a credibility/stability/proof release under a feature freeze. Until 0.8.0 ships, every PR
must map to an item in the 0.8.0 worklist; new-capability work goes to the post-0.8 backlog or stays
under `mixle.experimental`. See release-checklists/0.8.0.md and CONTRIBUTING.md.
-->

## Worklist item

<!-- The worklist/checklist item this PR advances, e.g. `Q5.3`, `A1.1`, `B7.1`. Required during the 0.8.0 freeze. -->

**Item:**

## Summary

<!-- What changed and why. If this is a new API or a behavior change, say so explicitly. -->

## Public API impact

- [ ] This PR does **not** change any public `__all__`.
- [ ] It changes the public surface, and I regenerated the manifest (`python scripts/gen_api_manifest.py`) and committed `api_manifest.json`.
- [ ] It adds public surface: a written exception is recorded in the release decision log (freeze rule — a change adding more public surface than it removes needs one).

## Claims impact

- [ ] No public claim (README, docs, benchmarks) is affected.
- [ ] A claim is affected and the relevant docs / claim-evidence ledger are updated with evidence at the required grade.

## Evidence

<!-- The exact command(s) run and the result, e.g. `pytest mixle/tests/foo_test.py -q` -> `12 passed`.
For stable-core changes, evidence should come from the built wheel, not an editable checkout. -->

```
```

## Checklist

- [ ] Tests added/updated and passing; `ruff check` + `ruff format --check` clean.
- [ ] Consumer suites for any edited module pass.
- [ ] I am not merging this PR myself, and I have not enabled auto-merge.
