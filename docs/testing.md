# Mixle Core testing

Document ID: CORE-DOC-TESTING-001
Version scope: 0.8.x development
Owner: PRJ-CORE

## Strategy

Tests are organized by claim, not only by module. A behavior change should
cover the successful path, invalid inputs, boundary values, preserved
compatibility, and any scientific invariant it claims.

Pytest collects both suffix-named files ending in _test.py and prefix-named
files beginning with test_. Test functions begin with test_, and legacy
unittest classes remain supported.

## Local tier

Run only directly affected nodes with serial execution and no marker filter.
Each local command has a hard 30-second deadline. A timed-out command is
evidence of an incomplete local check, not a pass.

## Hosted tiers

- fast jobs exercise the commit gate on supported Python and platform entries;
- full jobs exercise non-optional correctness with coverage;
- minimum-version jobs check declared dependency floors;
- clean-wheel jobs check packaging and import behavior;
- optional jobs exercise selected accelerators, connectors, and distributed
  backends when their dependencies are installed.

The workflow file is .github/workflows/tests.yml. Marker ownership is
centralized in mixle/tests/conftest.py.

## Evidence mapping

Record exact node IDs or file selections, revision, environment, elapsed time,
result, and known omissions. Scientific claims additionally record seed,
tolerance, data or fixture identity, and the invariant being checked.

## Coverage limits

Passing the base environment does not establish optional-backend parity.
Coverage percentage does not establish numerical validity. Stochastic success
at one seed does not establish calibration or convergence. Release evidence
must combine functional, negative, compatibility, scientific, packaging, and
supported-environment results applicable to the changed surface.
