# Mixle Core scientific validity

Document ID: CORE-DOC-SCIENTIFIC-VALIDITY-001
Version scope: 0.8.x development
Owner: PRJ-CORE

Core supplies scientific computing primitives; it does not make every model
scientifically valid by construction. Validity depends on the formulation,
data, assumptions, inference route, diagnostics, and decision context.

## Required assumptions

Record units, domains, coordinate conventions, data-generating and missingness
assumptions, likelihood and prior choices, known and unknown quantities,
identifiability limits, numerical precision, and excluded effects.

## Invariants

Applicable checks include finite canonical quantities, normalized
probabilities, support behavior, covariance symmetry and positive
semidefiniteness, deterministic seeded behavior, conservation or symmetry
where modeled, monotone objective behavior where promised, calibrated
coverage, and explicit failure on unsupported operations.

## Verification and validation

Prefer analytic invariants and manufactured examples, then trusted reference
solutions, independent implementations, controlled data, and expert review.
Shared code or assumptions do not provide independent corroboration.

Validation data used to choose a model is not independent confirmation.
Empirical law discovery, calibration, and model comparison must keep selection
and confirmation roles explicit.

## Uncertainty and limits

Report aleatoric, epistemic, measurement, model-discrepancy, and numerical
uncertainty when applicable. A calibrated result is valid only for the tested
population, model, data regime, and operating conditions. Low-power tests,
non-identifiability, distribution shift, extrapolation, and optional-backend
differences remain visible limitations.

## Reproducibility

Record source revision, environment, dependencies, inputs or digests,
configuration, seeds, precision, backend, stopping criteria, commands,
outputs, and tolerances. Portable hashes must not depend on process-randomized
runtime behavior.

The detailed numerical stability contract is in
stability-and-missing-data.rst. Release-critical scientific claims require the
evidence level declared by their status requirement.
