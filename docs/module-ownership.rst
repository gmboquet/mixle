Module ownership and migration
==============================

``module_ownership.json`` classifies every current top-level public module as
``retain``, ``narrow``, ``migrate``, ``deprecate`` or ``experimental``. A test
fails whenever a top-level module is added without a decision.

Classification is not deletion. A ``migrate`` decision identifies the future
owner while the current Core surface remains compatible. Removal requires a
published replacement, caller inventory, adapters, migration documentation,
and at least one governed release-window deprecation gate. ``narrow`` retains a
quantitative center while preventing new workflow ownership from accumulating.

The artifact is the machine-readable source; the maturity and API manifests
continue to gate what is actually public and supported in each release.
