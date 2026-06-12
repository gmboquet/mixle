# Test Suite

The suite is migrating to pytest as the collection and CI harness while keeping
the existing `unittest.TestCase` tests intact.  Pytest markers in
`pysp/tests/conftest.py` provide the logical organization.

## Common Commands

```sh
python -m pytest -m fast
python -m pytest
python -m pytest -m "slow and not optional and not benchmark"
python -m pytest -m "torch or mpi"
```

`fast` is the per-commit gate.  Full CI runs all non-optional tests.  Optional
extras such as torch, MPI, UMAP, and platform-specific accelerators are marked
separately so they can be enabled in dedicated jobs.

## Marker Conventions

- `distribution`: distribution interfaces, density math, encoders, samplers,
  estimators, and support behavior.
- `enumeration`: finite/infinite support enumerators and quantized indexes.
- `fisher`: Fisher views, sufficient statistics, and model metrics.
- `htsne`: affinity construction and embedding behavior.
- `hmm`, `pcfg`, `latent`, `bstats`, `automatic`: subsystem integration.
- `kernel`, `torch`, `numba`, `parallel`, `mpi`: implementation backends.
- `stochastic`, `slow`, `benchmark`, `optional`: CI scheduling tiers.

When adding a new file, add it to `FILE_MARKERS` in `conftest.py`.  If a single
class or method is heavier than the rest of its file, add a token rule or an
explicit `pytest.mark.*` decorator.
