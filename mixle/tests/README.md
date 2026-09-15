# Test Suite

The suite is migrating to pytest as the collection and CI harness while keeping
the existing `unittest.TestCase` tests intact.  Pytest markers in
`mixle/tests/conftest.py` provide the logical organization.

## Common Commands

```sh
python -m pytest -m fast
python -m pytest
python -m pytest -m full
python -m pytest -m "torch or mpi"
```

`fast` is the per-commit gate.  `full` is every test that is neither optional nor a benchmark --
`conftest.py` assigns it -- and is what full CI runs.  Pass one `-m` only: pytest keeps the last one,
so `-m full -m ""` silently selects everything, optional tiers included.  Optional
extras such as torch, MPI, UMAP, and platform-specific accelerators are marked
separately so they can be enabled in dedicated jobs.

## Marker Conventions

- `distribution`: distribution interfaces, density math, encoders, samplers,
  estimators, and support behavior.
- `enumeration`: finite/infinite support enumerators and quantized indexes.
- `fisher`: Fisher views, sufficient statistics, and model metrics.
- `htsne`: affinity construction and embedding behavior.
- `hmm`, `pcfg`, `latent`, `automatic`: subsystem integration. Bayesian
  behavior (conjugate priors, variational families) now lives in `mixle.stats`
  and is exercised under the `distribution`, `serialization`, and `stochastic`
  markers rather than a separate marker.
- `kernel`, `torch`, `numba`, `parallel`, `planner`, `mpi`: implementation
  backends and resource-planning surfaces.
- `stochastic`, `slow`, `benchmark`, `optional`: CI scheduling tiers.

When adding a new file, add it to `FILE_MARKERS` in `conftest.py`.  If a single
class or method is heavier than the rest of its file, add a token rule or an
explicit `pytest.mark.*` decorator.
