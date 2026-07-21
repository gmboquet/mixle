# Mixle benchmark harnesses

This directory contains reproducible, correctness-gated benchmark harnesses. It does **not** contain a
published 0.8.0 performance claim. A timing becomes release evidence only when it is generated from the
exact candidate, retains the environment and version stamp, passes the parity checks, and is reviewed under
`docs/benchmark-methodology.rst`.

Run the CPU comparison panel from this directory:

```console
python run_benchmarks.py --quick --reps 1  # development smoke; writes results/results_quick.json
python run_benchmarks.py                    # evidence candidate; writes results/results.json
```

The comparison packages are optional. Install `scikit-learn`, `pomegranate`, and `hmmlearn` to exercise the
full head-to-head panel. Missing packages are recorded as failures rather than silently removed. Every
comparison uses the same data, initialization, iteration budget, and fixed thread budget. A result is valid
only when its final likelihood agrees within the harness tolerance.

Run the CUDA capability panel on designated hardware:

```console
python gpu_scaling.py  # writes results/gpu_results.json
```

CUDA output is not a CPU/GPU speedup claim. It must identify the device, exact Mixle version and commit,
workload, timing, memory, and failures. Multi-GPU claims require a separate retained multi-process receipt.

## Result lifecycle

- `results/` is for artifacts produced by the current release line. The provenance gate rejects missing or
  stale release stamps.
- `archive/pre-0.8/` contains historical observations. They are useful engineering history, but they are not
  evidence for 0.8.0 and must not be quoted as current performance.
- Quick runs are smoke checks, not publishable evidence.
- Never copy a number from the archive into README, documentation, or release notes. Re-run the harness on
  the exact candidate instead.

The current 0.8.0 candidate therefore makes no numerical speedup or GPU-performance claim until the retained
hardware run is complete. Correctness and API support are covered separately by deterministic tests and the
backend support matrix.
