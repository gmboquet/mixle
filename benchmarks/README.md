# benchmarks

Reproducible timing for mixle's inference paths. The point is **honesty before any "fast" claim**:
no speed number ships in the docs/README until it is reproduced here, and every timing is reported
next to the recovered-parameter error so speed is never shown without correctness.

```sh
python benchmarks/inference_speed.py            # full (n=20000)
python benchmarks/inference_speed.py --quick    # n=2000, faster
```

What it shows:

- **Conjugate vs MCMC on the *same model*** — the headline. mixle's speed story is *choosing the closed
  form* (a single O(n) pass) instead of running thousands of MCMC draws, not racing samplers. The
  conjugate row should be orders of magnitude faster than the MCMC row at the same accuracy.
- **EM** on a Gaussian mixture and a Gaussian HMM (Baum–Welch).
- **A heterogeneous record** — `(category, real, count-sequence)` fit as one model — which the
  single-density PPLs (Stan/Pyro/NumPyro/PyMC) cannot express cleanly.
- **Optional rivals** (pomegranate / NumPyro) only when installed; otherwise their rows are skipped,
  never faked.

This harness is the prerequisite for the "blazing fast" pillar of `notes/MIXLE_POSITIONING_AND_ROADMAP.md`
(workstream A6) and the guide for the fusion / JIT work (A1–A3): optimize against these numbers.
