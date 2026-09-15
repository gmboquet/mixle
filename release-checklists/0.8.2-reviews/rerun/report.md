# mixle 0.8.2 candidate acda5378 -- independent re-run of the campaign's blocking and repaired findings

Candidate `acda53784b9dedae6e053d2c7e2735e9f9deb71f`, wheel sha256 `4812d906fe41a0f2aa236436ef43435951f979b6ae73c75365be6d4a9abd72bb`.
Every probe ran from /tmp with PYTHONPATH unset through `reviews-082-ten/tools/pyt.py`, and each output's first line is the
`mixle.__path__` it loaded. `verify_env.py` gives `source_commit acda5378...` for venv-full and venv-base (NOTES.md, line ENV).
Context runs on the previous candidate (866078be, `reviews-082-ten/venv-full`) and on 0.8.1 (`candidate-081/venv`) are marked
`c866078be`, or labelled inside the file. Paths below are relative to `rerun/`. The command trail is in NOTES.md.

Bottom line: of the 19 rows below, 17 hold on everything they name and Q07-F06 is still the documented limitation. **B5
(Q04-F11) is PARTIAL.** The two models it names are
now right. An ordinary 8-parameter location model is still 141 posterior sds off at the default budget, and so is `how='sample'`
(V01-F01, blocking). The other repairs hold on their named scope. Around them I found three real defects: B2 on container
components (V01-F02), B6 on record models fed a mapping or a structured array (V01-F03), and B2 in the engine kernels
(V01-F04). There are also six minor findings and one docs finding.

## Verdicts

| finding | defect | verdict | evidence |
|---|---|---|---|
| Q01-F01 | B1 quantile domain | HOLDS | out/b1_quantile.venv-full.txt, out/b1_quantile.venv-base.txt, out/rerun_repros_a.venv-full.txt |
| Q08-F08 | B1 quantile domain (Uniform/Laplace) | HOLDS | out/b1_quantile.venv-full.txt, out/b1_quantile.venv-base.txt |
| Q01-F02 | B2 support-limited mixing | HOLDS on the named families and routes (adjacent failures: V01-F02, V01-F04, V01-F05) | out/b2_unit.venv-{full,base}.txt, out/b2_mix_{repro,cont,count}.venv-full.txt, out/b2_mix_all.venv-base.txt, out/b2_adjacent.venv-full.txt, out/rerun_repros_a.venv-full.txt |
| Q02-F02 | B2 count families | HOLDS (a non-integer is refused at encode, per the owner ruling) | out/b2_unit.venv-{full,base}.txt, out/b2_mix_repro.venv-full.txt, out/b2_mix_count.venv-full.txt, out/rerun_repros_a.venv-full.txt |
| Q02-F01 | B3 terminal-state HMM readouts | HOLDS (adjacent minor: V01-F10) | out/b3_terminal.venv-{full,base}.txt, out/b3_quantized.venv-full.txt, out/b3_edges.venv-full.txt, out/rerun_repros_a.venv-full.txt |
| Q04-F01 | B4 restored conjugate predict | HOLDS (adjacent minor: V01-F06, V01-F07) | out/b4_conjugate_pickle.venv-full.txt, out/v_deepcopy_predict.txt, out/rerun_repros_a.venv-full.txt |
| Q04-F11 | B5 ensemble default budget | **PARTIAL**: the finding's two models hold; an 8-parameter location model is 141-149 sds off at the default and still 148 off at burn=20000 (V01-F01) | out/b5_ensemble_{diag,grouped,grouped2}.venv-full.txt, out/b5_adjacent.venv-full.txt, out/b5_dim_sweep_{a,b}.venv-full.txt, out/v_ensemble_default_d8.{venv-full,venv-base,c866078be}.txt, out/v_ensemble_d8_controls.venv-full.txt |
| Q07-F06 | B5 explicit short burn | known limitation, still wrong as documented (burn=500/draws=1000: 8.80 sds off, 8/18 groups); not counted | out/b5_ensemble_grouped2.venv-full.txt |
| Q05-F01 | B6 production front door | HOLDS: no verb reads column names or characters any more (residual on the finding's own Monitor line: V01-F09) | out/b6_scoring.venv-full.txt, out/b6_fit_verbs.venv-full.txt, out/b6_remaining.venv-full.txt, out/rerun_repros_a.venv-full.txt |
| Q09-F01 | B7 fitted TreeHMM JSON | HOLDS | out/b7_treehmm_json.venv-{full,base}.txt |
| Q06-F01 | vdata= front door | HOLDS | out/b6_fit_verbs.venv-full.txt (vdata section), out/rerun_repros_a.venv-full.txt |
| Q06-F02 | best_of DataFrame / dict_values | HOLDS | out/b6_fit_verbs.venv-full.txt, out/rerun_repros_a.venv-full.txt |
| Q06-F03 | learn_bayesian_network front door | HOLDS | out/b6_fit_verbs.venv-full.txt, out/b6_remaining.venv-full.txt, out/rerun_repros_b.venv-full.txt |
| Q06-F04 | production routes read frames | HOLDS | out/b6_scoring.venv-full.txt, out/rerun_repros_b.venv-full.txt |
| Q06-F05 | record models on lifecycle/production | HOLDS for DataFrame and dict rows (adjacent real: V01-F03 on a mapping or structured array) | out/b6_scoring.venv-full.txt, out/v_record_mapping_drift.venv-full.txt, out/rerun_repros_b.venv-full.txt |
| Q02-F08 | learn_bayesian_network spellings | HOLDS | out/rerun_repros_b.venv-full.txt, out/b6_fit_verbs.venv-full.txt |
| Q03-F04 | learn_structure front door | HOLDS on its surface (sibling learners its notes name: V01-F08) | out/b6_remaining.venv-full.txt, out/rerun_repros_b.venv-full.txt, out/v_learn_mixture_structure_frontdoor.venv-full.txt |
| Q02-F09 | TreeHMM write-only JSON | HOLDS | out/b7_treehmm_json.venv-{full,base}.txt, out/rerun_repros_b.venv-full.txt |
| Q01-F04 | diagnosis for partially -inf rows | HOLDS | out/b2_adjacent.venv-full.txt, out/rerun_repros_b.venv-full.txt |

## What was attacked, per defect

- **B1.** `probes/b1_quantile.py` walks every module under `mixle.stats` for classes that expose `quantile` and finds 31, all
  constructed. Each is called with q = -0.1, 1.1, NaN, inf, -inf, float32(1.1), float64 NaN, a 0-d array holding 1.5, -1e-12 and
  1+1e-15. Every out-of-domain q is refused with `<Family>.quantile: q must be in [0, 1]`. q = 0, 1, 0.3, int 0/1, float32 and a
  0-d array all answer. `density_quantile` refuses too, and `ops.quantize` still works on every continuous family. Zero failures
  on both venvs.
- **B2.**
  - Unit contract: 21 support-limited families, each with out-of-support values in list, float64, float32 and int spellings. The
    encoder admits the value, `seq_log_density` equals the scalar -inf, `seq_update` refuses weights 1 and 1e-9 and ignores
    weight 0 (the estimate matches the clean one), and `seq_initialize` ignores weight 0. NaN is refused everywhere, and count
    families refuse 1.5 and inf.
  - Latent models: 14 continuous families with a Gaussian sibling, across Mixture (2 seeds), float32, HeterogeneousMixture and HMM
    with and without numba, pass 84/84. 10 count pairs, across 3 seeds, float/list/int, Hetero and HMM with and without numba,
    pass 80/80. venv-base passes 174/175; the one exception is the ruling-covered Poisson 1.5.
  - The finding reproductions hold verbatim. NumpyEngine, TorchEngine and TorchEngine float32 fits, `init='kmeans++'`, `fit`,
    `best_of` and HMM `steady_state_init` all hold.
  - Failures:
    - Container components (V01-F02).
    - The `initialize()` verb (V01-F05).
    - Standalone engine kernels (V01-F04).
    - The scalar `update`/`initialize` zero-weight refusal on 14 families is deferred Q01-F03 and is not counted.
- **B3.** Brute-force enumeration over admissible paths checks `log_density`, `seq_posterior`, the marginals, `mode`, `viterbi`,
  `seq_viterbi` (blocked and numba layouts), `sample`, `samples`, `entropy`, `log_likelihood` and `posterior_predictive`. Cases: 8
  hand-built models (length-1 included) and 60 random models (K 2-4, L 1-6), mixed-length batches, the Quantized subclass (21/21),
  and a fitted 3-state model (max diff 4e-14, 0 inadmissible paths). All agree on both venvs. The one gap is impossible sequences
  (V01-F10). `seq_posterior(filtered=True)` does not force its last row onto terminal states; that is a filtering-semantics choice
  outside B3.
- **B4.** Nine hand-table conjugate pairs plus NIG, under how=auto/conjugate/posterior. The live `predict` integrates. After a
  pickle at protocol 2 or 5, `predict()` refuses by name, while `summary`, `explain_fit` and `sample` survive. A cross-process
  restore also refuses. Adjacent gaps: `deepcopy` (V01-F06) and the conjugate-mixture route (V01-F07).
- **B5.** DiagGaussian(5) at the default budget, seeds 0/1/2, is 0.19/0.08/0.06 sds off with sd ratio 0.97-1.06. `how='sample'` is
  identical, and `chains=2` gives r_hat 1.01. The 18-group model is 0.12/0.14/0.18 sds off with 0/18 groups beyond 3 sds, and the
  notebook spelling `max_its=300` and `chains=2` also hold. A 30-group model and a wide-prior scalar hold. **A d=8 or d=12 location
  model with ordinary spread fails (V01-F01).**
- **B6.** 21 input spellings across 6 fit verbs and the `vdata=` route: every table spelling fits with the same n on every verb, and
  str/bytes/matrix/masked/0-d/timedelta64/datetime64 and an empty mapping are refused by name, naming the verb. The scoring verbs
  (`Service.score`, `detect_drift`, `score_drift`, `Monitor.check`/`update`, `Service(reference=)`) match on DataFrame, mapping,
  structured array, generator and dict_values, and refuse by name on the others. Record models work with DataFrame and dict rows.
  Gaps: V01-F03, V01-F08, V01-F09.
- **B7.** Constructed, fitted (max_its 8 and 1, warm start), Q02-F09 hand-built, Poisson K=3, Categorical emissions, and a hand-built
  model whose memo is warm after scoring. Each round-trips through `dump_models`, `to_json`, `utils.to_json` and pickle. The loaded
  model scores identically and re-dumps after use. `Model.deploy` writes JSON and `Model.load` scores identically. Holds on both
  venvs.

## New defects

Full records, with observed, expected, reproduction and notes, are in `findings.json`.

### V01-F01 (blocking): B5 is partial. The ensemble default budget is still wrong on an 8-parameter location model.
DiagGaussian(8, mean=free(8), var=0.25), 300 rows, column means [0, 2, 1, 3, 4, -6, 9, -3]:
- `how='ensemble'` at the default budget is 141.37 sds off with seed 0 and 148.73 with seed 1 (v5 mean -1.91 against exact -5.99,
  reported mcse 0.008). `how='sample'` gives the same 141.37.
- burn=20000 is still 148.10 off. walkers=100 gives 0.05, NUTS 0.06 and HMC 0.10.
- venv-base gives the same numbers. On 866078be it was 136.7/151.4.
- A sweep with d=12 gives 17-36 sds off.

Mechanism: every real slot starts at the same scalar data mean, and 2(d+1) walkers cannot travel to coordinates far from it.
Evidence: `probes/v_ensemble_default_d8.py`, `out/v_ensemble_default_d8.*.txt`, `out/v_ensemble_d8_controls.venv-full.txt`,
`out/b5_dim_sweep_*.txt`.
```
import numpy as np
from mixle.ppl import DiagGaussian, free
RS = np.random.RandomState; locs = [0.0, 2.0, 1.0, 3.0, 4.0, -6.0, 9.0, -3.0]; r = RS(0)
X = np.stack([r.normal(l, 0.5, 300) for l in locs], axis=1)
v = free(8, name='v'); s = DiagGaussian(8, mean=v, var=np.full(8, .25)).fit(X.tolist(), how='ensemble', rng=RS(0)).summary()
print(s['v5']['mean'], X.mean(0)[5], s['v5']['mcse'])   # -1.9095 -5.9905 0.0079 (posterior sd 0.0289)
```

### V01-F02 (real): B2 does not reach support-limited leaves inside Composite/Sequence/Optional components.
Mixture[Composite(G, Poisson), Composite(G, Geometric)] on records with 32 zero counts:
- Refused at initialization by the Geometric support guard on seeds 1, 2 and 3.
- A warm start from the generating model fits, and the flat Mixture[Poisson, Geometric] on the same counts fits.
- Mixture[Sequence(Poisson), Sequence(Geometric)], HMM[Composite(...)] and Mixture[Optional(G), Optional(Exponential)] are also
  refused.

The same happens on venv-base and on 866078be. Cause: `supported_rows` exists only on leaf accumulators, and no combinator
aggregates it. Evidence: `probes/v_nested_support.py`, `out/v_nested_support.*.txt`, `out/b2_adjacent.venv-full.txt`.

### V01-F03 (real): a record model reports drift on identical data when given a mapping of columns or a structured array.
- `detect_drift(m, cols, cols)` returns True ('score KS 1.000', '0.000 scorable'). So do `Monitor.check(cols)` and
  `Service(reference=cols).check_drift(cols)`.
- `Monitor(m, est, df).update(cols)` then crashes with TypeError.
- Every fit verb and `Service.score` refuse the same mapping: `optimize`, `fit`, `Model.fit`, `fit_with_provenance`, `evaluate`.

Evidence: `probes/v_record_mapping_drift.py`, `out/v_record_mapping_drift.venv-full.txt`.

### V01-F04 (real): engine kernels score out-of-support rows as possible.
With numba installed, the NumpyEngine kernel scores:
- Bernoulli 2 as -2.05 and -1 as +0.49.
- Geometric 0 as -0.85 and -1 as -0.49.

The numpy scorer gives -inf for all four. Torch and the base NumpyEngine give +inf/NaN for LogSeries and NaN for Beta, InverseGamma
and LogGaussian.

Public route: `optimize(train, BernoulliEstimator(), vdata=[0,1,1,0,2,-1], engine=NumpyEngine())` fits and prints a validation
log-likelihood of -4.78; the default route refuses. On 866078be both routes refused at encode, so the B2 Bernoulli encoder change
opened this route.

Evidence: `probes/v_engine_oos_scores.py`, `probes/v_engine_vdata2.py`, `out/v_engine_oos_scores.*.txt`, `out/v_engine_vdata2.*.txt`,
`out/b2_engine_scorers.venv-full.txt`.

### V01-F05 (minor): `initialize()` on a mixture or HMM ignores component support.
`initialize(counts, Mixture[Poisson, Geometric])` and the Gaussian+Exponential Mixture and HMM equivalents are refused by the support
guard, while `optimize`, `fit` and `best_of` succeed. Evidence: `out/b2_adjacent.venv-full.txt`.

### V01-F06 (minor): `copy.deepcopy` of a live posterior fit loses `predict()`.
The error falsely says the posterior "was restored from a pickle". This is a regression against 0.8.1, where deep-copied fits
integrated (conjugate 2.300, mcmc 2.467). Evidence: `probes/v_deepcopy_predict.py`, `out/v_deepcopy_predict.txt`.

### V01-F07 (minor): a mixture-of-conjugate-priors fit cannot be pickled.
Pickling raises AttributeError on a local closure, contradicting "every posterior-bearing fit pickles". Evidence:
`out/b4_conjugate_pickle.venv-full.txt`.

### V01-F08 (minor): the other structure learners still use `list(data)`.
- `learn_mixture_structure(DataFrame|mapping)` and `mixture_structure_health(mot, DataFrame)` are refused with "record 0 is a single
  str".
- On `np.matrix`, `learn_mixture_structure` raises a raw TypeError.
- `learn_bayesian_network` and `learn_structure` on list-of-dict rows raise KeyError: 0, while `optimize` fits them.

Evidence: `out/v_learn_mixture_structure_frontdoor.venv-full.txt`, `out/b6_misc.venv-full.txt`.

### V01-F09 (minor): the finding's own `Monitor.update(df2)` line still retrains.
The new model is a categorical over the labels plus the frame's 2-tuples. Yet `optimize` and `fit_with_provenance` refuse that
frame, and the drift report calls it 0% scorable. Evidence: `out/b6_remaining.venv-full.txt`.

### V01-F10 (minor): `viterbi`/`seq_viterbi` return a path for an impossible sequence.
They return [0, 0, 0] for a sequence the terminal restriction makes impossible, where `latent_posterior().mode()` raises
ImpossiblePosteriorError. The same happens on non-terminal HMMs. Evidence: `out/b3_edges.venv-full.txt`.

### V01-F11 (docs): the CHANGELOG at acda5378 cites files the commit does not contain.
The cited `release-checklists/0.8.3-followups.md`, D-0217 and `release-checklists/0.8.2-reviews/pass-NN/` exist only uncommitted in
the worktree.

DONE rerun 11
