# mixle 0.8.1 release-candidate adversarial review — PASS 02

- Focus: latent-structure models — mixtures (MixtureEstimator, DiagonalGaussian, HeterogeneousMixture), HMMs and variants (lookback, segmental, tree, quantized), Markov chains, LDA, Chow-Liu / Bayesian networks, posterior/responsibility APIs.
- Wheel: `mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d`, tree `6040ea38` (branch release/0.8.1).
- Version verification (run from `<review-root>/reviews-081/pass-02/`):
  `python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"` →
  `0.8.1 <review-root>/candidate-081/venv/lib/python3.12/site-packages/mixle`
- Interpreter: Python 3.12.12; numpy 2.5.3, scipy 1.18.1; **numba is NOT installed** in the candidate venv (it is only the `[numba]` extra), which is exactly the base-install configuration and matters for F01.
- Every nbconvert run used `--ExecutePreprocessor.kernel_name=python3` (the venv's own kernelspec); a stale `mixle-notebooks-venv` kernel exists on the machine and was deliberately bypassed. macOS has no `timeout`; a small Python wrapper (`tmo.py`) enforced wall-clock limits.

## Executed corpus (all from the work dir, on the candidate wheel)

| # | Notebook (corpus-081/notebooks-repo/notebooks/) | exit | wall |
|---|---|---|---|
| 1 | data_science/hmm_from_scratch.ipynb | 0 | 30 s |
| 2 | data_science/gaussian_mixtures_in_depth.ipynb | 0 | 52 s |
| 3 | data_science/markov_chains_for_sequences.ipynb | 0 | 25 s |
| 4 | data_science/topic_modeling_lda.ipynb | 0 | 298 s |
| 5 | data_science/character_models_chow_liu.ipynb | 0 | 20 s |
| 6 | tutorials/latent_variable_models.ipynb | 0 | 72 s |
| 7 | data_science/model_based_clustering_evaluation.ipynb | 0 | 18 s |
| 8 | exploration_geoscience/well_log_facies_hmm.ipynb | 0 | 11 s |
| 9 | data_science/language_detection_hmm.ipynb | 0 | 22 s |
| 10 | data_science/latent_models_in_practice.ipynb | 0 | 57 s |

Executed copies: `executed-1.ipynb` … `executed-10.ipynb`; logs `nb-N.log`. Scan of all outputs: 0 error cells, no `nan`/`-inf` in any output. Stderr consists only of the intended "optimize() stopped at the max_its cap" UserWarnings (notebooks 6 and 10, where the notebook itself sets small `max_its`) and one DeprecationWarning (F10).

| # | Example (corpus-081/examples/) | exit | wall |
|---|---|---|---|
| 1 | gallery_structured_example.py | 0 | 9 s |
| 2 | semi_supervised_mixture_example.py | 0 | 6 s |
| 3 | lookback_hmm_example.py | 0 | 13 s |
| 4 | structured_hmm_example.py | 0 | 13 s |
| 5 | hierarchical_mixture_example.py | 0 | 26 s |
| 6 | joint_mixture_example.py | 0 | 6 s |
| 7 | latent_variable_models_example.py | 0 | 5 s |
| 8 | structure_learning_example.py | 0 | 11 s |

Logs `ex-N.log`. No example printed a false claim that I could detect; the semi-supervised example's three recovered components match its generating law to 2 decimals.

Attack scripts (all reproducible standalone): `atk_mix1.py`, `atk_mix2.py`, `atk_hmm1.py`, `atk_hmm2.py`, `atk_hmm_post.py`, `atk_lda.py`, `atk_mc.py`, `atk_mc2.py`, `atk_tree.py`, `atk_variants.py`, `ser_step.py`, `hang_probe.py`, `lb_recover.py`, `q_stop.py`; outputs `*.out`.

---

## P02-F01 (real) — `HiddenMarkovModelDistribution.seq_posterior` silently returns `None` on a base install (and always when `terminal_states` is set)

**Surface:** `mixle.stats.HiddenMarkovModelDistribution.seq_posterior`, also `QuantizedHiddenMarkovModelDistribution.seq_posterior`.

`hidden_markov.py:1608-1617`: `if not self.use_numba: return None`. `use_numba` defaults to `HAS_NUMBA`, which is `False` on the base install (numba is only an extra), and is forced to `False` whenever `terminal_states` is given (line 910). The docstring promises "forward-backward SMOOTHING marginals … matching every other `seq_posterior` in the package". The 0.8.0 changelog advertises "`seq_posterior` returns smoothing (not filtering) probabilities". The computation itself does not need numba: passing `use_numba=True` on this numba-less venv returns the correct list of `(T,k)` arrays (verified against brute-force enumeration to 7e-16).

**Reproduction:**
```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
true=S.HiddenMarkovModelDistribution(topics=[S.GaussianDistribution(-3,1.0),S.GaussianDistribution(3,1.0)],w=[0.5,0.5],
     transitions=[[0.9,0.1],[0.2,0.8]],len_dist=S.CategoricalDistribution({20:1.0}))
data=true.sampler(seed=1).sample(40)
m=optimize(data,S.HiddenMarkovModelEstimator([S.GaussianEstimator()]*2),max_its=30,rng=np.random.RandomState(1),print_iter=1000)
enc=m.dist_to_encoder().seq_encode(data)
print(m.use_numba, m.seq_posterior(enc))            # False None
print(m.seq_posterior(enc, filtered=True))          # None
m2=optimize(data,S.HiddenMarkovModelEstimator([S.GaussianEstimator()]*2,use_numba=True),max_its=30,rng=np.random.RandomState(1),print_iter=1000)
print(type(m2.seq_posterior(enc)))                  # <class 'list'>  -- works without numba installed
m3=optimize(data,S.HiddenMarkovModelEstimator([S.GaussianEstimator()]*2,terminal_states=[1],use_numba=True),max_its=3,rng=np.random.RandomState(1),print_iter=1000)
print(m3.use_numba, m3.seq_posterior(m3.dist_to_encoder().seq_encode(data)))   # False None
```
**Expected:** a list of `(T_i, k)` smoothing-marginal arrays (rows summing to 1), or an explicit error.
**Observed:** `None`, no warning, on the default configuration of a base `pip install mixle`; `None` unconditionally with `terminal_states`. `QuantizedHiddenMarkovModelEstimator(num_states=2, levels=[...])` fits give the same `None` (`atk_variants.py`, "post_type: NoneType").
**Notes:** `latent_posterior(seq).marginals()` and the `mixle.Model(...).fit(...).posterior(seq)` facade both work on the base install, so the changelog claim "`Model.posterior()` works on HMMs" holds; it is the documented low-level `seq_posterior` that silently no-ops. Every other `seq_posterior` I exercised (Mixture, LDA, TreeHMM, LookbackHMM) returns arrays on this install. `SegmentalHiddenMarkovModelDistribution` has no `seq_posterior`/`viterbi` at all (minor completeness gap, not counted separately).

## P02-F02 (real) — a fitted `ChowLiuTreeDistribution` cannot be serialized (`to_json`/`to_dict` raise)

**Surface:** `mixle.stats.ChowLiuTreeDistribution.to_json`, `.to_dict`; `mixle.utils.serialization.to_json`.

**Reproduction:**
```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
d=[tuple(str(v) for v in row) for row in [(1,1),(0,1),(1,0),(0,0),(1,1)]*20]
m=optimize(d, S.ChowLiuTreeEstimator([S.CategoricalEstimator()]*2), max_its=2, rng=np.random.RandomState(1), print_iter=1000)
m.to_json()     # SerializationError: callable <class 'str'> is not registered; use register_serializable_callable()
m.to_dict()     # same
m_int=optimize([(1,1),(0,1),(1,0),(0,0)]*20, S.ChowLiuTreeEstimator([S.IntegerCategoricalEstimator()]*2), max_its=2, rng=np.random.RandomState(1), print_iter=1000)
m_int.to_json() # SerializationError: callable <class 'int'> is not registered
```
**Expected:** a JSON string that `ChowLiuTreeDistribution.from_json` restores (as `IntegerChowLiuTreeDistribution`, MarkovChain, LDA, Mixture, HMM, Segmental/Tree/Quantized HMM all do — verified bit-identical log-densities).
**Observed:** `SerializationError` from the generic Chow-Liu distribution with any leaf family tried (categorical str, integer-categorical). The changelog says every mixle distribution "persists through the same safe artifact path"; this one cannot persist at all. The distribution stores a Python type (`str`/`int`) as its value coercer, which the serializer refuses.

## P02-F03 (real) — support-limited component families disagree between scalar and vectorized paths, making heterogeneous mixtures/HMMs unfit-able on legitimate data

**Surface:** `ExponentialDistribution.seq_log_density` (and the mixture/HMM encoders that call it); `MixtureDistribution`, `HeterogeneousMixtureDistribution`, `HiddenMarkovModelDistribution` with one Gaussian and one Exponential component.

**Reproduction:**
```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
e=S.ExponentialDistribution(1.0)
print(e.log_density(-1.0))                                       # -inf   (scalar: outside support = zero mass)
e.seq_log_density(e.dist_to_encoder().seq_encode([-1.0,1.0]))    # ValueError: Exponential requires x >= 0.
mm=S.MixtureDistribution([S.GaussianDistribution(-2,0.25), S.ExponentialDistribution(3.0)],[0.5,0.5])
print(mm.log_density(-1.0), mm.posterior(-1.0))                  # -2.9189  [1. 0.]   (scalar path: fine)
mm.seq_log_density(mm.dist_to_encoder().seq_encode([-1.0,1.0]))  # TypeError: MixtureDistribution could not encode the data ...
rng=np.random.RandomState(0); x=np.concatenate([rng.normal(-2,0.5,200), rng.exponential(3,200)])
optimize(x, S.MixtureEstimator([S.GaussianEstimator(), S.ExponentialEstimator()]), max_its=30, rng=np.random.RandomState(1), print_iter=1000)               # TypeError (same)
optimize(x, S.HeterogeneousMixtureEstimator([S.GaussianEstimator(), S.ExponentialEstimator()]), max_its=20, rng=np.random.RandomState(1), print_iter=1000)  # ValueError: Exponential requires x >= 0.
hm=S.HiddenMarkovModelDistribution(topics=[S.GaussianDistribution(-2,0.25), S.ExponentialDistribution(3.0)], w=[0.5,0.5], transitions=[[0.9,0.1],[0.1,0.9]])
print(hm.log_density([-1.0,1.0]))                                # -6.6535 (scalar path fine)
optimize([[-1.0,1.0,2.0,-2.0]]*20, S.HiddenMarkovModelEstimator([S.GaussianEstimator(), S.ExponentialEstimator()]), max_its=5, rng=np.random.RandomState(1), print_iter=1000)  # ValueError: Exponential requires x >= 0.
```
**Expected:** `seq_log_density` returns `-inf` for the out-of-support entry exactly as `log_density` does (the 0.8.0 changelog: components "score -inf, matching the mixture contract"); the mixture/HMM then assigns zero responsibility to the Exponential component for negative observations and the fit proceeds.
**Observed:** the vectorized path raises, so the model that `log_density` happily scores cannot be encoded, scored in batch, given responsibilities, or fitted whenever any observation lies outside one component's support. This is the fail-closed-guard-rejecting-legitimate-state pattern: a two-population "noise around zero + positive exponential tail" model is ordinary.

## P02-F04 (minor) — `LookbackHiddenMarkovModel` at `lag=0` with scalar emission families: internal crashes instead of a type rejection, and a sampler whose output the same model cannot score

**Surface:** `LookbackHiddenMarkovModelEstimator(lag=0)`, `LookbackHiddenMarkovModelDistribution(lag=0)`.

Module docstring: "With lag == 0 the model reduces to an ordinary hidden Markov model". The windows handed to the topics are length-1 *lists*, so scalar families are a type mismatch — but nothing says so.

**Reproduction:**
```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
rng=np.random.RandomState(0); seqs=[[float(v) for v in rng.normal(0,1,15)] for _ in range(40)]
optimize(seqs, S.LookbackHiddenMarkovModelEstimator([S.GaussianEstimator()]*2, lag=0), max_its=5, rng=np.random.RandomState(1), print_iter=1000)
# ValueError: shapes (600,1) and (600,) not aligned  (gaussian.py seq_update <- lookback seq_initialize); deterministic for every seed
optimize([list(rng.choice(['a','b'],10)) for _ in range(30)], S.LookbackHiddenMarkovModelEstimator([S.CategoricalEstimator()]*2, lag=0), max_its=5, rng=np.random.RandomState(1), print_iter=1000)
# ValueError: categorical labels must be hashable.
hb=S.LookbackHiddenMarkovModelDistribution([S.GaussianDistribution(-1,1),S.GaussianDistribution(1,1)], w=np.array([0.5,0.5]), transitions=np.array([[0.9,0.1],[0.1,0.9]]), lag=0, len_dist=S.CategoricalDistribution({5:1.0}))
s=hb.sampler(seed=1).sample(3)      # works: [[0.2398, 1.933, -2.124, 0.247, -1.062], ...]
hb.log_density(s[0])                # TypeError: float() argument must be a string or a real number, not 'list'
# Works: S.LookbackHiddenMarkovModelEstimator([S.SequenceEstimator(S.GaussianEstimator())]*2, lag=0)
```
**Expected:** either scalar emission families are accepted at lag=0 (the "ordinary HMM" reading) or the constructor/estimator rejects them with a message naming `SequenceEstimator`; in no case should `sampler()` accept a configuration that `log_density` cannot score.
**Observed:** deep internal errors; sampler/scorer inconsistency.

## P02-F05 (minor) — variance-floor repair disclosure is lost inside mixtures and HMMs

**Surface:** `fit_provenance().repairs` / `numerical_repairs()` on `MixtureDistribution`, `HiddenMarkovModelDistribution` with Gaussian / DiagonalGaussian components.

The 0.8.0 changelog: the Gaussian variance floor "is now reported the same way" (through `numerical_repairs()`).

**Reproduction:**
```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
g=optimize(np.full(100,3.0), S.GaussianEstimator(), max_its=3, print_iter=1000)
print(g, g.fit_provenance().repairs)     # GaussianDistribution(3.0, 9e-08)  ('variance-floored(0 -> 9e-08)',)
m=optimize(np.full(100,3.0), S.MixtureEstimator([S.GaussianEstimator()]*2), max_its=10, rng=np.random.RandomState(1), print_iter=1000)
print([str(c) for c in m.components], m.fit_provenance().repairs, m.numerical_repairs())   # both comps 9e-08, () ()
h=optimize([[1.0,1.0,1.0]]*30, S.HiddenMarkovModelEstimator([S.GaussianEstimator()]*2), max_its=10, rng=np.random.RandomState(1), print_iter=1000)
print([str(t) for t in h.topics], h.fit_provenance().repairs)       # both 1e-08, ()
d=optimize(np.tile([1.0,2.0],(50,1)), S.MixtureEstimator([S.DiagonalGaussianEstimator()]*2), max_its=5, rng=np.random.RandomState(1), print_iter=1000)
print(d.fit_provenance().repairs)        # ()  (plain DiagonalGaussianEstimator reports 'variance-floored(1e-08; onto a non-positive variance)')
```
**Expected:** the same floor applied inside a latent model is disclosed in the latent model's provenance.
**Observed:** `repairs=()` although every component was floored.

## P02-F06 (minor) — rejected-step early exit of `optimize()` is silent under the default `delta`, and provenance carries no stop reason

**Surface:** `mixle.inference.optimize` with `QuantizedHiddenMarkovModelEstimator` (any estimator whose M-step can propose a non-improving step).

**Reproduction:**
```python
import numpy as np, warnings, mixle.stats as S
from mixle.inference import optimize
warnings.simplefilter("always")
true=S.HiddenMarkovModelDistribution(topics=[S.CategoricalDistribution({'a':0.8,'b':0.15,'c':0.05}),S.CategoricalDistribution({'a':0.1,'b':0.2,'c':0.7})],w=[0.5,0.5],transitions=[[0.9,0.1],[0.2,0.8]],len_dist=S.CategoricalDistribution({40:1.0}))
d=true.sampler(seed=2).sample(150)
E=lambda: S.QuantizedHiddenMarkovModelEstimator(num_states=2, levels=['a','b','c'], k_max=10)
m=optimize(d, E(), max_its=60, rng=np.random.RandomState(1), print_iter=1000)
print(m.fit_provenance())   # iterations=4, max_iterations=60, converged=False, objective_gain=4.79e-09 -- NO warning emitted
m=optimize(d, E(), max_its=60, delta=None, rng=np.random.RandomState(1), print_iter=1000)   # same 4 iterations, but now a UserWarning explains "a proposed update was rejected"
```
**Expected:** the same explanatory warning on both paths (the changelog: "a silently truncated EM run now warns"), or a `stop_reason` in `FitProvenance`.
**Observed:** silent on the default path; the returned model is fine here (`monotone=False` converges to the identical objective at 6 iterations), but the caller sees `converged=False` with 56 unused iterations and no explanation.

## P02-F07 (minor) — `learn_bayesian_network` input-validation gaps

**Surface:** `mixle.inference.learn_bayesian_network`, `HeterogeneousBayesianNetwork.log_density`.

**Reproduction:**
```python
import numpy as np, mixle.stats as S
from mixle.inference import learn_bayesian_network
learn_bayesian_network([])                                    # IndexError: list index out of range
net=learn_bayesian_network([(1,2.0),(1,2.0,3.0)]*10); print(net)  # HeterogeneousBayesianNetwork(fields=2, ...) -- ragged rows silently truncated
r=np.random.RandomState(2); n=1500; a=r.randint(0,3,n); b=a*2.0+r.normal(0,0.5,n); c=np.where(r.rand(n)<0.9,a,r.randint(0,3,n)); dd=b+c+r.normal(0,0.3,n)
net=learn_bayesian_network([(int(a[i]),float(b[i]),int(c[i]),float(dd[i])) for i in range(n)])
print(net.log_density((0,np.nan,0,0.0)))        # nan  (propagated, not rejected / -inf)
print(net.log_density((0,0.0,0,0.0,1.0)))       # -1.223  (5-tuple scored against a 4-field net: extra field ignored)
net.log_density((0,0.0,0))                      # IndexError: tuple index out of range
```
**Expected:** a `ValueError` naming the problem for empty input, ragged rows, wrong width; NaN handled like the rest of the package (reject or `-inf`).
**Observed:** as above. (Weights are validated well: all-zero / negative weights are refused with a clear message.)

## P02-F08 (minor) — `HiddenMarkovModelDistribution` accepts `components=` but exposes no `.components`

**Surface:** `HiddenMarkovModelDistribution.__init__` (documented alias `components` for `topics`).

**Reproduction:**
```python
import mixle.stats as S
h=S.HiddenMarkovModelDistribution(components=[S.GaussianDistribution(0,1)], w=[1.0], transitions=[[1.0]])   # accepted
h.components   # AttributeError: 'HiddenMarkovModelDistribution' object has no attribute 'components'
```
**Expected:** the alias round-trips (`MixtureDistribution` exposes `.components`; the 0.8.0 changelog lists "HMM `components=`" under API consistency).
**Observed:** only `.topics` exists.

## P02-F09 (minor) — `LDAEstimator` turns ordinary corpora into an exception from `optimize()`

**Surface:** `mixle.stats.LDAEstimator` default alpha solver (`alpha_threshold=1e-8`, `max_alpha_iter=1000`).

**Reproduction:**
```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
from mixle.utils.optsutil import count_by_value
rng=np.random.RandomState(0); V=20
E=lambda k=3, **kw: S.LDAEstimator([S.IntegerCategoricalEstimator(min_val=0,max_val=V-1)]*k, **kw)
docs=[list(count_by_value(list(rng.randint(0,V,5000))).items()) for _ in range(50)]
optimize(docs, E(), max_its=5, rng=np.random.RandomState(1), print_iter=1000)
# LDAConvergenceError: lda_alpha_fixed_point did not converge after 1000/1000 iterations (iteration_budget_exhausted; residual=3.51405e-05)
one=[list(count_by_value(list(rng.randint(0,V,60))).items())]
optimize(one, E(), max_its=5, rng=np.random.RandomState(1), print_iter=1000)        # LDAConvergenceError (alpha_diverging)
optimize(one*50, E(), max_its=5, rng=np.random.RandomState(1), print_iter=1000)     # LDAConvergenceError (iteration_budget_exhausted; residual=1.15e-06)
```
**Expected:** a fitted model (alpha at the last iterate, or at the ceiling) plus a warning / provenance note, as every other estimator does for slow or non-convergence.
**Observed:** `optimize()` raises; the message does name `fixed_alpha=` and `alpha_threshold` as escapes (per the 0.8.0 changelog), and a 200-document mixed-topic corpus fits fine, so this is a usability gap rather than a wrong answer.

## P02-F10 (docs) — tutorial notebooks declare a `JointMixtureDistribution` law that mixle silently replaces

**Surface:** `tutorials/latent_variable_models.ipynb` cell 18; `data_science/latent_models_in_practice.ipynb` cell 55.

Both construct `JointMixtureDistribution([...],[...], w1, w2=[0.7,0.2,0.1], taus12, taus21=identity)` and mixle emits
`DeprecationWarning: w2 and taus21 describe a different joint law; the canonical law derived from w1 and taus12 is used. Pass joint_weights to avoid …`
(visible in `executed-6.ipynb` cell 18 stderr and `executed-10.ipynb` cell 55). The prose says the two blocks are "linked … with transition matrices taus12 / taus21", but the sample the notebook then fits is not drawn from the declared `w2`/`taus21`.

**Expected:** the tutorial passes a consistent joint law (or `joint_weights`) so that its declared parameters are the ones sampled.
**Observed:** deprecated, silently-overridden arguments in a tutorial. Not a candidate defect — the library's behaviour is correct and warned — but a notebook whose stated model is not the sampled one.

---

## Attacks that did not break anything

- **Gaussian mixture, degenerate data:** one cluster → 3 components (finite LL, posterior rows sum to 1); all-identical points (variance floored at 9e-08, finite); 2 distinct points → K=5 (three components collapse to weight 4.9e-16; `posterior(1.5)` = [0,0,⅓,⅓,⅓] is *correct* — verified from exact weights and component densities); single point; integer observations (int and float posteriors identical); NaN / inf in data refused with a clear message; empty data refused.
- **40 Gaussian-mixture seeds × 3 components, 25 DiagonalGaussian seeds × 3, 30 HMM seeds × 3 states, 20 LDA seeds, 15 TreeHMM seeds:** every fit finite, every posterior row sums to 1 within 1e-8, every transition row sums to 1.
- **`seq_log_density` vs per-item `log_density`, `seq_posterior` vs `posterior`:** agree to ≤1e-13 for Mixture, DiagonalGaussian mixture, HeterogeneousMixture, HMM, Lookback, Segmental, Tree, Quantized, LDA, MarkovChain, IntegerMarkovChain (lag 1 and 2), ChowLiu, IntegerChowLiu, BayesianNetwork.
- **HMM correctness vs brute-force enumeration (2 states, length 9):** marginal likelihood exact (diff 0.0), Viterbi path equals the argmax over all 512 paths, smoothing marginals match to 6.7e-16, `latent_posterior().marginals()` to 1.4e-15, `filtered=True` differs from smoothing as it should.
- **`use_numba=True` vs `False` HMM fits** (numba absent): identical weights, LL to 1e-13, same iteration count.
- **Sampler → fit → recover:** 2-state Gaussian HMM (mu −2.999/2.984, sig² 1.006/1.004, A [[.903,.097],[.199,.801]] from truth [[.9,.1],[.2,.8]]); LDA (topic correlations 0.999/0.999/0.975, alpha within 0.3); TreeHMM (mu ±2.9, A close); SegmentalHMM (segment means ±2.97/−2.99); Quantized HMM total mass over all length-2 strings = 1.0000; Lookback lag-1 total mass over all length-1 and length-3 strings = 1.0000. Caveat (not a defect): Lookback lag-1 EM lands on the symmetric near-uniform saddle for 3 of 6 seeds (LL gap ≈ 900 nats vs the truth; the other 3 seeds beat the true model's LL) and honestly reports `converged=False` at 300 iterations; `best_of` is the documented remedy.
- **JSON round trips** (`to_json`/`from_json`, log-densities compared): Mixture, DiagonalGaussian mixture, HeterogeneousMixture, HMM (Viterbi equal too), Segmental, Tree, Quantized, Lookback lag-1, LDA (posteriors equal), MarkovChain, IntegerMarkovChain, IntegerChowLiu — all diff 0.0, `fit_provenance` survives. `HeterogeneousBayesianNetwork` has no `to_json`/`from_json` *methods*, but `mixle.utils.serialization.to_json/from_json` round-trips it (lldiff 0.0), consistent with the changelog's "JSON serialization registration".
- **`fit_provenance()` claims:** `iterations`/`max_iterations`/`converged`/`final_objective` matched the actual iteration log and the recomputed log-likelihood in every fit checked (Mixture, HMM, LDA, MC, IntegerChowLiu, Quantized); the max_its-cap warning fires whenever `converged=False` at the cap.
- **Empty / length-1 / mixed-length / all-identical sequences:** HMM handles an all-empty corpus, an empty sequence inside a corpus (`log_density([])=0.0`, `viterbi([])=[]`), length-1 corpora, constant sequences of varying length; MarkovChain handles empty sequences in a corpus (`0.0`), a length-1-only corpus (uniform rows), an absorbing row (filled uniformly, as the changelog says), mixed int/str labels, and refuses an all-empty corpus with a clear error; LDA handles an empty document (`log_density([])=0.0`, posterior = normalized alpha) and empty trailing documents; Lookback lag-1 accepts sequences of exactly `lag` values in training and refuses shorter ones with a clear message; TreeHMM rejects empty trees, missing parents, cycles and multiple roots with precise messages, and scores a node order that lists a child before its parent identically.
- **Label sets that change between fit and score:** MarkovChain, categorical HMM, categorical mixture, LDA (out-of-vocabulary word, negative word id) all score `-inf`; `pseudo_count`+`levels` on MarkovChain gives finite mass to declared-but-unseen levels only; unseen labels at LDA fit time are refused with a support message. Mixture `posterior` on impossible evidence returns an all-zero row (documented) whereas HMM `seq_posterior` raises `ImpossibleEvidenceError` — inconsistent but both explicit.
- **Extreme values:** Gaussian mixture `log_density(1e154)` finite, `1e200` → `-inf` with posterior `[0,0]` (documented "all zero when impossible"); `inf`/`nan` observations raise `UnscorableObservation`.
- **Mixtures of non-generative factors:** `MixtureEstimator([MarkovChainEstimator()]*2)` (no `len_estimator`) is refused as "likelihood factors, not laws" — a deliberate policy stated in the changelog; with `len_estimator=CategoricalEstimator()` the mixture fits and posteriors are proper. A mixture of length-less HMMs is, inconsistently, accepted (noted, not filed).
- **Hangs/memory:** nothing hung. HMM 2000 seqs × 500 steps × 4 states and LDA 50 × 5000-token documents ran in seconds (LDA then raised F09, no hang). A 300-second stall seen once was my own stdin-fed script being detached by the harness, reproduced in 1 s when run unbuffered (`hang_probe.py`).
- **Quantized HMM:** `k_max`, `fixed_theta`, inferred `levels`, all-identical sequences (θ=0.25, `log_density([0,1,0])=-inf`), JSON round trip — all fine; without `k_max` θ drifts toward 1 as documented.
- **LDA sampler format:** `LDASampler.sample()` returns flat token lists that `LDAEstimator` refuses ("pair must have exactly two entries"); this is documented in the sampler docstring with a pointer to `count_by_value`, so not filed.
- **Corpus prose vs candidate:** apart from F10 I found no notebook or example statement that the candidate contradicts.

## Summary
- Counts: blocking 0, real 3 (F01, F02, F03), minor 6 (F04–F09), docs 1 (F10).
- Worst: **P02-F01** — on a base install (`pip install mixle`, no numba extra) the documented `HiddenMarkovModelDistribution.seq_posterior` returns `None` silently, and returns `None` unconditionally when `terminal_states` is set, although the marginals compute correctly when the gate is bypassed.
- Close second: **P02-F02** — a fitted `ChowLiuTreeDistribution` cannot be serialized at all.
- All 10 notebooks and 8 examples in this area execute cleanly on the candidate; no output contradicts the candidate except the deprecated joint-mixture construction in two tutorials.
- Nothing outside `<review-root>/reviews-081/pass-02/` was modified.
