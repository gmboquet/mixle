# Pass 02 -- latent-structure models: mixtures, HMMs and variants, Markov chains, LDA/PLSI, Chow-Liu and Bayesian networks, initialization and identifiability

*Recovery report.* The pass's execution, probes and notes were produced by a first reviewer (its command trail is `RECOVERED_NOTES.md`); this report and `findings.json` were written from that preserved evidence, plus the minimal confirming runs named below (`probes/p24_*`, `p25_*`, `p26_*`, `p27*_*`, `p28_*`, `p29_*`, `p30*_*` and the five remaining notebook comparisons), which were run through `tools/pyt.py` from `/tmp`.

## Header

- **Wheel:** `mixle-0.8.2-py3-none-any.whl`, sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da` (sdist `mixle-0.8.2.tar.gz` sha256 `71ec826b2960df13f9ca4513da83eac371e25ad942a479378e47d93f1a39c037`).
- **Commit:** `866078be520b22188110780be957150dc6da964c`, tree `7922a8c59283ecef6877104b9a7499afd52d9013` (branch `release/0.8.2`).
- **Work dir:** `REVIEW_ROOT/pass-02/` (probe scripts in `probes/`, their stdout in `out/`, executed notebooks in `corpus/notebooks/...` beside their `.log`, example stdout in `examples_out/`, regression-test copies in `tests_copy/`).
- **Verification** (`cd /tmp && <interpreter> REVIEW_ROOT/tools/verify_env.py`), verbatim:

`venv-full (notebooks, examples, most probes)`:

```
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-nonumba (p19 fingerprints)`:

```
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-base (p01, p19, p20, p27c, regression tests)`:

```
executable       <review-root>/venv-base/bin/python
mixle.__path__   <review-root>/venv-base/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`published 0.8.1 (before/after comparison only)`:

```
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

## Corpus executed on the candidate

Notebooks: executed in place in the APFS clone `corpus/`, on `venv-full`, through `tools/run_nb.sh` (timeout 1800 s), three at a time under machine contention (wall times are not performance evidence). All 16 exited 0.

| notebook | exit | wall (s) |
|---|---|---|
| `data_science/change_point_segmentation` | 0 | 26 |
| `data_science/gaussian_mixtures_in_depth` | 0 | 147 |
| `data_science/character_models_chow_liu` | 0 | 184 |
| `data_science/hmm_from_scratch` | 0 | 137 |
| `data_science/language_detection_hmm` | 0 | 198 |
| `data_science/enumerating_a_language_model` | 0 | 1619 |
| `data_science/latent_models_in_practice` | 0 | 1769 |
| `data_science/market_basket_ibp` | 0 | 473 |
| `data_science/markov_chains_for_sequences` | 0 | 30 |
| `data_science/model_based_clustering_evaluation` | 0 | 34 |
| `data_science/naive_bayes_text_classification` | 0 | 26 |
| `data_science/topic_modeling_lda` | 0 | 310 |
| `data_science/topic_modeling_plsi` | 0 | 16 |
| `data_science/projecting_an_llm_onto_a_lookback_hmm` | 0 | 348 |
| `data_science/learning_stop_words` | 0 | 2101 |
| `tutorials/latent_variable_models` | 0 | 58 |

Examples: `tools/pyt.py 1800 venv-full/bin/python source/examples/<name>.py` with `PYT_CWD=examples_out/<name>.cwd`. All 9 exited 0.

| script | exit | wall (s) |
|---|---|---|
| `structured_hmm_example` | 0 | 1224.7 |
| `lookback_hmm_example` | 0 | 353.6 |
| `latent_variable_models_example` | 0 | 185.8 |
| `hierarchical_mixture_example` | 0 | 258.1 |
| `joint_mixture_example` | 0 | 35.1 |
| `semi_supervised_mixture_example` | 0 | 23.3 |
| `mixture_reduction_benchmark` | 0 | 45.9 |
| `hidden_association_example` | 0 | 54.4 |
| `structure_learning_example` | 0 | 28.1 |

Fresh vs stored outputs (`compare_nb.py`, cell by cell; `out/cmp_<name>.txt`): no fresh output contradicts a prose claim. Differences found, all accounted for:

- Every notebook that fits with `print_iter=` now shows `Iteration N: ...` progress lines that the stored outputs replaced with a `print_iter has no effect without out=` warning (CHANGELOG 0.8.2: an explicit `print_iter=` prints without `out=`), wall-time prints, matplotlib object addresses, and warnings attributed to `best_of()` where the stored copy said `optimize()`.
- `data_science/language_detection_hmm`: cell 2's fit stops at `max_its=100` with the objective still moving by 4.85 nats (`Iteration 100 ... 4.854381e+00`) in the fresh run; train/test accuracies (0.973 / 0.967) are identical to the stored ones. No cap note appears in either copy (see 'not covered').
- `data_science/learning_stop_words`: the topic listings of cell 10 and its 'words with more shared than topic-specific mass' count (stored 813, fresh 679) differ, while the headline `precision@50=0.980, recall@50=0.120, MALLET hits=49/407` line is identical; the markdown makes no numeric claim about the changed numbers. Cells 8 and 18 (`initialize(...)`) newly emit the A-02 note (Q02-F03). The stored outputs cite paths under `mixle/.worktrees/release-082/`, i.e. an earlier 0.8.2 candidate, so a start-dependent difference is expected after R05-F08 (integer category codes no longer reach the numeric symmetry-breaking path).
- `tutorials/latent_variable_models`: cell 19's Viterbi path is the stored one with states relabelled (`[1 0 0 ...]` vs `[0 1 1 ...]`); cell 14's Gaussian parameters agree to 1e-14 and a `compiled-em: ...` progress line (written to `out`) is new; cell 20's tree HMM converged inside its budget in the fresh run (the stored copy carried a cap warning).
- `enumerating_a_language_model`, `topic_modeling_lda`, `projecting_an_llm_onto_a_lookback_hmm`: timing lines only. `character_models_chow_liu`, `hmm_from_scratch`, `market_basket_ibp`, `markov_chains_for_sequences`, `naive_bayes_text_classification`, `topic_modeling_plsi`: identical. `change_point_segmentation`, `gaussian_mixtures_in_depth`, `model_based_clustering_evaluation`, `latent_models_in_practice`: progress/warning lines only (numbers identical, e.g. the `latent_models_in_practice` HMM parameters agree to 1e-16).
- Examples: each script's printed claims are consistent with its numbers (e.g. `mixture_reduction_benchmark`: warm-started EM beat the closed form at 4/5 sizes and is monotone in M; `latent_variable_models_example`: fitted alpha [2.58, 2.53] vs true [1, 1] with TV distances [0.149, 0.119]; `lookback_hmm_example`: held-out gap 0.091 nats; `structured_hmm_example`: terminal-state lengths 2..18). `structure_learning_example` emits ~40 unconverged-fit warnings from `learn_mixture_structure`'s internal fits (the P09-F09 shape, already ledgered).

Regression tests on the base install (`run_tests_base.sh`, copies under `tests_copy/`, `venv-base`): `adversarial_review_082_repairs_test.py` 69 passed / 8 skipped, `latent_model_contract_repairs_test.py` 22 passed, `latent_initialization_symmetry_test.py` 12 passed, `latent_identifiability_disclosure_test.py` 9 passed, `component_family_identifiability_test.py` 9 passed, `hmm_terminal_states_test.py` 21 passed, `latent_readout_correctness_test.py` 24 passed (`out/tests_base.txt`). Green -- see Q02-F01 for why the terminal-state test passes.

## Findings

Severity counts: blocking 2, real 7, minor 5, docs 0 (14).

### Q02-F01 - blocking - A terminal-state HMM's latent_posterior (marginals/mode/sample), viterbi, seq_viterbi and posterior_predictive enforce only half of the restriction: they forbid a terminal state before the end but never require the final state to be terminal, so they answer for a law the model does not score

**Surface:** mixle.stats.HiddenMarkovModelDistribution.latent_posterior / .viterbi / .seq_viterbi / .posterior_predictive with terminal_states set (QuantizedHiddenMarkovModelDistribution inherits them); hidden_markov.py:1778-1780 and 1815-1817

**Repairs concerned:** R05-F01, P02-F01

**Reproduction:**

```python
# probes/p01_terminal_readouts.py (outputs out/p01_terminal_readouts_full.txt, _base.txt; 0.8.1: probes/p01_terminal_readouts_081.py -> out/p01_terminal_readouts_081.txt)
# probes/p01b_terminal_seq_viterbi.py (fitted model, seq_viterbi) -> out/p01b_terminal_seq_viterbi_full.txt
# cd /tmp && REVIEW_ROOT/tools/pyt.py 600 REVIEW_ROOT/venv-full/bin/python REVIEW_ROOT/pass-02/probes/p01_terminal_readouts.py
import numpy as np, mixle.stats as S
m = S.HiddenMarkovModelDistribution([S.GaussianDistribution(-3.0, 1.0), S.GaussianDistribution(3.0, 1.0)],
                                    w=[0.5, 0.5], transitions=[[0.8, 0.2], [0.2, 0.8]], terminal_states=[1])
x = [-3.0, -3.0, -3.0, -3.0]
print(m.seq_posterior(m.dist_to_encoder().seq_encode([x]))[0][-1])   # [0. 1.]   the scored law: the last state IS terminal
print(m.latent_posterior(x).marginals()[-1])                          # [1. 0.]   off by 1.0
print(m.viterbi(x), m.latent_posterior(x).mode())                     # [0 0 0 0] -- a path log_density gives -inf; admissible mode is [0,0,0,1]
print(m.viterbi([-3.0]))                                              # [0]       only [1] is admissible; length-1 sequences get no restriction at all
rng = np.random.RandomState(0); print(sum(int(np.asarray(m.latent_posterior(x).sample(rng))[-1]) != 1 for _ in range(200)))  # 200 of 200 samples end in a non-terminal state
```

**Observed:** Hand-built models (out/p01_terminal_readouts_full.txt, identical on venv-base): log_density and seq_posterior agree with brute-force enumeration over the admissible paths (z_1..z_{L-1} non-terminal, z_L terminal) to 1e-15 in every case, but latent_posterior().marginals() differs from the same enumeration by 1.000 (K=2, x=[-3,-3,-3,-3]: exact last row [0,1], returned [1,0]), by 1.000 on the length-1 sequence [-3.0], by 0.067 (K=3 terminal=[2]: exact [0,0,1], returned [0.057,0.010,0.933]) and by 0.998 (K=3 terminal=[1,2]); viterbi/mode return [0,0,0,0], [0] and [0,0,0] where the admissible modes are [0,0,0,1], [1] and [0,0,2]; FFBS .sample() ends in a non-terminal state on 200/200, 200/200, 12/200 and 200/200 draws; posterior_predictive reads the same object. Fitted model (the CHANGELOG's own P02-F01 reproduction, 40 sequences, out/p01b_terminal_seq_viterbi_full.txt): seq_posterior vs latent_posterior().marginals() max|diff| 0.9528, last row disagrees on 40/40 sequences, viterbi ends in a non-terminal state on 40/40, FFBS on 38/40; seq_viterbi returns [0,0,0,0], [0,0,0,1], [0] for the three hand-built sequences. 0.8.1 gives the same wrong readouts (out/p01_terminal_readouts_081.txt; there seq_posterior was None). StructuredHMM with terminal_states (structured_hmm.py) matches the enumeration exactly on the same inputs (out/p18_terminal_sampler_structured_full.txt), so the family disagrees with its own sibling.

**Expected:** Every readout of a terminal-state HMM is a functional of the posterior over the paths the model scores: log_density (hidden_markov.py:1169-1185, 'z_1..z_{L-1} non-terminal and z_L terminal') and terminal_forward_backward (696-720, which restricts the final position with lb[L-1] = where(term_mask, 0, -inf)). The marginals' last row must be supported on terminal states, the Viterbi path and every FFBS draw must end in one, and a length-1 sequence must be assigned to a terminal state. The migration guide (docs/migrations/0.8.2.md:63-67) and the CHANGELOG R05-F01 bullet say these routes now honour terminal_states; they honour it at positions 1..L-1 only.

**Notes:** Mechanism: viterbi (hidden_markov.py:1778-1780) and latent_posterior (1815-1817) fold the restriction as `log_b[:-1, terminal_mask] = -inf` and nothing else, guarded by `nn > 1`; the final position is left unrestricted, so the ordinary recursion sums/maximizes over paths that end in a non-terminal state -- paths _terminal_states_log_density and terminal_forward_backward exclude. Adding `log_b[-1, ~terminal_mask] = -inf` (and handling nn == 1) would make the folded recursion the restricted one. The regression test TerminalStateRoutesTest (mixle/tests/adversarial_review_082_repairs_test.py) enumerates 'admissible' paths with `if terminal and any(state in TERMINAL for state in path[:-1]): continue` -- the same half-law the code implements, without requiring path[-1] in TERMINAL -- so it passes for a reason other than correctness; hmm_terminal_states_test.py covers the forward likelihood and the sampler only. Concerns R05-F01 (the repair claimed for these routes) and P02-F01.

### Q02-F02 - blocking - P02-F03 does not hold for the count families: Poisson, Geometric and LogSeries encoders still refuse fractional, infinite and NaN values that the scalar path scores -inf (zero-weight rows included), and a Gaussian-plus-Poisson/Geometric mixture or HMM cannot be encoded or fitted on data with an out-of-support row

**Surface:** PoissonDistribution / GeometricDistribution / LogSeriesDistribution .dist_to_encoder().seq_encode and .seq_log_density; MixtureEstimator, HeterogeneousMixtureEstimator, HiddenMarkovModelEstimator with a Gaussian sibling; PoissonEstimator/GeometricEstimator/LogSeriesEstimator zero-weight exemption

**Repairs concerned:** P02-F03, R05-F02, R05-F05

**Reproduction:**

```python
# probes/p03_support_limited.py -> out/p03_support_limited_full.txt ; probes/p03b_count_families.py -> out/p03b_count_families_full.txt
import numpy as np, mixle.stats as S
from mixle.inference import optimize
d = S.PoissonDistribution(2.0)
print(d.log_density(1.5), d.log_density(np.inf))                    # -inf -inf
d.seq_log_density(d.dist_to_encoder().seq_encode([1.5]))            # ValueError: Poisson observations must be finite exact integers.  (Geometric, LogSeries: same)
acc = S.PoissonEstimator().accumulator_factory().make()
acc.seq_update(d.dist_to_encoder().seq_encode([1, 2, 3, 1.5]), np.array([1., 1., 1., 0.]), None)   # ValueError -- the zero-weight row is not exempt
rng = np.random.RandomState(0)
x = np.concatenate([rng.normal(-5, 1, 200).round(), rng.poisson(6, 200)]).astype(float)   # integer data, negative rows encode as -inf
optimize(x, S.MixtureEstimator([S.GaussianEstimator(), S.PoissonEstimator()]), max_its=15, rng=np.random.RandomState(1))
# ValueError: PoissonDistribution has support x in {0, 1, 2, ...}, but at least 27 observation(s) carrying weight are negative, fractional, NaN, or infinite ...
optimize(np.concatenate([rng.normal(-3, 0.5, 200), rng.gamma(3, 1, 200)]), S.MixtureEstimator([S.GaussianEstimator(), S.GammaEstimator()]), max_its=15, rng=np.random.RandomState(1))   # fits (the continuous families do)
```

**Observed:** Scalar-vs-vectorized survey over 13 support-limited families at x in {-1, 0, 1.5, 2, inf} (out/p03_support_limited_full.txt): only Poisson, Geometric and LogSeries disagree -- scalar -inf, vectorized ValueError 'observations must be finite exact integers' at x=1.5 and x=inf (and NaN, out/p03b_count_families_full.txt); negative integers encode as -inf on both routes. Zero-weight rows: -1 is exempt (statistics equal the clean ones) but 1.5 and inf raise at encode for all three families. Mixture(Gaussian, Poisson) and Mixture(Gaussian, Geometric) over float data holding negative rows: TypeError 'MixtureDistribution could not encode the data with all of its component encoders'; HMM(Gaussian, Poisson/Geometric): ValueError at encode; over integer data with negative integers, estimation still refuses on Mixture(Gauss,Poisson), Mixture(Gauss,Geometric), HMM(Gauss,Poisson) and Mixture(Poisson,Geometric) with 'at least N observation(s) carrying weight are negative ...'. The same shapes with Gamma, LogGaussian, Beta, Weibull, HalfNormal, Rayleigh, InverseGamma, InverseGaussian and Uniform siblings fit.

**Expected:** CHANGELOG P02-F03: 'The encoders now admit out-of-support observations and the vectorized scorers return the -inf the scalar path returns; fitting a law on one is refused by the accumulator (with a zero-weight row exempt, as EM produces); and mixture, heterogeneous-mixture and hidden-Markov initialization consult each component's support before handing it any responsibility.' The migration guide adds 'a zero-weight row is still exempt' for PoissonEstimator/GeometricEstimator/LogSeriesEstimator. Both should hold for these three families as they do for the continuous ones: -inf on the batch route for any out-of-support value, zero-weight rows exempt whatever their value, and a Gaussian sibling able to take the rows the count law cannot.

**Notes:** Mechanism (by grep, not a full read): `supported_rows` -- the hook `_state_draw_within_support` (hidden_markov.py:3525) and `_without_unsupported_responsibilities` (mixture.py:1523) consult at initialization -- is implemented by the ten continuous families only (inverse_gamma, beta, inverse_gaussian, exponential, log_gaussian, gamma, weibull, half_normal, rayleigh; pdist.py:1467 returns None = 'admits every row'), so a count component receives initial responsibility for rows outside its support and its accumulator's `refuse_unsupported_observations` then refuses them; the fractional/inf/NaN refusal comes from the encoder's whole-number check (`is_whole_number`, R05-F02) raising instead of encoding the value as impossible. A ledger-verbatim replay of P02-F03 (Exponential) passes; the defect is on the count siblings of the same shape.

### Q02-F03 - real - The A-02 unidentified-component note is raised from intermediate EM iterates (and by initialize()), so it reports a starvation the returned model does not have: a two-component categorical mixture whose components end with 147 and 153 effective rows against 49 free parameters each still emits 'this mixture fit left 2 of 2 component(s) with less data than they have parameters'

**Surface:** MixtureEstimator.estimate -> mixle.stats.latent.mixture._disclosing_component_support; mixle.inference.optimize / initialize; MixtureDistribution.component_row_mass

**Repairs concerned:** A-02, R07-F07

**Reproduction:**

```python
# probes/p29_a02_vocab_note.py -> out/p29_a02_vocab_note_full.txt ; probes/p24_a02_restarts.py -> out/p24_a02_restarts_full.txt
import warnings, numpy as np, mixle.stats as S
from mixle.inference import optimize
rng = np.random.RandomState(0); V, n = 50, 300
x = np.concatenate([rng.randint(0, V // 2, n // 2), rng.randint(V // 2, V, n // 2)])
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    m = optimize(x, S.MixtureEstimator([S.IntegerCategoricalEstimator(min_val=0, max_val=V - 1)] * 2), max_its=20, rng=np.random.RandomState(1))
print(m.component_row_mass)                                            # (146.8, 153.2)  -- 3x the 49 free parameters of each component
print([str(w.message)[:120] for w in wl if 'less data' in str(w.message)])   # one note: 'this mixture fit left 2 of 2 component(s) with less data than they have parameters: component 0 (49 free parameter(s)), component 1 (49 free parameter(s)) ...'
```

**Observed:** V=50, n=300: returned component_row_mass=(146.8, 153.2), each component has 49 free parameters, the note fired once. V=1000, n=300: the note fired 4 times for one optimize() call; the outlier corpus of p06/p24 fires it 5 times per optimize() call (under warnings 'always'). learning_stop_words (executed copy, cells 8 and 18): `initialize(cnt_docs, est, RandomState(1), 0.1)` -- an initialization, not a fit -- emits 'this mixture fit left 2 of 2 component(s) with less data than they have parameters: component 0 (5319 free parameter(s)), component 1 (5319 ...)'.

**Expected:** The note should describe the returned model: evaluate the criterion once on the final counts (the CHANGELOG A-02 bullet: 'A fitted mixture records how much data each component actually won ... and says so when a component holds fewer effective rows than it has free parameters'), and initialize() should not raise a fit-quality note.

**Notes:** Mechanism (mixture.py:2071-2115 and the comment above the message): `estimate` runs once per EM iteration and `_disclosing_component_support` evaluates counts < free_parameters on every call; the default init_p=0.1 hard subsample gives the first iterate roughly n/10 rows per component, which is below the parameter count of any moderately wide categorical component, so the note fires on iteration 1 whatever the final fit looks like. The message deliberately omits the row mass so that Python's once-per-location filter deduplicates it -- which also means the first (false) note is the one a user sees and a later true one from the same line is suppressed. Concerns A-02 and R07-F07 (the note's wording), not the record itself, which is correct on optimize()/fit()/best_of()/Model.fitted (out/p24_a02_restarts_full.txt).

### Q02-F04 - real - With the default init_p=0.1 the HMM start hands the initial-state law exact zeros for states that sequences do start in; EM can never revive a zero in w, and the fit reports converged=True at an optimum 44-373 nats below the one the same seed reaches with init_p=1.0 (3 of 6 seeds on 0.8.2, 1 of 6 on 0.8.1)

**Surface:** mixle.inference.optimize(..., init_p=0.1 default) with HiddenMarkovEstimator; HiddenMarkovAccumulator.seq_initialize / _initial_state_draw (hidden_markov.py:3345-3348, 3501); mixle.stats.latent._initialization.broken_symmetry_states

**Repairs concerned:** A-01, P09-F07

**Reproduction:**

```python
# probes/p15_a01_k3.py -> out/p15_a01_k3_{full,081}.txt ; probes/p27b_a01_start_model.py -> out/p27b_a01_start_model_{full,081}.txt ; probes/p27c_init_p.py -> out/p27c_init_p_{full,081,base}.txt
import numpy as np, mixle.stats as S
from mixle.inference import optimize
truth = S.HiddenMarkovModelDistribution([S.GaussianDistribution(-4, 1), S.GaussianDistribution(0, 1), S.GaussianDistribution(4, 1)], w=[0.1, 0.3, 0.6],
        transitions=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]], len_dist=S.CategoricalDistribution({25: 1.0}))
d3 = list(truth.sampler(1).sample(120))          # first observations fall 9 / 33 / 78 into the three regimes
E = lambda: S.HiddenMarkovEstimator([S.GaussianEstimator()] * 3, len_estimator=S.CategoricalEstimator())
ll = lambda m: float(np.sum(m.seq_log_density(m.dist_to_encoder().seq_encode(d3))))
for seed in (1, 2, 4):
    m = optimize(d3, E(), max_its=1, rng=np.random.RandomState(seed)); print(seed, np.round(m.w, 4), np.round([t.mu for t in m.topics], 2))
    # 1 [0.3479 0.     0.6521] [ 0.01 -3.96  4.01]   2 [0. 0. 1.] [-3.99  0.05  3.89]   4 [0.6551 0.     0.3449] [ 4.   -3.98 -0.02]
    a = optimize(d3, E(), max_its=300, rng=np.random.RandomState(seed)); b = optimize(d3, E(), max_its=300, init_p=1.0, rng=np.random.RandomState(seed))
    print(seed, ll(a), a.fit_provenance().converged, ll(b))   # -5754.66 True -5710.13 / -6083.32 True -5710.13 / -5754.66 True -5710.13
```

**Observed:** 0.8.2 (venv-full and venv-base identical): after one EM step from the default start, w has exact zeros on seeds 1, 2 and 4 ([0.3479, 0.0, 0.6521], [0.0, 0.0, 1.0], [0.6551, 0.0, 0.3449]) while the emission means are already at -4/0/4; the full runs converge (converged=True, 14-15 iterations) to log-likelihoods -5754.66, -6083.32 and -5754.66 against -5710.13 from seeds 0, 3, 5 and from the same three seeds with init_p=1.0 (14 iterations each). initialize(d3, est, RandomState(seed), 0.1) shows no zeros on these seeds. 0.8.1 (uniform draw): one seed of six (seed 5) starts with a zero; its other failures are unconverged after 300 iterations rather than converged at a starved optimum.

**Expected:** A start should not set a model parameter to exactly zero when the data support it (9 of 120 sequences begin in the starved regime); either the initial-state counts should be smoothed or the decline guard that already protects emission weight ('when a split would leave a state carrying no weight', CHANGELOG A-01) should also cover the first-position counts. A fit that converged with an initial state the start starved should at least be disclosed rather than reported as converged=True.

**Notes:** Mechanism: seq_initialize builds init_counts from the state drawn at each sequence's first position weighted by the 0/1 subsample mask (hidden_markov.py:3345-3348), unsmoothed; at p=0.1 only about 12 of the 120 first positions carry weight. The k-means++ assignment now places each of them in its true cluster, so a cluster that holds 7.5% of first observations receives none of the ~12 with probability ~e^-0.9, and w_k = 0 exactly; broken_symmetry_states declines only when a cluster carries no weight over ALL observations (_initialization.py:196-200), not over sequence starts. Pre-existing on 0.8.1 at a lower rate because the uniform draw spreads the kept first positions over states. Concerns A-01 / P09-F07 (the start) -- the 2-state case the ledger names is fixed (6/6 seeds recover, 0.8.1: 2/6).

### Q02-F05 - real - The public initialize() verb does not take the A-01 symmetry-breaking start: on 0.8.2 it still draws HMM states uniformly, bit-identical to 0.8.1, while optimize() takes the k-means++ start; its docstring's promise that seq_initialize() 'should produce the same initialized model' is now false

**Surface:** mixle.inference.initialize (mixle/stats/compute/sequence.py:645) with HiddenMarkovEstimator; used by notebooks learning_stop_words and latent_models_in_practice (`from mixle.inference import initialize`)

**Repairs concerned:** A-01, P09-F07, P09-F11

**Reproduction:**

```python
# probes/p27c_init_p.py -> out/p27c_init_p_{full,base,081}.txt
import numpy as np, mixle.stats as S
from mixle.inference import optimize, initialize
truth = S.HiddenMarkovModelDistribution([S.GaussianDistribution(-4, 1), S.GaussianDistribution(0, 1), S.GaussianDistribution(4, 1)], w=[0.1, 0.3, 0.6],
        transitions=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]], len_dist=S.CategoricalDistribution({25: 1.0}))
d3 = list(truth.sampler(1).sample(120))
E = lambda: S.HiddenMarkovEstimator([S.GaussianEstimator()] * 3, len_estimator=S.CategoricalEstimator())
m0 = initialize(d3, E(), np.random.RandomState(0), 1.0); print(np.round([t.mu for t in m0.topics], 2), np.round(m0.transitions, 2))   # [1.27 1.41 1.4], rows ~[0.33 0.33 0.33]: the symmetric start (same numbers on 0.8.1)
m1 = optimize(d3, E(), max_its=1, init_p=1.0, rng=np.random.RandomState(0)); print(np.round([t.mu for t in m1.topics], 2))          # [4.01 -4.01 -0.]  (0.8.1: [1.25 1.42 1.4])
```

**Observed:** On 0.8.2 (venv-full and venv-base) initialize(d3, est, RandomState(seed), p) for p in {0.1, 1.0} and seeds 0-5 returns emission means within 0.5 of each other and uniform transition rows, numerically identical to 0.8.1's output for the same calls; optimize(max_its=1) on the same seeds returns means at -4/0/4 on 0.8.2 and the symmetric ones on 0.8.1. The docstring of initialize() says 'Seq_initialize() is much more efficient, and should produce the same initialized model for the same data sets.'

**Expected:** CHANGELOG A-01: 'The hidden-Markov families start from a symmetry-breaking state assignment ... A k-means++ start over the emission stream ... replaces it' -- through every documented initialization route, or the CHANGELOG and the initialize() docstring should say the start reaches seq_initialize/optimize only.

**Notes:** Mechanism: mixle/stats/compute/sequence.py:645 `initialize` iterates observations through `accumulator.initialize(x, w, rng)` (the per-observation path); the k-means++ start lives only in HiddenMarkovAccumulator.seq_initialize via _initial_state_draw (hidden_markov.py:3345, 3398, 3501). Not an encoding-layout effect: identical on the numba and base environments. Concerns A-01, P09-F07, P09-F11.

### Q02-F06 - real - The variance-floor disclosure repaired for mixtures and HMMs (P02-F05) is still lost through Composite, Sequence, Optional, SemiSupervisedMixture, Lookback-HMM and Mixture(Composite) containers and through the auto-structure Composite that optimize(rows) returns: the leaf variance is floored to 9e-08 and fit_provenance().repairs / numerical_repairs() report ()

**Surface:** fit_provenance().repairs and numerical_repairs() on CompositeDistribution, SequenceDistribution, OptionalDistribution, SemiSupervisedMixtureDistribution, LookbackHiddenMarkovModelDistribution, MixtureDistribution over CompositeEstimator components, and the CompositeDistribution returned by optimize(rows) auto-structure

**Repairs concerned:** P02-F05

**Reproduction:**

```python
# probes/p12_disclosure_containers.py -> out/p12_disclosure_containers_full.txt ; probes/p04_nested_disclosure.py -> out/p04_nested_disclosure_full.txt
import numpy as np, mixle.stats as S
from mixle.inference import optimize, learn_bayesian_network
ss = optimize([(3.0, None)] * 50 + [(3.0, [(0, 1.0)])], S.SemiSupervisedMixtureEstimator([S.GaussianEstimator()] * 2), max_its=3, rng=np.random.RandomState(1))
print([d.sigma2 for d in ss.components], ss.fit_provenance().repairs, ss.numerical_repairs())     # [9e-08, 9e-08] () ()
lb = optimize([[3.0] * 5] * 20, S.LookbackHiddenMarkovModelEstimator([S.SequenceEstimator(S.GaussianEstimator(), len_estimator=S.CategoricalEstimator())] * 2, lag=0), max_its=3, rng=np.random.RandomState(1))
print([t.dist.sigma2 for t in lb.topics], lb.numerical_repairs())                                    # [9e-08, 9e-08] ()
c = optimize([(3.0, 4.0)] * 50, S.CompositeEstimator([S.GaussianEstimator(), S.GaussianEstimator()]), max_its=3)
print([d.sigma2 for d in c.dists], c.numerical_repairs())                                             # [9e-08, 1.6e-07] ()
rows = [('a' if i % 2 else 'b', 3.0, float(i % 3)) for i in range(120)]
print(optimize(rows).numerical_repairs(), learn_bayesian_network(rows).numerical_repairs())          # ()  ('field[1].variance-floored(0 -> 9e-08)',)
```

**Observed:** Composite(Gauss,Gauss), Composite(Gauss,Cat), Sequence(Gauss), Optional(Gauss), SemiSupervisedMixture(Gauss x2), Lookback lag-0 over Sequence(Gauss), Mixture(Composite(Gauss,Gauss)) and the auto-structure CompositeDistribution all carry floored leaf variances (9e-08 / 1.6e-07) with repairs=() and numerical_repairs()=(); the plain Gaussian, Mixture, HeterogeneousMixture, Mixture-of-Mixtures, HMM (topics[i]...), HMM len_dist and HeterogeneousBayesianNetwork on the same constant data disclose the floor with a path.

**Expected:** The same disclosure on every container: 'A numerical repair applied to a component is reported by its container' (CHANGELOG P02-F05); the migration guide's 'Variance floors applied inside mixtures and HMMs are disclosed' should extend to the sibling containers, or numerical_repairs() should not answer () for a model whose leaf was floored.

**Notes:** Same defect class as P02-F05 on sibling containers; SemiSupervisedMixture and Lookback HMM are latent-model containers in this pass's area, the others were reached by the same probe. Not ledgered (P03-F01 concerned the auto-structure return lacking the methods; it now has them and they answer ()).

### Q02-F07 - real - At lag >= 1 the SequenceDistribution window law that the new P02-F04 refusal recommends is not a probability law over sequences -- total mass 0.29 (length 3), 0.17 (length 4), 0.09 (lag 2) -- yet the family fits and scores it silently, and its sampler raises AttributeError

**Surface:** LookbackHiddenMarkovModelDistribution / LookbackHiddenMarkovModelEstimator with lag >= 1 and SequenceDistribution/SequenceEstimator emission (window) laws; the P02-F04 refusal message; LookbackHiddenMarkovModelDistribution.sampler at lag >= 1

**Repairs concerned:** P02-F04, R05-F04

**Reproduction:**

```python
# probes/p16_lookback_normalization.py -> out/p16_lookback_normalization_full.txt ; probes/p11_lookback_lag0.py -> out/p11_lookback_lag0_{full,081}.txt
import itertools, numpy as np, mixle.stats as S
from mixle.stats.latent.lookback_hidden_markov_model import LookbackHiddenMarkovModelDistribution as LB
win = lambda p, L: S.SequenceDistribution(S.IntegerCategoricalDistribution(0, p), len_dist=S.CategoricalDistribution({L: 1.0}))
for lag, n in [(0, 3), (1, 3), (1, 4), (2, 4)]:
    m = LB([win([0.7, 0.3], lag + 1), win([0.2, 0.8], lag + 1)], w=[0.6, 0.4], transitions=[[0.8, 0.2], [0.3, 0.7]], lag=lag,
           init_dist=([win([0.5, 0.5], lag), win([0.3, 0.7], lag)] if lag else None), len_dist=S.CategoricalDistribution({(n - lag + 1 if lag else n): 1.0}))
    print(lag, n, sum(np.exp(m.log_density(list(s))) for s in itertools.product([0, 1], repeat=n)))   # 1.000000, 0.294640, 0.165769, 0.094653
w2 = lambda mu: S.SequenceDistribution(S.GaussianDistribution(mu, 1.0), len_dist=S.CategoricalDistribution({2: 1.0}))
i1 = lambda mu: S.SequenceDistribution(S.GaussianDistribution(mu, 1.0), len_dist=S.CategoricalDistribution({1: 1.0}))
ok1 = LB([w2(-1), w2(1)], w=np.array([0.5, 0.5]), transitions=np.array([[0.9, 0.1], [0.1, 0.9]]), lag=1, init_dist=[i1(-1), i1(1)], len_dist=S.CategoricalDistribution({5: 1.0}))
ok1.sampler(seed=1).sample(3)        # AttributeError: 'SequenceSampler' object has no attribute 'sample_given'  (0.8.1 identical)
```

**Observed:** The mass of the lag-0 model over all length-3 binary strings is 1.000000; at lag 1 it is 0.294640 (length 3) and 0.165769 (length 4), at lag 2 0.094653 -- log_density is not a log-probability, and the values shrink with length, so likelihoods across lags or lengths are not comparable. optimize() with SequenceEstimator window laws at lag=1 fits without complaint (out/p11_lookback_lag0_full.txt, 'lag1 fit with a too-short sequence' and the p22 lookback fit), scalar and batch scores agree, and the sampler of the same object raises AttributeError on 0.8.2 and 0.8.1. The P02-F04 refusal on a scalar family at lag 1 says: 'Wrap a scalar family in a SequenceEstimator/SequenceDistribution (a length-2 sequence law)'.

**Expected:** The module docstring defines the window law as the conditional P(X(t) | X(t-lag:t-1), Z(t)) (e.g. IntegerMarkovChainDistribution), under which the model is normalized. Either the lag >= 1 refusal should recommend a conditional window law and the family should refuse a joint law over overlapping windows, or the docstring/message should state that with a joint window law log_density is an unnormalized score; and sampler() should raise a named error (or work) rather than AttributeError.

**Notes:** The lag-0 advice is right (a length-1 SequenceDistribution is exactly a scalar law) and the lag-0 sampler->score round trip is repaired; the same sentence at lag >= 1 sends the user to an improper model. Concerns P02-F04 (the message) and R05-F04 (the seq_posterior contract restated for this family).

### Q02-F08 - real - learn_bayesian_network does not apply the front-door normalization optimize()/fit() apply: a mapping of columns is fitted as two records of column NAMES (fields=1, n_observations=2), a bare str as its 17 characters, and bytes / numpy.matrix / 0-d array / list of scalars / pandas Series die on raw TypeErrors that the fit verbs refuse by name

**Surface:** mixle.inference.learn_bayesian_network (mixle/inference/bayesian_network.py); the data spellings R02-F07 and R06-F03 normalize for optimize/fit/Model.fit

**Repairs concerned:** R02-F07, P02-F07, R06-F03

**Reproduction:**

```python
# probes/p14_bn_frontdoor.py -> out/p14_bn_frontdoor_full.txt ; probes/p08_bayesnet.py -> out/p08_bayesnet_full.txt
import numpy as np
from mixle.inference import learn_bayesian_network, optimize
r = np.random.RandomState(2); a = r.randint(0, 3, 300); b = a * 2.0 + r.normal(0, 0.5, 300)
net = learn_bayesian_network({'a': a.tolist(), 'b': b.tolist()}); print(net, net.fit_provenance().n_observations)   # HeterogeneousBayesianNetwork(fields=1, edges=[none]) 2
print(optimize({'a': a.tolist(), 'b': b.tolist()}))                                                              # HeterogeneousBayesianNetwork(fields=2, edges=[0->1])
print(learn_bayesian_network('hello world hello').fit_provenance().n_observations)                               # 17
optimize('hello world hello')                                                                                    # ValueError: optimize() received a str, which iterates as its individual characters ...
learn_bayesian_network(b'hello')            # TypeError: object of type 'int' has no len()
learn_bayesian_network([1.0, 2.0, 3.0])     # TypeError: object of type 'float' has no len()
```

**Observed:** dict of two 300-element columns -> a one-field network fitted on the two key strings with a receipt claiming n_observations=2; str -> one-field categorical over 17 characters; bytes -> TypeError 'int' has no len(); np.matrix -> TypeError unhashable type: 'matrix'; 0-d array -> TypeError iteration over a 0-d array; [1.0, 2.0, 3.0] and pd.Series -> TypeError object of type 'float' has no len(). optimize()/fit() on the same dict build a two-field network with edge 0->1 and refuse the str by name. DataFrame, filtered DataFrame, structured array, masked array, generator and iterator inputs are handled (out/p14_bn_frontdoor_full.txt).

**Expected:** CHANGELOG R02-F07: learn_bayesian_network 'now takes the same conversion optimize already applied'; R06-F03: 'optimize/fit/best_of normalize a one-shot iterator, a structured array and a mapping of columns, and refuse a masked array, a numpy.matrix, a 0-dimensional array and a bare str/bytes by name ... One table now gets one answer whichever verb reads it'. learn_bayesian_network should give the same answer or the same named refusal.

**Notes:** The R02-F07 repair covers the DataFrame spelling only; the mapping-of-columns case is the identical defect (iterating the container yields its keys) on the spelling R06-F01/R06-F03 normalize for the fit verbs. Ledger P02-F07 (empty/ragged/NaN/width) is repaired and holds.

### Q02-F09 - real - TreeHiddenMarkovModelDistribution is write-only: to_json(), utils.to_json and dump_models refuse a hand-built or fitted tree HMM ('serialized state ... does not match its constructor-owned schema'), so the family has no JSON persistence path -- the class of defect P02-F02 closed for Chow-Liu and P05-F17 closed for eight other families

**Surface:** mixle.stats.TreeHiddenMarkovModelDistribution.to_json / mixle.utils.serialization.to_json / mixle.stats.dump_models on a TreeHiddenMarkovModelDistribution (hand-built or fitted with TreeHiddenMarkovEstimator)

**Repairs concerned:** P05-F17, P02-F02

**Reproduction:**

```python
# probes/p25_treehmm_json.py -> out/p25_treehmm_json_full.txt ; probes/p22_serialization_variants.py -> out/p22_serialization_variants_full.txt
import mixle.stats as S
tt = S.TreeHiddenMarkovModelDistribution([S.GaussianDistribution(-3, 1), S.GaussianDistribution(3, 1)], w=[0.5, 0.5], transitions=[[0.9, 0.1], [0.1, 0.9]], len_dist=S.CategoricalDistribution({3: 1.0}))
tt.to_json()
# SerializationError: to_json produced JSON that from_json cannot read back (SerializationError: serialized state for
# 'mixle.stats.latent.tree_hidden_markov_model.TreeHiddenMarkovModelDistribution' does not match its constructor-owned schema; define
# __pysp_setstate__ for validated custom state). Refusing to return a write-only serialization ... pickle the object instead.
```

**Observed:** Hand-built and fitted (TreeHiddenMarkovEstimator, 3 iterations) tree HMMs: to_json/from_json, utils.to_json and dump_models all raise SerializationError on 0.8.2; pickle round-trips. The 0.8.1 comparison run timed out (out/p25_treehmm_json_081.txt, exit TIMEOUT after 600 s inside the fitted-tree routes) and was not repeated.

**Expected:** The CHANGELOG 0.8.2 entry 'Eight more families round-trip through JSON' (P05-F17, P09-F05, P10-F05) and the 0.8.0 disclosure that only the directional and matrix modules cannot round-trip leave a tree HMM user expecting a JSON path; a fitted tree HMM (gallery_structured_example, P09-F07) should serialize like the other HMM variants (terminal-state, lookback, quantized, segmental, semi-supervised mixture and hierarchical mixture all round-trip, out/p22_serialization_variants_full.txt).

**Notes:** The refusal is loud (no silent write-only text escapes), which is why this is real rather than blocking. Not among the six families ledgered under P05-F17 nor the eight the CHANGELOG lists as repaired.

### Q02-F10 - minor - The A-02 note raised from optimize()/fit() advises 'restarts=', a keyword neither optimize() nor fit() nor best_of() accepts (only Model.fit does)

**Surface:** the A-02 unidentified-component note text (mixture.py _disclosing_component_support) as emitted from mixle.inference.optimize / fit; CHANGELOG R07-F07

**Repairs concerned:** A-02, R07-F07

**Reproduction:**

```python
# probes/p24_a02_restarts.py -> out/p24_a02_restarts_full.txt ; probes/p13_a02_routes.py -> out/p13_a02_routes_full.txt
import inspect, numpy as np, mixle.stats as S
from mixle.inference import optimize, fit, best_of
print(['restarts' in inspect.signature(f).parameters for f in (optimize, fit, best_of)])   # [False, False, False]
x = np.concatenate([np.random.RandomState(0).normal(0, 1, 400), [40.0]])
optimize(x, S.MixtureEstimator([S.GaussianEstimator()] * 2), max_its=50, rng=np.random.RandomState(3))
# UserWarning: this mixture fit left 1 of 2 component(s) with less data than they have parameters ... and restarts= or MixtureEstimator(..., init='dirichlet') distinguishes the two.
optimize(x, S.MixtureEstimator([S.GaussianEstimator()] * 2), restarts=3)   # TypeError: optimize() got an unexpected keyword argument 'restarts'
```

**Observed:** The note from optimize() reads '... an initialization that latched onto a small group, and restarts= or MixtureEstimator(..., init='dirichlet') distinguishes the two.'; passing restarts=3 to optimize() raises TypeError; only Model.fit(..., restarts=) exists (and there the record and the note work, out/p24_a02_restarts_full.txt).

**Expected:** R07-F07: 'When the library chose the mixture the note says so and points at what is reachable instead' -- the note should name Model(est).fit(data, restarts=) or omit restarts= on the verbs that lack it.

**Notes:** Cosmetic on its own; listed separately from Q02-F03 because it is a different defect (wrong remedy vs false trigger). The trail's earlier reading that the A-02 record is absent on the restart routes was an artifact of this TypeError and of reading the Model wrapper instead of Model.fitted.

### Q02-F11 - minor - HeterogeneousBayesianNetwork.seq_log_density does not check record width: a record wider than the network is silently scored on its first fields and a narrower one raises IndexError, where log_density refuses both by name

**Surface:** HeterogeneousBayesianNetwork.seq_log_density / its encoder's seq_encode (mixle/inference/bayesian_network.py) vs log_density

**Repairs concerned:** P02-F07, R02-F09

**Reproduction:**

```python
# probes/p14_bn_frontdoor.py -> out/p14_bn_frontdoor_full.txt ; probes/p08_bayesnet.py -> out/p08_bayesnet_full.txt
import numpy as np
from mixle.inference import learn_bayesian_network
r = np.random.RandomState(2); a = r.randint(0, 3, 300); b = a * 2.0 + r.normal(0, 0.5, 300)
net = learn_bayesian_network([(int(a[i]), float(b[i]), int(a[i] % 2)) for i in range(300)]); enc = net.dist_to_encoder()
print(net.seq_log_density(enc.seq_encode([(0, 0.0, 0, 9.0)])))   # [-1.33251289]  -- the 4-wide record is scored as if 3-wide
net.log_density((0, 0.0, 0, 9.0))                                 # ValueError: this network has 3 field(s) but the record has 4 ...
net.seq_log_density(enc.seq_encode([(0, 0.0)]))                   # IndexError: list index out of range
```

**Observed:** Wider record: batch route returns the 3-field score, scalar route refuses by name; narrower record: batch route IndexError, scalar route ValueError naming the widths. A batch mixing widths is refused by the ragged-record check (ValueError), so the silent case needs every record equally too wide.

**Expected:** Both routes should validate the width the same way (P02-F07 claims it for log_density; R02-F09 established route parity for NaN fields).

**Notes:** Validation gap on the batch route; scalar-vs-batch agreement on ordinary rows is exact (0.0 over 50 rows).

### Q02-F12 - minor - A HiddenMarkovModelDistribution without len_dist reports density_semantics EXACT and is accepted as a component of Mixture, HeterogeneousMixture, SemiSupervisedMixture and HMM, while a SequenceDistribution or MarkovChainDistribution without a length model is refused as a likelihood factor

**Surface:** MixtureDistribution / HeterogeneousMixtureDistribution / SemiSupervisedMixtureDistribution / HiddenMarkovModelDistribution component validation; HiddenMarkovModelDistribution.density_semantics(); migration guide 'Latent models' first bullet

**Repairs concerned:** none

**Reproduction:**

```python
# probes/p21_factor_refusal.py -> out/p21_factor_refusal_full.txt ; probes/p23_nested_hmm.py -> out/p23_nested_hmm_{full,081}.txt
import mixle.stats as S
hm = S.HiddenMarkovModelDistribution([S.CategoricalDistribution({'a': 0.5, 'b': 0.5})] * 2, w=[0.5, 0.5], transitions=[[0.5, 0.5], [0.5, 0.5]])
sd = S.SequenceDistribution(S.CategoricalDistribution({'a': 0.5, 'b': 0.5}))
print(hm.density_semantics(), sd.density_semantics())      # DensitySemantics.EXACT  DensitySemantics.LIKELIHOOD_FACTOR
S.MixtureDistribution([hm, hm], [0.5, 0.5])                # accepted
S.MixtureDistribution([sd, sd], [0.5, 0.5])                # TypeError: MixtureDistribution components must be generative probability laws; likelihood factors found ...
```

**Observed:** Estimator and distribution level alike: Sequence-no-len and MarkovChain-no-len are refused by Mixture, HetMixture, HMM and SemiSupervisedMixture; HMM-no-len is accepted by all four (and optimize() fits a mixture over HiddenMarkovEstimator without len_estimator).

**Expected:** A length-less HMM scores P(x_1..x_n | n) exactly as a length-less SequenceDistribution does; the migration guide's 'MixtureDistribution and the HMM families refuse a component that is a likelihood factor rather than a generative law' should apply to it, or density_semantics() should say why it is exempt.

**Notes:** Consistency gap in the generative-component predicate; no wrong number was produced by it in these probes. The nested HMM-of-HMMs fitted from sampled data (with len_dist) works identically on both versions.

### Q02-F13 - minor - LDAEstimator fitted on a corpus of empty documents returns a model whose fit_diagnostics say 'converged' and fit_provenance().converged=True, right after warning that the alpha solve was diverging to infinity (alpha ~ [60, 293, 645])

**Surface:** mixle.inference.optimize with LDAEstimator on documents holding no words (P02-F09 / P10-F02 non-convergence disclosure)

**Repairs concerned:** P02-F09, P10-F02

**Reproduction:**

```python
# probes/p09_lda.py -> out/p09_lda_full.txt
import numpy as np, mixle.stats as S
from mixle.inference import optimize
m = optimize([[], []], S.LDAEstimator([S.IntegerCategoricalEstimator(min_val=0, max_val=19)] * 3), max_its=3, rng=np.random.RandomState(1))
print(np.round(m.alpha, 2), m.fit_diagnostics.termination_reason, m.fit_diagnostics.converged, m.fit_provenance().converged)
# [ 60.11 292.56 645.27] converged True True   -- after "LDA's Dirichlet alpha solve stopped at 1000/1000 iterations (alpha_diverging ...)"
```

**Observed:** Two empty documents: a UserWarning says the alpha solve stopped at 1000/1000 iterations, alpha diverging; the returned model's diagnostics say termination_reason='converged', converged=True after 1 iteration, and fit_provenance().converged=True. An empty corpus ([]) is refused by name; a corpus of empty documents is not.

**Expected:** Refuse a corpus with no words by name (as optimize does for an empty corpus), or make the diagnostics and the warning tell one story.

**Notes:** The P02-F09/P10-F02 repair holds on the three ledger corpora (model + warning + receipts, out/p09_lda_full.txt); this is a degenerate-input edge of it.

### Q02-F14 - minor - LogGaussianDistribution at +inf: log_density raises UnscorableObservation while seq_log_density returns -inf -- the scalar route reads the observation as unknown and the batch route as impossible

**Surface:** LogGaussianDistribution.log_density vs .seq_log_density at x = inf

**Repairs concerned:** P02-F03

**Reproduction:**

```python
# probes/p03_support_limited.py -> out/p03_support_limited_full.txt (survey block)
import numpy as np, mixle.stats as S
d = S.LogGaussianDistribution(0.0, 1.0)
print(d.seq_log_density(d.dist_to_encoder().seq_encode([np.inf])))   # [-inf]
d.log_density(np.inf)                                                # raises UnscorableObservation
```

**Observed:** The only continuous family among the 13 surveyed whose scalar and batch routes disagree at +inf; every other value (-1, 0, 1.5, 2) agrees on both routes for LogGaussian.

**Expected:** The same answer on both routes (R05-F06 / R02-F09 shape: an impossible observation must not read as unknown on one route only).

**Notes:** Outside this pass's family list (pass 01) but reproduced here while varying P02-F03; reported for completeness.

## Attacks that did not break anything

- P02-F01 (base-install `seq_posterior`): exact against enumeration on venv-full and venv-base without `terminal_states` and, with it, exact on the `seq_posterior` route itself (max diff 1.4e-15, `out/p01_terminal_readouts_*.txt`); HMM/mixture/LDA/segmental fit-and-readout fingerprints identical across venv-full, venv-nonumba and venv-base (`out/p19_env_consistency_*.txt`).
- P02-F02 (Chow-Liu serialization): string, integer, mixed, float, bool, None, numpy int64/str_, tuple-valued, bytes, 1- and 3-field, IntegerChowLiu, and one field mixing int and str levels -- to_json/from_json, to_dict, dump/load_models and pickle all exact (`out/p10_chowliu_full.txt`); an unseen level scores -inf, a wrong-width record is refused by name.
- P02-F03 on the ten continuous families: scalar and vectorized agree at -1, 0, 1.5, 2 and inf (LogGaussian +inf excepted, Q02-F14); Mixture and HMM with Gamma, LogGaussian, Beta, Weibull, HalfNormal, Rayleigh, InverseGamma, InverseGaussian and Uniform siblings fit on data with negative rows; the fitted Gaussian+Exponential mixture scores negative rows identically on both routes and after JSON; zero-weight negative-integer rows are exempt on Poisson/Geometric/LogSeries; all-impossible rows are refused with the R05-F09 diagnosis (`out/p03_support_limited_full.txt`, `out/p03b_count_families_full.txt`).
- P02-F04 lag-0 lookback: the four ledger shapes (scalar Gaussian/Categorical fits, scalar-law sampler and log_density, `(n,1)` input) are all named with the SequenceEstimator/HiddenMarkovModelEstimator advice on 0.8.2 (raw shape/hashability errors on 0.8.1); the lag-0 window-law sampler->score round trip is exact; lag-1 scalar-vs-batch agreement 0; an empty sequence at lag 1 is refused by name (`out/p11_lookback_lag0_{full,081}.txt`).
- P02-F05 in Mixture, HeterogeneousMixture, Mixture-of-Mixtures, HMM and HMM `len_dist`: the floor is disclosed with a component path, survives JSON and pickle, and a healthy fit reports () (`out/p04_nested_disclosure_full.txt`).
- P02-F07 ledger cases: empty corpus, ragged rows, NaN/inf field (-inf on both routes), wrong width on `log_density`, unseen category (-inf), list/ndarray rows, single and two-row corpora, DataFrame / filtered DataFrame / structured array / masked array / generator / iterator inputs, weights validation (`out/p08_bayesnet_full.txt`, `out/p14_bn_frontdoor_full.txt`).
- P02-F08: `components=` accepted by HiddenMarkovModelDistribution, MixtureDistribution and HeterogeneousMixtureDistribution; `.components is .topics`; JSON round trip keeps it (read-only property, as R05 noted) (`out/p05_components_alias_full.txt`).
- P02-F09 / P10-F02: the three ledger corpora and the P10-F02 seed-3 corpus return a model with the alpha warning, `fit_diagnostics` and `fit_provenance().converged=False`; the strict `update_alpha` still exists; `fixed_alpha=` and `alpha_threshold=` escapes work; the returned model scores, JSON-round-trips with its diagnostics and pickles; zero/negative counts and out-of-range ids are refused by name; k=1 and more topics than documents fit (`out/p09_lda_full.txt`).
- R05-F04 (lookback `seq_posterior`): smoothing marginals exact against enumeration at lag 1 (lengths 5 and 2) and lag 2 (2.2e-16), `filtered=True` exact against the forward recursion, `viterbi_sequence` equals the enumerated mode, `terminal_states` raises NotImplementedError as documented (`out/p02_lookback_posterior_full.txt`).
- A-01 on the ledger's own shape: the 2-state Gaussian HMM recovers on 6/6 seeds (0.8.1: 2/6), DiagonalGaussian on 3/3; Categorical and Gamma emissions are bit-identical to 0.8.1 (R05-F08 holds, including the Gamma plateau); mixture fingerprints identical; all-identical rows, fewer rows than states, two distinct values and one long sequence fit without error (`out/p07_a01_init_{full,081}.txt`).
- A-02 record: present on optimize(), fit(), best_of() and Model.fitted (with and without `restarts=`), sums to n and equals the posterior column sums, absent after JSON (documented) and present after pickle, silent on healthy fits; HeterogeneousMixture and HMM carry no record (not claimed) (`out/p06_a02_row_mass_full.txt`, `out/p13_a02_routes_full.txt`, `out/p24_a02_restarts_full.txt`).
- Terminal-state generative side: 300 `terminal_states` samples all finite under `log_density`, `seq_log_density` exact, `seq_posterior` last-row mass 1.0 on the terminal state, the length-1 fraction matches w[terminal]; `terminal_values` samples all end in the terminal value, total mass 0.693 over lengths <= 6 (geometric tail) and 0.9958 for `terminal_states` over lengths <= 11 (P(len > 11) = 0.0036); StructuredHMM terminal readouts exact (`out/p18_terminal_sampler_structured_full.txt`).
- Cross-version: 0.8.1-written JSON and pickles of a mixture, an HMM, a terminal-state HMM and a Markov chain load on 0.8.2 (venv-full and venv-base) with max|dlogp| = 0 and a working `seq_posterior` (`out/p20_xver_*.txt`; the 0.8.1 writer died on its own Chow-Liu `to_json`, the pre-repair P02-F02, so its later models were not written).
- Serialization of fitted variants on 0.8.2: terminal-state HMM (restriction and `seq_posterior` survive), terminal-values HMM, lookback lag-1 (IntegerMarkovChain), quantized, segmental, semi-supervised mixture and hierarchical mixture -- `to_json/from_json`, `utils.to_json` and pickle exact (`out/p22_serialization_variants_full.txt`; the `dump/load_models` failures in that file are the probe indexing a single returned model, disproved by p10's correct call).
- Likelihood-factor refusal: Mixture, HeterogeneousMixture, HMM and SemiSupervisedMixture refuse length-less Sequence and MarkovChain components at estimator and distribution level (`out/p21_factor_refusal_full.txt`).
- Nested HMM-of-HMMs fitted from its own samples: identical on 0.8.2 and 0.8.1 for 3 seeds x pseudo_count None/1.0 (`out/p23_nested_hmm_{full,081}.txt`).
- The max_its cap note: emitted for plain, keyed (`keys=(None,None,'states')`), `init_estimator=`, Sequence(HMM) and Conditional{Sequence(HMM)} fits, with and without progress printing (`out/p30_cap_note_full.txt`, `out/p30b_cap_note_keys_full.txt`).
- A-02 note deduplication: two identical fits from one loop line raise one note under the default filter (Python's once-per-location rule; noted, not a finding).

## What was not covered

- `language_detection_hmm` cell 2: an HMM fit stopping at `max_its=100` with the objective still moving 4.85 nats shows no cap note in the stored or the fresh output; the minimal reproductions of every ingredient the cell uses (keyed emissions, `init_estimator=`, Conditional over Sequence(HMM), progress printing) all emit it, so the omission was not reproduced outside the notebook and is unresolved.
- Tree HMM on 0.8.1: the JSON round-trip comparison (`probes/p25_treehmm_json.py`) timed out after 600 s on the 0.8.1 interpreter inside the fitted-tree routes and was not repeated; Q02-F09 is stated for 0.8.2 only.
- Tree HMM `seq_viterbi` on the numba vs numpy kernels: the label sums differ (1811953 vs 1812050 over ~3.5e6 nodes) with log-likelihoods equal to 1e-14 relative (`out/p19_env_consistency_*.txt`); whether these are exact ties was not determined.
- `SemiSupervisedHiddenMarkovModelEstimator`: a fit over `(value, None)` rows raised 'state_prior likelihood potentials must be finite and non-negative' (`out/p22_serialization_variants_full.txt`); the correct spelling was not investigated.
- LDA: the ledger corpora show a `fit_diagnostics.termination_reason` that differs from the reason named in the first warning of the same fit (`alpha_diverging` vs `iteration_budget_exhausted`, `out/p09_lda_full.txt`); most likely different EM iterations, not resolved.
- Lookback at lag >= 1: a training sequence shorter than lag+1 is accepted by the fit (`out/p11_lookback_lag0_full.txt`); what it contributes was not checked.
- The count-family initialization mechanism in Q02-F02 was traced by grep (`supported_rows` implementers) rather than by reading the mixture initializer; the encoder refusal by grep to `is_whole_number`.
- Weighted `optimize()` calls in p03 used a `weights=` keyword `optimize()` does not take; the estimator-level zero-weight exemption was covered by p03b instead.
- Not attempted: Spark/RDD routes, Tree/segmental HMM readouts against enumeration, heterogeneous (per-state) emissions with `terminal_states`, the `mixle.stats.lookback_hmm` sibling module, PLSI beyond its notebook, and the `exec/` background logs.

DONE 02 14
