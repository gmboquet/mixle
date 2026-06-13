# Distribution API Naming Accounting

Date: 2026-06-13

Scope: runtime Python modules under `pysp/`, excluding `pysp/tests` and cache files. This is a documentation-only
accounting; no package code was changed for this report.

## Inventory Summary

The AST inventory found:

| Item | Count |
| --- | ---: |
| Runtime Python files scanned | 158 |
| Classes scanned | 802 |
| Module-level functions | 835 |
| Class methods | 4,803 |
| Distribution classes | 107 |
| Estimator classes | 102 |
| Sampler classes | 103 |
| Accumulator classes | 100 |
| Accumulator factory classes | 90 |
| Data encoder classes | 69 |
| Enumerator classes | 38 |

The public distribution API is mostly regular, but not regular enough to be predictable from names alone. The biggest
sources of inconsistency are:

1. Family stems are not always shared by the distribution, estimator, sampler, accumulator, encoder, and enumerator.
2. Constructor argument names mix short mathematical notation with descriptive names.
3. The public `keys` argument is stored internally as both `self.key` and `self.keys`.
4. Integer-specialized models use the public word `Integer`, but many modules still use abbreviated `int*` names.
5. A few legacy camelCase names remain.

## Recommended Convention

Use this as the target convention for new APIs and for compatibility aliases during migration.

### Class Families

For every public model family, use one stem:

`<Stem>Distribution`, `<Stem>Estimator`, `<Stem>Sampler`, `<Stem>Accumulator`,
`<Stem>AccumulatorFactory`, `<Stem>DataEncoder`, `<Stem>Enumerator`.

Examples:

| Current | Preferred |
| --- | --- |
| `HiddenMarkovModelDistribution`, `HiddenMarkovEstimator` | `HiddenMarkovModelDistribution`, `HiddenMarkovModelEstimator` |
| `QuantizedHiddenMarkovModelDistribution`, `QuantizedHiddenMarkovEstimator` | `QuantizedHiddenMarkovModelDistribution`, `QuantizedHiddenMarkovModelEstimator` |
| `ConditionalDistribution`, `ConditionalDistributionEstimator` | `ConditionalDistribution`, `ConditionalEstimator` |
| `GrammarEstimatorAccumulator` | `GrammarAccumulator` |
| `LDAEstimatorAccumulator` | `LDAAccumulator` |

Keep old names as aliases for at least one compatibility cycle.

### Constructor Argument Order

For public distribution constructors:

1. Required statistical parameters.
2. Optional statistical parameters and support bounds.
3. Structure children such as `len_dist`, `init_dist`, `given_dist`, `default_dist`.
4. `name=None`.
5. `keys=None`.
6. Backend/implementation flags such as `use_numba=False`, `low_memory=False`.

For public estimator constructors:

1. Required structural parameters.
2. Child estimators.
3. Fixed known parameters such as `fixed_weights`, `fixed_theta`, `fixed_alpha`.
4. `pseudo_count=None`.
5. `suff_stat=None`.
6. `name=None`.
7. `keys=None`.
8. Backend/implementation flags.

### Preferred Argument Names

| Concept | Preferred public name | Current variants to alias/deprecate |
| --- | --- | --- |
| Probability map | `prob_map` | `pmap` |
| Probability vector | `prob_vec` | `p_vec` |
| Log probability vector | `log_prob_vec` | `log_pvec`, `log_p_vec` |
| Probability matrix | `prob_mat` | `p_mat`, `pmat` |
| Mixture weights | `weights` | `w`, `w1`, `w2`, `fixed_w` |
| Fixed mixture weights | `fixed_weights` | `fixed_w` |
| Number of values | `num_values` | `num_vals` |
| Minimum integer support | `min_val` | `min_index` |
| Maximum integer support | `max_val` | `max_index` |
| Covariance | `covariance` | `covar`, `cov`, `sig2` for covariance matrices |
| Variance | `sigma2` | keep for scalar Gaussian variance |
| Rate parameter | `rate` or model-specific `lam` | `beta` for exponential rate |
| Iteration limit | `max_iter` | `max_its` |
| Length child distribution | `len_dist` | mostly consistent |
| Length child estimator | `len_estimator` | mostly consistent |
| Shared sufficient-stat key(s) | `keys` | keep public plural; avoid public `key` |

Recommendation: for mature classes, accept old spellings as aliases, canonicalize to the preferred names in
`__init__`, and serialize/pretty-print the preferred names.

### Integer Naming

The word `Integer` is appropriate when the model is not just discrete, but specifically assumes dense integer-coded
support or exploits integer indexing. Keep it in class names for those optimized variants.

Preferred module names should spell this out:

| Current module | Preferred module alias |
| --- | --- |
| `intrange` | `integer_categorical` |
| `intmultinomial` | `integer_multinomial` |
| `intsetdist` | `integer_bernoulli_set` |
| `int_spike` | `integer_uniform_spike` |
| `int_markovchain` | `integer_markov_chain` |
| `int_hidden_association` | `integer_hidden_association` |
| `int_plsi` | `integer_plsi` |
| `int_edit_setdist` | `integer_bernoulli_edit` |
| `int_edit_stepsetdist` | `integer_step_bernoulli_edit` |

For arbitrary finite support, prefer `Categorical`, `Multinomial`, or `BernoulliSet` without `Integer`.

### Function Names

Public methods should be snake_case. The remaining non-snake public names found were:

| Location | Current | Preferred |
| --- | --- | --- |
| `pysp.stats.llda` | `updateAlpha` | `update_alpha` |
| several estimators | `accumulatorFactory` | `accumulator_factory` |

`accumulatorFactory` appears in:

- `pysp.bstats.bernoulli.BernoulliEstimator`
- `pysp.bstats.dirichlet.DirichletEstimator`
- `pysp.bstats.intrange.IntegerCategoricalEstimator`
- `pysp.stats.grammar.GrammarEstimator`
- `pysp.stats.hidden_markov_ind_pi.IndPiHiddenMarkovEstimator`
- `pysp.stats.llda.LLDAEstimator`
- `pysp.stats.markov_transform.MarkovTransformEstimator`
- `pysp.stats.mvnmixture.GaussianMixtureEstimator`

Recommendation: keep these as aliases, but document `accumulator_factory` only.

## Distribution And Estimator Constructor Inventory

This table is the concrete API surface that should drive compatibility aliases.

### `pysp.stats`

| Class | Module | Constructor args |
| --- | --- | --- |
| `BernoulliDistribution` | `pysp.stats.bernoulli` | `p, name=None, keys=None` |
| `BernoulliEstimator` | `pysp.stats.bernoulli` | `pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `BetaDistribution` | `pysp.stats.beta` | `a, b, name=None, keys=None` |
| `BetaEstimator` | `pysp.stats.beta` | `pseudo_count=None, suff_stat=None, delta=1e-08, name=None, keys=None` |
| `BinomialDistribution` | `pysp.stats.binomial` | `p, n, min_val=None, name=None, keys=None` |
| `BinomialEstimator` | `pysp.stats.binomial` | `max_val=None, min_val=0, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `CategoricalDistribution` | `pysp.stats.categorical` | `pmap, default_value=0.0, name=None` |
| `CategoricalEstimator` | `pysp.stats.categorical` | `pseudo_count=None, suff_stat=None, default_value=False, name=None, keys=None` |
| `MultinomialDistribution` | `pysp.stats.catmultinomial` | `dist, len_dist=NullDistribution(), len_normalized=False, name=None` |
| `MultinomialEstimator` | `pysp.stats.catmultinomial` | `estimator, len_estimator=NullEstimator(), len_dist=None, len_normalized=False, name=None, keys=None` |
| `ChowLiuTreeDistribution` | `pysp.stats.chow_liu_tree` | `parents, marginal_dists, conditional_dists, default_dists=None, feature_order=None, parent_values=None, name=None` |
| `ChowLiuTreeEstimator` | `pysp.stats.chow_liu_tree` | `estimators, root=0, pseudo_count=None, mi_pseudo_count=None, default_policy='marginal', keys=None, name=None` |
| `CompositeDistribution` | `pysp.stats.composite` | `dists` |
| `CompositeEstimator` | `pysp.stats.composite` | `estimators, keys=None` |
| `ConditionalDistribution` | `pysp.stats.conditional` | `dmap, default_dist=NullDistribution(), given_dist=NullDistribution(), name=None, keys=None` |
| `ConditionalDistributionEstimator` | `pysp.stats.conditional` | `estimator_map, default_estimator=NullEstimator(), given_estimator=NullEstimator(), name=None, keys=None` |
| `DiracLengthMixtureDistribution` | `pysp.stats.dirac_length` | `len_dist, p, v=0, name=None` |
| `DiracLengthMixtureEstimator` | `pysp.stats.dirac_length` | `estimator, v=0, fixed_p=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)` |
| `DirichletDistribution` | `pysp.stats.dirichlet` | `alpha, name=None, keys=None` |
| `DirichletEstimator` | `pysp.stats.dirichlet` | `dim, pseudo_count=None, suff_stat=None, delta=1e-08, keys=None, use_mpe=False, name=None` |
| `DiagonalGaussianDistribution` | `pysp.stats.dmvn` | `mu, covar, name=None, keys=None` |
| `DiagonalGaussianEstimator` | `pysp.stats.dmvn` | `dim=None, pseudo_count=(None, None), suff_stat=(None, None), name=None, keys=None` |
| `ExponentialDistribution` | `pysp.stats.exponential` | `beta, name=None` |
| `ExponentialEstimator` | `pysp.stats.exponential` | `pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `GammaDistribution` | `pysp.stats.gamma` | `k, theta, name=None` |
| `GammaEstimator` | `pysp.stats.gamma` | `pseudo_count=(0.0, 0.0), suff_stat=(1.0, 0.0), threshold=1e-08, name=None, keys=None` |
| `GaussianDistribution` | `pysp.stats.gaussian` | `mu, sigma2, name=None` |
| `GaussianEstimator` | `pysp.stats.gaussian` | `pseudo_count=(None, None), suff_stat=(None, None), name=None, keys=None` |
| `GeometricDistribution` | `pysp.stats.geometric` | `p, name=None` |
| `GeometricEstimator` | `pysp.stats.geometric` | `pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `GrammarDistribution` | `pysp.stats.grammar` | `grammar, mix_p, decomp_level=0, lhs_delta=0, name=None, orig_n=100` |
| `GrammarEstimator` | `pysp.stats.grammar` | `pseudo_count=None, name=None` |
| `HeterogeneousPCFGDistribution` | `pysp.stats.heterogeneous_pcfg` | `binary_rules, terminal_rules, start=None, nonterminals=None, name=None` |
| `HeterogeneousPCFGEstimator` | `pysp.stats.heterogeneous_pcfg` | `binary_rules, terminal_rules, start=None, nonterminals=None, pseudo_count=None, name=None, keys=(None, None)` |
| `InducedHeterogeneousPCFGEstimator` | `pysp.stats.heterogeneous_pcfg` | `max_nonterminals, terminal_estimators, start='S', nonterminal_prefix='NT', terminal_rule_mass=0.5, rule_pseudo_count=0.001, prune_threshold=0.0, min_rule_prob=0.0, name=None, keys=(None, None)` |
| `HeterogeneousMixtureDistribution` | `pysp.stats.heterogenous_mixture` | `components, w, name=None` |
| `HeterogeneousMixtureEstimator` | `pysp.stats.heterogenous_mixture` | `estimators, fixed_weights=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)` |
| `HiddenAssociationDistribution` | `pysp.stats.hidden_association` | `cond_dist, given_dist=NullDistribution(), len_dist=NullDistribution(), name=None, keys=(None, None)` |
| `HiddenAssociationEstimator` | `pysp.stats.hidden_association` | `cond_estimator, given_estimator=NullEstimator(), len_estimator=NullEstimator(), pseudo_count=None, name=None, keys=(None, None)` |
| `HiddenMarkovModelDistribution` | `pysp.stats.hidden_markov` | `topics, w, transitions, taus=None, len_dist=NullDistribution(), name=None, terminal_values=None, use_numba=False` |
| `HiddenMarkovEstimator` | `pysp.stats.hidden_markov` | `estimators, len_estimator=NullEstimator(), pseudo_count=(None, None), name=None, keys=(None, None, None), use_numba=False` |
| `IndPiHiddenMarkovModelDistribution` | `pysp.stats.hidden_markov_ind_pi` | `topics, w, transitions, taus, len_dist=None, name=None, terminal_values=None, use_numba=True` |
| `IndPiHiddenMarkovEstimator` | `pysp.stats.hidden_markov_ind_pi` | `estimators, len_estimator=None, suff_stat=None, pseudo_count=(None, None), name=None, keys=(None, None, None), use_numba=True` |
| `HierarchicalMixtureDistribution` | `pysp.stats.hmixture` | `topics, mixture_weights, topic_weights, len_dist=NullDistribution(), name=None, keys=(None, None)` |
| `HierarchicalMixtureEstimator` | `pysp.stats.hmixture` | `estimators, num_mixtures, len_estimator=NullEstimator(), len_dist=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)` |
| `IndianBuffetProcessDistribution` | `pysp.stats.ibp` | `num_features, alpha=1.0, beta_params=None, feature_probs=None, min_prob=1e-128, name=None, keys=None, data_format='auto'` |
| `IndianBuffetProcessEstimator` | `pysp.stats.ibp` | `num_features, alpha=1.0, pseudo_count=None, suff_stat=None, estimate_alpha=True, min_alpha=1e-12, max_alpha=1000000000000.0, min_prob=1e-128, name=None, keys=None, data_format='auto'` |
| `ICLTreeDistribution` | `pysp.stats.icltree` | `dependency_list, conditional_log_densities, feature_order=None, name=None` |
| `ICLTreeEstimator` | `pysp.stats.icltree` | `num_features=None, num_states=None, pseudo_count=None, suff_stat=None, keys=None, name=None` |
| `IgnoredDistribution` | `pysp.stats.ignored` | `dist, name=None` |
| `IgnoredEstimator` | `pysp.stats.ignored` | `dist=NullDistribution(), pseudo_count=None, suff_stat=None, keys=None, name=None` |
| `IntegerBernoulliEditDistribution` | `pysp.stats.int_edit_setdist` | `log_edit_pmat, init_dist=NullDistribution(), name=None` |
| `IntegerBernoulliEditEstimator` | `pysp.stats.int_edit_setdist` | `num_vals, init_estimator=NullEstimator(), min_prob=1e-128, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `IntegerStepBernoulliEditDistribution` | `pysp.stats.int_edit_stepsetdist` | `log_edit_pmat, init_dist=None, name=None` |
| `IntegerStepBernoulliEditEstimator` | `pysp.stats.int_edit_stepsetdist` | `num_vals, init_estimator=NullEstimator(), min_prob=1e-128, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `IntegerHiddenAssociationDistribution` | `pysp.stats.int_hidden_association` | `state_prob_mat, cond_weights, alpha=0.0, prev_dist=NullDistribution(), len_dist=NullDistribution(), name=None, keys=(None, None), use_numba=False` |
| `IntegerHiddenAssociationEstimator` | `pysp.stats.int_hidden_association` | `num_vals, num_states, alpha=0.0, prev_estimator=NullEstimator(), len_estimator=NullEstimator(), suff_stat=None, pseudo_count=None, use_numba=False, name=None, keys=(None, None)` |
| `IntegerMarkovChainDistribution` | `pysp.stats.int_markovchain` | `num_values, cond_dist, lag=1, init_dist=NullDistribution(), len_dist=NullDistribution(), keys=None, name=None` |
| `IntegerMarkovChainEstimator` | `pysp.stats.int_markovchain` | `num_values, lag=1, init_estimator=NullEstimator(), len_estimator=NullEstimator(), init_dist=None, len_dist=None, pseudo_count=None, name=None, keys=None` |
| `IntegerPLSIDistribution` | `pysp.stats.int_plsi` | `state_word_mat, doc_state_mat, doc_vec, len_dist=NullDistribution(), name=None` |
| `IntegerPLSIEstimator` | `pysp.stats.int_plsi` | `num_vals, num_states, num_docs, len_estimator=NullEstimator(), pseudo_count=(None, None, None), suff_stat=(None, None, None), name=None, keys=(None, None, None)` |
| `IntegerUniformSpikeDistribution` | `pysp.stats.int_spike` | `k, num_vals, p, min_val=0, name=None` |
| `IntegerUniformSpikeEstimator` | `pysp.stats.int_spike` | `min_val=None, max_val=None, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `IntegerMultinomialDistribution` | `pysp.stats.intmultinomial` | `min_val=0, p_vec=None, len_dist=NullDistribution(), name=None, keys=None` |
| `IntegerMultinomialEstimator` | `pysp.stats.intmultinomial` | `min_val=None, max_val=None, len_estimator=NullEstimator(), len_dist=None, name=None, pseudo_count=None, suff_stat=None, keys=None` |
| `IntegerCategoricalDistribution` | `pysp.stats.intrange` | `min_val, p_vec, name=None` |
| `IntegerCategoricalEstimator` | `pysp.stats.intrange` | `min_val=None, max_val=None, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `IntegerBernoulliSetDistribution` | `pysp.stats.intsetdist` | `log_pvec, log_nvec=None, name=None, keys=None` |
| `IntegerBernoulliSetEstimator` | `pysp.stats.intsetdist` | `num_vals, min_prob=1e-128, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `JointMixtureDistribution` | `pysp.stats.jmixture` | `components1, components2, w1, w2, taus12, taus21, keys=(None, None, None), name=None` |
| `JointMixtureEstimator` | `pysp.stats.jmixture` | `estimators1, estimators2, suff_stat=None, pseudo_count=None, keys=(None, None, None), name=None` |
| `LaplaceDistribution` | `pysp.stats.laplace` | `mu, b, name=None, keys=None` |
| `LaplaceEstimator` | `pysp.stats.laplace` | `pseudo_count=None, suff_stat=None, min_scale=1e-08, name=None, keys=None` |
| `LDADistribution` | `pysp.stats.lda` | `topics, alpha, len_dist=NullDistribution(), gamma_threshold=1e-08` |
| `LDAEstimator` | `pysp.stats.lda` | `estimators, len_estimator=NullEstimator(), suff_stat=None, pseudo_count=None, keys=(None, None), fixed_alpha=None, gamma_threshold=1e-08, alpha_threshold=1e-08` |
| `LLDADistribution` | `pysp.stats.llda` | `topics, alphas, set_dist=None, len_dist=None, gamma_threshold=1e-08` |
| `LLDAEstimator` | `pysp.stats.llda` | `estimators, num_alphas, suff_stat=None, pseudo_count=None, keys=(None, None), fixed_alpha=None, gamma_threshold=1e-08, alpha_threshold=1e-08` |
| `LogGaussianDistribution` | `pysp.stats.log_gaussian` | `mu, sigma2, name=None` |
| `LogGaussianEstimator` | `pysp.stats.log_gaussian` | `pseudo_count=(None, None), suff_stat=(None, None), name=None, keys=None` |
| `LogisticDistribution` | `pysp.stats.logistic` | `loc=0.0, scale=1.0, name=None, keys=None` |
| `LogisticEstimator` | `pysp.stats.logistic` | `pseudo_count=None, suff_stat=None, min_scale=1e-08, name=None, keys=None` |
| `LookbackHiddenMarkovDistribution` | `pysp.stats.look_back_hmm` | `topics, w, transitions, lag=0, init_dist=None, len_dist=NullDistribution(), name=None` |
| `LookbackHiddenMarkovEstimator` | `pysp.stats.look_back_hmm` | `estimators, lag=0, init_estimators=None, len_estimator=NullEstimator(), suff_stat=None, pseudo_count=(None, None), name=None, keys=(None, None, None)` |
| `LookbackHiddenMarkovDistribution` | `pysp.stats.lookback_hmm` | `topics, w, transitions, lag=0, init_dist=None, len_dist=None, name=None` |
| `LookbackHiddenMarkovEstimator` | `pysp.stats.lookback_hmm` | `estimators, lag=0, init_estimators=None, len_estimator=None, suff_stat=None, pseudo_count=(None, None), name=None, keys=(None, None, None)` |
| `MarkovTransformDistribution` | `pysp.stats.markov_transform` | `init_prob_vec, cond_prob_mat, alpha=0.0, len_dist=None` |
| `MarkovTransformEstimator` | `pysp.stats.markov_transform` | `num_vals, alpha=0.0, len_estimator=None, suff_stat=None, pseudo_count=None, keys=(None, None)` |
| `MarkovChainDistribution` | `pysp.stats.markovchain` | `init_prob_map, transition_map, len_dist=NullDistribution(), default_value=0.0, name=None` |
| `MarkovChainEstimator` | `pysp.stats.markovchain` | `pseudo_count=None, levels=None, len_estimator=NullEstimator(), name=None, keys=None` |
| `MixtureDistribution` | `pysp.stats.mixture` | `components, w, name=None` |
| `MixtureEstimator` | `pysp.stats.mixture` | `estimators, fixed_weights=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)` |
| `MultivariateGaussianDistribution` | `pysp.stats.mvn` | `mu, covar, name=None, keys=None` |
| `MultivariateGaussianEstimator` | `pysp.stats.mvn` | `dim=None, pseudo_count=(None, None), suff_stat=(None, None), name=None, keys=None` |
| `GaussianMixtureDistribution` | `pysp.stats.mvnmixture` | `mu, sig2, w, name=None` |
| `GaussianMixtureEstimator` | `pysp.stats.mvnmixture` | `estimators, name=None, conj_prior_params=None, suff_stat=None, pseudo_count=None, keys=(None, None)` |
| `NegativeBinomialDistribution` | `pysp.stats.negative_binomial` | `r, p, name=None, keys=None` |
| `NegativeBinomialEstimator` | `pysp.stats.negative_binomial` | `r=1.0, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `NullDistribution` | `pysp.stats.null_dist` | `name=None` |
| `NullEstimator` | `pysp.stats.null_dist` | `pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `OptionalDistribution` | `pysp.stats.optional` | `dist, p=None, missing_value=None, name=None` |
| `OptionalEstimator` | `pysp.stats.optional` | `estimator, missing_value=None, est_prob=False, pseudo_count=None, name=None, keys=None` |
| `ParetoDistribution` | `pysp.stats.pareto` | `xm, alpha, name=None, keys=None` |
| `ParetoEstimator` | `pysp.stats.pareto` | `pseudo_count=None, suff_stat=None, min_denom=1e-12, name=None, keys=None` |
| `PointMassDistribution` | `pysp.stats.point_mass` | `value, name=None, keys=None` |
| `PointMassEstimator` | `pysp.stats.point_mass` | `value, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `PoissonDistribution` | `pysp.stats.poisson` | `lam, name=None` |
| `PoissonEstimator` | `pysp.stats.poisson` | `pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `QuantizedHiddenMarkovModelDistribution` | `pysp.stats.quantized_hmm` | `theta, levels, transition_exponents, emission_exponents, initial_exponents=None, init_mode='quantized', k_max=None, len_dist=NullDistribution(), name=None, terminal_values=None, use_numba=False` |
| `QuantizedHiddenMarkovEstimator` | `pysp.stats.quantized_hmm` | `num_states, levels=None, pseudo_count=None, k_max=None, fixed_theta=None, init_mode='quantized', len_estimator=NullEstimator(), name=None, keys=(None, None, None), use_numba=False, max_quant_its=50, split_collapsed=True, split_nats=math.log(2.0)` |
| `RayleighDistribution` | `pysp.stats.rayleigh` | `sigma, name=None, keys=None` |
| `RayleighEstimator` | `pysp.stats.rayleigh` | `pseudo_count=None, suff_stat=None, min_sigma=1e-08, name=None, keys=None` |
| `RecordDistribution` | `pysp.stats.record` | `fields, dists=None` |
| `RecordEstimator` | `pysp.stats.record` | `fields, estimators=None` |
| `SegmentalHiddenMarkovModelDistribution` | `pysp.stats.segmental_hmm` | `emissions, w, transitions, len_dist=NullDistribution(), name=None` |
| `SegmentalHiddenMarkovEstimator` | `pysp.stats.segmental_hmm` | `estimators, len_estimator=NullEstimator(), pseudo_count=(None, None), name=None, keys=(None, None, None)` |
| `SelectDistribution` | `pysp.stats.select` | `dists, choice_function` |
| `SelectEstimator` | `pysp.stats.select` | `estimators, choice_function` |
| `SequenceDistribution` | `pysp.stats.sequence` | `dist, len_dist=NullDistribution(), len_normalized=False, name=None` |
| `SequenceEstimator` | `pysp.stats.sequence` | `estimator, len_estimator=NullEstimator(), len_dist=NullDistribution(), len_normalized=False, name=None, keys=None` |
| `BernoulliSetDistribution` | `pysp.stats.setdist` | `pmap, min_prob=1e-128, name=None, keys=None` |
| `BernoulliSetEstimator` | `pysp.stats.setdist` | `min_prob=1e-128, pseudo_count=None, suff_stat=None, name=None, keys=None` |
| `SparseMarkovAssociationDistribution` | `pysp.stats.sparse_markov_transform` | `init_prob_vec, cond_prob_mat, alpha=0.0, len_dist=NullDistribution(), low_memory=False` |
| `SparseMarkovAssociationEstimator` | `pysp.stats.sparse_markov_transform` | `num_vals, alpha=0.0, len_estimator=NullEstimator(), suff_stat=None, pseudo_count=None, low_memory=True, keys=(None, None)` |
| `SpearmanRankingDistribution` | `pysp.stats.spearman_rho` | `sigma, rho=1.0, name=None, keys=None` |
| `SpearmanRankingEstimator` | `pysp.stats.spearman_rho` | `dim, rho=None, pseudo_count=None, suff_stat=None, name=None, keys=None, max_rho=1000000.0` |
| `SemiSupervisedMixtureDistribution` | `pysp.stats.ss_mixture` | `components, w, name=None` |
| `SemiSupervisedMixtureEstimator` | `pysp.stats.ss_mixture` | `estimators, suff_stat=None, pseudo_count=None, keys=(None, None), name=None` |
| `StudentTDistribution` | `pysp.stats.student_t` | `df, loc=0.0, scale=1.0, name=None, keys=None` |
| `StudentTEstimator` | `pysp.stats.student_t` | `df=5.0, pseudo_count=None, suff_stat=None, min_scale=1e-08, name=None, keys=None` |
| `TransformDistribution` | `pysp.stats.transform` | `dist, transform=None, density_correction=None, name=None, keys=None` |
| `TransformEstimator` | `pysp.stats.transform` | `estimator, transform=None, density_correction=None, name=None, keys=None` |
| `TreeHiddenMarkovModelDistribution` | `pysp.stats.tree_hmm` | `topics, w, transitions, len_dist=NullDistribution(), terminal_level=10, name=None, use_numba=False` |
| `TreeHiddenMarkovEstimator` | `pysp.stats.tree_hmm` | `estimators, len_estimator=NullEstimator(), pseudo_count=(None, None), name=None, keys=(None, None, None), use_numba=True` |
| `UniformDistribution` | `pysp.stats.uniform` | `low, high, name=None, keys=None` |
| `UniformEstimator` | `pysp.stats.uniform` | `pseudo_count=None, suff_stat=None, min_width=1e-08, name=None, keys=None` |
| `VonMisesFisherDistribution` | `pysp.stats.vmf` | `mu, kappa, name=None, keys=None` |
| `VonMisesFisherEstimator` | `pysp.stats.vmf` | `dim=None, pseudo_count=None, name=None, keys=None` |
| `WeibullDistribution` | `pysp.stats.weibull` | `shape, scale, name=None, keys=None` |
| `WeibullEstimator` | `pysp.stats.weibull` | `pseudo_count=None, suff_stat=None, min_shape=0.001, max_shape=1000.0, min_scale=1e-12, name=None, keys=None` |
| `WeightedDistribution` | `pysp.stats.weighted` | `dist, name=None` |
| `WeightedEstimator` | `pysp.stats.weighted` | `estimator, name=None` |

### `pysp.bstats`

| Class | Module | Constructor args |
| --- | --- | --- |
| `BernoulliDistribution` | `pysp.bstats.bernoulli` | `p, name=None, prior=default_prior, keys=None` |
| `BernoulliEstimator` | `pysp.bstats.bernoulli` | `name=None, keys=None, prior=default_prior` |
| `BayesianStreamingEstimator` | `pysp.bstats.bestimation` | `estimator, mode='posterior_carry', schedule=None, model=None, init_estimator=None, init_p=0.1, rng=np.random.RandomState(), num_chunks=1` |
| `BetaDistribution` | `pysp.bstats.beta` | `a, b, name=None, prior=None` |
| `BinomialDistribution` | `pysp.bstats.binomial` | `n, p, name=None, prior=default_prior, keys=None` |
| `BinomialEstimator` | `pysp.bstats.binomial` | `n, name=None, keys=None, prior=default_prior` |
| `DictDirichletDistribution` | `pysp.bstats.catdirichlet` | `alpha` |
| `CategoricalDistribution` | `pysp.bstats.categorical` | `prob_map, default_value=0.0, name=None, prior=default_prior` |
| `CategoricalEstimator` | `pysp.bstats.categorical` | `default_value=0.0, name=None, prior=default_prior, keys=(None,)` |
| `CompositeDistribution` | `pysp.bstats.composite` | `dists, name=None, keys=None` |
| `CompositeEstimator` | `pysp.bstats.composite` | `estimators, name=None, keys=None` |
| `ConditionalDistribution` | `pysp.bstats.conditional` | `dmap, cond_dist=None, default_dist=null_dist, pass_value=False` |
| `ConditionalDistributionEstimator` | `pysp.bstats.conditional` | `estimator_map, default_estimator=None, keys=None` |
| `DiracDistribution` | `pysp.bstats.dirac` | `value, name=None, prior=null_dist` |
| `DiracEstimator` | `pysp.bstats.dirac` | `value, prior=null_dist, keys=None` |
| `DirichletDistribution` | `pysp.bstats.dirichlet` | `alpha` |
| `DirichletEstimator` | `pysp.bstats.dirichlet` | `dim, pseudo_count=None, suff_stat=None, delta=1e-08, keys=None, use_mpe=False` |
| `DiagonalGaussianDistribution` | `pysp.bstats.dmvn` | `mu, covariance, name=None, prior=None` |
| `DiagonalGaussianEstimator` | `pysp.bstats.dmvn` | `dim=None, name=None, prior=None` |
| `DirichletProcessMixtureDistribution` | `pysp.bstats.dpm` | `components, w, a, g, component_priors, name=None, prior=default_prior` |
| `DirichletProcessMixtureEstimator` | `pysp.bstats.dpm` | `estimators, name=None, prior=default_prior, keys=(None, None)` |
| `ExponentialDistribution` | `pysp.bstats.exponential` | `lam, name=None, prior=default_prior` |
| `ExponentialEstimator` | `pysp.bstats.exponential` | `prior=default_prior, name=None, keys=(None,)` |
| `GammaDistribution` | `pysp.bstats.gamma` | `k, theta, name=None, prior=null_dist` |
| `GammaEstimator` | `pysp.bstats.gamma` | `pseudo_count=(0.0, 0.0), suff_stat=(1.0, 0.0), threshold=1e-08, name=None, prior=null_dist, keys=None` |
| `GaussianDistribution` | `pysp.bstats.gaussian` | `mu, sigma2, name=None, prior=default_prior` |
| `GaussianEstimator` | `pysp.bstats.gaussian` | `name=None, prior=default_prior, keys=(None, None)` |
| `GeometricDistribution` | `pysp.bstats.geometric` | `p, name=None, prior=default_prior, keys=None` |
| `GeometricEstimator` | `pysp.bstats.geometric` | `name=None, keys=None, prior=default_prior` |
| `HierarchicalDirichletProcessMixtureDistribution` | `pysp.bstats.hdpm` | `components, beta, alpha, gamma, group_weights=None, name=None, len_dist=null_dist` |
| `HierarchicalDirichletProcessMixtureEstimator` | `pysp.bstats.hdpm` | `estimators, gamma=1.0, alpha=1.0, name=None, keys=None, len_estimator=NullEstimator()` |
| `HiddenMarkovModelDistribution` | `pysp.bstats.hidden_markov` | `topics, w, transitions, name=None, prior=None, len_dist=null_dist` |
| `HiddenMarkovModelEstimator` | `pysp.bstats.hidden_markov` | `estimators, name=None, keys=None, prior=None, len_estimator=NullEstimator()` |
| `IgnoredDistribution` | `pysp.bstats.ignored` | `dist=null_dist` |
| `IgnoredEstimator` | `pysp.bstats.ignored` | `dist=null_dist, prior=null_dist, keys=None` |
| `IntegerCategoricalDistribution` | `pysp.bstats.intrange` | `prob_vec=None, default_value=0.0, min_index=0, name=None, prior=default_prior, min_val=None, p_vec=None` |
| `IntegerCategoricalEstimator` | `pysp.bstats.intrange` | `min_index=None, max_index=None, default_value=0.0, name=None, prior=default_prior, keys=(None,), min_val=None, max_val=None` |
| `LogGaussianDistribution` | `pysp.bstats.log_gaussian` | `mu, sigma2, name=None, prior=default_prior` |
| `LogGaussianEstimator` | `pysp.bstats.log_gaussian` | no explicit constructor args |
| `MarkovChainDistribution` | `pysp.bstats.markovchain` | `init_prob_vec, transition_mat, name=None, prior=None, len_dist=null_dist` |
| `MarkovChainEstimator` | `pysp.bstats.markovchain` | `num_states, name=None, keys=None, prior=None, len_estimator=NullEstimator()` |
| `MixtureDistribution` | `pysp.bstats.mixture` | `components, w, name=None, prior=None` |
| `MixtureEstimator` | `pysp.bstats.mixture` | `estimators, fixed_w=None, name=None, prior=default_prior, keys=(None, None)` |
| `MultivariateGaussianDistribution` | `pysp.bstats.mvn` | `mu, covar, name=None, prior=None` |
| `MultivariateGaussianEstimator` | `pysp.bstats.mvn` | `dim, name=None, keys=None, prior=None` |
| `MultivariateNormalGammaDistribution` | `pysp.bstats.mvngamma` | `mu, lam, a, b, name=None, prior=None` |
| `NormalGammaDistribution` | `pysp.bstats.normgamma` | `mu, lam, a, b, name=None, prior=None` |
| `NormalWishartDistribution` | `pysp.bstats.normwishart` | `mu, kappa, w_mat, nu, name=None, prior=None` |
| `NullDistribution` | `pysp.bstats.nulldist` | no explicit constructor args |
| `NullEstimator` | `pysp.bstats.nulldist` | `prior=None, keys=None` |
| `OptionalDistribution` | `pysp.bstats.optional` | `dist, p=0.5, missing_value=None, name=None, prior=default_prior, keys=None` |
| `OptionalEstimator` | `pysp.bstats.optional` | `estimator, missing_value=None, fixed_prob=None, name=None, keys=None, prior=default_prior` |
| `PoissonDistribution` | `pysp.bstats.poisson` | `lam, name=None, prior=default_prior, keys=None` |
| `PoissonEstimator` | `pysp.bstats.poisson` | `name=None, keys=None, prior=default_prior` |
| `SequenceDistribution` | `pysp.bstats.sequence` | `dist, len_dist=null_dist, name=None, len_normalized=False` |
| `SequenceEstimator` | `pysp.bstats.sequence` | `estimator, len_estimator=null_estimator, len_normalized=False, name=None, keys=(None, None)` |
| `BernoulliSetDistribution` | `pysp.bstats.setdist` | `pmap, name=None, prior=None` |
| `BernoulliSetEstimator` | `pysp.bstats.setdist` | `name=None, prior=default_prior, keys=(None,)` |
| `SymmetricDirichletDistribution` | `pysp.bstats.symdirichlet` | `alpha, dim=None` |

## Family Stem Outliers

These are the class families whose related class names are not predictable from a single stem.

| Area | Current issue | Recommendation |
| --- | --- | --- |
| Conditional | `ConditionalDistributionEstimator`, `ConditionalDistributionAccumulator`, etc. include `Distribution` in the stem. | Public aliases `ConditionalEstimator`, `ConditionalAccumulator`, `ConditionalAccumulatorFactory`, `ConditionalDataEncoder`, `ConditionalEnumerator`. |
| Grammar | `GrammarEstimatorAccumulator` exists, but the family stem would predict `GrammarAccumulator`. | Add alias `GrammarAccumulator`. |
| HMMs in `stats` | `HiddenMarkovModelDistribution` pairs with `HiddenMarkovEstimator`, `HiddenMarkovSampler`, etc. | Prefer the full `HiddenMarkovModel*` family, with old aliases. |
| Quantized HMM | `QuantizedHiddenMarkovModelDistribution` pairs with `QuantizedHiddenMarkovEstimator`. | Prefer `QuantizedHiddenMarkovModelEstimator`. |
| Segmental/tree HMM | Distribution names include `Model`, helper class names do not. | Prefer full `SegmentalHiddenMarkovModel*` and `TreeHiddenMarkovModel*` aliases. |
| IndPi HMM | `IndPiHiddenMarkovModelDistribution` pairs with `IndPiHiddenMarkovEstimator`. | Prefer `IndPiHiddenMarkovModel*` aliases. |
| Mixture-style accumulators | Several use `*EstimatorAccumulator` and `*EstimatorAccumulatorFactory`. | Prefer `*Accumulator` and `*AccumulatorFactory` as public names. |
| `mvnmixture` | `GaussianMixtureEstimatorAccumulatorFactory` does not match `GaussianMixtureAccumulatorFactory`. | Add canonical alias. |
| bstats | Many accumulator names include `EstimatorAccumulator`; bstats mostly lacks data encoders/enumerators. | Leave bstats narrower if intentional, but standardize names where public. |

## Module Name Outliers

| Current module | Issue | Recommendation |
| --- | --- | --- |
| `pysp.stats.heterogenous_mixture` | Misspells `heterogeneous`. | Add `pysp.stats.heterogeneous_mixture` alias; later migrate imports. |
| `pysp.stats.look_back_hmm` and `pysp.stats.lookback_hmm` | Duplicate modules with same family name. | Pick one canonical module; make the other compatibility-only. |
| `pysp.stats.mvn`, `pysp.stats.dmvn`, `pysp.stats.mvnmixture` | Abbreviated modules for public model families. | Add descriptive aliases: `multivariate_gaussian`, `diagonal_gaussian`, `gaussian_mixture`. |
| `pysp.stats.catmultinomial` | Non-standard module spelling. | Add `multinomial` alias if not conflicting. |
| `pysp.stats.null_dist` vs `pysp.bstats.nulldist` | Same class, different module convention. | Prefer one spelling, probably `null_dist` or `null_distribution`; alias the other. |
| `pysp.bstats.dirac` vs `pysp.stats.point_mass` | Same conceptual point-mass family uses two names. | Prefer `PointMass` for discrete public APIs; keep `Dirac` for continuous/measure language only if needed. |

## Keying Behavior

Current pattern:

- Most public constructors accept `keys`.
- Many distributions and estimators store `self.keys`.
- Many accumulators store `self.key` for scalar keys.
- Some multi-key accumulators split into names such as `weight_key`, `comp_key`, `state_key`, `trans_key`, `init_key`.

Recommended behavior:

1. Public API always uses `keys`.
2. Store public constructor state as `self.keys`.
3. Accumulators may expose derived internal fields such as `weight_key`, but should preserve `self.keys`.
4. Key tuple roles should be documented by family:
   - Mixture: `(weight_key, component_key)`.
   - HMM-like models: `(initial_key, transition_key, state_key)` or a clearly documented ordering.
   - PCFG: `(binary_rule_key, terminal_rule_key)`.
5. `validate_estimator_keys` should be the single enforcement point for tuple length and role consistency.

## Public Function Name Accounting

The package contains 777 unique module-level function names and 505 unique public method names. Most distribution
protocol names are consistent:

- Distributions: `density`, `log_density`, `seq_log_density`, `sampler`, `estimator`, `dist_to_encoder`,
  `enumerator`, `to_json`, `from_json`.
- Samplers: `sample`, `sample_seq`.
- Estimators: `accumulator_factory`, `estimate`.
- Accumulators: `initialize`, `seq_initialize`, `update`, `seq_update`, `combine`, `value`, `from_value`,
  `key_merge`, `key_replace`, `acc_to_encoder`, `scale`.
- Encoders: `seq_encode`, `__eq__`.
- Backends: `backend_seq_log_density`, `backend_seq_component_log_density`, `backend_stacked_*`,
  `compute_capabilities`, `compute_declaration`.

Current public function/method names that should change:

| Name | Kind | Recommendation |
| --- | --- | --- |
| `accumulatorFactory` | method alias | Keep alias, document and call `accumulator_factory`. |
| `updateAlpha` | module function | Keep alias, document and call `update_alpha`. |
| `get_*_estimator` | factory helpers | Acceptable, but future builder APIs could use `make_*_estimator` or `infer_*_estimator` to distinguish construction from retrieval. |
| `max_its` arguments | function arguments | Prefer `max_iter` in new APIs. |
| `num_vals` arguments | function/class arguments | Prefer `num_values` in new APIs. |

## Priority Migration Plan

1. Add aliases only, no behavior change:
   - `ConditionalEstimator`, `ConditionalAccumulator`, etc.
   - `HiddenMarkovModelEstimator`, `HiddenMarkovModelSampler`, etc.
   - `QuantizedHiddenMarkovModelEstimator`.
   - `GrammarAccumulator`.
   - `*AccumulatorFactory` aliases for the families that currently use `*EstimatorAccumulatorFactory`.
2. Normalize constructor aliases:
   - Accept `weights` anywhere `w` is accepted.
   - Accept `prob_map` for `pmap`.
   - Accept `prob_vec` for `p_vec`.
   - Accept `covariance` for `covar`.
   - Accept `num_values` for `num_vals`.
   - Accept `max_iter` for `max_its`.
3. Normalize serialized/pretty-printed names to preferred names while still decoding legacy names.
4. Add tests that assert the alias constructors produce identical models and JSON.
5. Update docs and examples to the preferred spellings.
6. After at least one release cycle, emit deprecation warnings for legacy spellings.

## Short Version

The package is close to a coherent convention. The best target is:

- Class names: one stem per family.
- Public constructors: descriptive names over abbreviations.
- `Integer` only for dense integer-indexed optimized variants.
- `keys` public everywhere; `self.keys` as stored state.
- Snake case only for public functions.
- Aliases first, warnings later, removals much later.

## Implementation Status

Priority migration plan, additive (non-breaking) phases, landed in the API-naming change:

- **Step 1 (class aliases) — done.** `pysp/utils/aliasing.py` adds the alias helper. Module-level
  aliases added across `pysp.stats` and `pysp.bstats`: every `*EstimatorAccumulator(Factory)` now
  has an `*Accumulator(Factory)` alias; family-stem aliases added for Conditional
  (`ConditionalEstimator`, `ConditionalAccumulator`, `ConditionalAccumulatorFactory`,
  `ConditionalDataEncoder`, `ConditionalEnumerator`), the HMM families
  (`HiddenMarkovModel*`, `QuantizedHiddenMarkovModelEstimator`, `SegmentalHiddenMarkovModel*`,
  `TreeHiddenMarkovModel*`, `IndPiHiddenMarkovModel*`), `GrammarAccumulator`, and
  `GaussianMixtureAccumulatorFactory`. The public ones are re-exported from `pysp.stats`.
- **Step 2 (constructor argument aliases) — done for the high-value renames.** Constructors accept
  both spellings and raise `TypeError` if both are passed: `weights`/`w` (Mixture, Heterogeneous,
  SemiSupervised, GaussianMixture, and the HMM/IndPi/Segmental/Tree/Lookback distributions),
  `prob_map`/`pmap` (categorical, setdist), `prob_vec`/`p_vec` (intrange, intmultinomial),
  `covariance`/`covar` (mvn, dmvn), `max_iter`/`max_its` (EM `run_em`/`RestartEM`, kernel `fit`).
  `num_values`/`num_vals` is done for the single-dimension estimators (markov_transform,
  sparse_markov_transform, intsetdist, int_edit_setdist, int_edit_stepsetdist). Remaining
  multi-dimension integer estimators (`int_plsi`, `int_hidden_association`, `int_spike`) are a
  mechanical follow-up.
- **Step 4 (tests) — done.** `pysp/tests/api_naming_aliases_test.py` covers the helper, the class
  aliases, and the constructor argument aliases (identical-model and mutual-exclusivity checks).
- **Steps 3, 5, 6 — staged follow-ups.** Serialized/pretty-printed names still emit legacy
  spellings (they round-trip through the new aliases), example rewrites beyond the README note are
  deferred, and deprecation warnings remain a later cycle, consistent with "aliases first, warnings
  later, removals much later".

