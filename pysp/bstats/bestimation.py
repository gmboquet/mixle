"""Model-fitting drivers for pysparkplug's Bayesian estimation (pysp.bstats).

The central routine is optimize(), which alternates accumulate-then-estimate
steps (seq_estimate) with evaluations of the penalized objective

	obj = data term + prior term,

where the data term is the observed-data log-likelihood (MAP/EM estimators)
or the sum of per-observation local ELBO contributions (variational
estimators exposing seq_local_elbo), and the prior term comes from
ParameterEstimator.model_log_density (log prior at the estimated parameters
for MAP, the data-independent ELBO terms for VB). With conjugate updates each
iteration cannot decrease obj, so optimize() stops once the improvement falls
below delta. Restarts (best_of), fixed-iteration runs (iterate),
validation-metric hill climbing (hill_climb), and data-splitting utilities
are also provided. All drivers accept local data, pandas DataFrames, or
pyspark RDDs via the dispatch in pysp.bstats.
"""
import numpy as np
import sys
import time
from pysp.bstats import initialize, seq_estimate, seq_log_density_sum, seq_encode, seq_log_density

def empirical_kl_divergence(dist1, dist2, enc_data):
	"""Empirical KL divergence between two models on encoded data.

	Both models are scored on the same encoded sample; their log-densities
	are normalized into empirical distributions over the sample (restricted
	to observations both models score finitely), and KL(p1 || p2) of those
	empirical distributions is returned.

	Args:
		dist1: First distribution (defines p1).
		dist2: Second distribution (defines p2).
		enc_data: Output of seq_encode() compatible with both models.

	Returns:
		Tuple (kl, n_bad_1, n_bad_2) with the empirical KL divergence and
		the number of observations each model failed to score (NaN/-inf).
	"""
	ll = seq_log_density(enc_data, estimate=(dist1, dist2), is_list=True)

	r1 = 0.0
	r2 = 0
	r3 = 0

	ll = np.hstack(ll)

	l1 = ll[0, :]
	l2 = ll[1, :]
	g1 = np.bitwise_and(l1 != -np.inf, ~np.isnan(l1))
	g2 = np.bitwise_and(l2 != -np.inf, ~np.isnan(l2))
	gg = np.bitwise_and(g1, g2)

	max_l1 = np.max(l1[gg])
	max_l2 = np.max(l2[gg])


	p1 = np.exp(l1[gg] - max_l1)
	p1 /= p1.sum()

	p2 = np.exp(l2[gg] - max_l2)
	p2 /= p2.sum()

	r1 = (p1*(np.log(p1) - np.log(p2))).sum()
	r2 = (~g1).sum()
	r3 = (~g2).sum()

	return r1, r2, r3

def k_fold_split_index(sz, k, rng):
	"""Assign sz items to k folds of (near-)equal size in random order.

	Args:
		sz (int): Number of items.
		k (int): Number of folds.
		rng (numpy.random.RandomState): Source of shuffling randomness.

	Returns:
		Numpy integer array of length sz with fold labels in [0, k).
	"""
	idx  = rng.rand(sz)
	sidx = np.argsort(idx)

	rv = np.zeros(sz, dtype=int)
	for i in range(k):
		rv[sidx[np.arange(start=i, stop=sz, step=k, dtype=int)]] = i

	return rv


def partition_data_index(sz, pvec, rng):
	"""Randomly partition index range [0, sz) into parts with proportions pvec.

	Args:
		sz (int): Number of items.
		pvec: Sequence of partition proportions (should sum to at most 1).
		rng (numpy.random.RandomState): Source of shuffling randomness.

	Returns:
		List of index arrays, one per entry of pvec.
	"""
	idx  = rng.rand(sz)
	sidx = np.argsort(idx)

	rv = []
	p_tot = 0
	prev_idx = 0

	for p in pvec:
		next_idx = int(round(sz*(p_tot + p), 0))
		rv.append(sidx[prev_idx:next_idx])
		p_tot += p
		prev_idx = next_idx

	return rv

def partition_data(data, pvec, rng):
	"""Randomly partition data into parts with proportions pvec.

	Args:
		data: Indexable sequence of observations.
		pvec: Sequence of partition proportions (should sum to at most 1).
		rng (numpy.random.RandomState): Source of shuffling randomness.

	Returns:
		List of observation lists, one per entry of pvec.
	"""
	idx_list = partition_data_index(len(data), pvec, rng)

	return [[data[i] for i in u] for u in idx_list]



def best_of(data, vdata, est, trials, max_its, init_p, delta, rng, init_estimator=None, enc_data=None, enc_vdata=None, out=sys.stdout, print_iter=1):
	"""Run several randomly-initialized EM fits and keep the best one.

	Each trial initializes a model from data, runs up to max_its
	accumulate-then-estimate iterations (stopping when the training
	log-likelihood gain drops below delta), and scores the result on the
	validation data; the model with the highest validation log-likelihood
	across trials is returned.

	Args:
		data: Training observations (iterable, DataFrame, or RDD).
		vdata: Validation observations used to pick the winning trial.
		est: ParameterEstimator used for the EM updates.
		trials (int): Number of random restarts.
		max_its (int): Maximum iterations per trial.
		init_p (float): Inclusion probability for the random initialization.
		delta (Optional[float]): Early-stopping threshold on the training
			log-likelihood gain (None disables early stopping).
		rng (numpy.random.RandomState): Source of initialization randomness.
		init_estimator: Estimator used only for initialization (defaults to
			est).
		enc_data: Optional pre-encoded training data.
		enc_vdata: Optional pre-encoded validation data.
		out: Stream for progress messages.
		print_iter (int): Progress is printed every print_iter iterations.

	Returns:
		Tuple (best validation log-likelihood, best model).
	"""
	rv_ll = -np.inf
	rv_mm = None


	if init_estimator is None:
		iest = est
	else:
		iest = init_estimator

	for kk in range(trials):

		mm = initialize(data, iest, rng, init_p)

		if enc_data is None:
			enc_data = seq_encode(data, mm)
		if enc_vdata is None:
			enc_vdata = seq_encode(vdata, mm)

		_, old_ll = seq_log_density_sum(enc_data, mm)
		#_, old_vll = seq_log_density_sum(enc_vdata, mm)

		for i in range(max_its):

			mm_next = seq_estimate(enc_data, est, mm)
			_, ll = seq_log_density_sum(enc_data, mm_next)
			#_, vll = seq_log_density_sum(enc_vdata, mm_next)

			#dvll = vll - old_vll
			dll = ll - old_ll

			if (i+1) % print_iter == 0:
				out.write('Iteration %d. LL=%f, delta LL=%e\n'% (i+1, ll, dll))

			if (dll >= 0) or (delta is None):
				mm = mm_next

			if (delta is not None) and (dll < delta):
				break

			old_ll = ll
			#old_vll = vll


		_, vll = seq_log_density_sum(enc_vdata, mm)
		out.write('Trial %d. VLL=%f\n' % (kk + 1, vll))

		if vll > rv_ll:
			rv_mm = mm
			rv_ll = vll

	return rv_ll, rv_mm


def _data_objective_sum(enc_data, model):
	"""Data-dependent part of the optimization objective.

	For variational models exposing seq_local_elbo this is the sum of the
	per-observation local ELBO contributions; otherwise it is the observed
	data log-likelihood at the current parameter estimates.
	"""
	if hasattr(model, 'seq_local_elbo'):
		return sum([model.seq_local_elbo(u[1]).sum() for u in enc_data])
	else:
		_, rv = seq_log_density_sum(enc_data, model)
		return rv


def _model_objective(estimator, model):
	"""Prior/global part of the optimization objective.

	For MAP estimators this is the log prior density of the estimated
	parameters; for variational estimators this is the data-independent part
	of the ELBO (prior cross-entropies plus variational entropies).
	"""
	if hasattr(estimator, 'model_log_density'):
		rv = estimator.model_log_density(model)
		return 0.0 if rv is None else rv
	return 0.0


def optimize(data, estimator, max_its=10, delta=1.0e-6, init_estimator=None, init_p=0.1, rng=np.random.RandomState(), prev_estimate=None, vdata=None, enc_data=None, enc_vdata=None, out=sys.stdout, print_iter=1):
	"""Iterate EM/VB updates, accepting steps that increase the penalized
	objective obj = data term + prior term and stopping when the improvement
	falls below delta.

	The data term is the log-likelihood (MAP estimators) or the local ELBO
	contributions (variational estimators with seq_local_elbo); the prior
	term comes from estimator.model_log_density. Convergence is checked on
	the combined objective, so the prior is part of the stopping rule.

	Args:
		data: Training observations (iterable, DataFrame, or RDD).
		estimator: ParameterEstimator used for the EM/VB updates.
		max_its (int): Maximum number of iterations.
		delta (Optional[float]): Stop when the objective gain drops below
			this value (None disables the convergence check).
		init_estimator: Estimator used only for initialization (defaults to
			estimator).
		init_p (float): Inclusion probability for the random initialization.
		rng (numpy.random.RandomState): Source of initialization randomness.
		prev_estimate: Warm-start model; skips initialization when given.
		vdata: Validation observations (defaults to data); the model with
			the best validation log-likelihood seen during the run is
			returned.
		enc_data: Optional pre-encoded training data.
		enc_vdata: Optional pre-encoded validation data.
		out: Stream for progress messages.
		print_iter (int): Progress is printed every print_iter iterations.

	Returns:
		The model with the highest validation log-likelihood encountered.
	"""
	div_error = np.geterr()
	np.seterr(divide='ignore')

	if init_estimator is None:
		iest = estimator
	else:
		iest = init_estimator

	if prev_estimate is None:
		mm = initialize(data, iest, rng, init_p)
	else:
		mm = prev_estimate

	if vdata is None:
		vdata = data

	if enc_data is None:
		enc_data = seq_encode(data, mm)

	if enc_vdata is None:
		enc_vdata = seq_encode(vdata, mm)

	_, old_vll = seq_log_density_sum(enc_vdata, mm)

	old_obj = _data_objective_sum(enc_data, mm) + _model_objective(estimator, mm)

	best_model = mm
	best_vll = old_vll

	for i in range(max_its):

		mm_next = seq_estimate(enc_data, estimator, mm)

		model_ll = _model_objective(estimator, mm_next)
		data_ll  = _data_objective_sum(enc_data, mm_next)
		obj      = data_ll + model_ll

		_, vll = seq_log_density_sum(enc_vdata, mm_next)

		dobj = obj - old_obj

		if (dobj >= 0) or (delta is None):
			mm = mm_next

		if (delta is not None) and (dobj < delta):
			out.write('Terminating %d. OBJ=%f, dOBJ=%e, LL=%f, MLL=%f, VLL=%f\n' % (i+1, obj, dobj, data_ll, model_ll, vll))
			break

		if (i+1) % print_iter == 0:
			out.write('Iteration %d. OBJ=%f, dOBJ=%e, LL=%f, MLL=%f, VLL=%f\n' % (i+1, obj, dobj, data_ll, model_ll, vll))

		old_obj = obj
		old_vll = vll

		if best_vll < vll:
			best_vll = vll
			best_model = mm

	np.seterr(divide=div_error['divide'])

	return best_model

def iterate(data, estimator, max_its, prev_estimate=None, init_p=0.1, rng=np.random.RandomState(), out=sys.stdout, is_encoded=False, init_estimator=None, print_iter=1):
	"""Run a fixed number of accumulate-then-estimate iterations.

	Unlike optimize(), no objective is tracked and no convergence check is
	performed; the model after max_its iterations is returned.

	Args:
		data: Training observations, or pre-encoded data when is_encoded is
			True.
		estimator: ParameterEstimator used for the EM/VB updates.
		max_its (int): Number of iterations to run.
		prev_estimate: Warm-start model; skips initialization when given.
		init_p (float): Inclusion probability for the random initialization.
		rng (numpy.random.RandomState): Source of initialization randomness.
		out: Stream for progress messages.
		is_encoded (bool): If True, data is already the output of
			seq_encode().
		init_estimator: Estimator used only for initialization (defaults to
			estimator).
		print_iter (int): Progress is printed every print_iter iterations.

	Returns:
		The model after max_its iterations.
	"""
	if init_estimator is None:
		iest = estimator
	else:
		iest = init_estimator

	if prev_estimate is None:
		mm = initialize(data, iest, rng, init_p)
	else:
		mm = prev_estimate

	if is_encoded:
		enc_data = data
	else:
		enc_data = seq_encode(data, mm)

	if hasattr(enc_data, 'cache'):
		enc_data.cache()

	t0 = time.time()
	for i in range(max_its):

		mm = seq_estimate(enc_data, estimator, mm)

		if (i+1) % print_iter == 0:
			out.write('Iteration %d\t E[dT]=%f.\n'% (i+1, (time.time()-t0)/float(i+1)))

	return mm


def hill_climb(data, vdata, estimator, prev_estimate, max_its, metric_lambda, best_estimate=None, enc_data=None, enc_vdata=None, out=sys.stdout, print_iter=1):
	"""Iterate EM updates, keeping the model that maximizes a validation metric.

	Every iteration applies seq_estimate to the training data; the returned
	model is the iterate with the best metric_lambda(vdata, model) score
	(ties broken by validation log-likelihood), which need not be the final
	iterate.

	Args:
		data: Training observations.
		vdata: Validation observations scored by metric_lambda.
		estimator: ParameterEstimator used for the EM updates.
		prev_estimate: Starting model.
		max_its (int): Number of iterations to run.
		metric_lambda: Callable (vdata, model) -> float; higher is better.
		best_estimate: Optional incumbent model to beat (defaults to
			prev_estimate).
		enc_data: Optional pre-encoded training data.
		enc_vdata: Optional pre-encoded validation data.
		out: Stream for progress messages.
		print_iter (int): Progress is printed every print_iter iterations.

	Returns:
		The model with the best validation metric encountered.
	"""
	mm = prev_estimate

	if enc_data is None:
		enc_data = mm.seq_encode(data)
		enc_data = [(len(data), enc_data)]
	if enc_vdata is None:
		enc_vdata = mm.seq_encode(vdata)
		enc_vdata = [(len(vdata), enc_vdata)]

	best_model = prev_estimate if best_estimate is None else best_estimate
	_, best_ll = seq_log_density_sum(enc_vdata, best_model)
	best_score = metric_lambda(vdata, best_model)

	for i in range(max_its):

		mm_next = seq_estimate(enc_data, estimator, mm)

		_, next_ll = seq_log_density_sum(enc_vdata, mm_next)
		next_score = metric_lambda(vdata, mm_next)

		if (next_score > best_score) or ((next_score == best_score) and (best_ll < next_ll)):
			best_model = mm_next
			best_ll    = next_ll
			best_score = next_score


		if i % print_iter == 0:
			out.write('Iteration %d. LL=%f, Best LL=%f, Best Score=%f\n'% (i+1, next_ll, best_ll, best_score))

		mm = mm_next

	return best_model


