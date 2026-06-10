import numpy as np
import sys
import time
from pysp.bstats import initialize, seq_estimate, seq_log_density_sum, seq_encode, seq_log_density

def empirical_kl_divergence(dist1, dist2, enc_data):

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

	idx  = rng.rand(sz)
	sidx = np.argsort(idx)

	rv = np.zeros(sz, dtype=int)
	for i in range(k):
		rv[sidx[np.arange(start=i, stop=sz, step=k, dtype=int)]] = i

	return rv


def partition_data_index(sz, pvec, rng):

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

	idx_list = partition_data_index(len(data), pvec, rng)

	return [[data[i] for i in u] for u in idx_list]



def best_of(data, vdata, est, trials, max_its, init_p, delta, rng, init_estimator=None, enc_data=None, enc_vdata=None, out=sys.stdout, print_iter=1):

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


