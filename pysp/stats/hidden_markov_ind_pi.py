from numpy.random import RandomState
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from pysp.stats.markovchain import MarkovChainDistribution
from pysp.stats.mixture import MixtureDistribution
import numba
import numpy as np
import math

class IndPiHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, topics, w, transitions, taus, len_dist=None, name=None, terminal_values=None, use_numba=True):

		self.use_numba = use_numba

		with np.errstate(divide='ignore'):

			self.topics           = topics
			self.nTopics          = len(topics)
#			self.nStates          = len(w)
			self.nStates          = len(w[0])
			self.w                = vec.make(w)
#			self.logW             = log(self.w)
			self.logW	      = log(np.sum(self.w,axis=0)/len(self.w))
			self.transitions      = np.reshape(transitions, (self.nStates, self.nStates))
			self.logTransitions   = log(self.transitions)
			self.terminal_values  = terminal_values
			self.len_dist         = len_dist
			self.name             = name

		if taus is not None:
			self.taus = vec.make(taus)
			self.logTaus = log(self.taus)
			self.has_topics = True
		else:
			self.taus = None
			self.has_topics = False

	def __str__(self):

		s1 = ','.join(map(str, self.topics))
		s2 = repr(list(self.w))
		s3 = repr([list(u) for u in self.transitions])
		if self.taus is None:
			s4 = repr(self.taus)
		else:
			s4 = repr([list(u) for u in self.taus])
		s5 = str(self.len_dist)
		s6 = repr(self.name)

		return 'IndPiHiddenMarkovModelDistribution([%s], %s, %s, %s, len_dist=%s, name=%s)'%(s1, s2, s3, s4, s5, s6)

	def density(self, x):
		return exp(self.log_density(x))

	def log_density(self, x):

		if x is None or len(x) == 0:
			if self.len_dist is not None:
				return self.len_dist.log_density(0)
			else:
				return 0.0


		if not self.has_topics:

			log_w      = self.logW
			num_states = self.nStates
			comps      = self.topics

			obs_log_likelihood = np.zeros(num_states, dtype=np.float64)
			obs_log_likelihood += log_w
#			obs_log_likelihood += np.sum(log_w,axis=0)
			for i in range(num_states):
				obs_log_likelihood[i] += comps[i].log_density(x[0])

			if np.max(obs_log_likelihood) == -np.inf:
				return -np.inf

			max_ll = obs_log_likelihood.max()
			obs_log_likelihood -= max_ll
			np.exp(obs_log_likelihood, out=obs_log_likelihood)
			sum_ll = np.sum(obs_log_likelihood)
			retval = np.log(sum_ll) + max_ll

			for k in range(1, len(x)):

				#  P(Z(t+1) | Z(t) = i) P(Z(t) = i | X(t), X(t-1), ...)
				np.dot(self.transitions.T, obs_log_likelihood, out=obs_log_likelihood)
				obs_log_likelihood /= obs_log_likelihood.sum()

				# log P(Z(t+1) | X(t), X(t-1), ...)
				np.log(obs_log_likelihood, out=obs_log_likelihood)

				# log P(X(t+1) | Z(t+1)=i) + log P(Z(t+1)=i | X(t), X(t-1), ...)
				for i in range(num_states):
					obs_log_likelihood[i] += comps[i].log_density(x[k])

				# P(X(t+1) | X(t), X(t-1), ...)  [prevent underflow]
				max_ll = obs_log_likelihood.max()
				obs_log_likelihood -= max_ll
				np.exp(obs_log_likelihood, out=obs_log_likelihood)
				sum_ll = np.sum(obs_log_likelihood)

				# P(X(t+1), X(t), ...)
				retval += np.log(sum_ll) + max_ll

			if self.len_dist is not None:
				retval += self.len_dist.log_density(len(x))

			return retval


		else:
			xIter   = iter(x)
			logW    = self.logW
			logTaus = self.logTaus
			nStates = self.nStates
			x0      = xIter.next()

			obsLogDensityByTopic  = [u.logDensity(x0) for u in self.topics]
			logLikelihoodByState  = [logW[i] + vec.weighted_log_sum(obsLogDensityByTopic, logTaus[i,:]) for i in range(nStates)]

			for x in xIter:
				obsLogDensityByTopic = [u.logDensity(x) for u in self.topics]
				logLikelihoodByState = [vec.weighted_log_sum(obsLogDensityByTopic, logTaus[:, i]) + vec.weighted_log_sum(obsLogDensityByTopic, logTaus[i, :]) for i in range(nStates)]

			rv = vec.log_sum(logLikelihoodByState)
			if self.len_dist is not None:
				rv += self.len_dist.log_density(len(x))

			return rv

	def seq_log_density(self, x):


		x0, x1 = x
		if x1 is None:

			num_states = self.nStates
			(tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x0
			w = self.w
			A = self.transitions

			max_len = len(idx_bands)
			num_seq = idx_mat.shape[0]

			good = idx_mat >= 0

			pr_obs = np.zeros((tot_cnt, num_states))
			ll_ret = np.zeros(num_seq)


			# Compute state likelihood vectors and scale the max to one
			for i in range(num_states):
				pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

			pr_max0 = pr_obs.max(axis=1, keepdims=True)
			pr_obs -= pr_max0
			np.exp(pr_obs, out=pr_obs)



			# Vectorized alpha pass
			band = idx_bands[0]
			alphas_prev = np.multiply(pr_obs[band[0]:band[1], :], w)
			temp = alphas_prev.sum(axis=1, keepdims=True)
			#temp2 = temp.copy()
			#temp2[temp2 == 0] = 1.0
			alphas_prev /= temp

			np.log(temp, out=temp)
			temp2 = pr_max0[band[0]:band[1], 0]
			ll_ret[good[:,0]] += temp[:,0] + temp2



			for i in range(1, max_len):
				band = idx_bands[i]
				has_next_loc = has_next[i-1]

				alphas_next = np.dot(alphas_prev[has_next_loc, :], A)
				alphas_next *= pr_obs[band[0]:band[1], :]
				pr_max = alphas_next.sum(axis=1, keepdims=True)
				#pr_max2 = pr_max.copy()
				#pr_max2[pr_max2 == 0] = 1.0
				alphas_next /= pr_max
				alphas_prev = alphas_next

				np.log(pr_max, out=pr_max)
				temp2 = pr_max0[band[0]:band[1], 0]
				ll_ret[good[:,i]] += pr_max[:,0] + temp2

			#nz = len_vec != 0
			#ll_ret[nz] /= len_vec[nz]

			ll_ret[np.isnan(ll_ret)] = -np.inf

			if self.len_dist is not None:
				ll_ret += self.len_dist.seq_log_density(len_enc)

			return ll_ret

		else:

			num_states = self.nStates
			(idx, sz, enc_data), len_enc = x1

			w = self.w
			A = self.transitions
			tot_cnt = len(idx)
			num_seq = len(sz)

			pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
			ll_ret = np.zeros(num_seq, dtype=np.float64)
			tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

			# Compute state likelihood vectors and scale the max to one
			for i in range(num_states):
				pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

			pr_max0 = pr_obs.max(axis=1)
			pr_obs -= pr_max0[:,None]
			np.exp(pr_obs, out=pr_obs)

			alpha_buff = np.zeros((num_seq, num_states), dtype=np.float64)
			next_alpha = np.zeros((num_seq, num_states), dtype=np.float64)

			w_sum = np.sum(w,axis=0)
			w_sum /= len(w)

			numba_seq_log_density(num_states, tz, pr_obs, w_sum, A, pr_max0, next_alpha, alpha_buff, ll_ret)

			if self.len_dist is not None:
				ll_ret += self.len_dist.seq_log_density(len_enc)

			return ll_ret

	def _seq_encode(self, x):

		cnt = len(x)
		len_vec = [len(u) for u in x]

		if self.len_dist is not None:
			len_enc = self.len_dist.seq_encode(len_vec)
		else:
			len_enc = None

		len_vec = np.asarray(len_vec)
		max_len = len_vec.max()
		#len_cnt = np.bincount(len_vec)

		seq_x = []
		idx_loc = 0
		idx_mat = np.zeros((cnt, max_len))-1
		idx_bands = []
		has_next = []
		idx_vec = []

		for i in range(max_len):
			i0 = idx_loc
			has_next_loc = []
			for j in range(cnt):
				if i < len_vec[j]:

					if i < (len_vec[j]-1):
						has_next_loc.append(idx_loc-i0)
					idx_vec.append(j)
					seq_x.append(x[j][i])
					idx_mat[j,i] = idx_loc
					idx_loc += 1

			has_next.append(np.asarray(has_next_loc))
			idx_bands.append((i0, idx_loc))

		tot_cnt = len(seq_x)
		enc_data = self.topics[0].seq_encode(seq_x)
		idx_vec = np.asarray(idx_vec)


		rv = ((tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc)
		return rv, None


	def seq_encode(self, x):

		if not self.use_numba:
			return self._seq_encode(x)

		idx = []
		xs  = []
		sz  = []

		for i, xx in enumerate(x):
			idx.extend([i]*len(xx))
			xs.extend(xx)
			sz.append(len(xx))

		if self.len_dist is not None:
			len_enc = self.len_dist.seq_encode(sz)
		else:
			len_enc = None

		idx = np.asarray(idx, dtype=np.int32)
		sz  = np.asarray(sz, dtype=np.int32)
		xs  = self.topics[0].seq_encode(xs)

		return None, ((idx, sz, xs), len_enc)

	def sampler(self, seed=None):
		return IndPiHiddenMarkovSampler(self, seed)

	def enumerator(self):
		"""Returns an enumerator over observation sequences in descending marginal probability order.

		Reuses HiddenMarkovModelEnumerator (same forward semantics) with this distribution's
		averaged initial state vector logW.
		"""
		from pysp.stats.hidden_markov import HiddenMarkovModelEnumerator
		return HiddenMarkovModelEnumerator(self, topics=self.topics, log_w=self.logW,
		                                   log_transitions=self.logTransitions, len_dist=self.len_dist,
		                                   path_root='IndPiHiddenMarkovModelDistribution')

	def estimator(self, pseudo_count=None):
		len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
		comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
		return IndPiHiddenMarkovEstimator(comp_ests, pseudo_count=(pseudo_count,pseudo_count), len_estimator=len_est)

class IndPiHiddenMarkovSampler(object):

	def __init__(self, dist, seed):
		self.num_states = dist.nStates
		self.dist       = dist
		self.rng        = RandomState(seed)

# cycle through available stateSamplers
		self.iter	= 0

		if dist.has_topics:
			self.obsSamplers = [MixtureDistribution(dist.topics, dist.taus[i,:]).sampler(seed=self.rng.randint(maxint)) for i in range(dist.nStates)]
		else:
			self.obsSamplers = [dist.topics[i].sampler(seed=self.rng.randint(maxint)) for i in range(dist.nStates)]

		if dist.len_dist is not None:
			self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(maxint))
		else:
			self.len_sampler = None

		if dist.terminal_values is None:
			self.terminal_set = None
		else:
			self.terminal_set = set(dist.terminal_values)

		tMap = {i: {k: dist.transitions[i,k] for k in range(dist.nStates)} for i in range(dist.nStates)}

# need a chain for each sequence

		self.stateSamplers = []
		for ws in self.dist.w:
#		for idist in self.dist:
#			pMap = {i: idist.w[i] for i in range(dist.nStates)}
#			self.stateSamplers.append(MarkovChainDistribution({i: idist.w[i] for i in range(dist.nStates)}, tMap).sampler(seed=self.rng.randint(maxint))
			pMap = {i: ws[i] for i in range(dist.nStates)}
			self.stateSamplers.append(MarkovChainDistribution(pMap,tMap).sampler(seed=self.rng.randint(maxint)))
#			self.stateSamplers.append(CategoricalDistribution(pMap).sampler(seed=self.rng.randint(maxint)))


#		self.stateSampler = MarkovChainDistribution(pMap, tMap).sampler(seed=self.rng.randint(maxint))  
#		self.stateSamplers = [MarkovChainDistribution({i: idist.w[i] for i in range(dist.nStates)}, tMap).sampler(seed=self.rng.randint(maxint)) for idist in self.dist ]


	def sample_seq(self, size=None):


		if size is None:
			n = self.len_sampler.sample()
#			stateSeq = self.stateSampler.sample_seq(n)
			stateSeq = self.stateSamplers[self.iter].sample_seq(n)

# stateSeq no longer one markov chain, sample according to position

#			stateSeq = [self.stateSamplers[i].sample() for i in range(n)]

			self.iter += 1
			if self.iter >= len(self.stateSamplers):
				self.iter = 0
			
			obsSeq   = [self.obsSamplers[stateSeq[i]].sample() for i in range(n)]
			

			return obsSeq

		else:
			n = self.len_sampler.sample(size=size)
#			stateSeq = [self.stateSampler.sample_seq(size=nn) for nn in n]
#			stateSeq = [self.stateSamplers[self.iter].sample_seq(size=nn) for nn in n]
			
			
#			obsSeq   = [[self.obsSamplers[j].sample() for j in nn] for nn in stateSeq]

#			stateSeq = []
			obsSeq = []
			for i in range(size):
				stateSeq = self.stateSamplers[self.iter].sample_seq(size=n[i])
				obsSeq.append([self.obsSamplers[j].sample() for j in stateSeq])

				self.iter += 1
				if self.iter >= len(self.stateSamplers):
					self.iter = 0

			return obsSeq

	def sample_terminal(self, terminal_set):

		z = self.stateSamplers[self.iter].sample_seq()
		rv = [self.obsSamplers[z].sample()]

		self.iter += 1
		if self.iter >= len(self.stateSamplers):
			self.iter = 0

		while rv[-1] not in terminal_set:
			z = self.stateSampler.sample_seq(v0=z)

			self.iter += 1
			if self.iter >= len(self.stateSamplers):
				self.iter = 0

			rv.append(self.obsSamplers[z].sample())

		return rv



	def sample(self, size=None):


		if self.len_sampler is not None:
			return self.sample_seq(size=size)

		elif self.terminal_set is not None:
			if size is None:
				return self.sample_terminal(self.terminal_set)
			else:
				return [self.sample_terminal(self.terminal_set) for i in range(size)]

		else:
			raise RuntimeError('IndPiHiddenMarkovSampler requires either a length distribution or terminal value set.')




class IndPiHiddenMarkovEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulators, len_accumulator=None,  keys=(None, None, None), init_counts=None):
		self.accumulators = accumulators
		self.num_states = len(accumulators)
#		self.init_counts = vec.zeros(self.num_states)
		self.init_counts = init_counts
		self.init_counts_initialized = True
		if self.init_counts is None:
			self.init_counts = np.array([])
			self.init_counts_initialized = False
		self.trans_counts = vec.zeros((self.num_states, self.num_states))
		self.state_counts = vec.zeros(self.num_states)
		self.len_accumulator = len_accumulator

		self.init_key = keys[0]
		self.trans_key = keys[1]
		self.state_key = keys[2]

	def update(self, x, weight, estimate):
		self.seq_update(estimate.seq_encode([x]), np.asarray([weight]), estimate)

	def initialize(self, x, weight, rng):

		n = len(x)

		if self.len_accumulator is not None:
			self.len_accumulator.initialize(n, weight, rng)

#		if self.init_counts is None:
#			self.init_counts = np.zeros((n,self.num_states))
			

		if n > 0:

			idx1 = rng.choice(self.num_states)

#			nr = rng.choice(n)

#			self.init_counts[nr][idx1]  += weight
#			self.init_counts[0][idx1]  += weight

			if not self.init_counts_initialized:
				self.init_counts = np.append(self.init_counts,np.zeros(self.num_states))
				self.init_counts = self.init_counts.reshape(( int(len(self.init_counts)/self.num_states),self.num_states))
				
#				self.init_counts[-1][idx1] += weight
				for idx1 in range(self.num_states):
					self.init_counts[-1][idx1] += weight


			#self.state_counts[idx1] += weight / float(n)
			self.state_counts[idx1] += weight

			for j in range(self.num_states):
				#w = weight/float(n) if j == idx1 else 0.0
				w = weight if j == idx1 else 0.0
				self.accumulators[j].initialize(x[0], w, rng)

			for i in range(1, len(x)):
				idx2 = rng.choice(self.num_states)
				#self.trans_counts[idx1,idx2] += weight/(float(n)-1)
				#self.state_counts[idx2] += weight/float(n)
				self.trans_counts[idx1,idx2] += weight
				self.state_counts[idx2] += weight

				for j in range(self.num_states):
					#w = weight/float(n) if j == idx2 else 0.0
					w = weight if j == idx2 else 0.0
					self.accumulators[j].initialize(x[i], w, rng)
				idx1 = idx2


	def seq_update(self, x, weights, estimate):

		x0, x1 = x

		if x1 is None:

			num_states = self.num_states
			(tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x0
			w = estimate.w
			A = estimate.transitions

			max_len = len(idx_bands)
			num_seq = idx_mat.shape[0]

			good = idx_mat >= 0

			pr_obs = np.zeros((tot_cnt, num_states))
			alphas = np.zeros((tot_cnt, num_states))


			# Compute state likelihood vectors and scale the max to one
			for i in range(num_states):
				pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)

			pr_max = pr_obs.max(axis=1, keepdims=True)
			pr_obs -= pr_max
			np.exp(pr_obs, out=pr_obs)


			# Vectorized alpha pass
			band = idx_bands[0]
			alphas_prev = alphas[band[0]:band[1], :]
			np.multiply(pr_obs[band[0]:band[1], :], w, out=alphas_prev)
			pr_sum = alphas_prev.sum(axis=1, keepdims=True)
			pr_sum[pr_sum == 0] = 1.0
			alphas_prev /= pr_sum

			for i in range(1, max_len):
				band = idx_bands[i]
				has_next_loc = has_next[i-1]
				alphas_next = alphas[band[0]:band[1], :]
				np.dot(alphas_prev[has_next_loc, :], A, out=alphas_next)
				alphas_next *= pr_obs[band[0]:band[1], :]
				pr_max = alphas_next.sum(axis=1, keepdims=True)

				pr_max[pr_max == 0] = 1.0

				alphas_next /= pr_max
				alphas_prev = alphas_next


			band2 = idx_bands[-1]
			prev_beta = np.ones((band2[1]-band2[0], num_states))
			alphas[band2[0]:band2[1], :] /= alphas[band2[0]:band2[1], :].sum(axis=1, keepdims=True)

			# Vectorized beta pass
			for i in range(max_len-2, -1, -1):
				band1 = idx_bands[i]
				band2 = idx_bands[i+1]
				has_next_loc = has_next[i]

				next_b = pr_obs[band2[0]:band2[1], :]
				prev_a = alphas[band1[0]:band1[1], :]
				prev_a = prev_a[has_next_loc, :]

				prev_beta *= next_b


				prev_a = np.reshape(prev_a, (prev_a.shape[0], prev_a.shape[1], 1))
				next_beta2 = np.reshape(prev_beta, (prev_beta.shape[0], 1, prev_beta.shape[1]))
				xi_loc = next_beta2*A
				next_beta = xi_loc.sum(axis=2)
				next_beta_max = next_beta.max(axis=1, keepdims=True)
				next_beta_max[next_beta_max == 0] = 1.0
				next_beta /= next_beta_max

				prev_beta = np.ones((band1[1] - band1[0], num_states))
				prev_beta[has_next_loc, :] = next_beta

				xi_loc *= prev_a
				#xi_loc = np.einsum('Bi,ij,Bj->Bij', prev_a, A, next_beta)
				xi_loc_sum = xi_loc.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
				len_vec_loc = np.reshape(len_vec[good[:, i+1]], (-1, 1, 1))-1
				weights_loc = np.reshape(weights[good[:, i+1]], (-1, 1, 1))
				#xi_loc *= weights_loc/(len_vec_loc*xi_loc_sum)

				xi_loc_sum[xi_loc_sum == 0] = 1.0

				xi_loc *= weights_loc / xi_loc_sum

				temp = xi_loc.sum(axis=2)
				temp_sum = temp.sum(axis=1, keepdims=True)
				temp_sum[temp_sum == 0] = 1.0
				temp /= temp_sum

				alphas[band1[0]+has_next_loc, :] = temp

				self.trans_counts += xi_loc.sum(axis=0)

			# Aggregate sufficient statistics
			for i in range(num_states):
				#alphas[:,i] *= weights[idx_vec]/np.maximum(len_vec[idx_vec], 1.0)
				alphas[:, i] *= weights[idx_vec]
				self.accumulators[i].seq_update(enc_data, alphas[:, i], estimate.topics[i])

			self.state_counts += alphas.sum(axis=0)

			band1 = idx_bands[0]
			temp = alphas[band1[0]:band1[1], :].sum(axis=1, keepdims=True)
			temp[temp == 0] = 1.0
			alphas[band1[0]:band1[1], :] *= np.reshape(weights[good[:,0]], (-1, 1))/temp

			self.init_counts += alphas[band1[0]:band1[1], :].sum(axis=0)

			if self.len_accumulator is not None:
				self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

		else:

			(idx, sz, enc_data), len_enc = x1

			tot_cnt = len(idx)
			seq_cnt = len(sz)
			num_states = estimate.nStates
			pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)


			max_len = sz.max()
			tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

			init_pvec = estimate.w
			tran_mat = estimate.transitions

			# Compute state likelihood vectors and scale the max to one
			for i in range(num_states):
				pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)

			pr_max = pr_obs.max(axis=1, keepdims=True)
			pr_obs -= pr_max
			np.exp(pr_obs, out=pr_obs)



			#alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
			#xi_acc = np.zeros((num_states, num_states), dtype=np.float64)
			#xi_buff = np.zeros((num_states, num_states), dtype=np.float64)
			#pi_acc = np.zeros(num_states, dtype=np.float64)
			#beta_buff = np.zeros(num_states, dtype=np.float64)
			#numba_baum_welch(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc, beta_buff, xi_buff)
			#self.init_counts  += pi_acc
			#self.trans_counts += xi_acc

			alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
			xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
			pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
			numba_baum_welch2(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)


#			self.init_counts  += pi_acc.sum(axis=0)
			self.init_counts = pi_acc
			self.trans_counts += xi_acc.sum(axis=0)


			#numba_baum_welch2.parallel_diagnostics(level=4)

			for i in range(num_states):
				self.accumulators[i].seq_update(enc_data, alphas[:, i], estimate.topics[i])

			self.state_counts += alphas.sum(axis=0)

			if self.len_accumulator is not None:
				self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)


	def combine(self, suff_stat):
		num_states, init_counts, state_counts, trans_counts, accumulators, len_acc = suff_stat

		self.init_counts  += init_counts
		self.state_counts += state_counts
		self.trans_counts += trans_counts

		for i in range(self.num_states):
			self.accumulators[i].combine(accumulators[i])

		if self.len_accumulator is not None and len_acc is not None:
			self.len_accumulator.combine(len_acc)

		return self

	def value(self):

		if self.len_accumulator is not None:
			len_val = self.len_accumulator.value()
		else:
			len_val = None

		return self.num_states, self.init_counts, self.state_counts, self.trans_counts, tuple([u.value() for u in self.accumulators]), len_val

	def from_value(self, x):
		num_states, init_counts, state_counts, trans_counts, accumulators, len_acc = x
		self.num_states = num_states
		self.init_counts = init_counts
		self.state_counts = state_counts
		self.trans_counts = trans_counts

		for i,v in enumerate(accumulators):
			self.accumulators[i].from_value(v)

		if self.len_accumulator is not None:
			self.len_accumulator.from_value(len_acc)

		return self

	def key_merge(self, stats_dict):

		if self.init_key is not None:
			if self.init_key in stats_dict:
				stats_dict[self.init_key] += self.init_counts
			else:
				stats_dict[self.init_key] = self.init_counts

		if self.trans_key is not None:
			if self.trans_key in stats_dict:
				stats_dict[self.trans_key] += self.trans_counts
			else:
				stats_dict[self.trans_key] = self.trans_counts

		if self.state_key is not None:
			if self.state_key in stats_dict:
				acc = stats_dict[self.state_key]
				for i in range(len(acc)):
					acc[i] = acc[i].combine(self.accumulators[i].value())
			else:
				stats_dict[self.state_key] = self.accumulators


		for u in self.accumulators:
			u.key_merge(stats_dict)

		if self.len_accumulator is not None:
			self.len_accumulator.key_merge(stats_dict)

	def key_replace(self, stats_dict):

		if self.init_key is not None:
			if self.init_key in stats_dict:
				self.init_counts = stats_dict[self.init_key]

		if self.trans_key is not None:
			if self.trans_key in stats_dict:
				self.trans_counts = stats_dict[self.trans_key]

		if self.state_key is not None:
			if self.state_key in stats_dict:
				self.accumulators = stats_dict[self.state_key]


		for u in self.accumulators:
			u.key_replace(stats_dict)

		if self.len_accumulator is not None:
			self.len_accumulator.key_replace(stats_dict)

class IndPiHiddenMarkovEstimatorAccumulatorFactory(object):

	def __init__(self, factories, len_factory, keys):
		self.factories = factories
		self.keys = keys
		self.len_factory = len_factory

	def make(self):
		len_acc = self.len_factory.make() if self.len_factory is not None else None
		return IndPiHiddenMarkovEstimatorAccumulator([self.factories[i].make() for i in range(len(self.factories))], len_accumulator=len_acc, keys=self.keys)



class IndPiHiddenMarkovEstimator(ParameterEstimator):

	def __init__(self, estimators, len_estimator=None, suff_stat=None, pseudo_count=(None,None), name=None, keys=(None, None, None), use_numba=True):

		self.num_states = len(estimators)
		self.estimators = estimators
		self.pseudo_count = pseudo_count
		self.suff_stat = suff_stat
		self.keys = keys
		self.len_estimator = len_estimator
		self.name = name

		self.use_numba = use_numba

	def accumulatorFactory(self):
		est_factories = [u.accumulatorFactory() for u in self.estimators]
		len_factory = self.len_estimator.accumulatorFactory() if self.len_estimator is not None else None
		return IndPiHiddenMarkovEstimatorAccumulatorFactory(est_factories, len_factory, self.keys)

	def estimate(self, nobs, suff_stat):

		num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = suff_stat


		if self.len_estimator is not None:
			len_dist = self.len_estimator.estimate(nobs, len_ss)
		else:
			len_dist = None

		topics = [self.estimators[i].estimate(state_counts[i], topic_ss[i]) for i in range(num_states)]

		w = np.zeros(np.shape(init_counts))


		for i in range(len(init_counts)):
			if self.pseudo_count[0] is not None:
				p1 = self.pseudo_count[0] / float(num_states)
				w[i] = init_counts[i] + p1
				if w[i].sum() > 0:
					w[i] /= w[i].sum()
			else:
				if init_counts[i].sum() > 0:
					w[i] = init_counts[i] / init_counts[i].sum()
				else:

#					w[i] = init_counts[i]

#					possibly incorrect to replace 0's with uniform
					w[i] = ([1.0/len(init_counts[i])]*len(init_counts[i]) )

			

		if self.pseudo_count[1] is not None:
			p2 = self.pseudo_count[1] / float(num_states*num_states)
			transitions = trans_counts + p2
			transitions /= transitions.sum(axis=1, keepdims=True)
		else:
			transitions = trans_counts / trans_counts.sum(axis=1, keepdims=True)

		return IndPiHiddenMarkovModelDistribution(topics, w, transitions, None, len_dist=len_dist, name=self.name, use_numba=self.use_numba)



@numba.njit('void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:])', parallel=True, fastmath=True, cache=True)
def numba_seq_log_density(num_states, tz, prob_mat, init_pvec, tran_mat, max_ll, next_alpha_mat, alpha_buff_mat, out):

	for n in numba.prange(len(tz) - 1):

		s0 = tz[n]
		s1 = tz[n+1]

		if s0 == s1:
			out[n] = 0
			continue

		next_alpha = next_alpha_mat[n,:]
		alpha_buff = alpha_buff_mat[n,:]

		llsum = 0
		alpha_sum = 0
		for i in range(num_states):
			temp = init_pvec[i] * prob_mat[s0, i]
			next_alpha[i] = temp
			alpha_sum += temp

		llsum += math.log(alpha_sum)
		llsum += max_ll[s0]

		for s in range(s0+1, s1):

			for i in range(num_states):
				alpha_buff[i] = next_alpha[i] / alpha_sum

			alpha_sum = 0
			for i in range(num_states):
				temp = 0.0
				for j in range(num_states):
					temp += tran_mat[j, i] * alpha_buff[j]
				temp *= prob_mat[s, i]
				next_alpha[i] = temp
				alpha_sum += temp

			llsum += math.log(alpha_sum)
			llsum += max_ll[s]

		out[n] = llsum



@numba.njit('void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:], float64[:,:])', cache=True)
def numba_baum_welch(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc, beta_buff, xi_buff):

	for n in range(len(tz)-1):

		s0 = tz[n]
		s1 = tz[n+1]

		if s0 == s1:
			continue

		weight_loc = weights[n]
		alpha_sum = 0
		for i in range(num_states):
			temp = init_pvec[i] * prob_mat[s0, i]
			alpha_loc[s0, i] = temp
			alpha_sum += temp
			#alpha_sum = temp if temp > alpha_sum else alpha_sum
		for i in range(num_states):
				alpha_loc[s0, i] /= alpha_sum

		for s in range(s0+1, s1):

			sm1 = s - 1
			alpha_sum = 0
			for i in range(num_states):
				temp = 0.0
				for j in range(num_states):
					temp += tran_mat[j, i] * alpha_loc[sm1, j]
				temp *= prob_mat[s, i]
				alpha_loc[s, i] = temp
				alpha_sum += temp
				#alpha_sum = temp if temp > alpha_sum else alpha_sum

			for i in range(num_states):
				alpha_loc[s, i] /= alpha_sum


		for i in range(num_states):
			alpha_loc[s1-1, i] *= weight_loc

		beta_sum = 1
		#beta_sum = 1/num_states
		prev_beta = np.empty(num_states, dtype=np.float64)
		prev_beta.fill(1/num_states)

		for s in range(s1 - 2, s0 - 1 , -1):

			sp1 = s + 1

			for j in range(num_states):
				beta_buff[j] = prev_beta[j] * prob_mat[sp1, j] / beta_sum

			xi_buff_sum = 0
			gamma_buff = 0
			beta_sum = 0
			for i in range(num_states):

				temp_beta = 0
				for j in range(num_states):
					temp = tran_mat[i, j] * beta_buff[j]
					temp_beta += temp
					temp *= alpha_loc[s,i]
					xi_buff[i, j] = temp
					xi_buff_sum += temp

				prev_beta[i] = temp_beta
				alpha_loc[s, i] *= temp_beta
				gamma_buff += alpha_loc[s, i]
				beta_sum += temp_beta
				#beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

			if gamma_buff > 0:
				gamma_buff = weight_loc / gamma_buff

			if xi_buff_sum > 0:
				xi_buff_sum = weight_loc / xi_buff_sum

			for i in range(num_states):
				alpha_loc[s, i] *= gamma_buff
				for j in range(num_states):
					xi_acc[i, j] += xi_buff[i,j] * xi_buff_sum

		for i in range(num_states):
			pi_acc[i] += alpha_loc[s0,i]



#@numba.njit('void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], float64[:,:])', parallel=True, fastmath=True)
@numba.njit('void(int64, int32[:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], float64[:,:])', parallel=True, fastmath=True, cache=True)
def numba_baum_welch2(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc):


	for n in numba.prange(len(tz)-1):

		s0 = tz[n]
		s1 = tz[n+1]


		if s0 == s1:
			continue

		beta_buff = np.zeros(num_states, dtype=np.float64)
		xi_buff = np.zeros((num_states,num_states), dtype=np.float64)

		weight_loc = weights[n]
		alpha_sum = 0
		for i in range(num_states):
#			temp = init_pvec[i] * prob_mat[s0, i]
			temp = init_pvec[n,i] * prob_mat[s0, i]

			alpha_loc[s0, i] = temp
			alpha_sum += temp
			#alpha_sum = temp if temp > alpha_sum else alpha_sum
		for i in range(num_states):
				if alpha_sum != 0.0:
					alpha_loc[s0, i] /= alpha_sum
				else:
#may not be correct to force uniform
					alpha_loc[s0,i] = 1.0/num_states
		for s in range(s0+1, s1):

			sm1 = s - 1
			alpha_sum = 0
			for i in range(num_states):
				temp = 0.0
				for j in range(num_states):
					temp += tran_mat[j, i] * alpha_loc[sm1, j]
				temp *= prob_mat[s, i]
				alpha_loc[s, i] = temp
				alpha_sum += temp
				#alpha_sum = temp if temp > alpha_sum else alpha_sum

			for i in range(num_states):
				alpha_loc[s, i] /= alpha_sum


		for i in range(num_states):
			alpha_loc[s1-1, i] *= weight_loc

		beta_sum = 1
		#beta_sum = 1/num_states
		prev_beta = np.empty(num_states, dtype=np.float64)
		prev_beta.fill(1/num_states)

		for s in range(s1 - 2, s0 - 1 , -1):

			sp1 = s + 1

			for j in range(num_states):
				beta_buff[j] = prev_beta[j] * prob_mat[sp1, j] / beta_sum

			xi_buff_sum = 0
			gamma_buff = 0
			beta_sum = 0
			for i in range(num_states):

				temp_beta = 0
				for j in range(num_states):
					temp = tran_mat[i, j] * beta_buff[j]
					temp_beta += temp
					temp *= alpha_loc[s,i]
					xi_buff[i, j] = temp
					xi_buff_sum += temp

				prev_beta[i] = temp_beta
				alpha_loc[s, i] *= temp_beta
				gamma_buff += alpha_loc[s, i]
				beta_sum += temp_beta
				#beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

			if gamma_buff > 0:
				gamma_buff = weight_loc / gamma_buff

			if xi_buff_sum > 0:
				xi_buff_sum = weight_loc / xi_buff_sum

			for i in range(num_states):
				alpha_loc[s, i] *= gamma_buff
				for j in range(num_states):
					xi_acc[n, i, j] += xi_buff[i,j] * xi_buff_sum

		for i in range(num_states):
#			if not np.isnan(alpha_loc[s0,i]):
			pi_acc[n,i] += alpha_loc[s0,i]

