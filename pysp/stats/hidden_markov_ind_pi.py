"""Create, estimate, and sample from a hidden Markov model with independent (per-sequence) initial state vectors.

Defines the IndPiHiddenMarkovModelDistribution, IndPiHiddenMarkovSampler, IndPiHiddenMarkovEstimatorAccumulator,
IndPiHiddenMarkovEstimatorAccumulatorFactory, IndPiHiddenMarkovEstimator, and the IndPiHiddenMarkovDataEncoder
classes for use with pysparkplug.

Data type: List[T] / Sequence[T] (determined by the emission distributions).

This is a variant of pysp.stats.hidden_markov where the initial state probabilities are NOT shared across
observation sequences. Instead, each observed sequence i carries its own initial state probability vector w[i],
so 'w' is a 2-d array with one row per sequence (rows sum to 1.0). The K-by-K transition matrix and the K
emission distributions are shared across all sequences:

	(1) Emissions:        P(X_t = x_t | Z_t = k), for k = {1,...,K} (all emission distributions of data type T).
	(2) Transitions:      p_mat(Z_t = i | Z_{t-1} = j), a K-by-K row-stochastic matrix.
	(3) Initial states:   p_mat(Z_1 = k | sequence i) = w[i, k], independently for each sequence i.

If included, the length of the sequences is modeled through P_len(T), a distribution with support on the
non-negative integers.

Note: Scalar 'log_density' evaluations average the per-sequence initial state vectors (logW is formed from the
column means of 'w'), while vectorized 'seq_' calls track per-sequence initial state vectors.

"""
from numpy.random import RandomState
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
	ParameterEstimator, DataSequenceEncoder, DistributionSampler, StatisticAccumulatorFactory
from pysp.stats.markovchain import MarkovChainDistribution
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.null_dist import NullDataEncoder
from pysp.utils.optional_deps import numba
from pysp.utils.aliasing import coalesce_alias, require, MISSING
import numpy as np
import math

class IndPiHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
	"""HMM with shared emissions/transitions and an independent initial state vector per observed sequence.

	Compatible with data type List[T], where T is the data type of the emission distributions.
	"""

	def compute_capabilities(self):
		from pysp.stats.capabilities import DistributionCapabilities, intersect_engine_ready
		if self.use_numba or self.has_topics or self.terminal_values is not None:
			return DistributionCapabilities(engine_ready=('numpy',), kernel_status='legacy_numpy')
		children = tuple(self.topics)
		if self.len_dist is not None:
			children = children + (self.len_dist,)
		return DistributionCapabilities(engine_ready=intersect_engine_ready(children),
		                                kernel_status='generic_latent')

	def compute_declaration(self):
		from pysp.stats.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec, declaration_for
		topic_children = tuple(declaration_for(topic) for topic in self.topics)
		length = None if self.len_dist is None else declaration_for(self.len_dist)
		children = tuple(child for child in topic_children + ((length,) if length is not None else ())
		                 if child is not None)
		roles = tuple('state_%d_emission' % i for i, child in enumerate(topic_children) if child is not None)
		if length is not None:
			roles += ('length',)
		return DistributionDeclaration(
			name='ind_pi_hidden_markov',
			distribution_type=type(self),
			parameters=(
				ParameterSpec('w', constraint='row_simplex_matrix'),
				ParameterSpec('transitions', constraint='row_simplex_matrix'),
				ParameterSpec('taus', constraint='row_simplex_matrix', differentiable=False),
			),
			statistics=(
				StatisticSpec('num_states', kind='metadata', additive=False, scales=False),
				StatisticSpec('initial_counts'),
				StatisticSpec('state_counts'),
				StatisticSpec('transition_counts'),
				StatisticSpec('emissions', kind='tuple'),
				StatisticSpec('length', kind='child_stat'),
			),
			support='independent_initial_hidden_state_sequence',
			children=children,
			child_roles=roles,
			differentiable=False,
		)

	def __init__(self, topics, w=MISSING, transitions=MISSING, taus=MISSING, len_dist=None, name=None, terminal_values=None, use_numba=True, weights=MISSING):
		"""IndPiHiddenMarkovModelDistribution object defining an HMM with per-sequence initial state vectors.

		Args:
			topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions all having type T.
			w (Union[List[List[float]], np.ndarray]): 2-d array of initial state probabilities with one row per
				observation sequence. Rows sum to 1.0.
			transitions (Union[List[List[float]], np.ndarray]): 2-d array of hidden state transition probabilities.
			taus (Optional[Union[List[List[float]], np.ndarray]]): If passed, emission distributions are treated as
				mixtures over 'topics' with state-dependent weights 'taus'.
			len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for sequence
				lengths with support on the non-negative integers.
			name (Optional[str]): Set name to object instance.
			terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.
			use_numba (bool): If True, use numba package for encoding and vectorized operations.

		Attributes:
			topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions all having type T.
			nTopics (int): Number of emission distributions.
			nStates (int): Number of hidden states.
			w (np.ndarray): 2-d array of per-sequence initial state probabilities.
			logW (np.ndarray): Log of the column means of 'w' (averaged initial state vector).
			transitions (np.ndarray): 2-d array of hidden state transition probabilities (nStates by nStates).
			logTransitions (np.ndarray): Log of above.
			taus (Optional[np.ndarray]): State-dependent topic mixture weights, or None.
			logTaus (Optional[np.ndarray]): Log of 'taus' if passed.
			has_topics (bool): True if taus is passed.
			len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional length distribution.
			name (Optional[str]): Name of object instance.
			terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.
			use_numba (bool): If True, use numba package for encoding and vectorized operations.

		"""
		w = coalesce_alias('w', w, 'weights', weights, default=MISSING)
		transitions = require('transitions', transitions, default=MISSING)
		taus = require('taus', taus, default=MISSING)
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
		"""Returns string representation of IndPiHiddenMarkovModelDistribution instance."""
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
		"""Returns the density of the HMM for an observed sequence x.

		See 'IndPiHiddenMarkovModelDistribution.log_density()' for details.

		Args:
			x (List[T]): Observed sequence of HMM emissions.

		Returns:
			Density of HMM for observed sequence x.

		"""
		return exp(self.log_density(x))

	def log_density(self, x):
		"""Returns the log-density of the HMM for an observed sequence x.

		The log-density is evaluated with a forward (alpha) pass using the averaged initial state vector
		'logW' (column means of 'w'), since a single sequence carries no information about which
		per-sequence initial state vector applies. If 'len_dist' is set, the log-density of the sequence
		length is added.

		Args:
			x (List[T]): Observed sequence of HMM emissions.

		Returns:
			Log-density of observed HMM sequence x.

		"""
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
			x0      = next(xIter)

			obsLogDensityByTopic  = [u.log_density(x0) for u in self.topics]
			logLikelihoodByState  = [logW[i] + vec.weighted_log_sum(obsLogDensityByTopic, logTaus[i,:]) for i in range(nStates)]

			for x in xIter:
				obsLogDensityByTopic = [u.log_density(x) for u in self.topics]
				logLikelihoodByState = [vec.weighted_log_sum(obsLogDensityByTopic, logTaus[:, i]) + vec.weighted_log_sum(obsLogDensityByTopic, logTaus[i, :]) for i in range(nStates)]

			rv = vec.log_sum(logLikelihoodByState)
			if self.len_dist is not None:
				rv += self.len_dist.log_density(len(x))

			return rv

	def seq_log_density(self, x):
		"""Vectorized log-density evaluation for an encoded sequence of iid HMM observations.

		Arg 'x' is the output of 'IndPiHiddenMarkovDataEncoder.seq_encode()'; either Tuple[enc, None] for
		the numpy encoding, or Tuple[None, enc_numba] for the numba encoding. The numba path averages
		the per-sequence initial state vectors of 'w' before the forward pass.

		Args:
			x: Encoded sequence of iid HMM observations (see IndPiHiddenMarkovDataEncoder.seq_encode()).

		Returns:
			Numpy array of log-density values, one per encoded sequence.

		"""
		x0, x1 = x
		if x1 is None:

			num_states = self.nStates
			(tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x0
			w = self.w
			A = self.transitions

			max_len = len(idx_bands)
			num_seq = idx_mat.shape[0]
			if w.shape[0] != num_seq:
				w_sum = np.sum(w, axis=0) / float(len(w))
				w = np.repeat(w_sum.reshape(1, -1), num_seq, axis=0)

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
			first_rows = np.flatnonzero(good[:, 0])
			alphas_prev = np.multiply(pr_obs[band[0]:band[1], :], w[first_rows, :])
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

	def backend_seq_log_density(self, x, engine):
		"""Engine-neutral scoring for the non-numba independent-initial-probability HMM layout."""
		from pysp.stats.backend import BackendScoringError, backend_seq_log_density

		x0, x1 = x
		if self.has_topics:
			if getattr(engine, 'name', None) == 'numpy':
				return self.seq_log_density(x)
			raise BackendScoringError('IndPi HMM backend scoring does not support topic-mixture emissions.')
		if self.terminal_values is not None:
			if getattr(engine, 'name', None) == 'numpy':
				return self.seq_log_density(x)
			raise BackendScoringError('IndPi HMM backend scoring does not support terminal-value semantics.')
		if x1 is not None:
			if getattr(engine, 'name', None) == 'numpy':
				return self.seq_log_density(x)
			raise BackendScoringError('IndPi HMM backend scoring requires the non-numba encoding.')
		if x0 is None:
			raise BackendScoringError('IndPi HMM backend scoring received an empty encoded layout.')

		(tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x0
		num_states = self.nStates
		max_len = len(idx_bands)
		num_seq = idx_mat.shape[0]
		w = self.w
		if w.shape[0] != num_seq:
			w_sum = np.sum(w, axis=0) / float(len(w))
			w = np.repeat(w_sum.reshape(1, -1), num_seq, axis=0)
		ll_ret = engine.zeros(num_seq)

		if max_len > 0 and tot_cnt > 0:
			good = idx_mat >= 0
			emission_scores = [backend_seq_log_density(topic, enc_data, engine) for topic in self.topics]
			log_pr_obs = engine.stack(emission_scores, axis=1)
			pr_max0 = engine.max(log_pr_obs, axis=1)
			pr_obs = engine.exp(log_pr_obs - pr_max0[:, None])

			band = idx_bands[0]
			row_idx = np.flatnonzero(good[:, 0])
			alphas_prev = pr_obs[band[0]:band[1], :] * engine.asarray(w[row_idx, :])
			temp = engine.sum(alphas_prev, axis=1, keepdims=True)
			alphas_prev = alphas_prev / temp
			ll_ret[engine.asarray(row_idx)] = engine.log(temp[:, 0]) + pr_max0[band[0]:band[1]]

			a_mat = engine.asarray(self.transitions)
			for i in range(1, max_len):
				band = idx_bands[i]
				row_idx = np.flatnonzero(good[:, i])
				alphas_next = engine.matmul(alphas_prev[engine.asarray(has_next[i-1]), :], a_mat)
				alphas_next = alphas_next * pr_obs[band[0]:band[1], :]
				pr_sum = engine.sum(alphas_next, axis=1, keepdims=True)
				alphas_prev = alphas_next / pr_sum
				ll_ret[engine.asarray(row_idx)] = ll_ret[engine.asarray(row_idx)] + \
					engine.log(pr_sum[:, 0]) + pr_max0[band[0]:band[1]]

		if self.len_dist is not None:
			ll_ret = ll_ret + backend_seq_log_density(self.len_dist, len_enc, engine)

		return ll_ret

	def _seq_encode(self, x):
		"""Deprecated: encode x with the numpy (non-numba) encoding.

		Use 'dist_to_encoder()' and 'IndPiHiddenMarkovDataEncoder.seq_encode()' instead.

		Args:
			x (List[List[T]]): A sequence of iid observations from the HMM.

		Returns:
			Encoded sequence (see IndPiHiddenMarkovDataEncoder._seq_encode()).

		"""
		encoder = self.dist_to_encoder()
		encoder.use_numba = False
		return encoder.seq_encode(x)


	def seq_encode(self, x):
		"""Deprecated: encode a sequence of iid HMM observations for vectorized 'seq_' calls.

		Use 'dist_to_encoder()' and 'IndPiHiddenMarkovDataEncoder.seq_encode()' instead.

		Args:
			x (List[List[T]]): A sequence of iid observations from the HMM.

		Returns:
			Encoded sequence (see IndPiHiddenMarkovDataEncoder.seq_encode()).

		"""
		return self.dist_to_encoder().seq_encode(x)

	def sampler(self, seed=None):
		"""Create an IndPiHiddenMarkovSampler object with seed passed.

		Args:
			seed (Optional[int]): Set seed for random sampling.

		Returns:
			IndPiHiddenMarkovSampler object.

		"""
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
		"""Create an IndPiHiddenMarkovEstimator for estimating models like this object instance.

		Args:
			pseudo_count (Optional[float]): Used to re-weight sufficient statistics in estimation.

		Returns:
			IndPiHiddenMarkovEstimator object.

		"""
		len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
		comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
		return IndPiHiddenMarkovEstimator(comp_ests, pseudo_count=(pseudo_count,pseudo_count), len_estimator=len_est)

	def dist_to_encoder(self):
		"""Returns IndPiHiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
		emission_encoder = self.topics[0].dist_to_encoder()
		len_encoder = self.len_dist.dist_to_encoder() if self.len_dist is not None else NullDataEncoder()
		return IndPiHiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder,
		                                    use_numba=self.use_numba)

class IndPiHiddenMarkovSampler(DistributionSampler):
	"""IndPiHiddenMarkovSampler object for sampling iid sequences from an IndPiHiddenMarkovModelDistribution.

	Cycles through one Markov chain state sampler per row of 'dist.w' so consecutive samples are drawn with
	the per-sequence initial state vectors of the model.
	"""

	def __init__(self, dist, seed):
		"""IndPiHiddenMarkovSampler object.

		Args:
			dist (IndPiHiddenMarkovModelDistribution): Distribution instance to sample from.
			seed (Optional[int]): Set seed on random number generator for sampling.

		Attributes:
			num_states (int): Number of hidden states in 'dist' object.
			dist (IndPiHiddenMarkovModelDistribution): Distribution instance to sample from.
			rng (RandomState): RandomState object with seed set for sampling.
			iter (int): Index of the next per-sequence state sampler to use (cycles through 'stateSamplers').
			obsSamplers (List[DistributionSampler]): Samplers for the emission distributions. Taken to be
				MixtureSampler objects if 'dist.has_topics' is True.
			len_sampler (Optional[DistributionSampler]): Sampler for sequence lengths if 'dist.len_dist' is set.
			terminal_set (Optional[Set[T]]): Set of values that terminate sampling, if set on 'dist'.
			stateSamplers (List[DistributionSampler]): One MarkovChainSampler per row of 'dist.w'.

		"""
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
		"""Sample iid HMM sequences with lengths drawn from the length sampler.

		If size is None, 1 sample is drawn and a List[T] is returned. If size > 0, 'size' samples are drawn
		and a List of length 'size' with HMM sequences (List[T]) is returned. Each draw cycles to the next
		per-sequence state sampler.

		Args:
			size (Optional[int]): Number of iid HMM sequences to sample.

		Returns:
			List[T] or List[List[T]] depending on size arg.

		"""

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
		"""Sample an HMM sequence until a terminal value is sampled from the emission distribution.

		Args:
			terminal_set (Set[T]): Set of values that terminate the HMM sequence.

		Returns:
			List[T] with length determined by samples to reach the first terminating value.

		"""
		z = self.stateSamplers[self.iter].sample_seq()
		rv = [self.obsSamplers[z].sample()]

		self.iter += 1
		if self.iter >= len(self.stateSamplers):
			self.iter = 0

		while rv[-1] not in terminal_set:
			z = self.stateSamplers[self.iter].sample_seq(v0=z)

			self.iter += 1
			if self.iter >= len(self.stateSamplers):
				self.iter = 0

			rv.append(self.obsSamplers[z].sample())

		return rv



	def sample(self, size=None):
		"""Draw iid samples from the HMM.

		If a 'len_sampler' is set, 'sample_seq()' is called. Otherwise, if 'terminal_set' is set,
		'sample_terminal()' is called.

		Args:
			size (Optional[int]): Number of iid HMM sequences to sample.

		Returns:
			List[T] or List[List[T]] depending on arg size.

		"""

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
	"""IndPiHiddenMarkovEstimatorAccumulator object for aggregating sufficient statistics from HMM observations.

	The initial state counts ('init_counts') are tracked per sequence (a 2-d array with one row per observed
	sequence), unlike pysp.stats.hidden_markov where they are pooled.
	"""

	def __init__(self, accumulators, len_accumulator=None,  keys=(None, None, None), init_counts=None, use_numba=True):
		"""IndPiHiddenMarkovEstimatorAccumulator object.

		Args:
			accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulator objects for the
				emission distributions.
			len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator object for
				the length distribution.
			keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for initial state counts,
				transition counts, and emission accumulators.
			init_counts (Optional[np.ndarray]): Optional 2-d array of per-sequence initial state counts.
			use_numba (bool): True if sequence encodings are for use with numba.

		Attributes:
			accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulator objects for the
				emission distributions.
			num_states (int): Total number of hidden states.
			init_counts (np.ndarray): Per-sequence initial state counts (one row per observed sequence).
			init_counts_initialized (bool): True if 'init_counts' was passed to the constructor.
			trans_counts (np.ndarray): 2-d matrix tracking transition updates from Baum-Welch.
			state_counts (np.ndarray): Expected number of times each state is observed.
			len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator object for
				the length distribution.
			use_numba (bool): True if sequence encodings are for use with numba.
			init_key (Optional[str]): Key for initial state counts.
			trans_key (Optional[str]): Key for state transitions.
			state_key (Optional[str]): Key for emission accumulators.

			_init_rng (bool): True if RandomState objects have been initialized for seq_initialize.
			_len_rng (Optional[RandomState]): RandomState for initializing length accumulator.
			_acc_rng (Optional[List[RandomState]]): RandomState objects for initializing emission accumulators.
			_idx_rng (Optional[RandomState]): RandomState for initializing state draws.

		"""
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

		self.use_numba = use_numba

		# protected for seq_initialize consistency.
		self._init_rng = False
		self._len_rng = None
		self._acc_rng = None
		self._idx_rng = None

	def update(self, x, weight, estimate):
		"""Update sufficient statistics of the accumulator with one observation sequence.

		Note: Not efficient. Encodes a singleton batch and delegates to 'seq_update()'.

		Args:
			x (List[T]): HMM observation sequence.
			weight (float): Weight for observation.
			estimate (IndPiHiddenMarkovModelDistribution): Previous estimate of the HMM.

		Returns:
			None.

		"""
		enc_x = estimate.dist_to_encoder().seq_encode([x])
		self.seq_update(enc_x, np.asarray([weight]), estimate)

	def _rng_initialize(self, rng):
		"""Set RandomState member variables used by seq_initialize.

		Args:
			rng (RandomState): RandomState object used to seed member RandomState objects.

		Returns:
			None.

		"""
		rng_seeds = rng.randint(maxrandint, size=2+self.num_states)
		self._idx_rng = RandomState(seed=rng_seeds[0])
		self._len_rng = RandomState(seed=rng_seeds[1])
		self._acc_rng = [RandomState(seed=rng_seeds[2+i]) for i in range(self.num_states)]
		self._init_rng = True

	def initialize(self, x, weight, rng):
		"""Initialize the accumulator with a single HMM observation sequence.

		Appends a row of 'weight' to the per-sequence initial state counts and assigns each emission to a
		random hidden state.

		Args:
			x (List[T]): HMM observation sequence.
			weight (float): Weight for observation.
			rng (RandomState): RandomState for random state assignments.

		Returns:
			None.

		"""

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

	def seq_initialize(self, x, weights, rng):
		"""Vectorized initialization of the accumulator from an encoded sequence of iid HMM observations.

		Mirrors 'initialize()': appends one row of weight to the per-sequence initial state counts for every
		non-empty sequence in the batch, assigns each emission to a random hidden state, and initializes the
		emission accumulators accordingly.

		Args:
			x: Encoded sequence of iid HMM observations (see IndPiHiddenMarkovDataEncoder.seq_encode()).
			weights (np.ndarray): Numpy array of weights for the observations.
			rng (RandomState): Used to seed member RandomState objects on first call.

		Returns:
			None.

		"""
		x0, x1 = x

		if not self._init_rng:
			self._rng_initialize(rng)

		if x1 is None:
			(tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x0

			if self.len_accumulator is not None:
				self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

			states = self._idx_rng.choice(self.num_states, size=tot_cnt)

			non_zero = len_vec > 0
			w_nz = weights[non_zero]

			if not self.init_counts_initialized:
				new_rows = np.zeros((len(w_nz), self.num_states))
				new_rows += np.reshape(w_nz, (-1, 1))
				self.init_counts = np.concatenate([np.reshape(self.init_counts, (-1, self.num_states)), new_rows], axis=0)

			self.state_counts += np.bincount(states, weights[idx_vec], minlength=self.num_states)

			max_len = len(idx_bands)
			for i in range(max_len - 1):
				both = np.bitwise_and(idx_mat[:, i] >= 0, idx_mat[:, i+1] >= 0)
				prev_state = states[idx_mat[both, i].astype(int)]
				next_state = states[idx_mat[both, i+1].astype(int)]
				temp = np.bincount(prev_state*self.num_states + next_state, weights[both], minlength=self.num_states**2)
				self.trans_counts += np.reshape(temp, (self.num_states, self.num_states))

			for j in range(self.num_states):
				w = weights[idx_vec].copy()
				w[states != j] = 0.0
				self.accumulators[j].seq_initialize(enc_data, w, self._acc_rng[j])

		else:
			(idx, sz, enc_data), len_enc = x1

			if self.len_accumulator is not None:
				self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

			tot_cnt = int(np.sum(sz))
			states = self._idx_rng.choice(self.num_states, size=tot_cnt)
			nz_idx, nz_idx_group = np.unique(idx, return_index=True)
			w_nz = weights[nz_idx]

			if not self.init_counts_initialized:
				new_rows = np.zeros((len(w_nz), self.num_states))
				new_rows += np.reshape(w_nz, (-1, 1))
				self.init_counts = np.concatenate([np.reshape(self.init_counts, (-1, self.num_states)), new_rows], axis=0)

			self.state_counts += np.bincount(states, weights[idx], minlength=self.num_states)

			sz_next = sz[nz_idx] - 1
			steps = np.zeros(len(sz_next), dtype=int)
			cond = steps < sz_next

			while np.any(cond):
				prev_state = states[nz_idx_group[cond] + steps[cond]]
				next_state = states[nz_idx_group[cond] + steps[cond] + 1]
				temp = np.bincount(prev_state*self.num_states + next_state, w_nz[cond], minlength=self.num_states**2)
				self.trans_counts += np.reshape(temp, (self.num_states, self.num_states))

				steps[cond] += 1
				cond = steps < sz_next

			for j in range(self.num_states):
				w = weights[idx].copy()
				w[states != j] = 0.0
				self.accumulators[j].seq_initialize(enc_data, w, self._acc_rng[j])

	def seq_update(self, x, weights, estimate):
		"""Vectorized Baum-Welch update of the accumulator from an encoded sequence of iid HMM observations.

		Note: For the numba encoding, the per-sequence initial state counts are replaced by the per-sequence
		posterior initial state probabilities (one row per sequence in the batch).

		Args:
			x: Encoded sequence of iid HMM observations (see IndPiHiddenMarkovDataEncoder.seq_encode()).
			weights (np.ndarray): Numpy array of weights for the observations.
			estimate (IndPiHiddenMarkovModelDistribution): Previous EM estimate of the HMM model.

		Returns:
			None.

		"""

		x0, x1 = x

		if x1 is None:

			num_states = self.num_states
			(tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc = x0
			w = estimate.w
			A = estimate.transitions

			max_len = len(idx_bands)
			num_seq = idx_mat.shape[0]
			if w.shape[0] != num_seq:
				w_sum = np.sum(w, axis=0) / float(len(w))
				w = np.repeat(w_sum.reshape(1, -1), num_seq, axis=0)

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
			if init_pvec.shape[0] != seq_cnt:
				w_sum = np.sum(init_pvec, axis=0) / float(len(init_pvec))
				init_pvec = np.repeat(w_sum.reshape(1, -1), seq_cnt, axis=0)

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
		"""Combine the sufficient statistics of the accumulator with the suff_stat arg.

		Sufficient statistics in suff_stat are a Tuple containing:
			suff_stat[0] (int): Number of hidden states.
			suff_stat[1] (np.ndarray): Per-sequence initial state counts.
			suff_stat[2] (np.ndarray): State counts.
			suff_stat[3] (np.ndarray): State transition counts.
			suff_stat[4] (Sequence): Emission distribution accumulator values.
			suff_stat[5] (Optional): Optional sufficient statistics of the length distribution.

		Args:
			suff_stat: See above for details.

		Returns:
			IndPiHiddenMarkovEstimatorAccumulator object.

		"""
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
		"""Returns sufficient statistics of the accumulator instance.

		Returned value rv is a Tuple containing:
			rv[0] (int): Number of hidden states.
			rv[1] (np.ndarray): Per-sequence initial state counts.
			rv[2] (np.ndarray): State counts.
			rv[3] (np.ndarray): State transition counts.
			rv[4] (Tuple): Emission distribution accumulator sufficient statistics.
			rv[5] (Optional): Optional sufficient statistics of the length distribution.

		Returns:
			See above for details.

		"""

		if self.len_accumulator is not None:
			len_val = self.len_accumulator.value()
		else:
			len_val = None

		return self.num_states, self.init_counts, self.state_counts, self.trans_counts, tuple([u.value() for u in self.accumulators]), len_val

	def from_value(self, x):
		"""Set the sufficient statistics of the accumulator instance to the value x.

		Arg x is a Tuple containing:
			x[0] (int): Number of hidden states.
			x[1] (np.ndarray): Per-sequence initial state counts.
			x[2] (np.ndarray): State counts.
			x[3] (np.ndarray): State transition counts.
			x[4] (Sequence): Emission distribution accumulator values.
			x[5] (Optional): Optional sufficient statistics of the length distribution.

		Args:
			x: See above for details.

		Returns:
			IndPiHiddenMarkovEstimatorAccumulator object.

		"""
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

	def scale(self, c):
		self.init_counts *= c
		self.state_counts *= c
		self.trans_counts *= c
		for acc in self.accumulators:
			acc.scale(c)
		if self.len_accumulator is not None:
			self.len_accumulator.scale(c)
		return self

	def key_merge(self, stats_dict):
		"""Merge the sufficient statistics of the accumulator instance into stats_dict for matching keys.

		Args:
			stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

		Returns:
			None.

		"""

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
		"""Replace the sufficient statistics of the accumulator instance with values in stats_dict for matching keys.

		Args:
			stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

		Returns:
			None.

		"""

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

	def acc_to_encoder(self):
		"""Returns IndPiHiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
		emission_encoder = self.accumulators[0].acc_to_encoder()
		len_encoder = self.len_accumulator.acc_to_encoder() if self.len_accumulator is not None else NullDataEncoder()
		return IndPiHiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder,
		                                    use_numba=self.use_numba)

class IndPiHiddenMarkovEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
	"""IndPiHiddenMarkovEstimatorAccumulatorFactory object for creating IndPiHiddenMarkovEstimatorAccumulator objects."""

	def __init__(self, factories, len_factory, keys, use_numba=True):
		"""IndPiHiddenMarkovEstimatorAccumulatorFactory object.

		Args:
			factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory objects for the
				emission distributions.
			len_factory (Optional[StatisticAccumulatorFactory]): Optional StatisticAccumulatorFactory for the
				length distribution.
			keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for initial state counts,
				transition counts, and emission accumulators.
			use_numba (bool): True if sequence encodings are for use with numba.

		"""
		self.factories = factories
		self.keys = keys
		self.len_factory = len_factory
		self.use_numba = use_numba

	def make(self):
		"""Returns an IndPiHiddenMarkovEstimatorAccumulator object."""
		len_acc = self.len_factory.make() if self.len_factory is not None else None
		return IndPiHiddenMarkovEstimatorAccumulator([self.factories[i].make() for i in range(len(self.factories))], len_accumulator=len_acc, keys=self.keys, use_numba=self.use_numba)



class IndPiHiddenMarkovEstimator(ParameterEstimator):
	"""IndPiHiddenMarkovEstimator object for estimating IndPiHiddenMarkovModelDistribution objects from
	aggregated sufficient statistics.
	"""

	def __init__(self, estimators, len_estimator=None, suff_stat=None, pseudo_count=(None,None), name=None, keys=(None, None, None), use_numba=True):
		"""IndPiHiddenMarkovEstimator object.

		Args:
			estimators (List[ParameterEstimator]): ParameterEstimator objects for the emission distributions.
			len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object for the length
				distribution.
			suff_stat (Optional[Any]): Kept for consistency with ParameterEstimator.
			pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo counts for initial state counts and
				state transitions.
			name (Optional[str]): Set name to object.
			keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for initial state counts,
				transition counts, and emission accumulators.
			use_numba (bool): If True, numba is used for sequence encoding and vectorized functions.

		Attributes:
			num_states (int): Number of hidden states.
			estimators (List[ParameterEstimator]): ParameterEstimator objects for the emission distributions.
			pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo counts for initial state counts and
				state transitions.
			suff_stat (Optional[Any]): Kept for consistency with ParameterEstimator.
			keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial state counts,
				transition counts, and emission accumulators.
			len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object for the length
				distribution.
			name (Optional[str]): Name of object instance.
			use_numba (bool): If True, numba is used for sequence encoding and vectorized functions.

		"""

		self.num_states = len(estimators)
		self.estimators = estimators
		self.pseudo_count = pseudo_count
		self.suff_stat = suff_stat
		self.keys = keys
		self.len_estimator = len_estimator
		self.name = name

		self.use_numba = use_numba

	def accumulator_factory(self):
		"""Returns an IndPiHiddenMarkovEstimatorAccumulatorFactory object."""
		est_factories = [u.accumulator_factory() for u in self.estimators]
		len_factory = self.len_estimator.accumulator_factory() if self.len_estimator is not None else None
		return IndPiHiddenMarkovEstimatorAccumulatorFactory(est_factories, len_factory, self.keys, use_numba=self.use_numba)

	def accumulatorFactory(self):
		"""Deprecated alias for accumulator_factory()."""
		return self.accumulator_factory()

	def estimate(self, nobs, suff_stat):
		"""Estimate an IndPiHiddenMarkovModelDistribution from aggregated sufficient statistics 'suff_stat'.

		Sufficient statistics in arg 'suff_stat' are a Tuple containing:
			suff_stat[0] (int): Number of hidden states.
			suff_stat[1] (np.ndarray): Per-sequence initial state counts (one row per observed sequence).
			suff_stat[2] (np.ndarray): State counts.
			suff_stat[3] (np.ndarray): State transition counts.
			suff_stat[4] (Sequence): Sufficient statistics for the emission distribution accumulators.
			suff_stat[5] (Optional): Optional sufficient statistics of the length distribution.

		Each row of the per-sequence initial state counts is normalized into a row of 'w'. Rows with zero
		counts are set to the uniform distribution.

		Args:
			nobs (Optional[float]): Number of observations used in estimation.
			suff_stat: See above for details.

		Returns:
			IndPiHiddenMarkovModelDistribution object.

		"""

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


class IndPiHiddenMarkovDataEncoder(DataSequenceEncoder):
	"""IndPiHiddenMarkovDataEncoder object for encoding sequences of iid HMM observations (List[List[T]])."""

	def __init__(self, emission_encoder, len_encoder=None, use_numba=True):
		"""IndPiHiddenMarkovDataEncoder object.

		Args:
			emission_encoder (DataSequenceEncoder): DataSequenceEncoder of type T for the observed emission
				values.
			len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder object for the sequence
				lengths. Set to NullDataEncoder if None is passed.
			use_numba (bool): If True, sequence encode for numba.

		Attributes:
			emission_encoder (DataSequenceEncoder): DataSequenceEncoder of type T for the observed emission
				values.
			len_encoder (DataSequenceEncoder): DataSequenceEncoder object for the sequence lengths.
			use_numba (bool): If True, sequence encode for numba.

		"""
		self.emission_encoder = emission_encoder
		self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
		self.use_numba = use_numba

	def __str__(self):
		"""Returns string representation of IndPiHiddenMarkovDataEncoder object instance."""
		s = 'IndPiHiddenMarkovDataEncoder(emission_encoder=' + str(self.emission_encoder) + ','
		s += 'len_encoder=' + str(self.len_encoder) + ','
		s += 'use_numba=' + str(self.use_numba) + ')'
		return s

	def __eq__(self, other):
		"""Check if other is equivalent to IndPiHiddenMarkovDataEncoder object instance.

		Args:
			other (object): Object to compare to IndPiHiddenMarkovDataEncoder object instance.

		Returns:
			True if other is an IndPiHiddenMarkovDataEncoder with equivalent member encoders and 'use_numba'.

		"""
		if isinstance(other, IndPiHiddenMarkovDataEncoder):
			return self.emission_encoder == other.emission_encoder \
				and self.len_encoder == other.len_encoder \
				and self.use_numba == other.use_numba
		else:
			return False

	def _seq_encode(self, x):
		"""Sequence encoding for iid HMM sequences for vectorized numpy functions that do not use numba.

		The returned value is Tuple[rv, None] with rv = (enc, len_enc), where enc is a Tuple of
			enc[0] (int): Total number of observed emissions from all HMM sequences.
			enc[1] (List[Tuple[int, int]]): Bands for the t^th observations of the HMM sequences.
			enc[2] (List[np.ndarray]): Per-band indices of sequences that have a next observed emission.
			enc[3] (np.ndarray): Numpy array of sequence lengths.
			enc[4] (np.ndarray): 2-d matrix mapping sequence i and step t to the row index of the encoded
				emissions, with -1 where sequence i is shorter than t.
			enc[5] (np.ndarray): Sequence index for each encoded emission (band order).
			enc[6]: Sequence encoded emission values in band order.
		and len_enc is the sequence encoded lengths (None when 'len_encoder' is the NullDataEncoder).

		Args:
			x (List[List[T]]): A sequence of iid observations from the HMM.

		Returns:
			Tuple[rv, None]. See above for details.

		"""

		cnt = len(x)
		len_vec = [len(u) for u in x]

		len_enc = self.len_encoder.seq_encode(len_vec)

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
		enc_data = self.emission_encoder.seq_encode(seq_x)
		idx_vec = np.asarray(idx_vec)


		rv = ((tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), len_enc)
		return rv, None


	def seq_encode(self, x):
		"""Sequence encode iid HMM observation sequences for vectorized 'seq_' calls.

		If 'use_numba' is False, returns the numpy encoding via '_seq_encode()' (Tuple[rv, None]).

		Otherwise the returned value is Tuple[None, rv_numba] with rv_numba = (enc, len_enc), where enc is
			enc[0] (np.ndarray): Sequence id for each observed emission.
			enc[1] (np.ndarray): Sequence lengths for each observed HMM sequence.
			enc[2]: Sequence encoded emission values in sequence order.
		and len_enc is the sequence encoded lengths (None when 'len_encoder' is the NullDataEncoder).

		Args:
			x (List[List[T]]): A sequence of iid observations from the HMM.

		Returns:
			Tuple[None, rv_numba] if use_numba, else Tuple[rv, None].

		"""

		if not self.use_numba:
			return self._seq_encode(x)

		idx = []
		xs  = []
		sz  = []

		for i, xx in enumerate(x):
			idx.extend([i]*len(xx))
			xs.extend(xx)
			sz.append(len(xx))

		len_enc = self.len_encoder.seq_encode(sz)

		idx = np.asarray(idx, dtype=np.int32)
		sz  = np.asarray(sz, dtype=np.int32)
		xs  = self.emission_encoder.seq_encode(xs)

		return None, ((idx, sz, xs), len_enc)


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



#@numba.njit('void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], float64[:,:])', parallel=True, fastmath=True, cache=True)
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

# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
IndPiHiddenMarkovAccumulator = IndPiHiddenMarkovEstimatorAccumulator
IndPiHiddenMarkovAccumulatorFactory = IndPiHiddenMarkovEstimatorAccumulatorFactory
IndPiHiddenMarkovModelAccumulator = IndPiHiddenMarkovEstimatorAccumulator
IndPiHiddenMarkovModelAccumulatorFactory = IndPiHiddenMarkovEstimatorAccumulatorFactory
IndPiHiddenMarkovModelDataEncoder = IndPiHiddenMarkovDataEncoder
IndPiHiddenMarkovModelEstimator = IndPiHiddenMarkovEstimator
IndPiHiddenMarkovModelSampler = IndPiHiddenMarkovSampler
