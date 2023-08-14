"""Create, estimate, and sample from an integer PLSI model.

Defines the IntegerPLSIDistribution, IntegerPLSISampler, IntegerPLSIAccumulatorFactory, IntegerPLSIAccumulator,
IntegerPLSIEstimator, and the IntegerPLSIDataEncoder classes for use with pysparkplug.

Consider an Integer PLSI model for a corpus of documents with S states, V word values, and D authors (doc_ids).

Let x (Tuple[int, Sequence[Tuple[int, float]]]) be an observation from a PLSI model, consisting of

    x = (d, [(v_0, c_0), (v_1, c_1), ..., (v_{k-1}, c_{k-1})]),

where the 'd' is some author (doc_id) in the corpus and each tuple (v_i, c_i) corresponds to a value-count couple
for some value 'v_i' in dictionary of words used in the corpus. Let w denote the distinct words {v_i} in the document
represented by x. The density for the PLSI model is given by

    p_mat(w, d) = P_len(nn)*p_mat(d) sum_{j=0}^{k-1} sum_{s=0}^{S-1} ( p_mat(v_j | s )p_mat(s | d) )^(c_j),

where P_len(nn) is the density of the length distribution for 'nn' representing the total number of words in
the document (i.e. nn = sum_i c_i), p_mat(d) is the probability of observing a document from author 'd', p_mat(v_j|s) is the
probability of observing word (integer-valued) given word-topic 's', and p_mat(s|d) are the weights for the word-topic for
author 'd'.

Note: To use this distribution, convert your words and authors of the corpus to unique integer keys.

"""
import numba
import numpy as np
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, StatisticAccumulatorFactory, \
    SequenceEncodableStatisticAccumulator, DataSequenceEncoder, DistributionSampler, ParameterEstimator
from pysp.stats.null_dist import NullDistribution, NullEstimator, NullDataEncoder, NullAccumulator, \
    NullAccumulatorFactory
from pysp.utils.optsutil import count_by_value
from pysp.arithmetic import maxrandint

from typing import List, Optional, Sequence, Tuple, Union, Any, TypeVar, Dict

T1 = TypeVar('T1') ## type for encoded sequence of lengths.
SS1 = TypeVar('SS1') ### type for value of length dist sufficient statistics.


class IntegerPLSIDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, state_word_mat: Union[List[List[float]], np.ndarray],
                 doc_state_mat: Union[List[List[float]], np.ndarray], doc_vec: Union[List[float], np.ndarray],
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None) -> None:
        """IntegerPLSIDistribution object defining an Integer PLSI distribution.

        Args:
            state_word_mat (Union[List[List[float]], np.ndarray]): Array-like of floats that contains a
                p_mat(word | states) for each word in corpus of documents. Cols should sum to 1.0
            doc_state_mat (Union[List[List[float]], np.ndarray]): Array-like of floats that contains a p_mat(doc | states)
                for each document id in corpus of documents. Rows should sum to 1.0
            doc_vec (Union[List[float], np.ndarray]): Array-like containing prior for documents p_mat(d) for each
                document id in corpus of documents. Should sum to 1.0
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for the length of
                each document (i.e. word count in an observed document). Should have support on positive integers.
            name (Optional[str]): Set name to object instance.

        Attributes:
            prob_mat (np.ndarray): 2-d numpy array of floats containing p_mat(word | states) in each row. Dimension is
                given by number of words times number of states.
            state_mat (np.ndarray): 2-d numpy array of floats containing p_mat(doc | states) in each row. Dimension is
                given by number of documents times number of states.
            doc_vec (np.ndarray): 1-d numpy array of floats containing p_mat(doc=d) for each entry. Length is equal to
                number of document ids.
            log_doc_vec (np.ndarray): 1-d numpy array of the log(p_mat(doc=d)).
            num_vals (int): Number of total words in corpus. (Number of rows in prob_mat).
            num_states (int): Number of word topics (mixture components). (Number of columns in prob_mat/state_mat).
            num_docs (int): Total number of document ids in corpus. (Number of rows in state_mat).
            name (Optional[str]): Optional name for object instance.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution object for the number of words per
                document. Defaults to the NullDistribution if None is passed.

        """
        self.prob_mat    = np.asarray(state_word_mat, dtype=np.float64)
        self.state_mat   = np.asarray(doc_state_mat, dtype=np.float64)
        self.doc_vec     = np.asarray(doc_vec, dtype=np.float64)
        self.log_doc_vec = np.log(self.doc_vec)
        self.num_vals    = self.prob_mat.shape[0]
        self.num_states  = self.prob_mat.shape[1]
        self.num_docs    = self.state_mat.shape[0]
        self.name        = name
        self.len_dist    = len_dist if len_dist is not None else NullDistribution()

    def __str__(self) -> str:
        """Return string representation of object instance."""
        s1 = ','.join(['[' + ','.join(map(str, self.prob_mat[i, :])) + ']' for i in range(len(self.prob_mat))])
        s2 = ','.join(['[' + ','.join(map(str, self.state_mat[i, :])) + ']' for i in range(len(self.state_mat))])
        s3 = ','.join(map(str, self.doc_vec))
        s4 = repr(self.name)
        s5 = str(self.len_dist)
        return 'IntegerPLSIDistribution([%s], [%s], [%s], name=%s, len_dist=%s)'%(s1, s2, s3, s4, s5)

    def density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Evaluate the density of PLSI model for an observation x.

        See log_density() for details on the density evaluation.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single observation of integer PLSI.

        Returns:
            Density evaluated at observed value x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Evaluate the log-density of PLSI model for an observation of x.

        Consider an Integer PLSI model for a corpus of documents with S states, V word values, and D documents ids
        (authors).

        Let x (Tuple[int, Sequence[Tuple[int, float]]]) be an observation from a PLSI model, consisting of
        x = (d, [(v_0, c_0), (v_1, c_1), ..., (v_{k-1}, c_{k-1})]), where the 'd' is some document d_id in the corpus and
        each tuple (v_i, c_i) corresponds to a value-count couple in the corpus. The log-likelihood is given by

        log(p_mat(x)) = log(p_mat(d)) + sum_{j=0}^{k-1} c_k*log( sum_{s=0}^{S-1} p_mat(d|s)p_mat(s|v_k) ) + log(P_len(nn)),

        where P_len(nn) is the density of the length distribution for 'nn' representing the total number of words in
        the document.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): (doc_id, [(value_id, count_for_value)]). See above for details.

        Returns:
            Log-density evaluated at a single observation x.

        """

        d_id = x[0]
        xv = np.asarray([u[0] for u in x[1]], dtype=int)
        xc = np.asarray([u[1] for u in x[1]], dtype=float)

        rv = 0.0
        rv += np.dot(np.log(np.dot(self.prob_mat[xv, :], self.state_mat[d_id, :])), xc)
        rv += np.log(self.doc_vec[d_id])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(np.sum(xc))

        return rv

    def component_log_density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> np.ndarray:
        """Evaluate the log-density for each state in the PLSI.

        Returns count*log(p_mat(W|S)) for each word-count pair in the document. Returned value is S by 1 where S is the
        number of components in the model.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single PLSI observation of form
                (doc_id, [(value_id, count_for_value)]).

        Returns:
            Numpy array of length S (num_states).

        """
        xv = np.asarray([u[0] for u in x[1]], dtype=int)
        xc = np.asarray([u[1] for u in x[1]], dtype=float)

        return np.dot(np.log(self.prob_mat[xv, :]).T, xc)

    def seq_log_density(self, x: Tuple[Optional[T1], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                           np.ndarray]]) -> np.ndarray:
        """Vectorized evaluation of the log-density for an encoded sequence of iid observation from a PLSI model.

        See log_density() function for details on the log-likelihood.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI model. See above for details.

        Returns:
            Numpy array of log-density evaluated at each observation in the encoded sequence.

        """
        nn, (xv, xc, xd, xi, xn, xm) = x
        cnt = len(xn)

        w = np.zeros(len(xv), dtype=np.float64)
        index_dot(self.prob_mat, xv, self.state_mat, xd, w)
        w = np.log(w, out=w)
        w *= xc

        rv = np.zeros(cnt, dtype=np.float64)
        bincount(xi, w, rv)
        rv += self.log_doc_vec[xm]

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(nn)

        return rv

    def seq_component_log_density(self, x: Tuple[Optional[T1], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                     np.ndarray, np.ndarray]]) -> np.ndarray:
        """Vectorized evaluation of the component log-density for each observation in an encoded sequence of iid PLSI
            observations.

        See component_log_density() function for details on component log-likelihood evaluation.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI model. See above for details.

        Returns:
            2-d numpy array containing N rows of num_state sized arrays.

        """
        nn, (xv, xc, xd, xi, xn, xm) = x
        rv = np.zeros((xi[-1]+1, self.num_states), dtype=np.float64)
        w_mat = self.prob_mat
        fast_seq_component_log_density(xv, xc, xd, xi, xm, w_mat, rv)
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IntegerPLSISampler':
        """Return an IntegerPLSISampler object from IntegerPLSIDistribution instance."""
        return IntegerPLSISampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerPLSIEstimator':
        """Create an IntegerPLSIEstimator object from IntegerPLSIDistribution instance.

        Args:
            pseudo_count (Optional[float]): Re-weight object instance sufficient statistics when passed to estimator.

        Returns:
            IntegerPLSIEstimator object.

        """
        if pseudo_count is None:
            return IntegerPLSIEstimator(num_vals=self.num_vals, num_states=self.num_states,num_docs=self.num_docs,
                                        len_estimator=self.len_dist.estimator(),name=self.name)
        else:
            pseudo_count = (pseudo_count, pseudo_count, pseudo_count)
            return IntegerPLSIEstimator(num_vals=self.num_vals, num_states=self.num_states, num_docs=self.num_docs,
                                        pseudo_count = pseudo_count,
                                        suff_stat=(self.prob_mat.T, self.state_mat, self.doc_vec),
                                        len_estimator=self.len_dist.estimator(), name=self.name)

    def dist_to_encoder(self) -> 'IntegerPLSIDataEncoder':
        """Returns IntegerPLSIDataEncoder object."""
        return IntegerPLSIDataEncoder(len_encoder=self.len_dist.dist_to_encoder())


class IntegerPLSISampler(DistributionSampler):

    def __init__(self, dist: IntegerPLSIDistribution, seed: Optional[int] = None) -> None:
        """IntegerPLSISampler object for sampling from IntegerPLSIDistribution.

        Args:
            dist (IntegerPLSIDistribution): IntegerPLSIDistribution instance to sampler from.
            seed (Optional[int]): Set seed for random number generator used in sampling.

        Attributes:
            rng (RandomState): RandomState object with seed set if passed.
            dist (IntegerPLSIDistribution): IntegerPLSIDistribution instance to sampler from.
            size_rng (RandomState): RandomState object for sampling the length of documents.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.size_rng = self.dist.len_dist.sampler(self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) \
            -> Union[Tuple[int, Sequence[Tuple[int, float]]], Sequence[Tuple[int, Sequence[Tuple[int, float]]]]]:
        """Generate iid samples from PLSI model.

        Args:
            size (Optional[int]): Number of samples to generate. Defaults to 0 if size is None.

        Returns:
            Sequence of iid PLSI samples if size is not None, else a single sample from PLSI model.

        """
        if size is None:
            d_id = self.rng.choice(self.dist.num_docs, p=self.dist.doc_vec)
            cnt = self.size_rng.sample()
            z = self.rng.multinomial(cnt, pvals=self.dist.state_mat[d_id, :])
            rv = []
            for i, n in enumerate(z):
                if n > 0:
                    rv.extend(self.rng.choice(self.dist.num_vals, p=self.dist.prob_mat[:, i], replace=True, size=n))

            return d_id, list(count_by_value(rv).items())

        else:
            return [self.sample() for i in range(size)]


class IntegerPLSIAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals: int, num_states: int, num_docs: int,
                 len_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)) -> None:
        """IntegerPLSIAccumulator object for aggregating sufficient statistics from observed data.

        Note: Keys in order, words/values, states, documents.

        Args:
            num_vals (int): Number of words in the corpus.
            num_states (int): Number of word-topics.
            num_docs (int): Number of authors (doc_ids) in the corpus.
            len_acc (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the length of documents.
                Should have support on non-negative integer values.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Optional keys for words, states, and
                authors (doc_ids).

        Attributes:
            num_vals (int): Number of words in the corpus.
            num_states (int): Number of word-topics or mixture components.
            num_docs (int): Number of authors (doc_ids) in the corpus.
            word_count (ndarray): Numpy array of shape num_states by num_vals for aggregating state/word counts.
            comp_count (ndarray): Numpy array (num_docs by num_states) for aggregating doc/state counts.
            doc_count (ndarray): Numpy array for aggregating counts of authors (prior on doc_ids).
            name (Optional[str]): Name of object instance.
            wc_key (Optional[str]): Key for merging 'word_count' with objects containing matching keys.
            sc_key (Optional[str]): Key for merging 'comp_count' with objects containing matching keys.
            dc_key (Optional[str]): Key for merging 'doc_count' with objects containing matching keys.
            len_acc (SequenceEncodableStatisticAccumulator): Accumulator object for the lengths of documents (total
                word counts). Defaults to the NullAccumulator if None is passed.

            _init_rng (bool): True if RandomState objects for accumulator have been initialized.
            _acc_rng (Optional[RandomState]): RandomState object for initializing the PLSI model.
            _len_rng (Optional[RandomState]): RandomState object for initializing the length accumulator.

        """
        self.num_vals   = num_vals
        self.num_states = num_states
        self.num_docs   = num_docs
        self.word_count = np.zeros((num_states, num_vals), dtype=np.float64)
        self.comp_count = np.zeros((num_docs, num_states), dtype=np.float64)
        self.doc_count  = np.zeros(num_docs, dtype=np.float64)
        self.name       = name
        self.wc_key, self.sc_key, self.dc_key = keys if keys is not None else (None, None, None)
        self.len_acc    = len_acc if len_acc is not None else NullAccumulator()

        ### Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[RandomState] = None
        self._len_rng: Optional[RandomState] = None

    def update(self, x: Tuple[int, Sequence[Tuple[int, float]]], weight: float, estimate: IntegerPLSIDistribution) \
            -> None:
        """Update the sufficient statistics of object instance for a single observation x.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): An observation from integer PLSI model.
            weight (float): Observation weight.
            estimate (IntegerPLSIDistribution): Prior estimate of IntegerPLSIDistribution object.

        Returns:
            None.

        """
        d_id = x[0]
        xv = np.asarray([u[0] for u in x[1]])
        xc = np.asarray([u[1] for u in x[1]])

        update = (estimate.prob_mat[xv, :] * estimate.state_mat[d_id, :]).T
        update *= xc*weight/np.sum(update, axis=0)
        self.comp_count[d_id, :] += np.sum(update, axis=1)
        self.word_count[:, xv] += update
        self.doc_count[d_id] += weight

        self.len_acc.update(np.sum(xc), weight, estimate.len_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize RandomState objects for accumulators from rng.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _acc_rng and _len_rng.

        Returns:
            None.

        """
        seeds = rng.randint(maxrandint, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._len_rng = RandomState(seed=seeds[1])
        self._init_rng = True

    def initialize(self, x: Tuple[int, Sequence[Tuple[int, float]]], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics of object instance with observation x.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): An observation from integer PLSI model.
            weight (float): Observation weight.
            rng (RandomState): RandomState object for seeded initialization.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        d_id = x[0]
        xv = np.asarray([u[0] for u in x[1]])
        xc = np.asarray([u[1] for u in x[1]])

        update = self._acc_rng.dirichlet(np.ones(self.num_states)/self.num_states, size=len(xc)).T
        update *= xc*weight
        self.word_count[:, xv] += update
        self.comp_count[d_id, :] += np.sum(update, axis=1)
        self.doc_count[d_id] += weight

        self.len_acc.update(np.sum(xc), weight, self._len_rng)

    def seq_initialize(self, x: Tuple[Optional[T1], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray]], weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of sufficient statistics form an encoded sequence of observations in arg 'x'.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI. See above for details.
            weights (ndarray): Weights for observations in encoded sequence.
            rng (RandomState): Used to initialize member RandomState variables.

        Returns:
            None.

        """
        nn, (xv, xc, xd, xi, xn, xm) = x

        if not self._init_rng:
            self._rng_initialize(rng)

        update = self._acc_rng.dirichlet(np.ones(self.num_states) / self.num_states, size=len(xv)).T
        update *= xc * weights[xi]
        self.word_count += vec_bincount3(xv, update, out=np.zeros_like(self.word_count, dtype=np.float64))
        self.doc_count += np.bincount(xm, weights, minlength=self.num_docs)
        self.comp_count += vec_bincount4(xd, update, out=np.zeros_like(self.comp_count, dtype=np.float64))

        self.len_acc.seq_initialize(nn, weights, self._len_rng)

    def seq_update(self, x: Tuple[Optional[T1], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                      np.ndarray]],
                   weights: np.ndarray, estimate: IntegerPLSIDistribution) -> None:
        """Vectorized update of sufficient statistics for encoded sequence of iid observations in x.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI model. See above for details.
            weights (np.ndarray): Numpy array of observation weights.
            estimate (IntegerPLSIDistribution): Prior estimate of IntegerPLSIDistribution object.

        Returns:
            None.

        """
        nn, (xv, xc, xd, xi, xn, xm) = x
        fast_seq_update(xv, xc, xd, xi, xm, weights, estimate.prob_mat, estimate.state_mat, self.word_count,
                        self.comp_count, self.doc_count)

        '''

        temp = xc*weights[xi]
        update  = estimate.prob_mat[xv, :] * estimate.state_mat[xd, :]

        temp /= np.sum(update, axis=1)
        update *= temp[:,None]

        #vec_bincount1(xv, update, self.word_count.T)
        #vec_bincount1(xd, update, self.comp_count)
        #bincount(xm, weights, self.num_docs)

        for i in range(self.num_states):
            self.word_count[i,:] += np.bincount(xv, weights=update[:,i], minlength=self.num_vals)
            self.comp_count[:,i] += np.bincount(xd, weights=update[:,i], minlength=self.num_docs)
        self.doc_count += np.bincount(xm, weights=weights, minlength=self.num_docs)
        '''
        self.len_acc.seq_update(nn, weights, estimate.len_dist)

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]) -> 'IntegerPLSIAccumulator':
        """Combine the sufficient statistics in arg 'suff_stat' with object instance.

        Arg 'suff_stat' is Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]] containing:
            suff_stat[0] (np.ndarray): State/word counts with matching dimension of num_states by num_vals.
            suff_stat[1] (np.ndarray): Doc/state counts with matching dimension of num_docs by num_states.
            suff_stat[2] (np.ndarray): Author counts with length (num_docs).
            suff_stat[3] (Optional[SS1]): Sufficient statistics for the length of document distribution having type SS1.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerPLSIAccumulator object.

        """
        self.word_count += suff_stat[0]
        self.comp_count += suff_stat[1]
        self.doc_count += suff_stat[2]

        self.len_acc.combine(suff_stat[3])

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        """Returns sufficient statistics of IntegerPLSIAccumulator object instance.

        Returned value 'suff_stat' is Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]] containing:
            suff_stat[0] (np.ndarray): State/word counts with matching dimension of num_states by num_vals.
            suff_stat[1] (np.ndarray): Doc/state counts with matching dimension of num_docs by num_states.
            suff_stat[2] (np.ndarray): Author counts with length (num_docs).
            suff_stat[3] (Optional[SS1]): Sufficient statistics for the length of document distribution having type SS1.

        """
        return self.word_count, self.comp_count, self.doc_count, self.len_acc.value()

    def from_value(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]) -> 'IntegerPLSIAccumulator':
        """Set the sufficient statistics of object instance to arg 'x' values.

        Arg 'x' is Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]] containing:
            x[0] (np.ndarray): State/word counts with matching dimension of num_states by num_vals.
            x[1] (np.ndarray): Doc/state counts with matching dimension of num_docs by num_states.
            x[2] (np.ndarray): Author counts with length (num_docs).
            x[3] (Optional[SS1]): Sufficient statistics for the length of document distribution having type SS1.

        Args:
            x: Aggregated sufficient statistics. See above for details.

        Returns:
            IntegerPLSIAccumulator object.

        """
        self.word_count = x[0]
        self.comp_count = x[1]
        self.doc_count = x[2]
        self.len_acc.from_value(x[3])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the sufficient statistics of object instance with matching keys.

        If wc_key is set, merge the state/word count variable.
        If sc_key is set, merge the doc/state count variable.
        If dc_key is set, merge the author count variable.

        Call key_merge() of accumulator for the length. Note nothing is done for this if default NullAccumulator is set.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to sufficient statistics.

        Returns:
            None.

        """
        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                stats_dict[self.wc_key] += self.word_count
            else:
                stats_dict[self.wc_key] = self.word_count

        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                stats_dict[self.sc_key] += self.comp_count
            else:
                stats_dict[self.sc_key] = self.comp_count

        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                stats_dict[self.dc_key] += self.doc_count
            else:
                stats_dict[self.dc_key] = self.doc_count

        self.len_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set the sufficient statistics of object instance to matching key values in arg 'stats_dict'.

        If wc_key is set, set the state/word count variable to matching key in stats_dict.
        If sc_key is set, set the doc/state count variable to matching key in stats_dict.
        If dc_key is set, set the author count variable to matching key in stats_dict.

        Call key_replace() of accumulator for the length. Note nothing is done for this if default NullAccumulator is
        set.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to sufficient statistics.

        Returns:
            None.

        """
        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                self.word_count = stats_dict[self.wc_key]
        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                self.comp_count = stats_dict[self.sc_key]
        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                self.doc_count = stats_dict[self.dc_key]

        self.len_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'IntegerPLSIDataEncoder':
        """Return an IntegerPLSIDataEncoder object."""
        len_encoder = self.len_acc.acc_to_encoder()
        return IntegerPLSIDataEncoder(len_encoder=len_encoder)

class IntegerPLSIAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, num_vals: int, num_states: int, num_docs: int,
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        """IntegerPLSIAccumulatorFactory object for creating IntegerPLSIAccumulator objects.

        Args:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_factory (Optional[StatisticsAccumulatorFactory]): Accumulator factory object for length distribution.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for merging
                word, state, and doc sufficient statistics with matching keys.
            name (Optional[str]): Set name for object.

        Attributes:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_factory (StatisticsAccumulatorFactory): Accumulator factory object for length distribution. Defaults
                to the NullAccumulatorFactory(). Should have support on non-negative integers.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for merging word, state, and doc
                sufficient statistics with matching keys.
            name (Optional[str]): Set name for object.

        """
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None, None)
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs
        self.name = name

    def make(self) -> 'IntegerPLSIAccumulator':
        """Returns IntegerPLSIAccumulator object."""
        return IntegerPLSIAccumulator(self.num_vals, self.num_states, self.num_docs, len_acc=self.len_factory.make(),
                                      keys=self.keys, name=self.name)

class IntegerPLSIEstimator(ParameterEstimator):

    def __init__(self, num_vals: int, num_states: int, num_docs: int,
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = (None, None, None),
                 suff_stat: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                           Optional[np.ndarray]]] = (None, None, None),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)) -> None:
        """IntegerPLSIEstimator for estimating integer PLSI distributions from aggregated sufficient statistics.

        Args:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object for the length of
                documents. Should have support on non-negative integers if not None.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float], Optional[float]]]): Optional re-weight
                sufficient statistics in 'estimate()' function.
            suff_stat (Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]): Optional
                Tuple of numpy arrays containing 'word_counts' (num_states by num_vals), 'state_counts' (num_docs by
                num_states), and doc_counts (length num_docs).
            name (Optional[str]): Set name to object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for merging word, state, and doc
                sufficient statistics with matching keys.

        Attributes:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_estimator (ParameterEstimator): Optional ParameterEstimator object for the length of documents. Should
                have support on non-negative integers. Defaults to NullEstimator() if None is passed.
            pseudo_count (Tuple[Optional[float], Optional[float], Optional[float]]): Optional re-weight sufficient
                statistics in 'estimate()' function. Defaults to (None, None, None) if None is passed.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]): Optional
                Tuple of numpy arrays containing 'word_counts' (num_states by num_vals), 'state_counts' (num_docs by
                num_states), and doc_counts (length num_docs). Defaults to (None, None, None) if None is passed.
            name (Optional[str]): Name of object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for merging word, state, and doc
                sufficient statistics with matching keys.
        """
        self.suff_stat     = suff_stat if suff_stat is not None else (None, None, None)
        self.pseudo_count  = pseudo_count if pseudo_count is not None else (None, None, None)
        self.num_vals      = num_vals
        self.num_states    = num_states
        self.num_docs      = num_docs
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.keys          = keys if keys is not None else (None, None, None)
        self.name          = name

    def accumulator_factory(self) -> 'IntegerPLSIAccumulatorFactory':
        """Returns IntegerPLSIAccumulatorFactory object."""
        len_est = self.len_estimator.accumulator_factory()
        return IntegerPLSIAccumulatorFactory(self.num_vals, self.num_states, self.num_docs, len_est, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]])\
            -> 'IntegerPLSIDistribution':
        """Estimate IntegerPLSIDistribution from aggregated sufficient statistics in arg 'suff_stat'.

        Args:
            nobs (Optional[float]): Optional number of observations used to accumulate 'suff_stat'.
            suff_stat: See above for details.

        Returns:
            IntegerPLSIDistribution object.

        """
        word_count, comp_count, doc_count, len_suff_stats = suff_stat

        if self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt*self.suff_stat[0].T
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        elif self.pseudo_count[0] is not None and self.suff_stat[0] is None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        else:
            word_prob_mat = word_count.T / np.sum(word_count, axis=1)

        if self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt * self.suff_stat[1]
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        elif self.pseudo_count[1] is not None and self.suff_stat[1] is None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        else:
            state_prob_mat = comp_count / np.sum(comp_count, axis=1, keepdims=True)

        if self.pseudo_count[2] is not None and self.suff_stat[2] is not None:
            adj_cnt = self.pseudo_count[1] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt*self.suff_stat[2]
            doc_prob_vec /= np.sum(doc_prob_vec)

        elif self.pseudo_count[2] is not None and self.suff_stat[2] is None:
            adj_cnt = self.pseudo_count[1] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt
            doc_prob_vec /= np.sum(doc_prob_vec)

        else:
            doc_prob_vec = doc_count / np.sum(doc_count)

        len_dist = self.len_estimator.estimate(None, len_suff_stats)

        return IntegerPLSIDistribution(word_prob_mat, state_prob_mat, doc_prob_vec, name=self.name, len_dist=len_dist)


class IntegerPLSIDataEncoder(DataSequenceEncoder):

    def __init__(self, len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder()) -> None:
        """IntegerPLSIDataEncoder object for encoding sequences of iid observations from a PLSI model.

        Args:
            len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder for the total number of words
                in each document.

        Attributes:
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for the total number of words in each document,
                defaulting to NullDataEncoder if None is passed.

        """
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Returns a string representation of object instance."""
        return 'IntegerPLSIDataEncoder(len_dist=' + str(self.len_encoder) + ')'

    def __eq__(self, other: object) -> bool:
        """Check if object is equivalent to instance of IntegerPLSIDataEncoder.

        Args:
            other (object): Other object to compare to instance.

        Returns:
            True if object is an instance of IntegerPLSIDataEncoder with matching 'len_encoder' attribute.

        """
        if isinstance(other, IntegerPLSIDataEncoder):
            return other.len_encoder == self.len_encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Tuple[int, Sequence[Tuple[int, float]]]])\
            -> Tuple[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Encode a sequence of iid PLSI observations for use with vectorized functions.

        Input arg 'x' is a sequence of iid PLSI observations having form

        x = [ (doc_id, [(value, count),...]),... ].

        The return value is a Tuple length 2. The first component contains data type Optional[T1] corresponding to the
        sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document d_id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document d_id's for the lengths above (len = len(x)).

        Args:
            x (Sequence[Tuple[int, Sequence[Tuple[int, float]]]]): See above for details.

        Returns:
            See above for details.

        """
        xv = []
        xc = []
        xd = []
        xi = []
        xn = []
        xm = []

        for i, (d_id, xx) in enumerate(x):

            v = [u[0] for u in xx]
            c = [u[1] for u in xx]

            xv.extend(v)
            xc.extend(c)
            xd.extend([d_id]*len(v))
            xi.extend([i]*len(v))
            xn.append(np.sum(c))
            xm.append(d_id)

        xv = np.asarray(xv, dtype=np.int32)
        xc = np.asarray(xc, dtype=np.float64)
        xd = np.asarray(xd, dtype=np.int32)
        xi = np.asarray(xi, dtype=np.int32)
        xn = np.asarray(xn, dtype=np.float64)
        xm = np.asarray(xm, dtype=np.int32)

        nn = self.len_encoder.seq_encode(xn)

        return nn, (xv, xc, xd, xi, xn, xm)


@numba.njit('void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], float64[:,:], float64[:], '
            'float64[:])', fastmath=True)
def fast_seq_log_density(xv, xc, xd, xi, xm, wmat, smat, dvec, out):
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    for i in range(n):
        ll = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        i3 = xi[i]
        for j in range(k):
            ll += wmat[i1,j]*smat[i2,j]
        out[i3] += cc*np.log(ll)
    for i in range(m):
        out[i] += dvec[xm[i]]

@numba.njit('void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], float64[:,:])', fastmath=True)
def fast_seq_component_log_density(xv, xc, xd, xi, xm, wmat, out):
    n = len(xv)
    k = wmat.shape[1]
    for i in range(n):
        cc = xc[i]
        i1 = xv[i]
        i3 = xi[i]
        for j in range(k):
            out[i3, j] += np.log(wmat[i1,j])*cc

@numba.njit('void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:], float64[:,:], float64[:,:], '
            'float64[:,:], float64[:,:], float64[:])', fastmath=True)
def fast_seq_update(xv, xc, xd, xi, xm, weights, wmat, smat, wcnt, scnt, dcnt):
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    posterior = np.zeros(k, dtype=np.float64)
    for i in range(n):
        norm_const = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        ww = weights[xi[i]]
        for j in range(k):
            temp = wmat[i1,j]*smat[i2,j]
            posterior[j] = temp
            norm_const  += temp
        norm_const = ww*cc/norm_const
        for j in range(k):
            temp = posterior[j]*norm_const
            wcnt[j,i1] += temp
            scnt[i2,j] += temp
    for i in range(m):
        dcnt[xm[i]] += weights[i]


@numba.njit('float64[:](float64[:,:], int32[:], float64[:,:], int32[:], float64[:])')
def index_dot(x, xi, y, yi, out):
    n = x.shape[1]
    for i in range(len(xi)):
        i1 = xi[i]
        i2 = yi[i]
        for j in range(n):
            out[i] += x[i1,j]*y[i2,j]
    return out

@numba.njit('float64[:](int32[:], float64[:], float64[:])')
def bincount(x, w, out):
    for i in range(len(x)):
        out[x[i]] += w[i]
    return out

@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount1(x, w, out):
    n = w.shape[1]
    for i in range(len(x)):
        for j in range(n):
            out[x[i],j] += w[i,j]
    return out

@numba.njit('float64[:,:](int32[:], float64[:,:], int32[:], float64[:,:])')
def vec_bincount2(x, w, y, out):
    for i in range(len(x)):
        out[x[i], :] += w[y[i], :]
    return out


@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount3(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Used to update comp counts for word/state probabilities.

    N = len(x)
    S = number of states.
    U = unique values in x can take on (unique words in corpus).

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): S by U matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[:, x[j]] += w[:, j]
    return out


@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount4(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Used to initialize doc/state counts.

    N = len(x)
    S = number of states.
    U = unique values in x can take on. (Unique number of authors).

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): U by S matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[x[j], :] += w[:, j]
    return out


