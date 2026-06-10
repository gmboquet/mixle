from typing import Optional, List, Tuple
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
import pysp.utils.vector as vec
from collections import defaultdict
from scipy.sparse import csc_matrix, lil_matrix
from pysp.utils.optsutil import count_by_value
import itertools
from pysp.arithmetic import maxrandint


class MarkovTransformDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, init_prob_vec, cond_prob_mat, alpha=0.0, len_dist=None):

        self.init_prob_vec = np.asarray(init_prob_vec, dtype=np.float64)
        self.cond_prob_mat = csc_matrix(cond_prob_mat, dtype=np.float64)
        self.len_dist = len_dist
        self.num_vals = len(init_prob_vec)
        self.alpha = alpha

    def __str__(self):
        s1 = ','.join(map(str,self.init_prob_vec))
        temp = self.cond_prob_mat.nonzero()
        tt = np.asarray(self.cond_prob_mat[temp[0],temp[1]]).flatten()
        s20 = ','.join(map(str, tt))
        s21 = ','.join(map(str, temp[0]))
        s22 = ','.join(map(str, temp[1]))
        s2 = '([%s], ([%s],[%s]))'%(s20, s21, s22)
        s3 = str(self.alpha)
        s4 = str(self.len_dist)
        return 'MarkovMixtureSetDistribution([%s], %s, alpha=%s, len_dist=%s)' % (s1, s2, s3, s4)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):

        nw = self.num_vals
        a  = self.alpha/nw
        b  = 1 - self.alpha

        xx, yy, zz = x

        ll1 = 0.0
        ll2 = 0.0
        ll3 = 0.0
        n1 = 0
        n2 = 0
        n3 = 0
        for u,c in xx:
            ll1 += np.log(self.init_prob_vec[u])*c
            n1 += c
        for u,c in yy:
            ll2 += np.log(self.init_prob_vec[u])*c
            n2 += c

        nn = n1*n2

        for w, cw in zz:
            loc_ll = 0.0
            for u, cu in xx:
                for v, cv in yy:
                    loc_ll += (b*self.cond_prob_mat[u*nw + v, w] + a)*cu*cv/nn
            ll3 += np.log(loc_ll)*cw
            n3 += cw

        rv = ll1 + ll2 + ll3

        if self.len_dist is not None:
            rv += self.len_dist.log_density([n1, n2, n3])

        return rv

    def seq_log_density(self, x):

        nw = self.num_vals
        a  = self.alpha/nw
        b  = 1 - self.alpha

        rv = np.zeros(len(x[0]), dtype=np.float64)

        for i,entry in enumerate(x[0]):

            xx, cx, yy, cy, zz, cz = entry

            ridx = np.reshape(xx*nw, (-1,1)) + np.reshape(yy, (1,-1))
            ridx = ridx.flatten()

            cc = np.reshape(cx, (-1,1)) * np.reshape(cy, (1,-1))
            cc = cc.flatten()
            cc /= cc.sum()

            loc_cprob = self.cond_prob_mat[:, zz]
            loc_cprob = ((loc_cprob[ridx,:].toarray().T) * b) + a
            ll3 = np.dot(np.log(np.dot(loc_cprob, cc)), cz)

            ll1 = np.dot(np.log(self.init_prob_vec[xx]), cx)
            ll2 = np.dot(np.log(self.init_prob_vec[yy]), cy)

            rv[i] = ll1 + ll2 + ll3

        if self.len_dist is not None:
            lln = self.len_dist.seq_log_density(x[1])
            rv += lln

        return rv


    def seq_encode(self, x):

        rv = []
        nn = []
        vset = set()

        for xx in x:
            rv0 = []
            nn0 = []
            for cvec in xx:
                rv0.append(np.asarray([v for v, c in cvec], dtype=int))
                rv0.append(np.asarray([c for v, c in cvec], dtype=float))
                nn0.append(np.sum(rv0[-1]))

            vset.update(itertools.product(rv0[0], rv0[2], rv0[4]))
            rv.append(tuple(rv0))
            nn.append(tuple(nn0))

        if self.len_dist is not None:
            nn = self.len_dist.seq_encode(nn)
        else:
            nn = None

        vv = np.zeros((len(vset),3), dtype=int)
        for i,vvv in enumerate(vset):
            vv[i,:] = vvv[:]

        return rv, nn, vv

    def sampler(self, seed=None):
        return MarkovTransformSampler(self, seed)


class MarkovTransformSampler(object):

    def __init__(self, dist: MarkovTransformDistribution, seed: Optional[int] = None):
        self.rng  = np.random.RandomState(seed)
        self.dist = dist
        #self.init_sampler  = np.random.RandomState(self.rng.tomaxint())
        #self.next_sampler  = np.random.RandomState(self.rng.tomaxint())
        #self.tran_sampler  = np.random.RandomState(self.rng.tomaxint())
        #self.flat_sampler  = np.random.RandomState(self.rng.tomaxint())
        self.size_sampler  = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None):

        if size is None:

            slens = self.size_sampler.sample()
            rng = np.random.RandomState(self.rng.randint(0, maxrandint))

            v1 = list(rng.choice(len(self.dist.init_prob_vec), p=self.dist.init_prob_vec, replace=True, size=slens[0]))
            v2 = list(rng.choice(len(self.dist.init_prob_vec), p=self.dist.init_prob_vec, replace=True, size=slens[1]))
            v3 = []

            z1 = list(rng.choice(len(v1), replace=True, size=slens[2]))
            z2 = list(rng.choice(len(v2), replace=True, size=slens[2]))
            nw = self.dist.num_vals

            for zz1, zz2 in zip(z1, z2):

                if rng.rand() > self.dist.alpha:
                    p = self.dist.cond_prob_mat[v1[zz1]*nw + v2[zz2], :].toarray().flatten()
                    v3.append(rng.choice(nw, p=p))
                else:
                    v3.append(rng.choice(nw))

            return list(count_by_value(v1).items()), list(count_by_value(v2).items()), list(count_by_value(v3).items())

        else:
            return [self.sample() for i in range(size)]



class MarkovTransformAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals, size_acc=None, keys=(None,None)):
        self.init_count = np.zeros(num_vals)
        self.trans_count = csc_matrix((num_vals*num_vals, num_vals))
        self.size_accumulator = size_acc
        self.num_vals = num_vals
        self.init_key = keys[0]
        self.trans_key = keys[1]

    def update(self, x, weight, estimate):

        nw = self.num_vals
        xx, yy, zz = x
        vx = np.asarray([u[0] for u in xx], dtype=int)
        cx = np.asarray([u[1] for u in xx], dtype=float)
        vy = np.asarray([u[0] for u in yy], dtype=int)
        cy = np.asarray([u[1] for u in yy], dtype=float)
        vz = np.asarray([u[0] for u in zz], dtype=int)
        cz = np.asarray([u[1] for u in zz], dtype=float)

        ridx = np.reshape(vx * nw, (-1, 1)) + np.reshape(vy, (1, -1))
        ridx = ridx.flatten()[:,None]

        cc = np.reshape(cx, (-1, 1)) * np.reshape(cy, (1, -1))
        cc = cc.flatten()[:,None]
        cs = cc.sum()

        temp = estimate.cond_prob_mat[ridx, vz].toarray()

        loc_cprob = temp * cc
        w = loc_cprob.sum(axis=0)
        loc_cprob *= (cz / w) * weight

        self.trans_count[ridx, vz] += loc_cprob
        self.init_count[vx] += cx
        self.init_count[vy] += cy

        if self.size_accumulator is not None:
            self.size_accumulator.update((cx.sum(), cy.sum(), cz.sum()), weight, estimate.len_dist)


    def initialize(self, x, weight, rng):

        nw = self.num_vals
        xx, yy, zz = x
        vx = np.asarray([u[0] for u in xx], dtype=int)
        cx = np.asarray([u[1] for u in xx], dtype=float)
        vy = np.asarray([u[0] for u in yy], dtype=int)
        cy = np.asarray([u[1] for u in yy], dtype=float)
        vz = np.asarray([u[0] for u in zz], dtype=int)
        cz = np.asarray([u[1] for u in zz], dtype=float)

        ridx = np.reshape(vx * nw, (-1, 1)) + np.reshape(vy, (1, -1))
        ridx = ridx.flatten()[:,None]

        cc = np.reshape(cx, (-1, 1)) * np.reshape(cy, (1, -1))
        cc = cc.flatten()

        loc_cprob = np.outer(cc, weight*cz)
        #umat = lil_matrix((nw * nw, nw))
        #umat[ridx, vz] = loc_cprob

        self.trans_count[ridx, vz] += loc_cprob
        #self.trans_count += umat
        self.init_count[vx] += cx
        self.init_count[vy] += cy

        if self.size_accumulator is not None:
            self.size_accumulator.initialize((cx.sum(), cy.sum(), cz.sum()), weight, rng)

    def seq_update(self, x, weights, estimate):

        nw = self.num_vals
        nzv = x[2]

        umat = csc_matrix((np.zeros(nzv.shape[0]), (nzv[:,0]*nw + nzv[:,1], nzv[:,2])), shape=(nw*nw, nw))

        for i,(entry,ww) in enumerate(zip(x[0], weights)):

            xx, cx, yy, cy, zz, cz = entry

            ridx = np.reshape(xx*nw, (-1,1)) + np.reshape(yy, (1,-1))
            ridx = ridx.flatten()[:,None]

            cc = np.reshape(cx, (-1,1)) * np.reshape(cy, (1,-1))
            cc = cc.flatten()[:,None]
            cs = cc.sum()

            temp = estimate.cond_prob_mat[ridx, zz].toarray()

            loc_cprob = temp * cc
            w = loc_cprob.sum(axis=0)
            loc_cprob *= (cz/w)*ww

            umat[ridx,zz] += loc_cprob
            self.init_count[xx] += cx
            self.init_count[yy] += cy

        if self.size_accumulator is not None:
            self.size_accumulator.seq_update(x[1], weights, estimate.len_dist)

        self.trans_count += umat

    def combine(self, suff_stat):

        init_count, trans_count, size_acc = suff_stat

        if self.size_accumulator is not None:
            self.size_accumulator.combine(size_acc)

        self.init_count += init_count
        self.trans_count += trans_count

        return self

    def value(self):
        if self.size_accumulator is not None:
            return self.init_count, self.trans_count, self.size_accumulator.value()
        else:
            return self.init_count, self.trans_count, None

    def from_value(self, x):

        init_count, trans_count, size_acc = x

        self.init_count = init_count
        self.trans_count = trans_count
        if self.size_accumulator is not None:
            self.size_accumulator.from_value(size_acc)

        return self



    def key_merge(self, stats_dict):

        '''
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators
        '''
        if self.size_accumulator is not None:
            self.size_accumulator.key_merge(stats_dict)


    def key_replace(self, stats_dict):

        '''
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc
        '''
        if self.size_accumulator is not None:
            self.size_accumulator.key_replace(stats_dict)


class MarkovTransformAccumulatorFactory(object):

    def __init__(self, num_vals, len_factory, keys):
        self.len_factory = len_factory
        self.keys = keys
        self.num_vals = num_vals

    def make(self):
        if self.len_factory is None:
            return MarkovTransformAccumulator(self.num_vals, size_acc=None, keys=self.keys)
        else:
            return MarkovTransformAccumulator(self.num_vals, size_acc=self.len_factory.make(), keys=self.keys)


class MarkovTransformEstimator(ParameterEstimator):

    def __init__(self, num_vals, alpha=0.0, len_estimator=None, suff_stat=None, pseudo_count=None, keys=(None, None)):
        self.keys = keys
        self.len_estimator=len_estimator
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.num_vals = num_vals
        self.alpha = alpha

    def accumulatorFactory(self):
        return MarkovTransformAccumulatorFactory(self.num_vals, self.len_estimator, self.keys)

    def estimate(self, nobs, suff_stat):

        init_count, trans_count, size_stats = suff_stat

        if self.len_estimator is not None:
            len_dist = self.len_estimator.estimate(nobs, size_stats)
        else:
            len_dist = None
        trans_count = trans_count.tocsc()
        row_sum = trans_count * csc_matrix(np.ones((trans_count.shape[1],1)))
        
        init_prob  = init_count / np.sum(init_count)
        trans_prob = trans_count.multiply(row_sum.power(-1))

        return MarkovTransformDistribution(init_prob, trans_prob, self.alpha, len_dist)

