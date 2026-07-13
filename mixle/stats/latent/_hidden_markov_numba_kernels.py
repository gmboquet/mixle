"""Shared numba Baum-Welch / forward kernels for the HMM family.

These forward-backward kernels are byte-for-byte identical across the HMM
variants (``hidden_markov`` and ``look_back_hmm``). They are consolidated here so
the numba JIT compiles and caches them once and the variants share the same
``cache=True`` object code.
"""

import math

import numpy as np

from mixle.utils.optional_deps import numba

# fastmath SUBSET shared by every kernel here: reassociation/contraction/approximations keep the
# SIMD wins, but ninf/nnan stay OFF -- ``numba_seq_log_density`` adds ``-np.inf`` to the
# log-likelihood on impossible observations, which full ``fastmath=True`` (LLVM ninf) declares
# undefined behavior and may fold away (the same policy as fused_codegen._njit).
_FASTMATH_SUBSET = {"reassoc", "contract", "arcp", "afn", "nsz"}


@numba.njit(
    "void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:])",
    parallel=True,
    fastmath=_FASTMATH_SUBSET,
    cache=True,
)
def numba_seq_log_density(num_states, tz, prob_mat, init_pvec, tran_mat, max_ll, next_alpha_mat, alpha_buff_mat, out):
    for n in numba.prange(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            out[n] = 0
            continue

        next_alpha = next_alpha_mat[n, :]
        alpha_buff = alpha_buff_mat[n, :]

        llsum = 0
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            next_alpha[i] = temp
            alpha_sum += temp

        llsum += math.log(alpha_sum) if alpha_sum > 0.0 else -np.inf  # guarded: math.log(0) raises off-numba
        llsum += max_ll[s0]
        if alpha_sum <= 0.0:  # impossible observation: log above gave -inf; clamp the divisor so the
            alpha_sum = 1.0  # recursion stays 0 (-> ll -inf) instead of 0/0 -> NaN

        for s in range(s0 + 1, s1):
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

            llsum += math.log(alpha_sum) if alpha_sum > 0.0 else -np.inf  # guarded: math.log(0) raises off-numba
            llsum += max_ll[s]
            if alpha_sum <= 0.0:  # impossible observation mid-sequence: keep ll -inf, avoid 0/0 -> NaN
                alpha_sum = 1.0

        out[n] = llsum


@numba.njit(
    "void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:], "
    "float64[:], float64[:,:])",
    cache=True,
)
def numba_baum_welch(
    num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc, beta_buff, xi_buff
):
    for n in range(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
        # alpha_sum = temp if temp > alpha_sum else alpha_sum
        if alpha_sum <= 0.0:  # impossible observation (zero emission in every state): keep alpha 0, avoid 0/0 -> NaN
            alpha_sum = 1.0
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):
            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
            if alpha_sum <= 0.0:  # impossible observation: keep alpha 0, avoid 0/0 -> NaN
                alpha_sum = 1.0
            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum

        for i in range(num_states):
            alpha_loc[s1 - 1, i] *= weight_loc

        beta_sum = 1
        # beta_sum = 1/num_states
        prev_beta = np.empty(num_states, dtype=np.float64)
        prev_beta.fill(1 / num_states)

        for s in range(s1 - 2, s0 - 1, -1):
            sp1 = s + 1

            if beta_sum <= 0.0:  # impossible observation: keep beta 0, avoid x/0 -> NaN/inf in the backward pass
                beta_sum = 1.0
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
                    temp *= alpha_loc[s, i]
                    xi_buff[i, j] = temp
                    xi_buff_sum += temp

                prev_beta[i] = temp_beta
                alpha_loc[s, i] *= temp_beta
                gamma_buff += alpha_loc[s, i]
                beta_sum += temp_beta
            # beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

            if gamma_buff > 0:
                gamma_buff = weight_loc / gamma_buff

            if xi_buff_sum > 0:
                xi_buff_sum = weight_loc / xi_buff_sum

            for i in range(num_states):
                alpha_loc[s, i] *= gamma_buff
                for j in range(num_states):
                    xi_acc[i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[i] += alpha_loc[s0, i]


@numba.njit(
    "void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], "
    "float64[:,:])",
    parallel=True,
    fastmath=_FASTMATH_SUBSET,
    cache=True,
)
def numba_baum_welch2(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc):
    for n in numba.prange(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        beta_buff = np.zeros(num_states, dtype=np.float64)
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
        # alpha_sum = temp if temp > alpha_sum else alpha_sum
        if alpha_sum <= 0.0:  # impossible observation (zero emission in every state): keep alpha 0, avoid 0/0 -> NaN
            alpha_sum = 1.0
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):
            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
            if alpha_sum <= 0.0:  # impossible observation: keep alpha 0, avoid 0/0 -> NaN
                alpha_sum = 1.0
            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum

        for i in range(num_states):
            alpha_loc[s1 - 1, i] *= weight_loc

        beta_sum = 1
        # beta_sum = 1/num_states
        prev_beta = np.empty(num_states, dtype=np.float64)
        prev_beta.fill(1 / num_states)

        for s in range(s1 - 2, s0 - 1, -1):
            sp1 = s + 1

            if beta_sum <= 0.0:  # impossible observation: keep beta 0, avoid x/0 -> NaN/inf in the backward pass
                beta_sum = 1.0
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
                    temp *= alpha_loc[s, i]
                    xi_buff[i, j] = temp
                    xi_buff_sum += temp

                prev_beta[i] = temp_beta
                alpha_loc[s, i] *= temp_beta
                gamma_buff += alpha_loc[s, i]
                beta_sum += temp_beta
            # beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

            if gamma_buff > 0:
                gamma_buff = weight_loc / gamma_buff

            if xi_buff_sum > 0:
                xi_buff_sum = weight_loc / xi_buff_sum

            for i in range(num_states):
                alpha_loc[s, i] *= gamma_buff
                for j in range(num_states):
                    xi_acc[n, i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[n, i] += alpha_loc[s0, i]


@numba.njit(
    "void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], "
    "float64[:,:])",
    parallel=True,
    fastmath=_FASTMATH_SUBSET,
    cache=True,
)
def numba_baum_welch_alphas(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc):
    for n in numba.prange(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        beta_buff = np.zeros(num_states, dtype=np.float64)
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
        # alpha_sum = temp if temp > alpha_sum else alpha_sum
        if alpha_sum <= 0.0:  # impossible observation (zero emission in every state): keep alpha 0, avoid 0/0 -> NaN
            alpha_sum = 1.0
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):
            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
            if alpha_sum <= 0.0:  # impossible observation: keep alpha 0, avoid 0/0 -> NaN
                alpha_sum = 1.0
            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum
