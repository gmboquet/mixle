"""Finite-state POMDP filtering, simulation, and fitting helpers.

The model represents action-conditioned transitions and observations, supports
belief updates and sequence likelihoods, and includes a Baum-Welch style fitting
path for known-action observation sequences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.models._result import FitResult


@dataclass
class PartiallyObservableMarkovDecisionProcessFilterResult:
    """Belief trajectories, log likelihood, and predictive observation terms."""

    beliefs: np.ndarray
    log_likelihood: float
    predictive_observation_probs: np.ndarray


@dataclass
class PartiallyObservableMarkovDecisionProcessFitResult(FitResult["PartiallyObservableMarkovDecisionProcessModel"]):
    """Baum-Welch style fit result for known-action PartiallyObservableMarkovDecisionProcess sequences."""


class PartiallyObservableMarkovDecisionProcessModel:
    """Finite-state PartiallyObservableMarkovDecisionProcess with action-conditioned transitions and observations.

    ``transition[a, i, j]`` is P(S_t=j | S_{t-1}=i, A_t=a).
    ``observation[a, j, o]`` is P(O_t=o | S_t=j, A_t=a).
    """

    def __init__(
        self,
        transition: Any,
        observation: Any,
        initial_belief: Any | None = None,
        rewards: Any | None = None,
        name: str | None = None,
    ) -> None:
        self.transition = _as_stochastic_3d(transition, "transition")
        self.observation = _as_observation(observation, self.transition.shape[0], self.transition.shape[2])
        self.num_actions = int(self.transition.shape[0])
        self.num_states = int(self.transition.shape[1])
        self.num_observations = int(self.observation.shape[2])
        if initial_belief is None:
            self.initial_belief = np.full(self.num_states, 1.0 / self.num_states, dtype=np.float64)
        else:
            self.initial_belief = _as_simplex(initial_belief, self.num_states, "initial_belief")
        self.rewards = None if rewards is None else np.asarray(rewards, dtype=np.float64)
        if self.rewards is not None and self.rewards.shape != (self.num_actions, self.num_states):
            raise ValueError("rewards must have shape (num_actions, num_states).")
        self.name = name

    def __str__(self) -> str:
        return (
            "PartiallyObservableMarkovDecisionProcessModel(num_states=%d, num_actions=%d, num_observations=%d, name=%r)"
            % (
                self.num_states,
                self.num_actions,
                self.num_observations,
                self.name,
            )
        )

    def belief_update(self, belief: Any, action: int, observation: int) -> tuple[np.ndarray, float]:
        """Update a belief after taking ``action`` and seeing ``observation``."""
        b = _as_simplex(belief, self.num_states, "belief")
        self._check_action_observation(action, observation)
        predictive = b.dot(self.transition[int(action)])
        obs_probs = self.observation[int(action), :, int(observation)]
        unnorm = predictive * obs_probs
        evidence = float(unnorm.sum())
        if evidence <= 0.0:
            return np.full(self.num_states, 1.0 / self.num_states, dtype=np.float64), 0.0
        return unnorm / evidence, evidence

    def filter(
        self, actions: Sequence[int], observations: Sequence[int], initial_belief: Any | None = None
    ) -> PartiallyObservableMarkovDecisionProcessFilterResult:
        """Run the forward filter and return posterior beliefs and log likelihood."""
        actions = np.asarray(actions, dtype=np.int64)
        observations = np.asarray(observations, dtype=np.int64)
        if actions.shape != observations.shape:
            raise ValueError("actions and observations must have the same length.")
        belief = (
            self.initial_belief
            if initial_belief is None
            else _as_simplex(initial_belief, self.num_states, "initial_belief")
        )
        beliefs = np.empty((len(actions), self.num_states), dtype=np.float64)
        pred_probs = np.empty(len(actions), dtype=np.float64)
        log_likelihood = 0.0
        for t, (a, o) in enumerate(zip(actions, observations)):
            belief, evidence = self.belief_update(belief, int(a), int(o))
            beliefs[t] = belief
            pred_probs[t] = evidence
            log_likelihood += np.log(max(evidence, 1.0e-300))
        return PartiallyObservableMarkovDecisionProcessFilterResult(beliefs, float(log_likelihood), pred_probs)

    def sequence_log_likelihood(
        self, actions: Sequence[int], observations: Sequence[int], initial_belief: Any | None = None
    ) -> float:
        """Return log P(observations | actions, model)."""
        return self.filter(actions, observations, initial_belief).log_likelihood

    def forward_backward(
        self, actions: Sequence[int], observations: Sequence[int], initial_belief: Any | None = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return state marginals, transition marginals, and sequence log likelihood."""
        actions = np.asarray(actions, dtype=np.int64)
        observations = np.asarray(observations, dtype=np.int64)
        if actions.shape != observations.shape:
            raise ValueError("actions and observations must have the same length.")
        if len(actions) == 0:
            return (np.zeros((0, self.num_states)), np.zeros((0, self.num_states, self.num_states)), 0.0)
        init = (
            self.initial_belief
            if initial_belief is None
            else _as_simplex(initial_belief, self.num_states, "initial_belief")
        )
        alpha, scales = self._forward_scaled(actions, observations, init)
        beta = self._backward_scaled(actions, observations, scales)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = self._transition_marginals(actions, observations, init, alpha, beta)
        return gamma, xi, float(np.sum(np.log(np.maximum(scales, 1.0e-300))))

    def predict_observation(self, belief: Any, action: int) -> np.ndarray:
        """Return P(O_t | belief, action) before observing O_t."""
        b = _as_simplex(belief, self.num_states, "belief")
        self._check_action(action)
        predictive = b.dot(self.transition[int(action)])
        return predictive.dot(self.observation[int(action)])

    def expected_reward(self, belief: Any, action: int) -> float:
        """Return E[R | belief, action] when rewards were supplied."""
        if self.rewards is None:
            raise ValueError("rewards were not supplied.")
        b = _as_simplex(belief, self.num_states, "belief")
        self._check_action(action)
        return float(np.dot(b, self.rewards[int(action)]))

    def sample(
        self, actions: Sequence[int], seed: int | None = None, initial_belief: Any | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample latent states and observations for a fixed action sequence."""
        rng = np.random.RandomState(seed)
        actions = np.asarray(actions, dtype=np.int64)
        belief = (
            self.initial_belief
            if initial_belief is None
            else _as_simplex(initial_belief, self.num_states, "initial_belief")
        )
        state = int(rng.choice(self.num_states, p=belief))
        states = np.empty(len(actions), dtype=np.int64)
        observations = np.empty(len(actions), dtype=np.int64)
        for t, action in enumerate(actions):
            self._check_action(int(action))
            state = int(rng.choice(self.num_states, p=self.transition[int(action), state]))
            obs = int(rng.choice(self.num_observations, p=self.observation[int(action), state]))
            states[t] = state
            observations[t] = obs
        return states, observations

    def _forward_scaled(
        self, actions: np.ndarray, observations: np.ndarray, initial_belief: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = np.empty((len(actions), self.num_states), dtype=np.float64)
        scales = np.empty(len(actions), dtype=np.float64)
        prev = initial_belief
        for t, (a, o) in enumerate(zip(actions, observations)):
            self._check_action_observation(int(a), int(o))
            row = prev.dot(self.transition[int(a)]) * self.observation[int(a), :, int(o)]
            scale = float(row.sum())
            scales[t] = scale
            alpha[t] = row / scale if scale > 0.0 else 1.0 / self.num_states
            prev = alpha[t]
        return alpha, scales

    def _backward_scaled(self, actions: np.ndarray, observations: np.ndarray, scales: np.ndarray) -> np.ndarray:
        beta = np.ones((len(actions), self.num_states), dtype=np.float64)
        for t in range(len(actions) - 2, -1, -1):
            a = int(actions[t + 1])
            o = int(observations[t + 1])
            beta[t] = self.transition[a].dot(self.observation[a, :, o] * beta[t + 1])
            beta[t] /= max(scales[t + 1], 1.0e-300)
        return beta

    def _transition_marginals(
        self,
        actions: np.ndarray,
        observations: np.ndarray,
        initial_belief: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        xi = np.empty((len(actions), self.num_states, self.num_states), dtype=np.float64)
        for t, (a, o) in enumerate(zip(actions, observations)):
            prev = initial_belief if t == 0 else alpha[t - 1]
            mat = prev[:, None] * self.transition[int(a)] * (self.observation[int(a), :, int(o)] * beta[t])[None, :]
            total = mat.sum()
            xi[t] = mat / total if total > 0.0 else 1.0 / (self.num_states * self.num_states)
        return xi

    def _check_action(self, action: int) -> None:
        if action < 0 or action >= self.num_actions:
            raise ValueError("action index out of range.")

    def _check_action_observation(self, action: int, observation: int) -> None:
        self._check_action(action)
        if observation < 0 or observation >= self.num_observations:
            raise ValueError("observation index out of range.")


def baum_welch_pomdp(
    sequences: Sequence[tuple[Sequence[int], Sequence[int]]],
    num_states: int,
    num_actions: int,
    num_observations: int,
    initial_model: PartiallyObservableMarkovDecisionProcessModel | None = None,
    max_its: int = 50,
    tol: float | None = 1.0e-8,
    pseudo_count: float = 1.0e-3,
    seed: int | None = None,
) -> PartiallyObservableMarkovDecisionProcessFitResult:
    """Fit a known-action finite PartiallyObservableMarkovDecisionProcess by Baum-Welch/EM."""
    if num_states <= 0 or num_actions <= 0 or num_observations <= 0:
        raise ValueError("state, action, and observation counts must be positive.")
    if len(sequences) == 0:
        raise ValueError("at least one sequence is required.")
    if pseudo_count < 0.0:
        raise ValueError("pseudo_count must be non-negative.")
    rng = np.random.RandomState(seed)
    if initial_model is None:
        model = _random_pomdp(num_states, num_actions, num_observations, rng)
    else:
        model = initial_model
    history: list[float] = []
    for _ in range(max(1, int(max_its))):
        init_counts = np.full(num_states, pseudo_count, dtype=np.float64)
        trans_counts = np.full((num_actions, num_states, num_states), pseudo_count, dtype=np.float64)
        obs_counts = np.full((num_actions, num_states, num_observations), pseudo_count, dtype=np.float64)
        ll = 0.0
        for actions, observations in sequences:
            actions_arr = np.asarray(actions, dtype=np.int64)
            obs_arr = np.asarray(observations, dtype=np.int64)
            gamma, xi, seq_ll = model.forward_backward(actions_arr, obs_arr)
            ll += seq_ll
            if len(actions_arr) == 0:
                continue
            init_counts += xi[0].sum(axis=1)
            for t, action in enumerate(actions_arr):
                trans_counts[int(action)] += xi[t]
                obs_counts[int(action), :, int(obs_arr[t])] += gamma[t]
        transition = _normalize_last_axis(trans_counts)
        observation = _normalize_last_axis(obs_counts)
        initial = init_counts / init_counts.sum()
        model = PartiallyObservableMarkovDecisionProcessModel(
            transition, observation, initial_belief=initial, name=model.name
        )
        history.append(float(ll))
        if len(history) > 1 and tol is not None and abs(history[-1] - history[-2]) < tol:
            break
    return PartiallyObservableMarkovDecisionProcessFitResult(model, history)


def _random_pomdp(
    num_states: int, num_actions: int, num_observations: int, rng: np.random.RandomState
) -> PartiallyObservableMarkovDecisionProcessModel:
    transition = rng.dirichlet(np.ones(num_states), size=(num_actions, num_states))
    observation = rng.dirichlet(np.ones(num_observations), size=(num_actions, num_states))
    initial = rng.dirichlet(np.ones(num_states))
    return PartiallyObservableMarkovDecisionProcessModel(transition, observation, initial_belief=initial)


def _as_stochastic_3d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("%s must be a three-dimensional array." % name)
    if arr.shape[1] != arr.shape[2]:
        raise ValueError("%s must have shape (actions, states, states)." % name)
    return _check_stochastic(arr, name)


def _as_observation(x: Any, num_actions: int, num_states: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] != num_states:
            raise ValueError("two-dimensional observation matrix must have shape (states, observations).")
        arr = np.broadcast_to(arr[None, :, :], (num_actions, arr.shape[0], arr.shape[1])).copy()
    if arr.ndim != 3 or arr.shape[0] != num_actions or arr.shape[1] != num_states:
        raise ValueError("observation must have shape (actions, states, observations).")
    return _check_stochastic(arr, "observation")


def _check_stochastic(arr: np.ndarray, name: str) -> np.ndarray:
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("%s probabilities must be finite and non-negative." % name)
    totals = arr.sum(axis=-1)
    if np.any(totals <= 0.0):
        raise ValueError("%s rows must have positive mass." % name)
    if not np.allclose(totals, 1.0):
        raise ValueError("%s rows must sum to one." % name)
    return arr


def _as_simplex(x: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != size:
        raise ValueError("%s must have length %d." % (name, size))
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("%s must contain finite non-negative values." % name)
    total = arr.sum()
    if total <= 0.0:
        raise ValueError("%s must have positive mass." % name)
    return arr / total


def _normalize_last_axis(x: np.ndarray) -> np.ndarray:
    totals = x.sum(axis=-1, keepdims=True)
    return np.divide(x, totals, out=np.zeros_like(x), where=totals > 0.0)
