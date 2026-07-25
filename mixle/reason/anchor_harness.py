"""Measured two-modality scientific-reasoning anchor.

The harness is deliberately lightweight enough for a focused release check,
but its evaluation boundary is real:

* independent training, calibration, and test splits;
* gravity and geochemistry both enter the inferred grade posterior;
* all coefficients, residual scales, and the gravity-only baseline are learned
  from training data rather than copied from the hidden generator;
* calibration and abstention thresholds are selected without test outcomes;
* abstained predictions are actually withheld and selective risk is reported;
* receiver-specific projections operate on each inferred test posterior.

Bootstrap fits create an empirical posterior over the two learned stages
``gravity -> density`` and ``(density, geochemistry) -> grade``. Component
variance propagates both stages' residual uncertainty. This is not a claim that
the synthetic task is a frontier application; it is a fast, falsifiable
integration contract for the reasoning surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import beta, norm

from mixle.reason.modality import ModalityGraph, ModalityView
from mixle.reason.task_projection import TaskReadout, read_out, task_sufficient_projection
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

_LITHOLOGIES = ("basalt", "shale", "granite")
_BOOTSTRAPS = 8
_INTERVAL_LEVEL = 0.90
_CALIBRATION_TOLERANCE = 0.10


@dataclass(frozen=True)
class _Dataset:
    gravity: np.ndarray
    geochemistry: np.ndarray
    density: np.ndarray
    grade: np.ndarray

    def __len__(self) -> int:
        return len(self.gravity)


@dataclass(frozen=True)
class _BootstrapModel:
    density_coef: np.ndarray
    density_variance: float
    grade_coef: np.ndarray
    grade_variance: np.ndarray
    baseline_coef: np.ndarray
    baseline_variance: float


@dataclass
class AnchorHarnessReport:
    """Measured train/calibration/test report for the anchor task."""

    modalities: list[str]
    hop_names: list[str]
    coverage_by_hop: dict[int, dict[str, float]]
    abstained_site_ids: list[int]
    abstain_rate: float
    answer_rate: float
    accepted_mae: float
    all_answer_mae: float
    driller_projection_components: int
    scout_projection_components: int
    driller_readout: str
    scout_readout: str
    projection_decision_agreement: float
    gravity_only_mae: float
    both_modalities_mae: float
    gravity_ablation_penalty: float
    test_coverage: float
    coverage_target: float
    coverage_lower_bound: float
    walk_is_calibrated: bool
    training_rows: int
    calibration_rows: int
    test_rows: int
    notes: list[str] = field(default_factory=list)

    @property
    def walk_mae(self) -> float:
        """Compatibility name for the two-stage, two-modality posterior mean MAE."""
        return self.both_modalities_mae

    def summary(self) -> str:
        """Render the measured evaluation boundary and results."""
        return "\n".join(
            [
                f"splits: train={self.training_rows}, calibration={self.calibration_rows}, test={self.test_rows}",
                f"modalities used in inference: {self.modalities}",
                f"learned stages: {self.hop_names}",
                f"test joint coverage={self.test_coverage:.3f}, lower bound={self.coverage_lower_bound:.3f}, "
                f"target={self.coverage_target:.3f}",
                f"answer rate={self.answer_rate:.3f}, accepted MAE={self.accepted_mae:.3f}, "
                f"all-answer MAE={self.all_answer_mae:.3f}",
                f"two-modality MAE={self.both_modalities_mae:.3f}, "
                f"learned gravity-only ablation MAE={self.gravity_only_mae:.3f}",
                f"projection decision agreement={self.projection_decision_agreement:.3f}",
            ]
        )


def run_anchor_harness(
    *,
    n_train: int = 1200,
    n_calibration: int | None = None,
    n_test: int = 200,
    seed: int = 0,
) -> AnchorHarnessReport:
    """Fit on training data, calibrate on a disjoint split, and evaluate once."""
    n_train = _positive_count(n_train, "n_train", minimum=40)
    n_test = _positive_count(n_test, "n_test", minimum=30)
    calibration_rows = max(30, n_test // 2) if n_calibration is None else _positive_count(
        n_calibration,
        "n_calibration",
        minimum=30,
    )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    rng = np.random.RandomState(int(seed))
    train = _generate_dataset(n_train, rng)
    calibration = _generate_dataset(calibration_rows, rng)
    test = _generate_dataset(n_test, rng)

    models = _fit_bootstraps(train, seed=int(seed))
    calibration_posteriors = [_grade_posterior(models, calibration, index) for index in range(len(calibration))]
    calibration_scale = _calibration_scale(calibration_posteriors, calibration.grade)
    calibration_posteriors = [_scale_posterior_variance(posterior, calibration_scale) for posterior in calibration_posteriors]
    test_posteriors = [
        _scale_posterior_variance(_grade_posterior(models, test, index), calibration_scale)
        for index in range(len(test))
    ]

    test_means = np.asarray([_mixture_mean(posterior) for posterior in test_posteriors])
    test_stds = np.asarray([_mixture_std(posterior) for posterior in test_posteriors])
    all_errors = np.abs(test_means - test.grade)
    both_modalities_mae = float(np.mean(all_errors))

    baseline_means = np.asarray(
        [np.mean([_predict_baseline(model, test.gravity[index]) for model in models]) for index in range(len(test))]
    )
    gravity_only_mae = float(np.mean(np.abs(baseline_means - test.grade)))

    calibration_stds = np.asarray([_mixture_std(posterior) for posterior in calibration_posteriors])
    abstain_threshold = float(np.quantile(calibration_stds, 0.90))
    accepted = test_stds <= abstain_threshold
    if not np.any(accepted):
        raise RuntimeError("calibration-derived abstention policy rejected every test case.")
    abstained_site_ids = np.flatnonzero(~accepted).astype(int).tolist()
    accepted_mae = float(np.mean(all_errors[accepted]))

    coverage_flags = np.asarray(
        [
            _posterior_interval(posterior, _INTERVAL_LEVEL, seed=int(seed) + index)[0]
            <= test.grade[index]
            <= _posterior_interval(posterior, _INTERVAL_LEVEL, seed=int(seed) + index)[1]
            for index, posterior in enumerate(test_posteriors)
        ],
        dtype=bool,
    )
    coverage = float(np.mean(coverage_flags))
    coverage_lower = _binomial_lower_bound(int(np.sum(coverage_flags)), len(coverage_flags))
    coverage_target = _INTERVAL_LEVEL
    calibrated = coverage_lower >= coverage_target - _CALIBRATION_TOLERANCE

    density_coverage = _density_coverage(models, test)
    coverage_by_hop = {
        1: {
            "coverage": density_coverage,
            "joint_coverage_target": coverage_target,
            "calibrated": density_coverage >= coverage_target - _CALIBRATION_TOLERANCE,
        },
        2: {
            "coverage": coverage,
            "coverage_lower_bound": coverage_lower,
            "joint_coverage_target": coverage_target,
            "calibrated": calibrated,
        },
    }

    driller_task = TaskReadout(
        "precise_grade",
        lambda mean: float(mean[0]),
    )
    scout_task = TaskReadout(
        "positive_grade",
        lambda mean: bool(mean[0] > 0.0),
        projection_error=_binary_decision_projection_error,
        max_projection_error=0.05,
    )
    projection_matches = []
    driller_components = []
    scout_components = []
    for posterior in test_posteriors:
        driller = task_sufficient_projection(posterior, driller_task)
        scout = task_sufficient_projection(posterior, scout_task)
        probe = np.array([_mixture_mean(posterior)])
        projection_matches.append(read_out(posterior, scout_task, probe) == read_out(scout, scout_task, probe))
        driller_components.append(driller.num_components)
        scout_components.append(scout.num_components)
    representative = int(np.argmin(np.abs(test_means - np.median(test_means))))
    representative_posterior = test_posteriors[representative]
    representative_probe = np.array([test_means[representative]])
    driller_view = task_sufficient_projection(representative_posterior, driller_task)
    scout_view = task_sufficient_projection(representative_posterior, scout_task)

    graph = ModalityGraph()
    graph.add(ModalityView("geochemistry", CategoricalDistribution(dict.fromkeys(_LITHOLOGIES, 1.0 / 3.0))))
    graph.add(ModalityView("gravity", GaussianDistribution(0.0, 1.0)))

    return AnchorHarnessReport(
        modalities=graph.modalities(),
        hop_names=["gravity_to_density", "density_and_geochemistry_to_grade"],
        coverage_by_hop=coverage_by_hop,
        abstained_site_ids=abstained_site_ids,
        abstain_rate=len(abstained_site_ids) / len(test),
        answer_rate=float(np.mean(accepted)),
        accepted_mae=accepted_mae,
        all_answer_mae=both_modalities_mae,
        driller_projection_components=int(round(float(np.mean(driller_components)))),
        scout_projection_components=int(round(float(np.mean(scout_components)))),
        driller_readout=str(read_out(driller_view, driller_task, representative_probe)),
        scout_readout=str(read_out(scout_view, scout_task, representative_probe)),
        projection_decision_agreement=float(np.mean(projection_matches)),
        gravity_only_mae=gravity_only_mae,
        both_modalities_mae=both_modalities_mae,
        gravity_ablation_penalty=gravity_only_mae - both_modalities_mae,
        test_coverage=coverage,
        coverage_target=coverage_target,
        coverage_lower_bound=coverage_lower,
        walk_is_calibrated=calibrated,
        training_rows=len(train),
        calibration_rows=len(calibration),
        test_rows=len(test),
        notes=[
            "All fitted coefficients and both baselines use training data only.",
            "Variance scaling and the abstention threshold use calibration data only.",
            "The test split is used once for reported coverage, risk, ablation, and projection metrics.",
        ],
    )


def _generate_dataset(n: int, rng: np.random.RandomState) -> _Dataset:
    probabilities = np.array([0.55, 0.30, 0.15])
    lithology_index = rng.choice(len(_LITHOLOGIES), size=n, p=probabilities)
    density = rng.normal(0.0, 1.0, size=n) + np.array([-0.4, 0.2, 0.8])[lithology_index]
    gravity = density + rng.normal(0.0, 0.35, size=n)
    grade_noise = np.array([0.25, 0.45, 1.10])[lithology_index]
    grade = 0.75 * density + np.array([-1.5, 0.25, 2.0])[lithology_index] + rng.normal(0.0, grade_noise)
    geochemistry = np.asarray(_LITHOLOGIES, dtype=object)[lithology_index]
    return _Dataset(gravity=gravity, geochemistry=geochemistry, density=density, grade=grade)


def _fit_bootstraps(train: _Dataset, *, seed: int) -> tuple[_BootstrapModel, ...]:
    rng = np.random.RandomState(seed + 1000)
    models = []
    for _ in range(_BOOTSTRAPS):
        indices = rng.randint(0, len(train), size=len(train))
        gravity = train.gravity[indices]
        density = train.density[indices]
        grade = train.grade[indices]
        geochemistry = train.geochemistry[indices]

        density_design = np.column_stack((np.ones(len(indices)), gravity))
        density_coef = np.linalg.lstsq(density_design, density, rcond=None)[0]
        density_residual = density - density_design @ density_coef

        grade_design = _grade_design(density, geochemistry)
        grade_coef = np.linalg.lstsq(grade_design, grade, rcond=None)[0]
        grade_residual = grade - grade_design @ grade_coef
        global_grade_variance = _residual_variance(grade_residual, grade_design.shape[1])
        grade_variance = np.asarray(
            [
                _residual_variance(grade_residual[geochemistry == lithology], 1)
                if np.sum(geochemistry == lithology) >= 3
                else global_grade_variance
                for lithology in _LITHOLOGIES
            ]
        )

        baseline_design = np.column_stack((np.ones(len(indices)), gravity))
        baseline_coef = np.linalg.lstsq(baseline_design, grade, rcond=None)[0]
        baseline_residual = grade - baseline_design @ baseline_coef
        models.append(
            _BootstrapModel(
                density_coef=density_coef,
                density_variance=_residual_variance(density_residual, density_design.shape[1]),
                grade_coef=grade_coef,
                grade_variance=grade_variance,
                baseline_coef=baseline_coef,
                baseline_variance=_residual_variance(baseline_residual, baseline_design.shape[1]),
            )
        )
    return tuple(models)


def _grade_design(density: np.ndarray, geochemistry: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.ones(len(density)),
            density,
            (geochemistry == _LITHOLOGIES[1]).astype(float),
            (geochemistry == _LITHOLOGIES[2]).astype(float),
        )
    )


def _grade_posterior(models: tuple[_BootstrapModel, ...], dataset: _Dataset, index: int) -> GaussianMixtureDistribution:
    means = []
    variances = []
    for model in models:
        density_mean = float(model.density_coef @ np.array([1.0, dataset.gravity[index]]))
        grade_row = _grade_design(np.array([density_mean]), np.array([dataset.geochemistry[index]], dtype=object))[0]
        grade_mean = float(model.grade_coef @ grade_row)
        lithology_index = _LITHOLOGIES.index(str(dataset.geochemistry[index]))
        propagated = float(
            model.grade_coef[1] ** 2 * model.density_variance + model.grade_variance[lithology_index]
        )
        means.append([grade_mean])
        variances.append([[max(propagated, np.finfo(float).eps)]])
    return GaussianMixtureDistribution(
        np.asarray(means),
        np.asarray(variances),
        np.full(len(models), 1.0 / len(models)),
    )


def _scale_posterior_variance(
    posterior: GaussianMixtureDistribution,
    scale: float,
) -> GaussianMixtureDistribution:
    return GaussianMixtureDistribution(posterior.mu.copy(), posterior.sig2 * scale**2, posterior.w.copy())


def _calibration_scale(posteriors: list[GaussianMixtureDistribution], truth: np.ndarray) -> float:
    means = np.asarray([_mixture_mean(posterior) for posterior in posteriors])
    stds = np.maximum(np.asarray([_mixture_std(posterior) for posterior in posteriors]), 1e-12)
    standardized = np.abs(truth - means) / stds
    target_quantile = float(norm.ppf(0.5 + _INTERVAL_LEVEL / 2.0))
    return max(float(np.quantile(standardized, _INTERVAL_LEVEL)) / target_quantile, 1e-6)


def _mixture_mean(posterior: GaussianMixtureDistribution) -> float:
    return float(np.sum(posterior.w * posterior.mu[:, 0]))


def _mixture_std(posterior: GaussianMixtureDistribution) -> float:
    mean = _mixture_mean(posterior)
    second = float(np.sum(posterior.w * (posterior.sig2[:, 0, 0] + posterior.mu[:, 0] ** 2)))
    return float(np.sqrt(max(second - mean**2, 0.0)))


def _posterior_interval(
    posterior: GaussianMixtureDistribution,
    level: float,
    *,
    seed: int,
) -> tuple[float, float]:
    draws = np.asarray(posterior.sampler(seed=seed).sample(600), dtype=float).reshape(-1)
    alpha = 1.0 - level
    return float(np.quantile(draws, alpha / 2.0)), float(np.quantile(draws, 1.0 - alpha / 2.0))


def _predict_baseline(model: _BootstrapModel, gravity: float) -> float:
    return float(model.baseline_coef @ np.array([1.0, gravity]))


def _density_coverage(models: tuple[_BootstrapModel, ...], test: _Dataset) -> float:
    covered = []
    for index in range(len(test)):
        means = np.asarray([model.density_coef @ np.array([1.0, test.gravity[index]]) for model in models])
        variance = float(np.mean([model.density_variance for model in models]) + np.var(means))
        center = float(np.mean(means))
        half_width = float(norm.ppf(0.5 + _INTERVAL_LEVEL / 2.0) * np.sqrt(variance))
        covered.append(center - half_width <= test.density[index] <= center + half_width)
    return float(np.mean(covered))


def _binary_decision_projection_error(components: tuple[Any, ...], weights: np.ndarray, merged: Any) -> float:
    def positive_probability(component: Any) -> float:
        mean = float(np.asarray(component.mu).reshape(-1)[0])
        variance = (
            float(np.asarray(component.covar)[0, 0])
            if hasattr(component, "covar")
            else float(component.sigma2)
        )
        return float(norm.cdf(mean / np.sqrt(variance)))

    original = float(
        sum(float(weight) * positive_probability(component) for weight, component in zip(weights, components))
    )
    return abs(original - positive_probability(merged))


def _residual_variance(residual: np.ndarray, parameters: int) -> float:
    dof = max(len(residual) - parameters, 1)
    return max(float(np.sum(np.square(residual)) / dof), np.finfo(float).eps)


def _binomial_lower_bound(hits: int, total: int) -> float:
    return 0.0 if hits == 0 else float(beta.ppf(0.05, hits, total - hits + 1))


def _positive_count(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)
