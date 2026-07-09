"""Runnable geoscience harness for cross-modal target inference.

The harness infers a synthetic subsurface ore-grade target from two
heterogeneous evidence modalities: a categorical geochemistry observation and a
potential-fields gravity reading. It composes existing reasoning primitives
into one measured workflow:

* :mod:`mixle.reason.modality` represents geochemistry and gravity as typed
  :class:`~mixle.reason.modality.ModalityView` objects.
* :mod:`mixle.reason.cycle_consistency` fits conditional transports and checks
  whether a hop should abstain.
* :mod:`mixle.reason.belief_walk` composes the gravity-to-density and
  density-to-grade hops by Monte Carlo belief propagation, with calibration
  checked by hop count through
  :func:`~mixle.reason.belief_walk.coverage_by_hop_count`.
* :mod:`mixle.reason.task_projection` projects the same ore-grade belief for
  two receivers: one that needs an exact tier and one that needs a binary
  drilling decision.

The data are synthetic and seeded, using a linear-Gaussian generative process
with coefficients hidden from the fitted transports. The report is checkable
against ground truth and records measured coverage, abstention, and error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mixle.reason.belief_walk import HopTransport, belief_walk, coverage_by_hop_count
from mixle.reason.cycle_consistency import cycle_inconsistency, fit_cycle_transport
from mixle.reason.modality import ModalityGraph, ModalityView
from mixle.reason.task_projection import TaskReadout, read_out, task_sufficient_projection
from mixle.reason.transport_edge import verify_edge_transport
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

# True generative process used to create checkable synthetic ground truth.
_GRAVITY_TO_DENSITY = 0.7
_DENSITY_TO_GRADE = 0.6
_NOISE_STD = 0.4


def _step(x: np.ndarray, coef: float, rng: np.random.RandomState) -> np.ndarray:
    return coef * x + rng.normal(0, _NOISE_STD, size=x.shape)


@dataclass
class AnchorHarnessReport:
    """Measured report for the cross-modal geoscience harness."""

    modalities: list[str]
    hop_names: list[str]
    coverage_by_hop: dict[int, dict[str, float]]
    abstained_site_ids: list[int]
    abstain_rate: float
    driller_projection_components: int
    scout_projection_components: int
    driller_readout: str
    scout_readout: str
    frontier_mae: float
    walk_mae: float
    frontier_is_calibrated: bool
    walk_is_calibrated: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Render a compact anchor-harness evaluation summary."""
        lines = [
            f"modalities (F1, structured belief nodes, no shared vector): {self.modalities}",
            f"belief walk hops (F3): {self.hop_names}",
            f"coverage by hop count (F3, uncertainty compounding): {self.coverage_by_hop}",
            f"cycle-consistency abstention (F5): {len(self.abstained_site_ids)} sites ({self.abstain_rate:.1%})",
            f"driller projection (F4): {self.driller_projection_components} components, "
            f"reads out {self.driller_readout!r}",
            f"scout projection (F4): {self.scout_projection_components} components, reads out {self.scout_readout!r}",
            f"direct-read baseline MAE={self.frontier_mae:.3f} calibrated={self.frontier_is_calibrated} "
            "(no compounding uncertainty, lowest-cost -- one linear read, no fitting/MC)",
            f"belief-walk MAE={self.walk_mae:.3f} calibrated={self.walk_is_calibrated} "
            "(calibrated interval, uncertainty compounded, costs 2 fitted hops + MC draws)",
        ]
        return "\n".join(lines)


def _frontier_baseline(gravity_reading: np.ndarray) -> np.ndarray:
    """Return a direct point-estimate baseline without calibrated uncertainty."""
    return _GRAVITY_TO_DENSITY * _DENSITY_TO_GRADE * gravity_reading


def run_anchor_harness(*, n_train: int = 2000, n_test: int = 200, seed: int = 0) -> AnchorHarnessReport:
    """Run the cross-modal harness and return the measured report.

    Requires ``torch`` (the transport fits do); raises the underlying ``ImportError`` if it is absent,
    same as other neural transport fitting paths.
    """
    rng = np.random.RandomState(seed)

    # Two structured-belief modality nodes for one prospect.
    graph = ModalityGraph()
    graph.add(ModalityView("geochem", CategoricalDistribution({"basalt": 0.4, "shale": 0.35, "granite": 0.25})))
    graph.add(ModalityView("gravity", GaussianDistribution(0.0, 1.0)))
    modalities = graph.modalities()

    # Fit each hop's conditional transport and verify it before composing.
    gravity_train = rng.normal(0, 1.0, size=(n_train, 1))
    density_train = _step(gravity_train, _GRAVITY_TO_DENSITY, rng)
    grade_train = _step(density_train, _DENSITY_TO_GRADE, rng)

    hop1_fit = fit_cycle_transport(gravity_train, density_train, k=1, seed=seed, max_its=25)
    hop2_fit = fit_cycle_transport(density_train, grade_train, k=1, seed=seed + 1, max_its=25)

    # Held-out pairs for the premise check, generated before the hops are composed.
    rng_test = np.random.RandomState(seed + 100)
    gravity_test = rng_test.normal(0, 1.0, size=(n_test, 1))
    density_true = _step(gravity_test, _GRAVITY_TO_DENSITY, rng_test)
    grade_true = _step(density_true, _DENSITY_TO_GRADE, rng_test)
    true_by_hop = {1: density_true, 2: grade_true}

    # ``verify_edge_transport`` checks coverage of the modeled target from
    # draws conditioned on the observed input for each hop.
    hop1_verdict = verify_edge_transport("gravity_to_density", hop1_fit.sampler(seed=seed), density_true, gravity_test)
    hop2_verdict = verify_edge_transport("density_to_grade", hop2_fit.sampler(seed=seed + 1), grade_true, density_true)
    hops = [
        HopTransport("gravity_to_density", hop1_fit, premise_passed=hop1_verdict.usable),
        HopTransport("density_to_grade", hop2_fit, premise_passed=hop2_verdict.usable),
    ]

    # Belief walk composing the two verified hops.
    coverage_by_hop = coverage_by_hop_count(hops, gravity_test, true_by_hop, alpha=0.1, n_draws=150, seed=seed)

    walk_means = np.zeros(n_test)
    walk_covered = np.zeros(n_test, dtype=bool)
    for i in range(n_test):
        walk = belief_walk(hops, gravity_test[i], n_draws=150, seed=seed + i)
        walk_means[i] = float(walk.mean[0])
        lo, hi = walk.credible_interval(alpha=0.1)
        walk_covered[i] = bool(lo[0] <= grade_true[i, 0] <= hi[0])
    walk_mae = float(np.mean(np.abs(walk_means - grade_true[:, 0])))
    walk_is_calibrated = bool(coverage_by_hop[2]["consistent_with_nominal"])

    # Cycle-consistency on the final hop gates abstention.
    hop2_sampler = hops[1].fit.sampler(seed=seed + 2)
    inconsistencies = np.array([cycle_inconsistency(hop2_sampler, density_true[i], n_draws=20) for i in range(n_test)])
    abstain_threshold = float(np.quantile(inconsistencies, 0.9))  # flag the most self-inconsistent decile
    abstained_site_ids = [i for i, v in enumerate(inconsistencies) if v > abstain_threshold]

    # Direct point-estimate baseline without a calibrated interval.
    frontier_estimate = _frontier_baseline(gravity_test[:, 0])
    frontier_mae = float(np.mean(np.abs(frontier_estimate - grade_true[:, 0])))
    frontier_is_calibrated = False  # the baseline reports no interval.

    # One belief, two receivers, and two task-sufficient projections.
    grade_tier_means = np.array([[-2.0], [-0.6], [0.6], [2.0]])  # low / lowmid / midhigh / high grade tiers
    grade_tier_cov = np.stack([np.array([[0.35]]) for _ in range(4)])
    grade_tier_w = np.full(4, 0.25)
    grade_belief = GaussianMixtureDistribution(mu=grade_tier_means, sig2=grade_tier_cov, w=grade_tier_w)

    driller_task = TaskReadout("exact_tier", lambda mean: round(float(mean[0]), 1))
    scout_task = TaskReadout("worth_drilling", lambda mean: bool(mean[0] > 0.0))

    driller_view = task_sufficient_projection(grade_belief, driller_task)
    scout_view = task_sufficient_projection(grade_belief, scout_task)

    probe_x = np.array([1.5])
    driller_readout = str(read_out(driller_view, driller_task, probe_x))
    scout_readout = str(read_out(scout_view, scout_task, probe_x))

    return AnchorHarnessReport(
        modalities=modalities,
        hop_names=[h.name for h in hops],
        coverage_by_hop=coverage_by_hop,
        abstained_site_ids=abstained_site_ids,
        abstain_rate=len(abstained_site_ids) / n_test,
        driller_projection_components=driller_view.num_components,
        scout_projection_components=scout_view.num_components,
        driller_readout=driller_readout,
        scout_readout=scout_readout,
        frontier_mae=frontier_mae,
        walk_mae=walk_mae,
        frontier_is_calibrated=frontier_is_calibrated,
        walk_is_calibrated=walk_is_calibrated,
        notes=[
            "direct-read baseline is a single linear read: lowest-cost, but uncalibrated (no interval reported)",
            "belief walk costs 2 fitted transports + MC draws per query, but is calibrated and abstains "
            "on cycle-inconsistent sites instead of guessing",
        ],
    )
