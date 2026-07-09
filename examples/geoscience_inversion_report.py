"""B7: Sense -> simulate -> invert -> report -- the track-M full-loop demo.

Domain framing lives ENTIRELY here (a toy geoscience seismic-inversion story), per the roadmap
card's own instruction: the library modules this example composes (``mixle.inference.scenario``,
``mixle.task.inverse``, ``mixle.reason.language_bridge``, ``mixle.task.calibrated_generator``) stay
domain-agnostic. Nothing below teaches mixle anything about geoscience -- it only teaches this
SCRIPT what "formation", "depth", and "amplitude" mean.

The story: a field crew logs three heterogeneous channels at each site -- formation TYPE (a text
label), a 3-station seismic AMPLITUDE array (the array modality), and SOURCE DEPTH (a scalar) -- and
mixle fits one joint over all three.

  1. **Sense.**  Synthetic multi-modal "field" records: (formation: str, amplitude: tuple[3xfloat],
     depth: float). ``learn_bayesian_network`` (M0's own substrate) discovers the DAG and fits it --
     this is "fit a joint" over heterogeneous data.
  2. **Simulate** (M2, ``mixle.inference.scenario.simulate``).  A "what-if": ``do(formation="salt")``
     -- roll out the implied depth/amplitude regime under that intervention, with a plausibility/ESS
     receipt.
  3. **Invert** (M3, ``mixle.task.inverse.learn_inverse``).  A NEW field observation arrives -- a
     noisy amplitude reading from an actual salt-formation site with an UNKNOWN depth. An amortized
     inverse posterior q(depth | amplitude) is trained against the same salt-regime forward physics
     M2 just characterized, then inverted on the real observation -- with SBC/coverage receipts that
     say whether the inversion is trustworthy, not just a point estimate.
  4. **Report** (M5's ``mixle.reason.language_bridge.PosteriorDescriber`` + A1's
     ``CalibratedGenerator``).  The depth posterior becomes a calibrated natural-language claim: draft
     candidate claims at several precision widths, score each against the posterior, and serve the
     best one only if it conformally clears a held-out threshold -- otherwise abstain rather than
     bluff a number.

Everything below is real, seeded computation; no printed number is invented. Run:
``python examples/geoscience_inversion_report.py``
"""

from __future__ import annotations

import numpy as np

from mixle.inference import certify
from mixle.inference.bayesian_network import learn_bayesian_network
from mixle.inference.scenario import Scenario, simulate
from mixle.reason.language_bridge import PosteriorDescriber, claim_score
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.task.inverse import learn_inverse

FORMATIONS = ("shale", "sand", "salt")
# formation -> (depth_mean_km, depth_std_km, per-station attenuation lengths)
FORMATION_PARAMS = {
    "shale": (2.0, 0.4, (1.6, 2.0, 2.4)),
    "sand": (3.0, 0.5, (2.2, 2.6, 3.0)),
    "salt": (4.5, 0.6, (3.2, 3.6, 4.0)),
}
FORMATION_WEIGHTS = (0.4, 0.35, 0.25)
A0 = 10.0  # source amplitude
SENSOR_NOISE = 0.15  # per-station amplitude noise, km-amplitude units
TRUE_DEPTH = 4.2  # the unknown depth at the real site M3 must recover
TRUE_FORMATION = "salt"


def _amplitude(depth: float, formation: str, rng: np.random.RandomState | None = None) -> tuple[float, ...]:
    """Exponential attenuation forward model: amplitude_i = A0 * exp(-depth / L_i) [+ station noise]."""
    _, _, lengths = FORMATION_PARAMS[formation]
    noise = rng.randn(len(lengths)) * SENSOR_NOISE if rng is not None else np.zeros(len(lengths))
    return tuple(float(A0 * np.exp(-depth / length) + n) for length, n in zip(lengths, noise))


def sense(n: int, seed: int) -> list[tuple]:
    """``n`` synthetic multi-modal field records: (formation: str, amplitude: 3-tuple, depth: float)."""
    rng = np.random.RandomState(seed)
    records = []
    for _ in range(n):
        formation = str(rng.choice(FORMATIONS, p=FORMATION_WEIGHTS))
        mean, std, _ = FORMATION_PARAMS[formation]
        depth = float(mean + std * rng.randn())
        records.append((formation, _amplitude(depth, formation, rng), depth))
    return records


def fit_joint(records: list[tuple]):
    """Discover and fit the heterogeneous DAG over (formation, amplitude, depth) -- M0's substrate."""
    return learn_bayesian_network(records, max_parents=2)


def what_if_salt(net, *, seed: int):
    """M2: ``do(formation="salt")`` -- roll out the implied depth/amplitude regime, receipted."""
    scenario = Scenario(interventions={0: TRUE_FORMATION}, evidence={}, horizon=1)
    sim = simulate(scenario, base=net, seed=seed)
    rows = sim.rollout(500)
    depths = np.array([r[2] for r in rows], dtype=float)
    amps = np.array([r[1] for r in rows], dtype=float)
    return sim, depths, amps


def invert_new_observation(depth_prior: GaussianDistribution, y_obs: np.ndarray, *, seed: int):
    """M3: train q(depth | amplitude) against the salt-regime forward physics, then invert ``y_obs``."""

    def forward_salt(theta_row: np.ndarray) -> np.ndarray:
        depth = float(np.ravel(theta_row)[0])
        return np.asarray(_amplitude(depth, TRUE_FORMATION), dtype=float)

    return learn_inverse(
        forward_salt,
        depth_prior,
        family="mdn",  # depth is 1-D; "flow" requires theta_dim >= 2 (see learn_inverse's docstring)
        n_sims=1500,
        n_sbc_replications=150,
        coverage_levels=(0.5, 0.9),
        y_obs=y_obs,
        rounds=1,
        seed=seed,
        m_steps=200,
        max_its=1,
    )


def build_calibration_set(inv_model, depth_prior: GaussianDistribution, *, n: int, seed: int):
    """(posterior, true_depth) pairs for the M5 describer's conformal calibration: fresh salt-regime
    sites, each observed once through the same noisy sensor as the real target site."""
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n):
        depth = float(depth_prior.sampler(seed=int(rng.randint(0, 2**31 - 1))).sample(1)[0])
        y = np.asarray(_amplitude(depth, TRUE_FORMATION), dtype=float) + SENSOR_NOISE * rng.randn(3)
        pairs.append((inv_model.posterior(y), depth))
    return pairs


def main() -> None:
    print("=" * 78)
    print("SENSE -> SIMULATE -> INVERT -> REPORT: a toy seismic-inversion full loop")
    print("=" * 78)

    # 1. sense + fit a joint
    records = sense(600, seed=0)
    net = fit_joint(records)
    cert = certify(net)
    print(f"\n[sense]     {len(records)} multi-modal field records (formation, amplitude[3], depth)")
    print(f"[fit joint] heterogeneous DAG over {len(net.factors)} fields:")
    for f in net.factors:
        print(f"              field[{f.child}] <- parents {getattr(f, 'parents', [])}  ({type(f).__name__})")
    print(f"            certificate: {cert.guarantee.name}")

    # 2. M2: a what-if scenario simulator
    sim, wi_depths, wi_amps = what_if_salt(net, seed=1)
    print(f"\n[simulate]  M2 what-if do(formation='{TRUE_FORMATION}'), {len(wi_depths)} rollout draws")
    print(f"            depth under the intervention:     mean={wi_depths.mean():.3f} std={wi_depths.std():.3f} km")
    print(f"            amplitude under the intervention:  mean={np.array2string(wi_amps.mean(axis=0), precision=3)}")
    print(f"            receipt: method={sim.receipt.method!r} ess_ratio={sim.receipt.ess_ratio}")

    # the M2 rollout's own empirical depth law becomes M3's prior -- the two stages share one belief
    depth_prior = GaussianDistribution(mu=float(wi_depths.mean()), sigma2=float(wi_depths.var()))

    # a new field observation: an actual site, unknown depth, one noisy amplitude reading
    obs_rng = np.random.RandomState(123)
    y_obs = np.asarray(_amplitude(TRUE_DEPTH, TRUE_FORMATION), dtype=float) + SENSOR_NOISE * obs_rng.randn(3)

    # 3. M3: invert the new observation
    inv_model = invert_new_observation(depth_prior, y_obs, seed=9)
    r = inv_model.receipts
    target_post = inv_model.posterior(y_obs)
    post_samples = target_post.sample(2000, seed=5)
    post_mean, post_std = float(post_samples.mean()), float(post_samples.std())
    print(f"\n[invert]    M3 amortized posterior q(depth | amplitude) trained on {inv_model.y_dim}-D observations")
    print(f"            new observation y_obs = {np.array2string(y_obs, precision=3)} (true depth = {TRUE_DEPTH} km)")
    print(
        f"            posterior: mean={post_mean:.3f} std={post_std:.3f} km, |error|={abs(post_mean - TRUE_DEPTH):.3f} km"
    )
    print(f"            SBC p-value={r.sbc_pvalue:.3g} (pass={r.sbc_pass})")
    print(
        f"            coverage@0.5={r.coverage[0.5]:.3f} (pass={r.coverage_pass[0.5]}), "
        f"coverage@0.9={r.coverage[0.9]:.3f} (pass={r.coverage_pass[0.9]})"
    )

    # 4. M5 + A1: a calibrated natural-language report
    calibration_set = build_calibration_set(inv_model, depth_prior, n=60, seed=999)
    describer = PosteriorDescriber(
        "depth_km", tol=0.1, k=3, alpha=0.2, width_multiples=(1.0, 3.0, 10.0), n_probe=300, seed=0
    )
    describer.calibrate(calibration_set, seed=0)
    claim = describer.describe(target_post, seed=0)

    print(f"\n[report]    M5 PosteriorDescriber calibrated on {len(calibration_set)} held-out sites through A1")
    if claim is None:
        print("            calibrated report: ABSTAIN -- no candidate claim conformally cleared the threshold")
    else:
        contained = claim.contains(TRUE_DEPTH)
        print(f'            calibrated report: "{claim.text()}"')
        print(f"            claim_score={claim_score(claim):.3f}, true depth {TRUE_DEPTH} km contained: {contained}")
        assert contained, "the served claim should bracket the true depth for this seeded run"

    print("\nOK: fit -> what-if -> invert -> calibrated report, one loop, real numbers throughout.")


if __name__ == "__main__":
    main()
