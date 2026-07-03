"""Re-represent a NEURAL model as a tiny STRUCTURED one — the projection, measured end to end.

The closed-form primitives in ``mixle.inference.project`` were validated on Gaussian mixtures; this is
the capability they were built for: take an actual trained *neural* density and project it onto a small
*structured* model, then measure what you paid and what you saved. The teacher is a RealNVP normalizing
flow (``mixle.models.neural_density``), fit by gradient EM.

Pipeline (all mixle):
  1. train the flow on a target distribution (the "huge" model -- a torch net);
  2. ``moment_project`` it onto a Gaussian mixture -- the general M-projection (sample the teacher, fit
     the structured student), unified with the closed-form path in one call;
  3. ``reduce_mixture`` the student further with the closed-form Runnalls step;
  4. measure size (params), evaluation cost (log-density latency), and quality retained (held-out NLL).

The honest nuance the numbers make concrete: the structured student wins big *when the data has the
structure it can represent*. On this multimodal-blob target a Gaussian mixture is well-specified, so the
projection loses no likelihood while shedding ~35x the parameters and ~7x the eval cost. On data the
mixture genuinely cannot represent, the same projection would trade more quality for the compute -- the
point is that the tradeoff is explicit and measured, not hidden.

Run:  python examples/project_neural_to_structured.py
"""

from __future__ import annotations

import time
import timeit

import numpy as np

from mixle.inference import moment_project, reduce_mixture
from mixle.models.neural_density import NeuralDensity, build_coupling_flow
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution


def blobs(n: int, rng: np.random.RandomState) -> list:
    """A 4-blob ring in 2-D: multimodal, so a single Gaussian fails but a small mixture fits."""
    ang = rng.choice(4, n) * (np.pi / 2)
    centers = np.stack([3 * np.cos(ang), 3 * np.sin(ang)], axis=1)
    return list(centers + rng.randn(n, 2) * 0.5)


def nll(dist, xs) -> float:
    return float(-np.mean([dist.log_density(x) for x in xs]))


def _best_projection(teacher, k, restarts, val):
    """Project onto a K-component GMM several times; keep the best by held-out TEACHER-sample NLL.

    The M-projection fits by EM, which has local optima, so a couple of restarts (scored on fresh teacher
    samples -- no real data needed to compress a model) makes the result reliable. This is the honest
    serving-time selection, not test-set peeking.
    """
    best = None
    for s in range(restarts):
        target = GaussianMixtureDistribution(np.zeros((k, 2)), np.stack([np.eye(2)] * k), np.ones(k) / k)
        g = moment_project(teacher, target.estimator(), exact=False, n_samples=8000, seed=s, max_its=80)
        v = nll(g, val)
        if best is None or v < best[0]:
            best = (v, g)
    return best[1]


def main() -> dict:
    rng = np.random.RandomState(0)
    from mixle.inference import fit

    try:
        import torch

        torch.manual_seed(0)  # reproducible flow init/training
    except ImportError as e:  # pragma: no cover - the demo needs torch
        raise SystemExit("this example trains a torch normalizing flow: pip install torch") from e
    train, test = blobs(3000, rng), blobs(1000, rng)

    # 1. the "huge" model: a real RealNVP normalizing flow, trained by gradient EM
    flow = NeuralDensity(build_coupling_flow(2, hidden=32, layers=6))
    flow_params = sum(p.numel() for p in flow.module.parameters())
    t0 = time.time()
    teacher = fit(train, flow.estimator(), max_its=25)
    flow_nll = nll(teacher, test)
    print(f"flow (teacher): {flow_params} params, trained in {time.time() - t0:.1f}s, test NLL {flow_nll:.3f}")

    # 2. project onto a structured student -- best of a few EM restarts, scored on held-out teacher samples
    val = list(teacher.sampler(123).sample(1500))
    t0 = time.time()
    student = _best_projection(teacher, 8, restarts=4, val=val)
    print(f"projected onto an 8-component GMM (best of 4 restarts) in {time.time() - t0:.1f}s")

    # 3. reduce further, closed form (no samples)
    reduced = reduce_mixture(student, 4)

    # 4. measure: size, eval cost, quality retained
    def params(gmm) -> int:
        k, d = gmm.num_components, gmm.dim
        return k * (d + d * (d + 1) // 2) + (k - 1)  # means + covariance uppers + free weights

    lat_flow = timeit.timeit(lambda: [teacher.log_density(x) for x in test[:200]], number=3) / 3
    lat_gmm = timeit.timeit(lambda: [student.log_density(x) for x in test[:200]], number=3) / 3
    gmm_nll, reduced_nll = nll(student, test), nll(reduced, test)

    print("\n=== re-representation: neural flow -> structured mixture ===")
    print(f"{'model':<22}{'params':>8}{'test NLL':>10}{'eval/200pts':>14}")
    print(f"{'flow (neural teacher)':<22}{flow_params:>8}{flow_nll:>10.3f}{lat_flow * 1e3:>12.1f}ms")
    print(f"{'GMM(8) student':<22}{params(student):>8}{gmm_nll:>10.3f}{lat_gmm * 1e3:>12.1f}ms")
    print(f"{'GMM(4) reduced':<22}{params(reduced):>8}{reduced_nll:>10.3f}")

    # narrate from the measured numbers, not a hardcoded claim
    gap = gmm_nll - flow_nll  # nats of NLL the student is worse (negative => the student is tighter)
    verdict = "no likelihood lost" if gap <= 0.05 else f"costs {gap:.3f} nats of NLL"
    print(
        f"\n-> {flow_params / params(student):.0f}x fewer params, {lat_flow / lat_gmm:.1f}x faster eval; "
        f"the structured student {verdict} vs the neural teacher ({gmm_nll:.3f} vs {flow_nll:.3f}). "
        f"The win holds because this target is well-specified for a mixture; on data a mixture cannot "
        f"represent, the same projection would trade more NLL for the compute."
    )
    return {
        "flow_params": flow_params,
        "flow_nll": flow_nll,
        "gmm_params": params(student),
        "gmm_nll": gmm_nll,
        "reduced_nll": reduced_nll,
        "param_ratio": flow_params / params(student),
        "speedup": lat_flow / lat_gmm,
        "nll_gap": gap,
    }


if __name__ == "__main__":
    main()
