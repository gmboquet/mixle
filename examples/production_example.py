"""The production / MLOps layer: reproducible artifacts, a model registry, serving, drift, and checkpoints.

Everything lives in ``mixle.inference.production``. This walks the lifecycle end to end on a small Gaussian model:

  1. fit a model *with provenance* -> a Header recording the data hash, model hash, training settings,
     timing, environment, and the per-iteration model-hash chain;
  2. verify the training lineage (iteration i+1 descends from i);
  3. register versioned models and promote one to a 'production' alias (an atomic swap);
  4. serve scoring through a Service (with activity logging) and check drift vs a reference sample;
  5. checkpoint a long fit and resume it from the latest checkpoint.

Takeaway: the deliverable is not "the calls succeeded" -- it is the CHAIN. Each step emits a hash that
the next step can check, so ``verify_lineage`` and ``Registry.verify_chain`` answer "is this served
model really the one the recorded fit produced, iteration by iteration, and the one that was
registered and promoted?" from the digests alone. What the digests do not bind: the header's dataset
hash, record count, and log-likelihood trace are recorded claims about the data, carried alongside
the chain, not covered by it (a rewritten ``dataset_hash`` still verifies). Losing the chain is the
failure mode; producing a model is the easy part.

Note on ``trust_code``: ``Registry.get`` / ``current`` / ``verify_chain`` take a keyword-only
``trust_code`` (default ``False``, and it must be exactly ``True`` or ``False`` -- not a truthy value).
The models here are pure statistical artifacts with no embedded code, so the safe default is left
alone; pass ``trust_code=True`` only for a registry whose artifacts you are willing to execute.

Fully self-contained: random data, a throwaway registry directory.
Run: ``python examples/production_example.py``
"""

import tempfile

import numpy as np

from mixle.inference import optimize
from mixle.inference.production import Registry, Service, detect_drift, fit_with_provenance, verify_lineage
from mixle.stats import GaussianEstimator, MixtureEstimator


def draw(rng: np.random.RandomState, loc: float, scale: float, size: int) -> list[float]:
    """A seeded Gaussian sample rounded to 6 decimals, so its bytes (and hence the data hash) are the
    same on every platform. The raw draws are not: ``RandomState.normal`` goes through libm's ``log``,
    which is not correctly rounded and differs by an ULP between macOS and glibc, so the unrounded
    sample hashed to a different value per operating system while every derived statistic agreed.
    """
    return np.round(rng.normal(loc, scale, size), 6).tolist()


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    data = draw(rng, 3.0, 2.0, 4000)

    # 1. fit with provenance: the model carries a self-describing Header.
    model, header = fit_with_provenance(data, GaussianEstimator(), max_its=30, seed=1)
    print("# provenance")
    print("  data hash   :", header.dataset_hash[:16], "...")
    print("  model hash  :", header.model_hash[:16], "...")
    print("  iterations  :", header.training["iterations"], "| final loglik %.1f" % header.final_loglik)
    env = header.environment
    mixle_version = env.get("mixle_version") or env.get("pysp_version") or "unknown"
    print("  git / mixle  :", env.get("git_commit") or "unknown", "/", mixle_version)

    # 2. the per-iteration lineage is a verifiable hash chain.
    print("# lineage verified:", verify_lineage(header))

    with tempfile.TemporaryDirectory() as root:
        reg = Registry(root)

        # 3. register two versions and promote one to production (an atomic alias swap).
        reg.register(model, "demo")  # v1
        drifted, _ = fit_with_provenance(draw(rng, 9.0, 2.0, 4000), GaussianEstimator(), max_its=30)
        reg.register(drifted, "demo")  # v2
        reg.promote("demo", "v1", alias="production")
        prod, _ = reg.current("demo", "production")
        print("# registry: versions %s, production -> mu=%.2f" % (reg.versions("demo"), prod.mu))

        # 4. serve scoring + check drift against the training sample as reference.
        svc = Service(prod, name="demo", reference=data)
        lp = svc.score(draw(rng, 3.0, 2.0, 500))
        print("# serving: scored %d records, mean loglik %.2f" % (len(lp), svc.health()["mean_loglik"]))
        report = detect_drift(prod, data, draw(rng, 9.0, 2.0, 500))  # shifted batch
        print("  drift on shifted batch:", report.drift, "(ks=%.2f)" % report.score["ks"])

        # 5. checkpoint a fit every 3 iterations, then resume from the latest checkpoint. The two
        # components overlap (means -1 and +1, unit scale) so EM is still climbing at iteration 9:
        # well-separated components reach their fixed point in about three iterations, after which
        # float noise decides whether a later iteration counts as an improvement or ends the loop,
        # and the number of checkpoints would then depend on the machine.
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        seqs = draw(rng, -1.0, 1.0, 3000) + draw(rng, 1.0, 1.0, 3000)
        optimize(
            seqs,
            est,
            max_its=9,
            delta=None,
            out=None,
            rng=np.random.RandomState(2),
            on_step=reg.checkpointer("run", every=3),
        )
        print("# checkpoints:", reg.versions("run"), "| chain intact:", reg.verify_chain("run"))
        mid, _ = reg.get("run")  # latest checkpoint
        optimize(seqs, est, max_its=10, delta=None, out=None, prev_estimate=mid)  # resume training
        print("  resumed from the latest checkpoint")
