"""The production / MLOps layer: reproducible artifacts, a model registry, serving, drift, and checkpoints.

Everything lives in ``mixle.inference.production``. This walks the lifecycle end to end on a small Gaussian model:

  1. fit a model *with provenance* -> a Header recording the data hash, model hash, training settings,
     timing, environment, and the per-iteration model-hash chain;
  2. verify the training lineage (iteration i+1 descends from i);
  3. register versioned models and promote one to a 'production' alias (an atomic swap);
  4. serve scoring through a Service (with activity logging) and check drift vs a reference sample;
  5. checkpoint a long fit and resume it from the latest checkpoint.

Fully self-contained: random data, a throwaway registry directory.
"""

import tempfile

import numpy as np

from mixle.inference import optimize
from mixle.inference.production import Registry, Service, detect_drift, fit_with_provenance, verify_lineage
from mixle.stats import GaussianEstimator, MixtureEstimator

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    data = rng.normal(3.0, 2.0, 4000).tolist()

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
        drifted, _ = fit_with_provenance(rng.normal(9.0, 2.0, 4000).tolist(), GaussianEstimator(), max_its=30)
        reg.register(drifted, "demo")  # v2
        reg.promote("demo", "v1", alias="production")
        prod, _ = reg.current("demo", "production")
        print("# registry: versions %s, production -> mu=%.2f" % (reg.versions("demo"), prod.mu))

        # 4. serve scoring + check drift against the training sample as reference.
        svc = Service(prod, name="demo", reference=data)
        lp = svc.score(rng.normal(3.0, 2.0, 500).tolist())
        print("# serving: scored %d records, mean loglik %.2f" % (len(lp), svc.health()["mean_loglik"]))
        report = detect_drift(prod, data, rng.normal(9.0, 2.0, 500).tolist())  # shifted batch
        print("  drift on shifted batch:", report.drift, "(ks=%.2f)" % report.score["ks"])

        # 5. checkpoint a fit every 3 iterations, then resume from the latest checkpoint.
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        seqs = np.concatenate([rng.normal(-5, 1, 3000), rng.normal(5, 1, 3000)]).tolist()
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
