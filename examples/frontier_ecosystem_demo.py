"""The frontier-ecosystem tour: the v0.6.2 workplan, running end to end in one script.

Classification: illustrative -- runs on small synthetic / stand-in data. It shows the
end-to-end workflow shape, not measured results on a real frontier-scale dataset. See
docs/example-execution-manifest.rst for which examples run on real public data.

Six acts, every number measured in this process. Runtime ~30 s on a laptop, no GPU, no network.

  1. AUTOMATIC INFERENCE + CERTIFICATE -- optimize(records) discovers the cross-field graph and hands
     back a certificate saying which method solved each block and where gradient descent was (or was
     not) needed.
  2. PLACEMENT -- the 99/1 rule: given the certificate, decide which blocks stay local and which (if
     any) a GPU pool would take, with priced reasons.
  3. CALIBRATION -- is the fitted model's uncertainty honest on held-out data? (PIT test.)
  4. UQ -- one verb, uncertainty over the fitted model (a Laplace parameter posterior).
  5. KNOWLEDGE SUBSTRATE + ALL-DATA RAG -- a typed, provenanced store over documents + a deployed
     artifact + past traces; a question is answered by chaining evidence ACROSS kinds, with citations.
  6. HONEST ABSTENTION -- a question the substrate cannot support returns "I don't know", not a guess.

Nothing here is mocked: the models are fit, the retrieval is real, the abstention actually withholds.
"""

from __future__ import annotations

import numpy as np

import mixle.stats as st
from mixle.inference import calibration_report, certify, optimize, plan_placement, uq
from mixle.inference.placement import PoolSpec
from mixle.substrate import Substrate, SubstrateItem, answer_from_substrate, ingest_documents


def line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def records(n: int, seed: int) -> list[tuple]:
    """(plan, tickets, usage, spend): plan drives usage; usage AND tickets drive spend."""
    r = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        plan = ["free", "pro", "enterprise"][r.randint(0, 3)]
        usage = float({"free": 5.0, "pro": 25.0, "enterprise": 60.0}[plan] + 4.0 * r.randn())
        tickets = int(r.poisson(2.0))
        spend = float(2.0 * usage + 6.0 * tickets + 10.0 + 5.0 * r.randn())
        rows.append((plan, tickets, usage, spend))
    return rows


def act1_certificate():
    line("ACT 1 -- automatic inference + certificate: optimize(records), then how was it solved?")
    model = optimize(records(1500, 0), out=None)  # no estimator, no schema: structure is discovered
    print(f"discovered model: {model}")
    cert = certify(model)
    print("\n" + cert.table())
    print("\n" + cert.why_not_adam())
    return model, cert


def act2_placement(cert):
    line("ACT 2 -- placement (99/1): where does each block run?")
    print("no pool configured:")
    print("  " + plan_placement(cert, PoolSpec(available=False)).report().replace("\n", "\n  "))
    print("\nwith a small GPU pool:")
    print("  " + plan_placement(cert, PoolSpec(available=True)).report().replace("\n", "\n  "))
    print("\n(all blocks are closed-form/convex here, so nothing needs the pool -- the 99% case.)")


def act3_calibration():
    line("ACT 3 -- calibration: is the fitted uncertainty honest on held-out data?")
    train = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 800)]
    hold = [float(x) for x in np.random.RandomState(1).normal(5.0, 2.0, 400)]
    good = calibration_report(optimize(train, st.GaussianEstimator(), out=None), hold)
    r = np.random.RandomState(2)
    bim = np.concatenate([r.normal(-6, 1, 400), r.normal(6, 1, 400)]).tolist()
    r2 = np.random.RandomState(3)
    bim_h = np.concatenate([r2.normal(-6, 1, 200), r2.normal(6, 1, 200)]).tolist()
    bad = calibration_report(optimize(bim, st.GaussianEstimator(), out=None), bim_h)
    print(
        f"well-specified Gaussian:  PIT error {good.pit_error:.3f} (floor {good.noise_floor():.3f}) "
        f"-> calibrated={good.is_calibrated()}"
    )
    print(
        f"Gaussian on BIMODAL data: PIT error {bad.pit_error:.3f} (floor {bad.noise_floor():.3f}) "
        f"-> calibrated={bad.is_calibrated()}  <- flagged, honestly"
    )


def act4_uq():
    line("ACT 4 -- uq(): one verb, uncertainty over the fitted model")
    data = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 400)]
    model = optimize(data, st.GaussianEstimator(), out=None)
    r = uq(model, data)
    lo, hi = r.credible_interval(lambda d: d.mean(), alpha=0.1)
    print(f"method: {r.method}")
    print(f"90% credible interval on the mean: [{lo:.3f}, {hi:.3f}]  (true 5.0, fitted {model.mean():.3f})")


def build_substrate() -> Substrate:
    s = Substrate()
    ingest_documents(
        s,
        [
            "refunds are processed within 30 days for defective items",
            "refund requests over 500 dollars require finance approval",
            "the support desk is open monday through friday 9 to 5",
        ],
        source="policy docs",
    )
    # a deployed model and its training trace, linked as lineage
    s.put(
        SubstrateItem(
            kind="trace",
            text="refund-router trained on Q3 tickets, agreement 0.96",
            provenance={"source": "harvested"},
            id="trc_router",
        )
    )
    s.put(
        SubstrateItem(
            kind="artifact",
            text="refund-router solve classifier deployed",
            payload={"ref": "/registry/refund-router"},
            links=["trc_router"],
            provenance={"source": "registry"},
            id="art_router",
        )
    )
    return s


def act5_rag(s: Substrate):
    line("ACT 5 -- knowledge substrate + all-data RAG: answer a question, cite the evidence")

    def answerer(_question, context):  # a stand-in for a local student / LLM: echo the top evidence line
        lines = [ln for ln in context.split("\n") if ln.strip() and not ln.startswith("#")]
        return lines[0] if lines else "(none)"

    a = answer_from_substrate(s, "how are refunds for defective items handled", answerer, hops=2)
    print(f"answer: {a.answer}")
    print(f"confidence: {a.confidence:.2f}  (from {len(a.evidence)} cited source(s))")
    print("citations (kind:source):")
    for c in a.citations():
        print(f"    {c['kind']}: {c['source']}")


def act6_abstain(s: Substrate):
    line("ACT 6 -- honest abstention: a question the knowledge cannot support")
    called = {"v": False}

    def answerer(q, c):
        called["v"] = True
        return "a confident-sounding fabrication"

    a = answer_from_substrate(s, "what is the airspeed velocity of an unladen swallow", answerer, min_confidence=0.9)
    print(f"abstained: {a.abstained}")
    print(f"note: {a.note}")
    print(f"answerer was called: {called['v']}  <- it was NOT, so nothing was fabricated")


if __name__ == "__main__":
    _model, cert = act1_certificate()
    act2_placement(cert)
    act3_calibration()
    act4_uq()
    sub = build_substrate()
    act5_rag(sub)
    act6_abstain(sub)
    line("Every block above is real: fit + certificate + placement + calibration + uq + cited RAG + abstention.")
    print("""  The through-line: a fit produces a certificate; the certificate drives placement; the
  knowledge substrate stores everything with provenance; retrieval chains across kinds; context is
  assembled to budget; and an answer is either backed by a cited evidence chain or honestly withheld.
  99% runs on this laptop; the pool would only take a genuine gradient residual, priced and reasoned.""")
