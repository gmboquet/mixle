"""Flagship app (G): cross-modal industrial triage, end to end through the reasoner.

Classification: illustrative -- runs on small synthetic / stand-in data. It shows the
end-to-end workflow shape, not measured results on a real frontier-scale dataset. See
docs/example-execution-manifest.rst for which examples run on real public data.

One runnable receipt touching every plane the workplan calls for:

  1. KNOWLEDGE   -- a typed substrate of policy docs + a resolution trace, secrets redacted on ingest.
  2. MODEL       -- create() fits a certified spend model from records (guarantee + why-not-ADAM).
  3. SKILL       -- the fitted model is registered as a reusable, findable skill.
  4. POOL        -- a fit job is submitted through the pool rails (budget-checked, local backend).
  5. REASONER    -- a Harness (support-triage) answers questions from the knowledge with citations,
                    escalates what it cannot support, and refuses malformed / oversized requests.
  6. MONITOR     -- a drift check over a fresh batch decides whether the model needs a re-fit.
  7. TRUST       -- every answer is grounded (factuality receipt) and no secret is ever indexed.

Measured in-process; a few seconds, no GPU, no network. This is the "reasoning is the product" thesis
in one file: knowledge + models + skills + pool + monitoring, consumed through one honest front door.
"""

from __future__ import annotations

import numpy as np

from mixle.inference import create
from mixle.inference.nonparametric import ks_2samp
from mixle.pool import PoolJob, submit
from mixle.substrate import Substrate, register_harness, safe_text, support_triage_harness
from mixle.substrate.harness import find_harnesses


def line(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    rng = np.random.RandomState(0)

    # 1. KNOWLEDGE -----------------------------------------------------------------------------------
    sub = Substrate()
    sub.add(kind="text", text=safe_text("Refunds are processed within 30 days of a written request."))
    sub.add(kind="text", text=safe_text("Enterprise support is staffed 24/7; free-tier is business hours."))
    # a leaked secret in an ingested trace is redacted BEFORE it is indexed
    sub.add(kind="trace", text=safe_text("case 4411: internal token sk-abcdefghij1234567890XYZ rotated"))

    # 2. MODEL -------------------------------------------------------------------------------------
    spend = [float(x) for x in rng.normal(50, 12, 400)]
    artifact = create(spend, calibrate=0.3, seed=0)
    line("MODEL: create() -> a certified artifact")
    print(f"guarantee: {artifact.guarantee.name} | calibrated: {artifact.is_calibrated()}")
    print(f"  {artifact.why().splitlines()[0]}")

    # 3. SKILL + 4. POOL ---------------------------------------------------------------------------
    from mixle.inference.skill import SkillRegistry, skill

    reg = SkillRegistry()
    skill("spend_model", artifact, description="sample or score customer spend", tags=["spend"], registry=reg)
    result = submit(PoolJob(run=lambda: create(spend, seed=1), kind="verb", reason="refit spend model", est_cost=0.0))
    line("SKILL + POOL")
    print(f"skill registered: {reg.best('customer spend').name}")
    print(f"pool job {result.status}: cost {result.cost} (local backend, budget-checked)")

    # 5. REASONER (Harness) ------------------------------------------------------------------------
    tickets: list[str] = []

    def escalate(req, inv):
        tickets.append(req)
        return f"ticket-{len(tickets)}"

    def answerer(question, context):
        return context.splitlines()[0] if context else ""

    harness = support_triage_harness(sub, answerer, escalate=escalate)
    register_harness(sub, harness, scope="support")

    line("REASONER: the triage harness answers, escalates, or refuses")
    for q in ["when are refunds processed", "what is the meaning of life", ""]:
        r = harness.handle(q)
        print(f"  {q!r:40} -> {r.status}" + (f": {r.answer}" if r.answer else f" ({r.reason})"))
    # a request carrying a secret is redacted before any action sees it
    leaked = harness.handle("my key sk-abcdefghij1234567890XYZ — when are refunds processed")
    print(f"  [request with a secret]           -> {leaked.status}, {leaked.redactions} redaction(s)")

    # 6. MONITOR -----------------------------------------------------------------------------------
    line("MONITOR: drift check decides whether to re-fit")
    fresh = [float(x) for x in np.random.RandomState(1).normal(50, 12, 300)]  # same regime
    drifted = [float(x) for x in np.random.RandomState(2).normal(70, 12, 300)]  # shifted
    for name, batch in [("in-distribution", fresh), ("shifted", drifted)]:
        p = ks_2samp(spend, batch).pvalue
        print(f"  {name:16}: KS p={p:.4f} -> {'DRIFT, re-fit' if p < 0.01 else 'no drift'}")

    # 7. TRUST -------------------------------------------------------------------------------------
    line("TRUST: grounded answer + no leaked secret in the index")
    from mixle.substrate import check_factuality, scan_substrate

    ans = harness.handle("when are refunds processed")
    receipt = check_factuality(sub, ans.answer or "")
    print(f"  answer grounded_fraction: {receipt.grounded_fraction}")
    print(f"  substrate secret scan: {scan_substrate(sub)['n_dirty']} dirty item(s)")
    print(f"  registered harnesses: {[h['harness'] for h in find_harnesses(sub)]}")

    print("\nknowledge + model + skill + pool + reasoner + monitor + trust — one honest front door.")


if __name__ == "__main__":
    main()
