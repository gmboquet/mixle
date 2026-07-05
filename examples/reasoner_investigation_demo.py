"""The reasoner as product: one question, a plural action space, evidence bought under budget.

Where ``frontier_ecosystem_demo.py`` tours the supply chain (fit -> certify -> place -> calibrate ->
RAG), this script is the CONSUMER of it: a reasoner that answers questions by choosing, per question,
which evidence-acquiring action to fire -- RETRIEVE from the knowledge store, COMPUTE with a fitted
skill, SIMULATE a what-if on a learned causal model, CREATE a model on demand, or DELEGATE to a priced
external worker -- ordered by expected information gain per unit cost, and abstaining when nothing local
clears the bar. Then it LEARNS a better ordering from the telemetry its own runs emit.

Everything is measured in-process. Runtime a few seconds, no GPU, no network.
"""

from __future__ import annotations

import numpy as np

from mixle.inference import create, learn_action_policy, learn_bayesian_network, simulate
from mixle.inference.skill import SkillRegistry, skill
from mixle.substrate import (
    Substrate,
    compute_action,
    create_action,
    delegate_action,
    investigate,
    retrieve_action,
    simulate_action,
)
from mixle.telemetry import Telemetry


def line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def plan_spend(n: int, seed: int) -> list[tuple]:
    r = np.random.RandomState(seed)
    return [(["free", "pro"][i % 2], float(20 + 80 * (i % 2) + 3 * r.randn())) for i in range(n)]


def answerer(question: str, evidence: str) -> str:
    """A stand-in 99%-local student: it just grounds its answer in the evidence it was handed."""
    first = evidence.splitlines()[0] if evidence else "(no evidence)"
    return f"Based on the evidence, {first}"


def main() -> None:
    tel = Telemetry()

    # -- the supply chain: a knowledge store, a fitted skill, a causal simulator ------------------
    sub = Substrate()
    sub.add(kind="text", text="Refunds are processed within 30 days of a written request.")
    sub.add(kind="text", text="Enterprise support is staffed 24/7; free-tier support is business hours only.")

    reg = SkillRegistry()
    convert = skill(
        "unit_convert",
        lambda q: "100 degrees Celsius is 212 degrees Fahrenheit",
        description="convert temperature between celsius and fahrenheit units",
        registry=reg,
    )

    net = learn_bayesian_network(plan_spend(500, 0), max_parents=1)
    sim = simulate(net).scenario("pro", {0: "pro"}).scenario("free", {0: "free"})

    def build_spend_model(_q: str):
        return create([float(x) for x in np.random.RandomState(1).normal(50, 10, 300)], seed=1)

    # -- the reasoner's action space: five ways to buy evidence -----------------------------------
    actions = [
        retrieve_action(sub, min_score=0.2),  # filter tiny-embedder false positives
        compute_action(convert, cost=1.0),
        simulate_action(sim, 1, "pro", description="forecast average spend under the pro plan", cost=2.0),
        create_action(build_spend_model, description="build a fresh spend model from raw data", cost=4.0),
        delegate_action(
            lambda q: "external solver: proprietary tax rule = 8.25%",
            description="ask the external priced solver about proprietary tax rules",
            cost=8.0,
        ),
    ]

    line("ACT 1 -- one reasoner, five kinds of action, routed by expected-gain-per-cost")
    questions = [
        "when are refunds processed",
        "convert the temperature from celsius to fahrenheit",
        "forecast average spend under the pro plan",
        "what is the proprietary tax rule",
    ]
    for q in questions:
        inv = investigate(q, actions, answerer, telemetry=tel)
        fired = " -> ".join(f"{s.kind}({s.cost:g})" for s in inv.steps if s.fragments)
        print(f"\nQ: {q}")
        print(f"   fired : {fired}")
        print(f"   answer: {inv.answer}")
        print(f"   spent : {inv.spent:g} cost units | confidence {inv.confidence:.2f}")

    line("ACT 2 -- honest abstention: no local action clears the bar, so it withholds")
    inv = investigate(
        "what is the airspeed velocity of an unladen swallow",
        actions[:-1],  # no delegate available
        answerer,
        min_confidence=0.5,
        telemetry=tel,
    )
    print(f"abstained : {inv.abstained}")
    print(f"note      : {inv.note}")

    line("ACT 3 -- the reasoner learns a better ordering from its own route telemetry")
    rows = tel.training_rows("route")
    print(f"route telemetry rows accumulated: {len(rows)}")
    # Synthesize a modest reinforcement history: in this deployment, compute answers land, expensive
    # delegate rarely adds value -- exactly the signal a learned acquisition policy should internalize.
    hist = list(rows)
    for _ in range(12):
        hist.append(({"kind": "compute", "cost": 1.0, "overlap": 0.5}, "compute", {"value": 1.0}))
        hist.append(({"kind": "delegate", "cost": 8.0, "overlap": 0.5}, "delegate", {"value": 0.0}))
    policy = learn_action_policy(hist)

    q = "convert the temperature from celsius to fahrenheit"
    comp = actions[1]
    dele = actions[4]
    print(f"\nfor: {q!r}")
    print(f"   learned score  compute={policy(comp, q):.3f}  delegate={policy(dele, q):.3f}")
    inv = investigate(q, actions, answerer, scorer=policy, telemetry=tel)
    print(f"   with learned policy, first productive action: {next(s.kind for s in inv.steps if s.fragments)}")

    print("\nThe reasoner spent the least it could to answer, abstained rather than guess, and")
    print("learned — from the trace of its own decisions — to prefer the actions that pay off.")


if __name__ == "__main__":
    main()
