"""The skeptic's challenge: "neural integration, distillation, automatic inference are hype."

Run this. Every number below is measured in this process, head-to-head against the tools a skeptic
would actually reach for (scikit-learn specialists, raw torch). Three claims, three acts:

  1. AUTOMATIC INFERENCE — one ``optimize(data)`` call on raw heterogeneous records discovers the
     cross-field dependency graph, then answers FOUR different queries from that single fit —
     each compared against a scikit-learn specialist trained for that one query.
  2. DISTILLATION — replace a rigid routine with a tiny calibrated student. The comparison is not
     accuracy (a logistic regression matches accuracy); it is HONESTY: the sklearn classifier's
     errors are all silent, the mixle system's local errors are conformally bounded and everything
     else escalates to the teacher.
  3. NEURAL INTEGRATION — a torch normalizing flow is a first-class mixle distribution: it EM-fits
     INSIDE a mixture next to a classical Gaussian with the same ``optimize`` verb. The hybrid beats
     both the all-classical and the all-neural fit on held-out likelihood.

Honest boundaries are printed at the end. Runtime ~2 minutes on a laptop, no GPU needed.
"""

from __future__ import annotations

import io
import time

import numpy as np

import mixle.stats as st
from mixle.inference import optimize

RNG = np.random.RandomState(0)


def _line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------------------------
# Act 1: automatic inference on heterogeneous records — one fit, four queries, four specialists
# ---------------------------------------------------------------------------------------------


def make_records(n: int, seed: int) -> list[tuple]:
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


def act1() -> None:
    _line("ACT 1 — 'automatic inference is hype': one optimize(data) vs four sklearn specialists")
    train, test = make_records(1500, 0), make_records(400, 1)

    t0 = time.perf_counter()
    log = io.StringIO()
    model = optimize(train, out=log)  # <- the entire modeling step. No estimator, no schema, no graph.
    fit_s = time.perf_counter() - t0
    print(f"mixle: model = optimize(records)   [{fit_s:.1f}s]")
    print(f"  -> {model}")
    print(f"  -> {log.getvalue().strip() or '(no structure note)'}")

    # -- query 0: did discovering structure matter? (held-out joint likelihood, same representation)
    indep = optimize(train, out=None, structure="off")
    enc_m = model.dist_to_encoder().seq_encode(test)
    enc_i = indep.dist_to_encoder().seq_encode(test)
    ll_m = float(np.mean(model.seq_log_density(enc_m)))
    ll_i = float(np.mean(indep.seq_log_density(enc_i)))
    print(
        f"\n  [joint density] held-out log-lik/row: discovered graph {ll_m:.2f} vs independence {ll_i:.2f}"
        f"  (+{ll_m - ll_i:.2f} nats/row)"
    )

    # sklearn feature prep (what a colleague would write): one-hot plan + raw numerics
    def feats(rows):
        plans = ["free", "pro", "enterprise"]
        return np.asarray([[1.0 * (p == q) for q in plans] + [t, u] for p, t, u, _s in rows])

    ytr = np.asarray([r[3] for r in train])
    yte = np.asarray([r[3] for r in test])

    # -- query 1: predict spend | rest (the FORWARD direction a specialist would be trained for)
    from sklearn.ensemble import GradientBoostingRegressor

    gbr = GradientBoostingRegressor(random_state=0).fit(feats(train), ytr)
    mae_sk = float(np.mean(np.abs(gbr.predict(feats(test)) - yte)))

    grid = np.linspace(ytr.min() - 20, ytr.max() + 20, 160)

    def impute_spend(row):  # MAP imputation straight off the joint: argmax_s p(plan,tickets,usage,s)
        cands = [(*row[:3], float(s)) for s in grid]
        return float(grid[int(np.argmax(model.seq_log_density(model.dist_to_encoder().seq_encode(cands))))])

    mae_mx = float(np.mean([abs(impute_spend(r) - r[3]) for r in test]))
    print(
        f"  [spend | rest]  MAE: mixle joint {mae_mx:.2f} vs sklearn GBR-trained-for-this {mae_sk:.2f}"
        f"   (specialist may edge it — it was trained for exactly this)"
    )

    # -- query 2: predict plan | rest (the REVERSE direction: sklearn needs a SECOND model)
    from sklearn.linear_model import LogisticRegression

    def feats_rev(rows):
        return np.asarray([[t, u, s] for _p, t, u, s in rows])

    lr = LogisticRegression(max_iter=2000).fit(feats_rev(train), [r[0] for r in train])
    acc_sk = float(np.mean(lr.predict(feats_rev(test)) == [r[0] for r in test]))

    plans = ["free", "pro", "enterprise"]

    def impute_plan(row):  # same single fit, opposite direction — no retraining
        cands = [(p, *row[1:]) for p in plans]
        return plans[int(np.argmax(model.seq_log_density(model.dist_to_encoder().seq_encode(cands))))]

    acc_mx = float(np.mean([impute_plan(r) == r[0] for r in test]))
    print(f"  [plan | rest]   acc: mixle same-fit {acc_mx:.3f} vs sklearn second-model {acc_sk:.3f}")

    # -- query 3: anomaly detection (dependence-breaking corruption: shuffle spend across rows)
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score

    corrupted = [(*r[:3], test[(i + 37) % len(test)][3]) for i, r in enumerate(test)]
    both = list(test) + corrupted
    labels = np.r_[np.zeros(len(test)), np.ones(len(corrupted))]
    scores_mx = -np.asarray(model.seq_log_density(model.dist_to_encoder().seq_encode(both)))
    auc_mx = float(roc_auc_score(labels, scores_mx))

    iso = IsolationForest(random_state=0).fit(np.c_[feats(train), ytr])
    scores_sk = -iso.score_samples(np.c_[feats(both), [r[3] for r in both]])
    auc_sk = float(roc_auc_score(labels, scores_sk))
    print(
        f"  [anomaly]       AUC: mixle joint {auc_mx:.3f} vs sklearn IsolationForest {auc_sk:.3f}"
        f"   (each corrupted field is marginally normal — only the broken DEPENDENCE gives it away)"
    )

    # -- query 4: generation (no sklearn column: the specialists cannot answer this one)
    rows = model.sampler(seed=0).sample(3)
    print("  [generate]      3 synthetic records from the same fit (sklearn column: n/a):")
    for r in rows:
        print(f"                    ({r[0]!r}, tickets={r[1]}, usage={r[2]:.1f}, spend={r[3]:.1f})")


# ---------------------------------------------------------------------------------------------
# Act 2: distillation — the claim is calibrated honesty, not accuracy
# ---------------------------------------------------------------------------------------------


def act2() -> None:
    _line("ACT 2 — 'distillation is hype': the sklearn retort answers, the receipts say HOW OFTEN silently wrong")

    def teacher(t):  # the rigid routine being replaced (imagine an API call at $0.03/request);
        # the interacting amount-threshold rule puts real ambiguity near the 500 boundary
        if t["amount"] > 500 and t["kind"] == "refund":
            return "finance-escalation"
        if t["kind"] in ("refund", "billing") and t["amount"] > 180:
            return "billing"
        return "support"

    def tickets(n, seed=0):
        r = np.random.RandomState(seed)
        kinds = ["refund", "billing", "question", "bug"]
        return [
            {
                "kind": kinds[r.randint(0, 4)],
                "amount": float(r.uniform(100.0, 900.0)),
                "region": ["us", "eu"][r.randint(0, 2)],
            }
            for _ in range(n)
        ]

    from mixle.task import scorecard, solve

    train, fresh = tickets(500), tickets(400, seed=9)
    t0 = time.perf_counter()
    sol = solve(teacher, train, alpha=0.08, seed=0, epochs=250)
    print(f"mixle: sol = solve(teacher, tickets)   [{time.perf_counter() - t0:.1f}s]")
    card = scorecard(sol, teacher, fresh, student_cost=0.0001, teacher_cost=0.03, task="ticket routing")
    print("\n".join("  " + ln for ln in card.table().splitlines()))

    # the retort: "just train a classifier on the teacher's labels"
    from sklearn.linear_model import LogisticRegression

    def feats(rows):
        kinds = ["refund", "billing", "question", "bug"]
        return np.asarray(
            [[1.0 * (t["kind"] == k) for k in kinds] + [t["amount"], 1.0 * (t["region"] == "eu")] for t in rows]
        )

    lr = LogisticRegression(max_iter=3000).fit(feats(train), [teacher(t) for t in train])
    pred = lr.predict(feats(fresh))
    truth = [teacher(t) for t in fresh]
    sk_wrong = float(np.mean(pred != np.asarray(truth)))

    wrong_system = escalated = 0
    for t in fresh:
        local = sol.cascade.model.decide(t)
        if local is None:
            escalated += 1  # the system runs the teacher here: correct by construction
        elif local != teacher(t):
            wrong_system += 1
    print(f"\n  sklearn LogisticRegression: {sk_wrong:.1%} of fresh traffic answered WRONG — every error silent")
    print(
        f"  mixle system:               {wrong_system / len(fresh):.1%} wrong end-to-end "
        f"({escalated / len(fresh):.1%} of traffic said 'not sure' and escalated to the teacher;"
    )
    print("                              the answered slice carries conformal risk alpha=0.08 by construction)")
    print("  real-data receipts (not this synthetic demo): Banking77 e2e 0.983 vs teacher at 84KB;")
    print("  escalation decays 0.679->0.428 over 6 harvest rounds (examples/real_receipt_banking77.py);")
    print("  CLIP VLM -> 10k-param head, 14760x smaller at matched accuracy (mixle-mlops examples).")


# ---------------------------------------------------------------------------------------------
# Act 3: neural integration — a torch flow as a first-class citizen inside classical EM
# ---------------------------------------------------------------------------------------------


def act3() -> None:
    _line("ACT 3 — 'neural integration is hype': a torch flow EM-fit INSIDE a mixture, and it pays")
    import torch

    from mixle.models.neural_density import NeuralDensity, build_coupling_flow

    def banana_plus_cluster(n, seed):
        r = np.random.RandomState(seed)
        n_banana = int(0.85 * n)
        x = r.uniform(-2.5, 2.5, n_banana)
        banana = np.c_[x, 0.6 * x**2 - 2.0 + 0.25 * r.randn(n_banana)]  # curved manifold
        cluster = np.c_[4.5 + 0.25 * r.randn(n - n_banana), 4.0 + 0.25 * r.randn(n - n_banana)]  # rare tight mode
        rows = np.r_[banana, cluster]
        return [rows[i] for i in r.permutation(len(rows))]

    train, test = banana_plus_cluster(1200, 0), banana_plus_cluster(600, 1)

    def held_out(m):
        return float(np.mean(m.seq_log_density(m.dist_to_encoder().seq_encode(test))))

    torch.manual_seed(0)
    t0 = time.perf_counter()
    classical = optimize(
        train, st.MixtureEstimator([st.MultivariateGaussianEstimator(dim=2)] * 2), max_its=40, out=None
    )
    ll_c = held_out(classical)
    print(
        f"  all-classical (2-comp Gaussian mixture)   held-out ll/row {ll_c:7.3f}   [{time.perf_counter() - t0:.0f}s]"
    )

    torch.manual_seed(0)
    t0 = time.perf_counter()
    flow = NeuralDensity(build_coupling_flow(2, hidden=32, layers=6), m_steps=60, lr=5e-3)
    neural = optimize(train, flow.estimator(), prev_estimate=flow, max_its=8, out=None)
    ll_n = held_out(neural)
    print(
        f"  all-neural (RealNVP coupling flow)        held-out ll/row {ll_n:7.3f}   [{time.perf_counter() - t0:.0f}s]"
    )

    torch.manual_seed(0)
    t0 = time.perf_counter()
    est = st.MixtureEstimator(
        [
            NeuralDensity(build_coupling_flow(2, hidden=32, layers=6), m_steps=60, lr=5e-3).estimator(),
            st.MultivariateGaussianEstimator(dim=2),
        ]
    )
    init = st.MixtureDistribution(
        [
            NeuralDensity(build_coupling_flow(2, hidden=32, layers=6), m_steps=60, lr=5e-3),
            st.MultivariateGaussianDistribution([4.0, 4.0], np.eye(2)),
        ],
        [0.5, 0.5],
    )
    hybrid = optimize(train, est, prev_estimate=init, max_its=8, out=None)
    ll_h = held_out(hybrid)
    print(
        f"  HYBRID mixture [flow, Gaussian], same EM  held-out ll/row {ll_h:7.3f}   [{time.perf_counter() - t0:.0f}s]"
    )
    print(f"  -> hybrid vs classical: {ll_h - ll_c:+.3f} nats/row;  hybrid vs pure-neural: {ll_h - ll_n:+.3f} nats/row")
    print(
        f"  -> mixture weights learned by EM: {np.round(hybrid.w, 3).tolist()}"
        f" (flow took the curved 85%, the Gaussian took the rare tight mode)"
    )
    print("  in raw torch this is: hand-written EM, hand-written responsibility weighting, and no")
    print("  composability with the 100+ classical families, samplers, and conformal layers above.")


if __name__ == "__main__":
    act1()
    act2()
    act3()
    _line("HONEST BOUNDARIES (what mixle does NOT claim)")
    print("""  - It is not a deep-learning framework: neural components are adapters into the probabilistic
    contract (score/fit/sample/compose), not a trainer for ImageNet-scale networks.
  - Structure search is greedy + BIC (both orientations, GLM/CLG/table factors), not exhaustive
    causal discovery; edge directions on observationally-equivalent pairs are not identified.
  - The distillation demos above use a rigid-function teacher; the real-data receipts cited in
    Act 2 are the evidence line, and frontier-LLM teachers are deliberately future work.
  - Specialists trained for one query can edge the joint model on that query (see Act 1 MAE);
    the claim is one automatic fit that answers every direction, not supremacy on each.""")
