"""The frontier-family lifecycle, end to end: checkpoint -> J2 ladder -> I1 artifacts -> J4 edge

Classification: illustrative -- runs on small synthetic / stand-in data. It shows the
end-to-end workflow shape, not measured results on a real frontier-scale dataset. See
docs/example-execution-manifest.rst for which examples run on real public data.
student -> served cascade, with real cost/quality receipts at every stage (roadmap B6).

This example does not build anything new -- every stage is existing, already-receipted machinery,
composed exactly the way F11 (:mod:`mixle.task.deploy_family`) composes it:

  1. TRAIN a small headline causal LM on F10's own four eval axes (perplexity/arithmetic/parity/
     induction), so it has genuine, above-chance capability -- the same curriculum
     ``checkpoint_family_ladder_test.py`` uses, at reduced steps for a laptop runtime.
  2. J2 (:mod:`mixle.task.checkpoint_family_ladder`) walks that headline down a two-rung size ladder,
     collecting J1/G3's own compression receipts and a fresh F10 eval report per rung, gated by
     ``track_regression`` against the previous rung.
  3. I1 (:mod:`mixle.models.unified_quantizer`) quantizes the headline and every rung's REAL parameter
     tensors into measured deployment artifacts (bytes, reconstruction error, chosen method per tensor).
  4. J4 (:mod:`mixle.task.frontier_to_native`) distills a *separate* frontier/teacher model into a
     small LNS-compressed, calibrated edge student and serves it behind a two-tier ``Cascade``.
  5. F11 (:mod:`mixle.task.deploy_family`) prices every J2/I1 artifact off ONE ``CostModel`` scaled by
     real measured bytes, reports the family's cost/quality frontier, and carries J4's own served-cascade
     receipt alongside it (unmerged -- the two quality axes measure different tasks; see
     ``deploy_family``'s module docstring for why).

Nothing here is mocked or hand-waved: the headline is really trained, the ladder really compresses it,
I1 really quantizes real tensors, J4 really distills/calibrates/serves a cascade, and every number
printed below comes out of that real execution.

Run: ``python examples/frontier_family_showcase.py`` (needs ``pip install "mixle[torch]"``; no network).
"""

from __future__ import annotations

import numpy as np

from mixle.models.eval_harness import markov_transition_matrix
from mixle.models.transformer import build_causal_lm
from mixle.task.checkpoint_family_ladder import RungSpec, build_checkpoint_family
from mixle.task.deploy_family import deploy_family
from mixle.task.distill import distill_records_from_labels
from mixle.task.economics import CostModel
from mixle.task.edge import footprint
from mixle.task.frontier_to_native import build_served_cascade, distill_to_lns_student, measure_cascade_receipt

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is optional
    raise ImportError("frontier_family_showcase requires torch: pip install 'mixle[torch]'") from exc


def line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --- Stage 1: a real, trained (not random-init) headline causal LM. ------------------------------------
# Same curriculum F10's own eval suite scores (markov perplexity, modular arithmetic, parity, induction)
# so the headline has genuine above-chance capability on every axis compression could degrade.


def train_headline(
    *, seed: int, vocab: int, d_model: int, n_layer: int, n_head: int, block: int, steps: int
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)

    trans = markov_transition_matrix(vocab)
    rng = np.random.default_rng(seed + 1000)
    ctx_len = min(block, 16)
    modulus = min(vocab - 2, 10)
    plus_id, eq_id = vocab - 2, vocab - 1
    bit_len = min(block, 5)

    def batch_perplexity(bs):
        seqs = np.empty((bs, ctx_len), dtype=np.int64)
        cur = rng.integers(0, vocab, size=bs)
        seqs[:, 0] = cur
        for t in range(1, ctx_len):
            nxt = np.array([rng.choice(vocab, p=trans[c]) for c in cur])
            seqs[:, t] = nxt
            cur = nxt
        return seqs[:, :-1], seqs[:, -1]

    def batch_arithmetic(bs):
        a = rng.integers(0, modulus, size=bs)
        b = rng.integers(0, modulus, size=bs)
        target = (a + b) % modulus
        seq = np.stack([a, np.full(bs, plus_id), b, np.full(bs, eq_id)], axis=1)
        return seq, target

    def batch_parity(bs):
        bits = rng.integers(0, 2, size=(bs, bit_len))
        target = bits.sum(axis=1) % 2
        return bits, target

    def batch_induction(bs):
        seqs = rng.integers(0, vocab, size=(bs, ctx_len))
        a = rng.integers(0, vocab, size=bs)
        b = rng.integers(0, vocab, size=bs)
        b = np.where(b == a, (b + 1) % vocab, b)
        plant_pos = rng.integers(0, ctx_len - 3, size=bs)
        for i in range(bs):
            p = int(plant_pos[i])
            seqs[i, p] = a[i]
            seqs[i, p + 1] = b[i]
            seqs[i, ctx_len - 1] = a[i]
        return seqs, b

    batchers = (batch_perplexity, batch_arithmetic, batch_parity, batch_induction)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    for step in range(steps):
        x_np, y_np = batchers[step % len(batchers)](64)
        x = torch.as_tensor(x_np.astype(np.int64))
        y = torch.as_tensor(y_np.astype(np.int64))
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


# --- Stage 4: a real, separate J4 edge tier -- a distilled/LNS/calibrated student behind a cascade. ----
# A different (business-tabular) task than the causal-LM family on purpose: this is the same real
# constraint F11's own docstring documents -- J2's family and J4's edge tier measure quality on
# different tasks, so their numbers are reported side by side rather than forced onto one axis.


def _churn_truth(records):
    out = []
    for r in records:
        score = (1.5 if r["region"] == "west" else -0.5) + 0.4 * r["spend"] + 0.3 * r["visits"]
        out.append("churn" if score < 1.0 else "retain")
    return out


def _churn_records(n, seed):
    rng = np.random.RandomState(seed)
    return [
        {
            "region": rng.choice(["west", "east"]),
            "spend": float(rng.normal(2.0, 1.5)),
            "visits": int(rng.poisson(3)),
        }
        for _ in range(n)
    ]


class _Teacher:
    """A frontier stand-in: a bigger structured model the edge student is distilled from."""

    def __init__(self, frontier) -> None:
        self.frontier = frontier

    def __call__(self, records):
        return self.frontier.batch(list(records))


def build_edge_cascade_receipt(*, n_train_frontier: int, n_train_student: int, n_cal: int, n_test: int):
    records = _churn_records(n_train_frontier, seed=0)
    labels = _churn_truth(records)
    frontier = distill_records_from_labels(
        records, labels, dim=512, hidden=[128], epochs=120, lr=1e-2, seed=0, task="frontier"
    )
    teacher = _Teacher(frontier)
    teacher_bytes = footprint(frontier).bytes

    train = _churn_records(n_train_student, seed=1)
    lns_student = distill_to_lns_student(teacher, train, min_gain=1.0, seed=0, task="lns_edge_student")

    cal = _churn_records(n_cal, seed=2)
    cost = CostModel(c_frontier=1.0, c_local=0.001)
    cascade = build_served_cascade(lns_student, teacher, cal, alpha=0.1, cost=cost)

    test_records = _churn_records(n_test, seed=3)
    truth = _churn_truth(test_records)
    return measure_cascade_receipt(cascade, test_records, truth, teacher_bytes=teacher_bytes)


def main() -> None:
    vocab, d_model, n_layer, n_head, block = 23, 16, 6, 2, 12

    line("STAGE 1: train the headline checkpoint")
    headline = train_headline(
        seed=7, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block, steps=1500
    )
    headline_params = sum(p.numel() for p in headline.parameters())
    print(f"headline trained: {headline_params} real parameters")

    line("STAGE 2: J2 -- checkpoint -> family ladder")
    calib_rng = np.random.RandomState(7)
    calibration_data = torch.as_tensor(calib_rng.randint(0, vocab, size=(150, block)), dtype=torch.long)
    rung_specs = [
        RungSpec(name="rung_mid", real_target="1B-equivalent stand-in", budget=0.4, trust_region=0.4, seed=0),
        RungSpec(name="rung_edge", real_target="edge-equivalent stand-in", budget=1.5, trust_region=1.5, seed=0),
    ]
    family = build_checkpoint_family(headline, rung_specs, calibration_data=calibration_data, eval_n_examples=256)
    print(f"rungs attempted: {[r.name for r in family.rungs]}")
    print(f"halted_at: {family.halted_at!r} (None means every rung stayed within its eval budget)")
    print(
        f"total calibration samples spent: {family.total_calibration_samples} "
        f"({family.total_calibration_fraction():.2%} of a full-sampling-KD-every-rung ladder)"
    )
    for r in family.rungs:
        print(f"  {r.name}: {r.n_params} params ({r.compression_ratio:.3f}x headline) -- {r.reason}")

    line("STAGE 4: J4 -- a separate frontier -> LNS edge student -> served cascade")
    edge_receipt = build_edge_cascade_receipt(n_train_frontier=1200, n_train_student=500, n_cal=150, n_test=250)
    print(edge_receipt.summary())
    assert edge_receipt.earns_its_complexity(), "the cascade should beat the extremes it sits between"

    line("STAGE 3 + 5: I1 quantized artifacts + F11's end-to-end serve receipt")
    serve_receipt = deploy_family(
        family, headline, edge_cascade_receipt=edge_receipt, cost=CostModel(c_frontier=1.0), seed=0
    )
    print(serve_receipt.summary())

    assert len(serve_receipt.points) == 1 + len(family.rungs)
    for p in serve_receipt.points:
        assert p.artifact.quantized_bytes < p.artifact.dense_bytes, f"{p.name}: I1 artifact did not shrink"
        assert p.cost_per_request > 0.0

    line("DONE")
    print(
        f"headline + {len(family.rungs)} rung(s), each I1-quantized and priced; "
        f"edge tier served {edge_receipt.n_requests} requests, "
        f"{edge_receipt.n_escalated} escalated to the frontier; "
        f"monotone family frontier: {serve_receipt.is_monotone_frontier()}"
    )


if __name__ == "__main__":
    main()
