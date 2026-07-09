"""Trust receipts on a VLM -- B4, composing three subsystems around B1's frozen 3-D encoder.

The claim this demo makes checkable: BEFORE any downstream training happens on top of a frozen
vision-ish encoder, you can already ask (and answer) "does this embedding space secretly merge two
regimes a naive single-cluster reading would conflate?" -- and you can wrap the whole investigation
in a replayable audit trail. Three already-existing subsystems, composed, not reimplemented:

1. **hvis** (:mod:`mixle.utils.hvis`) -- :func:`~mixle.utils.hvis.topology.model_fit_health` and
   :func:`~mixle.utils.hvis.hvis_map` run over the *raw* (standardized) patch/volume embeddings
   B1's frozen ``Conv3d`` encoder (``examples/multimodal_stage1_demo.py``) produces for two of its
   planted structures (blob, stripe). A deliberately UNDER-provisioned single-component reference
   model (K=1: "one visual regime, everything with a bright region") stands in for what a naive
   pre-training visual taxonomy might assume; ``model_fit_health``'s merged-regime detector (a
   deterministic 2-means split of the component's fiber coordinates) catches the fact that blob and
   stripe are actually two well-separated regimes in this space -- exactly the "before any training"
   framing B4 asks for, since nothing downstream of the frozen encoder has been fit yet.

2. **The disagreement gate** (:mod:`mixle.task.disagreement`) is text/``TaskModel``-native (a small
   distilled n-gram classifier over strings) -- it doesn't fit numeric embedding vectors directly, so
   this demo reuses what generalizes (:func:`~mixle.task.disagreement.measure_disagreement_mass` is
   fully duck-typed: it only calls ``student.batch(texts)`` and compares against ``teacher_labels``,
   so it works verbatim once "texts" are stringified embeddings) and wraps the two scorers as
   :class:`~mixle.task.llm.CallableLLM`\\ s per the module's own "an LLM can be a teacher" framing --
   a "small interpreter" that reads a single embedding axis (a cheap, lossy proxy of the kind a tiny
   local model might use) versus a "frontier teacher" standing in for an expensive/frontier judge
   (nearest full-embedding centroid -- it looks at the whole vector). A small local
   :class:`RegimeDisagreementGate` gives the same duck-typed ``ood_mask`` surface
   :class:`~mixle.task.disagreement.DisagreementGate` exposes, over embeddings instead of text, so the
   shape (not the text-specific training path) still matches the module's contract.

3. **The epistemic decision journal** (:mod:`mixle.epistemic`) wraps one end-to-end run: a toy
   :class:`~mixle.epistemic.portfolio.HypothesisPortfolio` over "which regime does this embedding
   belong to" (two hypotheses: blob, stripe) is updated one :func:`~mixle.epistemic.loop.step` at a
   time as embeddings arrive, using a real :class:`~mixle.epistemic.likelihood.LikelihoodStrategy`
   (:class:`~mixle.epistemic.likelihood.CallableLikelihood`, tier ``"simulation"`` -- the embeddings
   are synthetic-planted, not a real measurement). Every step is appended to an
   :class:`~mixle.epistemic.journal.EpistemicJournal`; ``.replay()`` reconstructs the belief
   trajectory from the journal alone and ``.verify()`` confirms no stored snapshot was tampered with.

Run: ``python examples/vlm_trust_receipts_demo.py``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multimodal_stage1_demo import build_module, synthetic_volume  # noqa: E402

from mixle.epistemic.journal import EpistemicJournal  # noqa: E402
from mixle.epistemic.likelihood import CallableLikelihood  # noqa: E402
from mixle.epistemic.loop import step  # noqa: E402
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio  # noqa: E402
from mixle.stats import DiagonalGaussianDistribution, MixtureDistribution  # noqa: E402
from mixle.task.disagreement import measure_disagreement_mass  # noqa: E402
from mixle.task.llm import CallableLLM  # noqa: E402
from mixle.utils.hvis import hvis_map  # noqa: E402

BLOB, STRIPE = 0, 2  # two of B1's planted structures -- the two "regimes" this demo investigates


# -- 1. embeddings from B1's frozen encoder ----------------------------------------------------------


def encode_volumes(labels: list[int], n_per_class: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Run B1's frozen ``Conv3d`` encoder over synthetic volumes -- no training, the encoder is
    exactly the one ``examples/multimodal_stage1_demo.py`` builds and never fits."""
    import torch

    encoder = build_module(seed=0).encoder  # frozen; build_module never trains it either
    rng = np.random.RandomState(seed)
    volumes, row_labels = [], []
    for label in labels:
        for _ in range(n_per_class):
            volumes.append(synthetic_volume(label, rng))
            row_labels.append(label)
    batch = torch.tensor(np.stack(volumes)).unsqueeze(1)
    with torch.no_grad():
        embeddings = encoder(batch).numpy().astype(np.float64)
    return embeddings, row_labels


def standardize(embeddings: np.ndarray) -> np.ndarray:
    mu, sd = embeddings.mean(axis=0), embeddings.std(axis=0)
    return (embeddings - mu) / np.maximum(sd, 1e-8)


# -- 2. hvis model_fit_health / hvis_map: is the encoder merging two regimes? ------------------------


def diagnose_merged_regimes(embeddings: np.ndarray) -> Any:
    """A deliberately under-provisioned K=1 reference model ("one visual regime") over the
    standardized embeddings, diagnosed by :func:`hvis_map` BEFORE any downstream training. If blob
    and stripe are genuinely separated in the encoder's own embedding space -- which they are, both
    being architecturally distinct signal shapes a frozen conv stack responds to very differently --
    the merged-regime detector catches it from the reference model's own residual structure alone."""
    dim = embeddings.shape[1]
    naive_one_regime = MixtureDistribution(
        [DiagonalGaussianDistribution(mu=np.zeros(dim), covar=embeddings.var(axis=0) + 1e-3)], [1.0]
    )
    return hvis_map(list(embeddings), mix_model=naive_one_regime, health=True, seed=0)


# -- 3. disagreement gate: small interpreter vs. frontier teacher, over embeddings -------------------


def _embedding_to_prompt(embedding: np.ndarray) -> str:
    return ",".join(f"{v:.6f}" for v in embedding)


def _prompt_to_embedding(prompt: str) -> np.ndarray:
    return np.array([float(v) for v in prompt.split(",")], dtype=np.float64)


def make_interpreter(threshold_dim: int = 0) -> CallableLLM:
    """The 'small interpreter': a single-axis threshold rule -- a cheap, lossy proxy of the kind a
    tiny local model might use, wrapped as a :class:`~mixle.task.llm.CallableLLM` (local callable
    standing in for a small locally-hosted model, per the module's own framing). Mostly right (this
    axis correlates strongly with the true regime) but not always -- exactly the failure mode a
    disagreement gate exists to catch."""

    def fn(prompt: str, system: str | None = None) -> str:
        e = _prompt_to_embedding(prompt)
        return "stripe" if e[threshold_dim] > 0 else "blob"

    return CallableLLM(fn)


def make_frontier_teacher(centroids: dict[str, np.ndarray]) -> CallableLLM:
    """The 'frontier teacher': nearest full-embedding centroid -- looks at the whole vector, not one
    axis, standing in for an expensive/frontier judge with the whole picture."""

    def fn(prompt: str, system: str | None = None) -> str:
        e = _prompt_to_embedding(prompt)
        return min(centroids, key=lambda k: float(np.linalg.norm(e - centroids[k])))

    return CallableLLM(fn)


class _LLMStudent:
    """Duck-typed ``.batch(texts) -> labels`` wrapper so :func:`measure_disagreement_mass` (which
    only ever calls ``student.batch``) runs unmodified against a :class:`CallableLLM`."""

    def __init__(self, llm: CallableLLM) -> None:
        self.llm = llm

    def batch(self, texts: list[str]) -> list[str]:
        return [self.llm.complete(t) for t in texts]


@dataclass
class RegimeDisagreementGate:
    """Same duck-typed ``ood_mask`` surface as :class:`mixle.task.disagreement.DisagreementGate`,
    over embeddings rather than text -- the interpreter/teacher pair disagreeing IS the escalation
    signal, so no separate trained classifier is needed the way the text gate needs one."""

    interpreter: CallableLLM
    teacher: CallableLLM

    def ood_mask(self, embeddings: list[np.ndarray]) -> np.ndarray:
        prompts = [_embedding_to_prompt(e) for e in embeddings]
        interp_labels = _LLMStudent(self.interpreter).batch(prompts)
        teacher_labels = _LLMStudent(self.teacher).batch(prompts)
        return np.array([a != b for a, b in zip(interp_labels, teacher_labels)])


def run_disagreement_gate(embeddings: np.ndarray, centroids: dict[str, np.ndarray]) -> dict:
    """Guarantee at least one disagreement DETERMINISTICALLY: append one hand-built embedding, nudged
    only on the interpreter's threshold axis away from its (otherwise dominant) nearest centroid, so
    the interpreter's one-axis rule and the teacher's whole-vector rule are forced to disagree on it
    regardless of what the frozen encoder's real embeddings happen to look like on a given run."""
    interpreter = make_interpreter()
    teacher = make_frontier_teacher(centroids)

    adversarial = centroids["stripe"].copy()
    adversarial[0] = -abs(centroids["stripe"][0]) - 0.5  # flips the interpreter's one-axis rule to "blob"
    assert interpreter.complete(_embedding_to_prompt(adversarial)) == "blob"
    assert teacher.complete(_embedding_to_prompt(adversarial)) == "stripe"  # still nearest to stripe overall

    rows = [*embeddings, adversarial]
    prompts = [_embedding_to_prompt(e) for e in rows]
    student = _LLMStudent(interpreter)
    teacher_labels = _LLMStudent(teacher).batch(prompts)
    disagreement_mass = measure_disagreement_mass(student, prompts, teacher_labels)

    gate = RegimeDisagreementGate(interpreter=interpreter, teacher=teacher)
    mask = gate.ood_mask(rows)
    return {"mass": disagreement_mass, "mask": mask, "n_flagged": int(mask.sum()), "n_total": len(mask)}


# -- 4. epistemic journal: wrap the run in a replayable audit trail ----------------------------------


def run_epistemic_journal(observations: list[np.ndarray], centroids: dict[str, np.ndarray]) -> EpistemicJournal:
    """One :func:`~mixle.epistemic.loop.step` per observed embedding over a 2-hypothesis portfolio
    ("which regime does this embedding belong to"), journaled as it goes."""
    hyps = [
        Hypothesis(id="blob", payload=centroids["blob"].tolist()),
        Hypothesis(id="stripe", payload=centroids["stripe"].tolist()),
    ]
    portfolio = HypothesisPortfolio(hyps, np.array([0.5, 0.5]), w_open=0.0)

    def gaussian_kernel(hypothesis: Hypothesis, observation: np.ndarray) -> float:
        mu = np.asarray(hypothesis.payload, dtype=np.float64)
        return float(np.exp(-0.5 * float(np.sum((observation - mu) ** 2))))

    likelihood = CallableLikelihood(gaussian_kernel, tier="simulation")

    journal = EpistemicJournal()
    for i, observation in enumerate(observations):
        outcome = step(portfolio, observation, likelihood)
        journal.append(outcome, rationale=f"observed embedding {i}", timestamp=float(i))
        portfolio = outcome.portfolio_after
    return journal


# -- putting it together ------------------------------------------------------------------------------


def run_demo(*, n_per_class: int = 40, seed: int = 0) -> dict:
    """The whole B4 pipeline, importable for the smoke test. Returns every receipt this demo composes."""
    raw, row_labels = encode_volumes([BLOB, STRIPE], n_per_class=n_per_class, seed=seed)
    embeddings = standardize(raw)

    blob_mask = np.array([label == BLOB for label in row_labels])
    centroids = {"blob": embeddings[blob_mask].mean(axis=0), "stripe": embeddings[~blob_mask].mean(axis=0)}

    fitted_map = diagnose_merged_regimes(embeddings)
    gate_result = run_disagreement_gate(embeddings[:5], centroids)  # a handful is plenty for the receipt
    journal = run_epistemic_journal([embeddings[0], embeddings[-1], centroids["blob"], centroids["stripe"]], centroids)

    return {
        "fit_health": fitted_map.fit_health,
        "map_summary": fitted_map.summary(),
        "gate_result": gate_result,
        "journal": journal,
        "journal_replay": journal.replay(),
        "journal_verified": journal.verify(),
    }


def main() -> None:
    result = run_demo()

    print("=== 1. hvis model_fit_health / hvis_map over B1's frozen embeddings (before any training) ===")
    print(result["map_summary"])
    merged = [d for d in result["fit_health"]["diagnosis"] if "merged" in d]
    print(f"merged-regime findings: {len(merged)}")
    for d in merged:
        print(f"  - {d}")

    print("\n=== 2. disagreement gate: small interpreter vs. frontier teacher ===")
    gate = result["gate_result"]
    print(f"disagreement mass: {gate['mass']:.2f}  flagged {gate['n_flagged']}/{gate['n_total']} examples")
    assert gate["n_flagged"] >= 1, "the gate should have flagged the hand-built adversarial case"

    print("\n=== 3. epistemic journal: replayable audit trail over one end-to-end run ===")
    journal = result["journal"]
    print(f"{len(journal)} decision record(s) journaled")
    trajectory = result["journal_replay"]
    print(f"replay reconstructed {len(trajectory)} portfolio snapshot(s) from the journal alone")
    print(f"journal.verify(): {result['journal_verified']}")
    assert result["journal_verified"], "the journal's stored snapshots should match their content-addresses"

    print("\n=== summary: three receipts, one run ===")
    print(f"health: {'merge flagged' if merged else 'no merge flagged'} on the frozen encoder's embedding space")
    print(f"gate:   {gate['n_flagged']}/{gate['n_total']} examples escalated to the frontier teacher")
    print(f"journal: {len(journal)} record(s), verify()={result['journal_verified']} -- audit trail intact")
    print(
        "OK: hvis health diagnostic ran, the disagreement gate flagged the adversarial case, "
        "and the journal's replay/verify confirms the audit trail."
    )


if __name__ == "__main__":
    main()
