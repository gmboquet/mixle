"""Optional local scientific-assistant workflow.

``Scientist`` assembles several Mixle surfaces into one offline-oriented object:

* modality encoders for images and text, loaded from the local Hugging Face
  cache when available;
* certified heads over encoder latents through ``study``;
* a substrate-backed ``ask`` workflow using local evidence, skills, and an
  optional local language model;
* factuality and provenance checks for produced answers.

The module is intentionally optional. Heavy assets such as CLIP, MiniLM, and
SmolLM2 are lazy-loaded and shared per process, and the package sets offline
Hugging Face defaults at import time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the laptop contract: local weights only
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_CLIP_ID = "openai/clip-vit-base-patch32"
_LM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
_CACHE: dict[str, Any] = {}


def _clip():
    if "clip" not in _CACHE:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(_CLIP_ID)
        model.eval()
        _CACHE["clip"] = (model, CLIPProcessor.from_pretrained(_CLIP_ID, use_fast=True), torch)
    return _CACHE["clip"]


def _lm():
    if "lm" not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_LM_ID)
        model = AutoModelForCausalLM.from_pretrained(_LM_ID)
        model.eval()
        _CACHE["lm"] = (model, tok, torch)
    return _CACHE["lm"]


# -- real modality leaves (C2, genuine towers) ------------------------------------------------------


def encode_images(images: Any, *, batch: int = 32) -> np.ndarray:
    """CLIP ViT-B/32 image features, ``(n, 512)`` -- the real image leaf. Accepts PIL images/arrays."""
    model, proc, torch = _clip()
    imgs = list(images)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            inp = proc(images=imgs[i : i + batch], return_tensors="pt")
            v = model.get_image_features(**inp)
            out.append(v.numpy())
    return np.concatenate(out, axis=0)


def encode_texts(texts: Any) -> np.ndarray:
    """MiniLM sentence embeddings ``(n, 384)`` -- the real text leaf."""
    if "st" not in _CACHE:
        from sentence_transformers import SentenceTransformer

        _CACHE["st"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(_CACHE["st"].encode(list(texts), show_progress_bar=False))


def generate(prompt: str, *, max_new_tokens: int = 96, temperature: float = 0.0) -> str:
    """One completion from the local LLM (SmolLM2-360M-Instruct) -- the 99%-local answerer."""
    model, tok, torch = _lm()
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
    )
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            **({"temperature": temperature, "top_p": 0.9} if temperature > 0 else {}),
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


# -- the certified perception head (study) ----------------------------------------------------------


@dataclass
class StudiedModel:
    """A certified cross-modal predictor: encoder latents -> closed-form head, with its receipts."""

    head: Any  # per-class Gaussian model over latents (closed form)
    classes: list[Any]
    certificate: Any
    qhat: float  # conformal threshold on the nonconformity score (abstention rail)
    alpha: float
    class_priors: np.ndarray
    train_seconds: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def _scores(self, z: np.ndarray) -> np.ndarray:
        """Per-class log posterior (up to a constant) at latents ``z``: log prior + Gaussian log-lik."""
        out = np.stack([g.seq_log_density(g.dist_to_encoder().seq_encode(list(z))) for g in self.head], axis=1)
        return out + np.log(self.class_priors)[None, :]

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Return the most likely class for each latent vector."""
        return np.asarray([self.classes[i] for i in np.argmax(self._scores(np.atleast_2d(z)), axis=1)])

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        """Return normalized class probabilities for each latent vector."""
        s = self._scores(np.atleast_2d(z))
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(axis=1, keepdims=True)

    def prediction_sets(self, z: np.ndarray) -> list[list[Any]]:
        """Conformal label sets at level 1-alpha; ambiguous cases may return multiple labels."""
        p = self.predict_proba(z)
        return [[self.classes[j] for j in range(p.shape[1]) if 1.0 - p[i, j] <= self.qhat] for i in range(len(p))]

    def abstains(self, z: np.ndarray) -> np.ndarray:
        """True where the conformal set is not a single label -- the 'do not trust a point guess' flag."""
        return np.asarray([len(s) != 1 for s in self.prediction_sets(np.atleast_2d(z))])


def study(
    latents: np.ndarray,
    labels: Any,
    *,
    alpha: float = 0.1,
    cal_frac: float = 0.25,
    seed: int = 0,
) -> StudiedModel:
    """Fit a CERTIFIED classifier over encoder latents: closed-form Gaussian class-conditionals + a
    split-conformal abstention rail. No gradient descent anywhere -- the certificate proves it."""
    import mixle.stats as st
    from mixle.inference import certify, optimize

    z = np.asarray(latents, dtype=np.float64)
    y = np.asarray(list(labels))
    t0 = time.time()
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(z))
    n_cal = max(1, int(round(cal_frac * len(z))))
    cal_idx, fit_idx = order[:n_cal], order[n_cal:]

    classes = sorted(set(y.tolist()))
    heads = []
    priors = []
    for c in classes:
        zc = z[fit_idx][y[fit_idx] == c]
        heads.append(optimize(list(zc), st.DiagonalGaussianEstimator(dim=z.shape[1]), out=None, max_its=1))
        priors.append(len(zc))
    priors = np.asarray(priors, dtype=float)
    priors = priors / priors.sum()

    model = StudiedModel(
        head=heads,
        classes=classes,
        certificate=certify(heads[0]),
        qhat=0.0,
        alpha=alpha,
        class_priors=priors,
        train_seconds=0.0,
    )
    # split-conformal calibration of the abstention rail: nonconformity = 1 - p(true class)
    p_cal = model.predict_proba(z[cal_idx])
    idx = {c: j for j, c in enumerate(classes)}
    scores = 1.0 - p_cal[np.arange(len(cal_idx)), [idx[c] for c in y[cal_idx]]]
    k = int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1
    model.qhat = float(np.sort(scores)[min(k, len(scores) - 1)])
    model.train_seconds = time.time() - t0
    model.provenance = {"n_fit": len(fit_idx), "n_cal": len(cal_idx), "alpha": alpha, "seed": seed}
    return model


# -- edge distillation: a foundation capability -> a torch-free, KB-sized artifact -------------------


@dataclass
class EdgeArtifact:
    """A capability compressed to run on a constrained device: the student + its footprint + retention."""

    model: Any  # the deployed student: call it on a raw input, no torch / no foundation model needed
    bytes: int
    torch_free: bool
    family: str
    teacher_accuracy: float
    student_accuracy: float
    agreement: float  # fraction of inputs where the student matches the teacher
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def retention(self) -> float:
        """The share of the teacher's accuracy retained by the edge student."""
        return self.student_accuracy / self.teacher_accuracy if self.teacher_accuracy else 0.0

    def render(self) -> str:
        """Render a compact human-readable edge-artifact receipt."""
        return (
            f"edge student ({self.family}, {self.bytes} bytes, torch_free={self.torch_free}): "
            f"teacher {self.teacher_accuracy:.3f} -> student {self.student_accuracy:.3f} "
            f"({self.retention:.0%} retained, {self.agreement:.3f} agreement)"
        )


def distill_to_edge(
    teacher_predict: Any,
    train_inputs: Any,
    val_inputs: Any,
    val_truth: Any,
    *,
    max_bytes: int = 500_000,
    torch_free: bool = True,
    n_init: int = 3,
    n_iter: int = 3,
    seed: int = 0,
) -> EdgeArtifact:
    """Compress a foundation capability into a torch-free edge artifact, with a retention receipt.

    ``teacher_predict`` is the capability (a raw_input -> label callable -- e.g. CLIP zero-shot, or a
    MiniLM + certified head). Its labels on ``train_inputs`` are the distillation target; the student
    learns them from the RAW inputs under a device byte budget, so the deployed artifact needs neither
    torch nor the foundation model. The receipt measures what survives: student accuracy vs the teacher's,
    and their agreement. Boundary: this works only when the student's own features carry the
    signal (text n-grams do; raw pooled pixels do NOT recover a vision foundation model)."""
    from mixle.task.edge import DeviceSpec, distill_for_edge

    truth = np.asarray(list(val_truth))
    train_labels = [teacher_predict(x) for x in train_inputs]
    val_labels = [teacher_predict(x) for x in val_inputs]
    teacher_acc = float((np.asarray(val_labels) == truth).mean())

    res = distill_for_edge(
        None,
        list(train_inputs),
        list(val_inputs),
        DeviceSpec(torch_free=torch_free, max_bytes=max_bytes),
        train_labels=train_labels,
        val_labels=val_labels,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
    )
    pred = np.asarray([res.model(x) for x in val_inputs])
    # labels may be ints or strings depending on the teacher; compare as strings to stay type-agnostic
    student_acc = float((pred.astype(str) == truth.astype(str)).mean())
    agreement = float((pred.astype(str) == np.asarray(val_labels).astype(str)).mean())
    return EdgeArtifact(
        model=res.model,
        bytes=int(res.footprint.bytes),
        torch_free=bool(res.footprint.torch_free),
        family=res.family,
        teacher_accuracy=teacher_acc,
        student_accuracy=student_acc,
        agreement=agreement,
        provenance={"max_bytes": max_bytes, "n_train": len(train_inputs), "seed": seed},
    )


# -- research proposals + conjectures (the don't-know-but-here's-how half) ---------------------------


@dataclass
class ResearchProposal:
    """A knowledge gap made actionable: what is missing and the ranked ways to acquire it."""

    question: str
    missing: str
    nearest_knowledge: list[dict[str, Any]] = field(default_factory=list)
    options: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def best(self) -> dict[str, Any] | None:
        """Return the highest-ranked acquisition option, if one exists."""
        return self.options[0] if self.options else None

    def render(self) -> str:
        """Render the proposal as a concise research plan."""
        lines = [f"I don't know: {self.missing}."]
        if self.nearest_knowledge:
            lines.append("Closest things I do know:")
            lines += [f"  - ({k['score']}) {k['text']}" for k in self.nearest_knowledge]
        lines.append("Ways we could find out (best first):")
        lines += [f"  {i + 1}. {o['how']}  [cost ~{o['cost']}]" for i, o in enumerate(self.options[:4])]
        return "\n".join(lines)


@dataclass
class Conjecture:
    """A curiosity-generated question: explicitly NOT knowledge -- a hypothesis with a proposed test."""

    question: str
    sources: list[str] = field(default_factory=list)  # the knowledge items that sparked it
    status: str = "conjecture"  # never 'fact' until an investigation answers it with provenance
    proposal: ResearchProposal | None = None

    def render(self) -> str:
        """Render the conjecture and its best proposed test."""
        head = f"[CONJECTURE] {self.question}"
        return head if self.proposal is None else f"{head}\n  test: {self.proposal.best()['how']}"


# -- the assembled reasoner --------------------------------------------------------------------------


class Scientist:
    """The laptop cross-modal scientific reasoner (see module docstring).

    Args:
        knowledge: a :class:`~mixle.substrate.Substrate` of what it may cite (built if omitted).
        max_entropy: the local LLM's semantic-entropy gate -- above it, the model may not answer
            from its own weights (it must ground in the substrate or abstain).
    """

    def __init__(self, knowledge: Any = None, *, max_entropy: float = 0.9) -> None:
        from mixle.substrate import Substrate

        self.knowledge = knowledge if knowledge is not None else Substrate()
        self.max_entropy = float(max_entropy)
        self._skills: list[Any] = []
        self._actions: list[Any] = []

    # -- knowledge + capability mounting ------------------------------------------------------------
    def learn(self, docs: Any, *, source: str = "user") -> int:
        """Ingest documents into the citable knowledge (secrets redacted before indexing)."""
        from mixle.substrate import ingest_documents, safe_text

        clean = [safe_text(str(d)) for d in docs]
        return len(ingest_documents(self.knowledge, clean, source=source))

    def add_action(self, action: Any) -> Scientist:
        """Mount a capability (a physics solver, a simulator, a fitted skill) as a reasoner action."""
        self._actions.append(action)
        return self

    # -- the verified-answer loop --------------------------------------------------------------------
    def ask(self, question: str, *, min_confidence: float = 0.2) -> Any:
        """Answer with citations or abstain. The local LLM composes ONLY from retrieved evidence, and
        its own uncertainty is measured: an answer it cannot ground or is not confident of is withheld."""
        from mixle.substrate import Reasoner

        def answerer(q: str, evidence: str) -> str:
            # a plain extraction prompt: the abstention decision is made by RETRIEVAL confidence and the
            # FACTUALITY check, not delegated to a 360M model's self-assessment (which it does poorly).
            top = evidence.splitlines()[0] if evidence else ""
            prompt = f"Read the passage and answer.\nPassage: {top}\nQ: {q}\nA:"
            return generate(prompt, max_new_tokens=48)

        reasoner = Reasoner(
            answerer,
            substrate=self.knowledge,
            retrieve_min_score=0.34,  # substantive content match, above low-signal embedder noise
            min_confidence=min_confidence,
        )
        for a in self._actions:
            reasoner.add_action(a)
        # verify=True runs check_factuality; an answer whose claims the substrate cannot ground is
        # withdrawn -- the local model's fluency never substitutes for provenance.
        inv = reasoner.ask(question, verify=True)
        if not inv.abstained and inv.factuality is not None and inv.factuality.grounded_fraction < 0.5:
            inv.answer = None
            inv.abstained = True
            inv.note = "answer could not be grounded in the retrieved evidence"
        return inv

    # -- "I don't know, but here is how we could find out" --------------------------------------------
    def propose(self, question: str, investigation: Any = None) -> ResearchProposal:
        """Turn an abstention into a research plan: what is missing, and ranked ways to acquire it.

        This is the difference between a dead end and a scientist: an explicit "I don't know" comes back
        with the acquisition options -- add knowledge, run a mounted capability, fit a model to data,
        simulate, or delegate outward -- each with what it would take and what it would settle. The
        ranking is EIG-per-cost over the mounted actions plus the generic acquisition strategies."""
        from mixle.substrate.act import relevance_of
        from mixle.substrate.retrieve import retrieve

        # what the substrate ALMOST knows: the nearest neighbors below the answer floor name the gap
        near = retrieve(self.knowledge, question, k=3)
        neighbors = [
            {"text": it.text[:120], "score": round(float(s), 3)} for it, s in zip(near.items, near.scores) if s > 0.05
        ]

        options: list[dict[str, Any]] = []
        # 1. mounted capabilities that are topically close but did not fire / did not suffice
        for a in self._actions:
            rel = relevance_of(a, question)
            if rel > 0.0:
                options.append(
                    {
                        "how": f"run the mounted {a.kind} capability {a.name!r}",
                        "kind": a.kind,
                        "relevance": round(rel, 3),
                        "cost": a.cost,
                        "score": round(rel / max(a.cost, 1e-9), 3),
                    }
                )
        # 2. the generic acquisition strategies, priced by convention (lower-cost -> higher-cost)
        generic = [
            ("ingest the missing source into the knowledge base (learn())", "retrieve", 1.0),
            ("fit a model to relevant data and query it (create()/study())", "create", 4.0),
            ("design an experiment or simulation whose outcome decides it (simulate())", "simulate", 3.0),
            ("delegate to an external model or expert, UQ-gated (external_action)", "delegate", 8.0),
        ]
        base_rel = 0.3  # a generic strategy is always weakly applicable; ranking is by cost
        for how, kind, cost in generic:
            options.append(
                {"how": how, "kind": kind, "relevance": base_rel, "cost": cost, "score": round(base_rel / cost, 3)}
            )
        options.sort(key=lambda o: -o["score"])

        missing = (
            "no stored knowledge is close to this question"
            if not neighbors
            else "nearby knowledge exists but none of it answers the question"
        )
        return ResearchProposal(
            question=question,
            missing=missing,
            nearest_knowledge=neighbors,
            options=options,
            note=getattr(investigation, "note", "") if investigation is not None else "",
        )

    def investigate(self, question: str, **kw: Any) -> Any:
        """``ask``, but an abstention comes back WITH its research proposal attached -- never a bare no."""
        inv = self.ask(question, **kw)
        if inv.abstained:
            inv.proposal = self.propose(question, inv)
        return inv

    # -- curiosity: conjectures with proposed tests, never asserted as fact ---------------------------
    def wonder(self, topic: str | None = None, *, n: int = 3, seed: int = 0) -> list[Conjecture]:
        """Generate testable conjectures from what it knows -- curiosity with receipts attached.

        Pairs of knowledge items (optionally biased toward ``topic``) are handed to the local LLM with
        the instruction to propose a QUESTION or HYPOTHESIS connecting them. Every output is labeled a
        CONJECTURE and carries a proposed test (the research-proposal machinery), and is checked NOT to
        already be answerable from the substrate -- curiosity about what it does not know, not
        rediscovery of what it does."""
        rng = np.random.RandomState(seed)
        items = [i for i in self.knowledge.all() if i.text]
        if topic:
            from mixle.substrate.retrieve import retrieve

            hits = retrieve(self.knowledge, topic, k=max(4, n * 2))
            items = [i for i in hits.items if i.text] or items
        if len(items) < 2:
            return []

        out: list[Conjecture] = []
        seen: set[str] = set()
        attempts = 0
        while len(out) < n and attempts < n * 4:
            attempts += 1
            a, b = (items[i] for i in rng.choice(len(items), size=2, replace=False))
            prompt = (
                f"Fact A: {a.text}\nFact B: {b.text}\n\n"
                "Propose ONE short, testable scientific question that connects these two facts. "
                "Reply with just the question."
            )
            q = generate(prompt, max_new_tokens=48, temperature=0.7).strip().split("\n")[0]
            for prefix in ("question:", "q:", "hypothesis:"):
                if q.lower().startswith(prefix):
                    q = q[len(prefix) :].strip()
            if not q or q.lower() in seen or len(q) < 12:
                continue
            seen.add(q.lower())
            probe = self.ask(q)
            if not probe.abstained:
                continue  # it already knows -- that is rediscovery, not curiosity
            out.append(
                Conjecture(
                    question=q,
                    sources=[a.id, b.id],
                    status="conjecture",
                    proposal=self.propose(q),
                )
            )
        return out

    # -- certified perception ------------------------------------------------------------------------
    @staticmethod
    def perceive(images: Any) -> np.ndarray:
        """Encode images into the shared scientific latent space."""
        return encode_images(images)

    @staticmethod
    def read(texts: Any) -> np.ndarray:
        """Encode texts into the shared scientific latent space."""
        return encode_texts(texts)

    @staticmethod
    def study(latents: np.ndarray, labels: Any, **kw: Any) -> StudiedModel:
        """Fit a certified perception head over latent vectors."""
        return study(latents, labels, **kw)

    @staticmethod
    def distill_to_edge(
        teacher_predict: Any, train_inputs: Any, val_inputs: Any, val_truth: Any, **kw: Any
    ) -> EdgeArtifact:
        """Compress a foundation capability into a torch-free edge artifact (see :func:`distill_to_edge`)."""
        return distill_to_edge(teacher_predict, train_inputs, val_inputs, val_truth, **kw)
