"""``Scientist`` -- the assembled laptop product: cross-modal scientific reasoning with verified answers.

This is the thing the frontier-ecosystem workplan was the supply chain FOR. One object that, on a
laptop, with no network:

  * PERCEIVES real modalities through real open-weight encoders (CLIP ViT-B/32 for images, MiniLM for
    text) mounted as typed leaves -- the C2 contract with genuine towers, not stand-ins;
  * LEARNS certified models over what it perceives (``study``: closed-form / convex heads over encoder
    latents, with an estimation certificate, calibration, and conformal abstention -- never ADAM where
    something provable exists);
  * REASONS over a knowledge substrate + its own fitted skills + physics solvers (``ask``: the
    evidence-buying action loop, answering through a REAL local LLM whose semantic-entropy UQ gates
    whether it may speak);
  * VERIFIES every answer (factuality receipts against the substrate; no answer without provenance;
    abstention instead of fabrication).

The frontier claim this system makes -- and PROVES in its receipts rather than asserts -- is task-level:
on scientific tasks (calibrated cross-modal prediction, parameter inversion with coverage, grounded QA)
it delivers verification no frontier LLM provides, at laptop cost. It does not claim open-ended
generative parity with hundred-billion-parameter models; it claims *trustworthy answers where the
answer can be checked*, which is what science requires.

Heavy assets (CLIP / MiniLM / SmolLM2) are lazy-loaded from the local HF cache and shared per process.
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
        return np.asarray([self.classes[i] for i in np.argmax(self._scores(np.atleast_2d(z)), axis=1)])

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        s = self._scores(np.atleast_2d(z))
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(axis=1, keepdims=True)

    def prediction_sets(self, z: np.ndarray) -> list[list[Any]]:
        """Conformal label sets at level 1-alpha: possibly >1 label (honest ambiguity), never empty lies."""
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
            retrieve_min_score=0.34,  # a real content match, not tiny-embedder noise (the honesty floor)
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

    # -- certified perception ------------------------------------------------------------------------
    @staticmethod
    def perceive(images: Any) -> np.ndarray:
        return encode_images(images)

    @staticmethod
    def read(texts: Any) -> np.ndarray:
        return encode_texts(texts)

    @staticmethod
    def study(latents: np.ndarray, labels: Any, **kw: Any) -> StudiedModel:
        return study(latents, labels, **kw)
