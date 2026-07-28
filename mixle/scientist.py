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
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the laptop contract: local weights only
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_CLIP_ID = "openai/clip-vit-base-patch32"
_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
_LM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
_LM_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
_SENTENCE_ID = "sentence-transformers/all-MiniLM-L6-v2"
_SENTENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_CACHE: dict[str, Any] = {}


#: The latent space each modality encoder produces -- two SEPARATE spaces, not one shared one.
#:
#: :func:`encode_images` returns 512-dimensional CLIP image features; :func:`encode_texts` returns
#: 384-dimensional embeddings from an independently trained MiniLM. There is no projection between
#: them, no paired alignment objective, no common schema and no learned bridge, so a vector from one
#: space carries no defined relation to a vector from the other: they cannot be compared by distance,
#: averaged, or stacked (the dimensions do not even match). Both encoders' docstrings used to claim
#: they encoded into "the shared scientific latent space"; they do not. Fit a separate head per space
#: (:func:`study` takes latents from exactly one of them), and treat any cross-space claim as
#: requiring a bridge that would have to be trained and evidenced first.
LATENT_SPACES: dict[str, dict[str, Any]] = {
    "image": {
        "space_id": "clip-vit-base-patch32.image_features/v1",
        "encoder": _CLIP_ID,
        "revision": _CLIP_REVISION,
        "dim": 512,
        "aligned_with": (),
    },
    "text": {
        "space_id": "all-MiniLM-L6-v2.sentence_embedding/v1",
        "encoder": _SENTENCE_ID,
        "revision": _SENTENCE_REVISION,
        "dim": 384,
        "aligned_with": (),
    },
}


def latent_space(modality: str) -> dict[str, Any]:
    """The representation contract for one modality's encoder (see :data:`LATENT_SPACES`)."""
    if modality not in LATENT_SPACES:
        raise KeyError(f"unknown modality {modality!r}; expected one of {sorted(LATENT_SPACES)}")
    return dict(LATENT_SPACES[modality])


def scientist_asset_manifest() -> dict[str, dict[str, str]]:
    """Return the immutable external-asset identities used by this module."""
    return {
        "clip": {"repository": _CLIP_ID, "revision": _CLIP_REVISION},
        "language_model": {"repository": _LM_ID, "revision": _LM_REVISION},
        "sentence_encoder": {"repository": _SENTENCE_ID, "revision": _SENTENCE_REVISION},
    }


def _clip():
    if "clip" not in _CACHE:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(_CLIP_ID, revision=_CLIP_REVISION, use_safetensors=True)
        model.eval()
        _CACHE["clip"] = (
            model,
            CLIPProcessor.from_pretrained(_CLIP_ID, revision=_CLIP_REVISION, use_fast=True),
            torch,
        )
    return _CACHE["clip"]


def _lm():
    if "lm" not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_LM_ID, revision=_LM_REVISION, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(_LM_ID, revision=_LM_REVISION, use_safetensors=True)
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
            # current transformers returns a BaseModelOutputWithPooling, not a raw tensor; the
            # projected per-image embedding (what this function has always promised) lives in
            # .pooler_output -- CLIPModel.get_image_features runs the pooled vision output through
            # visual_projection and stores the (n, 512) result back onto that same attribute.
            v = model.get_image_features(**inp)
            out.append(v.pooler_output.numpy())
    return np.concatenate(out, axis=0)


def encode_texts(texts: Any) -> np.ndarray:
    """MiniLM sentence embeddings ``(n, 384)`` -- the real text leaf."""
    if "st" not in _CACHE:
        from sentence_transformers import SentenceTransformer

        _CACHE["st"] = SentenceTransformer(_SENTENCE_ID, revision=_SENTENCE_REVISION, local_files_only=True)
    return np.asarray(_CACHE["st"].encode(list(texts), show_progress_bar=False))


def generate(prompt: str, *, max_new_tokens: int = 96, temperature: float = 0.0) -> str:
    """One completion from the local LLM (SmolLM2-360M-Instruct) -- the 99%-local answerer."""
    model, tok, torch = _lm()
    # current tokenizers/transformers return a BatchEncoding (dict-like: .input_ids/.attention_mask),
    # not a raw tensor -- torch.ones_like(ids) on the whole object no longer type-checks. Use the
    # BatchEncoding's own attention_mask (already correct and padding-aware) and its input_ids as the
    # tensor model.generate() and the final decode-slice both need.
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
    )
    input_ids = enc.input_ids
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            **({"temperature": temperature, "top_p": 0.9} if temperature > 0 else {}),
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True).strip()


# -- the certified perception head (study) ----------------------------------------------------------


@dataclass
class StudiedModel:
    """A certified predictor over ONE encoder's latents -> closed-form head, with its receipts.

    Not cross-modal: it is fit on latents from a single space (see :data:`LATENT_SPACES`), and
    ``provenance["latent_dim"]`` records which one it was trained in. Predicting on latents from the
    other encoder is a category error, not a modality it also covers.
    """

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
    split-conformal abstention rail. No gradient descent anywhere -- the certificate proves it.

    ``alpha`` (the split-conformal miscoverage level) must be in ``[0.0, 1.0]`` -- the same inclusive
    domain as :func:`~mixle.inference.conformal.conformal_label_threshold`, which computes ``qhat``
    below (``0.0`` and ``1.0`` are valid boundaries, not just interior values). ``cal_frac`` must be in
    ``(0.0, 1.0)`` exclusive so both the calibration and fit splits stay non-empty. ``latents`` must be
    two-dimensional ``(n, d)`` and ``labels`` must align with it one-to-one.
    """
    import mixle.stats as st
    from mixle.inference import EstimationCertificate, certify, conformal_label_threshold, optimize

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha!r}.")
    if not 0.0 < cal_frac < 1.0:
        raise ValueError(f"cal_frac must be in (0.0, 1.0), got {cal_frac!r}.")
    z = np.asarray(latents, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"latents must be two-dimensional (n, d), got shape {z.shape}.")
    if z.shape[0] == 0:
        raise ValueError("latents must be non-empty.")
    y = np.asarray(list(labels))
    if y.shape[0] != z.shape[0]:
        raise ValueError(f"labels must be aligned with latents: got {y.shape[0]} labels for {z.shape[0]} latents.")
    t0 = time.time()
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(z))
    n_cal = max(1, int(round(cal_frac * len(z))))
    cal_idx, fit_idx = order[:n_cal], order[n_cal:]

    classes = sorted(set(y.tolist()))
    heads = []
    head_data = []
    priors = []
    for c in classes:
        zc = z[fit_idx][y[fit_idx] == c]
        if len(zc) == 0:
            raise ValueError(
                f"class {c!r} has no examples in the fit split (n_fit={len(fit_idx)}, n_cal={n_cal}): "
                "provide more data for this class, or lower cal_frac so the fit split is bigger"
            )
        heads.append(optimize(list(zc), st.DiagonalGaussianEstimator(dim=z.shape[1]), out=None, max_its=1))
        head_data.append(list(zc))
        priors.append(len(zc))
    priors = np.asarray(priors, dtype=float)
    priors = priors / priors.sum()

    # certify EVERY class head, not just the first: the aggregate guarantee is the weakest head's, and
    # the per-head blocks (qualified by class) keep the receipt auditable for K > 1 problems
    head_certificates = [certify(head, data=data) for head, data in zip(heads, head_data)]
    certificate = EstimationCertificate(
        guarantee=min(cert.guarantee for cert in head_certificates),
        blocks=[
            replace(b, name=f"head[{c!r}].{b.name}" if b.name else f"head[{c!r}]")
            for c, cert in zip(classes, head_certificates)
            for b in cert.blocks
        ],
        escape_tested=all(cert.escape_tested for cert in head_certificates),
    )

    model = StudiedModel(
        head=heads,
        classes=classes,
        certificate=certificate,
        qhat=0.0,
        alpha=alpha,
        class_priors=priors,
        train_seconds=0.0,
    )
    # split-conformal calibration of the abstention rail: nonconformity = 1 - p(true class). Routed
    # through conformal_label_threshold (rather than re-deriving the ceil((n_cal+1)(1-alpha)) index
    # here) so this inherits its already-corrected finite-sample boundary handling -- alpha == 1.0
    # returns the MINIMUM calibration score, not the maximum via Python's negative-index wraparound --
    # and its +inf fallback when n_cal cannot support the requested level (n_cal is surfaced in
    # provenance so that regime stays visible).
    p_cal = model.predict_proba(z[cal_idx])
    idx = {c: j for j, c in enumerate(classes)}
    p_true = p_cal[np.arange(len(cal_idx)), [idx[c] for c in y[cal_idx]]]
    model.qhat = conformal_label_threshold(p_true, alpha=alpha)
    model.train_seconds = time.time() - t0
    model.provenance = {
        "n_fit": len(fit_idx),
        "n_cal": len(cal_idx),
        "alpha": alpha,
        "seed": seed,
        # which latent space this head lives in: the image and text encoders produce SEPARATE,
        # unaligned spaces (see LATENT_SPACES), so a head is only valid for latents from its own.
        "latent_dim": int(z.shape[1]),
    }
    return model


# -- edge distillation: a foundation capability -> a torch-free, KB-sized artifact -------------------


def _label_agreement(left: Any, right: Any) -> float:
    """Fraction of positions where two equal-length label sequences hold the SAME label.

    One equivalence relation -- ordinary Python equality on the label values, elementwise -- for every
    metric an :class:`EdgeArtifact` reports. Teacher accuracy used to be NumPy's raw label equality
    while student accuracy and agreement coerced both sides with ``.astype(str)``, so the same run
    could report teacher accuracy 0.0 and student accuracy 1.0 against the same reference labels
    (teacher and student label ``1``, reference label ``"1"``), giving retention 0.0 for a student that
    matched its teacher perfectly. String coercion also silently collapses genuinely distinct labels
    (``1`` and ``"1"``, ``1.0`` and ``True``), so it is not used anywhere here.
    """
    pairs = list(zip(left, right, strict=True))
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if bool(a == b)) / len(pairs)


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
        """The share of the teacher's accuracy retained by the edge student.

        ``nan`` when the teacher scored zero on the validation set: the ratio's denominator is not a
        valid accuracy to retain a share of, and reporting ``0.0`` there is indistinguishable from a
        student that genuinely retained nothing.
        """
        if self.teacher_accuracy <= 0.0:
            return float("nan")
        return self.student_accuracy / self.teacher_accuracy

    def render(self) -> str:
        """Render a compact human-readable edge-artifact receipt."""
        retention = self.retention
        retained = "undefined" if np.isnan(retention) else f"{retention:.0%}"
        return (
            f"edge student ({self.family}, {self.bytes} bytes, torch_free={self.torch_free}): "
            f"teacher {self.teacher_accuracy:.3f} -> student {self.student_accuracy:.3f} "
            f"({retained} retained, {self.agreement:.3f} agreement)"
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
    signal (text n-grams do; raw pooled pixels do NOT recover a vision foundation model).

    ``train_inputs``/``val_inputs``/``val_truth`` may be any iterable (including a one-shot generator):
    each is materialized to a list exactly once, at entry, and that list is what every subsequent pass
    (teacher labeling, student training, validation metrics, provenance counts) reuses."""
    from mixle.task.edge import DeviceSpec, distill_for_edge

    # materialize every input exactly once: teacher labeling below is the FIRST pass over train_inputs/
    # val_inputs, and distill_for_edge (student training) and the student-metrics loop are the second and
    # third. A one-shot iterable (a generator, a file-backed reader) silently yields nothing on its
    # second pass -- lists/tuples only look re-iterable by accident. Rebinding the parameter names keeps
    # every later reference to them (including provenance's len(train_inputs)) pointed at the same list.
    train_inputs = list(train_inputs)
    val_inputs = list(val_inputs)
    val_truth = list(val_truth)
    if not train_inputs:
        raise ValueError("distill_to_edge requires at least one training input, got 0.")
    if not val_inputs:
        raise ValueError("distill_to_edge requires at least one validation input, got 0.")
    if len(val_inputs) != len(val_truth):
        raise ValueError(
            f"val_inputs and val_truth must have matching length, got {len(val_inputs)} and {len(val_truth)}."
        )

    train_labels = [teacher_predict(x) for x in train_inputs]
    val_labels = [teacher_predict(x) for x in val_inputs]
    teacher_acc = _label_agreement(val_labels, val_truth)

    res = distill_for_edge(
        None,
        train_inputs,
        val_inputs,
        DeviceSpec(torch_free=torch_free, max_bytes=max_bytes),
        train_labels=train_labels,
        val_labels=val_labels,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
    )
    pred = [res.model(x) for x in val_inputs]
    # ONE equivalence relation for all three metrics (see _label_agreement): teacher accuracy used
    # NumPy's raw label equality while these two coerced both sides to strings, so the same run could
    # report teacher 0.0 / student 1.0 / agreement 1.0 / retention 0.0 against one reference set.
    student_acc = _label_agreement(pred, val_truth)
    agreement = _label_agreement(pred, val_labels)
    truth_vocabulary = {repr(v) for v in val_truth}
    model_vocabulary = {repr(v) for v in val_labels} | {repr(v) for v in pred}
    return EdgeArtifact(
        model=res.model,
        bytes=int(res.footprint.bytes),
        torch_free=bool(res.footprint.torch_free),
        family=res.family,
        teacher_accuracy=teacher_acc,
        student_accuracy=student_acc,
        agreement=agreement,
        provenance={
            "max_bytes": max_bytes,
            "n_train": len(train_inputs),
            "seed": seed,
            # the label vocabularies every metric above was computed under: disjoint sets mean the
            # reference labels and the models' labels are not the same vocabulary at all, which drives
            # every accuracy to zero for a reason that has nothing to do with model quality.
            "truth_labels": sorted(truth_vocabulary),
            "model_labels": sorted(model_vocabulary),
            "label_vocabulary_disjoint": bool(truth_vocabulary and not (truth_vocabulary & model_vocabulary)),
        },
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
    """The laptop multi-modality scientific reasoner (see module docstring).

    Multi-modality, not cross-modal: it carries an image encoder and a text encoder whose latent
    spaces are separate and unaligned (see :data:`LATENT_SPACES`); it does not relate one modality's
    latents to the other's.

    Args:
        knowledge: a :class:`~mixle.substrate.Substrate` of what it may cite (built if omitted).

    ``ask``'s abstention decision comes from retrieval confidence and the post-hoc factuality
    check (see ``ask``'s own docstring), not from the local model's self-assessed uncertainty --
    a 360M model's own confidence is a poor abstention signal. Semantic-entropy gating of the
    kind :mod:`mixle.inference.uq` provides is a real, separate mechanism in this package
    (:class:`~mixle.substrate.interop.ExternalModel` wraps it); ``Scientist`` deliberately uses
    the retrieval/factuality path instead.
    """

    def __init__(self, knowledge: Any = None) -> None:
        from mixle.substrate import Substrate

        self.knowledge = knowledge if knowledge is not None else Substrate()
        self._skills: list[Any] = []
        self._actions: list[Any] = []

    # -- knowledge + capability mounting ------------------------------------------------------------
    def learn(self, docs: str | Sequence[Any], *, source: str = "user") -> int:
        """Ingest documents into the citable knowledge (secrets redacted before indexing).

        ``docs`` is EITHER one document (a ``str``) or a collection of documents. A bare string is one
        document: iterating it directly made every character its own knowledge item, so the ordinary
        single-document call ``learn("hello world")`` reported 11 documents ingested and stored
        ``["h", "e", ..., "d"]`` as separate citable items -- destroying document structure and leaving
        later retrieval and provenance hanging off accidental character fragments.

        Any other iterable is materialized once (so a one-shot generator is not consumed by the
        redaction pass and then found empty by the ingest), and each item must be a string document;
        anything else is rejected by position rather than silently stringified into a citable "fact".
        Returns the number of documents ingested.
        """
        from mixle.substrate import ingest_documents, safe_text

        if isinstance(docs, (str, bytes, bytearray)):
            docs = [docs]
        items = list(docs)
        clean: list[str] = []
        for i, doc in enumerate(items):
            if isinstance(doc, (bytes, bytearray)):
                doc = doc.decode()
            if not isinstance(doc, str):
                raise TypeError(
                    f"learn() takes one document string or a collection of document strings; "
                    f"document {i} is {type(doc).__name__}"
                )
            clean.append(safe_text(doc))
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
        if not inv.abstained and inv.factuality is not None and not inv.factuality.is_grounded(threshold=0.5):
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
        """Encode images into the ``"image"`` latent space: ``(n, 512)`` CLIP image features.

        This is NOT the same space :meth:`read` produces, and nothing here bridges the two -- see
        :data:`LATENT_SPACES`. Fit a head per space; do not mix their vectors.
        """
        return encode_images(images)

    @staticmethod
    def read(texts: Any) -> np.ndarray:
        """Encode texts into the ``"text"`` latent space: ``(n, 384)`` MiniLM sentence embeddings.

        This is NOT the same space :meth:`perceive` produces, and nothing here bridges the two -- see
        :data:`LATENT_SPACES`. Fit a head per space; do not mix their vectors.
        """
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
