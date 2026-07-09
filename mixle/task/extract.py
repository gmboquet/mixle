"""Structured extraction: distill a teacher into a compact sequence tagger for typed text fields.

Use this module when brittle regular expressions or hand-written parsers are no longer enough for
semi-structured text such as invoices, support tickets, orders, or clinical notes. The student model learns
to extract fields such as ``id``, ``amount``, or ``date`` from labeled examples and can be retrained when
the source format changes. The model is token-level: tokenizer and orthographic features, then an embedding,
bidirectional GRU, per-token BIO tags, and a decoder back to ``{field: value}``. Labels can come from an LLM
teacher, a rule-based teacher, or gold annotations.

``distill_extractor(teacher, texts, fields)`` returns a callable :class:`~mixle.task.model.TaskModel`:
``model(text) -> {field: value}``. It saves through the same durable artifact (vocab + tag scheme in the
manifest, weights in safetensors) and loads in a fresh process. ``extraction_f1`` scores span-level fidelity.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.task.artifact import register_builder
from mixle.task.model import TaskModel

_TOKEN_RE = re.compile(r"\d+|[^\W\d_]+|[^\w\s]", re.UNICODE)
PAD, UNK = 0, 1
_N_FEATS = 7


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(token, start, end)`` triples: runs of digits, letters, or single punctuation."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _token_features(tok: str) -> list[float]:
    """Orthographic features that help extraction generalize to unseen tokens (numbers, caps, length)."""
    return [
        float(tok.isdigit()),
        float(tok.isalpha()),
        float(tok[:1].isupper()),
        float(tok.isupper()),
        float(any(c.isdigit() for c in tok)),
        min(len(tok), 12) / 12.0,
        float(bool(re.fullmatch(r"[^\w\s]", tok))),  # punctuation
    ]


def build_seq_tagger(vocab_size: int, n_tags: int, *, d_model: int = 64, hidden: int = 64, n_feats: int = _N_FEATS):
    """A bidirectional-GRU sequence tagger: ``(token_ids, features) -> per-token tag logits``."""
    import torch
    import torch.nn as nn

    class SeqTagger(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
            self.gru = nn.GRU(d_model + n_feats, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * hidden, n_tags)

        def forward(self, ids: Any, feats: Any) -> Any:
            h = torch.cat([self.emb(ids.long()), feats], dim=-1)
            out, _ = self.gru(h)
            return self.head(out)

    return SeqTagger()


register_builder("mixle.seq_tagger", build_seq_tagger)


class ExtractionIO:
    """``text -> {field: value}``: tokenize, tag (BIO), decode spans back to substrings of the original text."""

    kind = "extraction"

    def __init__(self, vocab: dict[str, int], fields: list[str], *, max_len: int = 128) -> None:
        self.vocab = dict(vocab)
        self.fields = list(fields)
        self.max_len = int(max_len)
        # tag scheme: O=0, then B-/I- per field
        self.tags = ["O"] + [f"{p}-{f}" for f in self.fields for p in ("B", "I")]
        self.tag_index = {t: i for i, t in enumerate(self.tags)}

    # --- encoding ---
    def _encode_tokens(self, toks: list[str]) -> tuple[list[int], list[list[float]]]:
        ids = [self.vocab.get(t.lower(), UNK) for t in toks]
        feats = [_token_features(t) for t in toks]
        return ids, feats

    def _batch(self, list_of_tokens: list[list[str]]):
        import torch

        m = max((len(t) for t in list_of_tokens), default=1)
        m = max(1, min(m, self.max_len))
        ids = np.zeros((len(list_of_tokens), m), dtype=np.int64)
        feats = np.zeros((len(list_of_tokens), m, _N_FEATS), dtype=np.float32)
        mask = np.zeros((len(list_of_tokens), m), dtype=bool)
        for i, toks in enumerate(list_of_tokens):
            toks = toks[:m]
            tid, tf = self._encode_tokens(toks)
            ids[i, : len(tid)] = tid
            feats[i, : len(tf)] = tf
            mask[i, : len(toks)] = True
        return torch.from_numpy(ids), torch.from_numpy(feats), mask

    # --- decoding ---
    def _decode(self, text: str, spans: list[tuple[str, int, int]], tag_ids: np.ndarray) -> dict[str, str]:
        out: dict[str, str] = {}
        i, n = 0, min(len(spans), len(tag_ids))
        while i < n:
            tag = self.tags[tag_ids[i]]
            if tag.startswith("B-"):
                field = tag[2:]
                start = spans[i][1]
                end = spans[i][2]
                j = i + 1
                while j < n and self.tags[tag_ids[j]] == f"I-{field}":
                    end = spans[j][2]
                    j += 1
                if field not in out:  # first occurrence wins
                    out[field] = text[start:end]
                i = j
            else:
                i += 1
        return out

    def predict(self, module: Any, text: str) -> dict[str, str]:
        """Extract fields from a single text record."""
        return self.predict_batch(module, [text])[0]

    def predict_batch(self, module: Any, texts: list[str]) -> list[dict[str, str]]:
        """Extract fields from a batch of text records."""
        import torch

        spans_per = [tokenize(t) for t in texts]
        toks_per = [[s[0] for s in spans] for spans in spans_per]
        ids, feats, mask = self._batch(toks_per)
        module.eval()
        with torch.no_grad():
            logits = module(ids, feats).cpu().numpy()
        tag_ids = logits.argmax(axis=-1)
        return [self._decode(texts[i], spans_per[i], tag_ids[i]) for i in range(len(texts))]

    def predict_with_confidence(self, module: Any, texts: list[str]) -> list[tuple[dict[str, str], float]]:
        """Extract each record and a confidence in ``[0, 1]``: the min per-token tag probability over tagged tokens.

        A low confidence or a missing field is an explicit signal that the format may be unfamiliar and should
        be escalated. Returns ``0.0`` when nothing was tagged.
        """
        import torch

        spans_per = [tokenize(t) for t in texts]
        toks_per = [[s[0] for s in spans] for spans in spans_per]
        ids, feats, mask = self._batch(toks_per)
        module.eval()
        with torch.no_grad():
            logits = module(ids, feats)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        tag_ids = probs.argmax(axis=-1)
        out = []
        for i in range(len(texts)):
            n = min(len(spans_per[i]), tag_ids.shape[1])
            tagged = [probs[i, k, tag_ids[i, k]] for k in range(n) if tag_ids[i, k] != 0]
            conf = float(min(tagged)) if tagged else 0.0
            out.append((self._decode(texts[i], spans_per[i], tag_ids[i]), conf))
        return out

    # --- persistence ---
    def to_spec(self) -> dict[str, Any]:
        """Serialize the extraction vocabulary, fields, and maximum sequence length."""
        return {"kind": self.kind, "vocab": self.vocab, "fields": self.fields, "max_len": self.max_len}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> ExtractionIO:
        """Reconstruct extraction IO from an artifact spec."""
        return cls(spec["vocab"], spec["fields"], max_len=spec.get("max_len", 128))


def _as_batched(teacher: Callable[..., Any]) -> Callable[[list[str]], list[dict[str, str]]]:
    def batched(texts: list[str]) -> list[dict[str, str]]:
        out = teacher(texts)
        if isinstance(out, (list, tuple)) and len(out) == len(texts):
            return [dict(o) for o in out]
        return [dict(teacher(t)) for t in texts]

    return batched


def _bio_labels(
    text: str, spans: list[tuple[str, int, int]], extraction: dict[str, str], io: ExtractionIO
) -> list[int]:
    """Turn a teacher's ``{field: value}`` into per-token BIO tag ids by locating each value's char span."""
    labels = [0] * len(spans)
    for field, value in extraction.items():
        if field not in io.fields or not value:
            continue
        pos = text.find(value)
        if pos < 0:
            continue
        lo, hi = pos, pos + len(value)
        inside = [k for k, (_t, s, e) in enumerate(spans) if s >= lo and e <= hi]
        for rank, k in enumerate(inside):
            labels[k] = io.tag_index[f"{'B' if rank == 0 else 'I'}-{field}"]
    return labels


def distill_extractor(
    teacher: Callable[..., Any],
    texts: Sequence[str],
    fields: Sequence[str],
    *,
    max_vocab: int = 5000,
    d_model: int = 64,
    hidden: int = 64,
    epochs: int = 60,
    lr: float = 5e-3,
    max_len: int = 128,
    seed: int = 0,
    device: str = "cpu",
    task: str = "",
) -> TaskModel:
    """Distill a teacher's extractions into a local sequence tagger; return ``model(text) -> {field: value}``."""
    import torch

    texts = [str(t) for t in texts]
    fields = list(fields)
    extractions = _as_batched(teacher)(texts)

    spans_per = [tokenize(t) for t in texts]
    from collections import Counter

    counts: Counter[str] = Counter()
    for spans in spans_per:
        counts.update(s[0].lower() for s in spans)
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, _ in counts.most_common(max_vocab):
        vocab.setdefault(tok, len(vocab))

    io = ExtractionIO(vocab, fields, max_len=max_len)
    cfg = {"vocab_size": len(vocab), "n_tags": len(io.tags), "d_model": d_model, "hidden": hidden}

    # build padded training tensors + label matrix
    toks_per = [[s[0] for s in spans] for spans in spans_per]
    ids, feats, mask = io._batch(toks_per)
    m = ids.shape[1]
    y = np.zeros((len(texts), m), dtype=np.int64)
    for i, (text, spans) in enumerate(zip(texts, spans_per)):
        lab = _bio_labels(text, spans[:m], extractions[i], io)
        y[i, : len(lab)] = lab
    yt = torch.from_numpy(y).to(device)
    maskt = torch.from_numpy(mask).to(device)

    torch.manual_seed(seed)
    module = build_seq_tagger(**cfg).to(device)
    ids, feats = ids.to(device), feats.to(device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    module.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = module(ids, feats)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), yt.reshape(-1), reduction="none")
        loss = (loss * maskt.reshape(-1).float()).sum() / maskt.sum().clamp(min=1)
        loss.backward()
        opt.step()

    student = TaskModel(
        module,
        io,
        builder="mixle.seq_tagger",
        config=cfg,
        task=task or "distilled field extractor",
        meta={"distilled": True, "fields": fields, "n_examples": len(texts), "vocab_size": len(vocab)},
    )
    student.meta["train_f1"] = extraction_f1(student, extractions, texts)
    return student


def extraction_f1(model: TaskModel, gold: Sequence[dict[str, str]], texts: Sequence[str]) -> float:
    """Micro-averaged field-level F1: a field counts as correct when the extracted value exactly matches gold."""
    pred = model.batch(list(texts))
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        for field, value in g.items():
            if p.get(field) == value:
                tp += 1
            else:
                fn += 1
        for field in p:
            if field not in g:
                fp += 1
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0
