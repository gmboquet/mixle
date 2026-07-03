"""Replace a regex scraper with a trained model: an LLM extracts fields once, a tiny local tagger learns it.

The task hardcoded logic usually owns -- scraping structured fields out of messy text -- done as a *trained*
model instead. An LLM teacher (here a local ``CallableLLM``; swap in ``OpenAICompatLLM(base_url, model)``)
extracts ``{id, amount, date, vendor}`` from example lines; ``distill_extractor`` trains a bi-GRU sequence
tagger to reproduce it; the result is a local ``model(text) -> {field: value}`` that runs with no LLM, no
network, and can be *retrained* when the format drifts (unlike a brittle regex).

Run: ``python task_extraction_example.py``  (needs ``pip install "mixle[torch]"``).
"""

from __future__ import annotations

import re

import numpy as np

from mixle.task import CallableLLM, distill_extractor, extraction_f1, llm_extractor

FIELDS = ["id", "amount", "date", "vendor"]
VENDORS = ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli", "Stark", "Wayne"]
TEMPLATES = [
    "INV-{id} {vendor} charged ${amount} on {date}",
    "Payment to {vendor} of ${amount} ref {id} dated {date}",
    "{date} | {vendor} | ${amount} | invoice {id}",
    "Receipt {id}: {vendor} ${amount} ({date})",
]


def make_lines(seed: int, n: int) -> list[str]:
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        out.append(
            str(rng.choice(TEMPLATES)).format(
                id=rng.randint(1000, 9999),
                vendor=str(rng.choice(VENDORS)),
                amount=f"{rng.randint(1, 999)}.{rng.randint(0, 99):02d}",
                date=f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            )
        )
    return out


def stub_llm(prompt, system=None):
    """Stand-in for a real extraction LLM: parses the line and returns the fields as JSON (verbatim substrings)."""
    text = prompt.split("Text:", 1)[-1].split("JSON:")[0]
    out = {}
    if m := re.search(r"(?:INV-|ref |invoice |Receipt )(\d{4})", text):
        out["id"] = m.group(1)
    if m := re.search(r"\$(\d+\.\d{2})", text):
        out["amount"] = m.group(1)
    if m := re.search(r"(\d{4}-\d{2}-\d{2})", text):
        out["date"] = m.group(1)
    if m := re.search(r"\b(" + "|".join(VENDORS) + r")\b", text):
        out["vendor"] = m.group(1)
    import json

    return json.dumps(out)


def main() -> None:
    teacher = llm_extractor(CallableLLM(stub_llm), FIELDS)
    train = make_lines(1, 400)

    print("1) the LLM extracts fields from a few example lines (the expensive teacher)")
    print(f"   e.g. {train[0]!r}\n        -> {teacher([train[0]])[0]}")

    print("\n2) distill a tiny local sequence tagger to reproduce it")
    model = distill_extractor(teacher, train, FIELDS, epochs=150, seed=0, task="transaction field extractor")
    print(f"   train F1: {model.meta['train_f1']:.3f}")

    print("\n3) the local model scrapes fields with no LLM, generalizing to unseen values")
    test = make_lines(999, 300)
    print(f"   held-out F1: {extraction_f1(model, teacher(test), test):.3f}")
    for line in make_lines(7, 3):
        print(f"   {line!r}\n     -> {model(line)}")


if __name__ == "__main__":
    main()
