"""Structured extraction (mixle.task.extract): a distilled sequence tagger scrapes typed fields from text.

The real task the toy classifier wasn't: pull {id, amount, date, vendor} out of messy transaction lines with a
trained bi-GRU tagger -- a learnable replacement for a regex scraper. The student should hit high field-level F1
on held-out text (including formats/values it never saw) and survive a fresh-process reload.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.extract import distill_extractor, extraction_f1, tokenize  # noqa: E402

FIELDS = ["id", "amount", "date", "vendor"]
VENDORS = ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli", "Stark", "Wayne"]


def _make(seed, n=400):
    """Synthetic transaction lines in varied templates; the 'teacher' returns the gold field spans."""
    rng = np.random.RandomState(seed)
    rows = []
    templates = [
        "INV-{id} {vendor} charged ${amount} on {date}",
        "Payment to {vendor} of ${amount} ref {id} dated {date}",
        "{date} | {vendor} | ${amount} | invoice {id}",
        "Receipt {id}: {vendor} ${amount} ({date})",
    ]
    for _ in range(n):
        vid = f"{rng.randint(1000, 9999)}"
        vendor = str(rng.choice(VENDORS))
        amount = f"{rng.randint(1, 999)}.{rng.randint(0, 99):02d}"
        date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        tmpl = str(rng.choice(templates))
        text = tmpl.format(id=vid, vendor=vendor, amount=amount, date=date)
        rows.append((text, {"id": vid, "amount": amount, "date": date, "vendor": vendor}))
    return rows


def _teacher_factory(gold_map):
    def teacher(texts):
        return [gold_map[t] for t in texts]

    return teacher


class TokenizeTest(unittest.TestCase):
    def test_spans_reconstruct_text(self):
        text = "INV-1234 Acme ${45.20}"
        for tok, s, e in tokenize(text):
            self.assertEqual(text[s:e], tok)


class ExtractionTest(unittest.TestCase):
    def _train(self, seed=0, epochs=120):
        rows = _make(seed)
        gold = {t: g for t, g in rows}
        teacher = _teacher_factory(gold)
        texts = [t for t, _ in rows]
        model = distill_extractor(teacher, texts, FIELDS, epochs=epochs, seed=0)
        return model, teacher

    def test_learns_to_extract_held_out(self):
        model, _ = self._train()
        self.assertGreaterEqual(model.meta["train_f1"], 0.9)
        test_rows = _make(seed=999)
        test_texts = [t for t, _ in test_rows]
        test_gold = [g for _, g in test_rows]
        f1 = extraction_f1(model, test_gold, test_texts)
        self.assertGreaterEqual(f1, 0.85)  # generalizes to unseen ids/amounts/dates

    def test_confidence_high_in_format_low_off_format(self):
        model, _ = self._train(epochs=120)
        io = model.adapter
        (rec_in, conf_in) = io.predict_with_confidence(model.model, ["INV-3333 Acme $9.99 on 2026-02-02"])[0]
        (_rec_out, conf_out) = io.predict_with_confidence(model.model, ["?!? nothing structured here at all ???"])[0]
        self.assertEqual(len(rec_in), 4)  # all fields found on a familiar format
        self.assertGreater(conf_in, conf_out)  # confidence drops on an unfamiliar line

    def test_returns_field_dict(self):
        # epochs=30 already reaches train_f1==1.0 on this fixture (verified empirically down to epochs=15
        # before the vendor/amount fields start dropping, epochs=10 misses vendor); 30 keeps a solid margin
        # while cutting training cost ~4x vs. the 120 used by the accuracy-threshold tests in this file.
        model, _ = self._train(epochs=30)
        out = model("Receipt 7777: Acme $12.34 (2026-05-05)")
        self.assertEqual(set(out).issubset(set(FIELDS)), True)
        self.assertEqual(out.get("vendor"), "Acme")
        self.assertEqual(out.get("amount"), "12.34")

    def test_fresh_process_reload(self):
        # this test only checks save/load round-trip equality (before == after), never model accuracy,
        # so epochs is not load-bearing for the claim -- verified the round-trip still holds bit-for-bit
        # down to epochs=3; kept at 10 (still a real trained model) for a >>10x margin over that floor.
        model, _ = self._train(epochs=10)
        text = "Payment to Globex of $88.10 ref 4321 dated 2026-03-03"
        before = model(text)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "extractor")
            model.save(path)
            out_file = os.path.join(d, "out.json")
            script = (
                "import torch; torch.set_num_threads(1)\n"
                "import json\n"
                "import mixle.task  # registers the seq_tagger builder + extraction adapter\n"
                "from mixle.task.model import TaskModel\n"
                f"m = TaskModel.load({path!r})\n"
                f"json.dump(m({text!r}), open({out_file!r}, 'w'))\n"
            )
            env = dict(os.environ, PYTHONPATH=os.getcwd())
            subprocess.run([sys.executable, "-c", script], check=True, env=env, cwd=os.getcwd())
            import json

            with open(out_file) as f:
                after = json.load(f)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
