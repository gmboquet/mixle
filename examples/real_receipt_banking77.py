"""The first REAL-data receipt: Banking77 (77 intents, real customer queries) through the solve loop.

Everything else in examples/ uses synthetic teachers; this one runs the loop against a pinned public
dataset. The "frontier" is an oracle stand-in (the dataset's gold labels, priced per call like an API),
so costs are modeled rather than billed. It prints a scorecard and an escalation-decay curve as fresh
measurements. The tracked source intentionally embeds no prior headline result. Run receipts become
release evidence only when they bind the exact candidate, dataset digests, dependencies, command,
duration, and output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mixle.task import scorecard, solve

BANKING77_SOURCE_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
BANKING77_FILES = {
    "train": {
        "url": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{BANKING77_SOURCE_COMMIT}/banking_data/train.csv"
        ),
        "sha256": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        "rows": 10003,
    },
    "test": {
        "url": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{BANKING77_SOURCE_COMMIT}/banking_data/test.csv"
        ),
        "sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        "rows": 3080,
    },
}


class Banking77UnavailableError(RuntimeError):
    """The immutable dataset could not be fetched from its authoritative source."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_banking77():
    """Fetch immutable source files, validate their bytes and split geometry, then parse them."""
    try:
        from datasets import DownloadManager, load_dataset
    except ImportError as exc:
        raise Banking77UnavailableError("datasets is not installed") from exc
    try:
        paths = DownloadManager().download({split: spec["url"] for split, spec in BANKING77_FILES.items()})
    except Exception as exc:  # noqa: BLE001 -- connector failures are reclassified; integrity failures are not
        raise Banking77UnavailableError(f"Banking77 download failed: {type(exc).__name__}: {exc}") from exc
    for split, path in paths.items():
        actual = _sha256(Path(path))
        expected = BANKING77_FILES[split]["sha256"]
        if actual != expected:
            raise ValueError(f"Banking77 {split} SHA-256 mismatch: expected {expected}, received {actual}")
    dataset = load_dataset("csv", data_files=paths)
    for split, spec in BANKING77_FILES.items():
        if len(dataset[split]) != spec["rows"]:
            raise ValueError(
                f"Banking77 {split} split has {len(dataset[split])} rows; expected {spec['rows']} "
                f"from source commit {BANKING77_SOURCE_COMMIT}"
            )
        if set(dataset[split].column_names) != {"text", "category"}:
            raise ValueError(f"Banking77 {split} columns do not match the pinned text/category schema")
    return dataset


def run(
    *,
    n_seed: int = 3000,
    n_round: int = 1000,
    n_rounds: int = 6,
    n_test: int = 1500,
    student: str = "mlp",
    verbose: bool = True,
    dataset=None,
) -> dict:
    """Run the Banking77 solve loop and return its measured results.

    The defaults reproduce the headline run in the module docstring. The sizes are parameters so a fast
    bounded version can be gated in CI (see ``real_receipt_banking77_smoke_test``); ``main`` keeps the
    full defaults. Returns ``{"card", "accuracy", "rounds"}`` where ``rounds`` is a list of
    ``{"escalation", "accuracy"}`` per improve round.
    """
    ds = load_banking77() if dataset is None else dataset
    train_all = [(r["text"], r["category"]) for r in ds["train"]]
    test = [(r["text"], r["category"]) for r in ds["test"]]
    rng = np.random.RandomState(0)
    rng.shuffle(train_all)

    gold = dict(train_all) | dict(test)

    def oracle(t: str) -> str:  # the frontier stand-in: always right, priced per call
        return gold[t]

    seed_texts = [t for t, _ in train_all[:n_seed]]
    rounds = [[t for t, _ in train_all[n_seed + i * n_round : n_seed + (i + 1) * n_round]] for i in range(n_rounds)]
    test_texts = [t for t, _ in test]

    kw = {"student": "generative", "pseudo_count": 4.0} if student == "generative" else {"epochs": 250}
    sol = solve(oracle, seed_texts, alpha=0.1, seed=0, **kw)
    card = scorecard(
        sol,
        oracle,
        test_texts[:n_test],
        task_truth=[gold[text] for text in test_texts[:n_test]],
        student_cost=0.0001,
        teacher_cost=0.03,
        task="banking77 intents (77 classes)",
    )
    if verbose:
        print(card.table())
        print("\nescalation-decay: serve fresh queries, harvest, improve — per round")
        print("round | escalation | end-to-end accuracy")

    round_results = []
    for i, chunk in enumerate(rounds):
        if not chunk:
            continue
        before = sol.cascade.stats.n_escalated
        answers = [sol(t) for t in chunk]
        esc = (sol.cascade.stats.n_escalated - before) / len(chunk)
        acc = float(np.mean([a == gold[t] for a, t in zip(answers, chunk)]))
        round_results.append({"escalation": esc, "accuracy": acc})
        if verbose:
            print(f"  {i}   |   {esc:.3f}    |   {acc:.3f}")
        sol.improve()

    if verbose:
        print("\nevery number above was measured by this run — change the seed and check.")
    return {
        "card": card,
        "metrics": card.as_dict(),
        "rounds": round_results,
        "dataset": {
            "name": "BANKING77",
            "source_commit": BANKING77_SOURCE_COMMIT,
            "splits": {
                split: {"rows": spec["rows"], "sha256": spec["sha256"]}
                for split, spec in BANKING77_FILES.items()
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generative", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run the bounded release-bundle configuration")
    parser.add_argument("--json", action="store_true", help="emit a strict JSON result")
    args = parser.parse_args(argv)
    sizes = {"n_seed": 1155, "n_round": 40, "n_rounds": 1, "n_test": 60} if args.smoke else {}
    result = run(
        student="generative" if args.generative or args.smoke else "mlp",
        verbose=not args.json,
        **sizes,
    )
    if args.json:
        print(
            json.dumps(
                {"artifact": "mixle.banking77_reproduction/v1", **{k: result[k] for k in ("metrics", "rounds", "dataset")}},
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
