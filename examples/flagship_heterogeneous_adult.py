"""Flagship A -- heterogeneous records in native form, on real public data (UCI Adult / Census Income).

Classification: evidence -- dataset: UCI Adult / Census Income (~32k rows, downloaded once). The
held-out generalization number below is a measured result on real data, not a synthetic stand-in.

Everything about this row is mixed: integer ``age`` and ``hours.per.week`` next to categorical
``workclass``, ``education``, ``sex``, and ``income``. Mixle takes those tuples as-is -- no one-hot, no
manual schema -- infers a model over them, and fits it in one ``optimize`` call. The honest receipt is a
generalization number: the mean log-density on a held-out split, next to the train split, so you can see
the model captured structure rather than memorizing.

Run: ``python examples/flagship_heterogeneous_adult.py`` (downloads Adult once, ~32k rows).
"""

from __future__ import annotations

import numpy as np

_FIELDS = ("age", "workclass", "education", "hours.per.week", "sex", "income")


def run(*, n_train: int = 4000, n_test: int = 1000, seed: int = 0, verbose: bool = True) -> dict:
    """Fit a heterogeneous model on real Adult records and return the train/held-out receipt.

    Sizes are parameters so a fast bounded version can be gated in CI. Returns
    ``{"model_type", "n_fields", "n_train", "n_test", "train_mean_log_density", "test_mean_log_density"}``.
    """
    from datasets import load_dataset

    from mixle.inference import optimize

    ds = load_dataset("scikit-learn/adult-census-income", split="train")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ds))[: n_train + n_test]
    records = [tuple(ds[int(i)][f] for f in _FIELDS) for i in order]
    train, test = records[:n_train], records[n_train : n_train + n_test]

    model = optimize(train, out=None)
    train_ll = float(np.mean([model.log_density(r) for r in train]))
    test_ll = float(np.mean([model.log_density(r) for r in test]))

    receipt = {
        "model_type": type(model).__name__,
        "n_fields": len(_FIELDS),
        "n_train": len(train),
        "n_test": len(test),
        "train_mean_log_density": train_ll,
        "test_mean_log_density": test_ll,
    }
    if verbose:
        print(f"fields: {_FIELDS}")
        print(f"model : {receipt['model_type']} over {receipt['n_fields']} heterogeneous fields")
        print(f"mean log-density  train={train_ll:.3f}  held-out={test_ll:.3f}")
        print("held-out close to train => the model generalized, not memorized.")
    return receipt


def main() -> None:
    run()


if __name__ == "__main__":
    main()
