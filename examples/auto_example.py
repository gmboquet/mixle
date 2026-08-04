"""``get_estimator``: derive an estimator tree from raw Python data, with no schema declared.

Three rows of mixed junk -- an int, an int that is sometimes ``None``, a string, and a variable-length
bag of ``(str, int)`` pairs -- are handed to :func:`mixle.utils.automatic.get_estimator`, which walks
the data and builds the matching estimator tree: a Composite over the four fields, an Optional wrapper
where a value went missing, a Categorical for the string, and a Sequence of Composites for the bag.

Takeaway: you do not write a schema and you do not one-hot anything. The shape of the data IS the
model spec, and the estimator that comes back is an ordinary estimator -- usable with ``initialize``
and ``estimate`` (or ``optimize``) exactly like a hand-built one. Print the fitted model to see the
tree the detector inferred. For the higher-level one-liner that also runs the fit, see
``quickstart_example.py`` (``mixle.propose`` / ``optimize``).
"""

import numpy as np

from mixle.inference import estimate, initialize
from mixle.stats import *
from mixle.utils.automatic import get_estimator

if __name__ == "__main__":
    # Note row 0's `None`: a genuinely missing second field, which the detector models as Optional
    # rather than inventing a sentinel value.
    data = [
        (1, None, "a", [("a", 1), ("b", 2)]),
        (3, 2, "b", [("a", 1), ("b", 2), ("c", 3)]),
        (3.1, 2, "a", [("a", 1), ("b", 2), ("c", 3)]),
    ]

    est = get_estimator(data, pseudo_count=1.0e-4)
    # `initialize` seeds parameters from the data; `estimate` then does one closed-form fit pass.
    init = initialize(data, est, np.random.RandomState(1))
    model = estimate(data, est, prev_estimate=init)

    print(str(model))
