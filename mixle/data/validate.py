"""Check that a dataset conforms to the spec a model expects, before fitting.

``check_dataset(model, data)`` derives the model's schema (:func:`mixle.data.schema.Schema.for_model`) and,
over a sample of records, verifies each record (a) coerces to the schema's logical types and (b) lies in
the model's support (finite log-density). It returns a report (and can ``raise`` on failure) so an invalid
feed -- wrong column types, out-of-support values, malformed records -- is caught up front rather than
surfacing later as a non-finite likelihood or a cryptic error deep in EM.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DataReport:
    """Result of checking sample records against a model-derived schema and support."""

    ok: bool
    n_checked: int
    schema: list[tuple[str, str]]
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"DataReport(ok={self.ok}, checked={self.n_checked}, schema=[{', '.join(n + ':' + t for n, t in self.schema)}])"
        return head if self.ok else head + "\n  " + "\n  ".join(self.issues[:20])


def _records(data: Any):
    return data.records() if hasattr(data, "records") and callable(data.records) else data


def check_dataset(
    model: Any, data: Any, *, sample: int = 1000, check_support: bool = True, raise_on_error: bool = False
) -> DataReport:
    """Validate ``data`` against the schema/support ``model`` expects (over the first ``sample`` records).

    Records both type-coercion failures (wrong shape/dtype for a field) and, when ``check_support`` is
    True, support violations (a value the model assigns probability 0 -> ``-inf`` log-density). With
    ``raise_on_error`` the first batch of issues is raised as a ``ValueError``."""
    from mixle.data.schema import Schema

    schema = Schema.for_model(model)
    schema_desc = [(f.name, repr(f.type)) for f in schema.fields]
    recs = list(itertools.islice(_records(data), sample))
    issues: list[str] = []
    conformed: list[Any] = []
    for i, r in enumerate(recs):
        try:
            conformed.append(schema.conform_record(r))
        except Exception as exc:
            issues.append(f"record {i}: does not conform to schema ({type(exc).__name__}: {exc})")
            conformed.append(None)
    if check_support:
        for i, (r, c) in enumerate(zip(recs, conformed)):
            if c is None:
                continue
            try:
                lp = model.log_density(r)
                if not np.isfinite(lp):
                    issues.append(f"record {i}: outside the model's support (log-density {lp})")
            except Exception as exc:
                issues.append(f"record {i}: could not be scored ({type(exc).__name__}: {exc})")
    report = DataReport(ok=not issues, n_checked=len(recs), schema=schema_desc, issues=issues)
    if raise_on_error and not report.ok:
        raise ValueError("dataset does not conform to the model spec:\n" + "\n".join(issues[:20]))
    return report
