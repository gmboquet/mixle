#!/usr/bin/env python
"""Source-checkout compatibility wrapper for the installed ``mixle-reproduce`` command."""

from mixle.reproduction import (
    build_receipt,
    claim_checks,
    environment,
    evaluate_claims,
    main,
    source_tree_provenance,
    wheel_provenance,
)

__all__ = [
    "build_receipt",
    "claim_checks",
    "environment",
    "evaluate_claims",
    "main",
    "source_tree_provenance",
    "wheel_provenance",
]


if __name__ == "__main__":
    raise SystemExit(main())
