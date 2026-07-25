"""Fail-safe loading for optional compiled engine extensions."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptionalExtension:
    """Result of loading one optional binary module and its required ABI symbols."""

    module_name: str
    values: tuple[Any, ...]
    status: str
    diagnostic: str | None

    @property
    def available(self) -> bool:
        return self.status == "available"


def load_optional_extension(module_name: str, symbols: tuple[str, ...]) -> OptionalExtension:
    """Load an optional extension without making a stale binary break the fallback.

    A genuinely absent module is reported as ``absent``. Loader/import failures and
    missing ABI symbols are reported as ``incompatible`` with the original diagnostic.
    Unexpected exceptions raised after the loader succeeds are deliberately not caught:
    an implementation defect must remain visible instead of masquerading as an absent
    accelerator.
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        status = "absent" if exc.name == module_name else "incompatible"
        return OptionalExtension(module_name, (), status, f"{type(exc).__name__}: {exc}")
    except (ImportError, OSError) as exc:
        return OptionalExtension(module_name, (), "incompatible", f"{type(exc).__name__}: {exc}")

    missing = tuple(symbol for symbol in symbols if not hasattr(module, symbol))
    if missing:
        return OptionalExtension(
            module_name,
            (),
            "incompatible",
            f"compiled extension is missing required symbol(s): {', '.join(missing)}",
        )
    return OptionalExtension(module_name, tuple(getattr(module, symbol) for symbol in symbols), "available", None)
