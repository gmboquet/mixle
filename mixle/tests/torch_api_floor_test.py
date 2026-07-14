"""Worklist D8.6 -- validate the Torch APIs mixle uses (a CPU correctness receipt).

D8.6 asks that the supported Torch lower bound be tested against the APIs actually used,
and that CUDA/FSDP claims not rest solely on CPU simulation. The GPU parity/memory/2-GPU
jobs require hardware this CI lacks; what *is* collectable here, on CPU, is a correctness
receipt: every ``torch.*`` symbol mixle references must resolve on the installed Torch,
and a curated set of critical nested APIs must be present. This catches a typo'd or
removed API before a user hits it, and substantiates the ``torch>=2.4`` floor used
by the FSDP2 and DCP distributed-training surface.

The honest split between this CPU correctness receipt and the still-pending GPU receipts
is recorded in ``release-checklists/0.8.0-gpu-claims.md``, whose consistency this file
also checks.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="D8.6 Torch-API check requires the torch extra")

pytestmark = pytest.mark.torch

ROOT = Path(__file__).resolve().parent.parent.parent
MIXLE_SRC = ROOT / "mixle"
GPU_LEDGER = ROOT / "release-checklists" / "0.8.0-gpu-claims.md"

_TORCH_REF = re.compile(r"\btorch\.([a-zA-Z_][a-zA-Z0-9_]*)")

# `torch.*` spellings that are NOT the torch package. safetensors exposes a submodule
# imported as `torch` (safetensors.torch.save_model), which the regex cannot distinguish.
_NOT_TORCH_PACKAGE = {"save_model", "load_model"}

# Critical nested APIs the neural leaves depend on -- checked explicitly.
_CRITICAL_NESTED = (
    "torch.nn.functional.cross_entropy",
    "torch.nn.Linear",
    "torch.nn.Module",
    "torch.optim.Adam",
    "torch.log_softmax",
    "torch.einsum",
    "torch.no_grad",
    "torch.cuda.is_available",
    "torch.distributed.checkpoint.async_save",
    "torch.distributed.device_mesh.init_device_mesh",
    "torch.distributed.fsdp.fully_shard",
    "torch.distributed.tensor.parallel.parallelize_module",
)


def _first_level_refs() -> set[str]:
    refs: set[str] = set()
    for path in MIXLE_SRC.rglob("*.py"):
        if "test" in path.name:
            continue
        for match in _TORCH_REF.finditer(path.read_text(encoding="utf-8", errors="ignore")):
            refs.add(match.group(1))
    return refs


def test_every_referenced_torch_api_resolves() -> None:
    refs = _first_level_refs() - _NOT_TORCH_PACKAGE
    missing = sorted(a for a in refs if not hasattr(torch, a))
    assert not missing, (
        f"mixle references torch APIs that do not exist on torch {torch.__version__}: "
        f"{missing} -- a typo, a removed API, or an API newer than the declared floor"
    )


def test_critical_nested_apis_resolve() -> None:
    for dotted in _CRITICAL_NESTED:
        obj = torch
        module_name = "torch"
        for seg in dotted.split(".")[1:]:
            if hasattr(obj, seg):
                obj = getattr(obj, seg)
                module_name = f"{module_name}.{seg}"
                continue
            module_name = f"{module_name}.{seg}"
            try:
                obj = importlib.import_module(module_name)
            except ModuleNotFoundError:
                pytest.fail(f"{dotted} is not available on torch {torch.__version__}")


def test_gpu_ledger_separates_correctness_from_gpu_receipts() -> None:
    """The GPU-claims ledger must exist and keep CUDA/FSDP claims honest."""
    if not GPU_LEDGER.is_file():
        pytest.skip("gpu-claims ledger not found")
    text = GPU_LEDGER.read_text(encoding="utf-8").lower()
    assert "correctness receipt" in text and "gpu receipt" in text, (
        "ledger must separate correctness (CPU) receipts from GPU receipts"
    )
    assert "unverified" in text, "ledger must mark GPU-unverified claims as such"
    assert "fsdp" in text and "cuda" in text, "ledger must cover the CUDA and FSDP claims"
