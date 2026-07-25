"""Canonical identities for executable inference artifacts.

The helpers in this module deliberately fail when an object cannot be represented without an
address-bearing ``repr``. A missing identity is not interchangeable with a stable identity in a
receipt, replay request, or reusable skill.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import platform
import sys
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from mixle.data.hashing import _canonical


def canonical_digest(value: Any) -> str:
    """Return a full SHA-256 digest of content with a stable, address-free identity."""
    return hashlib.sha256(_canonical(_stable(value))).hexdigest()


def _package_version(module_name: str) -> str | None:
    package = module_name.split(".", 1)[0]
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        module = sys.modules.get(package)
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else None


def _code_descriptor(code: types.CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code,
        "consts": [_code_descriptor(value) if isinstance(value, types.CodeType) else _stable(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _function_descriptor(function: types.FunctionType) -> dict[str, Any]:
    referenced_globals: dict[str, Any] = {}
    for name in function.__code__.co_names:
        if name not in function.__globals__:
            continue
        value = function.__globals__[name]
        if isinstance(value, types.ModuleType):
            referenced_globals[name] = {
                "module": value.__name__,
                "version": _package_version(value.__name__),
            }
        elif isinstance(value, (str, bytes, bool, int, float, type(None), tuple, list, dict)):
            referenced_globals[name] = _stable(value)
        elif isinstance(value, types.FunctionType) and value is not function:
            referenced_globals[name] = {
                "function": f"{value.__module__}.{value.__qualname__}",
                "code": _code_descriptor(value.__code__),
            }
        elif isinstance(value, type):
            referenced_globals[name] = {"type": f"{value.__module__}.{value.__qualname__}"}
    closure = []
    for cell in function.__closure__ or ():
        try:
            closure.append(_stable(cell.cell_contents))
        except ValueError:
            closure.append({"empty_cell": True})
    return {
        "kind": "python-function",
        "module": function.__module__,
        "qualname": function.__qualname__,
        "module_source_digest": _module_source_digest(function),
        "code": _code_descriptor(function.__code__),
        "defaults": _stable(function.__defaults__),
        "kwdefaults": _stable(function.__kwdefaults__),
        "closure": closure,
        "globals": referenced_globals,
    }


def _module_source_digest(function: types.FunctionType) -> str | None:
    """Digest the containing source file so transitive same-module helpers are also bound."""
    try:
        path = inspect.getsourcefile(function)
        if path is None:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _stable(value: Any, _active: set[int] | None = None) -> Any:
    """Represent ``value`` without process-specific addresses."""
    if value is Ellipsis:
        return {"ellipsis": True}
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex, np.generic)):
        return value.item() if isinstance(value, np.generic) else value
    if isinstance(value, slice):
        return {
            "slice": [
                _stable(value.start, _active),
                _stable(value.stop, _active),
                _stable(value.step, _active),
            ]
        }
    if isinstance(value, range):
        return {"range": [value.start, value.stop, value.step]}
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, types.CodeType):
        return _code_descriptor(value)
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__, "version": _package_version(value.__name__)}
    if isinstance(value, types.FunctionType):
        return _function_descriptor(value)
    if isinstance(value, types.BuiltinFunctionType):
        return {
            "kind": "builtin",
            "module": value.__module__,
            "qualname": value.__qualname__,
            "version": _package_version(value.__module__),
        }
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}

    active = set() if _active is None else _active
    identity = id(value)
    if identity in active:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                "mapping": [
                    (_stable(key, active), _stable(item, active))
                    for key, item in sorted(value.items(), key=lambda pair: _canonical(_stable(pair[0])))
                ]
            }
        if isinstance(value, (tuple, list)):
            return {"sequence": [_stable(item, active) for item in value], "type": type(value).__name__}
        if isinstance(value, (set, frozenset)):
            items = [_stable(item, active) for item in value]
            items.sort(key=_canonical)
            return {"set": items, "type": type(value).__name__}
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {field.name: _stable(getattr(value, field.name), active) for field in fields(value)},
            }
        if isinstance(value, types.MethodType):
            return {
                "method": _function_descriptor(value.__func__),
                "self": _stable(value.__self__, active),
            }
        if hasattr(value, "__dict__"):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": {
                    key: _stable(item, active)
                    for key, item in sorted(vars(value).items())
                    if not key.startswith("__") and not callable(item)
                },
            }
    finally:
        active.remove(identity)
    raise TypeError(f"{type(value).__module__}.{type(value).__qualname__} has no stable canonical identity")


def object_state_digest(value: Any) -> str:
    """Digest an object's canonical type and state."""
    return canonical_digest(_stable(value))


def implementation_digest(callable_object: Any) -> str:
    """Digest executable code, defaults, referenced constants, closure, and bound state."""
    if isinstance(callable_object, types.MethodType):
        descriptor = {
            "method": _function_descriptor(callable_object.__func__),
            "bound_state": _stable(callable_object.__self__),
        }
    elif isinstance(callable_object, types.FunctionType):
        descriptor = _function_descriptor(callable_object)
    elif isinstance(callable_object, types.BuiltinFunctionType):
        descriptor = _stable(callable_object)
    elif callable(callable_object):
        call = next(
            (namespace["__call__"] for cls in type(callable_object).__mro__ if "__call__" in (namespace := vars(cls))),
            None,
        )
        if not isinstance(call, types.FunctionType):
            raise TypeError(f"{type(callable_object).__name__} has no inspectable Python implementation")
        descriptor = {
            "call": _function_descriptor(call),
            "bound_state": _stable(callable_object),
        }
    else:
        raise TypeError(f"{type(callable_object).__name__} is not callable")
    return canonical_digest(descriptor)


def dependency_manifest() -> dict[str, Any]:
    """Return the executable platform and core dependency versions bound by inference receipts."""
    packages = {}
    for package in ("mixle", "numpy", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }


def dependency_digest() -> str:
    """Digest :func:`dependency_manifest`."""
    return canonical_digest(dependency_manifest())
