"""Pool execution primitives for offloading selected units of work.

The pool API keeps local execution as the default. A job is sent to a backend
only when a planner or caller provides a reason, an estimated cost, and a
budget. The default :class:`LocalBackend` executes the same protocol in-process,
so workflows can use the pool abstraction before a remote GPU backend exists.
"""

from __future__ import annotations

from mixle.pool.core import Backend, LocalBackend, PoolJob, PoolResult, submit

__all__ = ["PoolJob", "PoolResult", "Backend", "LocalBackend", "submit"]
