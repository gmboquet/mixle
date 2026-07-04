"""The pool plane (workstream H) -- offload a unit of work to a small GPU pool, get an artifact home.

99% local by default; a block or verb is offloaded only when the planner names the reason and the
economics price it worth the round-trip. v1 ships the abstraction (:class:`PoolJob`, :class:`Backend`,
:func:`submit`) and a :class:`LocalBackend` (the pool degraded to this machine) so it works end-to-end
and degrades gracefully to all-local. Real GPU backends plug into the same protocol behind the
budget + confirm rails.
"""

from __future__ import annotations

from mixle.pool.core import Backend, LocalBackend, PoolJob, PoolResult, submit

__all__ = ["PoolJob", "PoolResult", "Backend", "LocalBackend", "submit"]
