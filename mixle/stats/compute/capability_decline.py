"""Typed signal for data-independent optimized-kernel route negotiation."""


class KernelCapabilityDeclinedError(RuntimeError):
    """An optimized kernel cannot represent a model's static structure.

    Factories may catch this exception while selecting a route. Runtime data
    errors and implementation failures must use their natural exception types
    and must never be interpreted as permission to change algorithms.
    """

