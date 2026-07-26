"""Small collection and grouping utilities shared by legacy Mixle helpers.

These functions provide deterministic mapping, inverse mapping, grouping,
counting, and reduce-by-key operations for examples and older estimator code
that work over plain Python sequences.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from os import PathLike
from typing import TypeVar

import numpy as np

T = TypeVar("T")
T1 = TypeVar("T1")


def map_to_integers(x: Sequence[T], val_map: dict[T, int]) -> list[int]:
    """Map sequence of type T to integers.

    Args:
        x (Sequence[T]): Sequence of type T to be mapped to integers.
        val_map (Dict[T, int]): Dictionary mappings for type T to unique integer values.

    Returns:
        Returns x mapped to a list of integers.

    """
    rv = [None] * len(x)
    for i, u in enumerate(x):
        if u not in val_map:
            val_map[u] = len(val_map)
        rv[i] = val_map[u]
    return rv


def get_inv_map(val_map: dict[T, T1], *, multi: bool = False) -> dict[T1, T] | dict[T1, list[T]]:
    """Obtain an inverse dictionary mapping without silently losing collisions.

    Args:
        val_map (Dict[T1, T]): Dictionary mapping keys to values.
        multi: Return every source key in an insertion-ordered list for each
            value. When false, repeated values make the mapping non-invertible
            and raise ``ValueError``.

    Returns:
        Inverse mapping of val_map (value -> key), or a value -> keys
        multi-map when ``multi=True``.

    """
    if not isinstance(multi, bool):
        raise TypeError("multi must be a bool")
    grouped: dict[T1, list[T]] = defaultdict(list)
    for key, value in val_map.items():
        grouped[value].append(key)
    if multi:
        return dict(grouped)
    collisions = [value for value, keys in grouped.items() if len(keys) > 1]
    if collisions:
        raise ValueError(
            "mapping is not invertible because multiple keys map to "
            f"{collisions[0]!r}; pass multi=True to retain every key"
        )
    return {value: keys[0] for value, keys in grouped.items()}


def text_file(f: str | PathLike[str], *, encoding: str = "utf-8") -> list[str]:
    """Open a file and split by newline.

    Args
        f: File to be read-in and parsed.

    Returns:
        List of strings split on newline character.

    """
    with open(f, encoding=encoding) as fin:
        rv = fin.read()

    if rv is not None and len(rv) > 0 and rv[-1] == "\n":
        return rv[:-1].split("\n")
    else:
        return rv.split("\n")


def reduce_by_key(f: Callable[[T1, T1], T1], x: Sequence[tuple[T, T1]]) -> dict[T, T1]:
    """Reduce sequence of tuple of key value pairs under grouping function f.

    Args:
        f (Callable[[T1, T1], T1]): Function for reducing keys.
        x (Sequence[Tuple[T, T1]): A sequence of key/value pairs.

    Returns:
        Dictionary mapping key types T to value types T1.

    """
    rv = dict()

    for key, val in x:
        if key in rv:
            rv[key] = f(rv[key], val)
        else:
            rv[key] = val

    return rv


def sum_by_key(x: Sequence[tuple[T, T1]]) -> dict[T, T1]:
    """Sum values and return dictionary of items with their respective summed values.

    Args:
        x (Sequence[Tuple[T, T1]]): A sequence of tuples of key and value pairs.

    Returns:
        Dictionary of keys with summed values.

    """
    rv = dict()

    for key, val in x:
        if key in rv:
            rv[key] += val
        else:
            rv[key] = val

    return rv


def group_by_key(x: Sequence[tuple[T, T1]]) -> dict[T, list[T1]]:
    """Group keys and return dictionary of items with their respective values aggregated as a list.

    Args:
        x (Sequence[Tuple[T, T1]]): A sequence of tuples of key and value pairs.

    Returns:
        Dictionary mapping keys to list of values for respective keys.

    """
    rv = defaultdict(list)

    for key, val in x:
        rv[key].append(val)

    return rv


def group_by(f: Callable[[T], T1], x: Sequence[T]) -> dict[T1, list[T]]:
    """Maps values in x to key from mapping f. Dictionary mapping keys to list of grouped values in x is returned.

    Args:
        f (Callable[[T], T1]): Function mapping type T to its group 'key' of type T1.
        x (Sequence[T]): Sequence of values to be grouped.

    Returns:
        Dictionary mapping group id 'keys' (type T1) to Lists of values (type T).

    """
    rv = defaultdict(list)

    for val in x:
        key = f(val)
        rv[key].append(val)

    return rv


def count_by_value(x: Sequence[T] | np.ndarray) -> dict[T, int]:
    """Count the number of observations of a given value in arg 'x'.

    Args:
        x (Sequence[T]): A sequence of type T or numpy array of type T.

    Returns:
        Dictionary mapping value (type T) to value-count.

    """
    rv = dict()

    for u in x:
        rv[u] = rv.get(u, 0) + 1

    return rv


def flat_map(f: Callable[[T], Sequence[T1]], x: Sequence[T]) -> list[T1]:
    """Map values of x under mapping f().

    Args:
        f (Callable[[T], T1]): Maps values of x to sequence of type T1.
        x (Sequence[T]): Seuquence to be mapped under f.

    Returns:
        List of mapped type T1.

    """
    return [u for v in x for u in f(v)]


def least_occurring(x: Sequence[T], count: int | None = None, percent: float | None = None, keep_freq: bool = True):
    """Return the least frequent values by count or percentile cutoff."""
    if count is not None and percent is not None:
        raise ValueError("specify only one of count or percent")
    if count is not None:
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or int(count) < 0:
            raise ValueError("count must be a nonnegative integer")
        count = int(count)
    if percent is not None:
        if isinstance(percent, (bool, np.bool_)) or not isinstance(percent, (int, float, np.integer, np.floating)):
            raise ValueError("percent must be a finite number between 0 and 1")
        percent = float(percent)
        if not np.isfinite(percent) or not 0.0 <= percent <= 1.0:
            raise ValueError("percent must be a finite number between 0 and 1")

    cnt_map = list(count_by_value(x).items())
    cnt_map.sort(key=lambda item: item[1])

    if count is not None:
        n = min(len(cnt_map), count)
    elif percent is not None:
        n = 0 if percent == 0.0 else max(int(len(cnt_map) * percent), 1)
    else:
        return list(x)

    vals = [item[0] for item in cnt_map[:n]]

    if keep_freq:
        vset = set(vals)
        return [u for u in x if u in vset]  # always return a list (was a lazy filter object)
    return vals
