"""Grouping sessions by session attributes in a caller-controlled order.

Reports and tables must appear in a fixed, meaningful order (for example the
configured device list or the study's scenario ordering), not in whatever order
the data happens to arrive. The helpers here therefore iterate an explicit ordered
list of keys supplied by the caller and skip keys with no matching sessions.
"""
from __future__ import annotations

from typing import Callable, Hashable, TypeVar

from evaluation.shared.models import Session

_K = TypeVar("_K", bound=Hashable)


def group_sessions_ordered(
    sessions: list[Session],
    keys: list[_K],
    key_of: Callable[[Session], Hashable],
) -> list[tuple[_K, list[Session]]]:
    """Group sessions by ``key_of`` over an explicit ordered ``keys`` list.

    Returns one ``(key, matching_sessions)`` pair per key in ``keys`` order,
    omitting keys with no matching sessions. A session is grouped once by a single
    scan, so this avoids the repeated full-list rescans of ad-hoc filtering.
    """
    buckets: dict[Hashable, list[Session]] = {key: [] for key in keys}
    for session in sessions:
        bucket = buckets.get(key_of(session))
        if bucket is not None:
            bucket.append(session)
    return [(key, buckets[key]) for key in keys if buckets[key]]
