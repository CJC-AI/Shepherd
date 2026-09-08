import random
from datetime import datetime, timedelta


def random_datetime(
    start: datetime,
    end: datetime,
    rng: random.Random,
) -> datetime:
    """
    Return a random timezone-aware datetime between start and end.
    """

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError(
            "start and end must be timezone-aware"
        )

    if end < start:
        raise ValueError(
            "end must be greater than or equal to start"
        )

    if start == end:
        return start

    duration_seconds = int(
        (end - start).total_seconds()
    )

    offset_seconds = rng.randint(
        0,
        duration_seconds,
    )

    return start + timedelta(
        seconds=offset_seconds
    )


def random_progress(
    rng: random.Random,
    alpha: float = 2.0,
    beta: float = 5.0,
) -> float:
    """
    Generate a value between 0 and 1 using a beta distribution.
    """

    return rng.betavariate(alpha, beta)


def datetime_at_progress(
    start: datetime,
    end: datetime,
    progress: float,
) -> datetime:
    """
    Return a timestamp at a given fractional position
    between start and end.
    """

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError(
            "start and end must be timezone-aware"
        )

    if not 0.0 <= progress <= 1.0:
        raise ValueError(
            "progress must be between 0 and 1"
        )

    duration = end - start

    return start + duration * progress