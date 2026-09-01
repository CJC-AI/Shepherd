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