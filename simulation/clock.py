from datetime import datetime, timedelta, timezone


class SimulationClock:
    def __init__(self, start_time: datetime):
        if start_time.tzinfo is None:
            raise ValueError(
                "start_time must be timezone-aware"
            )

        self.current_time = start_time

    def now(self) -> datetime:
        return self.current_time

    def advance(
        self,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
    ) -> datetime:
        self.current_time += timedelta(
            seconds=seconds,
            minutes=minutes,
            hours=hours,
            days=days,
        )

        return self.current_time