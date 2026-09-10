from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class SimulationConfig:
    start_time: datetime
    duration_days: int
    customer_count: int

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(
            days=self.duration_days
        )


DEFAULT_SIMULATION_CONFIG = SimulationConfig(
    start_time=datetime(
        2026,
        6,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    ),
    duration_days=90,
    customer_count=10_000,
)