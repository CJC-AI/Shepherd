from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass
class DeviceState:
    device_id: UUID
    customer_id: UUID
    device_trust_score: float
    first_seen: datetime