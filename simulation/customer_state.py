from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from uuid import UUID

from simulation.config import CustomerArchetype


@dataclass
class CustomerState:
    customer_id: UUID
    archetype: CustomerArchetype
    customer_start_time: datetime

    home_country: str

    average_spend: float
    spend_std: float

    transactions_per_day: int

    usual_countries: List[str] = field(default_factory=list)
    known_devices: List[UUID] = field(default_factory=list)
    preferred_merchant_categories: List[str] = field(default_factory=list)

    last_transaction_timestamp: datetime | None = None

    account_ids: list[UUID] = field(default_factory=list)