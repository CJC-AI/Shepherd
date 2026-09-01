from dataclasses import dataclass
from uuid import UUID


@dataclass
class AccountState:
    account_id: UUID
    customer_id: UUID
    account_type: str
    currency: str
    balance: float