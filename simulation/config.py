from dataclasses import dataclass


@dataclass(frozen=True)
class CustomerArchetype:
    name: str
    transactions_per_day_min: int
    transactions_per_day_max: int
    average_spend: float
    spend_std: float
    international_probability: float
    device_count_min: int
    device_count_max: int

CONSERVATIVE = CustomerArchetype(
    name="conservative",
    transactions_per_day_min=1,
    transactions_per_day_max=4,
    average_spend=35.0,
    spend_std=12.0,
    international_probability=0.02,
    device_count_min=1,
    device_count_max=2,
)

PROFESSIONAL = CustomerArchetype(
    name="professional",
    transactions_per_day_min=3,
    transactions_per_day_max=10,
    average_spend=75.0,
    spend_std=25.0,
    international_probability=0.10,
    device_count_min=2,
    device_count_max=4,
)

TRAVELER = CustomerArchetype(
    name="traveler",
    transactions_per_day_min=4,
    transactions_per_day_max=12,
    average_spend=120.0,
    spend_std=50.0,
    international_probability=0.45,
    device_count_min=2,
    device_count_max=4,
)

HIGH_NET_WORTH = CustomerArchetype(
    name="high_net_worth",
    transactions_per_day_min=2,
    transactions_per_day_max=7,
    average_spend=800.0,
    spend_std=150.0,
    international_probability=0.25,
    device_count_min=2,
    device_count_max=4,
)

DIGITAL_NATIVE = CustomerArchetype(
    name="digital_native",
    transactions_per_day_min=8,
    transactions_per_day_max=25,
    average_spend=45.0,
    spend_std=20.0,
    international_probability=0.15,
    device_count_min=2,
    device_count_max=5,
)

CUSTOMER_ARCHETYPES = [
    CONSERVATIVE,
    PROFESSIONAL,
    TRAVELER,
    HIGH_NET_WORTH,
    DIGITAL_NATIVE
]