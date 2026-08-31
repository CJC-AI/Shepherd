COUNTRIES = [
    "NG", "GB", "DE", "FR", "ES", "IT",
    "NL", "BE", "PT", "AE", "US", "CA",
]

MERCHANT_CATEGORIES = [
    "groceries",
    "transport",
    "pharmacy",
    "utilities",
    "restaurants",
    "subscriptions",
    "electronics",
    "hotels",
    "travel",
    "luxury",
    "gaming",
    "digital_goods",
    "e_commerce",
    "food_delivery",
]

TRAVEL_DESTINATIONS = {
    "NG": ["GB", "FR", "AE", "DE", "US"],
    "GB": ["FR", "ES", "DE", "AE", "US"],
    "DE": ["FR", "ES", "IT", "NL", "GB"],
    "FR": ["ES", "IT", "DE", "GB", "BE"],
    "ES": ["FR", "PT", "IT", "DE", "GB"],
    "IT": ["FR", "DE", "ES", "GB", "NL"],
    "NL": ["DE", "BE", "FR", "GB", "ES"],
    "BE": ["FR", "NL", "DE", "GB", "ES"],
    "PT": ["ES", "FR", "GB", "DE", "NL"],
    "AE": ["GB", "FR", "DE", "US", "NG"],
    "US": ["GB", "FR", "DE", "CA", "AE"],
    "CA": ["US", "GB", "FR", "DE", "AE"],
}

ARCHETYPE_MERCHANT_CATEGORIES = {
    "conservative": [
        "groceries", "transport", "pharmacy", "utilities",
    ],
    "professional": [
        "restaurants", "transport", "groceries",
        "subscriptions", "electronics",
    ],
    "traveler": [
        "restaurants", "hotels", "travel",
        "transport", "e_commerce",
    ],
    "high_net_worth": [
        "luxury", "travel", "hotels",
        "restaurants", "electronics",
    ],
    "digital_native": [
        "subscriptions", "gaming", "digital_goods",
        "food_delivery", "e_commerce",
    ],
}