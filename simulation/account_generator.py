import random
import uuid

from simulation.account_state import AccountState
from simulation.customer_state import CustomerState


COUNTRY_CURRENCIES = {
    "NG": "NGN",
    "GB": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "ES": "EUR",
    "IT": "EUR",
    "NL": "EUR",
    "BE": "EUR",
    "PT": "EUR",
    "US": "USD",
    "CA": "CAD",
    "AE": "AED",
}


class AccountGenerator:
    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def generate_for_customer(
        self,
        customer: CustomerState,
    ) -> list[AccountState]:
        account_count = 1

        if self.random.random() < 0.10:
            account_count = 2

        accounts: list[AccountState] = []

        primary_currency = COUNTRY_CURRENCIES[
            customer.home_country
        ]

        for index in range(account_count):
            account_id = uuid.uuid4()

            if index == 0:
                account_type = "checking"
            else:
                account_type = (
                    "savings"
                    if self.random.random() < 0.75
                    else "checking"
                )

            currency = self._choose_currency(
                primary_currency=primary_currency,
                customer=customer,
                is_primary=index == 0,
            )

            balance = self._generate_starting_balance(
                customer=customer,
                account_type=account_type,
            )

            accounts.append(
                AccountState(
                    account_id=account_id,
                    customer_id=customer.customer_id,
                    account_type=account_type,
                    currency=currency,
                    balance=balance,
                )
            )

        customer.account_ids = [
            account.account_id
            for account in accounts
        ]

        return accounts

    def _choose_currency(
        self,
        primary_currency: str,
        customer: CustomerState,
        is_primary: bool,
    ) -> str:
        if is_primary:
            return primary_currency

        if self.random.random() < 0.75:
            return primary_currency

        alternate_currencies = [
            "EUR",
            "GBP",
            "USD",
            "NGN",
            "CAD",
            "AED",
        ]

        alternate_currencies = [
            currency
            for currency in alternate_currencies
            if currency != primary_currency
        ]

        return self.random.choice(alternate_currencies)

    def _generate_starting_balance(
        self,
        customer: CustomerState,
        account_type: str,
    ) -> float:
        base_multiplier = (
            8
            if account_type == "checking"
            else 15
        )

        raw_balance = (
            customer.average_spend
            * base_multiplier
        )

        variation = self.random.uniform(
            0.7,
            1.3,
        )

        balance = raw_balance * variation

        return round(
            max(balance, customer.average_spend),
            2,
        )