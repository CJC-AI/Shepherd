"""
Generate simulated bank accounts for customers.

This module creates AccountState objects from CustomerState objects while
assigning account types, currencies, starting balances, and deterministic
account identifiers.

Account identifiers are derived from the customer's deterministic UUID and
the account's position within that customer's account set. This allows the
same simulation configuration and random seed to reproduce the same account
identifiers.
"""

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
    """
    Generate simulated accounts for customers.

    Account identifiers are deterministic UUID5 values derived from the
    customer's UUID and the account's index.

    Parameters
    ----------
    seed:
        Optional random seed controlling stochastic account attributes such as
        account count, secondary currency selection, and starting balance.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the account generator.

        Parameters
        ----------
        seed:
            Optional random seed used for reproducible account attributes.
        """
        self.random = random.Random(seed)

    def generate_for_customer(
        self,
        customer: CustomerState,
    ) -> list[AccountState]:
        """
        Generate accounts belonging to one customer.

        Each customer receives one primary checking account and has a 10%
        probability of receiving a second account. Secondary accounts are
        either savings or checking accounts.

        Parameters
        ----------
        customer:
            CustomerState for which accounts should be generated.

        Returns
        -------
        list[AccountState]
            Generated account states belonging to the customer.
        """

        account_count = 1

        if self.random.random() < 0.10:
            account_count = 2

        accounts: list[AccountState] = []

        primary_currency = COUNTRY_CURRENCIES[
            customer.home_country
        ]

        for index in range(account_count):
            account_id = uuid.uuid5(
                customer.customer_id,
                f"account-{index}",
            )

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
        """
        Select the currency for an account.

        Primary accounts always use the customer's primary country currency.
        Secondary accounts use that currency 75% of the time and otherwise
        receive a randomly selected alternate supported currency.

        Parameters
        ----------
        primary_currency:
            Currency associated with the customer's home country.
        customer:
            CustomerState owning the account.
        is_primary:
            Whether the account is the customer's primary account.

        Returns
        -------
        str
            Three-letter currency code.
        """

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
        """
        Generate an initial account balance.

        Checking accounts use a smaller balance multiplier than savings
        accounts. A random variation is applied around the resulting base
        amount, while ensuring that the balance is never below the customer's
        average spend.

        Parameters
        ----------
        customer:
            CustomerState whose spending behavior determines the balance.
        account_type:
            Account type used to select the balance multiplier.

        Returns
        -------
        float
            Generated starting balance rounded to two decimal places.
        """

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