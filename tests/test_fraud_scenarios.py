"""
Validate the observable behavior of Shepherd's synthetic fraud scenarios.

This test generates one deterministic customer's baseline transactions and
then runs the fraud scenario engine four separate times, with only one fraud
scenario enabled on each run.

Each scenario is validated using transaction-observable properties rather
than the internal scenario name. This helps ensure that the synthetic fraud
labels correspond to measurable behavioral signals that a future feature
engineering pipeline could derive.
"""

from dataclasses import replace
from datetime import timedelta

from simulation.account_generator import AccountGenerator
from simulation.device_generator import DeviceGenerator
from simulation.fraud_config import FraudScenarioConfig
from simulation.fraud_scenario_engine import FraudScenarioEngine
from simulation.merchant_generator import MerchantGenerator
from simulation.simulation_config import DEFAULT_SIMULATION_CONFIG
from simulation.simulation_runner import SimulationRunner
from simulation.transaction_generator import TransactionGenerator


def build_fixture():
    """
    Build a deterministic customer transaction fixture.

    Returns
    -------
    tuple
        Customer, accounts, devices, and baseline transactions.
    """

    runner = SimulationRunner(
        DEFAULT_SIMULATION_CONFIG,
        seed=42,
    )

    customer = next(
        customer
        for customer in runner.generate_customers()
        if len(customer.known_devices) >= 2
    )

    accounts = AccountGenerator(
        seed=43,
    ).generate_for_customer(customer)

    devices = DeviceGenerator(
        seed=44,
    ).generate_for_customer(customer)

    merchants = MerchantGenerator(
        seed=45,
    ).generate(500)

    transactions = TransactionGenerator(
        seed=46,
    ).generate_for_customer(
        customer,
        accounts,
        devices,
        merchants,
    )

    return customer, accounts, devices, transactions


def run_single_scenario(
    scenario: str,
):
    """
    Run the fraud engine with exactly one enabled scenario.

    Parameters
    ----------
    scenario:
        Scenario name to enable.

    Returns
    -------
    tuple
        Customer, devices, baseline transactions, and fraud result.
    """

    customer, accounts, devices, baseline = (
        build_fixture()
    )

    weights = {
        "new_device": 0.0,
        "geographic_anomaly": 0.0,
        "amount_anomaly": 0.0,
        "velocity": 0.0,
    }

    weights[scenario] = 1.0

    config = FraudScenarioConfig(
        fraud_rate=0.05,
        new_device_weight=weights["new_device"],
        geographic_anomaly_weight=weights[
            "geographic_anomaly"
        ],
        amount_anomaly_weight=weights[
            "amount_anomaly"
        ],
        velocity_weight=weights["velocity"],
    )

    result = FraudScenarioEngine(
        config=config,
        seed=47,
    ).apply_for_customer(
        customer=customer,
        transactions=baseline,
        devices=devices,
    )

    return customer, devices, baseline, result


def test_new_device_scenario() -> None:
    """
    Verify that new-device fraud uses a recently introduced device.

    Raises
    ------
    AssertionError
        If any fraud transaction does not use a device that was first seen
        within the configured seven-day window before the transaction.
    """

    customer, devices, baseline, result = (
        run_single_scenario("new_device")
    )

    baseline_by_id = {
        transaction.transaction_id: transaction
        for transaction in baseline
    }

    device_by_id = {
        device.device_id: device
        for device in devices
    }

    fraud_transactions = [
        transaction
        for transaction in result.transactions
        if transaction.is_fraud
    ]

    assert result.scenario_counts["new_device"] == len(
        fraud_transactions
    )

    for transaction in fraud_transactions:
        device = device_by_id[transaction.device_id]

        age = (
            transaction.transaction_timestamp
            - device.first_seen
        )

        assert timedelta(0) <= age <= timedelta(days=7)

        assert transaction.transaction_timestamp >= (
            customer.customer_start_time
        )

        assert (
            baseline_by_id[transaction.transaction_id]
            .is_fraud
            is None
        )

    print(
        "new-device scenario: PASS "
        f"({len(fraud_transactions)} transactions)"
    )


def test_geographic_anomaly() -> None:
    """
    Verify that geographic fraud uses an unusual customer country.

    Raises
    ------
    AssertionError
        If a fraud transaction occurs in one of the customer's usual
        countries.
    """

    customer, _, baseline, result = (
        run_single_scenario("geographic_anomaly")
    )

    fraud_transactions = [
        transaction
        for transaction in result.transactions
        if transaction.is_fraud
    ]

    assert result.scenario_counts[
        "geographic_anomaly"
    ] == len(fraud_transactions)

    for transaction in fraud_transactions:
        assert (
            transaction.transaction_country
            not in customer.usual_countries
        )

    print(
        "geographic anomaly: PASS "
        f"({len(fraud_transactions)} transactions)"
    )


def test_amount_anomaly() -> None:
    """
    Verify that amount fraud increases the baseline amount by the configured
    anomaly multiplier.

    Raises
    ------
    AssertionError
        If a fraud transaction does not match the configured amount increase.
    """

    _, _, baseline, result = (
        run_single_scenario("amount_anomaly")
    )

    baseline_by_id = {
        transaction.transaction_id: transaction
        for transaction in baseline
    }

    fraud_transactions = [
        transaction
        for transaction in result.transactions
        if transaction.is_fraud
    ]

    assert result.scenario_counts[
        "amount_anomaly"
    ] == len(fraud_transactions)

    for transaction in fraud_transactions:
        original = baseline_by_id[
            transaction.transaction_id
        ]

        expected_amount = round(
            original.amount * 4.0,
            2,
        )

        assert transaction.amount == expected_amount

    print(
        "amount anomaly: PASS "
        f"({len(fraud_transactions)} transactions)"
    )


def test_velocity_scenario() -> None:
    """
    Verify that velocity fraud places transactions within the configured
    short temporal window of another customer transaction.

    Raises
    ------
    AssertionError
        If a fraud transaction is not part of a ten-minute temporal burst.
    """

    _, _, _, result = run_single_scenario(
        "velocity"
    )

    fraud_transactions = [
        transaction
        for transaction in result.transactions
        if transaction.is_fraud
    ]

    all_transactions = result.transactions

    assert result.scenario_counts[
        "velocity"
    ] == len(fraud_transactions)

    for fraud_transaction in fraud_transactions:
        nearest_distance = min(
            abs(
                (
                    fraud_transaction.transaction_timestamp
                    - other.transaction_timestamp
                ).total_seconds()
            )
            for other in all_transactions
            if other.transaction_id
            != fraud_transaction.transaction_id
        )

        assert nearest_distance <= 10 * 60

    print(
        "velocity scenario: PASS "
        f"({len(fraud_transactions)} transactions)"
    )


def main() -> None:
    """
    Execute all synthetic fraud scenario tests.

    Returns
    -------
    None
    """

    test_new_device_scenario()
    test_geographic_anomaly()
    test_amount_anomaly()
    test_velocity_scenario()

    print("all fraud scenario tests: PASS")


if __name__ == "__main__":
    main()