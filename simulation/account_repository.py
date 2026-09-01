from database.models import Account
from database.session import SessionLocal
from simulation.account_state import AccountState


def save_accounts(
    accounts: list[AccountState],
    batch_size: int = 1000,
) -> int:
    if not accounts:
        return 0

    total_saved = 0

    with SessionLocal() as session:
        for start in range(0, len(accounts), batch_size):
            batch = accounts[start:start + batch_size]

            account_rows = [
                Account(
                    account_id=account.account_id,
                    customer_id=account.customer_id,
                    account_type=account.account_type,
                    balance=account.balance,
                    currency=account.currency,
                )
                for account in batch
            ]

            session.add_all(account_rows)
            session.flush()

            total_saved += len(account_rows)

        session.commit()

    return total_saved