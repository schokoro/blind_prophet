import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/db/blind_prophet.db")


@app.command()
def init_db(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
) -> None:
    """Initialise the database and apply migrations."""
    from amnesiac.store import apply_migrations, get_connection

    conn = get_connection(db_path)
    apply_migrations(conn)
    conn.close()
    logger.info("Database initialised at %s", db_path)


@app.command()
def scrape(
    channel: Optional[str] = typer.Option(None, help="Telegram channel username (omit for all)"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
) -> None:
    """Scrape messages from Telegram channels into the database."""

    if channel is not None:
        async def _run_single() -> None:
            import yaml
            from telethon import TelegramClient

            from amnesiac.collect import scrape_channel
            from amnesiac.config import settings
            from amnesiac.store import apply_migrations, get_connection

            conn = get_connection(db_path)
            apply_migrations(conn)

            try:
                accounts_data = yaml.safe_load(Path("config/accounts.yaml").read_text())
                accounts = (accounts_data or {}).get("accounts", [])
                account = accounts[0] if accounts else None
            except FileNotFoundError:
                account = None

            if account:
                api_id = account["api_id"]
                api_hash = account["api_hash"]
                session_file = account["session_file"]
            else:
                api_id = settings.tg_api_id
                api_hash = settings.tg_api_hash
                session_file = settings.tg_session_path

            async with TelegramClient(session_file, api_id, api_hash) as client:
                if not await client.is_user_authorized():
                    raise RuntimeError(
                        "Telegram client is not authorized. Run an interactive login first."
                    )
                await scrape_channel(
                    client, channel, conn,
                    batch_size=settings.batch_size,
                    inter_batch_sleep=settings.inter_batch_sleep,
                )

            conn.close()

        asyncio.run(_run_single())
    else:
        from amnesiac.collect import run_all

        asyncio.run(run_all(db_path))


if __name__ == "__main__":
    app()
