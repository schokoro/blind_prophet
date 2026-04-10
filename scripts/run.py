import asyncio
import logging
from pathlib import Path

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
    channel: str = typer.Option(..., help="Telegram channel username"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
) -> None:
    """Scrape messages from a Telegram channel into the database."""

    async def _run() -> None:
        from telethon import TelegramClient

        from amnesiac.collect import scrape_channel
        from amnesiac.config import settings
        from amnesiac.store import apply_migrations, get_connection

        conn = get_connection(db_path)
        apply_migrations(conn)

        async with TelegramClient(
            settings.tg_session_path,
            settings.tg_api_id,
            settings.tg_api_hash,
        ) as client:
            if not await client.is_user_authorized():
                raise RuntimeError(
                    "Telegram client is not authorized. Run an interactive login first."
                )
            await scrape_channel(client, channel, conn)

        conn.close()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
