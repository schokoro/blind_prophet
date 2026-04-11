import asyncio
import logging
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import yaml
from telethon import TelegramClient

from amnesiac.collect.scraper import scrape_channel
from amnesiac.config import settings
from amnesiac.store import apply_migrations, get_connection

logger = logging.getLogger(__name__)


async def _scrape_account(
    client: TelegramClient,
    channels: list[str],
    conn,
    batch_size: int,
    inter_batch_sleep: float,
) -> None:
    for ch in channels:
        try:
            await scrape_channel(
                client, ch, conn,
                batch_size=batch_size,
                inter_batch_sleep=inter_batch_sleep,
            )
        except Exception:
            logger.exception("Error scraping @%s, skipping", ch)


async def run_all(db_path: str | Path) -> None:
    accounts_data = yaml.safe_load(Path("config/accounts.yaml").read_text())
    channels_data = yaml.safe_load(Path("config/channels.yaml").read_text())

    accounts = accounts_data.get("accounts", [])
    channels = channels_data.get("channels", [])

    if not accounts:
        logger.error("No accounts configured in config/accounts.yaml")
        return
    if not channels:
        logger.warning("No channels configured in config/channels.yaml")
        return

    # Round-robin assignment of channels to accounts
    assignment: dict[str, list[str]] = defaultdict(list)
    for channel, account in zip(channels, cycle(accounts)):
        key = account["session_file"]
        assignment[key].append(channel)

    conn = get_connection(db_path)
    apply_migrations(conn)

    account_map = {a["session_file"]: a for a in accounts}
    tasks = []

    for session_file, ch_list in assignment.items():
        account = account_map[session_file]
        client = TelegramClient(
            account["session_file"],
            account["api_id"],
            account["api_hash"],
        )
        await client.connect()

        if not await client.is_user_authorized():
            logger.warning(
                "Account %s is not authorized, skipping %d channels",
                account["phone"],
                len(ch_list),
            )
            await client.disconnect()
            continue

        tasks.append(
            _scrape_account(
                client, ch_list, conn,
                batch_size=settings.batch_size,
                inter_batch_sleep=settings.inter_batch_sleep,
            )
        )

    if tasks:
        await asyncio.gather(*tasks)

    conn.close()
