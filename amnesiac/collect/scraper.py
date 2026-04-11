import asyncio
import logging
import sqlite3

from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, FloodWaitError, UsernameNotOccupiedError

from amnesiac.store import (
    get_last_msg_id,
    insert_messages,
    upsert_channel,
)

logger = logging.getLogger(__name__)


async def scrape_channel(
    client: TelegramClient,
    channel_username: str,
    conn: sqlite3.Connection,
    batch_size: int = 200,
    inter_batch_sleep: float = 1.0,
) -> None:
    """Scrape messages from a Telegram channel and write them to the database."""
    entity = await client.get_entity(channel_username)
    title = getattr(entity, "title", None)

    channel_id = upsert_channel(conn, channel_username, title)
    min_id = get_last_msg_id(conn, channel_id) or 0

    logger.info("Scraping @%s (channel_id=%d, min_id=%d)", channel_username, channel_id, min_id)

    batch: list[dict] = []
    total_saved = 0
    batches_flushed = 0

    async def flush(rows: list[dict]) -> None:
        nonlocal total_saved, batches_flushed
        insert_messages(conn, rows)
        total_saved += len(rows)
        batches_flushed += 1
        logger.info(
            "@%s: flushed batch %d (%d messages, %d total saved)",
            channel_username,
            batches_flushed,
            len(rows),
            total_saved,
        )

    while True:
        try:
            async for msg in client.iter_messages(entity, min_id=min_id, reverse=True):
                if msg.fwd_from is not None:
                    continue
                if not msg.text:
                    continue

                media_type = type(msg.media).__name__ if msg.media else None
                batch.append(
                    {
                        "channel_id": channel_id,
                        "tg_message_id": msg.id,
                        "text": msg.text,
                        "date": msg.date.isoformat(),
                        "views": msg.views,
                        "forwards": msg.forwards,
                        "reply_to_msg_id": msg.reply_to_msg_id,
                        "media_type": media_type,
                        "raw_json": msg.to_json(),
                    }
                )

                if len(batch) >= batch_size:
                    await flush(batch)
                    await asyncio.sleep(inter_batch_sleep)
                    min_id = batch[-1]["tg_message_id"]
                    batch = []

            # Normal completion — exit the retry loop
            break

        except FloodWaitError as e:
            logger.warning("FloodWaitError: waiting %d seconds", e.seconds)
            await asyncio.sleep(e.seconds)
        except (ChannelPrivateError, UsernameNotOccupiedError):
            logger.error("Cannot access channel @%s", channel_username)
            raise

    if batch:
        await flush(batch)

    logger.info(
        "Done scraping @%s: %d messages saved in %d batches",
        channel_username,
        total_saved,
        batches_flushed,
    )
