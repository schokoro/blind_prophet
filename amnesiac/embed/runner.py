import logging
import sqlite3

from razdel import sentenize

from amnesiac.preprocess import process_message
from amnesiac.store import (
    apply_migrations,
    embed_documents,
    get_connection,
    get_unprocessed_messages,
    insert_embeddings,
    insert_processed_message,
)

logger = logging.getLogger(__name__)


async def run_embed(
    db_path,
    batch_size: int,
    lead_sentences: int,
    min_text_length: int,
) -> None:
    conn = get_connection(db_path)
    apply_migrations(conn)

    batch_num = 0
    try:
        while True:
            batch = get_unprocessed_messages(conn, batch_size)
            if not batch:
                break

            batch_num += 1
            to_embed: list[tuple[int, str]] = []
            n_valid = 0
            n_invalid = 0

            for msg in batch:
                processed_text, is_valid = process_message(msg["text"], min_text_length)
                pm_id = insert_processed_message(conn, msg["id"], processed_text, is_valid)
                if is_valid:
                    to_embed.append((pm_id, processed_text))
                    n_valid += 1
                else:
                    n_invalid += 1

            if to_embed:
                pm_ids = [pm_id for pm_id, _ in to_embed]
                leads = []
                for _, text in to_embed:
                    sentences = [s.text for s in sentenize(text)]
                    lead = " ".join(sentences[:lead_sentences])
                    leads.append(lead)

                vectors = await embed_documents(leads)
                insert_embeddings(conn, list(zip(pm_ids, vectors)))

            logger.info(
                "Batch %d done: %d valid, %d invalid", batch_num, n_valid, n_invalid
            )
    finally:
        conn.close()

    logger.info("Embedding complete: %d batches processed", batch_num)
