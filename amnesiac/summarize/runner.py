import logging
import os

from amnesiac.config import settings
from amnesiac.store import apply_migrations, get_connection, insert_summary
from amnesiac.summarize.retriever import retrieve
from amnesiac.summarize.summarizer import run_summarize

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


async def run_summarize_pipeline(db_path, run_date: str, force: bool = False) -> None:
    from openai import AsyncOpenAI

    conn = get_connection(db_path)
    try:
        apply_migrations(conn)
        if force:
            conn.execute("DELETE FROM summaries WHERE run_date = ?", (run_date,))
            conn.commit()

        client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=600.0,
        )
        model = settings.rag["summarize_model"]
        horizon_days = int(settings.rag["horizon_days"])

        retrieved = await retrieve(conn, run_date)
        summary = await run_summarize(client, retrieved, model, horizon_days)
        doc_count = sum(len(docs) for docs in retrieved.values())

        insert_summary(conn, run_date, summary, doc_count, model)
        logger.info("Summarization completed for %s (%s docs)", run_date, doc_count)
    finally:
        conn.close()
