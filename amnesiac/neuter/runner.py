"""Pipeline shell for applying summary neutering to stored summaries."""

import logging
from pathlib import Path

from amnesiac.store import apply_migrations, get_connection

logger = logging.getLogger(__name__)


async def run_neuter_pipeline(db_path: Path, run_date: str, *, force: bool = False) -> None:
    conn = get_connection(db_path)
    try:
        apply_migrations(conn)

        raw_row = conn.execute(
            "SELECT summary FROM summaries WHERE run_date = ?",
            (run_date,),
        ).fetchone()
        if raw_row is None:
            raise LookupError(f"No raw summary in 'summaries' for run_date={run_date}")

        existing_row = conn.execute(
            "SELECT 1 FROM neutered_summaries WHERE run_date = ?",
            (run_date,),
        ).fetchone()
        if existing_row is not None and not force:
            logger.info("neutered_summaries already has row for %s; pass --force to overwrite", run_date)
            return

        logger.info("neuter skeleton stage: pipeline not yet implemented (p02/p03 pending)")
    finally:
        conn.close()

