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


if __name__ == "__main__":
    app()
