import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer

from amnesiac.config import DEFAULT_DB_PATH
from amnesiac.utils import normalize_run_date

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
            from amnesiac.collect.runner import make_proxy
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

            async with TelegramClient(session_file, api_id, api_hash, proxy=make_proxy(settings)) as client:
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


@app.command()
def login(
    session_file: str = typer.Option(..., help="Session file name (without .session)"),
    api_id: int = typer.Option(..., help="Telegram api_id"),
    api_hash: str = typer.Option(..., help="Telegram api_hash"),
    phone: str = typer.Option("", help="Phone number (optional, informational only)"),
) -> None:
    """Authorize a Telegram account and upsert it in config/accounts.yaml."""
    import yaml
    from telethon import TelegramClient

    async def _authorize() -> None:
        client = TelegramClient(session_file, api_id, api_hash)
        await client.start()
        await client.disconnect()

    asyncio.run(_authorize())

    accounts_path = Path("config/accounts.yaml")
    if accounts_path.exists():
        data = yaml.safe_load(accounts_path.read_text()) or {}
    else:
        data = {}

    accounts = data.get("accounts") or []

    existing = next((a for a in accounts if a.get("session_file") == session_file), None)
    if existing:
        existing["api_id"] = api_id
        existing["api_hash"] = api_hash
        if phone:
            existing["phone"] = phone
    else:
        entry: dict = {"session_file": session_file, "api_id": api_id, "api_hash": api_hash}
        if phone:
            entry["phone"] = phone
        accounts.append(entry)

    data["accounts"] = accounts
    accounts_path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))
    print("Authorization successful. Account saved to config/accounts.yaml")


@app.command()
def embed(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
    batch_size: int = typer.Option(0, help="Batch size (0 = use params.yaml value)"),
    lead_sentences: int = typer.Option(0, help="Lead sentences (0 = use params.yaml value)"),
) -> None:
    """Preprocess messages and build embedding index."""
    import yaml

    from amnesiac.embed import run_embed

    params = yaml.safe_load(Path("config/params.yaml").read_text())
    preprocessing = params.get("preprocessing", {})

    if batch_size == 0:
        batch_size = preprocessing.get("embed_batch_size", 64)
    if lead_sentences == 0:
        lead_sentences = preprocessing.get("lead_sentences", 3)
    min_text_length = preprocessing.get("min_text_length", 50)

    asyncio.run(run_embed(db_path, batch_size, lead_sentences, min_text_length))


@app.command()
def summarize(
    date: str = typer.Option(..., help="Survey date (YYYY-MM or YYYY-MM-DD), e.g. 2021-10"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing summary for this date"),
) -> None:
    """Summarize news corpus for a given survey date and store result in DB."""
    from amnesiac.summarize.runner import run_summarize_pipeline

    date = normalize_run_date(date)
    asyncio.run(run_summarize_pipeline(db_path, date, force=force))


@app.command()
def neuter(
    date: str = typer.Option(..., help="Survey date (YYYY-MM or YYYY-MM-DD), e.g. 2021-12 or 2021-12-20"),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override MODEL_N from config.py for this run (e.g. 'openai/gpt-5.4')",
    ),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing record"),
    save_artifacts: bool = typer.Option(
        False,
        "--save-artifacts/--no-save-artifacts",
        help="Save per-iteration neutering audit artifacts under data/runs/neuter/",
    ),
) -> None:
    """Run neutering pipeline for a given run_date."""
    from amnesiac.neuter.runner import run_neuter_pipeline

    asyncio.run(
        run_neuter_pipeline(
            db_path,
            normalize_run_date(date),
            force=force,
            model_n_override=model,
            save_artifacts=save_artifacts,
        )
    )


@app.command()
def forecast(
    date: str = typer.Option(..., help="Survey date (YYYY-MM or YYYY-MM-DD), e.g. 2021-12"),
    condition: str = typer.Option(
        "both",
        "--condition",
        help="Forecast condition: 'raw', 'neutered', or 'both'",
    ),
    n_samples: int = typer.Option(3, "--n-samples", help="Number of samples per persona"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing forecast rows"),
) -> None:
    """Run forecast pipeline for a given run_date and store samples in DB."""
    from amnesiac.forecast.runner import run_forecast_pipeline

    date = normalize_run_date(date)
    asyncio.run(
        run_forecast_pipeline(
            db_path,
            date,
            condition=condition,
            force=force,
            n_samples=n_samples,
        )
    )


@app.command()
def series(
    dates: str = typer.Option(
        ...,
        "--dates",
        help="Comma-separated list of dates: YYYY-MM or YYYY-MM-DD, e.g. '2021-09,2022-03,2022-04'",
    ),
    n_samples: int = typer.Option(3, "--n-samples", help="Forecast samples per persona"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
) -> None:
    """Run summarize -> neuter -> forecast for each date in the series, with skip-existing and continue-on-error."""
    from amnesiac.forecast.series import format_summary_table, run_series_pipeline

    raw_dates = [d.strip() for d in dates.split(",") if d.strip()]
    if not raw_dates:
        raise typer.BadParameter("--dates must contain at least one date")

    normalized = [normalize_run_date(d) for d in raw_dates]
    logger.info("Series run_dates (normalized): %s", normalized)

    results = asyncio.run(run_series_pipeline(db_path, normalized, n_samples=n_samples))

    summary = format_summary_table(results)
    print()
    print("=" * 80)
    print("SERIES RUN SUMMARY")
    print("=" * 80)
    print(summary)
    print()

    failed = [
        r
        for r in results
        if any(s.status == "failed" for s in [r.summarize, r.neuter, r.forecast])
    ]
    if failed:
        logger.warning("Failed dates: %s", [r.run_date for r in failed])


@app.command()
def eval(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database file"),
) -> None:
    """Build final evaluation artifacts in data/eval/."""
    from scripts.build_eval import main as build_eval_main

    build_eval_main(db_path)


if __name__ == "__main__":
    app()
