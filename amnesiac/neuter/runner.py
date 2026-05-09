"""Pipeline shell for applying summary neutering to stored summaries."""

from datetime import datetime
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

from amnesiac.neuter.config import (
    JUDGE_BLIND_THRESHOLD,
    LLM_TIMEOUT_SECONDS,
    MODEL_J1,
    MODEL_J2,
    MODEL_N,
    OPENROUTER_BASE_URL,
)
from amnesiac.neuter.artifacts import NeuterArtifactWriter
from amnesiac.neuter.cycle import run_cycle
from amnesiac.neuter.holdout import run_j2_holdout
from amnesiac.store import apply_migrations, get_connection

logger = logging.getLogger(__name__)


async def run_neuter_pipeline(
    db_path: Path,
    run_date: str,
    *,
    force: bool = False,
    model_n_override: str | None = None,
    save_artifacts: bool = False,
) -> None:
    """Run full neutering pipeline for run_date and persist the result to neutered_summaries."""
    model_n = model_n_override if model_n_override else MODEL_N
    if model_n_override:
        logger.info("Using N model override: %s", model_n_override)

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

        try:
            parsed_date = datetime.strptime(run_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"run_date must use YYYY-MM-DD format; got {run_date!r}") from exc
        true_year = parsed_date.year
        true_month = parsed_date.month

        raw_summary = raw_row[0]
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to run the neutering pipeline")

        artifact_writer = None
        if save_artifacts:
            artifact_writer = NeuterArtifactWriter.create(run_date)
            logger.info("Saving neuter artifacts to %s", artifact_writer.artifact_dir)

        async with AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
        ) as client:
            cycle_result = await run_cycle(
                client,
                raw_summary,
                model_n=model_n,
                iteration_callback=artifact_writer.write_iteration if artifact_writer else None,
            )
            raw_holdout = await run_j2_holdout(
                client,
                raw_summary,
                true_year,
                true_month,
                label="raw",
            )
            neutered_holdout = await run_j2_holdout(
                client,
                cycle_result["summary"],
                true_year,
                true_month,
                label="neutered",
            )

        raw_period_id_score = raw_holdout["period_id_score"]
        neutered_period_id_score = neutered_holdout["period_id_score"]
        period_delta_vs_raw = raw_period_id_score - neutered_period_id_score
        judge_blind = 1 if raw_period_id_score < JUDGE_BLIND_THRESHOLD else 0

        conn.execute(
            """
            INSERT OR REPLACE INTO neutered_summaries (
                run_date,
                summary,
                neutering_status,
                final_iteration,
                q3_preservation,
                raw_period_id_score,
                neutered_period_id_score,
                period_delta_vs_raw,
                judge_blind,
                model_n,
                model_j1,
                model_j2
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_date,
                cycle_result["summary"],
                cycle_result["neutering_status"],
                cycle_result["final_iteration"],
                cycle_result["q3_preservation"],
                raw_period_id_score,
                neutered_period_id_score,
                period_delta_vs_raw,
                judge_blind,
                model_n,
                MODEL_J1,
                MODEL_J2,
            ),
        )
        conn.commit()
        if artifact_writer is not None:
            artifact_writer.write_final(
                {
                    "neutering_status": cycle_result["neutering_status"],
                    "final_iteration": cycle_result["final_iteration"],
                    "q3_preservation": cycle_result["q3_preservation"],
                    "q3_strength_net_shift": cycle_result["q3_strength_net_shift"],
                    "length_ratio": cycle_result["length_ratio"],
                    "residual_hard_fail_count": cycle_result["residual_hard_fail_count"],
                    "residual_warn_count": cycle_result["residual_warn_count"],
                    "raw_period_id_score": raw_period_id_score,
                    "neutered_period_id_score": neutered_period_id_score,
                    "period_delta_vs_raw": period_delta_vs_raw,
                    "judge_blind": judge_blind,
                    "model_n": model_n,
                    "model_j1": MODEL_J1,
                    "model_j2": MODEL_J2,
                }
            )
        logger.info(
            "neuter run_date=%s status=%s iter=%d q3=%.3f raw=%.3f neutered=%.3f delta=%.3f blind=%d",
            run_date,
            cycle_result["neutering_status"],
            cycle_result["final_iteration"],
            cycle_result["q3_preservation"],
            raw_period_id_score,
            neutered_period_id_score,
            period_delta_vs_raw,
            judge_blind,
        )
    finally:
        conn.close()
