"""Iterative neutering cycle entry point."""

import logging

from openai import AsyncOpenAI

from amnesiac.neuter.config import N_MAX, ROLLBACK_ON_HARD_RESIDUAL, Y_FLOOR
from amnesiac.neuter.judges import call_j1_q1, call_j1_q3, call_n_rewriter
from amnesiac.neuter.metrics import (
    RESIDUAL_FINGERPRINT_PATTERNS,
    q3_preservation_score,
    q3_strength_drift,
    residual_fingerprint_check,
)

logger = logging.getLogger(__name__)


async def run_cycle(client: AsyncOpenAI, raw_summary: str) -> dict:
    """
    Run up to N_MAX iterations of N → J1(Q1) → N rewrite → J1(Q3) → residual check
    on raw_summary. In-memory only; no disk I/O.

    Returns:
        {
            "summary": str,
            "neutering_status": str,
            "final_iteration": int,
            "q3_preservation": float,
            "q3_strength_net_shift": int,
            "length_ratio": float,
            "residual_hard_fail_count": int,
            "residual_warn_count": int,
        }
    """
    logger.info("Starting neutering cycle with N_MAX=%s", N_MAX)

    q3_baseline = await call_j1_q3(client, raw_summary)

    accepted_iteration = 0
    accepted_summary = raw_summary
    accepted_q3 = q3_baseline
    accepted_residual = residual_fingerprint_check(raw_summary, RESIDUAL_FINGERPRINT_PATTERNS)
    stop_status = None

    logger.debug(
        "Neutering baseline: hard=%s warn=%s",
        len(accepted_residual["hard_fail"]),
        len(accepted_residual["warn"]),
    )

    for iteration in range(1, N_MAX + 1):
        prev_summary = accepted_summary
        prev_q3 = accepted_q3

        q1 = await call_j1_q1(client, prev_summary)
        candidate_summary = await call_n_rewriter(
            client,
            prev_summary,
            q1.get("identifiers", []),
            prev_q3.get("signals", []),
        )
        q3_candidate = await call_j1_q3(client, candidate_summary)

        q3_preservation = q3_preservation_score(q3_baseline, q3_candidate)
        residual = residual_fingerprint_check(candidate_summary, RESIDUAL_FINGERPRINT_PATTERNS)
        length_ratio = len(candidate_summary) / len(prev_summary) if prev_summary else 0.0

        status = "continue"
        rollback = False
        if q3_preservation < Y_FLOOR:
            status = "signal_collapse"
            rollback = True
        elif residual["manual_fail"]:
            status = "residual_hard_fail"
            rollback = ROLLBACK_ON_HARD_RESIDUAL

        logger.info(
            "Neutering iter_%02d: status=%s q3=%.3f hard=%s warn=%s length_ratio=%.3f",
            iteration,
            status,
            q3_preservation,
            len(residual["hard_fail"]),
            len(residual["warn"]),
            length_ratio,
        )
        if residual["manual_fail"]:
            logger.warning(
                "iter_%02d residual_hard_fail matches: %s",
                iteration,
                [(m["pattern"], m["match"]) for m in residual["hard_fail"]][:30],
            )

        if status == "continue":
            accepted_iteration = iteration
            accepted_summary = candidate_summary
            accepted_q3 = q3_candidate
            accepted_residual = residual
            continue

        stop_status = status
        if rollback:
            logger.info("Rolling back rejected neutering iteration %s with status=%s", iteration, status)
        else:
            accepted_iteration = iteration
            accepted_summary = candidate_summary
            accepted_q3 = q3_candidate
            accepted_residual = residual
        break

    status_for_mapping = stop_status
    if stop_status is None:
        stop_status = "hard_capped"
        logger.info("Neutering cycle reached iteration cap without hard stop")

    if status_for_mapping == "signal_collapse":
        neutering_status = "signal_collapse"
    elif status_for_mapping == "residual_hard_fail":
        neutering_status = "residual_hard_fail"
    elif status_for_mapping is None and not accepted_residual["hard_fail"]:
        neutering_status = "neutered"
    else:
        neutering_status = "hard_capped"

    final_q3_drift = q3_strength_drift(q3_baseline, accepted_q3)
    return {
        "summary": accepted_summary,
        "neutering_status": neutering_status,
        "final_iteration": accepted_iteration,
        "q3_preservation": q3_preservation_score(q3_baseline, accepted_q3),
        "q3_strength_net_shift": final_q3_drift["strength_net_shift"],
        "length_ratio": len(accepted_summary) / len(raw_summary) if raw_summary else 0.0,
        "residual_hard_fail_count": len(accepted_residual["hard_fail"]),
        "residual_warn_count": len(accepted_residual["warn"]),
    }
