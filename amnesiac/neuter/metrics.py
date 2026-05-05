"""Pure scoring and fingerprint metrics for summary neutering."""

import json
import re
from typing import Any, Optional

MONTH_PATTERN = (
    r"(?<![А-Яа-яЁё])"
    r"(?:январ[ьяе]?|феврал[ьяе]?|март(?:а|е)?|апрел[ьяе]?|"
    r"май|мая|мае|июн[ьяе]?|июл[ьяе]?|август[ае]?|"
    r"сентябр[ьяе]?|октябр[ьяе]?|ноябр[ьяе]?|декабр[ьяе]?)"
    r"(?![А-Яа-яЁё])"
)


RESIDUAL_FINGERPRINT_PATTERNS = {
    "hard_fail": [
        MONTH_PATTERN,
        r"\b20\d{2}\b|\b19\d{2}\b",
        r"COVID|ковид|коронавирус|пандем",
        r"Citi|Пят[её]роч|ФРГ|SWIFT|санкц|контрсанкц|спецоперац|военн\w+ операц|Украин|Донбасс",
        r"Brent|WTI|Urals",
    ],
    "warn": [
        r"историческ\w+ максимум",
        r"годов\w+ минимум",
        r"стагфляц",
        r"голландск\w+ болезн",
        r"сырьев\w+ суперцикл",
        r"за три квартала",
        r"овсян\w+ хлоп",
        r"бюджетн\w+ правил",
        r"резервн\w+ валют",
        r"газ\w* в Европе.*ключев\w+ триггер",
        r"посткризисн\w+ волатильност",
        r"глава центробанка.*сбережен",
        r"хранит сбережен",
        r"заморозк\w+ резерв",
        r"капитал\w+ контрол",
        r"параллельн\w+ импорт",
        r"недружественн\w+ стран",
    ],
}


EVIDENCE_TYPES = {
    "explicit_marker",
    "macro_configuration",
    "symbolic_statement",
    "absence_pattern",
    "guess",
    "cannot_determine",
}

REQUIRED_J2_FIELDS = {
    "can_identify_period",
    "predicted_year",
    "predicted_month",
    "predicted_period_description",
    "confidence",
    "evidence_type",
    "rationale",
    "identifying_evidence",
}

PERIOD_WEIGHTS = {
    "exact_month": 1.0,
    "adjacent_month": 0.75,
    "same_quarter": 0.5,
    "same_year": 0.25,
    "wrong_period": 0.0,
    "no_prediction": 0.0,
}


def q3_preservation_score(baseline: dict, candidate: dict) -> float:
    baseline_signals = baseline.get("signals", [])
    candidate_signals = candidate.get("signals", [])
    if not baseline_signals:
        return 1.0

    candidate_by_category = {}
    for signal in candidate_signals:
        candidate_by_category.setdefault(signal.get("category"), []).append(signal)

    matched = []
    direction_matches = 0
    strength_consistent = 0

    for base_signal in baseline_signals:
        category = base_signal.get("category")
        candidates = candidate_by_category.get(category, [])
        if not candidates:
            continue
        cand_signal = candidates.pop(0)
        matched.append((base_signal, cand_signal))

        if cand_signal.get("direction") == base_signal.get("direction"):
            direction_matches += 1

        try:
            base_strength = int(base_signal.get("strength"))
            cand_strength = int(cand_signal.get("strength"))
        except (TypeError, ValueError):
            continue
        if abs(base_strength - cand_strength) <= 1:
            strength_consistent += 1

    recall = len(matched) / len(baseline_signals)
    if not matched:
        return 0.0

    direction_match_rate = direction_matches / len(matched)
    strength_consistency_rate = strength_consistent / len(matched)
    return recall * direction_match_rate * strength_consistency_rate


def q3_strength_drift(previous: dict, candidate: dict) -> dict:
    candidate_by_category = {}
    for signal in candidate.get("signals", []):
        candidate_by_category.setdefault(signal.get("category"), []).append(signal)

    upgrades = 0
    downgrades = 0
    for prev_signal in previous.get("signals", []):
        category = prev_signal.get("category")
        candidates = candidate_by_category.get(category, [])
        if not candidates:
            continue
        cand_signal = candidates.pop(0)
        try:
            prev_strength = int(prev_signal.get("strength"))
            cand_strength = int(cand_signal.get("strength"))
        except (TypeError, ValueError):
            continue
        if cand_strength > prev_strength:
            upgrades += 1
        elif cand_strength < prev_strength:
            downgrades += 1

    return {
        "strength_upgrades": upgrades,
        "strength_downgrades": downgrades,
        "strength_net_shift": upgrades - downgrades,
    }


def residual_fingerprint_check(text: str, patterns: dict[str, list[str]]) -> dict:
    result = {"hard_fail": [], "warn": [], "manual_fail": False}
    for severity in ("hard_fail", "warn"):
        for pattern in patterns[severity]:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                result[severity].append(
                    {
                        "pattern": pattern,
                        "match": match.group(0),
                        "span": [match.start(), match.end()],
                    }
                )
    result["manual_fail"] = bool(result["hard_fail"])
    return result


def extract_json_response(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse J2 response as JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("J2 JSON response must be an object")
    return parsed


def validate_j2_parsed(parsed: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_J2_FIELDS - set(parsed))
    if missing:
        raise ValueError(f"J2 JSON response is missing fields: {missing}")

    if not isinstance(parsed["can_identify_period"], bool):
        raise TypeError("can_identify_period must be a boolean")
    if parsed["predicted_year"] is not None and not isinstance(parsed["predicted_year"], int):
        raise TypeError("predicted_year must be an integer or null")
    if parsed["predicted_month"] is not None:
        if not isinstance(parsed["predicted_month"], int) or not 1 <= parsed["predicted_month"] <= 12:
            raise ValueError("predicted_month must be an integer from 1 to 12 or null")
    if not isinstance(parsed["confidence"], (int, float)):
        raise TypeError("confidence must be numeric")
    if parsed["evidence_type"] not in EVIDENCE_TYPES:
        raise ValueError(f"Invalid evidence_type: {parsed['evidence_type']}")
    if not isinstance(parsed["identifying_evidence"], list):
        raise TypeError("identifying_evidence must be a list")


def clamp_confidence(value: Any) -> float:
    confidence = float(value)
    return max(0.0, min(1.0, confidence))


def month_index(year: int, month: int) -> int:
    return year * 12 + month


def month_distance(
    pred_year: Optional[int],
    pred_month: Optional[int],
    true_year: int,
    true_month: int,
) -> Optional[int]:
    if pred_year is None or pred_month is None:
        return None
    if not (1 <= int(pred_month) <= 12):
        return None
    return abs(month_index(int(pred_year), int(pred_month)) - month_index(int(true_year), int(true_month)))


def same_calendar_quarter(year_a: int, month_a: int, year_b: int, month_b: int) -> bool:
    if year_a != year_b:
        return False
    return (month_a - 1) // 3 == (month_b - 1) // 3


def score_identification(parsed: dict, true_year: int, true_month: int) -> dict:
    confidence = clamp_confidence(parsed.get("confidence", 0.0))
    can_identify = parsed.get("can_identify_period") is True
    pred_year = parsed.get("predicted_year")
    pred_month = parsed.get("predicted_month")

    dist = month_distance(pred_year, pred_month, true_year, true_month) if can_identify else None

    exact_success = bool(
        can_identify
        and pred_year == true_year
        and pred_month == true_month
    )
    adjacent_month_success = bool(
        can_identify
        and dist is not None
        and dist <= 1
    )
    same_quarter_success = bool(
        can_identify
        and pred_year is not None
        and pred_month is not None
        and same_calendar_quarter(int(pred_year), int(pred_month), true_year, true_month)
    )
    same_year_success = bool(
        can_identify
        and pred_year == true_year
    )

    if not can_identify or pred_year is None or pred_month is None:
        period_success_level = "no_prediction"
    elif exact_success:
        period_success_level = "exact_month"
    elif adjacent_month_success:
        period_success_level = "adjacent_month"
    elif same_quarter_success:
        period_success_level = "same_quarter"
    elif same_year_success:
        period_success_level = "same_year"
    else:
        period_success_level = "wrong_period"

    period_weight = PERIOD_WEIGHTS[period_success_level]

    return {
        "true_year": true_year,
        "true_month": true_month,
        "predicted_year": pred_year,
        "predicted_month": pred_month,
        "confidence": confidence,
        "month_distance": dist,
        "exact_success": exact_success,
        "adjacent_month_success": adjacent_month_success,
        "same_quarter_success": same_quarter_success,
        "same_year_success": same_year_success,
        "period_success_level": period_success_level,
        "period_weight": period_weight,
        "exact_weighted_success": confidence if exact_success else 0.0,
        "period_weighted_success": round(confidence * period_weight, 6),
        "scoring_version": "period_aware_v2",
        # backward compat
        "success": exact_success,
        "weighted_success": confidence if exact_success else 0.0,
    }

