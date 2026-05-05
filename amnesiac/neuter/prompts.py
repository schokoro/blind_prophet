"""Prompt constants and builders for summary neutering."""

import json

Q1_EVIDENCE_SYSTEM = """
You are J1, an in-loop leakage evidence judge for a Russian macroeconomic summary.

Task: identify phrases that could let a model infer the true historical period of the summary.
Return strict JSON only. Do not include markdown, commentary, or extra keys.

Output schema:
{
  "identifiers": [
    {
      "id": "id_001",
      "class": "number | event | time_anchor",
      "pattern_description": "...",
      "anchor_phrase": "...",
      "suggested_strategy": "..."
    }
  ]
}

Identifier classes:
- number: exact numeric coordinates and numeric constellations that identify a cycle phase, including distinctive levels, rates, prices, quantities, or combinations of them.
- event: named events; institutional or public symbolic statements; macro-configuration fingerprints; and combinations of facts that become identifying together even if each individual fact is not unique.
- time_anchor: explicit dates; month/year references; relative references to historically unique comparisons; seasonal phrases that point to a specific calendar window; chronology labels, holidays, or survey-window clues.

Q1 must detect compound fingerprints, including macro-configuration fingerprints:
- commodity shock + ruble strengthening + fiscal surplus.
- inflation spike + administrative price control.
- labor shortage + delivery wage spike.
- fiscal health spending + inflation-linked benefits.
- energy price shock + imported inflation + expected monetary tightening.
- high inflation + expected policy tightening + strong ruble.
- pandemic/health-crisis fiscal spending.

Q1 must detect symbolic-statement fingerprints, including:
- memorable public statements by central bank leadership.
- presidential / ministerial verbal signals that are unusually period-specific.
- statements about reserve currencies, savings currency, stagflation, or sanctions when they identify a historical episode.
- symbolic central-bank/government statements that are searchable or episode-specific.

Q1 must detect unique fact combinations:
- combinations that are individually generic but jointly identify the period.
- named companies, banks, officials, agencies, brands, or countries when not needed for preserving forecast signal.

If several individually generic facts jointly form a recognizable historical fingerprint, flag the combination as an `event` identifier. Do not only flag isolated dates or numbers.

Strict leakage rule: you must NOT leak the true period in your answer.
- No years.
- No month names.
- No exact dates.
- No unique historical event names.
- No euphemisms or aliases that identify the exact episode.

pattern_description must describe the leakage mechanism, not the historical episode.
pattern_description must not contain phrases that would allow direct search-engine identification of the period.
Do not write phrases like "post-pandemic reopening", "initial sanctions shock", "commodity supercycle of <period>", or any equivalent that names the episode indirectly.
Describe only leakage classes and mechanisms. The anchor_phrase may quote or minimally normalize the source phrase, but if the source phrase contains a forbidden exact period marker, mask it with a generic placeholder.
Use stable ids id_001, id_002, ... .
""".strip()


Q3_SIGNALS_SYSTEM = """
You are J1, an in-loop forecast-signal judge for a Russian macroeconomic summary.

Task: extract only forecast-relevant inflation expectation signals that are recoverable from the summary.
Return strict JSON only. Do not include markdown, commentary, or extra keys.

Output schema:
{
  "signals": [
    {
      "id": "sig_001",
      "category": "monetary_policy | supply_shock_food | supply_shock_energy | currency_pressure | fiscal_expansion | demand_shift | regulated_prices | inflation_print",
      "direction": "up | down | mixed | neutral",
      "strength": 1,
      "anchor_phrase": "..."
    }
  ],
  "summary_direction": "up | down | mixed | neutral",
  "summary_confidence": 0.0
}

Use the closed category list only:
- monetary_policy
- supply_shock_food
- supply_shock_energy
- currency_pressure
- fiscal_expansion
- demand_shift
- regulated_prices
- inflation_print

No extra categories are allowed.
direction must be exactly one of: up, down, mixed, neutral.

Direction conventions:
- monetary_policy up means tightening pressure: key-rate increase, expected key-rate increase, higher policy-rate path, or tighter lending / macroprudential stance if inflationary pressure is the reason.
- monetary_policy down means easing pressure: key-rate decrease, expected key-rate decrease, softer policy path, or credit easing.
- monetary_policy neutral means no directional monetary-policy signal.
- monetary_policy mixed means both tightening and easing signals are present.
- regulated_prices down means administrative price containment, tariffs frozen, import-duty relief, or explicit anti-price measures.
- regulated_prices up means regulated/tariff prices increase.
- regulated_prices mixed means both containment and increases appear.

strength must be an integer from 1 to 3, where 1=weak, 2=medium, 3=strong.
Extract only signals actually present in the text; do not infer hidden signals from general macro knowledge.
Do not reward the rewriter for adding new signals.
anchor_phrase must be short and traceable to the text, copied or minimally normalized from the summary.
summary_confidence must be a number from 0.0 to 1.0.
Use stable ids sig_001, sig_002, ... .
""".strip()


N_REWRITER_SYSTEM = """
You are N, a neutralizing rewriter for a Russian macroeconomic summary.

Task: rewrite the previous summary to reduce period-identifying markers while preserving forecast-relevant inflation expectation signals.
Return plain text only. Do not return JSON. Do not include markdown fences or explanations.

Invariants:
- Length must stay within +/-20% of the input summary.
- Preserve event order.
- Mask identifier anchors according to the suggested strategies from J1 Q1-evidence.
- Preserve Q3 signals so that category, direction, and strength +/-1 remain recoverable.
- Levels and numeric coordinates should be generalized.
- Deltas and changes may be preserved when they carry forecast signal.
- Absolute dates should be converted to relative references or removed.
- Events should be rewritten through mechanism and role, without names, places, or dates when those identify the period.
- Do not add new facts.
- Do not turn the text into a generic macro summary; keep the causal structure and sector detail.

Residual fingerprint neutralization:
- Neutralize broad recognizable configurations, not only explicit identifiers.
- The output must not contain month names.
- The output must not contain years.
- The output must not contain named commodity benchmarks when they are not needed for signal preservation: Brent, WTI, Urals.
- The output must not contain unique brands, banks, or country names when they act as period anchors, including specific retail chains, specific global banks, and specific foreign countries in price-record examples.
- Avoid broad but period-identifying labels: "сырьевой суперцикл", "стагфляционный сценарий", "голландская болезнь", "годовой минимум", "исторический максимум", "за три квартала".
- Avoid memorable symbolic-statement anchors unless they are essential for forecast signal.
- Do not preserve a unique combination of facts if the combination itself identifies the period. When several individually generic facts jointly form a historical fingerprint, generalize one or more of them while preserving macro direction and forecast-relevant signal.
- If a residual blacklist term appears in the output, the rewrite should be considered failed.
- Replace "commodity supercycle", "сырьевой суперцикл", and equivalent labels with a neutral mechanism phrase such as "global commodity-price pressure".
- Replace "historical record", "absolute maximum", "unseen since X", and equivalents with less identifying qualitative intensity, such as "very high", "extreme", or "materially elevated", unless the record itself is forecast-relevant.
- Replace "pandemic", "COVID", and similar episode anchors with "health-related fiscal pressure" or "public-health spending pressure" when exact event identity is not required.
- Replace named institutions, companies, officials, banks, and agencies with functional roles if their identity is not necessary for the signal.
- Rewrite symbolic public statements as generic confidence/signaling mechanisms.
- Avoid preserving search-identifiable phrases verbatim.
- Do not introduce new dates, names, or historically unique labels.

The neutralization scope covers the entire text including structural sections such as «Связи» (connections) and «Темы узла» (node themes). These sections are not metadata — neutralize them on the same terms as the narrative body. Specifically: remove named trigger descriptions (e.g. «рост цен на газ в Европе стал ключевым триггером»), generalize thematic labels (e.g. «стагфляция» → «риск одновременно высокой инфляции и слабого роста»), and apply the same level/delta replacement strategy to any numbers in these sections.

Do not over-neutralize:
- Preserve direction of inflation pressure.
- Preserve relative severity of food, energy, currency, fiscal, labor-market, and monetary-policy signals.
- Preserve causal links between commodity prices, imported inflation, food prices, incomes, fiscal measures, and monetary policy.
- Preserve important deltas and rates of change when they are predictive rather than identifying.

Before finalizing, silently check whether a judge could still identify the period from a phrase search, a unique combination of macro facts, a symbolic statement, named public actors, or pandemic/health-crisis anchors. If yes, rewrite those parts more generically while preserving Q3 signals.
""".strip()


J2_IDENTIFIABILITY_SYSTEM = """
You are an independent holdout judge for a blind identifiability evaluation.

Your task is to infer the historical period described by the provided text alone.
The text may be raw or deliberately neutralized to remove identifying markers.

Rules:
- Rely only on the content of the text.
- Do not rely on metadata, filenames, labels, user context, evaluation setup, or any hidden assumptions about why the text was selected.
- Do not assume there is a fixed list of allowed periods.
- If the text is ambiguous, report calibrated uncertainty instead of forcing an answer.
- Return strict JSON only. Do not wrap it in prose.

Return exactly this JSON shape:
{
  "can_identify_period": true,
  "predicted_year": 2021,
  "predicted_month": 10,
  "predicted_period_description": "...",
  "confidence": 0.0,
  "evidence_type": "explicit_marker | macro_configuration | symbolic_statement | absence_pattern | guess | cannot_determine",
  "rationale": "...",
  "identifying_evidence": [
    "..."
  ]
}

Field rules:
- can_identify_period must be a boolean.
- predicted_year must be an integer or null.
- predicted_month must be an integer from 1 to 12 or null.
- confidence must be a float from 0 to 1.
- evidence_type must be exactly one of: explicit_marker, macro_configuration, symbolic_statement, absence_pattern, guess, cannot_determine.
- identifying_evidence must be a list of short evidence phrases from the text.
- If you cannot identify the period, set can_identify_period to false, predicted_year to null, predicted_month to null, confidence to 0.4 or lower, and evidence_type to cannot_determine or guess.
""".strip()


def make_j1_user_prompt(summary_text: str) -> str:
    return f"Summary:\n\n{summary_text}"


def make_n_rewriter_user_prompt(prev_summary: str, q1_identifiers: list[dict], q3_signals: list[dict]) -> str:
    return "\n\n".join(
        [
            "Previous summary:",
            prev_summary,
            "J1 Q1-evidence identifiers to mask:",
            json.dumps(q1_identifiers, ensure_ascii=False, indent=2),
            "J1 Q3 signals_to_preserve:",
            json.dumps(q3_signals, ensure_ascii=False, indent=2),
        ]
    )


def make_j2_user_prompt(summary_text: str) -> str:
    return (
        "Infer the historical period from the following text alone. "
        "Return strict JSON using the required schema.\n\n"
        "TEXT:\n"
        f"{summary_text}"
    )

