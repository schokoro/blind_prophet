import csv
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/blind_prophet_matplotlib")

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for PNG generation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from amnesiac.config import DEFAULT_DB_PATH

SERIES_DATES = [
    "2021-09-20",
    "2021-10-20",
    "2021-11-20",
    "2021-12-20",
    "2022-01-20",
    "2022-02-20",
    "2022-03-20",
    "2022-04-20",
    "2022-05-20",
    "2022-06-20",
]
WAR_START = datetime(2022, 2, 24)
EVAL_DIR = Path("data/eval")

# Status -> marker mapping for neutered plot
STATUS_MARKER = {
    "neutered": "o",  # filled circle
    "hard_capped": "^",  # triangle up
    "wrong_period_remap": "^",  # triangle up
    "residual_hard_fail": "x",  # cross
    "signal_collapse": "x",  # cross
    "judge_blind": "x",  # cross
    "missing": "x",
}
STATUS_LEGEND_LABEL = {
    "neutered": "neutered (clean)",
    "hard_capped": "with caveats",
    "residual_hard_fail": "method failed",
    "signal_collapse": "method failed",
}

# Color palette (consistent across all artifacts)
COLOR_ACTUAL = "#1a1a1a"  # near black for real inFOM
COLOR_RAW = "#d62728"  # red-ish for raw forecast
COLOR_NEUTERED = "#2ca02c"  # green-ish for neutered forecast
COLOR_BAND = None  # to be set as same color with alpha

CSV_COLUMNS = [
    "run_date",
    "infom_actual",
    "raw_mean",
    "raw_p25",
    "raw_p75",
    "raw_error",
    "neutered_mean",
    "neutered_p25",
    "neutered_p75",
    "neutered_error",
    "neutering_status",
    "period_delta",
    "q3_preservation",
    "model_n",
    "abs_error_diff_raw_vs_neutered",
]


def load_series_data(db_path: Path) -> dict:
    """
    Load all data needed for evaluation, returns dict with keys:
    - 'rows': list of dicts per (run_date, condition) with all aggregates
    - 'statuses': dict run_date -> neutering_status
    - 'metadata': dict run_date -> {q3, delta, model_n, final_iteration}
    """
    conn = sqlite3.connect(db_path)
    try:
        # Per-point aggregates
        query = """
        SELECT
            f.run_date,
            f.condition,
            ROUND(AVG(f.value), 3) AS forecast_mean,
            ROUND(AVG(f.value) - (
                SELECT median_12m FROM infom_expectations
                WHERE strftime('%Y-%m', survey_date) = strftime('%Y-%m', f.run_date)
            ), 3) AS error,
            (
                SELECT ROUND(median_12m, 3) FROM infom_expectations
                WHERE strftime('%Y-%m', survey_date) = strftime('%Y-%m', f.run_date)
            ) AS infom_actual,
            (SELECT MIN(value) FROM forecasts f2
             WHERE f2.run_date = f.run_date AND f2.condition = f.condition AND f2.value IS NOT NULL) AS p_min,
            (SELECT MAX(value) FROM forecasts f2
             WHERE f2.run_date = f.run_date AND f2.condition = f.condition AND f2.value IS NOT NULL) AS p_max,
            COUNT(*) AS n_samples,
            SUM(CASE WHEN f.value IS NULL THEN 1 ELSE 0 END) AS n_nulls
        FROM forecasts f
        WHERE f.run_date IN ({placeholders}) AND f.value IS NOT NULL
        GROUP BY f.run_date, f.condition
        ORDER BY f.run_date, f.condition
        """.format(placeholders=",".join("?" * len(SERIES_DATES)))
        rows = [
            dict(zip([col[0] for col in cur.description], r))
            for cur in [conn.execute(query, SERIES_DATES)]
            for r in cur.fetchall()
        ]

        # Compute IQR via two-pass because SQLite does not have percentile_cont.
        for row in rows:
            samples = conn.execute(
                """
                SELECT value
                FROM forecasts
                WHERE run_date = ? AND condition = ? AND value IS NOT NULL
                ORDER BY value
                """,
                (row["run_date"], row["condition"]),
            ).fetchall()
            values = [s[0] for s in samples]
            if values:
                n = len(values)
                row["p25"] = values[max(0, n // 4)]
                row["p75"] = values[min(n - 1, 3 * n // 4)]
                row["median"] = values[n // 2]
            else:
                row["p25"] = row["p75"] = row["median"] = None

        statuses = {}
        metadata = {}
        for run_date in SERIES_DATES:
            res = conn.execute(
                """SELECT neutering_status, ROUND(period_delta_vs_raw, 3),
                          ROUND(q3_preservation, 3), final_iteration, model_n
                   FROM neutered_summaries WHERE run_date = ?""",
                (run_date,),
            ).fetchone()
            if res:
                statuses[run_date] = res[0]
                metadata[run_date] = {
                    "delta": res[1],
                    "q3": res[2],
                    "final_iteration": res[3],
                    "model_n": res[4],
                }
            else:
                statuses[run_date] = "missing"
                metadata[run_date] = {}

        return {"rows": rows, "statuses": statuses, "metadata": metadata}
    finally:
        conn.close()


def build_csv_table(data: dict, output_path: Path = EVAL_DIR / "series_results.csv") -> None:
    """Write final E08 results as CSV."""
    records = _series_records(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record[key]) for key in CSV_COLUMNS})
        writer.writerow(_aggregate_csv_row(records))


def build_markdown_table(data: dict, output_path: Path = EVAL_DIR / "series_results.md") -> None:
    """Write final E08 results as a copy-pasteable Markdown table."""
    records = _series_records(data)
    headers = [
        "run_date",
        "inFOM",
        "raw mean",
        "raw IQR",
        "raw error",
        "neutered mean",
        "neutered IQR",
        "neutered error",
        "status",
        "delta",
        "q3",
        "model_n",
        "abs diff",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record["run_date"],
                    _md_number(record["infom_actual"]),
                    _md_number(record["raw_mean"]),
                    _md_iqr(record["raw_p25"], record["raw_p75"]),
                    _md_number(record["raw_error"]),
                    _md_number(record["neutered_mean"]),
                    _md_iqr(record["neutered_p25"], record["neutered_p75"]),
                    _md_number(record["neutered_error"]),
                    _status_label(record["neutering_status"]),
                    _md_number(record["period_delta"]),
                    _md_number(record["q3_preservation"]),
                    str(record["model_n"] or "-"),
                    _md_number(record["abs_error_diff_raw_vs_neutered"]),
                ]
            )
            + " |"
        )
    lines.append(_aggregate_markdown_row(records, len(headers)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_plot_raw(data: dict, output_path: Path = EVAL_DIR / "plot_raw.png") -> None:
    """Build raw forecast vs inFOM plot."""
    records = _series_records(data)
    fig, ax = plt.subplots(figsize=(16, 9))
    _plot_condition(ax, records, "raw", COLOR_RAW, "Прогноз модели на raw саммари vs фактический инФОМ")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_plot_neutered(data: dict, output_path: Path = EVAL_DIR / "plot_neutered.png") -> None:
    """Build neutered forecast vs inFOM plot."""
    records = _series_records(data)
    fig, ax = plt.subplots(figsize=(16, 9))
    _plot_condition(
        ax,
        records,
        "neutered",
        COLOR_NEUTERED,
        "Прогноз модели на neutered саммари vs фактический инФОМ",
        status_markers=True,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_plot_combined(data: dict, output_path: Path = EVAL_DIR / "plot_combined.png") -> None:
    """Build side-by-side raw and neutered plots."""
    records = _series_records(data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    raw_handles = _plot_condition(
        axes[0],
        records,
        "raw",
        COLOR_RAW,
        "Raw",
        show_legend=False,
    )
    neutered_handles = _plot_condition(
        axes[1],
        records,
        "neutered",
        COLOR_NEUTERED,
        "Neutered",
        status_markers=True,
        show_legend=False,
    )
    handles = _dedupe_handles(raw_handles + neutered_handles)
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("forecast vs инФОМ — raw vs neutered (n=10 точек)", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_narrative_template(
    data: dict, output_path: Path = EVAL_DIR / "narrative_template.md"
) -> None:
    """Write Markdown narrative template with DB-derived numbers."""
    records = _series_records(data)
    status_dates: dict[str, list[str]] = {}
    for record in records:
        status_dates.setdefault(record["neutering_status"], []).append(record["run_date"])

    raw_total = _sum_abs(records, "raw_error")
    neutered_total = _sum_abs(records, "neutered_error")
    improvement = None
    if raw_total not in (None, 0) and neutered_total is not None:
        improvement = (raw_total - neutered_total) / raw_total * 100

    lines = [
        "# E08 forecast — нарративный разбор серии",
        "",
        "## Сводка",
        "",
        "- **Точек в серии:** 10",
        "- **Распределение статусов нейтрализации:**",
    ]
    for status in sorted(status_dates):
        dates = ", ".join(status_dates[status])
        lines.append(f"  - {status}: {len(status_dates[status])} (даты: {dates})")
    lines.extend(
        [
            "- **Total abs error:**",
            f"  - raw: {_md_number(raw_total)}",
            f"  - neutered: {_md_number(neutered_total)}",
            f"  - улучшение: {_md_number(improvement)}%",
            "- **Sign-match по серии:** ... (заполнить руками)",
            "",
            "## Разбор по точкам",
            "",
        ]
    )

    for record in records:
        lines.extend(
            [
                f"### {record['run_date']} — {_period_label(record['run_date'])}",
                "",
                (
                    f"- инФОМ: {_md_number(record['infom_actual'])}, "
                    f"raw: {_md_number(record['raw_mean'])}, "
                    f"neutered: {_md_number(record['neutered_mean'])}"
                ),
                (
                    f"- raw error: {_md_number(record['raw_error'])}, "
                    f"neutered error: {_md_number(record['neutered_error'])}"
                ),
                (
                    f"- статус нейтрализации: {record['neutering_status']}, "
                    f"delta: {_md_number(record['period_delta'])}, "
                    f"q3: {_md_number(record['q3_preservation'])}"
                ),
                f"- модель N: {record['model_n'] or '-'}",
                "",
                (
                    "[ЗАПОЛНИ РУКАМИ]: содержательный комментарий — что точка показывает, "
                    "какие персоны двигают, что интересного."
                ),
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Build all evaluation artifacts."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {db_path}...")
    data = load_series_data(db_path)

    print("Building CSV table...")
    build_csv_table(data, EVAL_DIR / "series_results.csv")

    print("Building Markdown table...")
    build_markdown_table(data, EVAL_DIR / "series_results.md")

    print("Building plot_raw.png...")
    build_plot_raw(data, EVAL_DIR / "plot_raw.png")

    print("Building plot_neutered.png...")
    build_plot_neutered(data, EVAL_DIR / "plot_neutered.png")

    print("Building plot_combined.png...")
    build_plot_combined(data, EVAL_DIR / "plot_combined.png")

    print("Building narrative template...")
    build_narrative_template(data, EVAL_DIR / "narrative_template.md")

    print(f"\nAll artifacts in {EVAL_DIR}/")


def _series_records(data: dict) -> list[dict[str, Any]]:
    rows_by_key = {(row["run_date"], row["condition"]): row for row in data["rows"]}
    records = []
    for run_date in SERIES_DATES:
        raw = rows_by_key.get((run_date, "raw"), {})
        neutered = rows_by_key.get((run_date, "neutered"), {})
        metadata = data["metadata"].get(run_date, {})
        infom_actual = raw.get("infom_actual", neutered.get("infom_actual"))
        raw_error = raw.get("error")
        neutered_error = neutered.get("error")
        abs_error_diff = None
        if raw_error is not None and neutered_error is not None:
            abs_error_diff = abs(raw_error) - abs(neutered_error)
        records.append(
            {
                "run_date": run_date,
                "infom_actual": infom_actual,
                "raw_mean": raw.get("forecast_mean"),
                "raw_p25": raw.get("p25"),
                "raw_p75": raw.get("p75"),
                "raw_median": raw.get("median"),
                "raw_error": raw_error,
                "neutered_mean": neutered.get("forecast_mean"),
                "neutered_p25": neutered.get("p25"),
                "neutered_p75": neutered.get("p75"),
                "neutered_median": neutered.get("median"),
                "neutered_error": neutered_error,
                "neutering_status": data["statuses"].get(run_date, "missing"),
                "period_delta": metadata.get("delta"),
                "q3_preservation": metadata.get("q3"),
                "model_n": metadata.get("model_n"),
                "abs_error_diff_raw_vs_neutered": abs_error_diff,
            }
        )
    return records


def _aggregate_csv_row(records: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "run_date": "aggregate",
        "infom_actual": "-",
        "raw_mean": _csv_value(_mean(records, "raw_mean")),
        "raw_p25": "-",
        "raw_p75": "-",
        "raw_error": _csv_value(_sum_abs(records, "raw_error")),
        "neutered_mean": _csv_value(_mean(records, "neutered_mean")),
        "neutered_p25": "-",
        "neutered_p75": "-",
        "neutered_error": _csv_value(_sum_abs(records, "neutered_error")),
        "neutering_status": "-",
        "period_delta": "-",
        "q3_preservation": "-",
        "model_n": "-",
        "abs_error_diff_raw_vs_neutered": "-",
    }


def _aggregate_markdown_row(records: list[dict[str, Any]], n_cols: int) -> str:
    cells = [
        "**aggregate**",
        "**-**",
        f"**{_md_number(_mean(records, 'raw_mean'))}**",
        "**-**",
        f"**{_md_number(_sum_abs(records, 'raw_error'))}**",
        f"**{_md_number(_mean(records, 'neutered_mean'))}**",
        "**-**",
        f"**{_md_number(_sum_abs(records, 'neutered_error'))}**",
        "**-**",
        "**-**",
        "**-**",
        "**-**",
        "**-**",
    ]
    if len(cells) != n_cols:
        raise ValueError("Aggregate Markdown row has incorrect number of columns")
    return "| " + " | ".join(cells) + " |"


def _plot_condition(
    ax: Axes,
    records: list[dict[str, Any]],
    condition: str,
    color: str,
    title: str,
    *,
    status_markers: bool = False,
    show_legend: bool = True,
) -> list[Line2D]:
    dates = [datetime.strptime(record["run_date"], "%Y-%m-%d") for record in records]
    actual = [_plot_value(record["infom_actual"]) for record in records]
    median = [_plot_value(record[f"{condition}_median"]) for record in records]
    p25 = [_plot_value(record[f"{condition}_p25"]) for record in records]
    p75 = [_plot_value(record[f"{condition}_p75"]) for record in records]

    actual_line = ax.plot(
        dates,
        actual,
        color=COLOR_ACTUAL,
        marker="o",
        linewidth=3,
        label="инФОМ actual",
    )[0]
    forecast_line = ax.plot(
        dates,
        median,
        color=color,
        linewidth=2.5,
        label=f"{condition} forecast median",
    )[0]
    if not status_markers:
        forecast_line.set_marker("o")
    else:
        for date, value, record in zip(dates, median, records, strict=True):
            status = record["neutering_status"]
            ax.plot(
                [date],
                [value],
                color=color,
                marker=STATUS_MARKER.get(status, "x"),
                markersize=8,
                linestyle="None",
                markeredgewidth=2,
            )

    ax.fill_between(dates, p25, p75, color=color, alpha=0.2, label=f"{condition} IQR")
    ax.axvline(WAR_START, color="#808080", linestyle="--", linewidth=1.5, alpha=0.8, label="war start")
    y_top = _axis_top(actual + median + p25 + p75)
    ax.text(WAR_START, y_top, "war start", rotation=90, va="top", ha="right", color="#666666")
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("% инфляционных ожиданий, 12 месяцев")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(bottom=0, top=y_top)

    handles = [actual_line, forecast_line]
    handles.append(Line2D([0], [0], color=color, linewidth=8, alpha=0.2, label=f"{condition} IQR"))
    handles.append(Line2D([0], [0], color="#808080", linestyle="--", label="war start"))
    if status_markers:
        handles.extend(_status_legend_handles(color))
    if show_legend:
        ax.legend(handles=handles, loc="best")
    return handles


def _status_legend_handles(color: str) -> list[Line2D]:
    return [
        Line2D([0], [0], color=color, marker="o", linestyle="None", label="neutered"),
        Line2D([0], [0], color=color, marker="^", linestyle="None", label="with caveats"),
        Line2D([0], [0], color=color, marker="x", linestyle="None", label="method failed"),
    ]


def _dedupe_handles(handles: list[Line2D]) -> list[Line2D]:
    seen = set()
    deduped = []
    for handle in handles:
        label = handle.get_label()
        if label not in seen:
            seen.add(label)
            deduped.append(handle)
    return deduped


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _sum_abs(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record[key] is not None]
    if not values:
        return None
    return sum(abs(value) for value in values)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _md_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _md_iqr(p25: Any, p75: Any) -> str:
    if p25 is None or p75 is None:
        return "-"
    return f"{_md_number(p25)}–{_md_number(p75)}"


def _status_label(status: str) -> str:
    if status == "neutered":
        return f"✅ {status}"
    if status in {"hard_capped", "wrong_period_remap"}:
        return f"⚠️ {status}"
    if status in {"residual_hard_fail", "signal_collapse", "judge_blind", "missing"}:
        return f"❌ {status}"
    return status


def _plot_value(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def _axis_top(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 1.0
    return max(finite) * 1.12


def _period_label(run_date: str) -> str:
    labels = {
        "2021-10-20": "спокойный период (calm_2021 анкер)",
        "2022-03-20": "военный шок",
    }
    return labels.get(run_date, "спокойный период")


if __name__ == "__main__":
    main()
