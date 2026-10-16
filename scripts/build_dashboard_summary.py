"""Create small descriptive dashboard aggregates from the validated fire cohort.

Reads four required columns; never exports individual fire records or changes
models. Calendar durations are used consistently across reporting eras.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIZE_ORDER = ["Micro", "Small", "Medium", "Large", "1000+ acres"]
COLUMNS = ["FIRE_YEAR", "SIZE_BIN", "NWCG_CAUSE_CLASSIFICATION", "CONTAINMENT_DAYS_DATE_ONLY"]


def summarize_groups(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, observed=True, sort=True)
    summary = grouped.agg(
        count=("DAYS", "size"),
        same_calendar_day_count=("SAME_DAY", "sum"),
        over_3_days_count=("OVER_3", "sum"),
    )
    quantiles = grouped.DAYS.quantile([.5, .75, .9]).unstack(-1)
    quantiles.columns = ["median_days", "p75_days", "p90_days"]
    summary = summary.join(quantiles)
    summary["same_calendar_day_fraction"] = summary.same_calendar_day_count / summary["count"]
    summary["over_3_days_fraction"] = summary.over_3_days_count / summary["count"]
    return summary.reset_index()


def records(frame: pd.DataFrame) -> list[dict]:
    # pandas' JSON conversion resolves its integer/float scalars to JSON-native
    # values; validation below separately rejects nonfinite summary measures.
    return json.loads(frame.to_json(orient="records", double_precision=12))


def require_partition(summary: pd.DataFrame, count: int, name: str) -> None:
    if int(summary["count"].sum()) != count:
        raise AssertionError(f"{name} does not partition the fire cohort")
    if summary.empty or summary["count"].le(0).any():
        raise AssertionError(f"{name} has empty groups")
    numeric = summary.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError(f"{name} has nonfinite metrics")
    if not (summary.median_days.le(summary.p75_days) & summary.p75_days.le(summary.p90_days)).all():
        raise AssertionError(f"{name} has unordered quantiles")
    for metric in ["same_calendar_day_fraction", "over_3_days_fraction"]:
        if not summary[metric].between(0, 1).all():
            raise AssertionError(f"{name} has invalid proportions")


def build_summary(path: Path) -> dict:
    frame = pd.read_parquet(path, columns=COLUMNS)
    if frame.empty:
        raise ValueError("The dashboard cohort is empty")
    frame = frame.rename(columns={"CONTAINMENT_DAYS_DATE_ONLY": "DAYS"})
    frame["DAYS"] = pd.to_numeric(frame.DAYS, errors="raise")
    if not np.isfinite(frame.DAYS).all() or not frame.DAYS.between(0, 365).all():
        raise ValueError("Calendar durations must be finite and between 0 and 365 days")
    if not np.equal(frame.DAYS, np.floor(frame.DAYS)).all():
        raise ValueError("The calendar-day outcome must be integral")
    years = pd.to_numeric(frame.FIRE_YEAR, errors="raise")
    if not np.isfinite(years).all() or not np.equal(years, np.floor(years)).all():
        raise ValueError("Fire years must be finite integers")
    frame["FIRE_YEAR"] = years.astype(int)
    frame["SIZE_BIN"] = frame.SIZE_BIN.astype("string")
    if frame.SIZE_BIN.isna().any() or not frame.SIZE_BIN.isin(SIZE_ORDER).all():
        raise ValueError("Unexpected or missing acreage-bin labels")
    frame["CAUSE"] = frame.NWCG_CAUSE_CLASSIFICATION.astype("string").str.strip().fillna("Unknown")
    frame.loc[frame.CAUSE.eq(""), "CAUSE"] = "Unknown"
    frame["SAME_DAY"] = frame.DAYS.eq(0)
    frame["OVER_3"] = frame.DAYS.gt(3)
    frame["ERA_START"] = frame.FIRE_YEAR // 5 * 5
    count = len(frame)
    quantiles = frame.DAYS.quantile([.5, .75, .9]).to_numpy()
    cohort = {
        "count": int(count),
        "same_calendar_day_count": int(frame.SAME_DAY.sum()),
        "same_calendar_day_fraction": float(frame.SAME_DAY.mean()),
        "over_3_days_count": int(frame.OVER_3.sum()),
        "over_3_days_fraction": float(frame.OVER_3.mean()),
        "median_days": float(quantiles[0]), "p75_days": float(quantiles[1]), "p90_days": float(quantiles[2]),
    }
    year_size = summarize_groups(frame, ["FIRE_YEAR", "SIZE_BIN"])
    era_size = summarize_groups(frame, ["ERA_START", "SIZE_BIN"])
    observed_era = frame.groupby("ERA_START").FIRE_YEAR.agg(observed_year_min="min", observed_year_max="max")
    era_size = era_size.merge(observed_era, left_on="ERA_START", right_index=True, validate="many_to_one")
    era_size["era_end"] = era_size.ERA_START + 4
    era_size["era_label"] = era_size.ERA_START.astype(str) + "–" + era_size.era_end.astype(str)
    sizes = summarize_groups(frame, ["SIZE_BIN"])
    causes = summarize_groups(frame, ["CAUSE"])
    tables = {"year_by_size": year_size, "era_by_size": era_size, "size_overview": sizes, "cause_overview": causes}
    for name, table in tables.items():
        require_partition(table, count, name)
        if int(table.same_calendar_day_count.sum()) != cohort["same_calendar_day_count"]:
            raise AssertionError(f"{name} same-day counts do not match the cohort")
        if int(table.over_3_days_count.sum()) != cohort["over_3_days_count"]:
            raise AssertionError(f"{name} >3-day counts do not match the cohort")
    yearly_counts = frame.groupby("FIRE_YEAR").size()
    if not year_size.groupby("FIRE_YEAR")["count"].sum().equals(yearly_counts):
        raise AssertionError("Year-size counts do not recover year totals")
    era_counts = frame.groupby("ERA_START").size()
    if not era_size.groupby("ERA_START")["count"].sum().equals(era_counts):
        raise AssertionError("Era-size counts do not recover era totals")
    by_size = sizes.set_index("SIZE_BIN")["count"].sort_index()
    for table in [year_size, era_size]:
        if not table.groupby("SIZE_BIN")["count"].sum().sort_index().equals(by_size):
            raise AssertionError("Time aggregation does not recover size totals")
    rename = {"FIRE_YEAR": "year", "SIZE_BIN": "size_bin", "ERA_START": "era_start", "CAUSE": "cause"}
    for name, table in tables.items():
        table = table.rename(columns=rename)
        if "size_bin" in table:
            table["_size_order"] = table.size_bin.map({label: index for index, label in enumerate(SIZE_ORDER)})
            order = [key for key in ["year", "era_start"] if key in table] + ["_size_order"]
            table = table.sort_values(order).drop(columns="_size_order")
        tables[name] = table
    acquisition_path = ROOT / "reports" / "acquisition_summary.json"
    acquisition = json.loads(acquisition_path.read_text()) if acquisition_path.exists() else {}
    total_records = acquisition.get("total_records")
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(path.resolve()),
        "source_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "year_range": {"min": int(frame.FIRE_YEAR.min()), "max": int(frame.FIRE_YEAR.max())},
        "cohort": cohort,
        "population_selection": {
            "source_records": total_records,
            "usable_containment_records": int(count),
            "usable_fraction_of_source": float(count / total_records) if total_records else None,
            "source_missing_containment_dates": acquisition.get("missing_containment_dates"),
        },
        "size_order": SIZE_ORDER,
        **{name: records(table) for name, table in tables.items()},
        "definitions": {
            "duration": "Containment calendar date minus discovery calendar date, consistently normalized to midnight; containment differs from fire-out.",
            "same_calendar_day": "Calendar duration equals zero: containment occurred on the discovery calendar date. This does not mean zero elapsed time.",
            "over_3_days": "Containment calendar date is more than three days after discovery calendar date. This is not verified initial-attack failure.",
            "quantiles": "Unweighted empirical calendar-duration quantiles using linear interpolation; aggregated quantiles may be fractional despite integral record-level durations.",
            "size_bins": {"Micro": "0 < acres < 1", "Small": "1 <= acres < 10", "Medium": "10 <= acres < 100", "Large": "100 <= acres < 1000", "1000+ acres": "acres >= 1000"},
            "five_year_eras": "Five-year intervals aligned to multiples of five. The 1990–1994 era contains only observed 1992–1994 records; observed_year_min/max identify available years.",
            "cause": "Source NWCG ignition-cause classification; missing or blank classifications are Unknown. Ignition cause does not establish suppression policy.",
        },
        "caveats": [
            "All values in this summary are observed, unadjusted descriptions of the usable containment cohort, not model estimates or causal effects.",
            "Containment-date availability and reporting-agency composition change by year; observed trends can reflect changing reporting and selection.",
            "Only continental-U.S. records with positive final acreage and valid 0–365-day containment intervals enter this cohort; it is not all U.S. wildfire records.",
            "Final acreage is an eventual fire outcome. Comparing size bins describes similarly sized completed fires but does not isolate suppression capability.",
            "The first five-year era has three observed years. Counts across eras should not be compared as annual incidence without accounting for exposure years.",
            "Overall quantiles combine very different fire sizes, causes, locations, weather, and agencies; apparent changes can result from changing composition.",
            "Calendar-day summaries intentionally avoid mixing records with and without reported clock times; they cannot resolve changes within the same day.",
        ],
        "validation": {
            "passed": True,
            "checked": ["finite, integral, nonnegative 0–365-day durations", "finite integral fire years", "known size labels", "all grouping counts partition the cohort", "same-day and >3-day counts match across groupings", "year/era totals and size totals agree", "ordered finite quantiles and valid fractions", "JSON-native finite metrics"],
            "year_size_groups": int(len(year_size)), "era_size_groups": int(len(era_size)),
        },
    }
    # A strict serialization check covers native types and rejects any NaN/Inf.
    json.dumps(result, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed_fires.parquet")
    args = parser.parse_args()
    summary = build_summary(args.input)
    output = ROOT / "reports" / "dashboard_summary.json"
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "complete", "output": str(output), "cohort": summary["cohort"], "validation": summary["validation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
