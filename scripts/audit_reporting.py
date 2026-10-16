"""Audit reporting selection and observed containment trends without model changes.

Only raw-source columns needed for eligibility/coverage and a small set of cohort
columns are read. Original source values are never modified on disk.
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
PERIODS = {"early_1992_2005": (1992, 2005), "recent_2015_2024": (2015, 2024)}
FINE_EDGES = [0, .01, .1, .25, .5, 1, 2, 5, 10, 20, 50, 100,
              200, 500, 1000, 2000, 5000, 10000, 25000, 50000,
              100000, 250000, np.inf]


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=12))


def clock_minutes(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(4)
    hours = pd.to_numeric(text.str[:2], errors="coerce")
    minutes = pd.to_numeric(text.str[2:], errors="coerce")
    valid = text.str.fullmatch(r"\d{4}", na=False)
    valid &= (hours.between(0, 23) & minutes.between(0, 59)) | (hours.eq(24) & minutes.eq(0))
    return (hours * 60 + minutes).where(valid)


def weighted_quantiles(values, weights, probabilities=(.5, .9)) -> list[float]:
    """Inverse weighted empirical CDF, preserving calendar-day observations."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weighted quantiles require finite values and positive weights")
    order = np.argsort(values, kind="stable")
    ordered, cumulative = values[order], np.cumsum(weights[order])
    indices = np.searchsorted(cumulative, np.asarray(probabilities) * cumulative[-1], side="left")
    return [float(value) for value in ordered[np.minimum(indices, len(ordered) - 1)]]


def coverage_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = frame.groupby(keys, observed=True, sort=True, dropna=False).agg(
        source_records=("FOD_ID", "size"),
        geography_and_size_eligible_records=("PHYSICAL_ELIGIBLE", "sum"),
        containment_dates_count=("HAS_CONTAINMENT_DATE", "sum"),
        eligible_with_containment_date_count=("ELIGIBLE_CONTAINMENT_DATE", "sum"),
        usable_cohort_count=("IN_COHORT", "sum"),
        cohort_clock_complete_count=("COHORT_CLOCK_COMPLETE", "sum"),
        source_discovery_clock_available_count=("DISCOVERY_CLOCK_VALID", "sum"),
        source_containment_clock_available_count=("CONTAINMENT_CLOCK_VALID", "sum"),
        source_both_clocks_available_count=("BOTH_CLOCKS_VALID", "sum"),
    )
    result["containment_date_fraction_source"] = result.containment_dates_count / result.source_records
    denominator = result.geography_and_size_eligible_records.replace(0, np.nan)
    result["eligible_containment_date_fraction"] = result.eligible_with_containment_date_count / denominator
    result["usable_cohort_fraction_eligible"] = result.usable_cohort_count / denominator
    result["cohort_clock_complete_fraction"] = result.cohort_clock_complete_count / result.usable_cohort_count.replace(0, np.nan)
    result["cohort_calendar_only_count"] = result.usable_cohort_count - result.cohort_clock_complete_count
    return result.reset_index().rename(columns={"FIRE_YEAR": "year", "AGENCY": "agency"})


def cohort_statistics(group: pd.DataFrame) -> dict:
    quantiles = group.CALENDAR_DAYS.quantile([.5, .9]).to_numpy()
    timed = group.loc[group.CLOCK_COMPLETE]
    timed_quantiles = timed.WALL_CLOCK_DAYS.quantile([.5, .9]).to_numpy() if len(timed) else [None, None]
    return {
        "count": int(len(group)), "p50_days": float(quantiles[0]), "p90_days": float(quantiles[1]),
        "same_calendar_day_fraction": float(group.CALENDAR_DAYS.eq(0).mean()),
        "over_3_days_fraction": float(group.CALENDAR_DAYS.gt(3).mean()),
        "median_acres": float(group.FIRE_SIZE.median()),
        "p90_acres": float(group.FIRE_SIZE.quantile(.9)),
        "clock_complete_count": int(len(timed)), "clock_complete_fraction": float(len(timed) / len(group)),
        "clock_complete_p50_calendar_days": float(timed.CALENDAR_DAYS.median()) if len(timed) else None,
        "clock_complete_p90_calendar_days": float(timed.CALENDAR_DAYS.quantile(.9)) if len(timed) else None,
        "clock_complete_p50_wall_days": float(timed_quantiles[0]) if len(timed) else None,
        "clock_complete_p90_wall_days": float(timed_quantiles[1]) if len(timed) else None,
    }


def period_comparisons(cohort: pd.DataFrame, qualifying_agencies: list[str]) -> tuple[list[dict], list[dict]]:
    rows, deltas = [], []
    scopes = {"overall": None, "FS_only": "FS"}
    scopes.update({f"agency_{agency}_only": agency for agency in qualifying_agencies if agency != "FS"})
    for scope, agency in scopes.items():
        scoped = cohort if agency is None else cohort.loc[cohort.AGENCY.eq(agency)]
        for period, (start, end) in PERIODS.items():
            selected = scoped.loc[scoped.FIRE_YEAR.between(start, end)]
            period_rows = []
            for size in SIZE_ORDER:
                group = selected.loc[selected.SIZE_BIN.eq(size)]
                if group.empty:
                    continue
                row = {"scope": scope, "period": period, "year_min": start, "year_max": end,
                       "size_bin": size, **cohort_statistics(group)}
                period_rows.append(row)
            if sum(row["count"] for row in period_rows) != len(selected):
                raise AssertionError("Period-by-size summaries do not partition their cohort")
            rows.extend(period_rows)
        for size in SIZE_ORDER:
            selected = {row["period"]: row for row in rows if row["scope"] == scope and row["size_bin"] == size}
            early, recent = selected["early_1992_2005"], selected["recent_2015_2024"]
            deltas.append({
                "scope": scope, "size_bin": size, "early_count": early["count"], "recent_count": recent["count"],
                "early_p50_days": early["p50_days"], "recent_p50_days": recent["p50_days"],
                "p50_days_change": recent["p50_days"] - early["p50_days"],
                "early_p90_days": early["p90_days"], "recent_p90_days": recent["p90_days"],
                "p90_days_change": recent["p90_days"] - early["p90_days"],
                "same_calendar_day_percentage_point_change": 100 * (recent["same_calendar_day_fraction"] - early["same_calendar_day_fraction"]),
                "over_3_days_percentage_point_change": 100 * (recent["over_3_days_fraction"] - early["over_3_days_fraction"]),
            })
    return rows, deltas


def standardize(cohort: pd.DataFrame, minimum: int) -> tuple[list[dict], list[dict]]:
    """Hold early finer-acreage/agency composition fixed on shared support."""
    rows, deltas = [], []
    for scope in ["overall", "FS_only"]:
        scoped = cohort if scope == "overall" else cohort.loc[cohort.AGENCY.eq("FS")]
        for size in SIZE_ORDER:
            group = scoped.loc[scoped.SIZE_BIN.eq(size)]
            early = group.loc[group.FIRE_YEAR.between(*PERIODS["early_1992_2005"])]
            recent = group.loc[group.FIRE_YEAR.between(*PERIODS["recent_2015_2024"])]
            keys = ["FINE_SIZE", "AGENCY"] if scope == "overall" else ["FINE_SIZE"]
            early_counts = early.groupby(keys, observed=True).size().rename("early_count")
            recent_counts = recent.groupby(keys, observed=True).size().rename("recent_count")
            support = pd.concat([early_counts, recent_counts], axis=1).fillna(0)
            support = support.loc[support.early_count.ge(minimum) & support.recent_count.ge(minimum)].reset_index()
            if support.empty:
                raise ValueError(f"No shared strata for {scope}/{size}")
            for period, population, count_column in [
                ("early_1992_2005", early, "early_count"),
                ("recent_2015_2024", recent, "recent_count"),
            ]:
                retained = population.merge(support, on=keys, how="inner", validate="many_to_one")
                weights = retained.early_count / retained[count_column]
                if not np.isclose(weights.sum(), support.early_count.sum()):
                    raise AssertionError("Standardization weights do not recover early reference mass")
                p50, p90 = weighted_quantiles(retained.CALENDAR_DAYS, weights)
                raw = retained.CALENDAR_DAYS.quantile([.5, .9]).to_numpy()
                rows.append({
                    "scope": scope, "period": period, "size_bin": size,
                    "stratification": "finer_acreage_x_agency" if scope == "overall" else "finer_acreage",
                    "minimum_records_per_period_stratum": int(minimum), "common_strata": int(len(support)),
                    "period_total_count": int(len(population)), "retained_count": int(len(retained)),
                    "retained_fraction": float(len(retained) / len(population)),
                    "weighted_p50_days": p50, "weighted_p90_days": p90,
                    "raw_retained_p50_days": float(raw[0]), "raw_retained_p90_days": float(raw[1]),
                    "weighted_same_calendar_day_fraction": float(np.average(retained.CALENDAR_DAYS.eq(0), weights=weights)),
                    "weighted_over_3_days_fraction": float(np.average(retained.CALENDAR_DAYS.gt(3), weights=weights)),
                })
            a, b = rows[-2:]
            deltas.append({
                "scope": scope, "size_bin": size,
                "early_weighted_p50_days": a["weighted_p50_days"], "recent_weighted_p50_days": b["weighted_p50_days"],
                "weighted_p50_days_change": b["weighted_p50_days"] - a["weighted_p50_days"],
                "early_weighted_p90_days": a["weighted_p90_days"], "recent_weighted_p90_days": b["weighted_p90_days"],
                "weighted_p90_days_change": b["weighted_p90_days"] - a["weighted_p90_days"],
                "early_retained_fraction": a["retained_fraction"], "recent_retained_fraction": b["retained_fraction"],
                "weighted_over_3_days_percentage_point_change": 100 * (b["weighted_over_3_days_fraction"] - a["weighted_over_3_days_fraction"]),
            })
    return rows, deltas


def agency_stability(annual: pd.DataFrame, raw: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    """Screen every agency against a declared, non-causal coverage criterion."""
    target_years = list(range(1992, 2006)) + list(range(2015, 2025))
    rows, period_rows = [], []
    for agency in sorted(raw.AGENCY.unique()):
        observed = annual.loc[annual.agency.eq(agency)].set_index("year")
        target = observed.reindex(target_years)
        enough = target.geography_and_size_eligible_records.fillna(0).ge(50)
        covered = target.eligible_containment_date_fraction.ge(.90)
        enough_source = target.source_records.fillna(0).ge(50)
        covered_source = target.containment_date_fraction_source.ge(.90)
        row = {
            "agency": str(agency), "required_years": len(target_years),
            "observed_target_years": int(target.source_records.notna().sum()),
            "years_with_at_least_50_eligible_records": int(enough.sum()),
            "years_at_or_above_90pct_eligible_coverage": int(covered.sum()),
            "minimum_annual_eligible_records": int(target.geography_and_size_eligible_records.fillna(0).min()),
            "minimum_annual_source_records": int(target.source_records.fillna(0).min()),
            "minimum_observed_annual_eligible_coverage": float(target.eligible_containment_date_fraction.min()) if target.eligible_containment_date_fraction.notna().any() else None,
            "minimum_observed_annual_source_coverage": float(target.containment_date_fraction_source.min()) if target.containment_date_fraction_source.notna().any() else None,
            "qualifies_as_high_coverage_agency": bool((enough & covered).all()),
            "qualifies_using_all_source_denominator": bool((enough_source & covered_source).all()),
            "coverage_failure_years": [int(year) for year in target.index[~covered]],
            "volume_failure_years": [int(year) for year in target.index[~enough]],
        }
        rows.append(row)
        for period, (start, end) in PERIODS.items():
            selected = raw.loc[raw.AGENCY.eq(agency) & raw.FIRE_YEAR.between(start, end)]
            if selected.empty:
                period_rows.append({"agency": str(agency), "period": period, "source_records": 0,
                                    "geography_and_size_eligible_records": 0, "usable_cohort_count": 0,
                                    "minimum_annual_eligible_records": 0,
                                    "minimum_observed_annual_eligible_coverage": None,
                                    "minimum_observed_annual_source_coverage": None})
                continue
            result = json_records(coverage_table(selected, ["AGENCY"]))[0]
            years = target.loc[list(range(start, end + 1))]
            result.update({
                "period": period,
                "minimum_annual_eligible_records": int(years.geography_and_size_eligible_records.fillna(0).min()),
                "minimum_observed_annual_eligible_coverage": float(years.eligible_containment_date_fraction.min()) if years.eligible_containment_date_fraction.notna().any() else None,
                "minimum_observed_annual_source_coverage": float(years.containment_date_fraction_source.min()) if years.containment_date_fraction_source.notna().any() else None,
            })
            period_rows.append(result)
    qualified = [row["agency"] for row in rows if row["qualifies_as_high_coverage_agency"]]
    screen = {
        "criteria": "At least 50 CONUS positive-acreage source records and >=90% nonmissing containment dates in EACH of all 24 early/recent target years; absent years fail.",
        "target_years": target_years, "qualifying_agencies": qualified,
        "conclusion": "No reporting agency satisfies the specified annual high-coverage screen." if not qualified else "Qualifying agencies are higher-coverage descriptive sensitivities, not clean causal controls.",
        "threshold_caveat": "These screening thresholds are diagnostic choices, not a guarantee of reporting stability or a causal identification rule; values just below 90% should not be interpreted as a qualitative discontinuity.",
    }
    return rows, period_rows, screen


def window_sensitivity(cohort: pd.DataFrame, standardized_deltas: list[dict]) -> list[dict]:
    rows = []
    for size in SIZE_ORDER:
        group = cohort.loc[cohort.SIZE_BIN.eq(size)]
        windows = {}
        for name, bounds in {
            "short_early_1992_1994": (1992, 1994), "short_recent_2020_2024": (2020, 2024),
            "broad_early_1992_2005": (1992, 2005), "broad_recent_2015_2024": (2015, 2024),
        }.items():
            selected = group.loc[group.FIRE_YEAR.between(*bounds)]
            windows[name] = cohort_statistics(selected)
        short_change = windows["short_recent_2020_2024"]["p90_days"] - windows["short_early_1992_1994"]["p90_days"]
        broad_change = windows["broad_recent_2015_2024"]["p90_days"] - windows["broad_early_1992_2005"]["p90_days"]
        standardized = next(row for row in standardized_deltas if row["scope"] == "overall" and row["size_bin"] == size)
        rows.append({"size_bin": size, "windows": windows,
                     "short_window_p90_change_days": float(short_change),
                     "broad_window_p90_change_days": float(broad_change),
                     "broad_minus_short_p90_change_days": float(broad_change - short_change),
                     "standardized_broad_window_p90_change_days": standardized["weighted_p90_days_change"]})
    return rows


def run_audit(minimum: int) -> tuple[dict, list[dict]]:
    source_path, cohort_path = ROOT / "data" / "fpa_fod_all.parquet", ROOT / "data" / "processed_fires.parquet"
    raw = pd.read_parquet(source_path, columns=["FOD_ID", "FIRE_YEAR", "NWCG_REPORTING_AGENCY", "DISCOVERY_DATE", "CONT_DATE", "DISCOVERY_TIME", "CONT_TIME", "FIRE_SIZE", "LATITUDE", "LONGITUDE"])
    cohort = pd.read_parquet(cohort_path, columns=["FOD_ID", "FIRE_YEAR", "NWCG_REPORTING_AGENCY", "FIRE_SIZE", "SIZE_BIN", "CONTAINMENT_DAYS_DATE_ONLY", "CONTAINMENT_DAYS", "DURATION_RESOLUTION"])
    if raw.FOD_ID.duplicated().any() or cohort.FOD_ID.duplicated().any():
        raise AssertionError("Source and cohort fire identifiers must be unique")
    discovery = pd.to_datetime(raw.DISCOVERY_DATE, errors="coerce")
    containment = pd.to_datetime(raw.CONT_DATE, errors="coerce")
    days = (containment.dt.normalize() - discovery.dt.normalize()).dt.days
    discovery_clock, containment_clock = clock_minutes(raw.DISCOVERY_TIME), clock_minutes(raw.CONT_TIME)
    raw["PHYSICAL_ELIGIBLE"] = raw.LATITUDE.between(24, 50) & raw.LONGITUDE.between(-125, -66) & raw.FIRE_SIZE.gt(0)
    raw["HAS_CONTAINMENT_DATE"] = containment.notna()
    raw["ELIGIBLE_CONTAINMENT_DATE"] = raw.PHYSICAL_ELIGIBLE & raw.HAS_CONTAINMENT_DATE
    raw["DISCOVERY_CLOCK_VALID"] = discovery_clock.notna()
    raw["CONTAINMENT_CLOCK_VALID"] = containment_clock.notna()
    raw["BOTH_CLOCKS_VALID"] = raw.DISCOVERY_CLOCK_VALID & raw.CONTAINMENT_CLOCK_VALID
    precise = days + (containment_clock - discovery_clock) / 1440
    invalid_timed = raw.BOTH_CLOCKS_VALID & ~precise.between(0, 365)
    raw["IN_COHORT"] = raw.PHYSICAL_ELIGIBLE & days.between(0, 365) & ~invalid_timed
    raw["COHORT_CLOCK_COMPLETE"] = raw.IN_COHORT & raw.BOTH_CLOCKS_VALID
    raw["AGENCY"] = raw.NWCG_REPORTING_AGENCY.astype("string").str.strip().fillna("Unknown")
    expected = raw.loc[raw.IN_COHORT, ["FOD_ID"]].assign(EXPECTED_CALENDAR_DAYS=days.loc[raw.IN_COHORT])
    comparison = cohort.merge(expected, on="FOD_ID", how="outer", validate="one_to_one", indicator=True)
    if not comparison._merge.eq("both").all():
        raise AssertionError("Reconstructed raw eligibility differs from saved cohort IDs")
    if not np.equal(comparison.CONTAINMENT_DAYS_DATE_ONLY, comparison.EXPECTED_CALENDAR_DAYS).all():
        raise AssertionError("Saved calendar durations differ from normalized source dates")
    clock_ids = pd.Index(raw.loc[raw.COHORT_CLOCK_COMPLETE, "FOD_ID"])
    expected_clock_complete = cohort.FOD_ID.isin(clock_ids)
    if not expected_clock_complete.equals(cohort.DURATION_RESOLUTION.eq("wall_clock_minutes_timezone_unspecified")):
        raise AssertionError("Saved clock-completeness flags differ from source clocks")
    source_wall = pd.Series(precise.loc[raw.COHORT_CLOCK_COMPLETE].to_numpy(), index=raw.loc[raw.COHORT_CLOCK_COMPLETE, "FOD_ID"])
    if not np.allclose(cohort.loc[expected_clock_complete, "CONTAINMENT_DAYS"], cohort.loc[expected_clock_complete, "FOD_ID"].map(source_wall)):
        raise AssertionError("Saved wall-clock durations differ from source dates and clocks")
    statuses = np.select(
        [discovery.isna() & containment.isna(), discovery.isna() & containment.notna(),
         discovery.notna() & containment.isna(), days.lt(0), days.gt(365)],
        ["missing_both_dates", "missing_discovery_only", "missing_containment_only", "negative_calendar_interval", "over_365_calendar_days"],
        default="valid_0_365_calendar_days",
    )
    status_rows = []
    for status in ["missing_both_dates", "missing_discovery_only", "missing_containment_only", "negative_calendar_interval", "over_365_calendar_days", "valid_0_365_calendar_days"]:
        mask = statuses == status
        status_rows.append({"date_status": status, "source_records": int(mask.sum()),
                            "geography_and_size_eligible_records": int((mask & raw.PHYSICAL_ELIGIBLE).sum())})
    if sum(row["source_records"] for row in status_rows) != len(raw):
        raise AssertionError("Date-quality statuses do not partition the source")
    by_year, by_agency, by_year_agency = coverage_table(raw, ["FIRE_YEAR"]), coverage_table(raw, ["AGENCY"]), coverage_table(raw, ["FIRE_YEAR", "AGENCY"])
    for table in [by_year, by_agency, by_year_agency]:
        if int(table.source_records.sum()) != len(raw) or int(table.usable_cohort_count.sum()) != len(cohort):
            raise AssertionError("Coverage aggregation fails to recover source/cohort totals")
    stability_rows, agency_period_rows, stability_screen = agency_stability(by_year_agency, raw)
    period_coverage = []
    for scope in ["overall", "FS_only"]:
        scoped = raw if scope == "overall" else raw.loc[raw.AGENCY.eq("FS")]
        for period, (start, end) in PERIODS.items():
            selected = scoped.loc[scoped.FIRE_YEAR.between(start, end)].copy()
            selected["SCOPE"] = scope
            row = json_records(coverage_table(selected, ["SCOPE"]))[0]
            row.update(scope=scope, period=period)
            row.pop("SCOPE")
            period_coverage.append(row)
    cohort["AGENCY"] = cohort.NWCG_REPORTING_AGENCY.astype("string").str.strip().fillna("Unknown")
    cohort = cohort.rename(columns={"CONTAINMENT_DAYS_DATE_ONLY": "CALENDAR_DAYS", "CONTAINMENT_DAYS": "WALL_CLOCK_DAYS"})
    cohort["CLOCK_COMPLETE"] = expected_clock_complete
    cohort["SIZE_BIN"] = cohort.SIZE_BIN.astype("string")
    expected_bins = pd.cut(cohort.FIRE_SIZE, [0, 1, 10, 100, 1000, np.inf], right=False, labels=SIZE_ORDER).astype("string")
    if not cohort.SIZE_BIN.equals(expected_bins):
        raise AssertionError("Saved broad acreage bins differ from source acreage")
    cohort["FINE_SIZE"] = pd.cut(cohort.FIRE_SIZE, FINE_EDGES, right=False, labels=False)
    if cohort.FINE_SIZE.isna().any() or cohort.CALENDAR_DAYS.lt(0).any():
        raise AssertionError("Invalid finer acreage assignment or negative outcome")
    comparisons, deltas = period_comparisons(cohort, stability_screen["qualifying_agencies"])
    standardized, standardized_deltas = standardize(cohort, minimum)
    windows = window_sensitivity(cohort, standardized_deltas)
    # Independent numerical check of weighted-CDF implementation.
    values, weights = [0, 2, 9], [2, 3, 1]
    if not np.array_equal(weighted_quantiles(values, weights), np.quantile(np.repeat(values, weights), [.5, .9], method="inverted_cdf")):
        raise AssertionError("Weighted empirical quantile validation failed")
    nonmidnight_discovery = discovery.notna() & discovery.ne(discovery.dt.normalize())
    nonmidnight_containment = containment.notna() & containment.ne(containment.dt.normalize())
    primary_partition = [
        {"status": "geography_or_coordinate_out_of_scope", "count": int((~raw.PHYSICAL_ELIGIBLE).sum())},
        {"status": "eligible_missing_either_date", "count": int((raw.PHYSICAL_ELIGIBLE & days.isna()).sum())},
        {"status": "eligible_negative_calendar_interval", "count": int((raw.PHYSICAL_ELIGIBLE & days.lt(0)).sum())},
        {"status": "eligible_over_365_calendar_days", "count": int((raw.PHYSICAL_ELIGIBLE & days.gt(365)).sum())},
        {"status": "eligible_invalid_clock_interval", "count": int((raw.PHYSICAL_ELIGIBLE & days.between(0, 365) & invalid_timed).sum())},
        {"status": "included_in_cohort", "count": int(raw.IN_COHORT.sum())},
    ]
    if sum(row["count"] for row in primary_partition) != len(raw):
        raise AssertionError("Primary exclusion statuses do not partition the source")
    quality = {
        "date_status": status_rows,
        "primary_exclusion_partition": primary_partition,
        "geography_or_coordinate_ineligible_records": int((~(raw.LATITUDE.between(24, 50) & raw.LONGITUDE.between(-125, -66))).sum()),
        "nonpositive_or_missing_acreage_records": int((~raw.FIRE_SIZE.gt(0)).sum()),
        "geography_and_size_ineligible_records": int((~raw.PHYSICAL_ELIGIBLE).sum()),
        "total_excluded_from_cohort": int(len(raw) - len(cohort)),
        "invalid_timed_intervals_among_date_and_physical_eligible": int((raw.PHYSICAL_ELIGIBLE & days.between(0, 365) & invalid_timed).sum()),
        "source_nonmidnight_discovery_timestamps": int(nonmidnight_discovery.sum()),
        "source_nonmidnight_containment_timestamps": int(nonmidnight_containment.sum()),
        "cohort_nonmidnight_discovery_timestamps": int((nonmidnight_discovery & raw.IN_COHORT).sum()),
        "source_calendar_date_and_reported_fire_year_disagree": int((discovery.notna() & discovery.dt.year.ne(raw.FIRE_YEAR)).sum()),
        "source_nonmissing_invalid_discovery_clocks": int((raw.DISCOVERY_TIME.notna() & ~raw.DISCOVERY_CLOCK_VALID).sum()),
        "source_nonmissing_invalid_containment_clocks": int((raw.CONT_TIME.notna() & ~raw.CONTAINMENT_CLOCK_VALID).sum()),
    }
    report = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path), "cohort_source": str(cohort_path),
        "source_records": int(len(raw)), "cohort_records": int(len(cohort)),
        "periods": {name: {"year_min": bounds[0], "year_max": bounds[1]} for name, bounds in PERIODS.items()},
        "size_order": SIZE_ORDER, "date_quality": quality,
        "coverage_by_year": json_records(by_year), "coverage_by_agency": json_records(by_agency),
        "coverage_by_year_agency": json_records(by_year_agency), "coverage_by_period": period_coverage,
        "coverage_by_agency_period": agency_period_rows,
        "agency_stability": stability_rows, "stable_agency_screen": stability_screen,
        "period_comparisons": comparisons, "period_comparison_deltas": deltas,
        "standardized_comparisons": standardized, "standardized_comparison_deltas": standardized_deltas,
        "headline_window_sensitivity": windows,
        "definitions": {
            "coverage_denominators": "containment_date_fraction_source uses all source records; eligible fractions use CONUS coordinates and positive acreage; clock-complete fraction uses the usable containment cohort. Undefined zero-denominator fractions are null.",
            "calendar_outcome": "Normalized containment date minus normalized discovery date; source timestamps remain untouched. Containment is not fire-out.",
            "FS_only": "Records whose NWCG reporting agency is FS; this is not all fires on Forest Service land and does not hold reporting units, geography, weather, policy, or fire mix fixed.",
            "clock_complete": "Both source HHMM clocks are valid, including 2400 as end-of-day midnight, and the resulting interval remains within 0–365 days. Source clocks have unspecified timezones.",
            "raw_quantiles": "Unweighted empirical calendar-day quantiles using linear interpolation; wall-clock sensitivity quantiles use only clock-complete records.",
            "standardization": "Within each broad size bin and scope, retain finer-acreage/agency strata with at least the configured count in each period. Reweight recent records to early-period common-support stratum proportions; early records have unit weights. Overall stratifies by finer acreage and agency; FS-only by finer acreage.",
            "standardized_quantiles": "Inverse weighted empirical CDF (step quantiles), not interpolation. Interpret standardized and raw quantile differences with this small estimator distinction in mind.",
            "finer_acreage_edges": [float(value) if np.isfinite(value) else "infinity" for value in FINE_EDGES],
            "minimum_records_per_period_stratum": int(minimum),
        },
        "caveats": [
            "All results are observed descriptive comparisons or composition-standardized descriptive sensitivities, not causal estimates of suppression effectiveness.",
            "Aggregate Forest Service containment coverage masks a major recent decline; FS-only is another biased descriptive sensitivity, not a clean or stable reporting control.",
            "Missing containment outcomes need not resemble reported outcomes; neither complete-case analysis nor composition weighting identifies their unobserved durations.",
            "Early and recent periods span different numbers of years. Raw record counts are support sizes, not comparable annual incidence rates.",
            "Final acreage is a fire outcome, not a pre-incident severity measure; broad and finer size matching cannot isolate suppression capability.",
            "Finer-acreage/agency standardization addresses only those observed mixture differences on shared support; weather, geography, resource use, management policy, and reporting changes remain uncontrolled.",
            "Subday clock availability changes by year and agency. The primary comparison consistently uses calendar days; clocks-only sensitivities select a different population.",
            "Same-calendar-day containment is not instantaneous containment; calendar quantiles cannot resolve within-day improvements or delays.",
            "These audit summaries have no causal confidence intervals; they describe the available records and their measured composition.",
            "Changing the early/recent window endpoints changes the observed tail-duration contrast; neither a short nor broad window identifies a causal suppression effect.",
        ],
        "validation": {"passed": True, "checks": ["unique source and cohort IDs", "exact reconstructed eligibility versus cohort IDs", "calendar outcomes equal normalized source date subtraction", "clock flags and wall-clock outcomes match source clocks", "date-status partition", "coverage tables recover source and cohort totals", "period size partitions", "broad/finer acreage-bin consistency", "standardization mass and common support", "weighted quantiles agree with replicated empirical quantiles", "strict finite JSON serialization"]},
    }
    json.dumps(report, allow_nan=False)
    csv_rows = []
    for key in ["coverage_by_year", "coverage_by_agency", "coverage_by_year_agency", "coverage_by_period", "period_comparisons", "period_comparison_deltas", "standardized_comparisons", "standardized_comparison_deltas"]:
        csv_rows.extend({"row_type": key, **row} for row in report[key])
    csv_rows.extend({"row_type": "coverage_by_agency_period", **row} for row in agency_period_rows)
    csv_rows.extend({"row_type": "agency_stability", **{key: value for key, value in row.items() if not isinstance(value, list)}} for row in stability_rows)
    csv_rows.extend({"row_type": "headline_window_sensitivity", **{key: value for key, value in row.items() if key != "windows"}} for row in windows)
    csv_rows.extend({"row_type": "date_quality_status", **row} for row in status_rows)
    return report, csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-stratum", type=int, default=20)
    args = parser.parse_args()
    if args.minimum_stratum < 1:
        parser.error("minimum-stratum must be positive")
    report, rows = run_audit(args.minimum_stratum)
    output = ROOT / "reports" / "reporting_audit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps({"status": "complete", "source_records": report["source_records"], "cohort_records": report["cohort_records"], "date_quality": report["date_quality"], "validation": report["validation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
