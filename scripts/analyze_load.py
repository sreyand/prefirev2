"""Retrospective load associations, with uncertainty clustered by discovery day.

This is a descriptive complete-case analysis. It does not estimate a causal
effect of national fire load, an initial-attack outcome, or resource capacity.
Run after prepare_features.py. Use --self-test for numerical synthetic checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.stats import t as student_t
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
INPUT_COLUMNS = [
    "DISCOVERY_DAY", "CONTAINMENT_DAYS_DATE_ONLY", "CONTAINMENT_DAYS",
    "DURATION_RESOLUTION", "FIRE_YEAR", "STATE", "NWCG_REPORTING_AGENCY",
    "NWCG_CAUSE_CLASSIFICATION", "LOG_ACRES", "LATITUDE", "LONGITUDE",
    "DAY_OF_YEAR", "OBSERVED_UNCONTAINED_FIRES", "vapour_pressure_deficit_max",
    "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM",
    "DISTANCE_TO_PLACE_CENTROID_KM", "CENTROID_DEM_ELEVATION_M",
]
NUMERIC_CONTROLS = [
    "LOG_ACRES", "LATITUDE", "LONGITUDE", "DAY_OF_YEAR",
    "OBSERVED_UNCONTAINED_FIRES", "vapour_pressure_deficit_max",
    "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM",
    "DISTANCE_TO_PLACE_CENTROID_KM", "CENTROID_DEM_ELEVATION_M",
]
CAVEATS = [
    "Retrospective conditional association, not a causal effect or proof of an operational capacity threshold.",
    "The cohort excludes missing or invalid containment dates and intervals outside 0-365 days; availability changes by year and reporting agency.",
    "Observed daily load counts only the usable containment cohort and includes the focal fire and other fires discovered later on the same day; it is not total U.S. active fires or measured dispatch demand.",
    "Final fire acreage and observed uncontained load may themselves be consequences of suppression; conditioning on them does not identify suppression efficiency.",
    "A calendar duration of zero means containment on the discovery calendar date, not zero elapsed time; primary analysis uses calendar resolution consistently.",
    "Clocks-only sensitivity uses a different selected cohort and wall-clock outcome with unspecified timezones; DST and local weather-day alignment cannot be corrected from these fields.",
    "Weather at 25 fire-weighted centroids is a broad regional proxy; centroid DEM elevation is neither fire-site elevation nor terrain slope.",
    "Distance to 2024 Census place centroids is static and does not measure historical WUI, road access, or observed suppression policy.",
    "Discovery-day clustered intervals account for dependence within each sampled day, but do not fully account for serial or spatial dependence across days.",
    "The model is additive in the selected controls; load contrasts rely on that specification and may have limited support within individual year, state, or agency strata.",
    "Percent contrasts refer to the conditional geometric mean of days plus one, not the arithmetic mean of days or percent change in containment days.",
]


def stratified_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    """Sample year x agency strata, recording exact inverse selection weights."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len(frame) <= maximum:
        result = frame.copy()
        result["SAMPLING_WEIGHT"] = 1.0
        return result
    groups = frame.groupby(["FIRE_YEAR", "AGENCY"], observed=True, sort=True).indices
    if len(groups) > maximum:
        raise ValueError("Sample cap must retain at least one row per year x agency stratum")
    counts = np.array([len(indices) for indices in groups.values()], dtype=int)
    remaining = maximum - len(counts)
    quotas = remaining * (counts - 1) / (len(frame) - len(counts))
    allocation = 1 + np.floor(quotas).astype(int)
    remainder = maximum - int(allocation.sum())
    if remainder:
        order = np.argsort(-(quotas - np.floor(quotas)), kind="stable")
        allocation[order[:remainder]] += 1
    rng = np.random.default_rng(seed)
    selected, weights = [], []
    for indices, count in zip(groups.values(), allocation, strict=True):
        chosen = rng.choice(indices, count, replace=False)
        selected.append(chosen)
        weights.append(np.full(count, len(indices) / count))
    result = frame.iloc[np.concatenate(selected)].copy()
    result["SAMPLING_WEIGHT"] = np.concatenate(weights)
    if len(result) != maximum or not np.isclose(result.SAMPLING_WEIGHT.sum(), len(frame)):
        raise AssertionError("Stratified sampling weights do not reconstruct the population")
    return result


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    frame["DISCOVERY_DAY"] = pd.to_datetime(frame.DISCOVERY_DAY, errors="coerce").dt.normalize()
    for name in NUMERIC_CONTROLS + ["CONTAINMENT_DAYS_DATE_ONLY", "CONTAINMENT_DAYS", "FIRE_YEAR"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame[name] = frame[name].replace([np.inf, -np.inf], np.nan)
    missing = frame[NUMERIC_CONTROLS + ["DISCOVERY_DAY", "FIRE_YEAR", "CONTAINMENT_DAYS_DATE_ONLY"]].isna()
    exclusions = {name: int(count) for name, count in missing.sum().items() if count}
    valid = ~missing.any(axis=1)
    valid &= frame.CONTAINMENT_DAYS_DATE_ONLY.between(0, 365)
    valid &= frame.OBSERVED_UNCONTAINED_FIRES.ge(1)
    valid &= frame.wind_speed_10m_max.ge(0) & frame.ANTECEDENT_PRECIP_30D_MM.ge(0)
    valid &= frame.DISTANCE_TO_PLACE_CENTROID_KM.ge(0) & frame.vapour_pressure_deficit_max.ge(0)
    prepared = frame.loc[valid].copy().reset_index(drop=True)
    if prepared.empty:
        raise ValueError("No valid analysis rows")
    if not np.equal(prepared.CONTAINMENT_DAYS_DATE_ONLY, np.floor(prepared.CONTAINMENT_DAYS_DATE_ONLY)).all():
        raise ValueError("Calendar-day outcome contains fractional values")
    if prepared.groupby("DISCOVERY_DAY").OBSERVED_UNCONTAINED_FIRES.nunique().max() != 1:
        raise ValueError("National daily load differs between fires on the same discovery day")
    prepared["FIRE_YEAR"] = prepared.FIRE_YEAR.astype(int)
    prepared["STATE"] = prepared.STATE.astype("string").str.strip().str.upper().fillna("Unknown").astype(str)
    prepared["AGENCY"] = prepared.NWCG_REPORTING_AGENCY.fillna("Unknown").astype(str)
    prepared["CAUSE"] = prepared.NWCG_CAUSE_CLASSIFICATION.fillna("Unknown").astype(str)
    prepared["LOAD_LOG1P"] = np.log1p(prepared.OBSERVED_UNCONTAINED_FIRES)
    prepared["WIND_LOG1P"] = np.log1p(prepared.wind_speed_10m_max)
    prepared["PRECIP_LOG1P"] = np.log1p(prepared.ANTECEDENT_PRECIP_30D_MM)
    prepared["PLACE_DISTANCE_LOG1P"] = np.log1p(prepared.DISTANCE_TO_PLACE_CENTROID_KM)
    prepared["CENTROID_ELEVATION_KM"] = prepared.CENTROID_DEM_ELEVATION_M / 1000
    prepared["SEASON_SIN"] = np.sin(2 * np.pi * prepared.DAY_OF_YEAR / 365.25)
    prepared["SEASON_COS"] = np.cos(2 * np.pi * prepared.DAY_OF_YEAR / 365.25)
    info = {
        "input_rows": int(len(frame)), "eligible_rows": int(len(prepared)),
        "excluded_rows": int((~valid).sum()), "missing_or_nonfinite_by_field": exclusions,
    }
    return prepared, info


def daily_load(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("DISCOVERY_DAY", sort=True).OBSERVED_UNCONTAINED_FIRES.first()


def model_formula(frame: pd.DataFrame) -> str:
    loads = np.log1p(daily_load(frame).to_numpy(dtype=float))
    if len(np.unique(loads)) < 10:
        raise ValueError("Insufficient distinct daily loads for a nonlinear association")
    lower, upper = float(loads.min()), float(loads.max())
    knots = np.unique(np.quantile(loads, [.25, .5, .75]))
    knots = knots[(knots > lower) & (knots < upper)].tolist()
    load = f"bs(LOAD_LOG1P, knots={knots!r}, degree=3, include_intercept=False, lower_bound={lower!r}, upper_bound={upper!r})"
    return (
        f"OUTCOME_LOG1P ~ {load} + bs(LOG_ACRES, df=4, degree=3)"
        " + C(FIRE_YEAR) + C(STATE) + C(AGENCY) + C(CAUSE)"
        " + LATITUDE + LONGITUDE + SEASON_SIN + SEASON_COS"
        " + vapour_pressure_deficit_max + WIND_LOG1P + PRECIP_LOG1P"
        " + PLACE_DISTANCE_LOG1P + CENTROID_ELEVATION_KM"
    )


def fit_analysis(frame: pd.DataFrame, outcome: str, name: str, maximum: int, seed: int) -> dict:
    population = frame.loc[frame[outcome].between(0, 365)].copy()
    sample = stratified_sample(population, maximum, seed)
    sample["OUTCOME_LOG1P"] = np.log1p(sample[outcome])
    clusters = sample.DISCOVERY_DAY.astype("int64").to_numpy()
    n_clusters = len(np.unique(clusters))
    if n_clusters < 50:
        raise ValueError(f"Only {n_clusters} discovery-day clusters; at least 50 required")
    formula = model_formula(population)
    fitted = smf.wls(formula, data=sample, weights=sample.SAMPLING_WEIGHT).fit(
        cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True}, use_t=True,
    )
    if int(fitted.nobs) != len(sample):
        raise AssertionError("Regression silently dropped sampled rows")
    if not np.isfinite(fitted.params).all() or not np.isfinite(fitted.cov_params()).all().all():
        raise AssertionError("Nonfinite regression coefficients or clustered covariance")
    if fitted.model.rank < fitted.model.exog.shape[1]:
        raise ValueError("Regression design is rank deficient; inspect covariate support")
    daily = daily_load(population)
    quintile_edges = daily.quantile(np.linspace(0, 1, 6)).to_numpy(dtype=float)
    percentiles = [10, 30, 50, 70, 90]
    values = daily.quantile(np.array(percentiles) / 100).to_numpy(dtype=float)
    reference = float(values[0])
    # Controls cancel in additive contrasts. One actual row holds all other
    # features fixed; no synthetic category or marginal retransformation is used.
    probes = pd.concat([sample.iloc[[0]].copy()] * (len(values) + 1), ignore_index=True)
    probes["LOAD_LOG1P"] = np.log1p(np.r_[reference, values])
    # statsmodels 0.15 stores Patsy's specification as model_spec; earlier
    # versions expose the same object as design_info.
    specification = getattr(fitted.model.data, "model_spec", None)
    if specification is None:
        specification = fitted.model.data.design_info
    matrix = np.asarray(build_design_matrices([specification], probes)[0])
    differences = matrix[1:] - matrix[0]
    coefficients = fitted.params.to_numpy()
    covariance = fitted.cov_params().to_numpy()
    critical = float(student_t.ppf(.975, n_clusters - 1))
    contrasts = []
    for index, (percentile, value, difference) in enumerate(zip(percentiles, values, differences, strict=True)):
        estimate = float(difference @ coefficients)
        variance = float(difference @ covariance @ difference)
        if variance < -1e-10:
            raise AssertionError("Negative contrast variance")
        error = float(np.sqrt(max(0.0, variance)))
        low, high = estimate - critical * error, estimate + critical * error
        left, right = float(quintile_edges[index]), float(quintile_edges[index + 1])
        if index == 4:
            supported_days = daily.between(left, right, inclusive="both")
            supported_fires = population.OBSERVED_UNCONTAINED_FIRES.between(left, right, inclusive="both")
        else:
            supported_days = daily.between(left, right, inclusive="left")
            supported_fires = population.OBSERVED_UNCONTAINED_FIRES.between(left, right, inclusive="left")
        contrasts.append({
            "analysis": name, "load_percentile": percentile, "load_band": f"Q{index + 1}",
            "observed_daily_load": float(value), "reference_load": reference,
            "support_load_min": left, "support_load_max": right,
            "support_days": int(supported_days.sum()), "support_fires": int(supported_fires.sum()),
            "adjusted_log1p_days_difference": estimate, "ci_low": low, "ci_high": high,
            "adjusted_days_plus_one_percent_change": float(100 * np.expm1(estimate)),
            "percent_change_ci_low": float(100 * np.expm1(low)),
            "percent_change_ci_high": float(100 * np.expm1(high)),
        })
    if not np.allclose([contrasts[0][key] for key in ["adjusted_log1p_days_difference", "ci_low", "ci_high"]], 0):
        raise AssertionError("Reference contrast must be identically zero")
    if sum(row["support_fires"] for row in contrasts) != len(population):
        raise AssertionError("Load support bands do not partition the analysis population")
    if sum(row["support_days"] for row in contrasts) != len(daily):
        raise AssertionError("Load support bands do not partition discovery days")
    return {
        "status": "complete", "outcome": outcome, "outcome_transform": "log1p(days)",
        "scale": "Adjusted difference in log(days + 1); transformed percentages concern geometric mean(days + 1)",
        "population_rows": int(len(population)), "sampled_rows": int(len(sample)),
        "day_clusters": int(n_clusters), "population_discovery_days": int(len(daily)),
        "year_min": int(population.FIRE_YEAR.min()), "year_max": int(population.FIRE_YEAR.max()),
        "load_reference": {"percentile": 10, "observed_daily_load": reference},
        "adjusted_contrasts": contrasts,
        "fit_summary": {
            "weighted_r_squared": float(fitted.rsquared), "parameters": int(len(fitted.params)),
            "design_rank": int(fitted.model.rank), "condition_number": float(fitted.condition_number),
            "ci_level": .95, "covariance": "CR1 clustered by discovery calendar day",
            "ci_degrees_of_freedom": int(n_clusters - 1),
            "unweighted_sample_mean_days": float(sample[outcome].mean()),
            "population_mean_days": float(population[outcome].mean()),
        },
        "formula": formula,
    }


def self_test() -> None:
    rng = np.random.default_rng(90421)
    days = pd.date_range("1992-01-01", periods=730)
    discovery = np.repeat(days.to_numpy(), 12)
    size = len(discovery)
    day_load = rng.integers(30, 1800, len(days))
    load = np.repeat(day_load, 12)
    log_acres = rng.normal(1.5, 1.0, size)
    vpd = rng.uniform(.1, 4, size)
    latent = .15 + .8 * np.log1p(load / 300) + .08 * log_acres + .05 * vpd
    latent += np.repeat(rng.normal(0, .08, len(days)), 12) + rng.normal(0, .08, size)
    duration = np.expm1(latent).clip(0, 365)
    frame = pd.DataFrame({
        "DISCOVERY_DAY": discovery, "CONTAINMENT_DAYS_DATE_ONLY": np.floor(duration),
        "CONTAINMENT_DAYS": duration, "DURATION_RESOLUTION": "wall_clock_minutes_timezone_unspecified",
        "FIRE_YEAR": pd.DatetimeIndex(discovery).year, "STATE": rng.choice(["CA", "OR", "ok"], size),
        "NWCG_REPORTING_AGENCY": rng.choice(["FS", "BLM", "ST/C&L"], size),
        "NWCG_CAUSE_CLASSIFICATION": rng.choice(["Human", "Natural", "Unknown"], size),
        "LOG_ACRES": log_acres, "LATITUDE": rng.uniform(30, 48, size),
        "LONGITUDE": rng.uniform(-124, -70, size), "DAY_OF_YEAR": pd.DatetimeIndex(discovery).dayofyear,
        "OBSERVED_UNCONTAINED_FIRES": load, "vapour_pressure_deficit_max": vpd,
        "wind_speed_10m_max": rng.uniform(0, 60, size), "ANTECEDENT_PRECIP_30D_MM": rng.uniform(0, 200, size),
        "DISTANCE_TO_PLACE_CENTROID_KM": rng.uniform(0, 100, size), "CENTROID_DEM_ELEVATION_M": rng.uniform(-20, 3000, size),
    })
    prepared, info = prepare(frame)
    assert info["excluded_rows"] == 0 and "ok" not in prepared.STATE.unique()
    first = stratified_sample(prepared, 3000, 58)
    second = stratified_sample(prepared, 3000, 58)
    assert first.index.equals(second.index)
    assert np.isclose(first.SAMPLING_WEIGHT.sum(), len(prepared))
    population_groups = prepared.groupby(["FIRE_YEAR", "AGENCY"]).size()
    recovered_groups = first.groupby(["FIRE_YEAR", "AGENCY"]).SAMPLING_WEIGHT.sum()
    assert np.allclose(population_groups, recovered_groups.reindex(population_groups.index))
    result = fit_analysis(prepared, "CONTAINMENT_DAYS_DATE_ONLY", "self_test", 3000, 58)
    last = result["adjusted_contrasts"][-1]
    assert last["ci_low"] > 0, "Synthetic positive load relationship was not recovered"
    assert result["day_clusters"] > 600
    assert all(row["ci_low"] <= row["adjusted_log1p_days_difference"] <= row["ci_high"] for row in result["adjusted_contrasts"])
    invalid = frame.copy()
    invalid.loc[0, "OBSERVED_UNCONTAINED_FIRES"] += 1
    try:
        prepare(invalid)
    except ValueError as error:
        assert "differs between fires" in str(error)
    else:
        raise AssertionError("Inconsistent daily load was accepted")
    print("PASS: deterministic weighted stratification; support partitions; clustered contrast recovery; exact zero reference; inconsistent-load rejection.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed_fires.parquet")
    parser.add_argument("--max-sample", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    raw = pd.read_parquet(args.input, columns=INPUT_COLUMNS)
    frame, eligibility = prepare(raw)
    print(f"Primary analysis: {len(frame):,} eligible rows; sample cap {args.max_sample:,}", flush=True)
    primary = fit_analysis(frame, "CONTAINMENT_DAYS_DATE_ONLY", "primary_calendar_days", args.max_sample, args.seed)
    timed = frame.loc[frame.DURATION_RESOLUTION.eq("wall_clock_minutes_timezone_unspecified")].copy()
    print(f"Clock-complete sensitivity: {len(timed):,} eligible rows", flush=True)
    sensitivity = fit_analysis(timed, "CONTAINMENT_DAYS", "clocks_only_sensitivity", args.max_sample, args.seed + 1)
    acquisition_path = ROOT / "reports" / "acquisition_summary.json"
    acquisition = json.loads(acquisition_path.read_text()) if acquisition_path.exists() else {}
    weather_proxy = {key: acquisition.get(key) for key in [
        "weather_centroids", "weather_distance_km_p50", "weather_distance_km_p95",
    ]}
    caveats = list(CAVEATS)
    if weather_proxy["weather_distance_km_p50"] is not None and weather_proxy["weather_distance_km_p95"] is not None:
        caveats.insert(1, "Current fire-to-weather-centroid assignment distances are "
                       f"median {weather_proxy['weather_distance_km_p50']:.0f} km and "
                       f"95th percentile {weather_proxy['weather_distance_km_p95']:.0f} km; "
                       "weather adjustment cannot resolve local fire-site conditions.")
    report = {
        "status": "complete", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input.resolve()), "eligibility": eligibility,
        "method": {
            "name": "Retrospective weighted least-squares association with nonlinear observed load",
            "sample": "Deterministic proportional year x reporting-agency stratification; inverse sampling weights; at least one row per stratum",
            "max_sample_per_analysis": args.max_sample, "seed": args.seed,
            "load_contrasts": "P10/P30/P50/P70/P90 of daily load across distinct observed discovery days, relative to P10; support bands are day-load quintiles",
            "controls": ["Final acreage spline", "Year fixed effects", "State fixed effects", "Reporting-agency fixed effects", "Ignition-cause fixed effects", "Latitude and longitude", "Seasonal sine and cosine", "VPD", "log1p wind", "log1p antecedent rainfall", "log1p place-centroid distance", "Centroid DEM elevation"],
        },
        "primary": primary, "clocks_only_sensitivity": sensitivity,
        "weather_proxy": weather_proxy,
        "coverage_reports": ["reports/fire_coverage_by_year_agency.csv", "reports/duration_resolution_by_year_agency.csv"],
        "caveats": caveats,
    }
    directory = ROOT / "reports"
    directory.mkdir(exist_ok=True)
    output = directory / "load_analysis.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    contrasts = primary["adjusted_contrasts"] + sensitivity["adjusted_contrasts"]
    pd.DataFrame(contrasts).to_csv(directory / "load_analysis.csv", index=False)
    print(json.dumps({"status": "complete", "primary_sample": primary["sampled_rows"], "clock_sample": sensitivity["sampled_rows"], "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
