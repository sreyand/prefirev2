"""Retrospective prolonged-containment risk and conditional duration models.

Run from the project root: .venv/Scripts/python scripts/train_risk.py
Importable API: load_bundle(path), predict_batch(bundle, frame),
predict_scenario(bundle, row_dict). Bundle objects use only built-in sklearn
classes, so joblib artifacts do not depend on a pickled ``__main__`` class.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_pinball_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FEATURES = [
    "LOG_ACRES", "LATITUDE", "LONGITUDE", "DAY_OF_YEAR",
    "OBSERVED_UNCONTAINED_FIRES", "vapour_pressure_deficit_max",
    "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM",
    "DISTANCE_TO_PLACE_CENTROID_KM",
]
CATEGORICAL_FEATURES = ["STATE", "NWCG_REPORTING_AGENCY", "NWCG_CAUSE_CLASSIFICATION"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "CONTAINMENT_DAYS_DATE_ONLY"
QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_TIE_BREAK_AMPLITUDE = 1e-6
SCENARIO_THRESHOLD_TOLERANCE_DAYS = 1e-4
QUANTILE_FIX_EVIDENCE = {
    "reason": "At a tied zero initialization, native pinball boosting stalled at zero despite a supported nonzero conditional median.",
    "synthetic_rows": 5000, "synthetic_zero_fraction": 0.8,
    "supported_positive_subgroup_rows": 1000, "min_samples_leaf": 80,
    "true_conditional_medians_days": [0.0, 10.0],
    "original_predicted_medians_days": [0.0, 0.0],
    "corrected_predicted_medians_days": [0.0, 9.998978996758222],
    "correction": "Fixed-seed uniform numerical tie breaking of ±1e-6 added only to log1p-day fitting targets; original calendar-day outcomes remain unchanged for all evaluation.",
    "model_settings_unchanged": True,
}
SPLITS = {
    "training": (1992, 2014),
    "calibration": (2015, 2017),
    "validation": (2018, 2019),
    "test": (2020, 2024),
}
LIMITATIONS = [
    "Retrospective conditional associations, not causal effects or a dispatch-time forecast.",
    "Final acreage and discovery-day maximum weather are unavailable at ignition; final acreage may itself reflect suppression outcomes.",
    "Calendar-date discovery-to-containment days are the outcome. More than three calendar days is not an exact 72-hour threshold, fire-out time, or verified initial-attack failure.",
    "The selected cohort requires valid reported containment dates and 0–365 calendar-date days. Missing-outcome fires, excluded longer events, and reporting differences can bias results.",
    "Observed concurrency counts only the outcome-complete cohort, includes the focal fire, and is reconstructed using later reported containment dates. It is not measured crew availability or a census of all active fires.",
    "Weather uses 25 representative grid centroids, approximately 166 km from fires at the median and 348 km at the 95th percentile in this cohort. It is regional context rather than local fireground weather.",
    "Place-centroid distance is a proximity proxy rather than a WUI, road-access, or response-time measurement.",
    "P10–P90 is a conditional outcome interval, not a confidence interval or a guaranteed 80% coverage interval. Check held-out coverage before interpreting it.",
    "Changing one simulator input holds the others fixed; combinations can be unrealistic, and correlations do not establish intervention effects.",
]


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a stable feature order and normalize missing/unknown categories."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model feature columns: {sorted(missing)}")
    clean = frame.loc[:, FEATURES].copy()
    for feature in NUMERIC_FEATURES:
        clean[feature] = pd.to_numeric(clean[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for feature in CATEGORICAL_FEATURES:
        values = clean[feature].astype("string").fillna("Unknown").str.strip()
        clean[feature] = values.replace("", "Unknown").astype(object)
    clean["STATE"] = clean["STATE"].str.upper()
    return clean


def bounded_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    return frame if len(frame) <= maximum else frame.sample(n=maximum, random_state=seed)


def logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def load_bundle(path: str | Path = ROOT / "models" / "risk_bundle.joblib") -> dict[str, Any]:
    """Load a trusted local training artifact. Never load untrusted joblib files."""
    return joblib.load(path)


def predict_batch(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    clean = normalize_features(frame)
    matrix = bundle["preprocessor"].transform(clean)
    raw_probability = bundle["classifier"].predict_proba(matrix)[:, 1]
    probability = bundle["calibrator"].predict_proba(logit(raw_probability))[:, 1]
    unsorted = np.column_stack([
        np.clip(np.expm1(bundle["quantile_models"][quantile].predict(matrix)), 0, 365)
        for quantile in QUANTILES
    ])
    # Independently fitted quantiles can cross. Sorting guarantees a coherent
    # displayed range; held-out metrics below evaluate exactly this final output.
    ordered = np.sort(unsorted, axis=1)
    return {
        "probability_over_3_days": probability,
        "raw_probability_over_3_days": raw_probability,
        "p10_days": ordered[:, 0],
        "p50_days": ordered[:, 1],
        "p90_days": ordered[:, 2],
        "quantile_crossed": np.any(np.diff(unsorted, axis=1) < 0, axis=1),
    }


def predict_scenario(bundle: dict[str, Any], row_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a conditional scenario using exact FEATURE field names.

    LOG_ACRES = natural log(final acres), not log1p. Omitted inputs use the
    bundle's training-cohort defaults. Warnings identify defaults, categories
    absent from training, and inputs outside marginal training support.
    """
    values = dict(bundle["defaults"])
    warnings: list[str] = []
    omitted = [feature for feature in FEATURES if feature not in row_dict]
    if omitted:
        warnings.append("Training defaults used for: " + ", ".join(omitted) + ".")
    values.update({key: value for key, value in row_dict.items() if key in FEATURES})
    for feature in NUMERIC_FEATURES:
        try:
            values[feature] = float(values[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be a finite number.") from exc
        if not np.isfinite(values[feature]):
            raise ValueError(f"{feature} must be a finite number.")
        bounds = bundle["training_bounds"][feature]
        if values[feature] < bounds["min"] or values[feature] > bounds["max"]:
            warnings.append(f"{feature} is outside the observed training range; prediction extrapolates.")
        elif values[feature] < bounds["p01"] or values[feature] > bounds["p99"]:
            warnings.append(f"{feature} is in the outer 1% of its marginal training distribution.")
    for feature in CATEGORICAL_FEATURES:
        values[feature] = str(values[feature]).strip() or "Unknown"
        if feature == "STATE":
            values[feature] = values[feature].upper()
        if values[feature] not in bundle["categories"][feature]:
            warnings.append(f"Unseen {feature}={values[feature]!r}; encoded as an unknown category.")
    if not 1 <= values["DAY_OF_YEAR"] <= 366:
        raise ValueError("DAY_OF_YEAR must be between 1 and 366.")
    if values["OBSERVED_UNCONTAINED_FIRES"] < 0:
        raise ValueError("Observed fire concurrency cannot be negative.")
    for feature in ["vapour_pressure_deficit_max", "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM", "DISTANCE_TO_PLACE_CENTROID_KM"]:
        if values[feature] < 0:
            raise ValueError(f"{feature} cannot be negative.")
    prediction = predict_batch(bundle, pd.DataFrame([values]))
    if bool(prediction["probability_over_3_days"][0] > 0.5) != bool(prediction["p50_days"][0] > 3 + SCENARIO_THRESHOLD_TOLERANCE_DAYS):
        warnings.append("The separately fitted probability and median-duration models disagree on the three-day threshold for this scenario; treat these estimates as exploratory.")
    warnings.extend(bundle.get("inference_warnings", []))
    warnings.append("Conditional retrospective estimate: changing an input does not estimate a causal intervention.")
    return {
        key: float(prediction[key][0])
        for key in ["probability_over_3_days", "p10_days", "p50_days", "p90_days"]
    } | {"warnings": warnings}


def calibration_bins(y_binary: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    bin_index = np.minimum((probability * 10).astype(int), 9)
    rows = []
    for index in range(10):
        selected = bin_index == index
        if np.any(selected):
            rows.append({
                "bin_low": index / 10, "bin_high": (index + 1) / 10,
                "count": int(np.sum(selected)),
                "mean_predicted_probability": float(probability[selected].mean()),
                "observed_fraction_over_3_days": float(y_binary[selected].mean()),
            })
    return rows


def evaluate(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = frame[TARGET].to_numpy(dtype=float)
    binary = (target > 3).astype(int)
    predictions = predict_batch(bundle, frame)
    probability = predictions["probability_over_3_days"]
    bins = calibration_bins(binary, probability)
    baseline_rate = bundle["baselines"]["calibration_base_rate"]
    baseline_median = bundle["baselines"]["training_median_days"]
    baseline_quantiles = bundle["baselines"]["training_duration_quantiles"]
    metrics = {
        "rows": int(len(frame)),
        "target_base_rate": float(binary.mean()),
        "same_calendar_day_fraction": float(np.mean(target == 0)),
        "risk_median_threshold_disagreement_fraction": float(np.mean((probability > 0.5) != (predictions["p50_days"] > 3))),
        "roc_auc": float(roc_auc_score(binary, probability)) if len(np.unique(binary)) == 2 else None,
        "average_precision": float(average_precision_score(binary, probability)) if binary.sum() else None,
        "brier_score": float(brier_score_loss(binary, probability)),
        "raw_brier_score": float(brier_score_loss(binary, predictions["raw_probability_over_3_days"])),
        "baseline_brier_score": float(brier_score_loss(binary, np.full(len(frame), baseline_rate))),
        "baseline_average_precision": float(binary.mean()),
        "training_rate_baseline_brier_score": float(brier_score_loss(binary, np.full(len(frame), bundle["baselines"]["training_base_rate"]))),
        "log_loss": float(log_loss(binary, probability, labels=[0, 1])),
        "expected_calibration_error": float(sum(
            row["count"] / len(frame) * abs(row["mean_predicted_probability"] - row["observed_fraction_over_3_days"])
            for row in bins
        )),
        "median_mae_days": float(mean_absolute_error(target, predictions["p50_days"])),
        "baseline_median_mae_days": float(mean_absolute_error(target, np.full(len(frame), baseline_median))),
        "interval_coverage": float(np.mean((target >= predictions["p10_days"]) & (target <= predictions["p90_days"]))),
        "nominal_interval_coverage": 0.8,
        "mean_interval_width_days": float(np.mean(predictions["p90_days"] - predictions["p10_days"])),
        "baseline_interval_coverage": float(np.mean((target >= baseline_quantiles["p10_days"]) & (target <= baseline_quantiles["p90_days"]))),
        "quantile_crossing_fraction_before_sorting": float(predictions["quantile_crossed"].mean()),
        "calibration_bins": bins,
    }
    for quantile, key in zip(QUANTILES, ["p10_days", "p50_days", "p90_days"]):
        metrics[f"{key}_pinball_loss"] = float(mean_pinball_loss(target, predictions[key], alpha=quantile))
        metrics[f"baseline_{key}_pinball_loss"] = float(mean_pinball_loss(target, np.full(len(frame), baseline_quantiles[key]), alpha=quantile))
        metrics[f"fraction_at_or_below_{key}"] = float(np.mean(target <= predictions[key]))
    return metrics, predictions


def build_context(training: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    bounds = {}
    categories = {}
    for feature in NUMERIC_FEATURES:
        valid = training[feature].dropna()
        if valid.empty:
            defaults[feature] = 0.0
            bounds[feature] = dict(min=0.0, max=0.0, p01=0.0, p99=0.0)
        else:
            defaults[feature] = float(valid.median())
            bounds[feature] = {
                "min": float(valid.min()), "max": float(valid.max()),
                "p01": float(valid.quantile(0.01)), "p99": float(valid.quantile(0.99)),
            }
    for feature in CATEGORICAL_FEATURES:
        defaults[feature] = str(training[feature].mode().iloc[0])
        categories[feature] = sorted(training[feature].unique().tolist())
    state_locations = {
        str(state): {"LATITUDE": float(group.LATITUDE.median()), "LONGITUDE": float(group.LONGITUDE.median())}
        for state, group in training.groupby("STATE", observed=True)
        if group.LATITUDE.notna().any() and group.LONGITUDE.notna().any()
    }
    return {"defaults": defaults, "training_bounds": bounds, "categories": categories, "state_locations": state_locations}


def check_predictions(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    sample = frame.head(64)
    predicted = predict_batch(bundle, sample)
    for key in ["probability_over_3_days", "p10_days", "p50_days", "p90_days"]:
        assert np.isfinite(predicted[key]).all(), f"Nonfinite predictions: {key}"
    assert np.all((predicted["probability_over_3_days"] >= 0) & (predicted["probability_over_3_days"] <= 1))
    assert np.all(predicted["p10_days"] >= 0)
    assert np.all(predicted["p10_days"] <= predicted["p50_days"])
    assert np.all(predicted["p50_days"] <= predicted["p90_days"])
    assert np.all(predicted["p90_days"] <= 365)
    row = dict(bundle["defaults"])
    row["NWCG_REPORTING_AGENCY"] = "UNSEEN_AGENCY_FOR_CHECK"
    unknown = predict_scenario(bundle, row)
    assert any("Unseen" in warning for warning in unknown["warnings"])
    assert 0 <= unknown["probability_over_3_days"] <= 1
    row = dict(bundle["defaults"])
    row["OBSERVED_UNCONTAINED_FIRES"] = bundle["training_bounds"]["OBSERVED_UNCONTAINED_FIRES"]["max"] + 1
    extrapolation = predict_scenario(bundle, row)
    assert any("extrapolates" in warning for warning in extrapolation["warnings"])
    return {"finite_bounded_predictions": True, "ordered_quantiles": True, "unknown_category_handled": True, "extrapolation_warning": True}


def annotate_findings(bundle: dict[str, Any], training_target: np.ndarray, training_matrix: np.ndarray, test_metrics: dict[str, Any]) -> list[str]:
    """Record observed limitations without altering or selecting fitted models."""
    ranges = {}
    for quantile, key in zip(QUANTILES, ["p10_days", "p50_days", "p90_days"]):
        values = np.clip(np.expm1(bundle["quantile_models"][quantile].predict(training_matrix)), 0, 365)
        ranges[key] = {"min": float(values.min()), "max": float(values.max()), "constant": bool(np.ptp(values) <= 1e-12)}
    diagnostics = {
        "fit_sample_rows": int(len(training_target)),
        "fit_sample_same_calendar_day_fraction": float(np.mean(training_target == 0)),
        "fit_sample_median_days": float(np.median(training_target)),
        "native_training_prediction_ranges": ranges,
    }
    warnings = []
    findings = []
    if ranges["p50_days"]["constant"] and abs(ranges["p50_days"]["max"]) <= 1e-12:
        warnings.append("The median-duration model predicts same-calendar-day containment for every scenario and adds no individualized median-duration skill.")
        findings.append(f"P50 is constant at zero across the {len(training_target):,}-row fitting sample. Test median MAE is {test_metrics['median_mae_days']:.6f} days, equal to the constant-zero baseline; the median model adds no predictive value over that baseline.")
    if ranges["p10_days"]["constant"]:
        findings.append(f"P10 is also constant across the fitting sample ({ranges['p10_days']['min']:.6f} calendar days). Zero-heavy discrete outcomes and quantile fitting limit these lower-duration estimates.")
    if not ranges["p50_days"]["constant"]:
        findings.append(f"After the training/synthetic-verified numerical tie correction, P50 varies across the fitting sample. Exploratory test median MAE is {test_metrics['median_mae_days']:.6f} days versus {test_metrics['baseline_median_mae_days']:.6f} for the constant-zero baseline.")
    findings.append(f"P10–P90 test coverage is {test_metrics['interval_coverage']:.2%}, versus nominal 80%; the discrete, zero-heavy interval is not calibrated to 80% coverage.")
    if test_metrics["brier_score"] > test_metrics["raw_brier_score"]:
        warnings.append("Absolute probabilities show calibration drift in 2020–2024: the earlier-era probability adjustment performed worse than the raw model on that test era.")
        findings.append(f"The predeclared Platt adjustment improved validation Brier but worsened 2020–2024 test Brier ({test_metrics['brier_score']:.6f} adjusted versus {test_metrics['raw_brier_score']:.6f} raw). The original calibrated estimator is retained; test results were not used to select a replacement.")
    findings.append("The test era had already been inspected before a synthetic-proven quantile tie failure was corrected. Final repeated test scores are exploratory, not a fresh confirmatory evaluation.")
    warnings.append("These final scores are exploratory because the test era was evaluated before a numerical quantile-model correction.")
    bundle["inference_warnings"] = warnings
    bundle["metadata"]["duration_model_diagnostics"] = diagnostics
    bundle["metadata"]["evaluation_findings"] = findings
    bundle["metadata"]["quantile_numerical_correction"] = QUANTILE_FIX_EVIDENCE
    bundle["metadata"]["test_evaluation_status"] = "exploratory_repeat_after_synthetic_verified_numerical_fix"
    bundle["metadata"]["limitations"] = list(LIMITATIONS) + findings
    return findings


def check_zero_tie_quantile_regression() -> dict[str, Any]:
    """Reproduce and verify the numerical repair using no real fire outcomes."""
    features = np.arange(5000).reshape(-1, 1)
    target = np.where(features[:, 0] >= 4000, 10.0, 0.0)
    settings = dict(max_iter=140, learning_rate=0.07, max_leaf_nodes=31,
                    min_samples_leaf=80, l2_regularization=5,
                    early_stopping=True, n_iter_no_change=15, random_state=42)
    raw = np.log1p(target)
    smoothed = raw + np.random.default_rng(42).uniform(-QUANTILE_TIE_BREAK_AMPLITUDE, QUANTILE_TIE_BREAK_AMPLITUDE, len(target))
    original = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **settings).fit(features, raw)
    corrected = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **settings).fit(features, smoothed)
    original_predictions = np.clip(np.expm1(original.predict([[100], [4800]])), 0, 365)
    corrected_predictions = np.clip(np.expm1(corrected.predict([[100], [4800]])), 0, 365)
    assert corrected_predictions[0] < 0.01 and corrected_predictions[1] > 9, "Numerical tie correction fails a well-supported conditional-median example."
    return {"passed": True, "rows": 5000, "positive_subgroup_rows": 1000,
            "min_samples_leaf": 80, "true_medians_days": [0.0, 10.0],
            "original_medians_days": original_predictions.tolist(),
            "corrected_medians_days": corrected_predictions.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed_fires.parquet")
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-eval-rows", type=int, default=250_000)
    parser.add_argument("--max-iterations", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_eval_rows, args.max_iterations) < 1:
        raise ValueError("Row and iteration limits must be positive.")
    print("Loading date-comparable containment cohort...", flush=True)
    frame = pd.read_parquet(args.input, columns=FEATURES + [TARGET, "FIRE_YEAR", "DURATION_RESOLUTION"])
    total_rows = len(frame)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame["FIRE_YEAR"] = pd.to_numeric(frame["FIRE_YEAR"], errors="coerce")
    frame = frame.loc[frame[TARGET].between(0, 365) & frame["FIRE_YEAR"].between(1992, 2024)].copy()
    if not np.allclose(frame[TARGET].to_numpy(), np.rint(frame[TARGET].to_numpy()), rtol=0, atol=1e-9):
        raise ValueError("The date-comparable target must contain whole calendar days; normalize both source dates before subtraction.")
    frame.loc[:, FEATURES] = normalize_features(frame)
    splits: dict[str, pd.DataFrame] = {}
    split_report: dict[str, Any] = {}
    for name, (first, last) in SPLITS.items():
        full = frame.loc[frame.FIRE_YEAR.between(first, last)]
        if full.empty:
            raise ValueError(f"No eligible data in {name} years {first}–{last}.")
        maximum = args.max_train_rows if name == "training" else args.max_eval_rows
        split = bounded_sample(full, maximum, args.seed).copy()
        assert split.FIRE_YEAR.min() >= first and split.FIRE_YEAR.max() <= last
        splits[name] = split
        split_report[name] = {
            "start_year": first, "end_year": last,
            "eligible_rows": int(len(full)), "used_rows": int(len(split)),
            "sampled": bool(len(full) > maximum),
            "target_base_rate": float((full[TARGET] > 3).mean()),
            "duration_resolution_counts": {str(key): int(value) for key, value in full.DURATION_RESOLUTION.value_counts(dropna=False).items()},
        }
    assert SPLITS["training"][1] < SPLITS["calibration"][0] <= SPLITS["calibration"][1] < SPLITS["validation"][0] <= SPLITS["validation"][1] < SPLITS["test"][0]
    training = splits["training"]
    context = build_context(normalize_features(frame.loc[frame.FIRE_YEAR.between(*SPLITS["training"])]))
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUMERIC_FEATURES),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)
    matrix = preprocessor.fit_transform(normalize_features(training))
    # The encoder fits the bounded training sample. Its known-category lists
    # must match that fitted sample even when context summaries use all early
    # years, so rare omitted categories correctly trigger scenario warnings.
    context["categories"] = {
        feature: list(preprocessor.named_transformers_["categorical"].categories_[index])
        for index, feature in enumerate(CATEGORICAL_FEATURES)
    }
    categorical_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
    settings = dict(
        max_iter=args.max_iterations, learning_rate=0.07, max_leaf_nodes=31,
        min_samples_leaf=80, l2_regularization=5, categorical_features=categorical_mask,
        early_stopping=True, n_iter_no_change=15, random_state=args.seed,
    )
    target = training[TARGET].to_numpy(dtype=float)
    binary = (target > 3).astype(int)
    if len(np.unique(binary)) != 2:
        raise ValueError("Training must contain fires on both sides of the three-day threshold.")
    print(f"Training classifier on {len(training):,} pre-2015 fires...", flush=True)
    classifier = HistGradientBoostingClassifier(**settings).fit(matrix, binary)
    quantile_models = {}
    # The pinball gradient can be constant at a tied minimum prediction when
    # most observations are exactly zero, preventing supported positive groups
    # from learning. Sub-calendar-resolution, fixed-seed numerical tie breaking
    # repairs this synthetic-reproduced failure without changing the estimators
    # or evaluation targets. Amplitude was fixed using the synthetic case only.
    quantile_target = np.log1p(target) + np.random.default_rng(args.seed).uniform(
        -QUANTILE_TIE_BREAK_AMPLITUDE, QUANTILE_TIE_BREAK_AMPLITUDE, len(target)
    )
    for quantile in QUANTILES:
        print(f"Training P{int(100 * quantile)} duration model...", flush=True)
        quantile_models[quantile] = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **settings).fit(matrix, quantile_target)
    calibration = splits["calibration"]
    calibration_matrix = preprocessor.transform(normalize_features(calibration))
    calibration_binary = (calibration[TARGET].to_numpy() > 3).astype(int)
    if len(np.unique(calibration_binary)) != 2:
        raise ValueError("Calibration must contain both binary outcomes.")
    print("Calibrating probabilities using 2015–2017 only...", flush=True)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500, random_state=args.seed).fit(
        logit(classifier.predict_proba(calibration_matrix)[:, 1]), calibration_binary
    )
    bundle: dict[str, Any] = {
        "artifact_version": 2,
        "preprocessor": preprocessor, "classifier": classifier, "calibrator": calibrator,
        "quantile_models": quantile_models, **context,
        "feature_schema": {
            "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
            "target": TARGET, "threshold_days": 3,
            "acreage_transform": "natural_log", "duration_transform": "log1p",
            "units": {
                "LOG_ACRES": "natural log of final acres", "LATITUDE": "degrees", "LONGITUDE": "degrees",
                "DAY_OF_YEAR": "1–366", "OBSERVED_UNCONTAINED_FIRES": "observed cohort fires including focal fire",
                "vapour_pressure_deficit_max": "kPa", "wind_speed_10m_max": "km/h",
                "ANTECEDENT_PRECIP_30D_MM": "mm", "DISTANCE_TO_PLACE_CENTROID_KM": "km",
            },
        },
        "baselines": {
            "training_base_rate": float(binary.mean()),
            "calibration_base_rate": float(calibration_binary.mean()),
            "training_median_days": float(np.median(target)),
            "training_duration_quantiles": {key: float(np.quantile(target, q)) for q, key in zip(QUANTILES, ["p10_days", "p50_days", "p90_days"])},
        },
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
            "source_file": args.input.name, "source_rows": int(total_rows), "eligible_rows": int(len(frame)),
            "source_file_bytes": args.input.stat().st_size,
            "chronological_splits": split_report,
            "seed": args.seed, "max_iterations": args.max_iterations,
            "no_year_feature": True,
            "internal_early_stopping": "Random 10% validation wholly inside the pre-2015 fitting sample.",
            "quantiles": "Independently fitted log1p-day quantiles, inverse transformed, clipped to 0–365, and sorted per record.",
            "quantile_training_tie_break_amplitude_log_days": QUANTILE_TIE_BREAK_AMPLITUDE,
            "calibration": "Platt logistic calibration on the logit of raw probability; uses 2015–2017 only.",
            "training_feature_missing_fraction": {key: float(training[key].isna().mean()) for key in NUMERIC_FEATURES},
            "limitations": LIMITATIONS,
        },
    }
    print("Evaluating untouched validation and test eras...", flush=True)
    validation_metrics, _ = evaluate(bundle, splits["validation"])
    test_metrics, test_predictions = evaluate(bundle, splits["test"])
    findings = annotate_findings(bundle, target, matrix, test_metrics)
    checks = check_predictions(bundle, splits["test"])
    checks["whole_calendar_day_target"] = True
    checks["chronological_holdouts_separated"] = True
    bundle["metadata"]["synthetic_quantile_regression_check"] = check_zero_tie_quantile_regression()
    bundle["metadata"]["scenario_threshold_tolerance_days"] = SCENARIO_THRESHOLD_TOLERANCE_DAYS
    checks["synthetic_zero_tie_regression_passed"] = True
    reports = ROOT / "reports"
    models = ROOT / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    artifact = models / "risk_bundle.joblib"
    joblib.dump(bundle, artifact, compress=3)
    restored = load_bundle(artifact)
    before = predict_batch(bundle, splits["test"].head(8))
    after = predict_batch(restored, splits["test"].head(8))
    assert all(np.allclose(before[key], after[key]) for key in before)
    checks["serialized_round_trip"] = True
    results = {
        "target": TARGET, "threshold": "more than 3 calendar-date days",
        "chronological_splits": split_report,
        "test_metrics": test_metrics, "validation_metrics": validation_metrics,
        "calibration_bins": test_metrics["calibration_bins"],
        "baselines": bundle["baselines"],
        "training": {"classifier_iterations": int(classifier.n_iter_), "quantile_iterations": {str(q): int(model.n_iter_) for q, model in quantile_models.items()}, "settings": settings},
        "checks": checks, "limitations": bundle["metadata"]["limitations"], "findings": findings,
        "metadata": bundle["metadata"],
    }
    (reports / "risk_metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(results["calibration_bins"]).to_csv(reports / "risk_calibration_bins.csv", index=False)
    yearly_rows = []
    for year, group in splits["test"].groupby("FIRE_YEAR", observed=True):
        metrics, _ = evaluate(bundle, group)
        yearly_rows.append({"year": int(year), **{key: value for key, value in metrics.items() if key != "calibration_bins"}})
    pd.DataFrame(yearly_rows).to_csv(reports / "risk_test_by_year.csv", index=False)
    resolution_rows = []
    for resolution, group in splits["test"].groupby("DURATION_RESOLUTION", observed=True, dropna=False):
        metrics, _ = evaluate(bundle, group)
        resolution_rows.append({"duration_resolution": str(resolution), **{key: value for key, value in metrics.items() if key != "calibration_bins"}})
    pd.DataFrame(resolution_rows).to_csv(reports / "risk_test_by_resolution.csv", index=False)
    sample_rows = min(2000, len(splits["test"]))
    examples = splits["test"][["FIRE_YEAR", TARGET, "DURATION_RESOLUTION"]].iloc[:sample_rows].reset_index(drop=True)
    for key in ["probability_over_3_days", "p10_days", "p50_days", "p90_days"]:
        examples[key] = test_predictions[key][:sample_rows]
    examples.to_csv(reports / "risk_test_prediction_sample.csv", index=False)
    print(json.dumps({"artifact": str(artifact), "test_metrics": {key: value for key, value in test_metrics.items() if key != "calibration_bins"}, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
