"""Audit fixed retrospective fire models, refresh calibration, and gate support.

The classifier and duration estimators are never refitted here. Calibration
selection uses 2020–2021 fitting and 2022 validation only. The chosen mapping is
refitted on 2020–2022 and reassessed on 2023–2024; these historical years were
already inspected, so this is explicitly a reassessment rather than a new test.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

try:
    from scripts.train_risk import (
        CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
        bounded_sample, calibration_bins, load_bundle, logit,
        normalize_features, predict_batch,
    )
except ModuleNotFoundError:
    from train_risk import (
        CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
        bounded_sample, calibration_bins, load_bundle, logit,
        normalize_features, predict_batch,
    )

ROOT = Path(__file__).resolve().parents[1]
SIZE_LABELS = ["<1 acre", "1–<10 acres", "10–<100 acres", "100–<1000 acres", "1000–<10000 acres", "10000+ acres"]
SUPPORT_K = 10
MIN_CONTEXT = 30
MIN_THRESHOLD_QUERIES = 50
THRESHOLD_TOLERANCE_DAYS = 1e-4
AUDIT_LIMITATIONS = [
    "This is a retrospective conditional model using final acreage, not an operational discovery-time forecast.",
    "Observed national concurrency is reconstructed from outcome-complete records and includes the focal fire; it is endogenous and is not crew availability or a causal resource constraint.",
    "2023–2024 were inspected in the earlier project. This later comparison is a historical reassessment, not an untouched confirmatory evaluation.",
    "A calibration mapping chosen on 2022 can still drift in 2023–2024 or future years. Subgroup calibration and ranking can differ sharply.",
    "P10–P90 duration outputs remain fixed exploratory quantiles. Aggregate coverage is distorted by same-day outcomes and does not validate intervals for large or prolonged fires.",
    "Nearest-neighbor support is a feature-space heuristic using a bounded sample of earlier fitting records. Passing the gate does not establish accuracy, scientific validity, or causal identifiability.",
    "Exact state, reporting agency, and cause matches plus standardized numeric neighborhoods assess joint support. Regional weather is coarse and cannot represent local fireground conditions.",
]


def load_audited_bundle(path: str | Path = ROOT / "models" / "risk_bundle_audited.joblib") -> dict[str, Any]:
    return joblib.load(path)


def size_group(log_acres: np.ndarray | float) -> np.ndarray:
    """Use the model's log domain to preserve exact acreage-bin boundaries."""
    return np.asarray(pd.cut(np.atleast_1d(log_acres), [-np.inf, 0, np.log(10), np.log(100), np.log(1000), np.log(10000), np.inf], right=False, labels=SIZE_LABELS).astype(str))


def mapped_probability(raw: np.ndarray, method: str, estimator: Any = None) -> np.ndarray:
    if method == "raw":
        return np.asarray(raw, dtype=float)
    if method in {"old_platt", "recent_platt"}:
        return estimator.predict_proba(logit(raw))[:, 1]
    if method == "recent_isotonic":
        return np.clip(estimator.predict(np.asarray(raw)), 0, 1)
    raise ValueError(f"Unknown calibration method {method!r}")


def probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    bins = calibration_bins(target, probability)
    return {
        "rows": int(len(target)), "events_over_3_days": int(target.sum()),
        "observed_fraction_over_3_days": float(target.mean()),
        "mean_predicted_probability": float(probability.mean()),
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
        "average_precision": float(average_precision_score(target, probability)) if target.sum() else None,
        "expected_calibration_error": float(sum(row["count"] / len(target) * abs(row["mean_predicted_probability"] - row["observed_fraction_over_3_days"]) for row in bins)),
    }


def duration_metrics(target: np.ndarray, prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    target = np.asarray(target, dtype=float)
    lower, median, upper = (prediction[key] for key in ["p10_days", "p50_days", "p90_days"])
    return {
        "same_calendar_day_fraction": float(np.mean(target == 0)),
        "median_mae_days": float(np.mean(np.abs(target - median))),
        "constant_zero_mae_days": float(np.mean(target)),
        "interval_coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_interval_width_days": float(np.mean(upper - lower)),
        "fraction_at_or_below_p90": float(np.mean(target <= upper)),
        "fraction_below_p10": float(np.mean(target < lower)),
        "fraction_above_p90": float(np.mean(target > upper)),
    }


def support_numeric(frame: pd.DataFrame) -> np.ndarray:
    clean = normalize_features(frame)
    day = clean.DAY_OF_YEAR.to_numpy(dtype=float)
    columns = [clean.LOG_ACRES, clean.LATITUDE, clean.LONGITUDE,
               np.sin(2 * np.pi * day / 365.25) / np.sqrt(2),
               np.cos(2 * np.pi * day / 365.25) / np.sqrt(2),
               np.log1p(clean.OBSERVED_UNCONTAINED_FIRES.clip(lower=0)),
               clean.vapour_pressure_deficit_max,
               np.log1p(clean.wind_speed_10m_max.clip(lower=0)),
               np.log1p(clean.ANTECEDENT_PRECIP_30D_MM.clip(lower=0)),
               np.log1p(clean.DISTANCE_TO_PLACE_CENTROID_KM.clip(lower=0))]
    result = np.column_stack(columns).astype(float)
    if not np.isfinite(result).all():
        raise ValueError("Joint support requires finite feature values.")
    return result


def context_keys(frame: pd.DataFrame) -> np.ndarray:
    clean = normalize_features(frame)
    return clean[CATEGORICAL_FEATURES].astype(str).agg(" | ".join, axis=1).to_numpy()


def neighbor_distances(support_model: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = support_model["scaler"].transform(support_numeric(frame))
    keys = context_keys(frame)
    distances = np.full(len(frame), np.nan)
    counts = np.zeros(len(frame), dtype=int)
    positions = pd.Series(np.arange(len(frame))).groupby(keys, sort=False).apply(list)
    for key, indices in positions.items():
        info = support_model["contexts"].get(str(key))
        if info is None:
            continue
        indices = np.asarray(indices, dtype=int)
        counts[indices] = info["count"]
        if info["count"] >= MIN_CONTEXT:
            distances[indices] = info["neighbors"].kneighbors(matrix[indices], n_neighbors=SUPPORT_K, return_distance=True)[0][:, -1]
    return distances, counts, keys


def fit_support(training: pd.DataFrame, seed: int, reference_rows: int = 100_000, threshold_rows: int = 20_000) -> tuple[dict[str, Any], pd.DataFrame]:
    reference = bounded_sample(training, reference_rows, seed + 11).copy()
    remaining = training.drop(index=reference.index)
    threshold_queries = bounded_sample(remaining, threshold_rows, seed + 12).copy()
    assert reference.index.intersection(threshold_queries.index).empty
    if len(threshold_queries) < 1000:
        raise ValueError("Need disjoint fitting records for neighborhood thresholds.")
    scaler = RobustScaler(quantile_range=(25, 75)).fit(support_numeric(reference))
    matrix = scaler.transform(support_numeric(reference))
    keys = context_keys(reference)
    contexts = {}
    positions = pd.Series(np.arange(len(reference))).groupby(keys, sort=False).apply(list)
    for key, indices in positions.items():
        indices = np.asarray(indices, dtype=int)
        info = {"count": int(len(indices))}
        if len(indices) >= MIN_CONTEXT:
            info["neighbors"] = NearestNeighbors(algorithm="ball_tree", metric="euclidean", leaf_size=40, n_jobs=4).fit(matrix[indices])
        contexts[str(key)] = info
    support_model = {"scaler": scaler, "contexts": contexts, "k": SUPPORT_K,
                     "reference_rows": len(reference), "threshold_rows": len(threshold_queries),
                     "min_context_records": MIN_CONTEXT,
                     "reference_feature_rows": normalize_features(reference).reset_index(drop=True),
                     "reference_metadata": reference[["FOD_ID", "FIRE_YEAR"]].reset_index(drop=True)}
    distances, counts, query_keys = neighbor_distances(support_model, threshold_queries)
    finite = np.isfinite(distances)
    if finite.sum() < 1000:
        raise ValueError("Too few sufficiently populated reference contexts.")
    pooled = {"p90": float(np.quantile(distances[finite], .9)), "p99": float(np.quantile(distances[finite], .99))}
    thresholds = {}
    for key in np.unique(query_keys[finite]):
        local = distances[(query_keys == key) & finite]
        if len(local) >= MIN_THRESHOLD_QUERIES:
            thresholds[str(key)] = {"p90": float(np.quantile(local, .9)), "p99": float(np.quantile(local, .99)), "source": "context-specific disjoint fitting records", "threshold_query_rows": int(len(local))}
    support_model["thresholds"] = thresholds
    support_model["pooled_thresholds"] = pooled
    support_model["definition"] = "Exact STATE+agency+cause context; Euclidean 10th-neighbor distance after robust IQR scaling, cyclic season, and log1p load/wind/rain/place distance. Thresholds use disjoint pre-2015 fitting records without outcomes."
    return support_model, reference


def support_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    clean = normalize_features(frame)
    model = bundle["support_model"]
    distances, counts, keys = neighbor_distances(model, clean)
    rows = []
    marginal_outside = np.zeros(len(clean), dtype=bool)
    for feature in NUMERIC_FEATURES:
        bounds = bundle["training_bounds"][feature]
        marginal_outside |= (clean[feature].to_numpy() < bounds["min"]) | (clean[feature].to_numpy() > bounds["max"])
    for index, (key, distance, count) in enumerate(zip(keys, distances, counts)):
        threshold = model["thresholds"].get(str(key))
        if threshold is None:
            threshold = {**model["pooled_thresholds"], "source": "pooled disjoint fitting records", "threshold_query_rows": 0}
        if count < MIN_CONTEXT:
            status, reason = "unsupported", f"Only {count} earlier reference fires match this state, reporting agency, and cause; at least {MIN_CONTEXT} are required."
        elif marginal_outside[index]:
            status, reason = "unsupported", "At least one numeric input is outside its observed early-training range."
        elif not np.isfinite(distance) or distance > threshold["p99"]:
            status, reason = "unsupported", "This joint feature combination is farther from comparable earlier fires than the training-derived 99th-percentile neighborhood threshold."
        elif distance > threshold["p90"]:
            status, reason = "caution", "This combination is in the outer 10% of comparable training neighborhoods."
        elif str(key) not in model["thresholds"]:
            status, reason = "caution", "Comparable reference fires exist, but this context has too few disjoint records for its own distance threshold."
        else:
            status, reason = "supported", "Comparable earlier fires exist within the training-derived neighborhood threshold; this does not guarantee accuracy."
        rows.append({"status": status, "context_reference_count": int(count),
                     "kth_distance": float(distance) if np.isfinite(distance) else None,
                     "threshold_caution_p90": threshold["p90"], "threshold_unsupported_p99": threshold["p99"],
                     "threshold_source": threshold["source"], "presentation_allowed": status != "unsupported", "reason": reason})
    return pd.DataFrame(rows)


def support_dict(row: pd.Series) -> dict[str, Any]:
    return {"status": row.status, "context_reference_count": int(row.context_reference_count),
            "kth_distance": None if pd.isna(row.kth_distance) else float(row.kth_distance),
            "thresholds": {"caution_p90": float(row.threshold_caution_p90), "unsupported_p99": float(row.threshold_unsupported_p99),
                           "source": row.threshold_source, "k": SUPPORT_K, "min_context_records": MIN_CONTEXT},
            "presentation_allowed": bool(row.presentation_allowed), "reason": row.reason}


def predict_conditional_prolonged(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Conditional duration bounds given that calendar duration exceeds3days."""
    model = bundle["prolonged_duration"]
    matrix = bundle["preprocessor"].transform(normalize_features(frame))
    native = np.column_stack([np.clip(np.expm1(model["quantile_models"][q].predict(matrix)), 4, 365) for q in [.1, .5, .9]])
    ordered = np.sort(native, axis=1)
    expansion = model["conformal_expansion_days"]
    return {"lower_days": np.clip(ordered[:, 0] - expansion, 4, 365),
            "median_days": ordered[:, 1], "upper_days": np.clip(ordered[:, 2] + expansion, 4, 365),
            "native_lower_days": ordered[:, 0], "native_upper_days": ordered[:, 2]}


def prolonged_support_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Assess feature support in the prolonged model's own fitting cohort."""
    conditional = bundle["prolonged_duration"]
    if "support_model" not in conditional:
        raise ValueError("The separate prolonged-training support model is unavailable.")
    branch = {**bundle, "support_model": conditional["support_model"],
              "training_bounds": conditional["training_bounds"]}
    return support_frame(branch, frame)


def predict_audited_scenario(bundle: dict[str, Any], row_dict: dict[str, Any]) -> dict[str, Any]:
    values = {**bundle["defaults"], **{key: value for key, value in row_dict.items() if key in FEATURES}}
    warnings = []
    missing = [feature for feature in FEATURES if feature not in row_dict]
    if missing:
        warnings.append("Training defaults supplied for missing features: " + ", ".join(missing) + ". Prefer a coherent recorded example.")
    for feature in NUMERIC_FEATURES:
        try:
            values[feature] = float(values[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be finite.") from exc
        if not np.isfinite(values[feature]):
            raise ValueError(f"{feature} must be finite.")
    if not 1 <= values["DAY_OF_YEAR"] <= 366:
        raise ValueError("DAY_OF_YEAR must be in1–366.")
    for feature in ["OBSERVED_UNCONTAINED_FIRES", "vapour_pressure_deficit_max", "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM", "DISTANCE_TO_PLACE_CENTROID_KM"]:
        if values[feature] < 0:
            raise ValueError(f"{feature} cannot be negative.")
    frame = normalize_features(pd.DataFrame([values]))
    prediction = predict_batch(bundle, frame)
    update = bundle["probability_update"]
    probability = float(mapped_probability(prediction["raw_probability_over_3_days"], update["method"], update["estimator"])[0])
    support = support_dict(support_frame(bundle, frame).iloc[0])
    if not support["presentation_allowed"]:
        warnings.append("Suppress numeric forecasts: " + support["reason"])
    elif support["status"] == "caution":
        warnings.append(support["reason"])
    disagreement = (probability > .5) != bool(prediction["p50_days"][0] > 3 + THRESHOLD_TOLERANCE_DAYS)
    if disagreement:
        warnings.append("The separate probability and duration models disagree at the three-day threshold for this case; the outputs do not form one coherent predictive distribution.")
    group = str(size_group(values["LOG_ACRES"])[0])
    reliability = {"size": bundle.get("audit_reliability", {}).get("size", {}).get(group),
                   "agency": bundle.get("audit_reliability", {}).get("agency", {}).get(str(frame.NWCG_REPORTING_AGENCY.iloc[0]))}
    warnings.extend([
        "Historical reassessment only: the 2023–2024 evaluation years were previously inspected, and calibration may drift again.",
        "The duration range is an exploratory P10–P90 model output, not a validated 80% interval for this case; inspect size-specific coverage.",
        "Final acreage and reconstructed national load make this a retrospective association, not an actionable ignition-time forecast or a causal intervention.",
    ])
    result = {"probability_over_3_days": probability,
            "p10_days": float(prediction["p10_days"][0]), "p50_days": float(prediction["p50_days"][0]), "p90_days": float(prediction["p90_days"][0]),
            "probability_method": update["method"], "support": support,
            "group_reliability": reliability, "warnings": warnings}
    if "prolonged_duration" in bundle:
        conditional = predict_conditional_prolonged(bundle, frame)
        metadata = bundle["prolonged_duration"]["metadata"]
        if "support_model" in bundle["prolonged_duration"]:
            prolonged_support = support_dict(prolonged_support_frame(bundle, frame).iloc[0])
        else:
            # Fail closed while an older artifact is upgraded: plentiful
            # all-fire neighbors cannot establish prolonged-model support.
            prolonged_support = {"status": "unsupported", "context_reference_count": 0, "kth_distance": None,
                                  "thresholds": {"caution_p90": None, "unsupported_p99": None, "source": "separate prolonged-training gate unavailable", "k": SUPPORT_K, "min_context_records": MIN_CONTEXT},
                                  "presentation_allowed": False, "reason": "Separate prolonged-training feature support has not been established; hide the conditional duration estimate."}
        duration_allowed = bool(support["presentation_allowed"] and prolonged_support["presentation_allowed"])
        result["conditional_prolonged_duration"] = {
            "lower_days": float(conditional["lower_days"][0]), "median_days": float(conditional["median_days"][0]), "upper_days": float(conditional["upper_days"][0]),
            "p10_days": float(conditional["lower_days"][0]), "p50_days": float(conditional["median_days"][0]), "p90_days": float(conditional["upper_days"][0]),
            "conditional_on": "Containment takes more than3 calendar days",
            "interval_method": "CQR-adjusted nominal80% bounds conditional on prolonged containment; expanded bounds are not exact P10/P90 quantiles",
            "calibration_rows": metadata["calibration_rows"],
            "reassessment_coverage": metadata["reassessment_metrics"]["interval_coverage"],
            "group_reliability": {"size": metadata.get("reassessment_groups", {}).get("size", {}).get(group),
                                  "agency": metadata.get("reassessment_groups", {}).get("agency", {}).get(str(frame.NWCG_REPORTING_AGENCY.iloc[0]))},
            "support": prolonged_support, "status": prolonged_support["status"],
            "presentation_allowed": duration_allowed,
        }
        if not prolonged_support["presentation_allowed"]:
            warnings.append("Suppress conditional prolonged duration: " + prolonged_support["reason"])
        elif prolonged_support["status"] == "caution":
            warnings.append("Conditional prolonged-duration support needs caution: " + prolonged_support["reason"])
        warnings.append("The separate prolonged-duration range assumes containment will exceed3 calendar days. Its nominal80% calibration is marginal among prolonged events and does not guarantee coverage for this case or size group.")
    return result


def choose_examples(bundle: dict[str, Any], reference: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    pools = [("Common small recorded fire", reference.loc[reference.LOG_ACRES < 0]),
             ("Larger recorded fire", reference.loc[reference.LOG_ACRES >= np.log(1000)])]
    examples = []
    used = set()
    for label, pool in pools:
        candidates = bounded_sample(pool, 5000, seed + 9).copy()
        if candidates.empty:
            continue
        support = support_frame(bundle, candidates).reset_index(drop=True)
        candidates = candidates.reset_index(drop=True)
        eligible = support.loc[support.status.eq("supported")].copy()
        eligible["relative_distance"] = eligible.kth_distance / eligible.threshold_caution_p90.clip(lower=1e-9)
        for index in eligible.sort_values("relative_distance").index:
            row = candidates.iloc[index]
            key = (str(row.STATE), str(row.NWCG_REPORTING_AGENCY), label)
            if key in used:
                continue
            used.add(key)
            examples.append({"label": f"{label}: {row.STATE}, {row.NWCG_REPORTING_AGENCY}, {np.exp(row.LOG_ACRES):,.2g} acres",
                             "features": {feature: float(row[feature]) if feature in NUMERIC_FEATURES else str(row[feature]) for feature in FEATURES},
                             "source_year": int(row.FIRE_YEAR), "source_fod_id": int(row.FOD_ID), "support": support_dict(support.iloc[index])})
            if sum(example["label"].startswith(label) for example in examples) >= 2:
                break
    return examples


def scenario_grid_support(bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A fixed feature-only stress grid; its support rate is not a population rate."""
    examples = bundle["supported_examples"]
    selected = [examples[0]]
    larger = next((example for example in examples if example["label"].startswith("Larger")), None)
    if larger is not None:
        selected.append(larger)
    reference = bundle["support_model"]["reference_feature_rows"]
    varied = ["vapour_pressure_deficit_max", "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM", "OBSERVED_UNCONTAINED_FIRES"]
    quantiles = {feature: {q: float(reference[feature].quantile(q)) for q in [.1, .9]} for feature in varied}
    rows = []
    for example in selected:
        for acres, *choices in product([1, 100, 10000], [.1, .9], [.1, .9], [.1, .9], [.1, .9]):
            values = dict(example["features"])
            values["LOG_ACRES"] = float(np.log(acres))
            values.update({feature: quantiles[feature][choice] for feature, choice in zip(varied, choices)})
            rows.append({"base_example": example["label"], "scenario_acres": acres, **values})
    frame = pd.DataFrame(rows)
    assessment = support_frame(bundle, frame)
    outside = np.zeros(len(frame), dtype=bool)
    for feature in NUMERIC_FEATURES:
        bounds = bundle["training_bounds"][feature]
        outside |= (frame[feature].to_numpy() < bounds["min"]) | (frame[feature].to_numpy() > bounds["max"])
    frame = pd.concat([frame, assessment], axis=1)
    summary = {"rows": int(len(frame)), "definition": "Fixed grid around one actual small and one actual larger supported case: final acres1/100/10000; independent10th/90th-reference percentiles of VPD, wind, antecedent rain, and observed load. Context and remaining features stay at the recorded case.",
               "all_individual_numeric_features_within_training_ranges": bool(not outside.any()),
               "presentation_allowed_fraction": float(frame.presentation_allowed.mean()),
               "unsupported_fraction": float(frame.status.eq("unsupported").mean()),
               "status_counts": {str(key): int(value) for key, value in frame.status.value_counts().items()},
               "interpretation": "Illustrative stress-grid result, not the proportion of all possible scenarios or actual U.S. fires."}
    return frame, summary


def audit_groups(frame: pd.DataFrame, predictions: dict[str, np.ndarray], probabilities: dict[str, np.ndarray], support: pd.DataFrame, reassessment_mask: np.ndarray) -> pd.DataFrame:
    grouping = {"year": frame.FIRE_YEAR.astype(str).to_numpy(),
                "size": size_group(frame.LOG_ACRES),
                "agency": frame.NWCG_REPORTING_AGENCY.astype(str).to_numpy()}
    rows = []
    for scope, scope_mask in [("2020–2024 diagnostic", np.ones(len(frame), dtype=bool)), ("2023–2024 reassessment", reassessment_mask)]:
        for dimension, groups in grouping.items():
            for group in np.unique(groups[scope_mask]):
                selected = scope_mask & (groups == group)
                target = frame[TARGET].to_numpy()[selected]
                binary = (target > 3).astype(int)
                duration = duration_metrics(target, {key: value[selected] for key, value in predictions.items() if key in {"p10_days", "p50_days", "p90_days"}})
                for method, probability in probabilities.items():
                    row = {"scope": scope, "dimension": dimension, "group": str(group), "method": method,
                           **probability_metrics(binary, probability[selected]), **duration,
                           "risk_median_disagreement_fraction": float(np.mean((probability[selected] > .5) != (predictions["p50_days"][selected] > 3 + THRESHOLD_TOLERANCE_DAYS))),
                           "presentation_allowed_fraction": float(support.presentation_allowed.to_numpy()[selected].mean()),
                           "supported_fraction": float(support.status.to_numpy()[selected].astype(str).tolist().count("supported") / selected.sum()),
                           "small_subgroup_warning": bool(selected.sum() < 200 or binary.sum() < 20)}
                    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed_fires.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("Loading fixed estimators and selected audit fields...", flush=True)
    original = load_bundle(ROOT / "models" / "risk_bundle.joblib")
    frame = pd.read_parquet(args.input, columns=FEATURES + [TARGET, "FIRE_YEAR", "FOD_ID"])
    frame = frame.loc[frame[TARGET].between(0, 365) & frame.FIRE_YEAR.between(1992, 2024)].copy()
    frame.loc[:, FEATURES] = normalize_features(frame)
    fitted_seed = int(original["metadata"]["seed"])
    fitted_rows = int(original["metadata"]["chronological_splits"]["training"]["used_rows"])
    training = bounded_sample(frame.loc[frame.FIRE_YEAR.between(1992, 2014)], fitted_rows, fitted_seed)
    recent = frame.loc[frame.FIRE_YEAR.between(2020, 2024)].reset_index(drop=True)
    print(f"Predicting fixed models for {len(recent):,} recent records...", flush=True)
    prediction = predict_batch(original, recent)
    raw = prediction["raw_probability_over_3_days"]
    binary = (recent[TARGET].to_numpy() > 3).astype(int)
    fit_mask = recent.FIRE_YEAR.between(2020, 2021).to_numpy()
    select_mask = recent.FIRE_YEAR.eq(2022).to_numpy()
    refit_mask = recent.FIRE_YEAR.between(2020, 2022).to_numpy()
    reassess_mask = recent.FIRE_YEAR.between(2023, 2024).to_numpy()
    assert not np.any(refit_mask & reassess_mask) and not np.any(fit_mask & select_mask)
    platt = LogisticRegression(C=1e6, max_iter=500, random_state=args.seed).fit(logit(raw[fit_mask]), binary[fit_mask])
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw[fit_mask], binary[fit_mask])
    candidates = {"raw": None, "recent_platt": platt, "recent_isotonic": isotonic}
    selection = {method: probability_metrics(binary[select_mask], mapped_probability(raw[select_mask], method, estimator)) for method, estimator in candidates.items()}
    selected = min(selection, key=lambda method: (selection[method]["brier_score"], list(candidates).index(method)))
    print(f"2022-only Brier selects {selected}; refitting its mapping on2020–2022...", flush=True)
    if selected == "recent_platt":
        estimator = LogisticRegression(C=1e6, max_iter=500, random_state=args.seed).fit(logit(raw[refit_mask]), binary[refit_mask])
    elif selected == "recent_isotonic":
        estimator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw[refit_mask], binary[refit_mask])
    else:
        estimator = None
    selected_probability = mapped_probability(raw, selected, estimator)
    probabilities = {"raw": raw, "old_platt": prediction["probability_over_3_days"], "audited_selected": selected_probability}
    print("Building training-only joint-support neighborhoods...", flush=True)
    support_model, reference = fit_support(training, args.seed)
    audited = copy.copy(original)
    audited["probability_update"] = {"method": selected, "estimator": estimator}
    audited["support_model"] = support_model
    support = support_frame(audited, recent)
    print("Auditing recent calibration, intervals, subgroup errors, and support...", flush=True)
    groups = audit_groups(recent, prediction, probabilities, support, reassess_mask)
    reassessment = {}
    target = recent[TARGET].to_numpy()[reassess_mask]
    for method, probability in probabilities.items():
        reassessment[method] = {**probability_metrics(binary[reassess_mask], probability[reassess_mask]),
                               **duration_metrics(target, {key: value[reassess_mask] for key, value in prediction.items() if key in {"p10_days", "p50_days", "p90_days"}}),
                               "risk_median_disagreement_fraction": float(np.mean((probability[reassess_mask] > .5) != (prediction["p50_days"][reassess_mask] > 3 + THRESHOLD_TOLERANCE_DAYS)))}
    # Restrict historical reassessment diagnostics to cases that the UI will
    # actually permit; retain unsupported performance as a separate check.
    support_metrics = {}
    for status in ["supported", "caution", "unsupported"]:
        mask = reassess_mask & support.status.eq(status).to_numpy()
        if mask.sum():
            support_metrics[status] = {**probability_metrics(binary[mask], selected_probability[mask]),
                                      **duration_metrics(recent[TARGET].to_numpy()[mask], {key: value[mask] for key, value in prediction.items() if key in {"p10_days", "p50_days", "p90_days"}})}
    audited["supported_examples"] = choose_examples(audited, reference, args.seed)
    reliable = groups.loc[groups.scope.eq("2023–2024 reassessment") & groups.method.eq("audited_selected")]
    reliable = reliable.astype(object).where(pd.notna(reliable), None)
    audited["audit_reliability"] = {dimension: {row["group"]: {key: value for key, value in row.items() if key not in {"scope", "dimension", "group", "method"}} for row in reliable.loc[reliable.dimension.eq(dimension)].to_dict("records")} for dimension in ["size", "agency"]}
    grid, grid_summary = scenario_grid_support(audited)
    duration_strata = []
    for label, selection_mask in [("Same calendar day", target == 0), ("1–3 calendar days", (target > 0) & (target <= 3)), ("More than3 calendar days", target > 3)]:
        if selection_mask.any():
            duration_strata.append({"stratum": label, "rows": int(selection_mask.sum()), **duration_metrics(target[selection_mask], {key: value[reassess_mask][selection_mask] for key, value in prediction.items() if key in {"p10_days", "p50_days", "p90_days"}})})
    counts = support.loc[reassess_mask, "status"].value_counts().to_dict()
    support_summary = {"definition": support_model["definition"], "reference_rows": support_model["reference_rows"],
                       "threshold_rows": support_model["threshold_rows"], "contexts": len(support_model["contexts"]),
                       "contexts_with_local_thresholds": len(support_model["thresholds"]), "pooled_thresholds": support_model["pooled_thresholds"],
                       "reassessment_rows": int(reassess_mask.sum()), "reassessment_status_counts": {str(key): int(value) for key, value in counts.items()},
                       "reassessment_presentation_allowed_fraction": float(support.loc[reassess_mask, "presentation_allowed"].mean()),
                       "reassessment_metrics_by_support": support_metrics}
    protocol = {"classifier_and_quantile_estimators": "fixed original pre-2015 models; no refitting",
                "calibration_candidate_fit_years": [2020, 2021], "calibration_selection_year": 2022,
                "selection_criterion": "minimum Brier on2022; raw wins exact ties",
                "selected_mapping_refit_years": [2020, 2021, 2022], "reassessment_years": [2023, 2024],
                "reassessment_status": "Previously inspected historical years; reassessment only, not untouched evaluation.",
                "rows": {"candidate_fit": int(fit_mask.sum()), "selection": int(select_mask.sum()), "mapping_refit": int(refit_mask.sum()), "reassessment": int(reassess_mask.sum())},
                "support_thresholds": "Reference and threshold query records are disjoint subsets of the original fitting sample; outcome labels never enter support geometry or threshold selection."}
    checks = {"chronological_calibration_partition": True,
              "fixed_classifier_quantile_estimators": all(joblib.hash(audited[key]) == joblib.hash(original[key]) for key in ["classifier", "calibrator", "preprocessor", "quantile_models"]),
              "support_reference_threshold_queries_disjoint": True,
              "supported_examples_available": bool(audited["supported_examples"]),
              "finite_selected_probabilities": bool(np.isfinite(selected_probability).all()),
              "bounded_selected_probabilities": bool(np.all((selected_probability >= 0) & (selected_probability <= 1)))}
    assert all(checks.values())
    audited["audit_metadata"] = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                                "selection": selection, "reassessment_metrics": reassessment,
                                "support": support_summary, "limitations": AUDIT_LIMITATIONS,
                                "scenario_grid_support": grid_summary, "duration_strata": duration_strata,
                                "checks": checks}
    artifact = ROOT / "models" / "risk_bundle_audited.joblib"
    joblib.dump(audited, artifact, compress=3)
    restored = load_audited_bundle(artifact)
    example = audited["supported_examples"][0]["features"]
    before, after = predict_audited_scenario(audited, example), predict_audited_scenario(restored, example)
    assert before == after and after["support"]["presentation_allowed"]
    bad = dict(example, NWCG_REPORTING_AGENCY="UNSEEN_AUDIT_TEST_AGENCY")
    assert not predict_audited_scenario(restored, bad)["support"]["presentation_allowed"]
    checks["serialized_inference_round_trip"] = True
    checks["unsupported_context_suppresses_presentation"] = True
    # Checks are ordinary metadata; save once more so the artifact and report agree.
    joblib.dump(audited, artifact, compress=3)
    report = {"protocol": protocol, "calibration_selection": {"selected_method": selected, "candidate_2022_metrics": selection},
              "reassessment_metrics": reassessment, "support": support_summary,
              "supported_examples": audited["supported_examples"], "limitations": AUDIT_LIMITATIONS,
              "scenario_grid_support": grid_summary, "duration_strata": duration_strata,
              "checks": checks, "artifact": str(artifact),
              "groups_csv": "model_audit_groups.csv", "calibration_bins_csv": "model_audit_calibration_bins.csv"}
    reports = ROOT / "reports"
    (reports / "model_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    groups.to_csv(reports / "model_audit_groups.csv", index=False)
    bin_rows = []
    for year in sorted(recent.FIRE_YEAR.unique()):
        mask = recent.FIRE_YEAR.eq(year).to_numpy()
        for method, probability in probabilities.items():
            bin_rows.extend({"year": int(year), "method": method, **row} for row in calibration_bins(binary[mask], probability[mask]))
    pd.DataFrame(bin_rows).to_csv(reports / "model_audit_calibration_bins.csv", index=False)
    support_by_group = groups.loc[groups.method.eq("audited_selected"), ["scope", "dimension", "group", "rows", "presentation_allowed_fraction", "supported_fraction"]]
    support_by_group.to_csv(reports / "model_audit_support_coverage.csv", index=False)
    grid.to_csv(reports / "model_audit_scenario_support_grid.csv", index=False)
    pd.DataFrame(duration_strata).to_csv(reports / "model_audit_duration_strata.csv", index=False)
    print(json.dumps({"selected_method": selected, "reassessment_metrics": reassessment,
                      "support": {key: value for key, value in support_summary.items() if key != "reassessment_metrics_by_support"}, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
