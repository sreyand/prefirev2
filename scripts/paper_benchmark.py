"""Controlled retrospective classifier ablations inspired by EO4WildFires.

The cited paper predicts burned-area masks, not containment durations. This
benchmark borrows history, common splits, multi-metric evaluation, and filtered
training/full-evaluation controls. It never replaces the audited main bundle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TARGET = "CONTAINMENT_DAYS_DATE_ONLY"
NUMERIC = ["LOG_ACRES", "LATITUDE", "LONGITUDE", "DAY_OF_YEAR", "OBSERVED_UNCONTAINED_FIRES",
           "vapour_pressure_deficit_max", "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM",
           "DISTANCE_TO_PLACE_CENTROID_KM"]
CATEGORICAL = ["STATE", "NWCG_REPORTING_AGENCY", "NWCG_CAUSE_CLASSIFICATION"]
WEATHER = ["vapour_pressure_deficit_max", "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM"]
HISTORY_SOURCES = {"temperature_2m_max": "TEMP_MAX_C", "vapour_pressure_deficit_max": "VPD_MAX_KPA", "wind_speed_10m_max": "WIND_MAX_KMH"}
HISTORY = [f"PRIOR30_{label}_{stat}" for label in HISTORY_SOURCES.values() for stat in ["MEAN", "PEAK"]]
SPLITS = {"training": (1992, 2014), "calibration": (2015, 2017), "validation": (2018, 2019), "reassessment": (2023, 2024)}
SIZES = ["Micro", "Small", "Medium", "Large", "1000+ acres"]
RULE = {"minimum_relative_brier_improvement": .01, "maximum_ap_degradation": .005,
        "maximum_auc_degradation": .005, "maximum_micro_brier_degradation": .002,
        "paired_day_bootstrap_replicates": 500, "required_brier_difference_ci_upper": 0.0}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def id_hash(frame: pd.DataFrame, sorted_ids=True) -> str:
    ids = frame.FOD_ID.to_numpy(dtype=np.int64)
    if sorted_ids:
        ids = np.sort(ids)
    return hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()


def history_table(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["WEATHER_GRID_ID", "DISCOVERY_DAY"]).copy()
    daily["DISCOVERY_DAY"] = pd.to_datetime(daily.DISCOVERY_DAY).dt.normalize()
    if daily.duplicated(["WEATHER_GRID_ID", "DISCOVERY_DAY"]).any():
        raise ValueError("Duplicate weather grid/day")
    result = []
    for _, group in daily.groupby("WEATHER_GRID_ID", sort=True):
        if not group.DISCOVERY_DAY.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError("Weather history has nonconsecutive days")
        features = group[["WEATHER_GRID_ID", "DISCOVERY_DAY"]].copy()
        for source, label in HISTORY_SOURCES.items():
            shifted = pd.to_numeric(group[source], errors="raise").shift(1)
            features[f"PRIOR30_{label}_MEAN"] = shifted.rolling(30, min_periods=30).mean()
            features[f"PRIOR30_{label}_PEAK"] = shifted.rolling(30, min_periods=30).max()
        result.append(features)
    table = pd.concat(result, ignore_index=True)
    # Initial 30 source days lack a full antecedent window; no fire rows are
    # dropped. Fire merges below must have complete histories for every model.
    table = table.loc[table[HISTORY].notna().all(axis=1)].reset_index(drop=True)
    if not np.isfinite(table[HISTORY]).all().all():
        raise ValueError("Nonfinite history features")
    return table


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in NUMERIC + HISTORY:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("string").fillna("Unknown").str.strip().replace("", "Unknown").astype(object)
    frame["STATE"] = frame.STATE.str.upper()
    return frame


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    y = frame[TARGET].gt(3).to_numpy(dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(probability) != len(frame) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise AssertionError("Invalid probability array")
    result = {"rows": int(len(frame)), "positive_rows": int(y.sum()), "base_rate": float(y.mean()),
              "mean_predicted_probability": float(probability.mean()), "single_class": len(np.unique(y)) < 2}
    if result["single_class"]:
        return result | {"average_precision": None, "roc_auc": None, "brier_score": None,
                         "expected_calibration_error": None, "metric_note": "Classification metrics suppressed for a single-class diagnostic group."}
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if selected.any():
            ece += selected.mean() * abs(float(probability[selected].mean()) - float(y[selected].mean()))
    return result | {"average_precision": float(average_precision_score(y, probability)),
                     "roc_auc": float(roc_auc_score(y, probability)),
                     "brier_score": float(brier_score_loss(y, probability)),
                     "expected_calibration_error": float(ece)}


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def probability(bundle: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = bundle["preprocessor"].transform(frame)
    raw = bundle["classifier"].predict_proba(matrix)[:, 1]
    calibrated = bundle["calibrator"].predict_proba(logit(raw))[:, 1]
    return raw, calibrated


def fit_model(name: str, numeric: list[str], training: pd.DataFrame, calibration: pd.DataFrame,
              seed: int, iterations: int, logistic=False) -> dict:
    if logistic:
        preprocessing = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
        ])
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=seed)
    else:
        preprocessing = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64), CATEGORICAL),
        ], sparse_threshold=0)
        classifier = HistGradientBoostingClassifier(
            max_iter=iterations, learning_rate=.07, max_leaf_nodes=31, min_samples_leaf=80,
            l2_regularization=5, categorical_features=[False] * len(numeric) + [True] * len(CATEGORICAL),
            early_stopping=False, random_state=seed,
        )
    matrix = preprocessing.fit_transform(training)
    classifier.fit(matrix, training[TARGET].gt(3).to_numpy(dtype=int))
    raw_calibration = classifier.predict_proba(preprocessing.transform(calibration))[:, 1]
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500, random_state=seed).fit(
        logit(raw_calibration), calibration[TARGET].gt(3).to_numpy(dtype=int))
    return {"model_id": name, "numeric_features": numeric, "categorical_features": CATEGORICAL,
            "preprocessor": preprocessing, "classifier": classifier, "calibrator": calibrator,
            "training_rows": int(len(training)), "training_id_sha256": id_hash(training),
            "calibration_id_sha256": id_hash(calibration)}


def subgroups(model_id: str, phase: str, frame: pd.DataFrame, probability: np.ndarray) -> tuple[list[dict], list[dict]]:
    rows, duration = [], []
    for size in SIZES:
        mask = frame.SIZE_BIN.eq(size).to_numpy()
        if mask.any():
            rows.append({"model_id": model_id, "phase": phase, "group_type": "final_acreage_bin", "group": size,
                         **metrics(frame.loc[mask], probability[mask])})
    if sum(row["rows"] for row in rows) != len(frame):
        raise AssertionError("Size groups do not partition the evaluation sample")
    for label, mask in [("same_calendar_day", frame[TARGET].eq(0).to_numpy()),
                        ("one_to_three_calendar_days", frame[TARGET].between(1, 3).to_numpy()),
                        ("over_three_calendar_days", frame[TARGET].gt(3).to_numpy())]:
        if mask.any():
            duration.append({"model_id": model_id, "phase": phase, "group": label, "rows": int(mask.sum()),
                             "mean_predicted_probability": float(probability[mask].mean()),
                             "p50_predicted_probability": float(np.quantile(probability[mask], .5)),
                             "p90_predicted_probability": float(np.quantile(probability[mask], .9)),
                             "note": "Outcome-conditioned single-class diagnostic; no classification performance metrics reported."})
    if sum(row["rows"] for row in duration) != len(frame):
        raise AssertionError("Duration diagnostic groups do not partition the sample")
    return rows, duration


def brier_difference_ci(frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float]:
    y = frame[TARGET].gt(3).to_numpy(dtype=float)
    difference = (candidate - y) ** 2 - (baseline - y) ** 2
    table = pd.DataFrame({"day": frame.DISCOVERY_DAY.to_numpy(), "difference": difference})
    grouped = table.groupby("day").difference.agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), (RULE["paired_day_bootstrap_replicates"], len(grouped)))
    estimates = grouped["sum"].to_numpy()[draws].sum(axis=1) / grouped["size"].to_numpy()[draws].sum(axis=1)
    low, high = np.quantile(estimates, [.025, .975])
    return float(low), float(high)


def choose_candidate(validation: pd.DataFrame, results: list[dict], probabilities: dict, seed: int) -> dict:
    baseline_row = next(row for row in results if row["model_id"] == "hgb_original")
    baseline = baseline_row["calibrated_metrics"]
    original_probability = probabilities["hgb_original"]
    micro = validation.SIZE_BIN.eq("Micro").to_numpy()
    y_micro = validation.loc[micro, TARGET].gt(3).to_numpy(dtype=int)
    original_micro_brier = float(brier_score_loss(y_micro, original_probability[micro]))
    considered = []
    for index, row in enumerate(results):
        name = row["model_id"]
        if name in ["hgb_original", "hgb_train_ge1acre"]:
            continue
        values = row["calibrated_metrics"]
        relative = (baseline["brier_score"] - values["brier_score"]) / baseline["brier_score"]
        low, high = brier_difference_ci(validation, probabilities[name], original_probability, seed + index)
        micro_change = float(brier_score_loss(y_micro, probabilities[name][micro])) - original_micro_brier
        passed = relative >= RULE["minimum_relative_brier_improvement"] and high < 0
        passed &= values["average_precision"] >= baseline["average_precision"] - RULE["maximum_ap_degradation"]
        passed &= values["roc_auc"] >= baseline["roc_auc"] - RULE["maximum_auc_degradation"]
        passed &= micro_change <= RULE["maximum_micro_brier_degradation"]
        considered.append({"model_id": name, "passes_validation_screen": bool(passed),
                           "validation_relative_brier_improvement": float(relative),
                           "validation_brier_difference": values["brier_score"] - baseline["brier_score"],
                           "paired_day_bootstrap_brier_difference_ci95": [low, high],
                           "validation_micro_brier_difference": micro_change})
    eligible = [row for row in considered if row["passes_validation_screen"]]
    selected = min(eligible, key=lambda row: row["validation_brier_difference"])["model_id"] if eligible else None
    return {"decided_at_utc": utc(), "evaluation_years": [2018, 2019], "baseline_model_id": "hgb_original",
            "rule": RULE, "selected_model_id": selected, "considered": considered,
            "reassessment_used_for_selection": False,
            "status": "validation_eligible_experimental_candidate" if selected else "no_candidate_passed_validation_screen",
            "interpretation": "Exploratory validation screen, not automatic replacement or production promotion; confidence intervals ignore model-selection multiplicity and cross-day serial dependence."}


def write_json(report: dict) -> None:
    output = ROOT / "reports" / "paper_benchmark.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)


def add_reference_diagnostics(report: dict) -> None:
    """Add fixed-rate checks algebraically; never refit or alter selection."""
    calibration = report["split_manifest"]["calibration"]
    fixed_rate = calibration["positive_rows"] / calibration["used_rows"]
    references = {}
    for phase, key in [("validation", "validation_results"), ("reassessment_2023_2024", "reassessment_results")]:
        if key not in report:
            continue
        observed = report[key][0]["calibrated_metrics"]
        base = observed["base_rate"]
        references[phase] = {
            "model_id": "constant_calibration_rate", "rows": observed["rows"],
            "positive_rows": observed["positive_rows"], "base_rate": base,
            "mean_predicted_probability": float(fixed_rate), "single_class": False,
            "average_precision": float(base), "roc_auc": .5,
            "brier_score": float(base * (1 - fixed_rate) ** 2 + (1 - base) * fixed_rate ** 2),
            "expected_calibration_error": float(abs(base - fixed_rate)),
        }
    report["reference_baselines"] = {
        "definition": "Constant probability fixed to the 2015–2017 calibration-sample positive fraction; no validation/reassessment fitting.",
        "calibration_rate": float(fixed_rate), "metrics_by_phase": references,
    }
    report["metric_definitions"] = {
        "average_precision": "Noninterpolated average precision; prevalence is the constant-score reference, so values across different populations are not directly comparable.",
        "roc_auc": "ROC area; ranking discrimination, independent of a fixed classification threshold.",
        "brier_score": "Mean squared probability error; lower is better. Compare with fixed-rate baseline as well as other fitted models.",
        "expected_calibration_error": "Sample-weighted absolute observed-minus-predicted rate in ten equal-width probability bins. Coarse bins can conceal within-bin errors; this alone does not prove calibration.",
    }
    if "reassessment_results" not in report:
        return
    baseline = next(row for row in report["reassessment_results"] if row["model_id"] == "hgb_original")
    rows = []
    for row in report["reassessment_results"]:
        rows.append({
            "model_id": row["model_id"],
            "calibrated_brier_minus_raw": row["calibrated_metrics"]["brier_score"] - row["raw_metrics"]["brier_score"],
            "calibrated_brier_minus_original": row["calibrated_metrics"]["brier_score"] - baseline["calibrated_metrics"]["brier_score"],
            "calibrated_brier_minus_fixed_rate": row["calibrated_metrics"]["brier_score"] - references["reassessment_2023_2024"]["brier_score"],
        })
    selected = report["selection"]["selected_model_id"]
    selected_diagnostic = next((row for row in rows if row["model_id"] == selected), None)
    report["reassessment_diagnostics"] = {
        "model_comparisons": rows,
        "selected_candidate_result": "no_validation_eligible_candidate" if selected is None else (
            "validation_gain_not_reproduced_in_reassessment" if selected_diagnostic["calibrated_brier_minus_original"] >= 0 else
            "selected_candidate_has_lower_reassessment_brier_exploratory_only"),
        "selection_unchanged": True, "main_bundle_replaced": False,
    }


def export_csv(report: dict) -> None:
    rows = []
    for result in report["validation_results"] + report["reassessment_results"]:
        for version in ["raw", "calibrated"]:
            rows.append({"row_type": "overall_metrics", "model_id": result["model_id"], "phase": result["phase"], "probability_version": version, **result[f"{version}_metrics"]})
    rows += [{"row_type": "size_subgroup_calibrated", **row} for row in report["subgroup_results"]]
    rows += [{"row_type": "outcome_conditioned_diagnostic", **row} for row in report["outcome_conditioned_diagnostics"]]
    rows += [{"row_type": "matched_size_filtered_training_sensitivity", **row} for row in report["filtered_training_matched_size_reassessment"]]
    rows += [{"row_type": "fixed_rate_reference", "phase": phase, **row} for phase, row in report["reference_baselines"]["metrics_by_phase"].items()]
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "paper_benchmark.csv", index=False)


def add_artifact_provenance(report: dict) -> None:
    report["benchmark_artifact_snapshot"] = {
        "audited_bundle_path": "models/risk_bundle_audited.joblib",
        "sha256_during_benchmark": report.get("checks", {}).get("audited_main_sha256"),
        "scope": "Whole-file SHA-256 matched at the start and end of this benchmark execution. It is a historical snapshot, not a claim about the current artifact after later authorized support-gate metadata changes.",
    }
    source_path = ROOT / "models" / "risk_bundle.joblib"
    audited_path = ROOT / "models" / "risk_bundle_audited.joblib"
    if source_path.exists() and audited_path.exists():
        source, audited = joblib.load(source_path), joblib.load(audited_path)
        components = {}
        for name in ["classifier", "calibrator", "quantile_models"]:
            original_hash = joblib.hash(source[name], hash_name="sha1")
            audited_hash = joblib.hash(audited[name], hash_name="sha1")
            components[name] = {"source_joblib_sha1": original_hash, "audited_joblib_sha1": audited_hash,
                                "matches_source_estimator": original_hash == audited_hash}
        report["original_estimator_integrity"] = {
            "checked_at_utc": utc(), "source_artifact": str(source_path), "audited_artifact": str(audited_path),
            "components": components, "all_source_components_match": all(row["matches_source_estimator"] for row in components.values()),
            "scope": "Original probability classifier, old Platt calibrator, and all-fire quantile estimators compared directly with the preserved source artifact. The audited recent calibration, prolonged-duration models, and support-gate objects are separate components.",
        }


def self_test() -> None:
    days = pd.date_range("1992-01-01", periods=45)
    daily = pd.DataFrame({"WEATHER_GRID_ID": 0, "DISCOVERY_DAY": days,
                          **{source: np.arange(1, 46, dtype=float) for source in HISTORY_SOURCES}})
    history = history_table(daily)
    row = history.loc[history.DISCOVERY_DAY.eq(days[30])].iloc[0]
    assert row.PRIOR30_TEMP_MAX_C_MEAN == 15.5 and row.PRIOR30_TEMP_MAX_C_PEAK == 30
    changed = daily.copy()
    changed.loc[30, list(HISTORY_SOURCES)] = 9999
    assert history.loc[history.DISCOVERY_DAY.eq(days[30]), HISTORY].equals(
        history_table(changed).loc[lambda f: f.DISCOVERY_DAY.eq(days[30]), HISTORY])
    two = pd.DataFrame({TARGET: [0, 5]})
    result = metrics(two, np.array([0., 1.]))
    assert result["brier_score"] == 0 and result["average_precision"] == 1 and result["roc_auc"] == 1
    assert metrics(two.iloc[[0]], np.array([.2]))["average_precision"] is None
    reference_check = {"split_manifest": {"calibration": {"positive_rows": 1, "used_rows": 2}},
                       "validation_results": [{"calibrated_metrics": result}]}
    add_reference_diagnostics(reference_check)
    assert reference_check["reference_baselines"]["metrics_by_phase"]["validation"]["brier_score"] == .25
    ids = pd.DataFrame({"FOD_ID": [3, 1, 2]})
    assert id_hash(ids) == id_hash(ids.iloc[::-1]) and id_hash(ids, False) != id_hash(ids.iloc[::-1], False)
    print("PASS: strict prior-history window; discovery-day exclusion; common ID hashing; finite multi-metrics; single-class suppression.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--skip-filter-sensitivity", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.iterations <= 120 or not 1000 <= args.max_rows <= 100000:
        parser.error("Use 1–120 iterations and 1,000–100,000 rows per split")
    main_artifact = ROOT / "models" / "risk_bundle_audited.joblib"
    initial_main_hash = file_hash(main_artifact)
    columns = list(dict.fromkeys(NUMERIC + CATEGORICAL + [TARGET, "FOD_ID", "FIRE_YEAR", "FIRE_SIZE", "SIZE_BIN", "DISCOVERY_DAY", "WEATHER_GRID_ID"]))
    frame = pd.read_parquet(ROOT / "data" / "processed_fires.parquet", columns=columns)
    if frame.FOD_ID.duplicated().any() or not frame[TARGET].between(0, 365).all() or not np.equal(frame[TARGET], np.floor(frame[TARGET])).all():
        raise ValueError("Invalid IDs or calendar-day target")
    daily = pd.read_parquet(ROOT / "data" / "era5_daily.parquet", columns=["WEATHER_GRID_ID", "DISCOVERY_DAY"] + list(HISTORY_SOURCES))
    history = history_table(daily)
    history_path = ROOT / "data" / "paper_weather_features.parquet"
    history.to_parquet(history_path, index=False)
    splits, manifest = {}, {}
    for index, (name, years) in enumerate(SPLITS.items()):
        eligible = frame.loc[frame.FIRE_YEAR.between(*years)]
        selected = eligible.sample(n=args.max_rows, random_state=args.seed + index) if len(eligible) > args.max_rows else eligible.copy()
        original_ids = selected.FOD_ID.to_numpy().copy()
        selected = selected.merge(history, on=["WEATHER_GRID_ID", "DISCOVERY_DAY"], how="left", sort=False, validate="many_to_one")
        if len(selected) != min(args.max_rows, len(eligible)) or not np.array_equal(selected.FOD_ID.to_numpy(), original_ids):
            raise AssertionError("Weather-history merge changed event IDs or order")
        if selected[HISTORY].isna().any().any():
            raise AssertionError("Incomplete history would make ablations compare different fires")
        selected = clean(selected)
        splits[name] = selected
        manifest[name] = {"years": list(years), "eligible_rows": int(len(eligible)), "used_rows": int(len(selected)),
                          "id_sha256_sorted_int64_le": id_hash(selected), "id_sha256_fit_order_int64_le": id_hash(selected, False)}
        if name != "reassessment":
            manifest[name].update(positive_rows=int(selected[TARGET].gt(3).sum()), same_calendar_day_fraction=float(selected[TARGET].eq(0).mean()))
        else:
            manifest[name]["outcome_metrics_deferred_until_selection_locked"] = True
    for i, a in enumerate(splits):
        for b in list(splits)[i + 1:]:
            if np.intersect1d(splits[a].FOD_ID, splits[b].FOD_ID).size:
                raise AssertionError("Temporal split IDs overlap")
    training, calibration, validation = splits["training"], splits["calibration"], splits["validation"]
    configurations = [
        ("hgb_original", NUMERIC, False, False),
        ("hgb_without_weather", [f for f in NUMERIC if f not in WEATHER], False, False),
        ("hgb_without_observed_load", [f for f in NUMERIC if f != "OBSERVED_UNCONTAINED_FIRES"], False, False),
        ("hgb_with_prior30_history", NUMERIC + HISTORY, False, False),
        ("logistic_original", NUMERIC, True, False),
    ]
    if not args.skip_filter_sensitivity:
        configurations.append(("hgb_train_ge1acre", NUMERIC, False, True))
    bundles, validation_results, validation_probabilities, subgroup_rows, duration_rows = {}, [], {}, [], []
    training_rows = []
    for name, numeric, logistic, filtered in configurations:
        subset = training.loc[training.FIRE_SIZE.ge(1)] if filtered else training
        started = time.monotonic()
        print(f"Fitting {name}: {len(subset):,} rows, {len(numeric) + len(CATEGORICAL)} features...", flush=True)
        bundle = fit_model(name, list(numeric), subset, calibration, args.seed, args.iterations, logistic)
        bundles[name] = bundle
        raw, calibrated = probability(bundle, validation)
        validation_results.append({"model_id": name, "phase": "validation", "raw_metrics": metrics(validation, raw),
                                   "calibrated_metrics": metrics(validation, calibrated)})
        validation_probabilities[name] = calibrated
        group, diagnostic = subgroups(name, "validation", validation, calibrated)
        subgroup_rows.extend(group)
        duration_rows.extend(diagnostic)
        training_rows.append({"model_id": name, "training_rows": len(subset), "training_id_sha256": bundle["training_id_sha256"],
                              "calibration_id_sha256": bundle["calibration_id_sha256"], "features": numeric + CATEGORICAL,
                              "filtered_training": filtered, "fit_elapsed_seconds": float(time.monotonic() - started),
                              "iterations": int(bundle["classifier"].n_iter_) if not logistic else int(bundle["classifier"].n_iter_[0])})
        print(f"Validation {name}: Brier={validation_results[-1]['calibrated_metrics']['brier_score']:.6f}, AP={validation_results[-1]['calibrated_metrics']['average_precision']:.4f}", flush=True)
    selection = choose_candidate(validation, validation_results, validation_probabilities, args.seed)
    selection_digest = hashlib.sha256(json.dumps(selection, sort_keys=True, allow_nan=False).encode()).hexdigest()
    candidate_artifact = None
    if selection["selected_model_id"]:
        candidate = bundles[selection["selected_model_id"]]
        candidate.update({"artifact_kind": "experimental_retrospective_classifier_only", "target": TARGET,
                          "threshold": "more than three calendar days", "split_manifest": manifest,
                          "selection": selection, "selection_sha256": selection_digest,
                          "sklearn_version": sklearn.__version__, "created_utc": utc(),
                          "not_main_bundle_replacement": True})
        candidate_path = ROOT / "models" / "paper_candidate.joblib"
        joblib.dump(candidate, candidate_path, compress=3)
        restored = joblib.load(candidate_path)
        if not np.allclose(probability(candidate, validation.head(20))[1], probability(restored, validation.head(20))[1]):
            raise AssertionError("Candidate serialization changed probabilities")
        candidate_artifact = {"path": str(candidate_path), "model_id": selection["selected_model_id"],
                              "sha256": file_hash(candidate_path), "saved_before_reassessment": True,
                              "main_bundle_replaced": False, "classifier_only": True}
    report = {
        "schema_version": 1, "status": "validation_locked_reassessment_pending", "generated_utc": utc(),
        "paper_context": {"title": "Deep Learning Approaches for Wildfire Severity Prediction: A Comparative Study of Image Segmentation Networks and Visual Transformers on the EO4WildFires Dataset",
                          "published_url": "https://www.mdpi.com/2571-6255/7/11/374", "primary_preprint_url": "https://www.preprints.org/manuscript/202409.0702",
                          "application": "Transfers common-split, antecedent-history, multi-metric, and filtered-training/full-evaluation controls; does not reproduce burned-area segmentation or establish containment validity."},
        "outcome": {"field": TARGET, "definition": "Containment calendar date more than three days after discovery calendar date; same-day cases remain valid negative examples."},
        "split_manifest": manifest, "training_runs": training_rows,
        "feature_groups": {"original_numeric": NUMERIC, "categorical": CATEGORICAL, "original_weather": WEATHER,
                           "observed_load": ["OBSERVED_UNCONTAINED_FIRES"], "prior30_history": HISTORY},
        "history_provenance": {"source": "data/era5_daily.parquet", "output": str(history_path), "keys": ["WEATHER_GRID_ID", "DISCOVERY_DAY"],
                               "rows": int(len(history)), "source_sha256": file_hash(ROOT / "data" / "era5_daily.parquet"),
                               "window": "Discovery date minus30 through minus1, excluding discovery date; complete consecutive calendar days required.",
                               "aggregation": "Mean and maximum of daily maximum temperature, VPD, and wind; mean daily maximum is not mean hourly temperature/VPD/wind.",
                               "zero_fire_rows_dropped_for_history": True, "weather_centroids": 25,
                               "spatial_limitation": "Existing centroid assignments have median166km and P95348km separation; history does not improve spatial resolution."},
        "fixed_training_config": {"seed": args.seed, "maximum_rows_per_split": args.max_rows, "hgb_iterations": args.iterations,
                                  "hgb_learning_rate": .07, "hgb_max_leaves": 31, "hgb_min_leaf_rows": 80, "hgb_l2": 5,
                                  "hgb_early_stopping": False, "logistic": "Standardized numeric + one-hot categorical, C=1, lbfgs, maximum500 iterations",
                                  "calibration": "Each model receives Platt logistic adjustment of clipped raw logits, fitted only on identical2015–2017 rows; raw metrics also reported."},
        "validation_results": validation_results, "selection": selection, "selection_sha256": selection_digest,
        "candidate_artifact": candidate_artifact,
        "caveats": [
            "This is an exploratory repeated benchmark: validation and2023–2024 outcomes were already inspected in previous work; reassessment is not an untouched confirmatory future test.",
            "Model selection is frozen from2018–2019 before computing2023–2024 benchmark metrics; reassessment never changes that selection.",
            "Feature ablations measure conditional predictive contribution in this setup, not causal weather/load effects or suppression-capacity attribution.",
            "Final acreage and observed concurrency are retrospective and outcome-related; daily maximum weather includes conditions after discovery. These are not dispatch-time forecasts.",
            "Prior30 history excludes discovery-day weather but uses the same broad regional centroids; local-day alignment remains approximate because source clock timezones are unspecified.",
            "Reporting completeness changes sharply by agency/year; the selected containment cohort is not representative by default. Forest Service is not a stable coverage control.",
            "Filtering training to>=1acre is a distinct sensitivity, not removal of empty segmentation labels; same-day cases are retained in every fitting regime.",
            "Outcome-conditioned same-day/prolonged diagnostics are single-class; no binary classification performance metrics are reported for them.",
            "The separate candidate artifact contains only a probability classifier and requires a separate adapter; it does not replace main quantile/duration models.",
            "Paired discovery-day bootstrap intervals omit across-day dependence and selection multiplicity; candidate eligibility does not establish reproducible future improvement.",
        ],
    }
    write_json(report)
    print(f"Selection frozen before reassessment: {selection['selected_model_id']}", flush=True)
    reassessment = splits["reassessment"]
    reassessment_started = utc()
    reassessment_results, matched_size = [], []
    for name, bundle in bundles.items():
        raw, calibrated = probability(bundle, reassessment)
        reassessment_results.append({"model_id": name, "phase": "reassessment_2023_2024", "raw_metrics": metrics(reassessment, raw),
                                     "calibrated_metrics": metrics(reassessment, calibrated)})
        group, diagnostic = subgroups(name, "reassessment_2023_2024", reassessment, calibrated)
        subgroup_rows.extend(group)
        duration_rows.extend(diagnostic)
        if name in ["hgb_original", "hgb_train_ge1acre"]:
            selected = reassessment.FIRE_SIZE.ge(1).to_numpy()
            matched_size.append({"model_id": name, "evaluation_scope": "reassessment_only_final_acres_ge1_not_full_population", **metrics(reassessment.loc[selected], calibrated[selected])})
    if hashlib.sha256(json.dumps(selection, sort_keys=True, allow_nan=False).encode()).hexdigest() != selection_digest:
        raise AssertionError("Selection changed after reassessment")
    if file_hash(main_artifact) != initial_main_hash:
        raise AssertionError("Audited main artifact changed during benchmark")
    report.update(status="complete", completed_utc=utc(), reassessment_started_utc=reassessment_started,
                  reassessment_results=reassessment_results, subgroup_results=subgroup_rows,
                  outcome_conditioned_diagnostics=duration_rows, filtered_training_matched_size_reassessment=matched_size,
                  checks={"history_discovery_day_excluded": True, "common_split_ids_and_order": True,
                          "temporal_IDs_disjoint": True, "no_same_day_training_exclusion": True,
                          "audited_main_bundle_unchanged": True, "audited_main_sha256": initial_main_hash,
                          "selection_frozen_before_reassessment": True, "finite_bounded_probabilities": True,
                          "size_and_duration_group_partitions": True, "single_class_metrics_suppressed": True})
    add_reference_diagnostics(report)
    add_artifact_provenance(report)
    write_json(report)
    export_csv(report)
    print(json.dumps({"status": "complete", "selected_model_id": selection["selected_model_id"], "candidate_artifact": candidate_artifact,
                      "reassessment_metrics": [{"model_id": row["model_id"], **row["calibrated_metrics"]} for row in reassessment_results]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
