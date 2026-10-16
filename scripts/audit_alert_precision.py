"""Fixed-threshold historical precision/recall audit; no estimator fitting.

The assessed years have already been inspected. These are retrospective flags
using eventual fire size and reconstructed load, not operational early alerts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_model_support import load_audited_bundle, mapped_probability, support_frame
from scripts.train_risk import FEATURES, TARGET, normalize_features


THRESHOLDS = [.05, .10, .20, .30, .50, .70]
SIZES = ["Micro", "Small", "Medium", "Large", "1000+ acres"]
MAX_CURVE_POINTS = 500


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def thresholds(y: np.ndarray, probability: np.ndarray) -> list[dict]:
    y = np.asarray(y, dtype=bool)
    rows = []
    for threshold in THRESHOLDS:
        flag = probability >= threshold
        tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
        fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
        alerts, events = tp + fp, tp + fn
        if tp + fp + fn + tn != len(y):
            raise AssertionError("Confusion-matrix cells do not partition the scope")
        rows.append({"threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": float(tp / alerts) if alerts else None,
                     "recall": float(tp / events) if events else None,
                     "alert_count": alerts, "alert_rate": float(alerts / len(y)) if len(y) else None,
                     "false_alerts_per_100_alerts": float(100 * fp / alerts) if alerts else None})
    if any(rows[index]["alert_count"] < rows[index + 1]["alert_count"] for index in range(len(rows) - 1)):
        raise AssertionError("Alert counts increase with a stricter threshold")
    return rows


def curve(y: np.ndarray, probability: np.ndarray) -> list[dict]:
    precision, recall, cutoff = precision_recall_curve(y, probability)
    cutoff = np.r_[cutoff, np.nan]
    indices = np.unique(np.linspace(0, len(precision) - 1, min(len(precision), MAX_CURVE_POINTS), dtype=int))
    return [{"precision": float(precision[index]), "recall": float(recall[index]),
             "threshold": float(cutoff[index]) if np.isfinite(cutoff[index]) else None} for index in indices]


def summarize(frame: pd.DataFrame, probability: np.ndarray, raw_probability: np.ndarray) -> dict:
    y = frame[TARGET].gt(3).to_numpy(dtype=int)
    if not len(frame):
        raise ValueError("An alert-audit scope is empty")
    if not np.isfinite(probability).all() or not np.isfinite(raw_probability).all():
        raise AssertionError("Nonfinite probabilities")
    if ((probability < 0) | (probability > 1)).any() or ((raw_probability < 0) | (raw_probability > 1)).any():
        raise AssertionError("Probabilities outside [0,1]")
    both_classes = len(np.unique(y)) == 2
    return {"rows": int(len(frame)), "events_over_3_days": int(y.sum()), "prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, probability)) if both_classes else None,
            "raw_average_precision": float(average_precision_score(y, raw_probability)) if both_classes else None,
            "mean_predicted_probability": float(probability.mean()),
            "threshold_results": thresholds(y, probability),
            "precision_recall_curve": curve(y, probability) if both_classes else [],
            "raw_precision_recall_curve": curve(y, raw_probability) if both_classes else [],
            "single_class_scope": not both_classes}


def population(frame: pd.DataFrame, probability: np.ndarray, raw_probability: np.ndarray) -> dict:
    years, sizes = [], []
    for year in [2023, 2024]:
        selected = frame.FIRE_YEAR.eq(year).to_numpy()
        years.append({"year": year, **summarize(frame.loc[selected], probability[selected], raw_probability[selected])})
    for size in SIZES:
        selected = frame.SIZE_BIN.eq(size).to_numpy()
        if selected.any():
            sizes.append({"size_bin": size, **summarize(frame.loc[selected], probability[selected], raw_probability[selected])})
    if sum(row["rows"] for row in years) != len(frame) or sum(row["rows"] for row in sizes) != len(frame):
        raise AssertionError("Year/size summaries do not partition their population")
    return {"overall": summarize(frame, probability, raw_probability), "by_year": years, "by_size": sizes}


def self_test() -> None:
    y = np.array([0, 0, 1, 1])
    rows = thresholds(y, np.array([.01, .1, .2, .8]))
    at_point_one = next(row for row in rows if row["threshold"] == .1)
    assert [at_point_one[key] for key in ["TP", "FP", "FN", "TN"]] == [2, 1, 0, 1]
    assert np.isclose(at_point_one["precision"], 2 / 3) and at_point_one["recall"] == 1
    no_alerts = thresholds(y, np.zeros(4))
    assert all(row["precision"] is None and row["false_alerts_per_100_alerts"] is None and row["recall"] == 0 for row in no_alerts)
    points = curve(y, np.array([.01, .1, .2, .8]))
    assert len(points) <= MAX_CURVE_POINTS and points[-1]["recall"] == 0 and points[-1]["precision"] == 1
    print("PASS: threshold inclusivity; exact confusion counts; zero-alert handling; bounded PR curve.", flush=True)


def main() -> None:
    self_test()
    # This declaration is saved with its timestamp before loading/evaluating
    # the reassessment data; it never proposes an optimized operating threshold.
    protocol = {"declared_at_utc": utc(), "fixed_probability_thresholds": THRESHOLDS,
                "flag_rule": "mapped probability >= threshold (inclusive)",
                "threshold_optimization": False, "recommended_operating_threshold": None,
                "years": [2023, 2024], "sampling": "All usable records in the fixed historical reassessment years; no sampling.",
                "max_curve_points": MAX_CURVE_POINTS,
                "curve_thinning": "Evenly spaced full-curve indices, retaining endpoints; AP is computed exactly from all scores before thinning."}
    bundle_path = ROOT / "models" / "risk_bundle_audited.joblib"
    with bundle_path.open("rb") as handle:
        before = hashlib.file_digest(handle, "sha256").hexdigest()
    bundle = load_audited_bundle(bundle_path)
    columns = list(dict.fromkeys(FEATURES + [TARGET, "FOD_ID", "FIRE_YEAR", "SIZE_BIN"]))
    frame = pd.read_parquet(ROOT / "data" / "processed_fires.parquet", columns=columns,
                            filters=[("FIRE_YEAR", ">=", 2023), ("FIRE_YEAR", "<=", 2024)])
    if len(frame) != 128797 or frame.FOD_ID.duplicated().any() or not frame[TARGET].between(0, 365).all():
        raise AssertionError("Historical reassessment cohort changed or contains invalid records")
    clean = normalize_features(frame)
    matrix = bundle["preprocessor"].transform(clean)
    raw_probability = bundle["classifier"].predict_proba(matrix)[:, 1]
    mapping = bundle["probability_update"]
    probability = mapped_probability(raw_probability, mapping["method"], mapping["estimator"])
    support = support_frame(bundle, clean)
    allowed = support.presentation_allowed.to_numpy(dtype=bool)
    populations = {"all_records": population(frame, probability, raw_probability),
                   "presentation_allowed": population(frame.loc[allowed], probability[allowed], raw_probability[allowed])}
    report = {
        "schema_version": 1, "status": "complete", "generated_utc": utc(), "protocol": protocol,
        "target": {"field": TARGET, "positive_definition": "Containment calendar date is more than three days after discovery calendar date.",
                   "precision_definition": "Among flagged fires, the fraction whose recorded outcome exceeds three calendar days.",
                   "recall_definition": "Among fires recorded as exceeding three calendar days, the fraction flagged.",
                   "average_precision_definition": "Threshold-independent summary over the full precision-recall ranking; AP is not the precision at one operating threshold."},
        "model_provenance": {"path": str(bundle_path), "sha256_snapshot": before, "mapping": mapping["method"],
                             "classifier_joblib_sha1": joblib.hash(bundle["classifier"], hash_name="sha1"),
                             "no_refit_or_model_change": True,
                             "row_id_sha256_sorted_int64_le": hashlib.sha256(np.sort(frame.FOD_ID.to_numpy(dtype=np.int64)).astype("<i8").tobytes()).hexdigest()},
        "populations": populations,
        "support_selection": {"definition": "Feature-only all-fire presentation gate; supported and caution cases included, unsupported cases excluded. This is not the separate conditional prolonged-duration gate.",
                              "all_rows": int(len(frame)), "presentation_allowed_rows": int(allowed.sum()),
                              "excluded_rows": int((~allowed).sum()),
                              "status_counts": {str(key): int(value) for key, value in support.status.value_counts().items()}},
        "caveats": [
            "Historical reassessment of previously inspected 2023–2024 outcomes; not an untouched future or prospective operational validation.",
            "These are hypothetical retrospective flags: eventual acreage and reconstructed load are unavailable at discovery, so the displayed precision does not establish operational alert performance.",
            "Fixed thresholds were declared before this audit's calculations and are not optimized or recommended from these reassessment outcomes.",
            "Changing the probability threshold trades false flags against missed prolonged outcomes; high precision at a strict threshold can come with low recall and few flags.",
            "Isotonic calibration creates score ties; threshold inclusivity is explicit and AP can differ from the raw classifier even when ranking changes only through ties.",
            "The presentation-allowed subset changes the evaluated population and prevalence; its scores cannot be interpreted as an improvement for all fires or as an outcome-independent validation sample.",
            "Reporting completeness, coarse centroid weather, era drift, and uncertain source clock timezones remain unresolved. Calendar-date thresholds are not exact 72-hour containment measures.",
            "PR curves are thinned only for display; exact average precision uses all probabilities. Single-class scopes omit ranking metrics.",
        ],
        "checks": {"self_test": True, "fixed_128797_row_cohort": True, "confusion_counts_partition_each_scope": True,
                   "year_and_size_partitions": True, "zero_alert_metrics_null": True, "finite_bounded_probabilities": True,
                   "AP_computed_before_curve_thinning": True, "thresholds_not_optimized": True},
    }
    with bundle_path.open("rb") as handle:
        after = hashlib.file_digest(handle, "sha256").hexdigest()
    if before != after:
        # A concurrent authorized update of the separate prolonged-duration
        # support gate need not invalidate a fixed probability/support audit.
        current = load_audited_bundle(bundle_path)
        for component in ["classifier", "preprocessor", "probability_update", "support_model"]:
            if joblib.hash(current[component], hash_name="sha1") != joblib.hash(bundle[component], hash_name="sha1"):
                raise AssertionError(f"Relevant audited component changed during alert audit: {component}")
    report["model_provenance"]["whole_bundle_sha256_at_end"] = after
    report["model_provenance"]["snapshot_scope"] = "Probability classifier/mapping and feature-only all-fire support; separate conditional-duration support is not used."
    report["checks"]["relevant_probability_and_support_components_unchanged"] = True
    json.dumps(report, allow_nan=False)
    output = ROOT / "reports" / "alert_precision.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    rows = []
    for name, item in populations.items():
        scopes = [("overall", "all", item["overall"])]
        scopes += [("year", str(row["year"]), row) for row in item["by_year"]]
        scopes += [("size_bin", row["size_bin"], row) for row in item["by_size"]]
        for dimension, label, scope in scopes:
            for row in scope["threshold_results"]:
                rows.append({"population": name, "dimension": dimension, "group": label,
                             "rows": scope["rows"], "events_over_3_days": scope["events_over_3_days"],
                             "prevalence": scope["prevalence"], "average_precision": scope["average_precision"], **row})
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps({"status": "complete", "all_records": {key: value for key, value in populations["all_records"]["overall"].items() if "curve" not in key},
                      "presentation_allowed_rows": int(allowed.sum()), "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
