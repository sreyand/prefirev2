"""Fit a separate conditional prolonged-duration model and finite-rank CQR.

Existing classifier, calibration mapping, support gate, and unconditional
quantiles remain fixed. This model answers: if calendar containment duration
exceeds3days, what duration range does the historical conditional model imply?
"""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from scripts.audit_model_support import load_audited_bundle, predict_conditional_prolonged, predict_audited_scenario, size_group, support_frame, prolonged_support_frame, fit_support, support_dict
    from scripts.train_risk import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, normalize_features
except ModuleNotFoundError:
    from audit_model_support import load_audited_bundle, predict_conditional_prolonged, predict_audited_scenario, size_group, support_frame, prolonged_support_frame, fit_support, support_dict
    from train_risk import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, normalize_features

ROOT = Path(__file__).resolve().parents[1]


def conditional_metrics(target, prediction, baseline_median, baseline_quantiles):
    lower, median, upper = (prediction[key] for key in ["lower_days", "median_days", "upper_days"])
    return {"rows": int(len(target)), "median_mae_days": float(np.mean(np.abs(target - median))),
            "baseline_median_mae_days": float(np.mean(np.abs(target - baseline_median))),
            "interval_coverage": float(np.mean((target >= lower) & (target <= upper))),
            "native_interval_coverage": float(np.mean((target >= prediction["native_lower_days"]) & (target <= prediction["native_upper_days"]))),
            "mean_interval_width_days": float(np.mean(upper - lower)),
            "median_interval_width_days": float(np.median(upper - lower)),
            "fraction_below_lower": float(np.mean(target < lower)), "fraction_above_upper": float(np.mean(target > upper)),
            "constant_training_interval_coverage": float(np.mean((target >= baseline_quantiles[0]) & (target <= baseline_quantiles[1]))),
            "constant_training_interval_width_days": float(baseline_quantiles[1] - baseline_quantiles[0])}


def attach_prolonged_support(bundle, training, reassessment):
    """Add conditional-training geometry without refitting any prediction model."""
    conditional = bundle["prolonged_duration"]
    before = joblib.hash((conditional["quantile_models"], conditional["conformal_expansion_days"]))
    reference_rows = int(np.floor(.8 * len(training)))
    print(f"Building separate prolonged-training support: {reference_rows:,} reference records and disjoint remaining records...", flush=True)
    model, reference = fit_support(training, 42, reference_rows=reference_rows, threshold_rows=len(training) - reference_rows)
    model["definition"] = "PROLONGED-TRAINING COHORT ONLY. " + model["definition"]
    conditional["support_model"] = model
    conditional["training_bounds"] = {feature: {"min": float(training[feature].min()), "max": float(training[feature].max())} for feature in NUMERIC_FEATURES}
    base = support_frame(bundle, reassessment)
    branch = prolonged_support_frame(bundle, reassessment)
    allowed = base.presentation_allowed.to_numpy() & branch.presentation_allowed.to_numpy()
    prediction = predict_conditional_prolonged(bundle, reassessment)
    target = reassessment[TARGET].to_numpy(dtype=float)
    meta = conditional["metadata"]
    baseline_median = meta["baseline_training_median_days"]
    baseline_quantiles = meta["baseline_training_p10_p90_days"]
    unrestricted = conditional_metrics(target, prediction, baseline_median, baseline_quantiles)
    allowed_metrics = conditional_metrics(target[allowed], {key: value[allowed] for key, value in prediction.items()}, baseline_median, baseline_quantiles) if allowed.any() else None
    suppressed_metrics = conditional_metrics(target[~allowed], {key: value[~allowed] for key, value in prediction.items()}, baseline_median, baseline_quantiles) if (~allowed).any() else None
    # Coverage among all recent records is descriptive only; no outcome label
    # enters the geometry or its thresholds. Bound the query workload.
    all_recent = pd.read_parquet(ROOT / "data" / "processed_fires.parquet", columns=FEATURES + ["FOD_ID", "FIRE_YEAR"], filters=[("FIRE_YEAR", ">=", 2023), ("FIRE_YEAR", "<=", 2024)])
    eligible_recent_rows = len(all_recent)
    if len(all_recent) > 50_000:
        all_recent = all_recent.sample(n=50_000, random_state=42)
    all_base = support_frame(bundle, all_recent)
    all_branch = prolonged_support_frame(bundle, all_recent)
    all_allowed = all_base.presentation_allowed.to_numpy() & all_branch.presentation_allowed.to_numpy()
    summary = {"definition": model["definition"], "training_rows": int(len(training)), "reference_rows": model["reference_rows"],
               "threshold_rows": model["threshold_rows"], "contexts": len(model["contexts"]), "contexts_with_local_thresholds": len(model["thresholds"]),
               "minimum_context_reference_records": 30, "k": 10,
               "cohort_membership": "Conditional training consists of historical duration>3day records; inference uses only features and context, never the scenario's outcome.",
               "presentation_rule": "Probability requires all-fire support. Conditional duration requires BOTH all-fire and separate prolonged-training support.",
               "prolonged_reassessment_rows": int(len(reassessment)), "prolonged_reassessment_branch_status_counts": {str(key): int(value) for key,value in branch.status.value_counts().items()},
               "prolonged_reassessment_both_gates_allowed_fraction": float(allowed.mean()),
               "unrestricted_prolonged_reassessment_metrics": unrestricted,
               "both_gates_allowed_prolonged_reassessment_metrics": allowed_metrics,
               "suppressed_prolonged_reassessment_metrics": suppressed_metrics,
               "all_recent_feature_query": {"eligible_rows": eligible_recent_rows, "queried_rows": len(all_recent), "sampled": bool(eligible_recent_rows > len(all_recent)), "seed": 42,
                                            "conditional_branch_status_counts": {str(key): int(value) for key,value in all_branch.status.value_counts().items()},
                                            "both_gates_allowed_fraction": float(all_allowed.mean())}}
    # Keep existing recorded examples and put a case supported by both branches
    # first. Selection uses feature-space support, not containment outcomes.
    for example in bundle["supported_examples"]:
        assessment = predict_audited_scenario(bundle, example["features"])
        example["conditional_support"] = assessment["conditional_prolonged_duration"]["support"]
        example["conditional_presentation_allowed"] = assessment["conditional_prolonged_duration"]["presentation_allowed"]
    bundle["supported_examples"].sort(key=lambda example: not example["conditional_presentation_allowed"])
    if not bundle["supported_examples"][0]["conditional_presentation_allowed"]:
        candidates = reference.sample(n=min(2000,len(reference)),random_state=42)
        candidate_base = support_frame(bundle,candidates)
        candidate_branch = prolonged_support_frame(bundle,candidates)
        eligible = candidate_base.presentation_allowed.to_numpy() & candidate_branch.presentation_allowed.to_numpy()
        if not eligible.any():
            raise ValueError("No recorded example is supported by both branches.")
        position = int(np.flatnonzero(eligible)[0]); row = candidates.iloc[position]
        features = {feature: float(row[feature]) if feature in NUMERIC_FEATURES else str(row[feature]) for feature in FEATURES}
        bundle["supported_examples"].insert(0,{"label": f"Recorded context supported by both models: {row.STATE}, {row.NWCG_REPORTING_AGENCY}, {np.exp(row.LOG_ACRES):,.2g} acres",
                                             "features":features,"source_year":int(row.FIRE_YEAR),"source_fod_id":int(row.FOD_ID),
                                             "support":support_dict(candidate_base.iloc[position]),"conditional_support":support_dict(candidate_branch.iloc[position]),"conditional_presentation_allowed":True})
    regression = pd.read_parquet(ROOT / "data" / "processed_fires.parquet", columns=FEATURES + ["FOD_ID"], filters=[("FOD_ID", "==", 3455)])
    if len(regression) != 1:
        raise ValueError("Conditional-support regression record FOD3455 missing or duplicated.")
    check_case = predict_audited_scenario(bundle, regression.iloc[0].to_dict())
    assert check_case["support"]["presentation_allowed"], "Regression case must retain all-fire probability support."
    assert not check_case["conditional_prolonged_duration"]["presentation_allowed"]
    assert check_case["conditional_prolonged_duration"]["support"]["context_reference_count"] == 0
    assert joblib.hash((conditional["quantile_models"],conditional["conformal_expansion_days"])) == before
    summary["checks"] = {"separate_reference_and_threshold_records":True,"feature_only_inference_gate":True,
                           "conditional_prediction_models_and_expansion_unchanged":True,
                           "IN_FS_Human_zero_prolonged_context_suppressed":True,"IN_FS_Human_probability_retains_support":True,
                           "default_example_supported_by_both_models":bool(bundle["supported_examples"][0]["conditional_presentation_allowed"])}
    meta["support_gate"] = summary
    bundle["audit_metadata"]["conditional_prolonged_duration"] = meta
    return summary


def save_support_only(bundle, artifact, training, reassessment):
    summary = attach_prolonged_support(bundle, training, reassessment)
    temporary = artifact.with_suffix(".pending")
    joblib.dump(bundle,temporary,compress=3);os.replace(temporary,artifact)
    restored = load_audited_bundle(artifact)
    assert predict_audited_scenario(restored,restored["supported_examples"][0]["features"])["conditional_prolonged_duration"]["presentation_allowed"]
    report_path = ROOT / "reports" / "model_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["conditional_prolonged_duration"] = bundle["prolonged_duration"]["metadata"]
    report["supported_examples"] = bundle["supported_examples"]
    report["checks"]["separate_prolonged_training_support_gate"] = True
    temporary_report = report_path.with_suffix(".pending")
    temporary_report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8");os.replace(temporary_report,report_path)
    rows = [{"scope":"all prolonged reassessment",**summary["unrestricted_prolonged_reassessment_metrics"]}]
    if summary["both_gates_allowed_prolonged_reassessment_metrics"]:
        rows.append({"scope":"both support gates allow presentation",**summary["both_gates_allowed_prolonged_reassessment_metrics"]})
    if summary["suppressed_prolonged_reassessment_metrics"]:
        rows.append({"scope":"one or both gates suppress presentation",**summary["suppressed_prolonged_reassessment_metrics"]})
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "model_audit_prolonged_support.csv",index=False)
    print(json.dumps({"separate_prolonged_support":summary,"default_example":bundle["supported_examples"][0]["label"]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-only",action="store_true",help="Update only the separate prolonged-training support gate; preserve all fitted models and calibration.")
    args=parser.parse_args()
    artifact = ROOT / "models" / "risk_bundle_audited.joblib"
    bundle = load_audited_bundle(artifact)
    preserved_hashes = {key: joblib.hash(bundle[key]) for key in ["classifier", "calibrator", "preprocessor", "quantile_models", "probability_update"]}
    print("Loading only prolonged records and selected features...", flush=True)
    frame = pd.read_parquet(ROOT / "data" / "processed_fires.parquet", columns=FEATURES + [TARGET, "FIRE_YEAR", "FOD_ID"], filters=[(TARGET, ">", 3), (TARGET, "<=", 365), ("FIRE_YEAR", ">=", 1992), ("FIRE_YEAR", "<=", 2024)])
    training = frame.loc[frame.FIRE_YEAR.between(1992, 2014)]
    calibration = frame.loc[frame.FIRE_YEAR.between(2020, 2022)]
    reassessment = frame.loc[frame.FIRE_YEAR.between(2023, 2024)]
    assert len(training) > 1000 and len(calibration) > 100 and len(reassessment) > 100
    assert training.FIRE_YEAR.max() < calibration.FIRE_YEAR.min() <= calibration.FIRE_YEAR.max() < reassessment.FIRE_YEAR.min()
    if args.support_only:
        if "prolonged_duration" not in bundle:
            raise ValueError("Fit the conditional prolonged model before a support-only upgrade.")
        save_support_only(bundle,artifact,training,reassessment)
        return
    training_target = training[TARGET].to_numpy(dtype=float)
    matrix = bundle["preprocessor"].transform(normalize_features(training))
    settings = dict(max_iter=140, learning_rate=.07, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=5,
                    categorical_features=[False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES),
                    early_stopping=True, n_iter_no_change=15, random_state=42)
    target = np.log1p(training_target) + np.random.default_rng(42).uniform(-1e-6, 1e-6, len(training))
    quantiles = {}
    for q in [.1, .5, .9]:
        print(f"Fitting conditional prolonged P{int(q*100)} on {len(training):,} pre2015 events...", flush=True)
        quantiles[q] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **settings).fit(matrix, target)
    bundle["prolonged_duration"] = {"quantile_models": quantiles, "conformal_expansion_days": 0.0, "metadata": {}}
    calibration_prediction = predict_conditional_prolonged(bundle, calibration)
    calibration_target = calibration[TARGET].to_numpy(dtype=float)
    scores = np.maximum.reduce([calibration_prediction["native_lower_days"] - calibration_target,
                                calibration_target - calibration_prediction["native_upper_days"], np.zeros(len(calibration))])
    # Finite-sample split-conformal order statistic, with nonnegative expansion.
    # Exchangeability is not guaranteed across years; later coverage is reported.
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * .8)))
    expansion = float(np.partition(scores, rank - 1)[rank - 1])
    bundle["prolonged_duration"]["conformal_expansion_days"] = expansion
    print(f"CQR calibration n={len(scores):,}; finite rank={rank}; expansion={expansion:.4f} days. Reassessing2023–2024...", flush=True)
    prediction = predict_conditional_prolonged(bundle, reassessment)
    reassessment_target = reassessment[TARGET].to_numpy(dtype=float)
    baseline_median = float(np.median(training_target))
    baseline_quantiles = np.quantile(training_target, [.1, .9]).tolist()
    metrics = conditional_metrics(reassessment_target, prediction, baseline_median, baseline_quantiles)
    calibration_expanded = predict_conditional_prolonged(bundle, calibration)
    calibration_metrics = conditional_metrics(calibration_target, calibration_expanded, baseline_median, baseline_quantiles)
    group_rows = []
    for dimension, groups in {"size": size_group(reassessment.LOG_ACRES), "year": reassessment.FIRE_YEAR.astype(str).to_numpy(), "agency": reassessment.NWCG_REPORTING_AGENCY.astype(str).to_numpy()}.items():
        for group in np.unique(groups):
            mask = groups == group
            group_rows.append({"dimension": dimension, "group": str(group), **conditional_metrics(reassessment_target[mask], {key: value[mask] for key, value in prediction.items()}, baseline_median, baseline_quantiles), "small_subgroup_warning": bool(mask.sum() < 200)})
    support = support_frame(bundle, reassessment)
    for status in ["supported", "caution", "unsupported"]:
        mask = support.status.eq(status).to_numpy()
        if mask.any():
            group_rows.append({"dimension": "support", "group": status, **conditional_metrics(reassessment_target[mask], {key: value[mask] for key, value in prediction.items()}, baseline_median, baseline_quantiles), "small_subgroup_warning": bool(mask.sum() < 200)})
    checks = {"chronological_training_calibration_reassessment": True,
              "nonnegative_finite_rank_expansion": bool(expansion >= 0 and np.isfinite(expansion)),
              "finite_ordered_conditional_bounds": bool(np.isfinite(prediction["lower_days"]).all() and np.all(prediction["lower_days"] <= prediction["median_days"]) and np.all(prediction["median_days"] <= prediction["upper_days"])),
              "bounds_4_to_365_calendar_days": bool(np.all(prediction["lower_days"] >= 4) and np.all(prediction["upper_days"] <= 365)),
              "original_estimators_and_probability_mapping_unchanged": all(joblib.hash(bundle[key]) == value for key, value in preserved_hashes.items())}
    assert all(checks.values())
    metadata = {"conditional_on": "Observed calendar-date containment duration>3days", "training_years": [1992, 2014], "training_rows": int(len(training)),
                "calibration_years": [2020, 2022], "calibration_rows": int(len(calibration)),
                "reassessment_years": [2023, 2024], "reassessment_status": "Historical reassessment of previously inspected years, not untouched confirmation",
                "nominal_coverage": .8, "finite_order_statistic_rank": rank, "conformal_expansion_days": expansion,
                "method": "Separate prolonged-only log1p-duration HGB quantiles plus nonnegative finite-rank CQR expansion; lower/upper clipped to4–365calendar days",
                "tie_break_amplitude_log_days": 1e-6, "seed": 42, "settings": settings,
                "calibration_metrics": calibration_metrics, "reassessment_metrics": metrics,
                "baseline_training_median_days": baseline_median, "baseline_training_p10_p90_days": baseline_quantiles,
                "reassessment_groups": {dimension: {row["group"]: {key: value for key, value in row.items() if key not in {"dimension", "group"}} for row in group_rows if row["dimension"] == dimension} for dimension in ["size", "agency", "year", "support"]},
                "limitations": ["Conditional on the prolonged event occurring; this is not an unconditional duration forecast.",
                                "Marginal calibration among observed prolonged fires does not guarantee individual, size-specific, or agency-specific coverage.",
                                "Conformal exchangeability across historical years is uncertain; inspect later observed coverage and subgroup widths.",
                                "Final acreage and coarse regional weather remain retrospective inputs; observed load remains endogenous.",
                                "Finite bounds describe the selected0–365day outcome-complete cohort and exclude longer and missing-outcome fires."],
                "checks": checks}
    bundle["prolonged_duration"]["metadata"] = metadata
    attach_prolonged_support(bundle,training,reassessment)
    bundle["audit_metadata"]["conditional_prolonged_duration"] = metadata
    temporary = artifact.with_suffix(".pending")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, artifact)
    restored = load_audited_bundle(artifact)
    example = restored["supported_examples"][0]["features"]
    output = predict_audited_scenario(restored, example)
    assert output["conditional_prolonged_duration"]["presentation_allowed"]
    json.dumps(output, allow_nan=False)
    bad = predict_audited_scenario(restored, {**example, "NWCG_REPORTING_AGENCY": "UNSEEN_PROLONGED_CHECK"})
    assert not bad["conditional_prolonged_duration"]["presentation_allowed"]
    report_path = ROOT / "reports" / "model_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["conditional_prolonged_duration"] = metadata
    report["checks"]["conditional_prolonged_support_gate"] = True
    report["checks"]["separate_prolonged_training_support_gate"] = True
    temporary_report = report_path.with_suffix(".pending")
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary_report, report_path)
    pd.DataFrame(group_rows).to_csv(ROOT / "reports" / "model_audit_prolonged_groups.csv", index=False)
    support_summary = metadata["support_gate"]
    support_rows = [{"scope": "all prolonged reassessment", **support_summary["unrestricted_prolonged_reassessment_metrics"]}]
    if support_summary["both_gates_allowed_prolonged_reassessment_metrics"]:
        support_rows.append({"scope": "both support gates allow presentation", **support_summary["both_gates_allowed_prolonged_reassessment_metrics"]})
    if support_summary["suppressed_prolonged_reassessment_metrics"]:
        support_rows.append({"scope": "one or both gates suppress presentation", **support_summary["suppressed_prolonged_reassessment_metrics"]})
    pd.DataFrame(support_rows).to_csv(ROOT / "reports" / "model_audit_prolonged_support.csv", index=False)
    print(json.dumps({"conditional_prolonged_metadata": metadata, "larger_supported_example": output}, indent=2), flush=True)


if __name__ == "__main__":
    main()
