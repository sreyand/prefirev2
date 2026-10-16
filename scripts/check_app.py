"""Exercise recorded profiles, audited inference gates, and incident-source plots."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from streamlit.runtime.media_file_manager import MediaFileManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_model_support import load_audited_bundle, predict_audited_scenario
from scripts.train_risk import FEATURES, NUMERIC_FEATURES


def clean_run(app):
    app.run()
    assert not app.exception, [error.message for error in app.exception]
    return app


def markdown(app):
    return "\n".join(str(element.value) for element in app.markdown)


def captions(app):
    return "\n".join(str(element.value) for element in app.caption)


def outcome_values(app):
    return re.findall(r'<div class="outcome">([^<]*)</div>', markdown(app))


def probability_label(value):
    return f"{value:.1%}"


def duration_label(value):
    value = int(round(value))
    return "Same day" if value == 0 else f"About {value} day{'s' if value != 1 else ''}"


def range_label(prediction):
    lower = max(0, math.floor(prediction["p10_days"]))
    upper = max(lower, math.ceil(prediction["p90_days"]))
    if upper == 0:
        return "Same calendar day"
    if lower == 0:
        return f"Same day to {upper} day{'s' if upper != 1 else ''}"
    return f"{lower}–{upper} days"


def assert_predictions_render(app, expected):
    assert all(result["support"]["presentation_allowed"] for result in expected)
    assert outcome_values(app) == [probability_label(result["probability_over_3_days"]) for result in expected]
    rendered = markdown(app)
    allowed_conditionals = 0
    for prediction in expected:
        conditional = prediction["conditional_prolonged_duration"]
        if not conditional["presentation_allowed"]:
            continue
        allowed_conditionals += 1
        assert f"Typical duration: **{duration_label(conditional['median_days'])}**" in rendered
        assert f"Estimated duration range: **{range_label(conditional)}**" in rendered
    assert rendered.count("If containment takes more than three days") == allowed_conditionals
    assert any(expander.label == "Support and reliability details" for expander in app.expander)


def source_profile_matches_record(example):
    source = pd.read_parquet(ROOT / "data/processed_fires.parquet", columns=FEATURES + ["FOD_ID"],
                             filters=[("FOD_ID", "==", example["source_fod_id"])])
    assert len(source) == 1, "A recorded reference profile must identify exactly one source fire"
    for feature in FEATURES:
        observed, saved = source.iloc[0][feature], example["features"][feature]
        if feature in NUMERIC_FEATURES:
            assert math.isclose(float(observed), float(saved), rel_tol=1e-10, abs_tol=1e-10), feature
        else:
            assert str(observed) == str(saved), feature


def assert_profile_controls(app, example):
    row = example["features"]
    assert app.selectbox(key="state").value == row["STATE"]
    assert app.selectbox(key="agency").value == row["NWCG_REPORTING_AGENCY"]
    for field, key in [("LATITUDE", "latitude"), ("LONGITUDE", "longitude"),
                       ("DISTANCE_TO_PLACE_CENTROID_KM", "proximity"), ("DAY_OF_YEAR", "season_day")]:
        assert math.isclose(float(app.number_input(key=key).value), float(row[field]), rel_tol=1e-10, abs_tol=1e-10), key
    for index in range(2):
        assert math.isclose(app.number_input(key=f"acres_{index}").value, math.exp(row["LOG_ACRES"]), rel_tol=1e-6, abs_tol=1e-6)
        assert app.number_input(key=f"load_{index}").value == row["OBSERVED_UNCONTAINED_FIRES"]
        assert app.selectbox(key=f"cause_{index}").value == row["NWCG_CAUSE_CLASSIFICATION"]
        assert app.selectbox(key=f"weather_{index}").value == "Recorded profile"


def reset_button(app):
    return next(button for button in app.button if button.label == "Reset both cases to this profile")


def numeric_array(value):
    """Decode Plotly's binary transport as well as ordinary JSON arrays."""
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"])).astype(float)
    return np.asarray([np.nan if item is None else item for item in value], dtype=float)


def labeled(elements, label):
    return next(element for element in elements if element.label == label)


def assert_precision_panel(app, population, threshold):
    """Verify user-visible flag tradeoffs against saved confusion counts."""
    chosen = next(row for row in population["threshold_results"] if row["threshold"] == threshold)
    assert chosen["TP"] + chosen["FN"] == population["events_over_3_days"]
    assert chosen["TP"] + chosen["FP"] == chosen["alert_count"]
    assert sum(chosen[key] for key in ["TP", "FP", "FN", "TN"]) == population["rows"]
    precision = chosen["TP"] / chosen["alert_count"]
    recall = chosen["TP"] / population["events_over_3_days"]
    assert math.isclose(precision, chosen["precision"])
    assert math.isclose(recall, chosen["recall"])
    assert labeled(app.metric, "Precision").value == f"{precision:.1%}"
    assert labeled(app.metric, "Recall").value == f"{recall:.1%}"
    assert labeled(app.metric, "False flags / all flags").value == f"{chosen['FP']:,} / {chosen['alert_count']:,}"
    assert f"This threshold misses **{chosen['FN']:,}** prolonged fires" in markdown(app)
    assert f"{population['rows']:,} historical records · {population['events_over_3_days']:,} prolonged fires · average precision {population['average_precision']:.3f}" in captions(app)
    # Verify curve orientation, selected threshold point, and a population-
    # specific prevalence baseline; a global curve reused after filtering fails.
    specs = [json.loads(plot.proto.spec) for plot in app.get("plotly_chart")]
    figure = next(spec for spec in specs if any(trace.get("name") == "Historical precision–recall" for trace in spec["data"]))
    curve = next(trace for trace in figure["data"] if trace.get("name") == "Historical precision–recall")
    source_curve = population["precision_recall_curve"]
    np.testing.assert_allclose(numeric_array(curve["x"]), [point["recall"] for point in source_curve])
    np.testing.assert_allclose(numeric_array(curve["y"]), [point["precision"] for point in source_curve])
    point = next(trace for trace in figure["data"] if trace.get("name") == f"{threshold:.0%} threshold")
    np.testing.assert_allclose(numeric_array(point["x"]), [recall])
    np.testing.assert_allclose(numeric_array(point["y"]), [precision])
    assert any(math.isclose(shape["y0"], population["prevalence"]) and math.isclose(shape["y1"], population["prevalence"])
               for shape in figure["layout"].get("shapes", []))


def main():
    checked_sources = [ROOT / "app.py", ROOT / "scripts/check_app.py",
                       ROOT / "scripts/audit_model_support.py", ROOT / "scripts/acquire_operational_data.py"]
    for path in checked_sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    bundle = load_audited_bundle()
    examples = bundle["supported_examples"]
    source_profile_matches_record(examples[0])
    source_profile_matches_record(examples[1])
    expected_initial = predict_audited_scenario(bundle, examples[0]["features"])
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception, [error.message for error in app.exception]
    assert app.button_group(key="view").value == "Findings"
    assert list(app.button_group(key="view").options) == ["Findings", "Compare fires", "Incident reports", "Model & data"]
    app.button_group(key="view").set_value("Compare fires")
    clean_run(app)
    assert_profile_controls(app, examples[0])
    assert_predictions_render(app, [expected_initial, expected_initial])
    assert "inputs match" in markdown(app)

    # A modest change stays close to a real supported record and must flow from
    # each widget through the audited API to the displayed probability/bounds.
    base = examples[0]["features"]
    app.number_input(key="acres_1").set_value(round(math.exp(base["LOG_ACRES"]) * 1.2, 6))
    app.number_input(key="load_1").set_value(int(base["OBSERVED_UNCONTAINED_FIRES"]) + 10)
    app.selectbox(key="weather_1").set_value("Custom")
    clean_run(app)
    app.number_input(key="vpd_1").set_value(round(float(base["vapour_pressure_deficit_max"]), 2) + 0.1)
    app.number_input(key="wind_1").set_value(round(float(base["wind_speed_10m_max"])) + 1.0)
    app.number_input(key="rain_1").set_value(round(float(base["ANTECEDENT_PRECIP_30D_MM"])) + 1.0)
    clean_run(app)
    changed = dict(base, LOG_ACRES=math.log(app.number_input(key="acres_1").value),
                   OBSERVED_UNCONTAINED_FIRES=app.number_input(key="load_1").value,
                   vapour_pressure_deficit_max=app.number_input(key="vpd_1").value,
                   wind_speed_10m_max=app.number_input(key="wind_1").value,
                   ANTECEDENT_PRECIP_30D_MM=app.number_input(key="rain_1").value)
    expected_changed = predict_audited_scenario(bundle, changed)
    assert_predictions_render(app, [expected_initial, expected_changed])
    note = captions(app)
    for label in ["Changed in Fire B:", "final size", "observed load", "dryness", "wind", "prior rain"]:
        assert label in note, label

    # BIA is an existing fitted category but has only a small exact state/cause
    # neighborhood for the initial recorded profile. Pick a guaranteed blocked
    # existing category from the API rather than relying on a particular slider.
    unsupported_agency = next(agency for agency in bundle["categories"]["NWCG_REPORTING_AGENCY"]
                              if not predict_audited_scenario(bundle, dict(base, NWCG_REPORTING_AGENCY=agency))["support"]["presentation_allowed"])
    app.selectbox(key="agency").set_value(unsupported_agency)
    clean_run(app)
    assert outcome_values(app) == [], "Unsupported cases must have no numeric outcome card"
    assert "insufficient comparable training records" in "\n".join(str(item.value) for item in app.warning)
    assert "comparison is unavailable" in markdown(app)
    assert "Typical duration:" not in markdown(app)
    assert "Original all-fire median:" not in markdown(app)
    reset_button(app).click()
    clean_run(app)
    assert_profile_controls(app, examples[0])
    assert_predictions_render(app, [expected_initial, expected_initial])
    # Reset must also restore custom weather values that were previously edited.
    app.selectbox(key="weather_1").set_value("Custom")
    clean_run(app)
    for field, key in [("vapour_pressure_deficit_max", "vpd_1"), ("wind_speed_10m_max", "wind_1"),
                       ("ANTECEDENT_PRECIP_30D_MM", "rain_1")]:
        assert math.isclose(app.number_input(key=key).value, float(base[field]), rel_tol=1e-10, abs_tol=1e-10), key
    app.number_input(key="acres_1").set_value(10000.0)
    app.number_input(key="wind_1").set_value(120.0)
    clean_run(app)
    app.selectbox(key="profile_index").set_value(1)
    clean_run(app)
    assert_profile_controls(app, examples[1])
    expected_second = predict_audited_scenario(bundle, examples[1]["features"])
    assert_predictions_render(app, [expected_second, expected_second])
    app.selectbox(key="weather_1").set_value("Custom")
    clean_run(app)
    for field, key in [("vapour_pressure_deficit_max", "vpd_1"), ("wind_speed_10m_max", "wind_1"),
                       ("ANTECEDENT_PRECIP_30D_MM", "rain_1")]:
        assert math.isclose(app.number_input(key=key).value, float(examples[1]["features"][field]), rel_tol=1e-10, abs_tol=1e-10), key

    # Ensure an actually supported prolonged branch also retains API-equivalent
    # bounds after adding its independent conditional-training support gate.
    both_index = next(index for index, example in enumerate(examples)
                      if predict_audited_scenario(bundle, example["features"])["conditional_prolonged_duration"]["presentation_allowed"])
    app.selectbox(key="profile_index").set_value(both_index)
    clean_run(app)
    both_prediction = predict_audited_scenario(bundle, examples[both_index]["features"])
    assert_predictions_render(app, [both_prediction, both_prediction])

    # Regression: IN / FS / Human has ample all-fire neighbors but no fitting
    # prolonged outcomes. Its probability can be displayed, while its duration
    # branch must be gated separately in the UI and in the downloaded JSON.
    regression_source = pd.read_parquet(ROOT / "data/processed_fires.parquet", columns=FEATURES,
                                        filters=[("FOD_ID", "==", 3455)])
    assert len(regression_source) == 1
    regression = {feature: float(regression_source.iloc[0][feature]) if feature in NUMERIC_FEATURES
                  else str(regression_source.iloc[0][feature]) for feature in FEATURES}
    regression_prediction = predict_audited_scenario(bundle, regression)
    assert regression_prediction["support"]["presentation_allowed"]
    conditional = regression_prediction["conditional_prolonged_duration"]
    assert not conditional["presentation_allowed"]
    assert not conditional["support"]["presentation_allowed"]
    for field, key in [("STATE", "state"), ("NWCG_REPORTING_AGENCY", "agency")]:
        app.selectbox(key=key).set_value(regression[field])
    for field, key in [("LATITUDE", "latitude"), ("LONGITUDE", "longitude"),
                       ("DAY_OF_YEAR", "season_day"), ("DISTANCE_TO_PLACE_CENTROID_KM", "proximity")]:
        value = int(regression[field]) if field == "DAY_OF_YEAR" else regression[field]
        app.number_input(key=key).set_value(value)
    for index in range(2):
        app.number_input(key=f"acres_{index}").set_value(round(math.exp(regression["LOG_ACRES"]), 6))
        app.number_input(key=f"load_{index}").set_value(int(regression["OBSERVED_UNCONTAINED_FIRES"]))
        app.selectbox(key=f"cause_{index}").set_value(regression["NWCG_CAUSE_CLASSIFICATION"])
        app.selectbox(key=f"weather_{index}").set_value("Custom")
    clean_run(app)
    for index in range(2):
        for field, key in [("vapour_pressure_deficit_max", "vpd"), ("wind_speed_10m_max", "wind"),
                           ("ANTECEDENT_PRECIP_30D_MM", "rain")]:
            app.number_input(key=f"{key}_{index}").set_value(regression[field])
    exported_comparisons = []
    original_media_add = MediaFileManager.add

    def capture_comparison(manager, data, mimetype, coordinates, file_name=None, is_for_static_download=False):
        if file_name == "prefire_comparison.json":
            exported_comparisons.append(json.loads(data.decode("utf-8") if isinstance(data, bytes) else data))
        return original_media_add(manager, data, mimetype, coordinates, file_name, is_for_static_download)

    with patch.object(MediaFileManager, "add", new=capture_comparison):
        clean_run(app)
    assert outcome_values(app) == [probability_label(regression_prediction["probability_over_3_days"])] * 2
    assert "Typical duration:" not in markdown(app)
    assert "Estimated duration range:" not in markdown(app)
    assert "If containment takes more than three days" not in markdown(app)
    assert "Prolonged-duration estimate hidden" in "\n".join(str(item.value) for item in app.info)
    assert exported_comparisons, "Capture the actual downloadable comparison payload"
    for result in exported_comparisons[-1]["predictions"]:
        assert result["estimate"] is not None
        hidden = result["estimate"]["conditional_prolonged_duration"]
        assert hidden["estimate"] is None
        assert not hidden["support"]["presentation_allowed"]
        assert not {"lower_days", "upper_days", "median_days", "p10_days", "p50_days", "p90_days"}.intersection(hidden)

    app.button_group(key="view").set_value("Incident reports")
    clean_run(app)
    operational = json.loads((ROOT / "reports/operational_examples.json").read_text())
    incident = operational["examples"][0]
    observed = pd.read_parquet(ROOT / "data/operational_incident_daily.parquet", filters=[("INCIDENT_ID", "==", incident["incident_id"])])
    assert len(observed) == incident["observed_report_days"] == 72
    assert f"{incident['observed_report_days']} reported days across {incident['calendar_span_days']} calendar days" in captions(app)
    assert str(incident["FOD_ID"]) in captions(app)
    assert "Gaps remain gaps" in captions(app)
    source_context = pd.read_parquet(ROOT / "data/operational_fpa_context.parquet", filters=[("FOD_ID", "==", incident["FOD_ID"])])
    assert len(source_context) == 1
    assert source_context.OPERATIONAL_INCIDENT_ID.iloc[0] == incident["incident_id"]
    assert source_context.OPERATIONAL_LINK_ROLE.iloc[0] == "direct_incident_context"
    observed = observed.set_index(pd.to_datetime(observed.REPORT_DATE)).sort_index()
    calendar = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    observed = observed.reindex(calendar)
    plots = app.get("plotly_chart")
    assert len(plots) == 3
    specs = [json.loads(plot.proto.spec) for plot in plots]
    for index, field in [(0, "TOTAL_PERSONNEL"), (1, "TOTAL_AERIAL")]:
        trace = specs[index]["data"][0]
        actual = numeric_array(trace["y"])
        assert len(actual) == incident["calendar_span_days"] == 74
        assert np.isnan(actual).sum() == 2
        assert trace["connectgaps"] is False
        np.testing.assert_allclose(actual, observed[field].to_numpy(dtype=float), equal_nan=True)
    percentage = specs[2]["data"][1]
    assert percentage["connectgaps"] is False
    np.testing.assert_allclose(numeric_array(percentage["y"]), observed.CONTAINMENT_PERCENT_EXPLICIT_PCT.to_numpy(dtype=float), equal_nan=True)

    app.button_group(key="view").set_value("Model & data")
    clean_run(app)
    assert "How reliable are these estimates?" in "\n".join(str(item.value) for item in app.subheader)
    assert "Duration range reliability" in "\n".join(str(item.value) for item in app.subheader)
    assert any(expander.label == "Model experiments and research reference" for expander in app.expander)
    assert "That paper predicts burned area; our target is containment duration" in markdown(app)
    alerts = json.loads((ROOT / "reports/alert_precision.json").read_text())
    all_records = alerts["populations"]["all_records"]["overall"]
    supported_records = alerts["populations"]["presentation_allowed"]["overall"]
    threshold_label = "Illustrative probability threshold for a high-risk flag"
    assert labeled(app.radio, "Evaluation population").value == "all_records"
    assert labeled(app.selectbox, threshold_label).value == 0.2
    assert all_records["rows"] == 128797 and all_records["events_over_3_days"] == 5114
    assert_precision_panel(app, all_records, 0.2)
    labeled(app.selectbox, threshold_label).set_value(0.5)
    clean_run(app)
    assert_precision_panel(app, all_records, 0.5)
    labeled(app.radio, "Evaluation population").set_value("presentation_allowed")
    clean_run(app)
    assert supported_records["rows"] == 95226 and supported_records["events_over_3_days"] == 3683
    assert round(supported_records["average_precision"], 4) == 0.1927
    assert round(all_records["average_precision"], 4) == 0.2246
    assert not math.isclose(supported_records["average_precision"], all_records["average_precision"])
    assert_precision_panel(app, supported_records, labeled(app.selectbox, threshold_label).value)
    app.button_group(key="view").set_value("Findings")
    clean_run(app)
    app.radio(key="load_variant").set_value("clocks_only_sensitivity")
    clean_run(app)
    result = {"status": "passed", "views_rendered": 4,
              "source_profiles_match_real_records": True, "coherent_defaults_match_audited_api": True,
              "conditional_prolonged_bounds_match_api": True, "custom_weather_and_changed_case_match_api": True,
              "changed_inputs_explained": True, "unsupported_forecast_numbers_suppressed": True,
              "conditional_support_gate_independent": True, "conditional_range_hidden_without_prolonged_support": True,
              "unsupported_conditional_download_sanitized": True,
              "reset_restores_all_controls": True, "profile_callback_resets_stale_inputs": True,
              "incident_direct_source_join_verified": True, "incident_reports_count": 72,
              "incident_calendar_span_days": 74, "incident_plot_gaps_preserved": True,
              "explicit_pct_only_curve_matches_source": True, "paper_section_rendered": True,
              "fixed_threshold_precision_recall_match_confusion_counts": True,
              "threshold_change_updates_false_flags_and_missed_cases": True,
              "precision_recall_curve_and_baseline_match_population": True,
              "support_filter_changes_population_counts_and_average_precision": True,
              "precision_recall_all_record_rows": all_records["rows"],
              "precision_recall_supported_rows": supported_records["rows"],
              "clock_sensitivity_selectable": True, "sources_compile": True}
    (ROOT / "reports/app_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
