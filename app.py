"""PreFIRE: clear findings and comparisons backed by the saved research models."""
from pathlib import Path
import calendar
import html
import json
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scripts.audit_model_support import load_audited_bundle, predict_audited_scenario

ROOT = Path(__file__).resolve().parent
ACCENT = "#e7ad66"
MUTED = "#a4afa7"
SIZE_LABELS = {"Micro": "Under 1 acre", "Small": "1–10 acres", "Medium": "10–100 acres", "Large": "100–1,000 acres", "1000+ acres": "1,000+ acres"}
STATE_NAMES = dict(zip(
    "AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split(),
    ["Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]))
WEATHER = {"Milder": (1.0, 12.0, 80.0), "Typical": (2.0, 20.0, 40.0), "Hot / dry": (4.0, 30.0, 5.0)}

st.set_page_config(page_title="PreFIRE · Understand containment", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
header[data-testid="stHeader"] {visibility:hidden;height:0}
.block-container {max-width:1160px;padding-top:1.5rem;padding-bottom:3rem}
h1 {font-size:clamp(1.7rem,3.7vw,2.5rem)!important;letter-spacing:-.035em;line-height:1.13!important;max-width:860px}
h2 {font-size:1.5rem!important;letter-spacing:-.025em}
h3 {font-size:1.15rem!important;letter-spacing:-.015em}
p, li {line-height:1.6}
[data-testid="stMetricValue"] {font-size:1.8rem!important;font-weight:600}
[data-testid="stMetricLabel"] {color:#a4afa7}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:12px}
[data-testid="stForm"] {border:0;padding:0}
.brand {font-weight:700;font-size:1rem;letter-spacing:.06em;color:#e7ad66;margin-bottom:1rem}
.brand span {color:#89968d;font-size:.7rem;font-weight:400;letter-spacing:.1em;margin-left:.75rem}
.intro {max-width:760px;color:#bac3bc;font-size:1rem;line-height:1.65;margin-bottom:.5rem}
.meta {font-size:.76rem;color:#89968d;margin-bottom:1.1rem}
.finding {font-size:1.3rem;line-height:1.4;font-weight:550;margin:.3rem 0 .5rem;max-width:850px}
.quiet {font-size:.82rem;line-height:1.6;color:#a4afa7}
.case-title {font-size:.85rem;color:#e7ad66;letter-spacing:.08em;font-weight:600;margin:0 0 .75rem}
.outcome {font-size:1.65rem;line-height:1.15;font-weight:600;letter-spacing:-.02em;margin:.25rem 0 .75rem}
.kicker {font-size:.74rem;color:#a4afa7;margin-top:.65rem}
.result-note {padding:.85rem 1rem;background:#242b25;border-radius:8px;font-size:.9rem;line-height:1.6;margin:.8rem 0}
@media(max-width:700px){.block-container{padding-top:1rem;padding-left:1rem;padding-right:1rem}.brand{margin-bottom:.7rem}.outcome{font-size:1.4rem}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def _json(path, version):
    return json.loads(Path(path).read_text())


@st.cache_data
def _csv(path, version):
    return pd.read_csv(path)


@st.cache_resource
def _models(path, version):
    return load_audited_bundle(path)


def report(name):
    path = ROOT / "reports" / name
    return _json(str(path), path.stat().st_mtime_ns)


def csv_report(name):
    path = ROOT / "reports" / name
    return _csv(str(path), path.stat().st_mtime_ns)


def models():
    path = ROOT / "models/risk_bundle_audited.joblib"
    return _models(str(path), path.stat().st_mtime_ns)


def navigate(view):
    st.session_state["view"] = view


def plot_style(fig, x, y, height=330):
    return fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font=dict(family="Arial, sans-serif", color="#c8d0ca", size=12),
                             margin=dict(l=0, r=15, t=15, b=0), height=height,
                             xaxis_title=x, yaxis_title=y, legend_title_text="",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                             yaxis=dict(gridcolor="#2b322e", zerolinecolor="#505b53"),
                             xaxis=dict(showgrid=False))


def probability(value):
    return f"{value:.1%}"


def duration(value):
    days = int(round(value))
    return "Same day" if days == 0 else f"About {days} day{'s' if days != 1 else ''}"


def duration_range(prediction):
    low = max(0, int(math.floor(prediction["p10_days"])))
    high = max(low, int(math.ceil(prediction["p90_days"])))
    if high == 0:
        return "Same calendar day"
    if low == 0:
        return f"Same day to {high} day{'s' if high != 1 else ''}"
    return f"{low}–{high} days"


@st.cache_data
def incident_days(path, version, incident_id):
    return pd.read_parquet(path, filters=[("INCIDENT_ID", "==", incident_id)])


def reset_profile(bundle):
    row = bundle["supported_examples"][st.session_state["profile_index"]]["features"]
    for field, key in [("STATE", "state"), ("NWCG_REPORTING_AGENCY", "agency"),
                       ("LATITUDE", "latitude"), ("LONGITUDE", "longitude"),
                       ("DISTANCE_TO_PLACE_CENTROID_KM", "proximity")]:
        st.session_state[key] = row[field]
    st.session_state["season_day"] = int(row["DAY_OF_YEAR"])
    for i in range(2):
        st.session_state[f"acres_{i}"] = round(math.exp(row["LOG_ACRES"]), 6)
        st.session_state[f"load_{i}"] = int(row["OBSERVED_UNCONTAINED_FIRES"])
        st.session_state[f"cause_{i}"] = row["NWCG_CAUSE_CLASSIFICATION"]
        st.session_state[f"weather_{i}"] = "Recorded profile"
        for field, key in [("vapour_pressure_deficit_max", "vpd"), ("wind_speed_10m_max", "wind"), ("ANTECEDENT_PRECIP_30D_MM", "rain")]:
            st.session_state[f"{key}_{i}"] = float(row[field])


def outcome(prediction, letter):
    st.markdown(f"**Fire {letter} · model estimate**")
    support = prediction["support"]
    if not support["presentation_allowed"]:
        st.warning("Estimate hidden · insufficient comparable training records")
        st.write(support["reason"])
        st.caption("Load another recorded profile or move the changed inputs closer to it.")
        return
    if support["status"] == "caution":
        st.warning("Unusual combination · interpret cautiously")
    else:
        st.caption("Comparable training records found · accuracy still varies")
    st.markdown('<div class="kicker">Chance of taking more than 3 calendar days</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="outcome">{probability(prediction["probability_over_3_days"])}</div>', unsafe_allow_html=True)
    st.progress(prediction["probability_over_3_days"])
    conditional = prediction.get("conditional_prolonged_duration")
    if conditional and not conditional["presentation_allowed"]:
        st.info("Prolonged-duration estimate hidden · insufficient comparable prolonged fires")
        st.caption(conditional.get("support", {}).get("reason", "The duration branch needs its own conditional-training support."))
    elif conditional:
        st.markdown("**If containment takes more than three days:**")
        if conditional.get("support", {}).get("status") == "caution":
            st.caption("Limited support within the prolonged-fire training records; interpret this duration cautiously.")
        st.write(f"Typical duration: **{duration(conditional['median_days'])}**")
        st.write(f"Estimated duration range: **{duration_range(conditional)}**")
        st.caption("Total days from discovery to containment, rounded outward; this range applies only to the prolonged-fire branch.")
        group = conditional.get("group_reliability", {}).get("size")
        if group:
            st.caption(f"For prolonged fires in this size group, the range covered {group['interval_coverage']:.0%} of {group['rows']:,} reassessed outcomes.")


st.markdown('<div class="brand">PreFIRE<span>CONTAINMENT EXPLORER</span></div>', unsafe_allow_html=True)
st.title("Which fires take longer to contain?")
st.markdown('<div class="intro">Explore recorded containment times, compare historical fire conditions, and inspect reported resource assignments.</div>', unsafe_allow_html=True)
required = ["dashboard_summary.json", "model_audit.json", "reporting_audit.json", "load_analysis.json"]
if not all((ROOT / "reports" / name).exists() for name in required):
    st.error("The research outputs are not ready. Run the project pipeline to generate them.")
    st.stop()
summary, audit, reporting = report("dashboard_summary.json"), report("model_audit.json"), report("reporting_audit.json")
metrics = audit["reassessment_metrics"]["audited_selected"]
st.markdown(f'<div class="meta">{summary["cohort"]["count"]/1e6:.2f} million usable records · 1992–2024 · Retrospective research</div>', unsafe_allow_html=True)
view = st.segmented_control("Explore", ["Findings", "Compare fires", "Incident reports", "Model & data"],
                            default=None if "view" in st.session_state else "Findings", required=True, key="view", label_visibility="collapsed", width="stretch")
st.write("")

if view == "Findings":
    st.markdown('<div class="finding">Larger fires show longer reported containment times. Reporting changes complicate the comparison.</div>', unsafe_allow_html=True)
    st.write("Most recorded fires were contained on their discovery date. For larger fires, recent records show longer durations; the size of that change depends on the years and agencies compared.")
    scope = st.radio("Reporting scope", ["overall", "FS_only"], format_func=lambda x: "All reporting agencies" if x == "overall" else "U.S. Forest Service only", horizontal=True)
    balanced = st.toggle("Balance the acreage and agency mix", value=False, help="Reweight shared fine-acreage × agency groups to their early-period mix. This is a descriptive sensitivity, not a causal adjustment.")
    source = reporting["standardized_comparisons"] if balanced else reporting["period_comparisons"]
    frame = pd.DataFrame(source)
    frame = frame.loc[frame.scope.eq(scope)]
    qkey = "weighted_p90_days" if balanced else "p90_days"
    big = frame.loc[frame.size_bin.eq("1000+ acres")].set_index("period")
    early, recent = big.loc["early_1992_2005"], big.loc["recent_2015_2024"]
    a, b = st.columns(2)
    a.metric("Contained on the discovery date", f"{summary['cohort']['same_calendar_day_fraction']:.0%}")
    a.caption("All usable records · 1992–2024")
    b.metric("90th-percentile time · 1,000+ acres", f"{early[qkey]:.0f} → {recent[qkey]:.0f} days")
    b.caption("1992–2005 → 2015–2024 · selected reporting scope")
    fig = go.Figure()
    for period, label, color in [("early_1992_2005", "1992–2005", MUTED), ("recent_2015_2024", "2015–2024", ACCENT)]:
        group = frame.loc[frame.period.eq(period)].set_index("size_bin").reindex(summary["size_order"])
        fig.add_trace(go.Bar(x=[SIZE_LABELS[x] for x in group.index], y=group[qkey], name=label, marker_color=color,
                             hovertemplate="%{x}<br>%{y:.1f} calendar days<extra>%{fullData.name}</extra>"))
    st.plotly_chart(plot_style(fig, "Final fire size", "90th-percentile containment time (days)"), width="stretch")
    if balanced:
        st.caption("The plotted value is the 90th percentile under the early-period fine-acreage / agency mix, on shared support. It is not the raw fraction of recent fires. Final acreage is known after the fire; weather, location, policy, and reporting can still differ.")
    else:
        st.caption("About nine in ten recorded fires were contained by the plotted duration. Final acreage is known after the fire. Equal-size groups still differ in weather, location, policy, and reporting.")
    if balanced:
        st.caption(f"1,000+ acres: shared groups retain {early['retained_fraction']:.1%} of early and {recent['retained_fraction']:.1%} of recent records. Weights use the early-period composition.")
    coverage = pd.DataFrame(reporting["coverage_by_period"])
    c = coverage.loc[coverage.scope.eq(scope)].set_index("period")
    st.warning(f"Containment-date coverage changed from {c.loc['early_1992_2005','eligible_containment_date_fraction']:.0%} to {c.loc['recent_2015_2024','eligible_containment_date_fraction']:.0%} in this reporting scope. Fires missing containment dates are absent from the duration comparison.")
    st.divider()
    st.subheader("Does higher observed fire load explain delays?")
    st.write("The adjusted association is small and uneven. These records do not establish a national resource breaking point.")
    with st.expander("Explore the adjusted load analysis"):
        variant = st.radio("Duration measure", ["primary_calendar_days", "clocks_only_sensitivity"], horizontal=True,
                           format_func=lambda x: "Calendar dates" if x.startswith("primary") else "Clock-complete records", key="load_variant")
        data = csv_report("load_analysis.csv")
        c = data.loc[data.analysis.eq(variant)]
        fig = go.Figure(go.Scatter(x=c.observed_daily_load, y=c.adjusted_days_plus_one_percent_change, mode="lines+markers", line_color=ACCENT,
                                  error_y=dict(type="data", symmetric=False, array=c.percent_change_ci_high-c.adjusted_days_plus_one_percent_change,
                                               arrayminus=c.adjusted_days_plus_one_percent_change-c.percent_change_ci_low)))
        fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
        st.plotly_chart(plot_style(fig, "Observed fires awaiting containment (count)", "Adjusted change in geometric mean(days + 1) (%)"), width="stretch")
        st.caption("Relative to 10th-percentile load; 95% intervals cluster by discovery day. Controls: weather, final acreage, location, season, year, agency and cause. Cross-day serial dependence is not accounted for.")
        st.write("Load is reconstructed using later containment dates and only outcome-complete fires, including the focal fire. It does not measure available crews. Final acreage and load can themselves reflect fire outcomes, so adjusted associations are not causal resource effects.")
    st.button("Compare two sets of conditions", type="primary", on_click=navigate, args=("Compare fires",))

elif view == "Compare fires":
    bundle = models()
    st.subheader("Start with a recorded fire. Change one condition.")
    st.write("Both cases start from the same real profile. Change Fire B to explore model associations. Final acreage and reconstructed load make this a retrospective comparison.")
    if "profile_index" not in st.session_state:
        st.session_state["profile_index"] = 0
        reset_profile(bundle)
    examples = bundle["supported_examples"]
    st.selectbox("Recorded reference profile", range(len(examples)), format_func=lambda i: examples[i]["label"], key="profile_index", on_change=reset_profile, args=(bundle,))
    example = examples[st.session_state["profile_index"]]
    base = example["features"]
    st.caption(f"Recorded profile · {example['source_year']} · FOD_ID {example['source_fod_id']}. Both cases use the same starting conditions.")
    st.button("Reset both cases to this profile", on_click=reset_profile, args=(bundle,))
    with st.expander("Shared location, season, and reporting context"):
        left, right = st.columns(2)
        state = left.selectbox("State", bundle["categories"]["STATE"], key="state")
        agency = right.selectbox("Reporting agency", bundle["categories"]["NWCG_REPORTING_AGENCY"], key="agency")
        latitude = left.number_input("Latitude", 24.0, 50.0, key="latitude", format="%.5f")
        longitude = right.number_input("Longitude", -125.0, -66.0, key="longitude", format="%.5f")
        season = left.number_input("Discovery day of year", 1, 366, key="season_day")
        proximity = right.number_input("Distance to a place centroid (km)", 0.0, 2000.0, key="proximity")
        st.caption("Coordinates and season come from the selected profile. If changing state, also update coordinates; the support check assesses the combination. Proximity is a static place-centroid proxy.")
    cases = []
    for i, col in enumerate(st.columns(2)):
        with col, st.container(border=True):
            st.markdown(f'<div class="case-title">FIRE {"A" if i == 0 else "B"}</div>', unsafe_allow_html=True)
            acres = st.number_input("Final fire size (acres)", .001, 1000000.0, step=10.0, key=f"acres_{i}")
            preset = st.selectbox("Weather", ["Recorded profile", "Milder", "Typical", "Hot / dry", "Custom"], key=f"weather_{i}")
            if preset == "Custom":
                vpd = st.number_input("Vapor-pressure deficit (kPa)", 0.0, 10.0, step=.1, key=f"vpd_{i}")
                wind = st.number_input("Wind speed (km/h)", 0.0, 120.0, step=1.0, key=f"wind_{i}")
                rain = st.number_input("Previous 30-day rain (mm)", 0.0, 1000.0, step=5.0, key=f"rain_{i}")
            elif preset == "Recorded profile":
                vpd, wind, rain = (base[x] for x in ["vapour_pressure_deficit_max", "wind_speed_10m_max", "ANTECEDENT_PRECIP_30D_MM"])
            else:
                vpd, wind, rain = WEATHER[preset]
            st.caption(f"VPD {vpd:.2f} kPa · wind {wind:.1f} km/h · prior rain {rain:.1f} mm")
            load = st.number_input("Fires awaiting containment", 0, 5000, step=50, key=f"load_{i}", help="Count within the selected historical records, including the focal fire. This is not a crew-availability measure.")
            cause = st.selectbox("Ignition cause", bundle["categories"]["NWCG_CAUSE_CLASSIFICATION"], key=f"cause_{i}")
            row = dict(base, STATE=state, NWCG_REPORTING_AGENCY=agency, LATITUDE=latitude, LONGITUDE=longitude, DAY_OF_YEAR=season,
                       DISTANCE_TO_PLACE_CENTROID_KM=proximity, LOG_ACRES=math.log(acres), OBSERVED_UNCONTAINED_FIRES=load,
                       NWCG_CAUSE_CLASSIFICATION=cause, vapour_pressure_deficit_max=vpd, wind_speed_10m_max=wind, ANTECEDENT_PRECIP_30D_MM=rain)
            cases.append({"row": row})
    predictions = [predict_audited_scenario(bundle, case["row"]) for case in cases]
    labels = {"LOG_ACRES": "final size", "OBSERVED_UNCONTAINED_FIRES": "observed load", "NWCG_CAUSE_CLASSIFICATION": "cause", "vapour_pressure_deficit_max": "dryness", "wind_speed_10m_max": "wind", "ANTECEDENT_PRECIP_30D_MM": "prior rain"}
    changed = [label for key, label in labels.items() if cases[0]["row"][key] != cases[1]["row"][key]]
    st.caption("Changed in Fire B: " + (", ".join(changed) if changed else "nothing yet — the two cases are identical."))
    for col, letter, p in zip(st.columns(2), ["A", "B"], predictions):
        with col:
            outcome(p, letter)
    allowed = all(p["support"]["presentation_allowed"] for p in predictions)
    if not allowed:
        result = "A comparison is unavailable because at least one case lacks sufficient training support."
    elif not changed:
        result = "The estimates match because the inputs match. Change one condition in Fire B to explore the difference."
    else:
        delta = predictions[1]["probability_over_3_days"]-predictions[0]["probability_over_3_days"]
        result = f"Fire B's estimated chance changes by {delta*100:+.1f} percentage points. This is a model association, not an effect of changing resources or policy."
    st.markdown(f'<div class="result-note">{html.escape(result)}</div>', unsafe_allow_html=True)
    st.caption("Probability and prolonged duration require separate support checks. The conditional range targets 80% coverage among prolonged fires; coverage varies by size and agency. Calendar days are date differences, not exact 24-hour intervals.")
    with st.expander("Support and reliability details"):
        for letter, p in zip(["A", "B"], predictions):
            st.write(f"**Fire {letter} · {p['support']['status']}**")
            st.write(p["support"]["reason"])
            st.caption(f"Exact state × agency × cause reference count: {p['support']['context_reference_count']:,}. Distance thresholds use earlier training features; passing does not guarantee accuracy.")
            conditional_support = p.get("conditional_prolonged_duration", {}).get("support", {})
            if conditional_support:
                st.write(f"Prolonged-duration support: {conditional_support['status']}; {conditional_support['context_reference_count']:,} conditional-training reference cases in this context.")
                st.caption(conditional_support["reason"])
            for warning in p["warnings"]:
                if "disagree" in warning:
                    st.warning(warning)
    export = []
    for p in predictions:
        estimate = dict(p) if p["support"]["presentation_allowed"] else None
        if estimate and not p.get("conditional_prolonged_duration", {}).get("presentation_allowed", False):
            estimate["conditional_prolonged_duration"] = {"estimate": None, "support": p.get("conditional_prolonged_duration", {}).get("support")}
        export.append({"support": p["support"], "estimate": estimate})
    st.download_button("Save this comparison", json.dumps({"cases": cases, "predictions": export}, indent=2), "prefire_comparison.json", "application/json")

elif view == "Incident reports":
    st.subheader("What resources were reported on an actual incident?")
    st.write("Follow reported personnel, aerial assignments, burned acreage, and containment across an incident. This view describes deployments; it does not estimate how many resources were available or needed.")
    upgrade, examples = report("data_upgrade.json"), report("operational_examples.json")
    choices = examples["examples"]
    chosen = st.selectbox("Incident", range(len(choices)), format_func=lambda i: f"{choices[i]['name']} · {choices[i]['state']} · {choices[i]['source_year']}")
    case = choices[chosen]
    path = ROOT / "data/operational_incident_daily.parquet"
    data = incident_days(str(path), path.stat().st_mtime_ns, case["incident_id"]).sort_values("REPORT_DATE")
    st.caption(f"FOD_ID {case['FOD_ID']} · exact direct link · {case['observed_report_days']} reported days across {case['calendar_span_days']} calendar days. Examples were selected for reporting coverage and state diversity.")
    data = data.set_index(pd.to_datetime(data.REPORT_DATE)).reindex(pd.date_range(data.REPORT_DATE.min(), data.REPORT_DATE.max(), freq="D"))
    cols = st.columns(2)
    for col, field, title, unit in [(cols[0], "TOTAL_PERSONNEL", "Personnel assigned", "Reported / source-smoothed personnel"), (cols[1], "TOTAL_AERIAL", "Aerial resources assigned", "Reported / source-smoothed aerial resources")]:
        with col:
            st.markdown(f"**{title}**")
            fig = go.Figure(go.Scatter(x=data.index, y=data[field], mode="lines+markers", line_color=ACCENT, connectgaps=False))
            st.plotly_chart(plot_style(fig, "Report date", unit, 270), width="stretch")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.ACRES, name="Reported acres", line_color=ACCENT, connectgaps=False))
    fig.add_trace(go.Scatter(x=data.index, y=data.CONTAINMENT_PERCENT_EXPLICIT_PCT, name="Contained (%)", yaxis="y2", line_color="#85bdb5", connectgaps=False))
    fig = plot_style(fig, "Report date", "Reported burned acres", 280)
    fig.update_layout(yaxis2=dict(title="Contained (%) · explicit units only", overlaying="y", side="right", range=[0, 100], showgrid=False), margin=dict(r=65))
    st.plotly_chart(fig, width="stretch")
    st.caption("Gaps remain gaps; values are not interpolated. Source smoothing can produce fractional assignments. Containment percentages appear only when source units explicitly say PCT. Resources can respond to fire growth, so these curves do not establish a resource effect.")
    st.info("ICS-209-PLUS v3 covers 1999–2020 and favors significant incidents. The linked sample covers 2.04% of FPA records in those years; median final size is 300 acres versus 0.75 acres overall. Complex/member resources must not be added together.")
    st.markdown("Source: [ICS-209-PLUS v3](https://doi.org/10.6084/m9.figshare.19858927.v3)")
    st.download_button("Download these incident observations", data.dropna(subset=["INCIDENT_ID"]).to_csv(index=False), "prefire_incident.csv", "text/csv")

else:
    st.subheader("How reliable are these estimates?")
    st.write("The model is useful for exploring historical patterns, but its precision–recall tradeoff remains weak. High thresholds miss most prolonged fires.")
    raw, old = audit["reassessment_metrics"]["raw"], audit["reassessment_metrics"]["old_platt"]
    prolonged = audit.get("conditional_prolonged_duration", {})
    st.caption("2023–2024 historical reassessment · these years were previously inspected; this is not an untouched future test.")
    st.caption("Average precision summarizes the precision–recall curve. The selected threshold below shows the actual share of flags that were correct.")
    if (ROOT / "reports/alert_precision.json").exists():
        alerts = report("alert_precision.json")
        with st.expander("Precision, missed prolonged fires, and false alerts", expanded=True):
            population = st.radio("Evaluation population", ["all_records", "presentation_allowed"], format_func=lambda x: "All usable 2023–2024 records" if x == "all_records" else "Records passing the probability support check", horizontal=True)
            pr = alerts["populations"][population]["overall"]
            st.caption(f"{pr['rows']:,} historical records · {pr['events_over_3_days']:,} prolonged fires · average precision {pr['average_precision']:.3f}. Support filtering changes the evaluation population.")
            fixed_thresholds = [row["threshold"] for row in pr["threshold_results"]]
            threshold = st.selectbox("Illustrative probability threshold for a high-risk flag", fixed_thresholds, index=fixed_thresholds.index(.2), format_func=lambda x: f"{x:.0%}")
            chosen = next(row for row in pr["threshold_results"] if row["threshold"] == threshold)
            columns = st.columns(3)
            columns[0].metric("Precision", f"{chosen['precision']:.1%}" if chosen["precision"] is not None else "No flags", help="Share of flagged fires that truly took more than three days.")
            columns[1].metric("Recall", f"{chosen['recall']:.1%}", help="Share of prolonged fires that were flagged.")
            columns[2].metric("False flags / all flags", f"{chosen['FP']:,} / {chosen['alert_count']:,}")
            st.write(f"This threshold misses **{chosen['FN']:,}** prolonged fires. Raising the threshold generally trades fewer flags for more missed cases; the model does not choose an operational threshold.")
            curve = pd.DataFrame(pr["precision_recall_curve"])
            fig = go.Figure(go.Scatter(x=curve.recall, y=curve.precision, mode="lines", line_color=ACCENT, name="Historical precision–recall"))
            fig.add_hline(y=pr["prevalence"], line_dash="dash", line_color=MUTED)
            if chosen["precision"] is not None:
                fig.add_trace(go.Scatter(x=[chosen["recall"]], y=[chosen["precision"]], mode="markers", name=f"{threshold:.0%} threshold", marker=dict(color="#85bdb5", size=10)))
            fig = plot_style(fig, "Recall · share of prolonged fires flagged", "Precision · share of flags truly prolonged", 300)
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("Dashed line: prevalence baseline. Thresholds were fixed before this analysis; none was optimized on these outcomes. These are retrospective flags using final acreage and outcome-based load, not dispatch-time alerts.")
    st.divider()
    with st.expander("Additional model metrics"):
        st.write(f"Probability Brier error: {metrics['brier_score']:.4f} (lower is better). ROC AUC: {metrics['roc_auc']:.3f} (0.5 is random ranking; 1 is perfect). These do not replace the precision–recall check.")
    st.subheader("Duration range reliability")
    pm = prolonged["reassessment_metrics"]
    st.write(f"In the 2023–2024 reassessment, that conditional range covered {pm['interval_coverage']:.1%} of {pm['rows']:,} prolonged outcomes. Mean absolute error of the predicted median was {pm['median_mae_days']:.2f} days, versus {pm['baseline_median_mae_days']:.2f} for a constant median. Coverage was about 75% for fires of 1,000+ acres; individual estimates remain uncertain.")
    with st.expander("Inspect subgroup performance"):
        groups = csv_report("model_audit_prolonged_groups.csv")
        dimension = st.radio("Group by", ["size", "agency"], horizontal=True)
        group = groups.loc[groups.dimension.eq(dimension)]
        fig = go.Figure(go.Bar(x=group.group, y=group.interval_coverage*100, marker_color=ACCENT, customdata=group[["rows"]], hovertemplate="%{x}<br>Coverage %{y:.1f}%<br>%{customdata[0]:,} prolonged cases<extra></extra>"))
        fig.add_hline(y=80, line_dash="dash", line_color=MUTED)
        st.plotly_chart(plot_style(fig, "Prolonged-fire group", "Observed interval coverage (%)"), width="stretch")
        st.caption("Outcome-conditioned reassessment. The nominal 80% target is marginal across prolonged cases, not a guarantee for each subgroup.")
    st.divider()
    with st.expander("Model experiments and research reference"):
        paper = report("paper_benchmark.json")
        st.markdown("Methods adapted from [Sykas et al.](https://www.mdpi.com/2571-6255/7/11/374): consistent model comparisons, prior weather history, and small-event sensitivity. That paper predicts burned area; our target is containment duration. Detailed methods were checked in the [author preprint](https://www.preprints.org/manuscript/202409.0702).")
        names = {"hgb_original": "Original features", "hgb_without_weather": "Without weather", "hgb_without_observed_load": "Without observed load", "hgb_with_prior30_history": "+ 30-day history", "logistic_original": "Logistic baseline", "hgb_train_ge1acre": "Train on ≥1 acre only"}
        score = st.radio("Benchmark measure", ["Probability error (Brier)", "Average precision"], horizontal=True)
        score_key = "brier_score" if score.startswith("Probability") else "average_precision"
        fig = go.Figure()
        for phase, label, color in [("validation_results", "2018–2019 validation", MUTED), ("reassessment_results", "2023–2024 reassessment", ACCENT)]:
            rows = paper[phase]
            baseline_phase = "validation" if phase == "validation_results" else "reassessment_2023_2024"
            constant = paper["reference_baselines"]["metrics_by_phase"][baseline_phase]
            fig.add_trace(go.Bar(x=[names[x["model_id"]] for x in rows] + ["Fixed-rate baseline"], y=[x["calibrated_metrics"][score_key] for x in rows] + [constant[score_key]], name=label, marker_color=color))
        st.plotly_chart(plot_style(fig, "Controlled model / feature experiment", score), width="stretch")
        st.caption("100,000 common records per primary split, with fixed settings and older calibration. These experiments use a different fitting sample and calibration from the active model; the ≥1-acre training run uses a subset. All are evaluated on the same full population.")
        st.write("No tested feature change consistently improved newer-year results. Removing load helped validation but worsened later performance; adding weather history helped only slightly. The experimental candidates remain separate from the active model.")
    st.divider()
    st.subheader("Important data limits")
    st.write("37.5% of source fires lack containment dates; reporting changes over time. Weather is regional context, a median 166 km from fires. Final acreage and reconstructed load make the estimates retrospective. The model does not measure available crews, vegetation, soil moisture, or local terrain.")
    with st.expander("Reporting coverage by year"):
        coverage = pd.DataFrame(reporting["coverage_by_year_agency"])
        agency = st.selectbox("Coverage agency", ["All"] + sorted(coverage.agency.unique()))
        c = pd.DataFrame(reporting["coverage_by_year"]) if agency == "All" else coverage.loc[coverage.agency.eq(agency)]
        fig = go.Figure(go.Scatter(x=c.year, y=c.eligible_containment_date_fraction*100, mode="lines+markers", line_color=ACCENT))
        st.plotly_chart(plot_style(fig, "Fire year", "Eligible records with containment dates (%)", 260), width="stretch")
        st.caption("Denominator: continental study bounds and positive acreage before filtering containment outcomes.")
    st.markdown("Sources: [USFS FPA-FOD](https://research.fs.usda.gov/firelab/products/dataandtools/us-wildfires) · [Open-Meteo / ERA5](https://open-meteo.com/en/docs/historical-weather-api) · [ICS-209-PLUS](https://research.fs.usda.gov/firelab/products/dataandtools/ics-209-plus)")


