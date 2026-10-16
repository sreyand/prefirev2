"""Acquire and audit the authors' ICS-209-PLUS v3 operational wildfire data.

Run from the project root: .venv/Scripts/python scripts/acquire_operational_data.py
Outputs deliberately remain separate from the prediction cohort. Incident personnel
are deployments to selected large incidents, not workforce capacity or shortages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = "https://api.figshare.com/v2/articles/19858927/versions/3"
LANDING = "https://figshare.com/articles/dataset/19858927/3"
PAPER = "https://doi.org/10.1038/s41597-023-01955-0"
FILES = {
    "ics209plus-wildfire.zip": (38766504, 48717736, "564b028f2c724ab7855f95bbaca74683"),
    "ics209plus-reference.zip": (38766492, 88397, "5fec186428418aa28b4c8b2d5fb0e38d"),
}
STEMS = {
    "sitreps": "ics209-plus-wf_sitreps_1999to2020.csv",
    "incidents": "ics209-plus-wf_incidents_1999to2020.csv",
    "complexes": "ics209-plus-wf_complex_associations_1999to2020.csv",
}
DAILY_COLUMNS = [
    "INCIDENT_ID", "INCIDENT_NAME", "INCIDENT_NUMBER", "INC209R_IDENTIFIER",
    "FIRE_EVENT_ID", "START_YEAR", "INCTYP_ABBREVIATION", "COMPLEX",
    "REPORT_FROM_DATE", "REPORT_TO_DATE", "DISCOVERY_DATE", "POO_STATE",
    "POO_LATITUDE", "POO_LONGITUDE", "ACRES", "NEW_ACRES",
    "PCT_CONTAINED_COMPLETED", "PCT_CONT_COMPL_UOM_ABBREV",
    "TOTAL_PERSONNEL", "TOTAL_AERIAL", "SUPPRESSION_METHOD",
    "SUPPRESSION_METHOD_FULLNAME", "PERCENT_FS", "PERCENT_C", "PERCENT_M",
    "PERCENT_PZP", "EST_IM_COST_TO_DATE", "CRIT_RES_NEEDS_NARR",
    "IMT_MGMT_ORG_DESC", "REPORT_DAY_SPAN",
]
LIMITATIONS = [
    "ICS-209 selects large or otherwise significant incidents; it is not a representative sample of all fires. Coverage is 1999-2020, with reporting changes and gaps especially before 2006.",
    "TOTAL_PERSONNEL and TOTAL_AERIAL describe incident assignments in source reports. They do not measure available crews, unfilled requests, national workforce capacity, resource shortages, or causal suppression effectiveness.",
    "ICS-209-PLUS has source cleaning/smoothing and fractional resource values. Source zeros are retained and may include unreported values; missing-resource semantics cannot be recovered from this cleaned release.",
    "Only an incident's latest source report on each reported calendar day is retained. Gaps are not filled; source report dates have no unified UTC interpretation.",
    "Complex and member incidents can overlap. Explicit complex relationships are retained; personnel must not be summed across member/complex rows or duplicated FPA links. No national resource total is produced.",
    "Personnel over 10,000 and aerial resources over 100 are marked for manual review, not silently corrected. These are quality-review thresholds, not physical capacity limits.",
    "Reports may start after discovery; source incident summaries and resource peaks use later information. They must not be treated as ignition-time features or added to forecasting models without temporal alignment.",
    "Expected containment and growth cessation are not observed actual containment dates. This acquisition does not invent containment outcomes or alter the existing risk/load models.",
    "Prescribed-burn (RX) rows remain in the original source CSVs but are excluded from the operational Parquet analysis products. WFU and CX categories are retained and explicitly labeled.",
    "Critical-resource-needs text is retained as reported and has not been validated or classified into a shortage indicator.",
    "PCT_CONTAINED_COMPLETED can be reported in PCT, MMA, or unspecified units. The safe containment-percent column is populated only for explicit PCT units with values in 0-100.",
]


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def fetch(session: requests.Session, url: str, path: Path, maximum: int) -> None:
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with session.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError(f"Received HTML instead of data: {response.url}")
        received = 0
        with temporary.open("wb") as target:
            for chunk in response.iter_content(1024 * 1024):
                received += len(chunk)
                if received > maximum:
                    raise ValueError(f"Download exceeded the bounded size for {path.name}")
                target.write(chunk)
    temporary.replace(path)


def original_csvs(raw: Path) -> dict[str, Path]:
    """Extract only the three known CSVs; retain their exact original bytes."""
    extracted = raw / "ics209plus_v3"
    extracted.mkdir(exist_ok=True)
    outputs = {}
    with zipfile.ZipFile(raw / "ics209plus-wildfire.zip") as archive:
        for key, filename in STEMS.items():
            member = "ics209plus-wildfire/" + filename
            target = extracted / filename
            expected_size = archive.getinfo(member).file_size
            if not target.exists() or target.stat().st_size != expected_size:
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            outputs[key] = target
    return outputs


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def write_examples(root: Path, summary: pd.DataFrame, daily: pd.DataFrame, context: pd.DataFrame) -> dict:
    """Select illustrative incidents by reporting coverage, never their outcomes."""
    usable = daily.assign(
        personnel_usable=daily.TOTAL_PERSONNEL.gt(0) & ~daily.PERSONNEL_REVIEW_FLAG,
        aerial_positive=daily.TOTAL_AERIAL.gt(0) & ~daily.AERIAL_REVIEW_FLAG,
        pct_usable=daily.CONTAINMENT_PERCENT_EXPLICIT_PCT.notna(),
    ).groupby("INCIDENT_ID", observed=True).agg(
        personnel_positive_fraction=("personnel_usable", "mean"),
        aerial_positive_fraction=("aerial_positive", "mean"),
        explicit_pct_fraction=("pct_usable", "mean"),
    ).reset_index()
    candidates = summary.merge(usable, on="INCIDENT_ID", validate="one_to_one")
    span = (candidates.LAST_REPORT_DATE - candidates.FIRST_REPORT_DATE).dt.days + 1
    candidates["calendar_span_days"] = span
    candidates["report_day_fraction_of_span"] = candidates.OBSERVED_REPORT_DAYS / span
    direct = context.loc[context.OPERATIONAL_LINK_ROLE.eq("direct_incident_context") & context.IN_VALID_CONTAINMENT_COHORT].copy()
    candidates = candidates.loc[
        candidates.INCIDENT_ID.isin(direct.OPERATIONAL_INCIDENT_ID)
        & candidates.REPORTING_SCOPE.eq("standalone_or_unmapped")
        & candidates.OBSERVED_REPORT_DAYS.ge(15)
        & candidates.personnel_positive_fraction.ge(0.9)
        & candidates.aerial_positive_fraction.ge(0.5)
        & candidates.explicit_pct_fraction.ge(0.8)
        & candidates.report_day_fraction_of_span.ge(0.8)
        & candidates.PERSONNEL_REVIEW_DAYS.eq(0)
        & candidates.AERIAL_REVIEW_DAYS.eq(0)
        & candidates.POO_STATE.notna()
    ].sort_values(["OBSERVED_REPORT_DAYS", "INCIDENT_ID"], ascending=[False, True])
    chosen = candidates.drop_duplicates("POO_STATE").head(5)
    examples = []
    for _, incident in chosen.iterrows():
        linked = direct.loc[direct.OPERATIONAL_INCIDENT_ID.eq(incident.INCIDENT_ID)].sort_values("FOD_ID")
        fire = linked.iloc[0]
        examples.append({
            "incident_id": incident.INCIDENT_ID, "name": str(incident.INCIDENT_NAME).strip(),
            "source_year": int(incident.START_YEAR), "state": str(incident.POO_STATE),
            "FOD_ID": int(fire.FOD_ID), "link_role": "direct_incident_context",
            "linked_fpa_record_count": len(linked),
            "first_report_date": incident.FIRST_REPORT_DATE.date().isoformat(),
            "last_report_date": incident.LAST_REPORT_DATE.date().isoformat(),
            "observed_report_days": int(incident.OBSERVED_REPORT_DAYS),
            "calendar_span_days": int(incident.calendar_span_days),
            "report_day_fraction_of_span": float(incident.report_day_fraction_of_span),
            "personnel_positive_fraction": float(incident.personnel_positive_fraction),
            "aerial_positive_fraction": float(incident.aerial_positive_fraction),
            "explicit_pct_fraction": float(incident.explicit_pct_fraction),
            "containment_unit_label": "Reported containment/completion (%) — explicit PCT units only",
        })
    if len(examples) != 5:
        raise ValueError("Insufficient directly linked, well-reported examples in distinct states")
    payload = {
        "source_doi": "10.6084/m9.figshare.19858927.v3",
        "selection_method": "Eligible direct official FPA links in the valid CONUS containment cohort; no mapped complex/member incidents. Select by report coverage only, rank by observed report days descending then incident ID, keep one per state. No selection on containment outcome or direction of resource trends.",
        "criteria": {"minimum_observed_report_days": 15, "minimum_report_day_fraction_of_span": 0.8,
                     "minimum_personnel_positive_fraction": 0.9, "minimum_aerial_positive_fraction": 0.5,
                     "minimum_explicit_pct_fraction": 0.8, "resource_review_flag_days": 0},
        "default_incident_id": examples[0]["incident_id"], "examples": examples,
        "plot_fields": {"date": "REPORT_DATE", "personnel": "TOTAL_PERSONNEL", "aerial_resources": "TOTAL_AERIAL",
                        "burned_area_acres": "ACRES", "containment_percent": "CONTAINMENT_PERCENT_EXPLICIT_PCT"},
        "presentation_notes": ["These are illustrative well-reported incidents, not a representative sample.",
                               "Show personnel, aerial resources, and acres on separate labeled axes/panels.",
                               "Preserve gaps: reindex to calendar days without filling values before plotting lines.",
                               "Personnel/aerial values are source-reported or source-smoothed assignments; fractional values are preserved.",
                               "Personnel and fire size can rise together because both respond to fire severity. This plot does not establish that resources caused containment outcomes.",
                               "A source zero may reflect zero assignment or unreported information; do not interpret zeros as verified absence."]
    }
    (root / "reports/operational_examples.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data/raw/new_sources"
    raw.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "PreFIRE-research/0.2 (public operational-data audit)"
    metadata_path = raw / "ics209plus_metadata.json"
    fetch(session, ARTICLE, metadata_path, 1_000_000)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["id"] != 19858927 or metadata["version"] != 3:
        raise ValueError("Expected the pinned ICS-209-PLUS article version 3")
    manifest = []
    for filename, (file_id, expected_bytes, expected_md5) in FILES.items():
        target = raw / filename
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        fetch(session, url, target, expected_bytes)
        if target.stat().st_size != expected_bytes or digest(target, "md5") != expected_md5:
            raise ValueError(f"Published size/MD5 validation failed for {filename}")
        manifest.append({"path": str(target.relative_to(root)), "url": url,
                         "bytes": expected_bytes, "published_md5": expected_md5,
                         "sha256": digest(target)})
    originals = original_csvs(raw)
    for filename in originals.values():
        manifest.append({"path": str(filename.relative_to(root)), "bytes": filename.stat().st_size,
                         "sha256": digest(filename), "source_archive": "ics209plus-wildfire.zip"})

    source = pd.read_csv(originals["sitreps"], usecols=DAILY_COLUMNS, low_memory=False)
    incidents = pd.read_csv(originals["incidents"], low_memory=False)
    complexes = pd.read_csv(originals["complexes"], low_memory=False)
    source_rows = len(source)
    for frame in [source, incidents]:
        frame["INCIDENT_ID"] = normalize_id(frame.INCIDENT_ID)
    if incidents.INCIDENT_ID.duplicated().any():
        raise ValueError("Incident summary contains duplicate IDs")
    retained_types = ["WF", "WFU", "CX"]
    incidents = incidents.loc[incidents.INCTYP_ABBREVIATION.isin(retained_types)].copy()
    excluded_type_rows = int((~source.INCTYP_ABBREVIATION.isin(retained_types)).sum())
    source = source.loc[source.INCTYP_ABBREVIATION.isin(retained_types)].copy()
    summary_mismatch = source.loc[~source.INCIDENT_ID.isin(incidents.INCIDENT_ID)]
    excluded_summary_rows = len(summary_mismatch)
    excluded_summary_ids = sorted(summary_mismatch.INCIDENT_ID.unique().tolist())
    # The published daily table contains a few IDs absent from the incident
    # summary, and one incident whose final summary classifies it as RX. Keep
    # those original rows intact in CSV and disclose their exclusion explicitly.
    source = source.loc[source.INCIDENT_ID.isin(incidents.INCIDENT_ID)].copy()
    for column in ["REPORT_TO_DATE", "REPORT_FROM_DATE", "DISCOVERY_DATE"]:
        source[column] = pd.to_datetime(source[column], format="mixed", errors="coerce")
    if source.REPORT_TO_DATE.isna().any() or source.INCIDENT_ID.isna().any():
        raise ValueError("Source reports require a valid date and incident ID")
    source["REPORT_DATE"] = source.REPORT_TO_DATE.dt.normalize()
    counts = source.groupby(["INCIDENT_ID", "REPORT_DATE"], observed=True).size().rename("SOURCE_REPORTS_ON_DAY")
    source["SOURCE_ROW_ORDER"] = np.arange(len(source))
    daily = source.sort_values(["INCIDENT_ID", "REPORT_TO_DATE", "SOURCE_ROW_ORDER"], kind="stable").drop_duplicates(["INCIDENT_ID", "REPORT_DATE"], keep="last")
    daily = daily.merge(counts.reset_index(), on=["INCIDENT_ID", "REPORT_DATE"], how="left", validate="one_to_one")
    daily.drop(columns="SOURCE_ROW_ORDER", inplace=True)
    for column in ["TOTAL_PERSONNEL", "TOTAL_AERIAL", "ACRES", "NEW_ACRES", "PCT_CONTAINED_COMPLETED", "EST_IM_COST_TO_DATE"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["PERSONNEL_REVIEW_FLAG"] = daily.TOTAL_PERSONNEL.lt(0) | daily.TOTAL_PERSONNEL.gt(10_000)
    daily["AERIAL_REVIEW_FLAG"] = daily.TOTAL_AERIAL.lt(0) | daily.TOTAL_AERIAL.gt(100)
    daily["SOURCE_PERSONNEL_ZERO"] = daily.TOTAL_PERSONNEL.eq(0)
    daily["SOURCE_AERIAL_ZERO"] = daily.TOTAL_AERIAL.eq(0)
    daily["SOURCE_PERSONNEL_FRACTIONAL"] = daily.TOTAL_PERSONNEL.notna() & daily.TOTAL_PERSONNEL.mod(1).ne(0)
    daily["CONTAINMENT_PERCENT_EXPLICIT_PCT"] = daily.PCT_CONTAINED_COMPLETED.where(
        daily.PCT_CONT_COMPL_UOM_ABBREV.eq("PCT") & daily.PCT_CONTAINED_COMPLETED.between(0, 100)
    )
    for column in ["CPLX_INCIDENT_ID", "MEMBER_INCIDENT_ID"]:
        complexes[column] = normalize_id(complexes[column])
    edges = complexes[["CPLX_INCIDENT_ID", "MEMBER_INCIDENT_ID"]].dropna().drop_duplicates()
    edges = edges.loc[edges.CPLX_INCIDENT_ID.ne(edges.MEMBER_INCIDENT_ID)]
    parents = edges.groupby("MEMBER_INCIDENT_ID").CPLX_INCIDENT_ID.agg(lambda group: " | ".join(sorted(set(group))))
    daily["ASSOCIATED_COMPLEX_IDS"] = daily.INCIDENT_ID.map(parents)
    daily["HAS_COMPLEX_PARENT"] = daily.INCIDENT_ID.isin(edges.MEMBER_INCIDENT_ID)
    daily["HAS_MAPPED_MEMBERS"] = daily.INCIDENT_ID.isin(edges.CPLX_INCIDENT_ID)

    aggregate = daily.groupby("INCIDENT_ID", observed=True).agg(
        FIRST_REPORT_DATE=("REPORT_DATE", "min"), LAST_REPORT_DATE=("REPORT_DATE", "max"),
        OBSERVED_REPORT_DAYS=("REPORT_DATE", "size"),
        SOURCE_MEDIAN_PERSONNEL=("TOTAL_PERSONNEL", "median"), SOURCE_PEAK_PERSONNEL=("TOTAL_PERSONNEL", "max"),
        SOURCE_MEDIAN_AERIAL=("TOTAL_AERIAL", "median"), SOURCE_PEAK_AERIAL=("TOTAL_AERIAL", "max"),
        PERSONNEL_REVIEW_DAYS=("PERSONNEL_REVIEW_FLAG", "sum"), AERIAL_REVIEW_DAYS=("AERIAL_REVIEW_FLAG", "sum"),
        PERSONNEL_ZERO_REPORT_DAYS=("SOURCE_PERSONNEL_ZERO", "sum"),
    ).reset_index()
    summary_columns = ["INCIDENT_ID", "INCIDENT_NAME", "INCIDENT_NUMBER", "START_YEAR", "INCTYP_ABBREVIATION",
                       "COMPLEX", "DISCOVERY_DATE", "FINAL_ACRES", "POO_STATE", "POO_LATITUDE", "POO_LONGITUDE",
                       "FUEL_MODEL", "SUPPRESSION_METHOD", "SUP_METHOD_INITIAL", "SUP_METHOD_FINAL", "SUP_PERCENT_FS"]
    summary = incidents[summary_columns].merge(aggregate, how="left", on="INCIDENT_ID", validate="one_to_one")
    summary["ASSOCIATED_COMPLEX_IDS"] = summary.INCIDENT_ID.map(parents)
    summary["HAS_COMPLEX_PARENT"] = summary.INCIDENT_ID.isin(edges.MEMBER_INCIDENT_ID)
    summary["HAS_MAPPED_MEMBERS"] = summary.INCIDENT_ID.isin(edges.CPLX_INCIDENT_ID)
    summary["REPORTING_SCOPE"] = np.where(summary.HAS_MAPPED_MEMBERS | summary.COMPLEX.eq(True), "complex", np.where(summary.HAS_COMPLEX_PARENT, "complex_member", "standalone_or_unmapped"))

    fpa_columns = ["FOD_ID", "FIRE_YEAR", "FIRE_NAME", "FIRE_SIZE", "STATE", "DISCOVERY_DATE", "CONT_DATE",
                   "ICS_209_PLUS_INCIDENT_JOIN_ID", "ICS_209_PLUS_COMPLEX_JOIN_ID"]
    fpa = pd.read_parquet(root / "data/fpa_fod_all.parquet", columns=fpa_columns)
    incident_ids = set(summary.INCIDENT_ID)
    fpa["DIRECT_ICS_MATCH"] = normalize_id(fpa.ICS_209_PLUS_INCIDENT_JOIN_ID).isin(incident_ids)
    fpa["COMPLEX_ICS_MATCH"] = normalize_id(fpa.ICS_209_PLUS_COMPLEX_JOIN_ID).isin(incident_ids)
    fpa["ANY_ICS_MATCH"] = fpa.DIRECT_ICS_MATCH | fpa.COMPLEX_ICS_MATCH
    eligible = fpa.FIRE_YEAR.between(1999, 2020)
    coverage = []
    for year, group in fpa.groupby("FIRE_YEAR", observed=True):
        coverage.append({"year": int(year), "fpa_records": len(group),
                         "fpa_direct_join_id_present": int(group.ICS_209_PLUS_INCIDENT_JOIN_ID.notna().sum()),
                         "direct_exact_matches": int(group.DIRECT_ICS_MATCH.sum()), "complex_exact_matches": int(group.COMPLEX_ICS_MATCH.sum()),
                         "any_exact_matches": int(group.ANY_ICS_MATCH.sum()),
                         "matched_fraction_all_fires": float(group.ANY_ICS_MATCH.mean())})
    links = []
    for role, column, mask in [("direct", "ICS_209_PLUS_INCIDENT_JOIN_ID", fpa.DIRECT_ICS_MATCH),
                               ("complex", "ICS_209_PLUS_COMPLEX_JOIN_ID", fpa.COMPLEX_ICS_MATCH)]:
        part = fpa.loc[mask, ["FOD_ID", "FIRE_YEAR", column]].rename(columns={column: "INCIDENT_ID"})
        part["LINK_ROLE"] = role
        links.append(part)
    links = pd.concat(links, ignore_index=True).drop_duplicates()
    context = fpa.loc[fpa.ANY_ICS_MATCH].copy()
    # An explicitly matched complex provides shared context; never allocate its
    # personnel repeatedly to member fires or interpret these rows as unique crews.
    context["OPERATIONAL_INCIDENT_ID"] = context.ICS_209_PLUS_COMPLEX_JOIN_ID.where(context.COMPLEX_ICS_MATCH, context.ICS_209_PLUS_INCIDENT_JOIN_ID)
    context["OPERATIONAL_LINK_ROLE"] = np.where(context.COMPLEX_ICS_MATCH, "shared_complex_context", "direct_incident_context")
    context["FPA_CONTAINMENT_DATE_PRESENT"] = context.CONT_DATE.notna()
    renamed = summary.rename(columns={column: "ICS_" + column for column in summary if column != "INCIDENT_ID"})
    context = context.merge(renamed, how="left", left_on="OPERATIONAL_INCIDENT_ID", right_on="INCIDENT_ID", validate="many_to_one").drop(columns="INCIDENT_ID")
    context["FIRST_REPORT_LAG_DAYS"] = (context.ICS_FIRST_REPORT_DATE - pd.to_datetime(context.DISCOVERY_DATE)).dt.total_seconds() / 86400
    context["FIRST_REPORT_PRECEDES_FPA_DISCOVERY"] = context.FIRST_REPORT_LAG_DAYS.lt(0)
    modeled_ids = pd.read_parquet(root / "data/fires_containment.parquet", columns=["FOD_ID"]).FOD_ID
    context["IN_VALID_CONTAINMENT_COHORT"] = context.FOD_ID.isin(modeled_ids)

    output_frames = {
        "operational_incident_daily.parquet": daily,
        "operational_incidents.parquet": summary,
        "operational_complex_links.parquet": edges,
        "operational_fpa_links.parquet": links,
        "operational_fpa_context.parquet": context,
    }
    outputs = []
    for name, frame in output_frames.items():
        output = root / "data" / name
        frame.to_parquet(output, index=False)
        outputs.append({"path": str(output.relative_to(root)), "rows": len(frame), "columns": list(frame.columns),
                        "bytes": output.stat().st_size, "sha256": digest(output)})
    assert not daily.duplicated(["INCIDENT_ID", "REPORT_DATE"]).any()
    assert not summary.INCIDENT_ID.duplicated().any()
    assert not context.FOD_ID.duplicated().any()
    assert not links.duplicated(["FOD_ID", "LINK_ROLE", "INCIDENT_ID"]).any()
    assert not daily.INCTYP_ABBREVIATION.eq("RX").any()
    assert daily.INCIDENT_ID.isin(summary.INCIDENT_ID).all()
    examples = write_examples(root, summary, daily, context)
    report = {
        "status": "acquired_and_validated_separate_operational_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"name": metadata["title"], "version": 3, "doi": "10.6084/m9.figshare.19858927.v3",
                   "landing_page": LANDING, "metadata_api": ARTICLE, "methods_paper": PAPER,
                   "license": metadata["license"], "attribution": "St. Denis, Short, McConnell, Cook, Buckland, Mietkiewicz and Balch; ICS-209-PLUS v3 (2023)."},
        "source_manifest": manifest,
        "raw_counts": {"sitreps": source_rows, "incident_summaries": int(pd.read_csv(originals["incidents"], usecols=["INCIDENT_ID"]).shape[0]),
                       "complex_association_rows": len(complexes)},
        "retained": {"incident_types": retained_types, "daily_incident_records": len(daily), "incident_summaries": len(summary),
                     "source_reports_excluded_by_incident_type": excluded_type_rows,
                     "source_reports_excluded_missing_or_rx_incident_summary": excluded_summary_rows,
                     "excluded_missing_or_rx_summary_ids": excluded_summary_ids,
                     "duplicate_or_earlier_same_day_reports_removed": int(len(source) - len(daily)),
                     "first_report_date": daily.REPORT_DATE.min().isoformat(), "last_report_date": daily.REPORT_DATE.max().isoformat(),
                     "nonself_complex_member_edges": len(edges)},
        "quality": {"daily_personnel_zero": int(daily.SOURCE_PERSONNEL_ZERO.sum()),
                    "daily_personnel_fractional": int(daily.SOURCE_PERSONNEL_FRACTIONAL.sum()),
                    "daily_personnel_review_flag": int(daily.PERSONNEL_REVIEW_FLAG.sum()),
                    "daily_aerial_review_flag": int(daily.AERIAL_REVIEW_FLAG.sum()),
                    "summaries_without_retained_daily_reports": int(summary.OBSERVED_REPORT_DAYS.isna().sum()),
                    "daily_explicit_pct_containment_values": int(daily.CONTAINMENT_PERCENT_EXPLICIT_PCT.notna().sum()),
                    "daily_containment_units": {str(key): int(value) for key, value in daily.PCT_CONT_COMPL_UOM_ABBREV.fillna("Unspecified").value_counts().items()},
                    "max_source_personnel": float(daily.TOTAL_PERSONNEL.max()), "max_source_aerial": float(daily.TOTAL_AERIAL.max())},
        "fpa_linkage": {"method": "Exact official direct/complex join IDs only; no fuzzy matches or old-edition numeric FOD IDs.",
                        "fpa_records_all_years": len(fpa), "fpa_records_1999_2020": int(eligible.sum()),
                        "direct_exact_matches": int(fpa.DIRECT_ICS_MATCH.sum()), "complex_exact_matches": int(fpa.COMPLEX_ICS_MATCH.sum()),
                        "unique_fpa_records_with_operational_context": len(context),
                        "fraction_1999_2020_with_context": float(fpa.loc[eligible, "ANY_ICS_MATCH"].mean()),
                        "context_records_in_valid_containment_cohort": int(context.IN_VALID_CONTAINMENT_COHORT.sum()),
                        "matched_median_final_acres": float(context.FIRE_SIZE.median()),
                        "all_1999_2020_median_final_acres": float(fpa.loc[eligible, "FIRE_SIZE"].median()),
                        "median_first_report_lag_days": float(context.FIRST_REPORT_LAG_DAYS.median()),
                        "first_report_precedes_fpa_discovery_records": int(context.FIRST_REPORT_PRECEDES_FPA_DISCOVERY.sum()),
                        "negative_lag_note": "Can reflect earlier shared-complex reporting or disagreement between source dates. Inspect before longitudinal alignment; no date corrections were invented.",
                        "by_year": coverage},
        "outputs": outputs,
        "illustrative_examples": {"path": "reports/operational_examples.json", "count": len(examples["examples"]), "selection": "reporting coverage and distinct states; not containment outcome"},
        "join_contract": {"daily_key": ["INCIDENT_ID", "REPORT_DATE"], "incident_key": ["INCIDENT_ID"],
                          "fpa_context_key": ["FOD_ID"], "link_edges_key": ["FOD_ID", "LINK_ROLE", "INCIDENT_ID"],
                          "resource_units": {"TOTAL_PERSONNEL": "Source incident personnel assignments (may be smoothed); not unique people over time.",
                                             "TOTAL_AERIAL": "Source aerial-support resource assignments; not flight hours or available fleet."},
                          "prediction_integration": "Not added to existing model inputs. Use as historical incident-explorer context; temporal/selection analysis is required before predictive use."},
        "checks": {"published_archive_size_and_md5": True, "daily_key_unique": True, "incident_key_unique": True,
                   "fpa_context_key_unique": True, "prescribed_burns_excluded": True, "raw_csvs_preserved": True},
        "limitations": LIMITATIONS,
    }
    (reports / "data_upgrade.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"outputs": [{"path": out["path"], "rows": out["rows"]} for out in outputs],
                      "linkage": {key: value for key, value in report["fpa_linkage"].items() if key != "by_year"},
                      "quality": report["quality"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
