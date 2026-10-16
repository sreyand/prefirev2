"""Engineer descriptive controls without asserting causal or policy effects."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def observed_concurrency(fires):
    """Daily inclusive discovery-to-containment counts for the observed cohort."""
    starts = pd.to_datetime(fires.DISCOVERY_DATE).dt.normalize().value_counts()
    ends = (pd.to_datetime(fires.CONT_DATE).dt.normalize() + pd.Timedelta(days=1)).value_counts()
    days = pd.date_range(starts.index.min(), ends.index.max(), freq="D")
    counts = starts.reindex(days, fill_value=0).sub(ends.reindex(days, fill_value=0)).cumsum()
    assert counts.min() >= 0 and counts.iloc[-1] == 0
    return counts.rename("OBSERVED_UNCONTAINED_FIRES")


def main():
    fires = pd.read_parquet(DATA / "enriched_fpa_fod.parquet")
    fires["STATE"] = fires.STATE.astype("string").str.strip().str.upper()
    fires["LOG_ACRES"] = np.log(fires.FIRE_SIZE)
    fires["DAY_OF_YEAR"] = fires.DISCOVERY_DAY.dt.dayofyear
    # Boundaries: [0,1), [1,10), [10,100), [100,1000), [1000,infinity).
    fires["SIZE_BIN"] = pd.cut(fires.FIRE_SIZE, [0, 1, 10, 100, 1000, np.inf], right=False,
                               labels=["Micro", "Small", "Medium", "Large", "1000+ acres"])
    fires["CAUSE_TYPE"] = fires.NWCG_CAUSE_CLASSIFICATION.fillna("Unknown")
    national = observed_concurrency(fires)
    daily = national.to_frame().rename_axis("DAY").reset_index()
    # Retain all occurrence records for discovery counts; do not impute their end dates.
    all_fires = pd.read_parquet(DATA / "fpa_fod_all.parquet", columns=["DISCOVERY_DATE", "LATITUDE", "LONGITUDE", "FIRE_SIZE"])
    all_fires = all_fires.loc[all_fires.LATITUDE.between(24, 50) & all_fires.LONGITUDE.between(-125, -66) & all_fires.FIRE_SIZE.gt(0)]
    discoveries = pd.to_datetime(all_fires.DISCOVERY_DATE).dt.normalize().value_counts()
    daily["ALL_REPORTED_DISCOVERIES"] = daily.DAY.map(discoveries).fillna(0).astype(int)
    daily.to_parquet(DATA / "daily_observed_load.parquet", index=False)
    fires["OBSERVED_UNCONTAINED_FIRES"] = fires.DISCOVERY_DAY.map(national)
    fires["ALL_REPORTED_DISCOVERIES_ON_DAY"] = fires.DISCOVERY_DAY.map(discoveries)
    fires["OBSERVED_UNCONTAINED_SAME_STATE"] = 0
    for state, group in fires.groupby("STATE"):
        counts = observed_concurrency(group)
        fires.loc[group.index, "OBSERVED_UNCONTAINED_SAME_STATE"] = group.DISCOVERY_DAY.map(counts).to_numpy()
    fires["OBSERVED_LOAD_TIER"] = pd.cut(fires.OBSERVED_UNCONTAINED_FIRES, [0, 50, 150, 300, 500, np.inf],
                                        right=False, labels=["Low", "Moderate", "High", "Severe", "Extreme"])
    # A descriptive size/duration proxy, not an observed dispatch-level attack outcome.
    fires["SIZE_OR_DURATION_ESCAPE_PROXY"] = (fires.CONTAINMENT_DAYS.gt(3) | fires.FIRE_SIZE.gt(100)).astype("int8")
    fires.to_parquet(DATA / "processed_fires.parquet", index=False)
    notes = {
        "rows": len(fires),
        "concurrency_scope": "Only fires with valid reported containment dates and 0–365-day intervals; includes the focal fire.",
        "concurrency_endpoint": "Containment, not fire-out; not a census of active U.S. wildfires or resource assignments.",
        "causal_scope": "Final acreage and observed concurrency may be outcomes of suppression itself; they are not guaranteed pre-treatment confounders.",
        "escape_proxy": "Duration >3 days or final acreage >100 is a descriptive proxy, not verified initial-attack failure.",
        "size_boundaries": "Left-inclusive bins; 1000 acres belongs to 1000+ acres.",
    }
    (ROOT / "reports" / "feature_notes.json").write_text(json.dumps(notes, indent=2))
    print(json.dumps(notes, indent=2), flush=True)


if __name__ == "__main__":
    main()
