"""Check the saved data contract using real acquisition outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    base = pd.read_parquet(DATA / "fires_containment.parquet", columns=["FOD_ID", "CONTAINMENT_DAYS", "CONTAINMENT_DAYS_DATE_ONLY"])
    columns = ["FOD_ID", "FIRE_YEAR", "DISCOVERY_DAY", "CONTAINMENT_DAYS", "LATITUDE", "LONGITUDE", "FIRE_SIZE",
               "WEATHER_GRID_ID", "WEATHER_DISTANCE_KM", "DISTANCE_TO_PLACE_CENTROID_KM",
               "temperature_2m_max", "wind_speed_10m_max", "precipitation_sum", "vapour_pressure_deficit_max",
               "ANTECEDENT_PRECIP_30D_MM", "Total (USD)", "TOTAL_SUPPRESSION_2024_USD"]
    joined = pd.read_parquet(DATA / "enriched_fpa_fod.parquet", columns=columns)
    assert len(base) >= 150000
    assert base.FOD_ID.is_unique and joined.FOD_ID.is_unique
    assert np.array_equal(np.sort(base.FOD_ID), np.sort(joined.FOD_ID)), "Enrichment lost or duplicated IDs"
    required = [c for c in columns if c not in ["Total (USD)", "TOTAL_SUPPRESSION_2024_USD"]]
    assert joined[required].notna().all().all()
    assert joined.CONTAINMENT_DAYS.between(0, 365).all()
    assert joined.LATITUDE.between(24, 50).all() and joined.LONGITUDE.between(-125, -66).all()
    assert joined.FIRE_SIZE.gt(0).all()
    assert joined.WEATHER_DISTANCE_KM.ge(0).all() and joined.DISTANCE_TO_PLACE_CENTROID_KM.ge(0).all()
    assert joined.wind_speed_10m_max.ge(0).all()
    assert joined.precipitation_sum.ge(0).all() and joined.ANTECEDENT_PRECIP_30D_MM.ge(0).all()
    assert joined.vapour_pressure_deficit_max.ge(0).all()
    assert joined.FIRE_YEAR.between(1992, 2024).all()
    assert joined.loc[joined.FIRE_YEAR.le(2023), "Total (USD)"].notna().all()
    assert joined.loc[joined.FIRE_YEAR.eq(2024), "Total (USD)"].isna().all()
    weather = pd.read_parquet(DATA / "era5_daily.parquet")
    assert not weather.duplicated(["WEATHER_GRID_ID", "DISCOVERY_DAY"]).any()
    assert weather.WEATHER_GRID_ID.nunique() == 25
    for _, group in weather.groupby("WEATHER_GRID_ID"):
        expected = pd.date_range("1991-12-01", "2024-12-31")
        assert np.array_equal(group.DISCOVERY_DAY.sort_values().to_numpy(), expected.to_numpy())
    processed = pd.read_parquet(DATA / "processed_fires.parquet", columns=["FOD_ID", "SIZE_BIN", "OBSERVED_UNCONTAINED_FIRES", "OBSERVED_UNCONTAINED_SAME_STATE"])
    assert len(processed) == len(joined) and processed.FOD_ID.is_unique
    assert processed.notna().all().all()
    assert processed.OBSERVED_UNCONTAINED_FIRES.ge(processed.OBSERVED_UNCONTAINED_SAME_STATE).all()
    assert processed.OBSERVED_UNCONTAINED_SAME_STATE.ge(1).all()
    daily = pd.read_parquet(DATA / "daily_observed_load.parquet")
    assert daily.DAY.is_unique and daily.OBSERVED_UNCONTAINED_FIRES.ge(0).all()
    assert daily.OBSERVED_UNCONTAINED_FIRES.iloc[-1] == 0
    assert daily.OBSERVED_UNCONTAINED_FIRES.sum() == (base.CONTAINMENT_DAYS_DATE_ONLY + 1).sum()
    sources = json.loads((DATA / "raw" / "sources.json").read_text())
    for relative, details in sources.items():
        with (ROOT / relative).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == details["sha256"], relative
    result = {"status": "passed", "enriched_records": len(joined), "weather_rows": len(weather),
              "weather_locations": 25, "source_checksums_verified": len(sources),
              "raw_fire_records": pq.ParquetFile(DATA / "fpa_fod_all.parquet").metadata.num_rows,
              "missing_cost_years": sorted(joined.loc[joined["Total (USD)"].isna(), "FIRE_YEAR"].unique().tolist())}
    (ROOT / "reports" / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
