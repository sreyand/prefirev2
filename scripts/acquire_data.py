"""Reproducible acquisition of PreFIRE research inputs. Run from project root."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import pyogrio
import requests
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
REPORTS = ROOT / "reports"
for folder in (RAW, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "PreFIRE-research/0.1 (noncommercial academic data preparation)"
MANIFEST = RAW / "sources.json"
sources = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def require_same_source(source, destination):
    """Refuse stale caches when a user supplies a different input under the same name."""
    if destination.exists() and source.resolve() != destination.resolve():
        with source.open("rb") as incoming, destination.open("rb") as cached:
            if hashlib.file_digest(incoming, "sha256").digest() != hashlib.file_digest(cached, "sha256").digest():
                raise ValueError(f"Source changed: {source}. Use a fresh data directory before rebuilding from a different input.")


def record(path, url, **details):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    sources[str(path.relative_to(ROOT))] = dict(
        source=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
        bytes=path.stat().st_size,
        sha256=digest, **details)
    MANIFEST.write_text(json.dumps(sources, indent=2), encoding="utf-8")


def fetch(url, path, params=None):
    if path.exists():
        return path
    attempt = 0
    rate_wait_started = None
    while attempt < 5:
        try:
            response = SESSION.get(url, params=params, timeout=180)
            if response.status_code == 429:
                reason = response.json().get("reason", response.text) if "json" in response.headers.get("Content-Type", "") else response.text
                if "daily" in reason.lower() or "monthly" in reason.lower():
                    raise RuntimeError(f"Quota requires a later-day resume: {reason}")
                if rate_wait_started is None:
                    rate_wait_started = time.monotonic()
                if time.monotonic() - rate_wait_started > 4200:
                    raise RuntimeError(f"Rate limit has persisted over 70 minutes: {reason}")
                delay = 300 if "hourly" in reason.lower() else 60
                print(f"Rate limit: {reason} Waiting {delay}s; cached downloads retained.", flush=True)
                for _ in range(delay // 60):
                    time.sleep(60)
                continue
            if response.status_code >= 500:
                delay = max(15 * (attempt + 1), int(response.headers.get("Retry-After", "0")))
                print(f"Retry {response.status_code}: waiting {delay}s", flush=True)
                time.sleep(delay)
                attempt += 1
                continue
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError(f"HTML received instead of data: {response.url}")
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(response.content)
            temp.replace(path)
            record(path, response.url)
            return path
        except (requests.RequestException, ValueError):
            if 'response' in locals() and 400 <= response.status_code < 500 and response.status_code != 429:
                raise
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
            attempt += 1
    raise RuntimeError(f"Download exhausted retries: {url}")


def load_fires(zip_path):
    destination = RAW / "S_USA.Fire_FPA_FOD_7th_Fires.gdb.zip"
    require_same_source(zip_path, destination)
    if not destination.exists():
        shutil.copy2(zip_path, destination)
        record(destination, str(zip_path), origin="User-supplied USFS FPA-FOD seventh edition")
    gdb = RAW / "S_USA.Fire_FPA_FOD_7th_Fires.gdb"
    if not gdb.exists():
        with zipfile.ZipFile(destination) as archive:
            for member in archive.namelist():
                target = (RAW / member).resolve()
                if not target.is_relative_to(RAW.resolve()):
                    raise ValueError("Unsafe archive member")
            archive.extractall(RAW)
    output = DATA / "fpa_fod_all.parquet"
    if output.exists():
        return pd.read_parquet(output)
    layers = pyogrio.list_layers(gdb)
    print("Geodatabase layers:", layers.tolist(), flush=True)
    layer = next(str(name) for name, _ in layers if "Fires" in name)
    _, table = pyogrio.read_arrow(gdb, layer=layer, read_geometry=False)
    fires = table.to_pandas()
    fires.columns = fires.columns.str.upper()
    fires.to_parquet(output, index=False)
    print(f"Imported {len(fires):,} fire records", flush=True)
    return fires


def profile(fires):
    f = fires.copy()
    for name in ("DISCOVERY_DATE", "CONT_DATE"):
        f[name] = pd.to_datetime(f[name], errors="coerce")
    days = (f.CONT_DATE.dt.normalize() - f.DISCOVERY_DATE.dt.normalize()).dt.days
    conus = f.LATITUDE.between(24, 50) & f.LONGITUDE.between(-125, -66)
    valid = conus & f.FIRE_SIZE.gt(0) & days.between(0, 365)
    yearly = f.assign(HAS_CONTAINMENT=f.CONT_DATE.notna(), VALID_DURATION=valid).groupby("FIRE_YEAR").agg(
        records=("FOD_ID", "size"), containment_dates=("HAS_CONTAINMENT", "sum"),
        valid_conus_durations=("VALID_DURATION", "sum"))
    yearly["containment_date_fraction"] = yearly.containment_dates / yearly.records
    yearly.to_csv(REPORTS / "fire_coverage_by_year.csv")
    agency = f.assign(HAS_CONTAINMENT=f.CONT_DATE.notna(), VALID_DURATION=valid).groupby(
        ["FIRE_YEAR", "NWCG_REPORTING_AGENCY"], dropna=False).agg(
            records=("FOD_ID", "size"), containment_dates=("HAS_CONTAINMENT", "sum"),
            valid_conus_durations=("VALID_DURATION", "sum"))
    agency["containment_date_fraction"] = agency.containment_dates / agency.records
    agency.to_csv(REPORTS / "fire_coverage_by_year_agency.csv")
    result = f.loc[valid].copy()
    result["CONTAINMENT_DAYS_DATE_ONLY"] = days.loc[valid]
    # Source clocks have no declared timezone. Use same-record wall clocks only,
    # preserve date-only durations when either time is absent/invalid.
    def clock(series):
        text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(4)
        hh = pd.to_numeric(text.str[:2], errors="coerce")
        mm = pd.to_numeric(text.str[2:], errors="coerce")
        okay = (hh.between(0, 23) & mm.between(0, 59)) | ((hh == 24) & (mm == 0))
        return (hh * 60 + mm).where(okay)
    dclock, cclock = clock(result.DISCOVERY_TIME), clock(result.CONT_TIME)
    precise = result.CONTAINMENT_DAYS_DATE_ONLY + (cclock - dclock) / 1440
    invalid_timed = dclock.notna() & cclock.notna() & ~precise.between(0, 365)
    invalid_timed_count = int(invalid_timed.sum())
    result = result.loc[~invalid_timed].copy()
    precise = precise.loc[result.index]
    dclock, cclock = dclock.loc[result.index], cclock.loc[result.index]
    good = precise.between(0, 365) & dclock.notna() & cclock.notna()
    result["CONTAINMENT_DAYS"] = precise.where(good, result.CONTAINMENT_DAYS_DATE_ONLY)
    result["DURATION_RESOLUTION"] = np.where(good, "wall_clock_minutes_timezone_unspecified", "calendar_days")
    result["STATE"] = result.STATE.astype("string").str.strip().str.upper()
    result["DISCOVERY_DAY"] = result.DISCOVERY_DATE.dt.normalize()
    result.groupby(["FIRE_YEAR", "NWCG_REPORTING_AGENCY", "DURATION_RESOLUTION"], dropna=False).size().rename("records").to_csv(
        REPORTS / "duration_resolution_by_year_agency.csv")
    summary = dict(total_records=len(f), valid_conus_containment_records=len(result),
                   min_year=int(f.FIRE_YEAR.min()), max_year=int(f.FIRE_YEAR.max()),
                   missing_containment_dates=int(f.CONT_DATE.isna().sum()),
                   invalid_timed_intervals_excluded=invalid_timed_count,
                   timed_durations=int(good.sum()))
    (REPORTS / "acquisition_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return result


def cpi_costs(csv_path):
    destination = RAW / "nifc_suppression_costs.csv"
    require_same_source(csv_path, destination)
    if not destination.exists():
        shutil.copy2(csv_path, destination)
        record(destination, str(csv_path), official_source="https://www.nifc.gov/fire-information/statistics/suppression-costs")
    costs = pd.read_csv(destination)
    costs = costs.loc[pd.to_numeric(costs.Year, errors="coerce").notna()].copy()
    costs["Year"] = costs.Year.astype(int)
    assert costs.Year.is_unique
    assert np.allclose(costs["Forest Service (USD)"] + costs["DOI Agencies (USD)"], costs["Total (USD)"])
    try:
        path = fetch("https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems", RAW / "bls_all_items.txt")
        cpi = pd.read_csv(path, sep="\t")
        cpi.columns = cpi.columns.str.strip()
        cpi = cpi.loc[cpi.series_id.str.strip().eq("CUUR0000SA0") & cpi.period.str.strip().eq("M13"), ["year", "value"]]
        cpi.columns = ["Year", "CPI_U_ANNUAL"]
    except Exception as error:
        print(f"BLS bulk unavailable ({error}); using BLS public API", flush=True)
        pieces = []
        for year in range(1985, 2025, 10):
            path = RAW / f"bls_cpi_{year}_{year+9}.json"
            if not path.exists():
                request = {"seriesid": ["CUUR0000SA0"], "startyear": str(year), "endyear": str(year+9)}
                response = SESSION.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", json=request, timeout=90)
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") != "REQUEST_SUCCEEDED":
                    raise ValueError(payload)
                path.write_text(json.dumps(payload))
                record(path, response.url, request=request)
            payload = json.loads(path.read_text())
            pieces.extend(payload["Results"]["series"][0]["data"])
        monthly = pd.DataFrame(pieces)
        monthly = monthly.loc[monthly.period.ne("M13")].copy()
        monthly["Year"] = monthly.year.astype(int)
        monthly["CPI"] = pd.to_numeric(monthly.value)
        assert not monthly.duplicated(["Year", "period"]).any()
        groups = monthly.groupby("Year").CPI.agg(["mean", "count"])
        cpi = groups.loc[groups["count"].eq(12), ["mean"]].reset_index().rename(columns={"mean": "CPI_U_ANNUAL"})
    cpi.to_csv(DATA / "cpi_annual.csv", index=False)
    base = float(cpi.loc[cpi.Year.eq(2024), "CPI_U_ANNUAL"].iloc[0])
    costs = costs.merge(cpi, on="Year", how="left", validate="one_to_one")
    assert costs.CPI_U_ANNUAL.notna().all()
    costs["TOTAL_SUPPRESSION_2024_USD"] = costs["Total (USD)"] * base / costs.CPI_U_ANNUAL
    costs.to_parquet(DATA / "nifc_costs_real.parquet", index=False)
    return costs


def places(fires):
    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_place_national.zip"
    path = fetch(url, RAW / "2024_Gaz_place_national.zip")
    with zipfile.ZipFile(path) as archive:
        p = pd.read_csv(archive.open(archive.namelist()[0]), sep="\t", dtype={"GEOID": str})
    p.columns = p.columns.str.strip()
    p = p.loc[p.INTPTLAT.between(24, 50) & p.INTPTLONG.between(-125, -66)].reset_index(drop=True)
    p.to_parquet(DATA / "census_places_2024.parquet", index=False)
    def xyz(lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    distance, index = cKDTree(xyz(p.INTPTLAT, p.INTPTLONG)).query(xyz(fires.LATITUDE, fires.LONGITUDE))
    fires["DISTANCE_TO_PLACE_CENTROID_KM"] = 2 * 6371.0088 * np.arcsin(np.clip(distance / 2, 0, 1))
    fires["NEAREST_PLACE_GEOID"] = p.GEOID.to_numpy()[index]
    return fires


def weather(fires, count):
    directory = RAW / "weather"
    directory.mkdir(exist_ok=True)
    gridpath = DATA / "weather_centroids.csv"
    coordinates = fires[["LATITUDE", "LONGITUDE"]].to_numpy()
    scale = np.array([1.0, np.cos(np.radians(38))])
    if gridpath.exists():
        grid = pd.read_csv(gridpath)
        assert len(grid) == count, "Existing grid differs; use a fresh project for a new grid."
    else:
        rng = np.random.default_rng(20261014)
        sample = coordinates[rng.choice(len(coordinates), min(100000, len(coordinates)), replace=False)]
        centers, _ = kmeans2(sample * scale, count, iter=40, minit="++", rng=rng)
        centers = centers / scale
        centers = centers[np.lexsort((centers[:, 1], centers[:, 0]))]
        grid = pd.DataFrame(centers, columns=["latitude", "longitude"])
        grid.insert(0, "WEATHER_GRID_ID", range(count))
        grid.to_csv(gridpath, index=False)
    # Great-circle nearest centroid matching, not raw longitude degrees.
    def xyz(coords):
        lat, lon = np.radians(coords).T
        return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    distance, index = cKDTree(xyz(grid[["latitude", "longitude"]].to_numpy())).query(xyz(coordinates))
    fires["WEATHER_GRID_ID"] = index
    fires["WEATHER_DISTANCE_KM"] = 2 * 6371.0088 * np.arcsin(np.clip(distance / 2, 0, 1))
    start = int(fires.FIRE_YEAR.min())
    end = int(fires.FIRE_YEAR.max())
    frames = []
    variables = "temperature_2m_max,wind_speed_10m_max,precipitation_sum,vapour_pressure_deficit_max"
    for row in grid.itertuples():
        pieces = []
        for year in range(start, end + 1, 5):
            last = min(year + 4, end)
            path = directory / f"era5_grid_{row.WEATHER_GRID_ID:02d}_{year}_{last}.json"
            params = dict(latitude=row.latitude, longitude=row.longitude, start_date=f"{year-1}-12-01" if year == start else f"{year}-01-01",
                          end_date=f"{last}-12-31", daily=variables, models="era5", timezone="GMT",
                          wind_speed_unit="kmh", temperature_unit="celsius", precipitation_unit="mm")
            fresh = not path.exists()
            fetch("https://archive-api.open-meteo.com/v1/archive", path, params)
            payload = json.loads(path.read_text())
            if "daily" not in payload:
                raise ValueError(f"No daily weather in {path}")
            daily = pd.DataFrame(payload["daily"])
            daily["CENTROID_DEM_ELEVATION_M"] = payload.get("elevation")
            daily["MODEL_LATITUDE"] = payload["latitude"]
            daily["MODEL_LONGITUDE"] = payload["longitude"]
            pieces.append(daily)
            if fresh:
                time.sleep(0.5)
        daily = pd.concat(pieces, ignore_index=True).sort_values("time")
        daily["DISCOVERY_DAY"] = pd.to_datetime(daily.pop("time"))
        assert daily.DISCOVERY_DAY.is_unique
        daily["ANTECEDENT_PRECIP_30D_MM"] = daily.precipitation_sum.shift(1).rolling(30, min_periods=30).sum()
        daily["WEATHER_GRID_ID"] = row.WEATHER_GRID_ID
        frames.append(daily)
        print(f"Weather {row.WEATHER_GRID_ID + 1}/{count}: {len(daily):,} days cached", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_parquet(DATA / "era5_daily.parquet", index=False)
    result = fires.merge(daily, on=["WEATHER_GRID_ID", "DISCOVERY_DAY"], how="left", validate="many_to_one")
    weather_cols = variables.split(",") + ["ANTECEDENT_PRECIP_30D_MM"]
    assert result[weather_cols].notna().all().all(), "Weather coverage has gaps"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire-zip", type=Path, required=True)
    parser.add_argument("--nifc-csv", type=Path, required=True)
    parser.add_argument("--centroids", type=int, default=25)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    fires = profile(load_fires(args.fire_zip))
    fires.to_parquet(DATA / "fires_containment.parquet", index=False)
    if args.profile_only:
        return
    costs = cpi_costs(args.nifc_csv)
    fires = places(fires)
    fires = weather(fires, args.centroids)
    fires = fires.merge(costs.rename(columns={"Year": "FIRE_YEAR"}), on="FIRE_YEAR", how="left", validate="many_to_one")
    fires.to_parquet(DATA / "enriched_fpa_fod.parquet", index=False)
    summarypath = REPORTS / "acquisition_summary.json"
    summary = json.loads(summarypath.read_text())
    summary.update(enriched_records=len(fires), weather_centroids=args.centroids,
                   weather_distance_km_p50=float(fires.WEATHER_DISTANCE_KM.median()),
                   weather_distance_km_p95=float(fires.WEATHER_DISTANCE_KM.quantile(.95)),
                   missing_cost_records=int(fires["Total (USD)"].isna().sum()),
                   cost_max_year=int(costs.FIRE_YEAR.max()) if "FIRE_YEAR" in costs else int(costs.Year.max()),
                   limitations=["Containment is not fire-out; no FIREOUTDATETIME alias is created.",
                                "Source timestamps have unspecified timezones; do not relabel as UTC.",
                                "25-centroid ERA5 weather is a regional proxy, not local fire weather.",
                                "Centroid DEM elevation used by Open-Meteo is not fire-site elevation or slope.",
                                "2024 Census place centroids are a static proximity proxy, not historical WUI or policy.",
                                "Annual national federal spending cannot be assigned as individual fire cost.",
                                "Missing containment records affect representativeness; see annual coverage report."])
    summarypath.write_text(json.dumps(summary, indent=2))
    print("COMPLETE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
