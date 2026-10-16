"""Paired regional-centroid versus near-fire ERA5 weather diagnostic."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw/weather_probe"
RAW.mkdir(parents=True,exist_ok=True)
VARIABLES=["temperature_2m_max","wind_speed_10m_max","precipitation_sum","vapour_pressure_deficit_max"]


def great_circle(lat1,lon1,lat2,lon2):
    a,b,c,d=map(np.radians,[lat1,lon1,lat2,lon2])
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(np.clip(h,0,1)))


def main():
    cols=["FOD_ID","FIRE_YEAR","DISCOVERY_DAY","LATITUDE","LONGITUDE","STATE","SIZE_BIN","WEATHER_DISTANCE_KM","ANTECEDENT_PRECIP_30D_MM"]+VARIABLES
    fires=pd.read_parquet(ROOT/"data/processed_fires.parquet",columns=cols)
    sample_path=ROOT/"data/weather_probe_sample.parquet"
    if sample_path.exists():
        sample=pd.read_parquet(sample_path)
    else:
        rng=np.random.default_rng(20261014)
        fires["QUARTER"]=fires.DISCOVERY_DAY.dt.quarter
        groups=[]
        for year in range(1992,2025,4):
            for quarter in range(1,5):
                pool=fires.loc[fires.FIRE_YEAR.eq(year)&fires.QUARTER.eq(quarter)]
                counts=pool.DISCOVERY_DAY.value_counts()
                days=counts.loc[counts.ge(20)].index.sort_values()
                if len(days)==0: continue
                day=days[int(rng.integers(len(days)))]
                daily=pool.loc[pool.DISCOVERY_DAY.eq(day)]
                chosen=[]
                for _,size_group in daily.groupby("SIZE_BIN",observed=True):
                    chosen.extend(rng.choice(size_group.index,min(2,len(size_group)),replace=False).tolist())
                remaining=daily.index.difference(chosen)
                if len(chosen)<10 and len(remaining):
                    chosen.extend(rng.choice(remaining,min(10-len(chosen),len(remaining)),replace=False).tolist())
                groups.append(fires.loc[chosen])
        sample=pd.concat(groups,ignore_index=True)
        assert sample.FOD_ID.is_unique
        sample.to_parquet(sample_path,index=False)
    records=[]
    manifest=[]
    session=requests.Session()
    for day,group in sample.groupby("DISCOVERY_DAY",sort=True):
        group=group.sort_values("FOD_ID")
        path=RAW/f"era5_{day:%Y%m%d}.json"
        params={"latitude":",".join(f"{x:.6f}" for x in group.LATITUDE),"longitude":",".join(f"{x:.6f}" for x in group.LONGITUDE),
                "start_date":f"{day-pd.Timedelta(days=30):%Y-%m-%d}","end_date":f"{day:%Y-%m-%d}",
                "models":"era5","timezone":"GMT","daily":",".join(VARIABLES),"wind_speed_unit":"kmh","temperature_unit":"celsius","precipitation_unit":"mm"}
        if not path.exists():
            for attempt in range(5):
                response=session.get("https://archive-api.open-meteo.com/v1/archive",params=params,timeout=120)
                if response.status_code==429:
                    reason=response.json().get("reason","")
                    if "hourly" in reason.lower() or "daily" in reason.lower():
                        raise RuntimeError("Probe paused by provider quota; cached batches retained: "+reason)
                    print("Minute quota; waiting60 seconds",flush=True)
                    time.sleep(60)
                    continue
                response.raise_for_status()
                path.write_bytes(response.content)
                break
            else: raise RuntimeError("Weather probe retries exhausted")
            time.sleep(.5)
        payload=json.loads(path.read_text())
        if isinstance(payload,dict): payload=[payload]
        assert len(payload)==len(group)
        with path.open("rb") as file: digest=hashlib.file_digest(file,"sha256").hexdigest()
        manifest.append({"path":str(path.relative_to(ROOT)),"sha256":digest,"endpoint":"https://archive-api.open-meteo.com/v1/archive","parameters":params})
        for (_,fire),local in zip(group.iterrows(),payload,strict=True):
            daily=pd.DataFrame(local["daily"])
            assert len(daily)==31 and daily.time.iloc[-1]==f"{day:%Y-%m-%d}"
            row=fire.to_dict()
            for var in VARIABLES: row["NEAR_FIRE_"+var]=float(daily[var].iloc[-1])
            row["NEAR_FIRE_ANTECEDENT_PRECIP_30D_MM"]=float(daily.precipitation_sum.iloc[:-1].sum())
            row["NEAR_FIRE_GRID_LATITUDE"]=local["latitude"]
            row["NEAR_FIRE_GRID_LONGITUDE"]=local["longitude"]
            row["NEAR_FIRE_DISTANCE_KM"]=float(great_circle(fire.LATITUDE,fire.LONGITUDE,local["latitude"],local["longitude"]))
            records.append(row)
        print(f"Matched point-weather batch {day:%Y-%m-%d}: {len(group)} fires",flush=True)
    paired=pd.DataFrame(records)
    assert paired.FOD_ID.is_unique and len(paired)==len(sample)
    for var in VARIABLES+["ANTECEDENT_PRECIP_30D_MM"]:
        assert paired["NEAR_FIRE_"+var].notna().all()
    paired.to_parquet(ROOT/"data/fine_weather_probe.parquet",index=False)
    errors={}
    for var in VARIABLES+["ANTECEDENT_PRECIP_30D_MM"]:
        diff=paired[var]-paired["NEAR_FIRE_"+var]
        errors[var]={"median_absolute_difference":float(diff.abs().median()),"p90_absolute_difference":float(diff.abs().quantile(.9)),"mean_signed_centroid_minus_near_fire":float(diff.mean())}
    report={"status":"complete","paired_fires":len(paired),"discovery_days":int(paired.DISCOVERY_DAY.nunique()),
            "years":sorted(paired.FIRE_YEAR.unique().tolist()),"original_distance_km_median":float(paired.WEATHER_DISTANCE_KM.median()),
            "near_fire_distance_km_median":float(paired.NEAR_FIRE_DISTANCE_KM.median()),"near_fire_distance_km_p95":float(paired.NEAR_FIRE_DISTANCE_KM.quantile(.95)),
            "paired_variable_differences":errors,"selection":"Fixed-seed nine-year x four-quarter day sample, up to two cases per acreage bin then fill to ten per day; outcome-complete cohort only.",
            "scope":"A paired weather diagnostic, not a population-weighted national error estimate; these data do not replace the original models' features.",
            "same_source":"Both use ERA5/Open-Meteo with GMT days and the same daily definitions. Near-fire coordinates select local ERA5 cells; still approximately25km reanalysis, not observed local fireground weather.",
            "caveats":["Only36 discovery days; clustered and size-enriched sample.","Requested elevations differ by location; this diagnostic measures the complete centroid proxy difference, not pure horizontal distance alone.","Calendar-day alignment to unknown source timezones remains approximate.","Existing fitted models still use the25-centroid features; applying point values to those models without retraining is not justified."],
            "validation":{"same_model_and_units":True,"paired_identifiers_unique":True,"all_expected_days_present":True,"no_missing_probe_weather":True}}
    (ROOT/"reports/weather_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (RAW/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
