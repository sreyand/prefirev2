# PreFIRE v2 containment workbench

PreFIRE studies historical wildfire containment duration, prolonged-containment risk, and the association between observed fire concurrency and duration. The core dataset is USFS FPA-FOD seventh edition, 1992–2024; the current outcome-complete continental-U.S. cohort contains 1,651,130 records.

The models use **final acreage** and reconstructed observed load. Their outputs are retrospective conditional associations, not ignition-time forecasts, dispatch recommendations, causal resource effects, or proof that suppression capability has declined.

## Run locally

Place the preserved source files at:

- `data/raw/S_USA.Fire_FPA_FOD_7th_Fires.gdb.zip`
- `data/raw/nifc_suppression_costs.csv`

From the project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_pipeline.ps1
.\launch.ps1
```

The workbench is served at [127.0.0.1:8501](http://127.0.0.1:8501). Data, model artifacts, and computation remain in this project. The virtual environment, downloaded data, and generated Parquet/model files are excluded from Git; preserve the raw files and source manifests separately when moving the project.

Acquisition caches bulk downloads, ERA5 centroid responses, the weather probe, and the pinned ICS-209-PLUS files. Reruns resume from those caches rather than downloading the bulk fire archive again. Missing weather batches can still make API requests and encounter provider quotas; cached batches are retained if that happens.

For an existing, fully built project, optional acquisition steps can be skipped:

```powershell
.\run_pipeline.ps1 -SkipAcquisition -SkipWeatherAudit -SkipOperationalData
```

The weather and operational skip switches require their existing reports. Use them only when those reports were generated from the same source/cohort version. Model training and the paper-inspired benchmark still run. To inspect only the source fire records without acquiring enrichment, run `scripts/acquire_data.py --profile-only` with the required source paths.

## Pipeline order and artifacts

`run_pipeline.ps1` stops at the first failing step and runs these dependencies in order:

1. `acquire_data.py`: source profiling, cached acquisition, normalization, and enrichment; optional with `-SkipAcquisition`.
2. `prepare_features.py` and `validate_data.py`: calendar-comparable outcome, size bins, daily discoveries, observed concurrency, and validation.
3. `audit_reporting.py`: date/clock reconstruction, reporting availability, common-support composition standardization, and window sensitivity.
4. `audit_weather.py`: cached paired centroid/near-fire weather diagnostic; optional with `-SkipWeatherAudit`.
5. `acquire_operational_data.py`: cached ICS-209-PLUS acquisition, exact identifier links, resource-history validation, and examples; optional with `-SkipOperationalData`.
6. `build_dashboard_summary.py`: observed cohort summaries for the interface.
7. `train_risk.py` and `analyze_load.py`: original risk/duration estimators and adjusted retrospective load association.
8. `audit_model_support.py`: temporal calibration reassessment, subgroup audit, joint-support gate, and coherent recorded examples.
9. `calibrate_risk.py`: the separate prolonged-only duration model and conformal range calibration. This must run **after** `audit_model_support.py`, which creates its input bundle.
10. `paper_benchmark.py`: separate controlled classifier ablations; it does not replace the audited simulator bundle.
11. `audit_alert_precision.py`: fixed-threshold precision, recall, false flags, and PR curves; probability support populations are separate.
12. `check_app.py`: end-to-end application checks after all model/audit artifacts exist.

The main data outputs are `data/fpa_fod_all.parquet`, `data/fires_containment.parquet`, `data/era5_daily.parquet`, `data/enriched_fpa_fod.parquet`, and `data/processed_fires.parquet`. `data/daily_observed_load.parquet` retains the observed daily load series. Supporting inputs include Census place centroids, CPI-U, and nominal/2024-dollar NIFC federal spending.

`data/raw/sources.json` records source URLs, acquisition timestamps, checksums, and API parameters where applicable. The operational and near-fire weather acquisitions have their own provenance manifests. Keep these with exported results.

## Outcome and reporting sensitivity

The primary outcome is `CONTAINMENT_DAYS_DATE_ONLY`: normalized containment calendar date minus normalized discovery calendar date. **Zero means the same calendar date, not zero elapsed time.** More than three calendar days is not an exact 72-hour operational threshold. Containment and fire-out are different endpoints; this project does not measure time until fire-out.

Original clock fields are retained as wall-clock values with unspecified timezones. `DURATION_RESOLUTION` identifies records with valid source clocks; their fractional-day durations support a separate sensitivity analysis. Mixing their sub-day values with date-only records would change the measurement population over time. Source timezones and daylight-saving transitions cannot be reconstructed reliably.

The cohort requires positive acreage, continental-U.S. coordinates, valid discovery/containment dates, and a 0–365-day interval; invalid clock intervals are also screened. Missing containment outcomes are substantial and vary by reporting agency and year. Complete-case coverage is not national representativeness, and Forest Service-only records are not a stable reporting control.

`reports/reporting_audit.json` and CSV compare broad periods 1992–2005 versus 2015–2024, clock-complete subsets, and finer acreage/reporting-agency strata on common support. Recent strata are reweighted to the early-period composition. The audit also compares the shorter 1992–1994 versus 2020–2024 windows. These are descriptive sensitivities: window endpoints and mixture standardization can change the observed tail-duration contrast without identifying a suppression effect. Retained common-support fractions are reported; excluded strata are not silently extrapolated.

## Audited prolonged-fire probability

The original `models/risk_bundle.joblib` preserves the historical estimators: a deterministic 200,000-record fitting sample from 1992–2014, original probability calibration in 2015–2017, validation in 2018–2019, and an earlier 2020–2024 evaluation. A tied-zero quantile fitting failure was reproduced on synthetic data and corrected with fixed-seed, training-only numerical smoothing of ±1e-6 in log1p-duration. Repeated historical evaluation is labeled exploratory.

The simulator uses the separate **`models/risk_bundle_audited.joblib`**. Its classifier and original duration estimators are unchanged. Calibration candidates—raw probabilities, recent Platt scaling, and recent isotonic regression—fit on 2020–2021 and are selected solely by 2022 Brier score. The selected isotonic mapping is then fitted on 2020–2022 and reassessed on 2023–2024. Those later years were previously inspected, so this is a historical reassessment, not an untouched confirmatory test.

On 128,797 reassessment records, the selected mapping has Brier score 0.03426, compared with 0.03600 raw and 0.03858 from the old Platt mapping. Raw ranking AUC is 0.7743; isotonic ties slightly change AUC and average precision. Better aggregate calibration does not make every probability reliable: size- and agency-specific errors remain material. `reports/model_audit.json`, `model_audit_groups.csv`, and `model_audit_calibration_bins.csv` expose those differences rather than presenting a single national score as case-level validation.

Average precision is a summary over the precision–recall curve, not precision at one threshold. `audit_alert_precision.py` evaluates predeclared probability cutoffs (5%, 10%, 20%, 30%, 50%, 70%) without optimizing on 2023–2024 outcomes. In all 128,797 reassessment records, a 20% cutoff gives 28.9% precision / 26.7% recall (1,364 true and 3,353 false flags). A 50% cutoff gives 76.4% precision / 6.88% recall (352 true and 109 false flags), missing 4,762 of the 5,114 prolonged fires. These retrospective tradeoffs do not establish useful operational alerts. Results, PR curves, yearly/size breakdowns, and probability-supported-population comparisons are in `reports/alert_precision.json` and CSV.

## Duration if containment exceeds three days

The principal displayed duration range answers **“If containment takes more than three calendar days, how long might it take?”** It is not the unconditional median duration and does not imply the prolonged event will occur.

`calibrate_risk.py` fits separate conditional P10/P50/P90 histogram-boosted regressors on all 35,496 prolonged 1992–2014 records, using the existing feature transform. It calibrates a nonnegative conformalized-quantile-regression expansion on 5,450 prolonged 2020–2022 records, using the finite order statistic `ceil((n + 1) × 0.8)`. Evaluation uses the original unsmoothed calendar outcomes; bounds are clipped to the cohort's conditional 4–365-day domain. The expansion is 0.08972 days.

Among 5,114 prolonged 2023–2024 reassessment records, the resulting range covers **80.13%**, with mean width **21.38 days**. Conditional median MAE is **9.35 days**, versus **10.47 days** for the early-training constant median. The constant training P10–P90 interval spans 38 days and covers a different, higher fraction of later outcomes; width and coverage must be assessed together.

This is marginal calibration among observed prolonged fires, not an individual-fire or subgroup guarantee. Conditional coverage is approximately **74.7% for 1,000–<10,000 acres** and **74.8% for 10,000+ acres**, with 379 and 119 reassessment cases respectively. Across-year exchangeability is uncertain. The expanded lower/upper bounds are not exact P10/P90 quantiles or confidence intervals. Subgroup coverage, width, and sample counts are in `reports/model_audit_prolonged_groups.csv` and the conditional-model section of `model_audit.json`.

The original unconditional quantiles remain available for exploratory comparison. Their apparently high aggregate coverage is dominated by same-day outcomes: on 2023–2024 prolonged cases, their range covers only 15.86%. They must not be advertised as a validated likely-duration range for prolonged fires.

## Joint support and scenario comparisons

The simulator starts from actual recorded feature rows, including supported small and larger examples. The full feature combination matters: separately plausible slider values need not describe a combination seen in the fitting data.

The probability model's all-fire joint support uses 100,000 records from the original fitting sample as a reference and 20,000 disjoint fitting records for distance thresholds. It matches state, reporting agency, and cause exactly, then measures the tenth-neighbor distance across robustly scaled numeric controls, cyclic season, and transformed skewed features. At least 30 matching context records are required. Training-derived 90th/99th-percentile thresholds distinguish supported, caution, and unsupported combinations; sparse contexts use an explicitly labeled pooled threshold. Numeric values outside the observed training ranges are also unsupported. Within the specified fitting cohort, outcome values do not enter neighborhood geometry, thresholds, or example ranking.

The conditional duration model has a **separate prolonged-training support gate**: 28,396 reference and 7,100 disjoint threshold records from its own 35,496 historical prolonged fitting cases. It uses the same feature-only algorithm and a minimum of 30 matching context references, with numeric ranges derived from that conditional cohort. Membership in the historical conditional training cohort requires duration >3 days; inference checks only the supplied features and never looks up a scenario's outcome. Probability requires the all-fire gate; conditional duration requires **both gates**. A context with many ordinary-fire neighbors but no prolonged training cases can therefore retain a probability estimate while its conditional duration is hidden. Missing conditional-support artifacts fail closed.

The gate does not improve or recalibrate the fitted duration estimators. Unrestricted prolonged reassessment remains 5,114 cases with 80.13% coverage and 9.35-day median MAE. Both gates permit 2,466 of those cases (48.22%); their separately reported coverage is 79.36%, median MAE 10.14 days, and mean interval width 24.14 days. Metrics for the 2,648 suppressed cases are retained too. On a deterministic 50,000-record sample of all 128,797 recent records, both gates permit conditional-duration presentation for 28.98%; this sampled fraction is not an exact population rate. See `model_audit_prolonged_support.csv` and the conditional model's `support_gate` section in `model_audit.json`. To upgrade only this gate without retraining models, run `scripts/calibrate_risk.py --support-only`.

An unsupported all-fire combination suppresses probability and duration displays; an unsupported prolonged-training combination suppresses conditional duration only. A supported label only means comparable earlier feature combinations exist in the relevant training cohort; it does not establish accuracy or causal validity. The following percentages describe the all-fire gate. On 2023–2024 records, 51.17% are supported, 22.77% caution, and 26.07% unsupported. In a fixed 96-combination illustrative stress grid, every individual input stays within its training range, yet 34.38% are jointly unsupported. That grid is a diagnostic, not a population frequency estimate. See `model_audit_support_coverage.csv` and `model_audit_scenario_support_grid.csv`.

Changing final acreage, weather, cause, or observed load holds other controls fixed. The resulting difference is a conditional model association and can be sensitive to unrealistic combinations; it is not the effect of deploying additional crews or changing weather.

## Load, operational resources, and environmental limits

`analyze_load.py` estimates a weighted retrospective association between log1p(calendar duration) and nonlinear observed national load, adjusting for final acreage, season, year, state, reporting agency, cause, geography, weather, and available proxies. It uses deterministic year-by-agency stratification, inverse sampling weights, and discovery-day clustered uncertainty. Percentage contrasts describe geometric mean(days + 1), not arithmetic mean duration. Clock-complete results are separate.

Observed concurrency includes the focal fire and only records with usable containment outcomes. Its construction uses later reported containment dates; duration and final acreage can themselves reflect suppression outcomes. It is not total national active-fire load, available capacity, unfilled requests, or a causal resource bottleneck. No exact crew-rationing threshold or causal attribution waterfall is inferred.

The separate **ICS-209-PLUS v3 operational dataset** covers selected large/significant incidents in 1999–2020. It contains reported personnel, aerial resources, acres, and explicitly unit-checked containment/completion fields. Exact official incident/complex identifiers link 37,236 FPA records to context, about 2.04% of FPA records in that period. It is a selected operational history, not a representative national sample. Complex/member resources must not be added together; gaps are not filled, source zeros/fractional values are retained, and resource outliers are flagged. Personnel assignments do not measure available workforce or shortages, and source report peaks are not ignition-time features. The operational records remain separate from the prediction inputs. See `reports/data_upgrade.json`, `reports/operational_examples.json`, and `data/operational_*.parquet`.

Main weather inputs are ERA5 daily reanalysis at 25 fixed fire-weighted regional centroids, not fire-site observations. Assignment distance is approximately 166 km at the median and 348 km at the 95th percentile. GMT weather days and fire dates with unknown source timezones introduce day-alignment uncertainty. Centroid DEM elevation is an Open-Meteo downscaling input, not fire-site elevation or slope. **Fire-site terrain, historical WUI boundaries, road access, and observed suppression policy are not supplied.** Census place-centroid distance is a static proximity proxy; ignition cause does not establish a let-burn decision.

`audit_weather.py` compares 360 size-enriched cases on 36 discovery days against near-fire ERA5 requests. In that diagnostic, median absolute differences are 2.1°C for maximum temperature, 2.6 km/h for maximum wind, 0.325 kPa for VPD, and 13.7 mm for preceding 30-day rainfall. The sample is clustered and not population weighted. Near-fire cells remain approximately 25-km reanalysis, not local observed fireground weather. These probe values do not replace the trained models' centroid features; substituting them at inference without retraining would change the feature distribution. Cached probe responses and checks are recorded in `reports/weather_audit.json` and `data/raw/weather_probe/manifest.json`.

NIFC federal suppression spending is annual national context, not fire-specific expense. CPI-U converts available supplied years to 2024 dollars; absent spending years remain missing. Spending-duration comparisons alone do not establish financial or operational efficiency.

## Paper-inspired controlled benchmark

The reference is Sykas, Zografakis, and Demestichas, *Deep Learning Approaches for Wildfire Severity Prediction: A Comparative Study of Image Segmentation Networks and Visual Transformers on the EO4WildFires Dataset*. The [published article](https://www.mdpi.com/2571-6255/7/11/374) is in *Fire* 2024, 7(11), 374. The readable [author preprint, version 1, posted 9 September 2024](https://www.preprints.org/manuscript/202409.0702), is explicitly not peer reviewed; it links the later [published DOI](https://doi.org/10.3390/fire7110374). The publisher page was inaccessible during the project audit, so detailed method inspection used that identified preprint version.

That work predicts burned-area size/shape masks from earth-observation inputs. It does not validate containment-duration prediction, suppression effectiveness, or this project's tabular outcomes. Segmentation IoU/F1 scores cannot be compared directly with containment AUC, calibration error, or duration MAE.

`paper_benchmark.py` transfers only experimental design ideas: common event IDs and chronological splits, strictly preceding 30-day weather summaries, feature ablations, multiple metrics, and filtered-training/full-evaluation sensitivity. It compares the original feature set, removal of weather, removal of observed load, additional prior-30-day temperature/VPD/wind means and peaks, a logistic baseline, and training restricted to fires at least one acre. The full evaluation population remains explicit, with matched-size diagnostics also reported. Previous-weather features exclude the discovery day but retain the same coarse centroids.

The benchmark caps each split at 100,000 records and uses its own common fitting/calibration/validation protocol. Its numbers are not a controlled direct comparison with the differently calibrated deployed audited bundle. A candidate is saved separately only if the predeclared validation screen passes; it never replaces the simulator's probability, support, or conditional-duration models. Results, provenance, selection rules, and limitations are in `reports/paper_benchmark.json` and CSV. Historical reassessment remains exploratory because those later years were already inspected.

The validation-selected experiment removed observed load: Brier improved from 0.032426 to 0.032032 (1.21%), but reversed on 2023–2024 reassessment (0.038275 to 0.039113). The antecedent-history experiment gave a smaller validation improvement and failed the predefined screen. Logistic regression was better on the later reassessment but worse in validation; it was not selected after seeing the later scores. These results expose temporal instability. The experimental candidate remains separate from the active model. Small-event sensitivity and calibration-era constant baselines are reported alongside the full-population metrics.

## Sources and attribution

- [USFS FPA-FOD](https://research.fs.usda.gov/firelab/products/dataandtools/us-wildfires)
- [NIFC suppression costs](https://www.nifc.gov/fire-information/statistics/suppression-costs)
- [ERA5/Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api)
- [BLS CPI series](https://www.bls.gov/cpi/factsheets/cpi-series-ids.htm)
- [Census 2024 Gazetteer files](https://www.census.gov/geographies/reference-files/2024/geo/gazetter-file.html)
- [ICS-209-PLUS v3, CC BY 4.0](https://doi.org/10.6084/m9.figshare.19858927.v3), with its [methods paper](https://doi.org/10.1038/s41597-023-01955-0)

Preserve source attribution, version identifiers, checksums, and the stated analysis populations when sharing artifacts. The research question about suppression effectiveness remains open.
