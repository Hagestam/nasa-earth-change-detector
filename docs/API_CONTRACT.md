# NASA POWER Data Ingestion Contract

## Selected source and parameter

- Provider: NASA Prediction of Worldwide Energy Resources (POWER).
- Endpoint: `https://power.larc.nasa.gov/api/temporal/daily/regional`.
- Parameter: `PRECTOTCORR` (corrected total precipitation), Agroclimatology (`AG`) community.
- Product: POWER's observationally corrected MERRA-2 meteorological precipitation product.
- NASA-reported units: `mm/day` for the daily precipitation value. NASA's methodology describes MERRA-2 precipitation comparisons as one-day accumulated precipitation and describes coarser accumulations as sums over days.
- Analysis units: the adapter treats each daily value as a one-day accumulated amount in `mm`, because the analysis engine aggregates daily amounts by summation. This is an explicit unit normalization: `mm/day × 1 day = mm`, so the numeric value is unchanged. Both reported and analysis units plus the operation are recorded in provenance. Monthly/annual totals and trend slopes therefore carry `mm` and `mm/year` units respectively.
- Temporal resolution: daily, requested in UTC using `time-standard=UTC`. The daily value is an accumulation over a one-day interval, not an hourly precipitation rate. The adapter applies no scaling; `aggregate_precipitation` sums these daily amounts to calendar months/years when coverage requirements are met.
- Historical availability: the daily POWER meteorology API documents data from 1981-01-01 through near real time. Availability for a specific cell/date is not guaranteed.
- Spatial resolution: request is a bounding box; output is one record per returned grid cell and date. NASA documents regional daily examples on a 0.5° grid. `PRECTOTCORR` is based on MERRA-2 meteorology (native grid approximately 0.5° latitude by 0.625° longitude); preserve the coordinates returned for each feature as authoritative rather than synthesizing coordinates from the requested bounds.

NASA's daily API permits one parameter in a regional request. The configured request uses that one parameter and JSON output. The caller supplies ISO dates and numeric bounding-box bounds. Request failures, timeouts, malformed JSON/schema, negative precipitation, invalid coordinates, and out-of-range response dates produce explicit exceptions. NASA's `-999` no-data sentinel is converted to missing (`NaN`); zero remains a valid observation, and no missing observation is imputed.

## Analysis-engine adapter

`backend/app/services/nasa_power.py::fetch_regional_precipitation()` returns a Pandas DataFrame in long form:

| Column | Meaning |
| --- | --- |
| `time` | Daily timestamp parsed from NASA's YYYYMMDD key; request is UTC. |
| `latitude` | Returned grid-cell latitude. |
| `longitude` | Returned grid-cell longitude. |
| `precipitation` | Daily accumulated `PRECTOTCORR` amount in `mm` (numeric value unchanged from NASA's `mm/day` daily value), or NaN when missing. |

Run `analysis.validate_environmental_data(..., expected_frequency="daily")` per `(latitude, longitude)` group; validating the full long-form table as one series would mistake the same daily timestamps at different cells for duplicate observations. For monthly or annual analysis, call `nasa_power.aggregate_regional_precipitation(data, period=...)`. This small adapter applies the existing `analysis.aggregate_precipitation()` independently to each cell, returning `time`, `latitude`, `longitude`, and `precipitation` plus coverage columns. Pass those columns to `analysis.analyze_spatial_trends()`. The analysis functions do not need modification.

The DataFrame attributes retain provider, product, endpoint, parameter, NASA-reported units (`mm/day`), analysis units (`mm`), time resolution/interpretation, requested extent/date range, missing sentinel, source header/messages, and transformations. The explicit daily unit normalization is required because the analysis engine sums daily amounts and preserves the input unit label. Other adapter operations are limited to parsing dates/coordinates/numeric values and mapping the documented missing sentinel to NaN. No values are filled.

## Limitations

- POWER precipitation is a gridded reanalysis/observation-corrected product, not a direct rain-gauge record. Bias, representativeness, and source-product limitations remain relevant to interpretation.
- NASA says the API may return missing values for otherwise supported dates; the adapter retains these as NaN. The existing analysis aggregation excludes periods that fail coverage and does not impute them.
- A regional bounding box may include only cells returned by NASA. The service does not invent edge cells or extrapolate outside returned geometry.
- The existing analysis engine has no upstream provenance-column contract; it reads its four required columns and stores metadata through its explicit `source_provenance` and `units` arguments/DataFrame attrs.
- Automated tests mock NASA responses and do not depend on the live service. Live endpoint availability, query-size constraints for long daily regional requests, and exact regional payload variations remain operational integration concerns.

## Backend HTTP API

The FastAPI application is `backend.app.main:app`. Start it from the repository root with:

```powershell
uvicorn backend.app.main:app --reload --app-dir .
```

`GET /health` reports process readiness. `POST /api/v1/analysis/precipitation` runs the complete ingestion and analysis pipeline. Its request body is:

```json
{
  "start_date": "2000-01-01",
  "end_date": "2020-12-31",
  "bbox": {
    "latitude_min": -5.0,
    "latitude_max": 5.0,
    "longitude_min": 30.0,
    "longitude_max": 42.0
  },
  "period": "annual",
  "coverage_threshold": 1.0,
  "alpha": 0.05,
  "fdr_method": "by",
  "n_resamples": 499,
  "include_interpretation": false
}
```

`period` may be `monthly` or `annual`; it selects both the precipitation aggregation and the matching trend method. `include_interpretation` is optional and defaults to false. When true, the backend sends the just-computed analysis records and context to the interpreter. Configure `LLM_API_KEY` (or `OPENAI_API_KEY`) and `LLM_MODEL` (or `OPENAI_MODEL`) in the process environment. No API key is accepted in the request.

For browser clients hosted on another origin, configure `CORS_ORIGINS` as a comma-separated allowlist. The development default permits the common local React dev-server origins `http://localhost:5173` and `http://127.0.0.1:5173`; production deployments should set the exact frontend origin(s).

Successful responses contain `request`, `data` (units, provenance, aggregation, row counts), `validation` (per-cell diagnostics), `analysis` (summary, testing family, settings, and spatial results), and `interpretation` (null, or a completed/unavailable status). Each spatial result includes the engine's trend status/label, coordinates, slope, confidence interval, p-value and adjusted q-value where calculated. Missing numeric values are serialized as JSON `null`.

Request validation errors return HTTP 422. NASA POWER request and response failures return HTTP 502. Unexpected processing failures return HTTP 500 with a generic message; diagnostic details are written to the backend logger. The synchronous data clients and statistical services run in a worker thread so they do not block the ASGI event loop. The optional LLM failure is reported under `interpretation` while retaining the completed statistical results.

## References

- [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) — supported formats, regional request example, daily availability, time standards, and regional one-parameter limit.
- [NASA POWER Meteorology data sources and precipitation methodology](https://power.larc.nasa.gov/docs/methodology/data/sources_meteorology/) — MERRA-2 observational correction for `PRECTOTCORR`, precipitation units and accumulation interpretation, and validation caveats.
- [NASA POWER Temporal APIs](https://power.larc.nasa.gov/docs/services/api/temporal/) — daily and regional temporal service descriptions.
- [NASA POWER Parameter Dictionary](https://power.larc.nasa.gov/parameters/) — parameter catalog.
