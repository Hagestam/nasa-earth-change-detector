"""NASA POWER daily regional precipitation ingestion.

The normalized output is long-form and can be passed directly to
``analysis.analyze_spatial_trends`` after aggregation to the desired cadence.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any, Literal, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .analysis import aggregate_precipitation


NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
PARAMETER = "PRECTOTCORR"
MISSING_SENTINEL = -999
SOURCE_UNITS = "mm/day"
ANALYSIS_UNITS = "mm"


class NasaPowerError(RuntimeError):
    """Base exception for NASA POWER ingestion failures."""


class NasaPowerRequestError(NasaPowerError):
    """Raised when NASA POWER cannot serve the requested data."""


class NasaPowerResponseError(NasaPowerError):
    """Raised when a NASA POWER response does not match its expected schema."""


def _as_date(value: str | date | datetime, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD).") from exc


def _validate_request(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    latitude_min: float,
    latitude_max: float,
    longitude_min: float,
    longitude_max: float,
) -> tuple[date, date]:
    start, end = _as_date(start_date, "start_date"), _as_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date.")
    if start < date(1981, 1, 1):
        raise ValueError("NASA POWER daily MERRA-2 data start on 1981-01-01.")
    bounds = (latitude_min, latitude_max, longitude_min, longitude_max)
    if not all(math.isfinite(float(value)) for value in bounds):
        raise ValueError("Geographic bounds must be finite numbers.")
    if not (-90 <= latitude_min < latitude_max <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= min < max <= 90.")
    if not (-180 <= longitude_min < longitude_max <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= min < max <= 180.")
    return start, end


def build_regional_request_url(
    *,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    latitude_min: float,
    latitude_max: float,
    longitude_min: float,
    longitude_max: float,
) -> str:
    """Build a daily regional request for PRECTOTCORR in UTC."""
    start, end = _validate_request(
        start_date, end_date, latitude_min, latitude_max, longitude_min, longitude_max
    )
    query = {
        "parameters": PARAMETER,
        "community": "ag",
        "latitude-min": latitude_min,
        "latitude-max": latitude_max,
        "longitude-min": longitude_min,
        "longitude-max": longitude_max,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "utc",
    }
    return f"{NASA_POWER_BASE_URL}?{urlencode(query)}"


def _read_json(url: str, timeout: float) -> Mapping[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "AfricaEnvironmentalChangeMonitor/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise NasaPowerRequestError(f"NASA POWER returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise NasaPowerRequestError(f"NASA POWER request failed: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NasaPowerResponseError("NASA POWER returned invalid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise NasaPowerResponseError("NASA POWER JSON root must be an object.")
    return payload


def _parse_point_feature(feature: Mapping[str, Any], parameter: str) -> list[dict[str, Any]]:
    geometry = feature.get("geometry")
    properties = feature.get("properties")
    if not isinstance(geometry, Mapping) or not isinstance(properties, Mapping):
        raise NasaPowerResponseError("Regional feature must contain geometry and properties objects.")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
        raise NasaPowerResponseError("Regional feature has invalid point coordinates.")
    longitude, latitude = coordinates[0], coordinates[1]
    parameter_data = properties.get("parameter")
    if not isinstance(parameter_data, Mapping):
        parameter_data = properties.get("parameters")
    if not isinstance(parameter_data, Mapping) or parameter not in parameter_data:
        raise NasaPowerResponseError(f"Regional feature is missing parameter {parameter}.")
    dates = parameter_data[parameter]
    if not isinstance(dates, Mapping):
        raise NasaPowerResponseError(f"Regional feature {parameter} values must be keyed by date.")
    rows = []
    for raw_date, raw_value in dates.items():
        rows.append({"time": raw_date, "latitude": latitude, "longitude": longitude, "precipitation": raw_value})
    return rows


def normalize_regional_response(
    payload: Mapping[str, Any],
    *,
    requested_start: str | date | datetime,
    requested_end: str | date | datetime,
    bounds: Mapping[str, float],
    parameter: str = PARAMETER,
) -> pd.DataFrame:
    """Convert NASA POWER regional JSON into analysis-ready long form.

    NASA's ``-999`` no-data sentinel and non-finite values become NaN. Other
    malformed observations are rejected instead of being silently coerced.
    """
    if not isinstance(payload, Mapping):
        raise NasaPowerResponseError("NASA POWER JSON root must be an object.")
    messages = payload.get("messages", [])
    errors = payload.get("errors", [])
    if errors:
        raise NasaPowerResponseError("NASA POWER reported: " + "; ".join(map(str, errors)))
    parameter_catalog = payload.get("parameters")
    reported_parameter = parameter_catalog.get(parameter) if isinstance(parameter_catalog, Mapping) else None
    if not isinstance(reported_parameter, Mapping) or not isinstance(reported_parameter.get("units"), str):
        raise NasaPowerResponseError(f"Response is missing unit metadata for parameter {parameter}.")
    reported_units = reported_parameter["units"]
    if reported_units != SOURCE_UNITS:
        raise NasaPowerResponseError(
            f"NASA POWER reported {reported_units!r} for {parameter}; expected documented units {SOURCE_UNITS!r}."
        )

    rows: list[dict[str, Any]] = []
    features = payload.get("features")
    if isinstance(features, list):
        for feature in features:
            if not isinstance(feature, Mapping):
                raise NasaPowerResponseError("Regional response contains a malformed feature.")
            rows.extend(_parse_point_feature(feature, parameter))
    else:
        # Some POWER responses encode regional values as parameter -> date ->
        # grid coordinate key. Require coordinates explicitly in the key.
        parameters = payload.get("parameters")
        parameter_data = parameters.get(parameter) if isinstance(parameters, Mapping) else None
        if not isinstance(parameter_data, Mapping):
            raise NasaPowerResponseError(f"Response is missing regional parameter {parameter}.")
        for raw_date, cells in parameter_data.items():
            if not isinstance(cells, Mapping):
                raise NasaPowerResponseError("Regional parameter data must include grid-cell coordinates.")
            for key, raw_value in cells.items():
                if isinstance(key, str) and ("," in key or " " in key):
                    parts = key.replace(",", " ").split()
                    if len(parts) < 2:
                        raise NasaPowerResponseError("Invalid regional coordinate key.")
                    longitude, latitude = float(parts[0]), float(parts[1])
                elif isinstance(raw_value, Mapping) and {"latitude", "longitude", "value"} <= raw_value.keys():
                    latitude, longitude = raw_value["latitude"], raw_value["longitude"]
                    raw_value = raw_value["value"]
                else:
                    raise NasaPowerResponseError("Regional grid coordinates are absent from response.")
                rows.append({"time": raw_date, "latitude": latitude, "longitude": longitude, "precipitation": raw_value})

    if not rows:
        raise NasaPowerResponseError("NASA POWER response contains no regional observations.")
    frame = pd.DataFrame(rows, columns=["time", "latitude", "longitude", "precipitation"])
    try:
        frame["time"] = pd.to_datetime(frame["time"], format="%Y%m%d", errors="raise")
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="raise")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="raise")
        raw_values = frame["precipitation"]
        numeric = pd.to_numeric(raw_values, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise NasaPowerResponseError(f"Regional response contains invalid dates or coordinates: {exc}") from exc
    invalid_nonmissing = raw_values.notna() & numeric.isna()
    if invalid_nonmissing.any():
        raise NasaPowerResponseError("Regional response contains non-numeric precipitation values.")
    header = payload.get("header", {})
    fill_value = header.get("fill_value", MISSING_SENTINEL) if isinstance(header, Mapping) else MISSING_SENTINEL
    try:
        fill_value = float(fill_value)
    except (TypeError, ValueError) as exc:
        raise NasaPowerResponseError("NASA POWER header contains an invalid fill value.") from exc
    numeric = numeric.astype(float)
    numeric[(numeric == fill_value) | ~numeric.map(math.isfinite)] = float("nan")
    if (numeric.dropna() < 0).any():
        raise NasaPowerResponseError("Regional response contains negative precipitation values other than the missing sentinel.")
    frame["precipitation"] = numeric
    if not frame["latitude"].between(-90, 90).all() or not frame["longitude"].between(-180, 180).all():
        raise NasaPowerResponseError("Regional response contains coordinates outside geographic bounds.")
    start, end = _as_date(requested_start, "requested_start"), _as_date(requested_end, "requested_end")
    dates = frame["time"].dt.date
    if ((dates < start) | (dates > end)).any():
        raise NasaPowerResponseError("Regional response contains dates outside the requested range.")
    if frame.duplicated(["time", "latitude", "longitude"]).any():
        raise NasaPowerResponseError("Regional response contains duplicate cell-date observations.")

    frame = frame.sort_values(["latitude", "longitude", "time"], kind="stable").reset_index(drop=True)
    parameter_meta = payload.get("parameters", {})
    source_metadata = {
        "provider": "NASA POWER",
        "product": "MERRA-2 corrected precipitation",
        "endpoint": NASA_POWER_BASE_URL,
        "parameter": parameter,
        "source_units": SOURCE_UNITS,
        "units": ANALYSIS_UNITS,
        "temporal_resolution": "daily UTC; daily precipitation value",
        "temporal_aggregation": "daily precipitation accumulation; NASA reports mm/day, normalized to a one-day amount in mm (1 day × daily value; numeric value unchanged)",
        "spatial_resolution": "NASA POWER daily regional grid; actual feature coordinates are authoritative. The underlying MERRA-2 meteorological grid is approximately 0.5° latitude by 0.625° longitude.",
        "requested_bounds": dict(bounds),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "missing_sentinel": fill_value,
        "source_header": dict(header) if isinstance(header, Mapping) else {},
        "parameter_metadata": dict(parameter_meta) if isinstance(parameter_meta, Mapping) else {},
        "source_messages": list(messages) if isinstance(messages, list) else [],
        "transformations": [
            "NASA POWER -999 sentinel mapped to missing (NaN)",
            "timestamps parsed as daily UTC",
            "source unit mm/day normalized to one-day accumulated amount mm; numerical value unchanged because each record spans one day",
        ],
    }
    frame.attrs["units"] = ANALYSIS_UNITS
    frame.attrs["source_provenance"] = source_metadata
    frame.attrs["frequency"] = "daily"
    frame.attrs["missing_observation_count"] = int(frame["precipitation"].isna().sum())
    return frame


def fetch_regional_precipitation(
    *,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    latitude_min: float,
    latitude_max: float,
    longitude_min: float,
    longitude_max: float,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch precipitation for a bounding box and return analysis-ready rows."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number of seconds.")
    start, end = _validate_request(
        start_date, end_date, latitude_min, latitude_max, longitude_min, longitude_max
    )
    bounds = {
        "latitude_min": latitude_min,
        "latitude_max": latitude_max,
        "longitude_min": longitude_min,
        "longitude_max": longitude_max,
    }
    url = build_regional_request_url(
        start_date=start, end_date=end,
        latitude_min=latitude_min, latitude_max=latitude_max,
        longitude_min=longitude_min, longitude_max=longitude_max,
    )
    payload = _read_json(url, timeout)
    return normalize_regional_response(
        payload,
        requested_start=start,
        requested_end=end,
        bounds=bounds,
    )


def aggregate_regional_precipitation(
    data: pd.DataFrame,
    *,
    period: Literal["monthly", "annual"] = "monthly",
    source_frequency: Literal["daily", "monthly"] = "daily",
    coverage_threshold: float = 1.0,
) -> pd.DataFrame:
    """Aggregate each cell separately into engine-ready spatial long form.

    Output columns include the exact ``time``, ``latitude``, ``longitude``,
    and ``precipitation`` contract accepted by ``analyze_spatial_trends``.
    Daily-to-annual requests can be staged through monthly totals and passed
    with ``source_frequency="monthly"``. Missing observations remain missing;
    no interpolation is performed.
    """
    required = {"time", "latitude", "longitude", "precipitation"}
    if not required.issubset(data.columns):
        raise ValueError(f"Regional data is missing required columns: {sorted(required - set(data.columns))}")
    if data.empty:
        raise ValueError("Regional data must contain at least one observation.")
    units = data.attrs.get("units")
    provenance = data.attrs.get("source_provenance", {})
    outputs: list[pd.DataFrame] = []
    for (latitude, longitude), cell in data.groupby(["latitude", "longitude"], sort=True):
        aggregate = aggregate_precipitation(
            cell[["time", "precipitation"]],
            period=period,
            source_frequency=source_frequency,
            units=units,
            coverage_threshold=coverage_threshold,
            source_provenance=provenance,
        ).reset_index()
        aggregate["time"] = aggregate["period"].dt.to_timestamp()
        aggregate["latitude"] = float(latitude)
        aggregate["longitude"] = float(longitude)
        outputs.append(aggregate.drop(columns="period"))
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["latitude", "longitude", "time"], kind="stable"
    ).reset_index(drop=True)
    result.attrs["units"] = units
    result.attrs["source_provenance"] = dict(provenance)
    result.attrs["aggregation"] = {
        "period": period,
        "source_frequency": source_frequency,
        "coverage_threshold": coverage_threshold,
        "incomplete_periods": "missing; not imputed",
    }
    return result
