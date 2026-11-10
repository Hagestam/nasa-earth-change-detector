"""Scientific analysis services for precipitation trend monitoring.

The module keeps calculations deterministic for fixed inputs and settings. It
does not fetch data or generate environmental observations; callers provide
observed precipitation with timestamps and provenance.
"""

from __future__ import annotations

from math import erfc, sqrt
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

Frequency = Literal["daily", "monthly", "annual"]
Period = Literal["monthly", "annual"]


def _as_series(
    data: pd.Series | pd.DataFrame,
    *,
    value_column: str | None = None,
    timestamp_column: str | None = None,
) -> pd.Series:
    """Return a sorted numeric Series with a DatetimeIndex."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        time_col = timestamp_column or next(
            (name for name in ("timestamp", "time", "date") if name in frame.columns), None
        )
        if time_col is not None:
            frame.index = pd.to_datetime(frame.pop(time_col), errors="coerce")
        if value_column is None:
            value_column = "precipitation" if "precipitation" in frame.columns else None
        if value_column is None or value_column not in frame.columns:
            raise ValueError("Provide a precipitation value column.")
        values = frame[value_column]
    elif isinstance(data, pd.Series):
        values = data.copy()
    else:
        raise TypeError("data must be a pandas Series or DataFrame")

    if isinstance(values.index, pd.PeriodIndex):
        index = values.index.to_timestamp(how="start")
    else:
        index = pd.to_datetime(values.index, errors="coerce")
    if index.isna().any():
        raise ValueError("One or more timestamps are invalid.")
    numeric = pd.to_numeric(values, errors="coerce")
    original = values.to_numpy()
    if ((~pd.isna(original)) & pd.isna(numeric.to_numpy())).any():
        raise ValueError("Precipitation values must be numeric or missing.")
    result = pd.Series(numeric.to_numpy(dtype=float), index=pd.DatetimeIndex(index), name="precipitation")
    return result.sort_index()


def _frequency_alias(frequency: str) -> str:
    aliases = {"daily": "D", "monthly": "M", "annual": "Y", "yearly": "Y"}
    if frequency not in aliases:
        raise ValueError("frequency must be daily, monthly, or annual")
    return aliases[frequency]


def _period_keys(index: pd.DatetimeIndex, frequency: str) -> pd.PeriodIndex:
    if frequency == "daily":
        return index.to_period("D")
    if frequency == "monthly":
        return index.to_period("M")
    if frequency in ("annual", "yearly"):
        return index.to_period("Y")
    raise ValueError("frequency must be daily, monthly, or annual")


def validate_environmental_data(
    data: pd.Series | pd.DataFrame,
    expected_frequency: Frequency = "monthly",
    value_column: str | None = None,
    timestamp_column: str | None = None,
) -> dict[str, Any]:
    """Validate timestamps, missing/invalid values, duplicates, and cadence.

    Missing values are reported, not imputed. The returned diagnostics are
    JSON-friendly except for no embedded data objects; callers can retain their
    original input for subsequent processing.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
            time_col = timestamp_column or next(
                (name for name in ("timestamp", "time", "date") if name in frame.columns), None
            )
            if time_col is None:
                raise ValueError("A timestamp column or DatetimeIndex is required.")
            parsed = pd.to_datetime(frame[time_col], errors="coerce")
            raw = frame[value_column or ("precipitation" if "precipitation" in frame else "")]
        elif isinstance(data, pd.Series):
            parsed = pd.to_datetime(data.index, errors="coerce")
            raw = data
        else:
            raise TypeError("data must be a pandas Series or DataFrame")
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "observation_count": 0,
            "valid_observation_count": 0,
            "missing_value_count": 0,
            "invalid_value_count": 0,
            "missing_period_count": None,
            "duplicate_timestamp_count": 0,
            "frequency": expected_frequency,
        }

    invalid_timestamp_count = int(pd.isna(parsed).sum())
    if invalid_timestamp_count:
        errors.append(f"{invalid_timestamp_count} invalid timestamp(s).")
    valid_mask = ~pd.isna(parsed)
    timestamps = pd.DatetimeIndex(parsed[valid_mask])
    raw_values = raw.to_numpy()[valid_mask]
    converted = pd.to_numeric(pd.Series(raw_values), errors="coerce")
    values = converted.to_numpy(dtype=float)
    source_missing = pd.isna(raw_values)
    numeric_mask = np.isfinite(values)
    explicit_missing_count = int(source_missing.sum())
    non_numeric_or_nonfinite_count = int((~numeric_mask & ~source_missing).sum())
    invalid_value_count = non_numeric_or_nonfinite_count
    negative_count = int((numeric_mask & (values < 0)).sum())
    invalid_value_count += negative_count
    if negative_count:
        errors.append(f"{negative_count} negative precipitation value(s).")
    if non_numeric_or_nonfinite_count:
        errors.append(f"{non_numeric_or_nonfinite_count} non-numeric or non-finite value(s).")
    if explicit_missing_count:
        warnings.append(f"{explicit_missing_count} explicit missing value(s); none were imputed.")
    if len(timestamps) == 0:
        errors.append("No valid timestamps are available.")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "observation_count": len(parsed),
            "valid_observation_count": 0,
            "missing_value_count": explicit_missing_count,
            "invalid_value_count": invalid_value_count,
            "missing_period_count": None,
            "duplicate_timestamp_count": 0,
            "frequency": expected_frequency,
        }

    aliases = {"daily": "D", "monthly": "M", "annual": "Y"}
    if expected_frequency not in aliases:
        errors.append("expected_frequency must be daily, monthly, or annual")
        missing_period_count = None
        duplicates = 0
    else:
        keys = _period_keys(timestamps, expected_frequency)
        duplicate_mask = keys.duplicated(keep=False)
        duplicates = int(duplicate_mask.sum())
        if duplicates:
            errors.append(f"{duplicates} duplicate timestamp period(s) at {expected_frequency} cadence.")
        observed_keys = pd.PeriodIndex(keys.unique()).sort_values()
        expected = pd.period_range(observed_keys.min(), observed_keys.max(), freq=aliases[expected_frequency])
        missing_period_count = int(len(expected.difference(observed_keys)))
        if missing_period_count:
            warnings.append(f"{missing_period_count} missing {expected_frequency} period(s); no values were filled.")

    valid_observation_count = int((numeric_mask & (values >= 0)).sum())
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "observation_count": int(len(parsed)),
        "valid_observation_count": valid_observation_count,
        "missing_value_count": explicit_missing_count + (missing_period_count or 0),
        "invalid_value_count": invalid_value_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "missing_period_count": missing_period_count,
        "duplicate_timestamp_count": duplicates,
        "frequency": expected_frequency,
    }


def aggregate_precipitation(
    data: pd.Series | pd.DataFrame,
    period: Period = "monthly",
    source_frequency: Frequency = "daily",
    units: str | None = None,
    coverage_threshold: float = 1.0,
    source_provenance: dict[str, Any] | None = None,
    value_column: str | None = None,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    """Aggregate precipitation amounts and mark incomplete periods missing.

    Daily amounts are summed to calendar months/years; monthly amounts are
    summed to years. Rates must be converted to amounts by the caller. Output
    columns are ``precipitation``, ``observations``, ``expected_observations``
    and ``coverage``. Units and provenance are retained in ``DataFrame.attrs``.
    """
    if period not in ("monthly", "annual"):
        raise ValueError("period must be monthly or annual")
    if not 0 < coverage_threshold <= 1:
        raise ValueError("coverage_threshold must be in (0, 1]")
    if source_frequency not in ("daily", "monthly", "annual"):
        raise ValueError("source_frequency must be daily, monthly, or annual")
    series = _as_series(data, value_column=value_column, timestamp_column=timestamp_column)
    if series.index.has_duplicates:
        raise ValueError("Duplicate timestamps must be resolved before aggregation.")
    if _period_keys(series.index, source_frequency).has_duplicates:
        raise ValueError(f"More than one observation occurs within a declared {source_frequency} period.")
    if ((series.dropna() < 0) | ~np.isfinite(series.dropna())).any():
        raise ValueError("Precipitation values must be finite and non-negative.")

    keys = _period_keys(series.index, source_frequency)
    source = pd.Series(series.to_numpy(), index=keys, name="precipitation")
    if source.empty:
        raise ValueError("No observations to aggregate.")
    target_freq = "M" if period == "monthly" else "Y"
    start, end = source.index.min(), source.index.max()
    target_index = pd.period_range(start, end, freq=target_freq)

    rows: list[dict[str, Any]] = []
    for key in target_index:
        if source_frequency == "daily":
            mask = source.index.asfreq(target_freq) == key
            expected = int(key.days_in_month) if period == "monthly" else int(key.days_in_year)
        elif source_frequency == "monthly":
            mask = source.index.asfreq(target_freq) == key
            expected = 1 if period == "monthly" else 12
        else:
            if period == "monthly":
                raise ValueError("Annual source values cannot be disaggregated to monthly totals.")
            mask = source.index == key
            expected = 1
        selected = source.loc[mask]
        valid = selected.dropna()
        observed = int(len(valid))
        coverage = float(observed / expected) if expected else 0.0
        complete_edges = True
        if source_frequency == "daily":
            expected_start = key.start_time.normalize()
            expected_end = key.end_time.normalize()
            if period == "annual":
                expected_start = pd.Timestamp(year=key.year, month=1, day=1)
                expected_end = pd.Timestamp(year=key.year, month=12, day=31)
            actual_days = pd.DatetimeIndex(series.index[(series.index.to_period(target_freq) == key)])
            complete_edges = bool(len(actual_days) and actual_days.min().normalize() == expected_start and actual_days.max().normalize() == expected_end)
        elif source_frequency == "monthly" and period == "annual":
            actual_months = pd.PeriodIndex(source.index[mask]).month
            complete_edges = bool(len(actual_months) and min(actual_months) == 1 and max(actual_months) == 12)
        elif source_frequency == "monthly" and period == "monthly":
            complete_edges = observed == 1
        elif source_frequency == "annual":
            complete_edges = observed == 1

        amount = float(valid.sum()) if observed else np.nan
        accepted = observed > 0 and coverage >= coverage_threshold and complete_edges
        rows.append({
            "period": key,
            "precipitation": amount if accepted else np.nan,
            "observations": observed,
            "expected_observations": expected,
            "coverage": coverage,
        })

    result = pd.DataFrame(rows).set_index("period")
    result.index.name = "period"
    result.attrs["units"] = units
    result.attrs["source_provenance"] = dict(source_provenance or {})
    result.attrs["aggregation"] = {
        "period": period,
        "source_frequency": source_frequency,
        "coverage_threshold": coverage_threshold,
        "incomplete_periods": "missing; not imputed",
    }
    return result


def _mk_score(values: np.ndarray) -> int:
    values = values[np.isfinite(values)]
    score = 0
    for i in range(len(values) - 1):
        score += int(np.sign(values[i + 1:] - values[i]).sum())
    return score


def _mk_variance(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    tie_term = sum(int(count) * (int(count) - 1) * (2 * int(count) + 5) for count in counts)
    return float((n * (n - 1) * (2 * n + 5) - tie_term) / 18)


def _z_from_score(score: int, variance: float) -> float:
    if variance <= 0:
        return 0.0
    correction = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
    return float((score - correction) / sqrt(variance))


def _two_sided_normal_p(z_value: float) -> float:
    return min(1.0, max(0.0, erfc(abs(z_value) / sqrt(2.0))))


def _sen_slopes(series: pd.Series, seasonal: bool) -> np.ndarray:
    points = series.dropna()
    if isinstance(points.index, pd.PeriodIndex):
        periods = points.index
    else:
        periods = points.index.to_period("M" if seasonal else "Y")
    years = np.asarray([period.year + ((period.month - 1) / 12 if seasonal else 0) for period in periods], dtype=float)
    values = points.to_numpy(dtype=float)
    months = np.asarray([period.month for period in periods]) if seasonal else np.zeros(len(points), dtype=int)
    slopes: list[float] = []
    for i in range(len(values) - 1):
        later = np.arange(i + 1, len(values))
        if seasonal:
            later = later[months[later] == months[i]]
        for j in later:
            elapsed = years[j] - years[i]
            if elapsed > 0:
                slopes.append(float((values[j] - values[i]) / elapsed))
    return np.asarray(slopes, dtype=float)


def _sen_slope(series: pd.Series, seasonal: bool) -> float | None:
    slopes = _sen_slopes(series, seasonal)
    return float(np.median(slopes)) if len(slopes) else None


def _annual_hamed_rao(series: pd.Series, alpha: float) -> dict[str, Any]:
    """Hamed–Rao variance correction for annual Mann–Kendall testing."""
    values_full = series.to_numpy(dtype=float)
    valid = np.isfinite(values_full)
    values = values_full[valid]
    n = len(values)
    score = _mk_score(values)
    variance = _mk_variance(values)
    slope = _sen_slope(series, seasonal=False)
    correction_factor = 1.0
    used_lags: list[int] = []
    if n >= 4 and slope is not None:
        positions = np.arange(len(values_full), dtype=float)
        detrended = values_full[valid] - slope * positions[valid]
        ranks = np.full(len(values_full), np.nan, dtype=float)
        ranks[valid] = pd.Series(detrended).rank(method="average").to_numpy(dtype=float)
        max_lag = min(len(values_full) - 1, 10)
        threshold = NormalDist().inv_cdf(1 - alpha / 2) / sqrt(n)
        correction_sum = 0.0
        for lag in range(1, max_lag + 1):
            left, right = ranks[:-lag], ranks[lag:]
            pair_mask = np.isfinite(left) & np.isfinite(right)
            if int(pair_mask.sum()) < 4:
                continue
            x, y = left[pair_mask], right[pair_mask]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            rho = float(np.corrcoef(x, y)[0, 1])
            if abs(rho) > threshold:
                used_lags.append(lag)
                correction_sum += (n - lag) * (n - lag - 1) * (n - lag - 2) * rho
        correction_factor = max(0.0, 1.0 + 2.0 * abs(correction_sum) / (n * (n - 1) * (n - 2)))
    adjusted_variance = variance * correction_factor
    z_value = _z_from_score(score, adjusted_variance)
    p_value = _two_sided_normal_p(z_value)
    return {
        "statistic": int(score),
        "variance": float(adjusted_variance),
        "z_score": z_value,
        "p_value": p_value,
        "test_method": "hamed_rao_modified_mann_kendall",
        "autocorrelation_adjustment": {
            "method": "Hamed-Rao rank autocorrelation variance correction",
            "lags_used": used_lags,
            "variance_inflation_factor": correction_factor,
        },
    }


def _seasonal_matrix(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    periods = series.index.to_period("M")
    first_year, last_year = int(periods.year.min()), int(periods.year.max())
    years = np.arange(first_year, last_year + 1)
    matrix = np.full((len(years), 12), np.nan, dtype=float)
    row_by_year = {year: index for index, year in enumerate(years)}
    for timestamp, value in series.items():
        period = timestamp.to_period("M")
        matrix[row_by_year[period.year], period.month - 1] = value
    return years.astype(float), matrix


def _seasonal_score(matrix: np.ndarray) -> int:
    return int(sum(_mk_score(matrix[:, month]) for month in range(matrix.shape[1])))


def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    chosen: list[int] = []
    while len(chosen) < n:
        start = int(rng.integers(0, n))
        chosen.extend((start + offset) % n for offset in range(block_length))
    return np.asarray(chosen[:n], dtype=int)


def _moving_block_bootstrap(
    series: pd.Series,
    *,
    seasonal: bool,
    slope: float,
    observed_score: int,
    n_resamples: int,
    block_length: int | None,
    seed: int,
    confidence_level: float,
) -> tuple[float, tuple[float, float] | None, int]:
    """Block-bootstrap a no-trend p-value and Sen-slope percentile interval."""
    rng = np.random.default_rng(seed)
    if seasonal:
        years, matrix = _seasonal_matrix(series)
        n = len(years)
        length = block_length or max(2, int(round(n ** (1 / 3))))
        length = min(max(1, length), n)
        centered_time = years - np.median(years)
        baseline = np.full(12, np.nan)
        residuals = np.full_like(matrix, np.nan)
        for month in range(12):
            valid = np.isfinite(matrix[:, month])
            if valid.any():
                baseline[month] = float(np.median(matrix[valid, month] - slope * centered_time[valid]))
                residuals[valid, month] = matrix[valid, month] - baseline[month] - slope * centered_time[valid]
        null_scores: list[int] = []
        slope_samples: list[float] = []
        for _ in range(n_resamples):
            indices = _block_indices(n, length, rng)
            null_matrix = baseline[None, :] + residuals[indices, :]
            null_scores.append(_seasonal_score(null_matrix))
            fitted_matrix = baseline[None, :] + slope * centered_time[:, None] + residuals[indices, :]
            temp = _matrix_series(years, fitted_matrix)
            sample_slope = _sen_slope(temp, seasonal=True)
            if sample_slope is not None:
                slope_samples.append(sample_slope)
    else:
        full = series.to_numpy(dtype=float)
        n = len(full)
        length = block_length or max(2, int(round(n ** (1 / 3))))
        length = min(max(1, length), n)
        positions = np.arange(n, dtype=float)
        valid = np.isfinite(full)
        observed_years = positions[valid]
        fitted = float(np.nanmedian(full - slope * positions)) + slope * positions
        residuals = full - fitted
        null_scores = []
        slope_samples = []
        baseline = float(np.nanmedian(full - slope * positions))
        for _ in range(n_resamples):
            indices = _block_indices(n, length, rng)
            null_values = baseline + residuals[indices]
            null_scores.append(_mk_score(null_values))
            fitted_values = baseline + slope * positions + residuals[indices]
            fitted_values[~valid] = np.nan
            sample = pd.Series(fitted_values, index=series.index)
            sample_slope = _sen_slope(sample, seasonal=False)
            if sample_slope is not None:
                slope_samples.append(sample_slope)

    p_value = (1 + sum(abs(score) >= abs(observed_score) for score in null_scores)) / (1 + len(null_scores))
    interval = None
    if slope_samples:
        tail = (1.0 - confidence_level) / 2.0
        lower, upper = np.quantile(np.asarray(slope_samples), [tail, 1.0 - tail])
        interval = (float(lower), float(upper))
    return float(p_value), interval, int(length)


def _matrix_series(years: np.ndarray, matrix: np.ndarray) -> pd.Series:
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row, year in enumerate(years.astype(int)):
        for month in range(12):
            dates.append(pd.Timestamp(year=year, month=month + 1, day=15))
            values.append(float(matrix[row, month]))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="precipitation")


def calculate_significance(
    data: pd.Series | pd.DataFrame,
    temporal_frequency: Frequency = "annual",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    n_resamples: int = 499,
    block_length: int | None = None,
    random_seed: int = 0,
    value_column: str | None = None,
    timestamp_column: str | None = None,
) -> dict[str, Any]:
    """Calculate Mann–Kendall significance and a Sen-slope interval.

    Monthly data use seasonal Mann–Kendall with a year-block bootstrap of the
    trend-free residuals for the p-value and slope interval. Annual data use
    the Hamed–Rao variance-corrected Mann–Kendall test and the same block
    bootstrap for the slope interval. The bootstrap is reproducible by seed.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    series = _as_series(data, value_column=value_column, timestamp_column=timestamp_column)
    if series.index.has_duplicates:
        raise ValueError("Duplicate timestamps must be resolved before trend analysis.")
    if ((series.dropna() < 0) | ~np.isfinite(series.dropna())).any():
        raise ValueError("Precipitation values must be finite and non-negative.")
    if temporal_frequency not in ("monthly", "annual"):
        raise ValueError("temporal_frequency must be monthly or annual")

    if temporal_frequency == "monthly":
        if series.index.to_period("M").has_duplicates:
            raise ValueError("Monthly trend analysis requires at most one value per calendar month.")
        series.index = series.index.to_period("M").to_timestamp(how="start")
        seasonal = True
        score = _seasonal_score(_seasonal_matrix(series)[1])
        slope = _sen_slope(series, seasonal=True)
        years, matrix = _seasonal_matrix(series)
        season_counts = np.isfinite(matrix).sum(axis=0)
        p_value, interval, chosen_block = _moving_block_bootstrap(
            series,
            seasonal=True,
            slope=float(slope or 0.0),
            observed_score=score,
            n_resamples=n_resamples,
            block_length=block_length,
            seed=random_seed,
            confidence_level=confidence_level,
        )
        result = {
            "statistic": score,
            "variance": None,
            "z_score": None,
            "p_value": p_value,
            "test_method": "seasonal_mann_kendall_year_block_bootstrap",
            "autocorrelation_adjustment": {
                "method": "moving year-block bootstrap of trend-free seasonal residuals",
                "block_length_years": chosen_block,
                "season_observation_counts": {str(month + 1): int(count) for month, count in enumerate(season_counts)},
            },
        }
    else:
        annual_periods = series.index.to_period("Y")
        if annual_periods.has_duplicates:
            raise ValueError("Annual trend analysis requires one annual total per calendar year; aggregate subannual data first.")
        series.index = series.index.to_period("Y").to_timestamp(how="start")
        seasonal = False
        annual_index = pd.period_range(series.index.min().to_period("Y"), series.index.max().to_period("Y"), freq="Y").to_timestamp(how="start")
        series = series.reindex(annual_index)
        slope = _sen_slope(series, seasonal=False)
        result = _annual_hamed_rao(series, alpha)
        p_value, interval, chosen_block = _moving_block_bootstrap(
            series,
            seasonal=False,
            slope=float(slope or 0.0),
            observed_score=int(result["statistic"]),
            n_resamples=n_resamples,
            block_length=block_length,
            seed=random_seed,
            confidence_level=confidence_level,
        )
        if series.isna().any():
            result["p_value"] = p_value
            result["variance"] = None
            result["z_score"] = None
            result["hamed_rao_p_value"] = None
            result["test_method"] = "annual_mann_kendall_year_block_bootstrap_with_missing_periods"
            result["autocorrelation_adjustment"] = {
                "method": "moving year-block bootstrap of trend-free annual residuals; used because annual periods are missing",
                "block_length_years": chosen_block,
            }
        else:
            result["hamed_rao_p_value"] = result["p_value"]
            result["bootstrap_p_value_diagnostic"] = p_value
        result["autocorrelation_adjustment"]["confidence_interval_block_length_years"] = chosen_block

    result["slope"] = slope
    result["slope_confidence_interval"] = interval
    result["confidence_level"] = confidence_level
    result["alpha"] = alpha
    result["sample_size"] = int(series.notna().sum())
    result["valid_pair_count"] = int(len(_sen_slopes(series, seasonal)))
    result["confidence_interval_method"] = "moving_block_bootstrap_percentile"
    result["bootstrap"] = {"resamples": n_resamples, "random_seed": random_seed}
    return result


def calculate_trend(
    data: pd.Series | pd.DataFrame,
    temporal_frequency: Frequency = "annual",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    n_resamples: int = 499,
    block_length: int | None = None,
    random_seed: int = 0,
    min_observations: int = 10,
    min_observations_per_season: int = 3,
    min_seasons: int = 10,
    value_column: str | None = None,
    timestamp_column: str | None = None,
) -> dict[str, Any]:
    """Estimate trend direction, Sen slope, significance, and uncertainty.

    Eligibility thresholds are operational defaults and should be versioned
    configuration in production. Insufficient records have no p-value/label.
    """
    series = _as_series(data, value_column=value_column, timestamp_column=timestamp_column)
    if series.index.has_duplicates:
        raise ValueError("Duplicate timestamps must be resolved before trend analysis.")
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    valid = series.dropna()
    reasons: list[str] = []
    if temporal_frequency not in ("monthly", "annual"):
        raise ValueError("temporal_frequency must be monthly or annual")
    if temporal_frequency == "annual":
        annual_periods = series.index.to_period("Y")
        if annual_periods.has_duplicates:
            raise ValueError("Annual trend analysis requires one annual total per calendar year; aggregate subannual data first.")
    if len(valid) < min_observations:
        reasons.append(f"Requires at least {min_observations} valid observations; found {len(valid)}.")
    season_counts: dict[int, int] = {}
    if temporal_frequency == "monthly":
        month_keys = valid.index.to_period("M").month
        season_counts = {month: int(sum(month_keys == month)) for month in range(1, 13)}
        eligible_seasons = sum(count >= min_observations_per_season for count in season_counts.values())
        if eligible_seasons < min_seasons:
            reasons.append(
                f"Requires {min_observations_per_season} observations in at least {min_seasons} calendar months; found {eligible_seasons}."
            )
    if reasons:
        return {
            "status": "insufficient_data",
            "label": "insufficient data",
            "direction": "undetermined",
            "significant": None,
            "p_value": None,
            "slope": None,
            "slope_confidence_interval": None,
            "sample_size": int(len(valid)),
            "season_observation_counts": {str(k): v for k, v in season_counts.items()},
            "insufficient_data_reasons": reasons,
        }
    result = calculate_significance(
        valid,
        temporal_frequency=temporal_frequency,
        alpha=alpha,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        block_length=block_length,
        random_seed=random_seed,
    )
    slope = result["slope"]
    p_value = result["p_value"]
    significant = bool(p_value < alpha)
    direction = "increasing" if slope is not None and slope > 0 else "decreasing" if slope is not None and slope < 0 else "no_direction"
    if significant and direction == "increasing":
        label = "increasing trend"
    elif significant and direction == "decreasing":
        label = "decreasing trend"
    else:
        label = "no statistically significant trend detected"
    return {
        "status": "ok",
        "label": label,
        "direction": direction,
        "significant": significant,
        **result,
        "season_observation_counts": {str(k): v for k, v in season_counts.items()} if season_counts else None,
        "insufficient_data_reasons": [],
    }


def _fdr_adjust(p_values: np.ndarray, method: Literal["by", "bh"] = "by") -> np.ndarray:
    """Return Benjamini–Yekutieli or Benjamini–Hochberg adjusted q-values."""
    if method not in ("by", "bh"):
        raise ValueError("fdr_method must be 'by' or 'bh'")
    p = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    factor = sum(1 / rank for rank in range(1, m + 1)) if method == "by" else 1.0
    adjusted_sorted = np.minimum.accumulate((ranked * m * factor / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def analyze_spatial_trends(
    data: pd.DataFrame,
    temporal_frequency: Frequency = "annual",
    time_column: str = "time",
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    value_column: str = "precipitation",
    alpha: float = 0.05,
    fdr_method: Literal["by", "bh"] = "by",
    n_resamples: int = 499,
    block_length: int | None = None,
    random_seed: int = 0,
    min_observations: int = 10,
    min_observations_per_season: int = 3,
    min_seasons: int = 10,
    units: str | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Analyze each (latitude, longitude) cell and apply spatial FDR.

    ``data`` is long-form with one observation per cell and timestamp. Multiple
    grid cells are independent analysis units; only eligible raw p-values enter
    the declared correction family.
    """
    required = {time_column, latitude_column, longitude_column, value_column}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required spatial columns: {sorted(missing_columns)}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if fdr_method not in ("by", "bh"):
        raise ValueError("fdr_method must be 'by' or 'bh'")
    frame = data.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    if frame[time_column].isna().any():
        raise ValueError("Spatial input contains invalid timestamps.")
    if frame[[latitude_column, longitude_column]].isna().any().any():
        raise ValueError("Spatial coordinates must be present for every observation.")
    coordinates = frame[[latitude_column, longitude_column]].apply(pd.to_numeric, errors="coerce")
    if coordinates.isna().any().any() or not np.isfinite(coordinates.to_numpy(dtype=float)).all():
        raise ValueError("Spatial coordinates must be finite numeric values.")
    if not coordinates[latitude_column].between(-90, 90).all() or not coordinates[longitude_column].between(-180, 180).all():
        raise ValueError("Latitude/longitude coordinates are outside valid geographic bounds.")
    frame[latitude_column] = coordinates[latitude_column]
    frame[longitude_column] = coordinates[longitude_column]
    if frame.duplicated([time_column, latitude_column, longitude_column]).any():
        raise ValueError("Duplicate timestamp/grid-cell observations must be resolved first.")

    results: list[dict[str, Any]] = []
    grouped = frame.groupby([latitude_column, longitude_column], sort=True, dropna=False)
    for cell_number, ((latitude, longitude), cell) in enumerate(grouped):
        series = pd.Series(cell[value_column].to_numpy(), index=pd.DatetimeIndex(cell[time_column]), name="precipitation")
        cell_seed = int(np.random.SeedSequence([random_seed, cell_number]).generate_state(1)[0])
        try:
            trend = calculate_trend(
                series,
                temporal_frequency=temporal_frequency,
                alpha=alpha,
                n_resamples=n_resamples,
                block_length=block_length,
                random_seed=cell_seed,
                min_observations=min_observations,
                min_observations_per_season=min_observations_per_season,
                min_seasons=min_seasons,
            )
        except (TypeError, ValueError, ArithmeticError) as exc:
            trend = {
                "status": "invalid_data",
                "label": "invalid data",
                "direction": "undetermined",
                "significant": None,
                "p_value": None,
                "slope": None,
                "slope_confidence_interval": None,
                "sample_size": int(cell[value_column].notna().sum()),
                "insufficient_data_reasons": [str(exc)],
            }
        results.append({
            "latitude": float(latitude),
            "longitude": float(longitude),
            "cell_id": f"{float(latitude):.6f},{float(longitude):.6f}",
            "units": units,
            "source_provenance": dict(source_provenance or {}),
            **trend,
        })

    eligible_indices = [index for index, result in enumerate(results) if result.get("p_value") is not None]
    p_values = np.asarray([results[index]["p_value"] for index in eligible_indices], dtype=float)
    q_values = _fdr_adjust(p_values, method=fdr_method)
    for index, q_value in zip(eligible_indices, q_values):
        result = results[index]
        result["q_value"] = float(q_value)
        result["significant"] = bool(q_value < alpha)
        direction = result.get("direction")
        if result["significant"] and direction == "increasing":
            result["label"] = "increasing trend"
        elif result["significant"] and direction == "decreasing":
            result["label"] = "decreasing trend"
        else:
            result["label"] = "no statistically significant trend detected"
        result["status"] = "ok"
    for index, result in enumerate(results):
        if index not in eligible_indices:
            result["q_value"] = None
    output = pd.DataFrame(results)
    output.attrs["testing_family"] = {
        "method": fdr_method,
        "eligible_hypotheses": len(eligible_indices),
        "excluded_cells": len(results) - len(eligible_indices),
        "alpha": alpha,
        "temporal_frequency": temporal_frequency,
    }
    output.attrs["source_provenance"] = dict(source_provenance or {})
    output.attrs["units"] = units
    output.attrs["analysis_settings"] = {
        "alpha": alpha,
        "fdr_method": fdr_method,
        "n_resamples": n_resamples,
        "block_length_years": block_length,
        "random_seed": random_seed,
        "min_observations": min_observations,
    }
    return output


def generate_analysis_summary(results: pd.DataFrame | list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize spatial classifications without conflating missing tests.

    Counts increasing/decreasing only when the adjusted test is significant;
    nonsignificant eligible locations and insufficient-data locations are
    counted separately.
    """
    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if frame.empty:
        return {
            "location_count": 0,
            "increasing_locations": 0,
            "decreasing_locations": 0,
            "statistically_insignificant_locations": 0,
            "insufficient_data_locations": 0,
            "invalid_data_locations": 0,
            "significant_locations": 0,
        }
    labels = frame.get("label", pd.Series("insufficient data", index=frame.index)).fillna("insufficient data")
    insufficient = labels.eq("insufficient data")
    increasing = labels.eq("increasing trend")
    decreasing = labels.eq("decreasing trend")
    nonsignificant = labels.eq("no statistically significant trend detected")
    return {
        "location_count": int(len(frame)),
        "increasing_locations": int(increasing.sum()),
        "decreasing_locations": int(decreasing.sum()),
        "statistically_insignificant_locations": int(nonsignificant.sum()),
        "insufficient_data_locations": int(insufficient.sum()),
        "invalid_data_locations": int(labels.eq("invalid data").sum()),
        "significant_locations": int((increasing | decreasing).sum()),
    }
