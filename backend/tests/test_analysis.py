"""Unit tests for the precipitation analysis service.

Deterministic synthetic series are used only to verify algorithm behavior;
they are not environmental measurements or project findings.
"""

import unittest

import numpy as np
import pandas as pd

from backend.app.services.analysis import (
    aggregate_precipitation,
    analyze_spatial_trends,
    calculate_significance,
    calculate_trend,
    generate_analysis_summary,
    validate_environmental_data,
)


class ValidationTests(unittest.TestCase):
    def test_reports_gaps_missing_values_negative_values_and_duplicates(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-03-01", "2020-04-01"])
        series = pd.Series([1.0, 2.0, np.nan, -1.0], index=dates)
        result = validate_environmental_data(series, expected_frequency="monthly")
        self.assertFalse(result["valid"])
        self.assertEqual(result["duplicate_timestamp_count"], 2)
        self.assertEqual(result["missing_period_count"], 1)
        self.assertEqual(result["missing_value_count"], 2)
        self.assertEqual(result["invalid_value_count"], 1)

    def test_accepts_zero_as_valid_precipitation(self):
        series = pd.Series([0.0, 1.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
        result = validate_environmental_data(series, expected_frequency="monthly")
        self.assertTrue(result["valid"])
        self.assertEqual(result["valid_observation_count"], 2)

    def test_reports_non_numeric_measurement_as_invalid_not_missing(self):
        series = pd.Series(["not-a-measurement", 1.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
        result = validate_environmental_data(series, expected_frequency="monthly")
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_value_count"], 1)
        self.assertEqual(result["missing_value_count"], 0)


class AggregationTests(unittest.TestCase):
    def test_daily_to_monthly_marks_incomplete_edge_period_and_keeps_provenance(self):
        dates = pd.date_range("2020-01-01", "2020-02-28", freq="D")
        series = pd.Series(1.0, index=dates, name="precipitation")
        result = aggregate_precipitation(
            series,
            period="monthly",
            source_frequency="daily",
            units="mm",
            coverage_threshold=0.9,
            source_provenance={"dataset": "synthetic test fixture"},
        )
        self.assertEqual(result.loc[pd.Period("2020-01", freq="M"), "precipitation"], 31.0)
        self.assertTrue(np.isnan(result.loc[pd.Period("2020-02", freq="M"), "precipitation"]))
        self.assertEqual(result.attrs["units"], "mm")
        self.assertEqual(result.attrs["source_provenance"]["dataset"], "synthetic test fixture")

    def test_monthly_to_annual_sums_only_complete_year(self):
        dates = pd.date_range("2019-01-01", periods=24, freq="MS")
        series = pd.Series(np.ones(24), index=dates)
        result = aggregate_precipitation(series, period="annual", source_frequency="monthly", units="mm")
        self.assertEqual(result.loc[pd.Period("2019", freq="Y"), "precipitation"], 12.0)
        self.assertEqual(result.loc[pd.Period("2020", freq="Y"), "precipitation"], 12.0)


def monthly_trend(years=20, slope=0.2):
    dates = pd.date_range("2000-01-01", periods=years * 12, freq="MS")
    values = np.array([month + 1 + year * slope for year in range(years) for month in range(12)], dtype=float)
    return pd.Series(values, index=dates, name="precipitation")


class TrendTests(unittest.TestCase):
    def test_monthly_uses_seasonal_test_and_returns_deterministic_results(self):
        series = monthly_trend()
        first = calculate_trend(series, temporal_frequency="monthly", n_resamples=99, random_seed=7)
        second = calculate_trend(series, temporal_frequency="monthly", n_resamples=99, random_seed=7)
        self.assertEqual(first["test_method"], "seasonal_mann_kendall_year_block_bootstrap")
        self.assertEqual(first["direction"], "increasing")
        self.assertAlmostEqual(first["slope"], 0.2, places=6)
        self.assertEqual(first["p_value"], second["p_value"])
        self.assertIsNotNone(first["slope_confidence_interval"])

    def test_annual_uses_hamed_rao_and_reports_insufficient_data(self):
        dates = pd.date_range("2000-01-01", periods=18, freq="YS")
        series = pd.Series(np.arange(18, dtype=float) * 2, index=dates)
        result = calculate_significance(series, temporal_frequency="annual", n_resamples=99, random_seed=4)
        self.assertEqual(result["test_method"], "hamed_rao_modified_mann_kendall")
        self.assertGreater(result["slope"], 0)
        self.assertIn("bootstrap_p_value_diagnostic", result)
        short = calculate_trend(series.iloc[:5], temporal_frequency="annual")
        self.assertEqual(short["label"], "insufficient data")
        self.assertIsNone(short["p_value"])

    def test_annual_gaps_use_block_bootstrap_inference(self):
        dates = pd.date_range("2000-01-01", periods=18, freq="YS")
        series = pd.Series(np.arange(18, dtype=float) + 10, index=dates).drop(dates[8])
        result = calculate_significance(series, temporal_frequency="annual", n_resamples=99, random_seed=4)
        self.assertEqual(result["test_method"], "annual_mann_kendall_year_block_bootstrap_with_missing_periods")
        self.assertIsNone(result["hamed_rao_p_value"])
        self.assertIsNone(result["z_score"])

    def test_annual_trend_requires_aggregated_annual_totals(self):
        dates = pd.date_range("2000-01-01", periods=24, freq="MS")
        series = pd.Series(np.ones(24), index=dates)
        with self.assertRaisesRegex(ValueError, "aggregate subannual data first"):
            calculate_trend(series, temporal_frequency="annual")


class SpatialTests(unittest.TestCase):
    def test_spatial_cells_are_corrected_and_summary_separates_categories(self):
        dates = pd.date_range("2000-01-01", periods=20, freq="YS")
        rows = []
        for year_number, date in enumerate(dates):
            rows.append({"time": date, "latitude": 1.0, "longitude": 2.0, "precipitation": 10 + year_number})
            rows.append({"time": date, "latitude": 3.0, "longitude": 4.0, "precipitation": 30 - year_number})
            rows.append({"time": date, "latitude": 5.0, "longitude": 6.0, "precipitation": 10.0})
        rows.append({"time": dates[0], "latitude": 7.0, "longitude": 8.0, "precipitation": 1.0})
        spatial = analyze_spatial_trends(
            pd.DataFrame(rows),
            temporal_frequency="annual",
            n_resamples=99,
            random_seed=12,
            min_observations=10,
        )
        self.assertEqual(spatial.attrs["testing_family"]["method"], "by")
        self.assertEqual(spatial.attrs["testing_family"]["excluded_cells"], 1)
        self.assertTrue(spatial.loc[spatial["latitude"] == 1.0, "q_value"].notna().all())
        summary = generate_analysis_summary(spatial)
        self.assertEqual(summary["location_count"], 4)
        self.assertEqual(summary["increasing_locations"], 1)
        self.assertEqual(summary["decreasing_locations"], 1)
        self.assertEqual(summary["statistically_insignificant_locations"], 1)
        self.assertEqual(summary["insufficient_data_locations"], 1)


if __name__ == "__main__":
    unittest.main()
