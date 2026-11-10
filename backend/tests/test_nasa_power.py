import json
import unittest
from datetime import date
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pandas as pd

from backend.app.services import analysis, nasa_power


def regional_payload(value=1.25):
    return {
        "header": {"title": "NASA/POWER regional daily data", "api_version": "test"},
        "messages": [],
        "parameters": {"PRECTOTCORR": {"units": "mm/day", "longname": "Corrected Precipitation"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [36.625, -1.75, 1000]},
                "properties": {
                    "parameter": {"PRECTOTCORR": {"20200101": value, "20200102": -999}}
                },
            }
        ],
    }


class NasaPowerTests(unittest.TestCase):
    def test_builds_configured_regional_request_and_checks_dates_and_bounds(self):
        url = nasa_power.build_regional_request_url(
            start_date="1981-01-01",
            end_date=date(1981, 1, 31),
            latitude_min=-2,
            latitude_max=-1,
            longitude_min=36,
            longitude_max=37,
        )
        self.assertIn("/api/temporal/daily/regional?", url)
        self.assertIn("parameters=PRECTOTCORR", url)
        self.assertIn("time-standard=utc", url)
        self.assertIn("start=19810101", url)
        with self.assertRaises(ValueError):
            nasa_power.build_regional_request_url(
                start_date="2021-01-02", end_date="2021-01-01",
                latitude_min=0, latitude_max=1, longitude_min=0, longitude_max=1,
            )

    @patch("backend.app.services.nasa_power._read_json", return_value=regional_payload())
    def test_successful_fetch_normalizes_response_and_preserves_provenance(self, read_json):
        result = nasa_power.fetch_regional_precipitation(
            start_date="2020-01-01", end_date="2020-01-02",
            latitude_min=-2, latitude_max=-1, longitude_min=36, longitude_max=37,
        )
        self.assertEqual(list(result.columns), ["time", "latitude", "longitude", "precipitation"])
        self.assertEqual(result.loc[0, "precipitation"], 1.25)
        self.assertTrue(pd.isna(result.loc[1, "precipitation"]))
        self.assertEqual(result.loc[0, "time"], pd.Timestamp("2020-01-01"))
        self.assertEqual(result.loc[0, "latitude"], -1.75)
        self.assertEqual(result.loc[0, "longitude"], 36.625)
        self.assertEqual(result.attrs["units"], "mm")
        self.assertEqual(result.attrs["source_provenance"]["parameter"], "PRECTOTCORR")
        self.assertEqual(result.attrs["source_provenance"]["source_units"], "mm/day")
        self.assertEqual(result.attrs["missing_observation_count"], 1)
        read_json.assert_called_once()

    @patch("backend.app.services.nasa_power.urlopen", side_effect=HTTPError("url", 503, "Unavailable", {}, BytesIO(b"down")))
    def test_http_failure_is_reported(self, _urlopen):
        with self.assertRaises(nasa_power.NasaPowerRequestError):
            nasa_power._read_json("https://example.test", timeout=1)

    @patch("backend.app.services.nasa_power.urlopen", side_effect=URLError("offline"))
    def test_network_failure_is_reported(self, _urlopen):
        with self.assertRaises(nasa_power.NasaPowerRequestError):
            nasa_power._read_json("https://example.test", timeout=1)

    def test_normalizer_rejects_invalid_response_shapes_and_observations(self):
        with self.assertRaises(nasa_power.NasaPowerResponseError):
            nasa_power.normalize_regional_response({}, requested_start="2020-01-01", requested_end="2020-01-02", bounds={})
        bad = regional_payload("not-a-number")
        with self.assertRaises(nasa_power.NasaPowerResponseError):
            nasa_power.normalize_regional_response(bad, requested_start="2020-01-01", requested_end="2020-01-02", bounds={})
        negative = regional_payload(-2)
        with self.assertRaises(nasa_power.NasaPowerResponseError):
            nasa_power.normalize_regional_response(negative, requested_start="2020-01-01", requested_end="2020-01-02", bounds={})

    def test_unit_and_temporal_handling_are_explicitly_normalized(self):
        result = nasa_power.normalize_regional_response(
            regional_payload(2.5), requested_start="2020-01-01", requested_end="2020-01-02", bounds={}
        )
        provenance = result.attrs["source_provenance"]
        self.assertEqual(provenance["temporal_resolution"], "daily UTC; daily precipitation value")
        self.assertEqual(provenance["source_units"], "mm/day")
        self.assertEqual(provenance["units"], "mm")
        self.assertIn("one-day accumulated amount mm", " ".join(provenance["transformations"]))
        self.assertEqual(result.loc[0, "precipitation"], 2.5)

    @patch("backend.app.services.nasa_power.urlopen", side_effect=TimeoutError("request timed out"))
    def test_timeout_is_reported(self, _urlopen):
        with self.assertRaisesRegex(nasa_power.NasaPowerRequestError, "request timed out"):
            nasa_power._read_json("https://example.test", timeout=0.01)

    def test_result_can_be_aggregated_and_passed_into_spatial_analysis(self):
        # Test-only mocked records; no environmental measurements are embedded in production code.
        dates = pd.date_range("2000-01-01", "2011-12-31", freq="D")
        payload = {
            "parameters": {"PRECTOTCORR": {"units": "mm/day", "longname": "Corrected Precipitation"}},
            "features": [
                {"geometry": {"type": "Point", "coordinates": [36.625, -1.75]},
                 "properties": {"parameter": {"PRECTOTCORR": {stamp.strftime("%Y%m%d"): 1.0
                     for stamp in dates}}}}
            ]
        }
        daily = nasa_power.normalize_regional_response(
            payload, requested_start="2000-01-01", requested_end="2011-12-31", bounds={}
        )
        monthly = nasa_power.aggregate_regional_precipitation(
            daily, period="monthly", coverage_threshold=1.0,
        )
        self.assertTrue(monthly["precipitation"].notna().all())
        self.assertEqual(monthly.iloc[0]["precipitation"], 31.0)
        self.assertEqual(monthly.attrs["units"], "mm")
        # The adapter provides the exact long-form columns consumed by the spatial engine.
        analyzed = analysis.analyze_spatial_trends(
            monthly[["time", "latitude", "longitude", "precipitation"]],
            temporal_frequency="monthly", min_observations=10, min_observations_per_season=1,
            min_seasons=10, n_resamples=19,
            source_provenance=daily.attrs["source_provenance"], units=daily.attrs["units"],
        )
        self.assertEqual(len(analyzed), 1)
        self.assertEqual(analyzed.iloc[0]["source_provenance"]["provider"], "NASA POWER")

    def test_response_dates_must_be_in_requested_range(self):
        with self.assertRaises(nasa_power.NasaPowerResponseError):
            nasa_power.normalize_regional_response(
                regional_payload(), requested_start="2020-01-02", requested_end="2020-01-02", bounds={}
            )

    def test_rejects_unexpected_units_in_api_metadata(self):
        payload = regional_payload()
        payload["parameters"]["PRECTOTCORR"]["units"] = "mm"
        with self.assertRaises(nasa_power.NasaPowerResponseError):
            nasa_power.normalize_regional_response(
                payload, requested_start="2020-01-01", requested_end="2020-01-02", bounds={}
            )


if __name__ == "__main__":
    unittest.main()
