import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import interpretation, nasa_power


def mocked_regional_response(start="2000-01-01", end="2011-12-31"):
    """Create a deterministic, synthetic API fixture for pipeline tests only."""
    dates = pd.date_range(start, end, freq="D")
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [36.625, -1.75, 1200]},
                "properties": {
                    "parameter": {"PRECTOTCORR": {stamp.strftime("%Y%m%d"): 1.0 for stamp in dates}}
                },
            }
        ],
        "header": {"fill_value": -999, "title": "mocked NASA POWER data"},
        "messages": [],
        "parameters": {"PRECTOTCORR": {"units": "mm/day", "longname": "Corrected Precipitation"}},
        "times": {"data": 0, "process": 0},
    }


class ApiPipelineTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def request_body(self, **overrides):
        body = {
            "start_date": "2000-01-01",
            "end_date": "2011-12-31",
            "bbox": {
                "latitude_min": -2,
                "latitude_max": -1,
                "longitude_min": 36,
                "longitude_max": 37,
            },
            "period": "annual",
            "n_resamples": 19,
            "include_interpretation": False,
        }
        body.update(overrides)
        return body

    @patch("backend.app.services.nasa_power._read_json", side_effect=lambda _url, _timeout: mocked_regional_response())
    def test_pipeline_fetches_aggregates_analyzes_and_returns_structured_results(self, read_json):
        response = self.client.post("/api/v1/analysis/precipitation", json=self.request_body())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["data"]["source_provenance"]["provider"], "NASA POWER")
        self.assertEqual(body["data"]["units"], "mm")
        self.assertEqual(body["analysis"]["period"], "annual")
        self.assertEqual(body["analysis"]["summary"]["location_count"], 1)
        self.assertEqual(len(body["analysis"]["results"]), 1)
        result = body["analysis"]["results"][0]
        self.assertEqual(result["latitude"], -1.75)
        self.assertEqual(result["longitude"], 36.625)
        self.assertEqual(result["status"], "ok")
        self.assertIn("p_value", result)
        self.assertIn("slope_confidence_interval", result)
        self.assertEqual(body["validation"]["cells_checked"], 1)
        self.assertEqual(body["validation"]["cells_with_errors"], 0)
        self.assertIsNone(body["interpretation"])
        read_json.assert_called_once()

    @patch("backend.app.services.nasa_power._read_json", side_effect=lambda _url, _timeout: mocked_regional_response())
    def test_monthly_request_reaches_seasonal_trend_method(self, _read_json):
        response = self.client.post(
            "/api/v1/analysis/precipitation",
            json=self.request_body(period="monthly", min_observations_per_season=3, min_seasons=10),
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["analysis"]["results"][0]
        self.assertEqual(result["test_method"], "seasonal_mann_kendall_year_block_bootstrap")

    @patch("backend.app.services.interpretation.interpret_analysis_results", return_value={
        "executive_summary": "Test interpretation based on computed results.",
        "trend_explanation": "The computed series has no detected trend.",
        "statistical_interpretation": "The statistical result is not significant.",
        "environmental_implications": [],
        "recommendations": [],
        "further_investigation": [],
        "limitations": [],
    })
    @patch("backend.app.services.nasa_power._read_json", side_effect=lambda _url, _timeout: mocked_regional_response())
    def test_optional_interpretation_receives_backend_computed_analysis(self, _read_json, interpret):
        response = self.client.post(
            "/api/v1/analysis/precipitation",
            json=self.request_body(include_interpretation=True),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["interpretation"]["status"], "completed")
        self.assertEqual(response.json()["interpretation"]["result"]["recommendations"], [])
        self.assertIsInstance(interpret.call_args.args[0], pd.DataFrame)

    @patch("backend.app.services.interpretation.interpret_analysis_results", side_effect=interpretation.InterpretationConfigurationError("Set LLM_API_KEY to enable interpretation."))
    @patch("backend.app.services.nasa_power._read_json", side_effect=lambda _url, _timeout: mocked_regional_response())
    def test_interpretation_failure_does_not_discard_completed_statistics(self, _read_json, _interpret):
        response = self.client.post(
            "/api/v1/analysis/precipitation",
            json=self.request_body(include_interpretation=True),
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["analysis"]["summary"]["location_count"], 1)
        self.assertEqual(body["interpretation"]["status"], "unavailable")

    @patch("backend.app.services.nasa_power.fetch_regional_precipitation", side_effect=nasa_power.NasaPowerRequestError("NASA service unavailable"))
    def test_nasa_failure_is_returned_as_upstream_error(self, _fetch):
        response = self.client.post("/api/v1/analysis/precipitation", json=self.request_body())
        self.assertEqual(response.status_code, 502)
        self.assertIn("NASA service unavailable", response.json()["detail"])

    @patch("backend.app.services.nasa_power.fetch_regional_precipitation", side_effect=nasa_power.NasaPowerResponseError("invalid provider payload"))
    def test_invalid_upstream_payload_is_returned_as_bad_gateway(self, _fetch):
        response = self.client.post("/api/v1/analysis/precipitation", json=self.request_body())
        self.assertEqual(response.status_code, 502)

    @patch("backend.app.services.nasa_power.fetch_regional_precipitation", side_effect=RuntimeError("internal error details"))
    def test_unexpected_processing_failure_is_sanitized(self, _fetch):
        response = self.client.post("/api/v1/analysis/precipitation", json=self.request_body())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "The precipitation analysis pipeline failed.")

    def test_invalid_request_and_date_order_are_rejected(self):
        invalid_bbox = self.request_body()
        invalid_bbox["bbox"]["latitude_min"] = -3
        invalid_bbox["bbox"]["latitude_max"] = -5
        response = self.client.post("/api/v1/analysis/precipitation", json=invalid_bbox)
        self.assertEqual(response.status_code, 422)

        invalid_dates = self.request_body(start_date="2011-12-31", end_date="2000-01-01")
        response = self.client.post("/api/v1/analysis/precipitation", json=invalid_dates)
        self.assertEqual(response.status_code, 422)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
