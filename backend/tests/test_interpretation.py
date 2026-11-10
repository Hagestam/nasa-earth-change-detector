"""Tests for the interpretation boundary and LLM response validation."""

import json
import unittest

from backend.app.services.interpretation import (
    InterpretationAPIError,
    InterpretationConfigurationError,
    InterpretationValidationError,
    OUTPUT_FIELDS,
    interpret_analysis_results,
    validate_interpretation_response,
)


ANALYSIS = {
    "status": "ok",
    "label": "increasing trend",
    "direction": "increasing",
    "significant": True,
    "p_value": 0.01,
    "slope": 0.2,
    "slope_confidence_interval": [0.1, 0.3],
    "sample_size": 24,
    "test_method": "seasonal_mann_kendall_year_block_bootstrap",
    "units": "mm/year",
}


def valid_interpretation():
    return {
        "executive_summary": "The supplied result reports an increasing trend.",
        "trend_explanation": "The Sen slope is 0.2 mm/year, with an interval from 0.1 to 0.3.",
        "statistical_interpretation": "The engine classified this result as an increasing trend; the supplied p-value is 0.01.",
        "environmental_implications": ["This pattern may warrant checking local rainfall impacts against independent observations."],
        "recommendations": ["Compare this cell with independent precipitation records before making operational decisions."],
        "further_investigation": ["Review missingness and station observations for the analyzed period."],
        "limitations": ["A trend does not identify its cause or establish environmental impact."],
    }


def responses_api_result(text):
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
    }


class InterpretationValidationTests(unittest.TestCase):
    def test_response_has_exact_required_shape(self):
        result = validate_interpretation_response(valid_interpretation(), evidence=ANALYSIS)
        self.assertEqual(set(result), set(OUTPUT_FIELDS))

    def test_rejects_numeric_claim_absent_from_statistics(self):
        result = valid_interpretation()
        result["executive_summary"] = "The slope is 9 mm/year."
        with self.assertRaisesRegex(InterpretationValidationError, "numeric claim"):
            validate_interpretation_response(result, evidence=ANALYSIS)

    def test_rejects_missing_keys_wrong_types_and_invalid_json(self):
        result = valid_interpretation()
        result.pop("limitations")
        with self.assertRaisesRegex(InterpretationValidationError, "invalid schema"):
            validate_interpretation_response(result)
        result = valid_interpretation()
        result["recommendations"] = "not an array"
        with self.assertRaisesRegex(InterpretationValidationError, "list of non-empty strings"):
            validate_interpretation_response(result)
        with self.assertRaisesRegex(InterpretationValidationError, "not valid JSON"):
            validate_interpretation_response("not json")


class InterpretationClientTests(unittest.TestCase):
    def setUp(self):
        self.environ = {
            "LLM_API_KEY": "unit-test-key",
            "LLM_MODEL": "test-model",
            "LLM_API_URL": "https://example.invalid/v1/responses",
        }

    def test_sends_only_calculated_results_and_returns_validated_json(self):
        calls = []

        def fake_transport(url, headers, body, timeout):
            calls.append((url, headers, body, timeout))
            return responses_api_result(json.dumps(valid_interpretation()))

        result = interpret_analysis_results(ANALYSIS, environ=self.environ, transport=fake_transport)
        self.assertEqual(result, valid_interpretation())
        self.assertEqual(calls[0][0], self.environ["LLM_API_URL"])
        self.assertEqual(calls[0][1]["Authorization"], "Bearer unit-test-key")
        self.assertFalse(calls[0][2]["store"])
        self.assertEqual(calls[0][2]["text"]["format"]["type"], "json_schema")
        self.assertIn("p_value", calls[0][2]["input"][1]["content"][0]["text"])

    def test_rejects_fabricated_response_number(self):
        interpretation = valid_interpretation()
        interpretation["executive_summary"] = "The precipitation slope is 8 mm/year."
        with self.assertRaisesRegex(InterpretationValidationError, "numeric claim"):
            interpret_analysis_results(
                ANALYSIS,
                environ=self.environ,
                transport=lambda *_: responses_api_result(json.dumps(interpretation)),
            )

    def test_handles_missing_configuration_and_refusal(self):
        with self.assertRaises(InterpretationConfigurationError):
            interpret_analysis_results(ANALYSIS, environ={})
        refusal = {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "declined"}]}]}
        with self.assertRaises(InterpretationAPIError):
            interpret_analysis_results(ANALYSIS, environ=self.environ, transport=lambda *_: refusal)

    def test_rejects_invalid_analysis_before_calling_api(self):
        invalid = dict(ANALYSIS, p_value=1.2)
        called = []
        with self.assertRaisesRegex(InterpretationValidationError, "invalid p_value"):
            interpret_analysis_results(invalid, environ=self.environ, transport=lambda *_: called.append(True))
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
