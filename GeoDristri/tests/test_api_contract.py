import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import prediction_service  # noqa: E402


class UnifiedPredictContractTests(unittest.TestCase):
    @patch.object(prediction_service, "predict_urban_models")
    @patch.object(prediction_service, "predict_forest_models")
    @patch.object(prediction_service, "predict_weather_models")
    @patch.object(prediction_service, "predict_drought_models")
    @patch.object(prediction_service, "predict_crop_model")
    @patch.object(prediction_service, "ndvi_proxy")
    @patch.object(prediction_service, "reverse_geocode_state")
    @patch.object(prediction_service, "fetch_environmental_context")
    @patch.object(prediction_service, "geocode")
    def test_unified_predict_returns_domain_grouped_contract(
        self,
        geocode_mock,
        env_mock,
        reverse_mock,
        ndvi_mock,
        crop_mock,
        drought_mock,
        weather_mock,
        forest_mock,
        urban_mock,
    ):
        geocode_mock.return_value = (18.52, 73.85)
        reverse_mock.return_value = "Maharashtra"
        ndvi_mock.return_value = 0.42
        env_mock.return_value = {
            "current": {"temperature": 31.2, "windspeed": 14.3, "weathercode": 1, "precipitation": 12.0},
            "aggregate": {
                "avg_temp": 30.0,
                "max_temp": 34.0,
                "min_temp": 25.0,
                "precip_week": 45.0,
                "precip_day": 12.0,
                "wind_kmh": 14.3,
                "wind_ms": 4.0,
                "wind_bin": 4.0,
                "humidity": 62.0,
                "solar": 18.0,
                "sunshine_hours": 6.8,
                "rainfall_annualized": 2346.4,
            },
            "raw": {},
        }
        crop_mock.return_value = {
            "recommended_crop": "Rice",
            "confidence": 88.0,
            "top_predictions": [{"crop": "Rice", "confidence": 88.0}],
            "season": "Kharif",
            "state": "Maharashtra",
        }
        drought_mock.return_value = {
            "estimated_spei": -0.8,
            "primary_category": "Moderately Dry",
            "top_categories": [{"category": "Moderately Dry", "confidence": 77.0}],
            "drought_active": True,
            "drought_status_code": 1,
            "drought_status_confidence": 81.0,
            "alert_level": "YELLOW",
        }
        weather_mock.return_value = {
            "primary_event": "Flood",
            "primary_event_confidence": 72.0,
            "top_events": [{"event": "Flood", "confidence": 72.0}],
            "intensity": 6.2,
            "intensity_label": "YELLOW",
            "season": "Monsoon",
            "mei_index": 0.4,
            "anomalies": {"temperature_anomaly_c": 2.0, "precipitation_anomaly_mm": 30.0, "wind_anomaly_kmph": 5.0},
        }
        forest_mock.return_value = {
            "matched_state": "Maharashtra",
            "deforestation_alert_code": 1,
            "deforestation_alert_label": "Mild Deforestation",
            "deforestation_alert_confidence": 74.0,
            "future_ndvi": 0.39,
            "future_forest_cover_sqkm": 62000.0,
            "aqi_impact_score": 88.0,
            "human_impact_score": 41.0,
            "effects_report": {"air_quality": {"category": "Moderate"}},
        }
        urban_mock.return_value = {
            "future_population_millions": 1500.0,
            "future_urbanization_rate": 41.5,
            "future_urban_population_millions": 622.5,
            "future_infrastructure_pressure_score": 79.0,
            "future_growth_rate": 1.12,
            "infrastructure_needs": {"pressure_level": "HIGH", "new_schools": 1200},
            "effects_report": {"economic": ["Growth"]},
            "location_specific_rates": {"city_urb_rate": None, "state_urb_rate": 45.0},
        }

        result = prediction_service.unified_predict(location="Pune")

        self.assertIn("domains", result)
        self.assertIn("reports", result)
        self.assertIn("automation", result)
        self.assertIn("crop_intelligence", result["domains"])
        self.assertIn("drought_monitoring", result["domains"])
        self.assertIn("weather_disaster", result["domains"])
        self.assertIn("forest_health", result["domains"])
        self.assertIn("urban_growth", result["domains"])
        self.assertEqual(result["domains"]["crop_intelligence"]["best_crop"]["recommended_crop"], "Rice")
        self.assertEqual(result["domains"]["forest_health"]["alert"]["deforestation_alert_label"], "Mild Deforestation")
        self.assertEqual(result["domains"]["urban_growth"]["population"]["future_population_millions"], 1500.0)
        self.assertEqual(result["reports"]["infrastructure_needs"]["pressure_level"], "HIGH")
        self.assertEqual(result["ensemble"]["overall_alert_level"], "YELLOW")
        self.assertEqual(result["automation"]["manual_input_required"], 0)


if __name__ == "__main__":
    unittest.main()
