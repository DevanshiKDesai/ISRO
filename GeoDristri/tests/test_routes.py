import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402


MOCK_PREDICTION = {
    "location": "AOI-10:30",
    "coordinates": {"lat": 18.52, "lon": 73.85},
    "state": "Maharashtra",
    "inputs": {"temperature": 31.0, "wind_speed": 12.0, "precipitation": 10.0},
    "indices": {"ndvi": 0.44, "ndwi": -0.1, "ndbi": 0.08},
    "domains": {
        "crop_intelligence": {"best_crop": {"recommended_crop": "Rice", "confidence": 88.0}},
        "drought_monitoring": {
            "category": {"primary_category": "Moderately Dry", "estimated_spei": -0.8, "top_categories": []},
            "status": {"drought_active": True, "confidence": 81.0},
        },
        "weather_disaster": {
            "event": {"primary_event": "Flood", "confidence": 72.0, "top_events": []},
            "intensity": {"intensity": 6.2, "label": "YELLOW"},
        },
        "forest_health": {
            "alert": {"deforestation_alert_label": "Mild Deforestation", "confidence": 74.0},
            "future_ndvi": {"future_ndvi": 0.39},
            "future_cover": {"future_forest_cover_sqkm": 62000.0},
            "aqi_impact": {"aqi_impact_score": 88.0},
            "human_impact": {"human_impact_score": 41.0, "effects_report": {"air_quality": {"category": "Moderate"}}},
        },
        "urban_growth": {
            "population": {"future_population_millions": 1500.0},
            "urbanization": {"future_urbanization_rate": 41.5},
            "infrastructure": {"future_infrastructure_pressure_score": 79.0, "infrastructure_needs": {"pressure_level": "HIGH"}},
            "urban_population": {"future_urban_population_millions": 622.5},
            "growth": {"future_growth_rate": 1.12},
        },
    },
    "predictions": {
        "crop_yield_predictor": {"recommended_crop": "Rice", "confidence": 88.0},
        "drought_category_model": {"primary_category": "Moderately Dry", "estimated_spei": -0.8, "top_categories": []},
        "drought_status_model": {"drought_active": True, "confidence": 81.0},
        "weather_event_model": {"primary_event": "Flood", "confidence": 72.0, "top_events": []},
        "weather_intensity_model": {"intensity": 6.2, "label": "YELLOW"},
        "forest_alert_model": {"deforestation_alert_label": "Mild Deforestation", "confidence": 74.0},
        "forest_ndvi_model": {"future_ndvi": 0.39},
        "forest_cover_model": {"future_forest_cover_sqkm": 62000.0},
        "forest_aqi_model": {"aqi_impact_score": 88.0},
        "forest_human_model": {"human_impact_score": 41.0, "effects_report": {"air_quality": {"category": "Moderate"}}},
        "urban_pop_model": {"future_population_millions": 1500.0},
        "urban_urb_model": {"future_urbanization_rate": 41.5},
        "urban_infra_model": {"future_infrastructure_pressure_score": 79.0, "infrastructure_needs": {"pressure_level": "HIGH"}},
        "urban_upop_model": {"future_urban_population_millions": 622.5},
        "urban_grow_model": {"future_growth_rate": 1.12},
    },
    "reports": {
        "effects_report": {"forest": {"air_quality": {"category": "Moderate"}}, "urban": {"economic": ["Growth"]}},
        "infrastructure_needs": {"pressure_level": "HIGH"},
    },
    "ensemble": {"overall_alert_level": "YELLOW", "signals": ["crop=Rice"]},
    "automation": {"manual_input_required": 0, "models_run": ["crop_yield_predictor"]},
    "weather_snapshot": {"temperature": 31.0, "windspeed": 12.0, "precipitation": 10.0},
    "errors": [],
}


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("routes.aoi.unified_predict", return_value=MOCK_PREDICTION)
    @patch("routes.aoi.send_alert_email", return_value=True)
    def test_aoi_analyze_returns_grouped_domains(self, _email_mock, _predict_mock):
        response = self.client.post(
            "/aoi/analyze",
            json={
                "lat": 18.52,
                "lng": 73.85,
                "north": 18.57,
                "south": 18.47,
                "east": 73.90,
                "west": 73.80,
                "aoi_name": "Pune AOI",
                "user_email": "demo@example.com",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["alert_level"], "YELLOW")
        self.assertIn("domains", body)
        self.assertIn("reports", body)
        self.assertEqual(body["domains"]["crop_intelligence"]["best_crop"]["recommended_crop"], "Rice")

    @patch("routes.tools.unified_predict", return_value=MOCK_PREDICTION)
    def test_tool_disaster_returns_full_prediction(self, _predict_mock):
        response = self.client.post("/tool/disaster", json={"location": "Pune"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("domains", body)
        self.assertIn("predictions", body)
        self.assertEqual(body["predictions"]["weather_event_model"]["primary_event"], "Flood")

    @patch("routes.chat.classify", return_value=["weather"])
    @patch("routes.chat.DISPATCH", {"weather": lambda _msg: "Weather reply"})
    def test_chat_endpoint_returns_reply(self, _classify_mock):
        response = self.client.post("/chat", json={"message": "weather in Pune"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "weather")
        self.assertEqual(body["reply"], "Weather reply")


if __name__ == "__main__":
    unittest.main()
