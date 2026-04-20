from typing import Any

from fastapi import APIRouter, Query, Response

from models.schemas import AOIRequest
from services.export_service import dict_to_csv
from services.prediction_service import unified_predict
from utils.notifications import send_alert_email


router = APIRouter(prefix="/aoi")


@router.post("/analyze")
def analyze_aoi(req: AOIRequest, export: str = Query(default="json")) -> Any:
    try:
        prediction = unified_predict(location=req.aoi_name, lat=req.lat, lon=req.lng)
    except Exception as exc:
        return {"error": f"AOI analysis failed: {exc}"}

    crop_result = prediction["predictions"]["crop_yield_predictor"]
    drought_result = prediction["predictions"]["drought_category_model"]
    drought_status = prediction["predictions"]["drought_status_model"]
    forest_result = prediction["predictions"]["forest_alert_model"]
    forest_human = prediction["predictions"]["forest_human_model"]
    urban_infra = prediction["predictions"]["urban_infra_model"]
    level = prediction["ensemble"]["overall_alert_level"]

    summary_parts = []
    if "error" not in crop_result:
        summary_parts.append(f"Crop recommendation is {crop_result['recommended_crop']}.")
    if "error" not in drought_result:
        summary_parts.append(
            f"Drought outlook is {drought_result['primary_category']} and drought is "
            f"{'active' if drought_status.get('drought_active') else 'not active'}."
        )
    if "error" not in forest_result:
        summary_parts.append(f"Forest outlook is {forest_result['deforestation_alert_label']}.")
    if "error" not in urban_infra:
        summary_parts.append(
            f"Urban infrastructure pressure is {urban_infra['future_infrastructure_pressure_score']}."
        )
    if not summary_parts:
        summary_parts.append("AOI analyzed with limited ML coverage.")
    summary = " ".join(summary_parts)

    email_sent = False
    if req.user_email and level != "GREEN":
        email_sent = send_alert_email(req.user_email, req.aoi_name, level, summary, {"lat": req.lat, "lng": req.lng})

    payload = {
        "lat": req.lat,
        "lng": req.lng,
        "aoi_name": req.aoi_name,
        "indices": prediction["indices"],
        "temperature": prediction["inputs"]["temperature"],
        "wind_speed": prediction["inputs"]["wind_speed"],
        "precipitation": prediction["inputs"]["precipitation"],
        "is_water": prediction.get("is_water", False),
        "alert_level": level,
        "summary": summary,
        "domains": prediction["domains"],
        "reports": prediction["reports"],
        "crop_prediction": crop_result,
        "drought_prediction": {"category": drought_result, "status": drought_status},
        "weather_prediction": prediction["domains"]["weather_disaster"],
        "forest_prediction": prediction["domains"]["forest_health"],
        "urban_prediction": prediction["domains"]["urban_growth"],
        "ml_prediction": prediction,
        "email_sent": email_sent,
        "errors": prediction["errors"],
    }
    if export.lower() == "csv":
        return Response(content=dict_to_csv(payload), media_type="text/csv")
    return payload
