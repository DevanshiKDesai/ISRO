from typing import Any

from fastapi import APIRouter, Query, Response

from models.schemas import AlertEmailRequest
from services.export_service import dict_to_csv
from services.model_registry import registry
from services.prediction_service import unified_predict
from utils.notifications import send_alert_email


router = APIRouter(prefix="/tool")


@router.post("/population")
def get_population(payload: dict[str, Any], export: str = Query(default="json")) -> Any:
    location = payload.get("location") or payload.get("state")
    if not location:
        return {"error": "Location or state is required."}
    try:
        prediction = unified_predict(location=location)
        result = prediction
        if export.lower() == "csv":
            return Response(content=dict_to_csv(result), media_type="text/csv")
        return result
    except Exception as exc:
        return {"result": f"Population prediction failed: {exc}"}


@router.post("/forest")
def get_forest(payload: dict[str, Any], export: str = Query(default="json")) -> Any:
    location = payload.get("location") or payload.get("state")
    if not location:
        return {"error": "Location or state is required."}
    try:
        prediction = unified_predict(location=location)
        result = prediction
        if export.lower() == "csv":
            return Response(content=dict_to_csv(result), media_type="text/csv")
        return result
    except Exception as exc:
        return {"result": f"Forest prediction failed: {exc}"}


@router.post("/crop")
def get_crop(payload: dict[str, Any], export: str = Query(default="json")) -> Any:
    location = payload.get("location") or payload.get("state")
    if not location:
        return {"error": "Location or state is required."}
    try:
        prediction = unified_predict(location=location)
        result = prediction
        if export.lower() == "csv":
            return Response(content=dict_to_csv(result), media_type="text/csv")
        return result
    except Exception as exc:
        return {"result": f"Crop prediction failed: {exc}"}


@router.post("/drought")
def get_drought(payload: dict[str, Any], export: str = Query(default="json")) -> Any:
    location = payload.get("location") or payload.get("state")
    if not location:
        return {"error": "Location or state is required."}
    try:
        prediction = unified_predict(location=location)
        result = prediction
        if export.lower() == "csv":
            return Response(content=dict_to_csv(result), media_type="text/csv")
        return result
    except Exception as exc:
        return {"result": f"Drought prediction failed: {exc}"}


@router.post("/disaster")
def predict_disaster(payload: dict[str, Any], export: str = Query(default="json")) -> Any:
    location = payload.get("location")
    lat = payload.get("lat")
    lon = payload.get("lon")
    
    if not location and (lat is None or lon is None):
        return {"error": "Location or coordinates are required."}
        
    try:
        prediction = unified_predict(
            location=location,
            lat=lat,
            lon=lon,
            temp=payload.get("temp"),
            wind=payload.get("wind"),
            precip=payload.get("precip"),
        )
        if export.lower() == "csv":
            return Response(content=dict_to_csv(prediction), media_type="text/csv")
        return prediction
    except Exception as exc:
        return {"error": f"Failed: {exc}", "location": location}


@router.post("/alert/email")
def email_alert(req: AlertEmailRequest) -> dict[str, Any]:
    ok = send_alert_email(req.to_email, req.aoi_name, req.level, req.summary, {"lat": req.lat, "lng": req.lng})
    return {
        "success": ok,
        "message": f"Alert sent to {req.to_email}" if ok else "Email failed. Set SMTP_USER + SMTP_PASSWORD in Space secrets.",
    }
