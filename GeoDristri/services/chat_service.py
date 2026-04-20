import re
from typing import Any

from services.model_registry import registry
from services.prediction_service import unified_predict
from utils.constants import INDIAN_PLACES, INTENT_MAP, WMO


def classify(msg: str) -> list[str]:
    text = msg.lower()
    matched = [intent for intent, keywords in INTENT_MAP.items() if any(keyword in text for keyword in keywords)]
    return matched if matched else ["unknown"]


def extract_loc(msg: str) -> str:
    caps = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", msg)
    for candidate in caps:
        if candidate in INDIAN_PLACES:
            return candidate
    lookup = {place.lower(): place for place in INDIAN_PLACES}
    for word in re.findall(r"\b[a-z]{4,}\b", msg.lower()):
        if word in lookup:
            return lookup[word]
    return caps[0] if caps else None


def extract_year(msg: str) -> int:
    years = re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", msg)
    return int(years[0]) if years else 2017


def _full_model_summary(pred: dict[str, Any]) -> list[str]:
    crop = pred["predictions"]["crop_yield_predictor"]
    drought = pred["predictions"]["drought_category_model"]
    drought_status = pred["predictions"]["drought_status_model"]
    weather = pred["predictions"]["weather_event_model"]
    intensity = pred["predictions"]["weather_intensity_model"]
    forest = pred["predictions"]["forest_alert_model"]
    forest_ndvi = pred["predictions"]["forest_ndvi_model"]
    urban_pop = pred["predictions"]["urban_pop_model"]
    urban_urb = pred["predictions"]["urban_urb_model"]
    urban_infra = pred["predictions"]["urban_infra_model"]
    urban_growth = pred["predictions"]["urban_grow_model"]

    lines = [f"Alert: {pred['ensemble']['overall_alert_level']}"]
    if "error" not in crop:
        lines.append(f"Crop model: {crop['recommended_crop']} ({crop['confidence']}%)")
    if "error" not in drought:
        drought_flag = "active" if drought_status.get("drought_active") else "inactive"
        lines.append(f"Drought category model: {drought['primary_category']}")
        lines.append(f"Drought status model: {drought_flag}")
    if "error" not in weather:
        lines.append(f"Weather event model: {weather['primary_event']} ({weather['confidence']}%)")
    if "error" not in intensity:
        lines.append(f"Weather intensity model: {intensity['intensity']}/10 ({intensity['label']})")
    if "error" not in forest:
        lines.append(f"Forest alert model: {forest['deforestation_alert_label']} ({forest['confidence']}%)")
    if "error" not in forest_ndvi:
        lines.append(f"Forest NDVI model: future NDVI {forest_ndvi['future_ndvi']}")
    if "error" not in urban_pop:
        lines.append(f"Urban population model: {urban_pop['future_population_millions']}M")
    if "error" not in urban_urb:
        lines.append(f"Urbanization model: {urban_urb['future_urbanization_rate']}%")
    if "error" not in urban_infra:
        lines.append(f"Urban infra model: score {urban_infra['future_infrastructure_pressure_score']}")
    if "error" not in urban_growth:
        lines.append(f"Urban growth model: {urban_growth['future_growth_rate']}%")
    if pred["errors"]:
        lines.append("Errors: " + "; ".join(pred["errors"]))
    return lines


def chat_weather(msg: str) -> str:
    loc = extract_loc(msg)
    if not loc:
        return "Please specify a location (e.g., 'Weather in Pune')."
    pred = unified_predict(location=loc)
    weather = pred["weather_snapshot"]
    lines = [f"Weather ML for {loc}", f"Current: {weather['temperature']}C, wind {weather['windspeed']} km/h, precip {weather['precipitation']} mm"]
    code = pred.get("weather_snapshot", {}).get("weathercode")
    if code is not None:
        lines.append(f"Condition: {WMO.get(code, f'Code {code}')}")
    lines.extend(_full_model_summary(pred))
    return "\n".join(lines)


def chat_disaster(msg: str) -> str:
    loc = extract_loc(msg)
    if not loc:
        return "Please specify a location (e.g., 'Disaster risk for Chennai')."
    pred = unified_predict(location=loc)
    lines = [f"Multi-model disaster analysis for {loc}"]
    lines.extend(_full_model_summary(pred))
    return "\n".join(lines)


def chat_crop(msg: str) -> str:
    loc = extract_loc(msg)
    if not loc:
        return "Please specify a location (e.g., 'Crop yield in Punjab')."
    pred = unified_predict(location=loc)
    crop = pred["predictions"]["crop_yield_predictor"]
    if "error" in crop:
        return f"Crop model unavailable for {loc}: {crop['error']}"
    top = ", ".join(f"{item['crop']} {item['confidence']}%" for item in crop["top_predictions"])
    lines = [
        f"Crop ML for {loc}",
        f"Recommended crop: {crop['recommended_crop']} ({crop['confidence']}%)",
        f"Top predictions: {top}",
    ]
    lines.extend(_full_model_summary(pred))
    return "\n".join(lines)


def chat_drought(msg: str) -> str:
    loc = extract_loc(msg)
    if not loc:
        return "Please specify a location (e.g., 'Drought analysis for Rajasthan')."
    pred = unified_predict(location=loc)
    drought = pred["predictions"]["drought_category_model"]
    status = pred["predictions"]["drought_status_model"]
    if "error" in drought:
        return f"Drought models unavailable for {loc}: {drought['error']}"
    top = ", ".join(f"{item['category']} {item['confidence']}%" for item in drought["top_categories"])
    lines = [
        f"Drought ML for {loc}",
        f"Category: {drought['primary_category']}",
        f"Active drought: {'Yes' if status['drought_active'] else 'No'}",
        f"Estimated SPEI: {drought['estimated_spei']}",
        f"Top categories: {top}",
    ]
    lines.extend(_full_model_summary(pred))
    return "\n".join(lines)


def chat_forest(msg: str) -> str:
    loc = extract_loc(msg)
    if not loc:
        return "Please specify a location (e.g., 'Forest cover in Maharashtra')."
    pred = unified_predict(location=loc)
    forest = pred["predictions"]["forest_alert_model"]
    forest_ndvi = pred["predictions"]["forest_ndvi_model"]
    forest_cover = pred["predictions"]["forest_cover_model"]
    forest_aqi = pred["predictions"]["forest_aqi_model"]
    forest_human = pred["predictions"]["forest_human_model"]
    if "error" in forest:
        return f"Forest models unavailable for {state}: {forest['error']}"
    lines = [
        f"Forest ML for {state}",
        f"Alert: {forest['deforestation_alert_label']} ({forest['confidence']}%)",
        f"Future NDVI: {forest_ndvi['future_ndvi']}",
        f"Future cover: {forest_cover['future_forest_cover_sqkm']} sq km",
        f"AQI impact: {forest_aqi['aqi_impact_score']}",
        f"Human impact: {forest_human['human_impact_score']}",
    ]
    lines.extend(_full_model_summary(pred))
    return "\n".join(lines)


def chat_population(msg: str) -> str:
    loc = extract_loc(msg)
    if not loc:
        return "Please specify a location (e.g., 'Population growth in Delhi')."
    pred = unified_predict(location=loc)
    urban_pop = pred["predictions"]["urban_pop_model"]
    urban_urb = pred["predictions"]["urban_urb_model"]
    urban_infra = pred["predictions"]["urban_infra_model"]
    urban_upop = pred["predictions"]["urban_upop_model"]
    urban_growth = pred["predictions"]["urban_grow_model"]
    if "error" in urban_pop:
        return f"Urban models unavailable for {loc}: {urban_pop['error']}"
    lines = [
        f"Urban ML for {loc}",
        f"Future population: {urban_pop['future_population_millions']}M",
        f"Future urbanization: {urban_urb['future_urbanization_rate']}%",
        f"Future urban population: {urban_upop['future_urban_population_millions']}M",
        f"Future infra score: {urban_infra['future_infrastructure_pressure_score']}",
        f"Future growth rate: {urban_growth['future_growth_rate']}%",
    ]
    lines.extend(_full_model_summary(pred))
    return "\n".join(lines)


def chat_ndvi(msg: str) -> str:
    loc = extract_loc(msg)
    if not loc:
        return "Please specify a location (e.g., 'NDVI for Kerala')."
    pred = unified_predict(location=loc)
    indices = pred["indices"]
    return f"Indices for {loc}\nNDVI: {indices['ndvi']}\nNDWI: {indices['ndwi']}\nNDBI: {indices['ndbi']}"


def smart_fallback(msg: str) -> str:
    loc = extract_loc(msg)
    parts = [f"Could not classify the request cleanly. Closest location: {loc}", "Available capabilities:"]
    parts.extend(f"- {name}: {'ready' if ok else 'unavailable'}" for name, ok in registry.model_status.items())
    return "\n".join(parts)


DISPATCH = {
    "weather": chat_weather,
    "disaster": chat_disaster,
    "forest": chat_forest,
    "crop": chat_crop,
    "drought": chat_drought,
    "population": chat_population,
    "ndvi": chat_ndvi,
}
