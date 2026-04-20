from datetime import datetime
from typing import Any, Optional
import difflib
import numpy as np
import requests

from services.model_registry import registry
from utils.alerts import (
    alert_score,
    current_crop_season,
    current_weather_season,
    drought_alert_level,
    encode_spatiotemporal,
    estimate_spei,
    severity_from_intensity,
)
from utils.external_apis import (
    geocode,
    get_altitude,
    get_mei_index,
    get_soil_data,
    live_weather,
    ndvi_proxy,
    reverse_geocode_state,
)


def apply_geographic_rules(probs: np.ndarray, state: str, season: str, anomalies: dict[str, float]) -> np.ndarray:
    if registry.weather_event_encoder is None:
        return probs
    classes = list(registry.weather_event_encoder.classes_)
    adjusted = probs.copy()

    def boost(event_name: str, factor: float) -> None:
        if event_name in classes:
            adjusted[classes.index(event_name)] *= factor

    precip = anomalies["precipitation_anomaly_mm"]
    wind = anomalies["wind_anomaly_kmph"]
    temp = anomalies["temperature_anomaly_c"]

    if state in registry.coastal_states and season in {"Monsoon", "Post-Monsoon"} and wind > 10:
        boost("Cyclone", 2.5)
    if state in registry.hilly_states and precip > 50:
        boost("Landslide", 2.2)
    if state in registry.dry_states and season == "Summer":
        if temp > 3:
            boost("Heatwave", 2.0)
        if precip < -30:
            boost("Drought", 2.0)
    if season == "Monsoon" and precip > 80:
        boost("Flood", 1.8)
        boost("Cloudburst", 1.6)
    if wind > 15 and precip > 30:
        boost("Thunderstorm", 1.7)

    total = adjusted.sum()
    return adjusted / total if total > 0 else adjusted


def fetch_environmental_context(
    lat: float,
    lon: float,
    temp: Optional[float] = None,
    wind: Optional[float] = None,
    precip: Optional[float] = None,
) -> dict[str, Any]:
    weather = live_weather(lat, lon)
    current = weather.get("current_weather", {})
    daily = weather.get("daily", {})
    hourly = weather.get("hourly", {})

    def values(name: str) -> list[float]:
        return [v for v in daily.get(name, []) if v is not None]

    temp_max = values("temperature_2m_max")
    temp_min = values("temperature_2m_min")
    precip_vals = values("precipitation_sum")
    wind_vals = values("windspeed_10m_max")
    solar_vals = values("shortwave_radiation_sum")
    sunshine_vals = values("sunshine_duration")
    humidity_vals = [v for v in hourly.get("relativehumidity_2m", []) if v is not None]

    avg_temp = round(float(np.mean([(mx + mn) / 2 for mx, mn in zip(temp_max, temp_min)])), 3) if temp_max and temp_min else None
    max_temp = round(float(np.mean(temp_max)), 3) if temp_max else None
    min_temp = round(float(np.mean(temp_min)), 3) if temp_min else None
    precip_week = round(float(sum(precip_vals)), 3) if precip_vals else 0.0
    precip_day = round(float(precip_vals[0]), 3) if precip_vals else 0.0
    wind_kmh_avg = round(float(np.mean(wind_vals)), 3) if wind_vals else None
    humidity = round(float(np.mean(humidity_vals)), 3) if humidity_vals else 55.0
    solar = round(float(np.mean(solar_vals)), 3) if solar_vals else 18.0
    sunshine_hours = round(float(np.mean(sunshine_vals)) / 3600, 2) if sunshine_vals else 7.0

    current_temp = float(temp) if temp is not None else float(current.get("temperature", avg_temp or 28.0))
    current_wind = float(wind) if wind is not None else float(current.get("windspeed", wind_kmh_avg or 12.0))
    current_precip = float(precip) if precip is not None else precip_day
    aggregate_precip = float(precip) if precip is not None else precip_week
    aggregate_avg_temp = float(temp) if temp is not None else float(avg_temp if avg_temp is not None else current_temp)
    aggregate_max_temp = float(temp) if temp is not None else float(max_temp if max_temp is not None else current_temp)
    aggregate_min_temp = float(temp) if temp is not None else float(min_temp if min_temp is not None else current_temp)
    aggregate_wind_kmh = float(wind) if wind is not None else float(wind_kmh_avg if wind_kmh_avg is not None else current_wind)

    return {
        "raw": weather,
        "current": {
            "temperature": round(current_temp, 2),
            "windspeed": round(current_wind, 2),
            "weathercode": current.get("weathercode"),
            "precipitation": round(current_precip, 2),
        },
        "aggregate": {
            "avg_temp": round(aggregate_avg_temp, 3),
            "max_temp": round(aggregate_max_temp, 3),
            "min_temp": round(aggregate_min_temp, 3),
            "precip_week": round(aggregate_precip, 3),
            "precip_day": round(current_precip, 3),
            "wind_kmh": round(aggregate_wind_kmh, 3),
            "wind_ms": round(aggregate_wind_kmh / 3.6, 3),
            "wind_bin": round((aggregate_wind_kmh / 3.6) / 0.5) * 0.5,
            "humidity": round(humidity, 3),
            "solar": round(solar, 3),
            "sunshine_hours": round(sunshine_hours, 2),
            "rainfall_annualized": round(aggregate_precip * (365 / 7), 2),
        },
    }


def predict_crop_model(state: str, month: int, env: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    if registry.crop_model is None or registry.crop_target_encoder is None or not registry.crop_feature_cols:
        raise RuntimeError("Crop model assets not loaded")

    soil = get_soil_data(lat, lon)
    altitude = get_altitude(lat, lon)
    season = current_crop_season(month)
    features: dict[str, Any] = {
        "State_Name": state,
        "Season": season,
        "N": 68,
        "P": 53,
        "K": 78,
        "rainfall": env["aggregate"]["rainfall_annualized"],
        "humidity": env["aggregate"]["humidity"],
        "temperature": env["aggregate"]["avg_temp"],
        "pH": soil["pH"],
        "Soil_Type": "Neutral",
        "Irrigation_Method": "Rainfed",
        "Fertilizer_Used_kg": 120.0,
        "Pesticide_Usage_kg": 10.0,
        "Soil_Moisture": soil["Soil_Moisture"],
        "Sunshine_hours": env["aggregate"]["sunshine_hours"],
        "Wind_speed": env["aggregate"]["wind_kmh"],
        "Altitude_m": altitude,
        "Organic_Carbon": soil["Organic_Carbon"],
        "Soil_Texture": "Loamy",
    }
    for col in ("State_Name", "Season", "Soil_Type", "Irrigation_Method", "Soil_Texture"):
        encoder = registry.crop_encoders.get(col)
        if encoder is not None:
            raw = features.get(col, encoder.classes_[0])
            features[col] = int(encoder.transform([raw])[0]) if raw in encoder.classes_ else 0

    matrix = np.array([[features.get(col, 0) for col in registry.crop_feature_cols]])
    probs = registry.crop_model.predict_proba(matrix)[0] if hasattr(registry.crop_model, "predict_proba") else None
    if probs is None:
        label = str(registry.crop_target_encoder.inverse_transform([int(registry.crop_model.predict(matrix)[0])])[0])
        return {"recommended_crop": label, "confidence": None, "top_predictions": [{"crop": label, "confidence": None}]}
    top_idx = np.argsort(probs)[::-1][:5]
    top_predictions = [
        {
            "crop": str(registry.crop_target_encoder.inverse_transform([idx])[0]),
            "confidence": round(float(probs[idx]) * 100, 1),
        }
        for idx in top_idx
    ]
    return {
        "recommended_crop": top_predictions[0]["crop"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "season": season,
        "state": state,
    }


def predict_drought_models(lat: float, lon: float, month: int, env: dict[str, Any]) -> dict[str, Any]:
    if registry.drought_category_model is None or registry.drought_status_model is None or not registry.drought_feature_cols:
        raise RuntimeError("Drought model assets not loaded")

    spei = estimate_spei(env["aggregate"]["precip_week"], env["aggregate"]["avg_temp"], env["aggregate"]["humidity"])
    st = encode_spatiotemporal(lat, lon, month)
    season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]
    feat_map = {
        "Relative Humidity (%)": env["aggregate"]["humidity"],
        "Max Temp (°C)": env["aggregate"]["max_temp"],
        "Min Temp (°C)": env["aggregate"]["min_temp"],
        "Wind Speed (m/s)": env["aggregate"]["wind_ms"],
        "Avg Temperature (°C)": env["aggregate"]["avg_temp"],
        "Solar Radiation": env["aggregate"]["solar"],
        "Precipitation (mm)": env["aggregate"]["precip_week"],
        "Drought Index (SPEI)": spei,
        "lat_sin": st["lat_sin"],
        "lat_cos": st["lat_cos"],
        "lon_sin": st["lon_sin"],
        "lon_cos": st["lon_cos"],
        "month_sin": st["month_sin"],
        "month_cos": st["month_cos"],
        "month": month,
        "season": season,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "temp_range": round(env["aggregate"]["max_temp"] - env["aggregate"]["min_temp"], 3),
        "heat_stress": round(env["aggregate"]["max_temp"] * (1 - env["aggregate"]["humidity"] / 100), 3),
        "precip_humidity": round(env["aggregate"]["precip_week"] * env["aggregate"]["humidity"], 3),
        "aridity_index": round(env["aggregate"]["solar"] / (env["aggregate"]["precip_week"] + 1), 3),
        "wind_evap": round(env["aggregate"]["wind_ms"] * env["aggregate"]["max_temp"], 3),
        "Wind Speed (m/s) (bins)": env["aggregate"]["wind_bin"],
    }
    matrix = np.array([[feat_map[col] for col in registry.drought_feature_cols]])
    category_probs = registry.drought_category_model.predict_proba(matrix)[0]
    top_idx = np.argsort(category_probs)[::-1][:3]
    top_categories = [
        {
            "category": str(registry.drought_category_encoder.inverse_transform([idx])[0]) if registry.drought_category_encoder is not None else str(idx),
            "confidence": round(float(category_probs[idx]) * 100, 1),
        }
        for idx in top_idx
    ]
    status_value = int(registry.drought_status_model.predict(matrix)[0])
    status_probs = registry.drought_status_model.predict_proba(matrix)[0] if hasattr(registry.drought_status_model, "predict_proba") else None
    status_confidence = round(float(status_probs[status_value]) * 100, 1) if status_probs is not None else None
    return {
        "estimated_spei": spei,
        "primary_category": top_categories[0]["category"],
        "top_categories": top_categories,
        "drought_active": bool(status_value),
        "drought_status_code": status_value,
        "drought_status_confidence": status_confidence,
        "alert_level": drought_alert_level(top_categories[0]["category"], bool(status_value)),
    }


def predict_weather_models(state: str, month: int, env: dict[str, Any]) -> dict[str, Any]:
    if any(
        item is None
        for item in (
            registry.weather_event_model,
            registry.weather_intensity_model,
            registry.weather_state_encoder,
            registry.weather_season_encoder,
            registry.weather_event_encoder,
        )
    ) or not registry.weather_feature_cols:
        raise RuntimeError("Weather model assets not loaded")

    season = current_weather_season(month)
    anomalies = {
        "temperature_anomaly_c": round(env["aggregate"]["avg_temp"] - 25.0, 2),
        "precipitation_anomaly_mm": round(env["aggregate"]["precip_week"] - 100.0, 2),
        "wind_anomaly_kmph": round(env["aggregate"]["wind_kmh"] - 20.0, 2),
    }
    state_enc = int(registry.weather_state_encoder.transform([state])[0]) if state in registry.weather_state_encoder.classes_ else 0
    season_enc = int(registry.weather_season_encoder.transform([season])[0]) if season in registry.weather_season_encoder.classes_ else 0
    mei = get_mei_index()
    feature_map = {
        "state_enc": state_enc,
        "month": month,
        "season_enc": season_enc,
        "is_coastal": int(state in registry.coastal_states),
        "is_hilly": int(state in registry.hilly_states),
        "is_dry": int(state in registry.dry_states),
        "precipitation_anomaly_mm": anomalies["precipitation_anomaly_mm"],
        "mei_index": mei,
        "temperature_anomaly_c": anomalies["temperature_anomaly_c"],
        "wind_anomaly_kmph": anomalies["wind_anomaly_kmph"],
        "duration_days": 5,
        "precip_wind": round(anomalies["precipitation_anomaly_mm"] * anomalies["wind_anomaly_kmph"], 3),
        "temp_mei": round(anomalies["temperature_anomaly_c"] * mei, 3),
        "precip_temp": round(anomalies["precipitation_anomaly_mm"] * anomalies["temperature_anomaly_c"], 3),
        "wind_duration": round(anomalies["wind_anomaly_kmph"] * 5, 3),
        "precip_sq": round(anomalies["precipitation_anomaly_mm"] ** 2, 3),
        "wind_sq": round(anomalies["wind_anomaly_kmph"] ** 2, 3),
    }
    event_matrix = np.array([[feature_map[col] for col in registry.weather_feature_cols]])
    raw_probs = registry.weather_event_model.predict_proba(event_matrix)[0]
    adjusted = apply_geographic_rules(raw_probs, state, season, anomalies)
    top_idx = np.argsort(adjusted)[::-1][:3]
    top_events = [
        {
            "event": str(registry.weather_event_encoder.inverse_transform([idx])[0]),
            "confidence": round(float(adjusted[idx]) * 100, 1),
        }
        for idx in top_idx
    ]
    intensity_cols = registry.weather_intensity_feature_cols or registry.weather_feature_cols
    intensity_matrix = np.array([[feature_map.get(col, 0) for col in intensity_cols]])
    intensity = round(float(np.clip(registry.weather_intensity_model.predict(intensity_matrix)[0], 1, 10)), 1)
    severity = severity_from_intensity(intensity)
    return {
        "primary_event": top_events[0]["event"],
        "primary_event_confidence": top_events[0]["confidence"],
        "top_events": top_events,
        "intensity": intensity,
        "intensity_label": severity,
        "season": season,
        "mei_index": mei,
        "anomalies": anomalies,
    }


def calculate_indices(precip: float, ndvi: Optional[float], temp: float, wind: float) -> dict[str, Optional[float]]:
    ndwi = round(max(-1, min(1, (precip / 200) - 0.3)), 3)
    ndbi = round(max(-1, min(1, 0.4 - (ndvi or 0.3))), 3)
    # Normalized Burn Ratio (NBR) proxy - often used for fire/burn detection
    # Proxying with drought/temp/ndvi indicators since we don't have SWIR bands
    nbr = round(max(-1, min(1, (ndvi or 0.4) - (temp / 100) - (wind / 100))), 3)
    
    # Calculate a normalized risk level (0 to 1)
    base_risk = alert_score(None, temp, wind, precip)
    normalized_risk = round(base_risk / 11, 2) # alert_score max is ~11

    return {
        "ndvi": ndvi, 
        "ndwi": ndwi, 
        "ndbi": ndbi, 
        "nbr": nbr,
        "risk_level": normalized_risk
    }


def match_forest_state(state_name: str) -> str:
    known_states = list(registry.forest_state_data.keys())
    if not known_states:
        raise RuntimeError("Forest state data not loaded")
    for state in known_states:
        if state_name.lower() == state.lower():
            return state
    for state in known_states:
        if state_name.lower() in state.lower() or state.lower() in state_name.lower():
            return state
    aliases = {
        "odisha": "Orissa",
        "jammu and kashmir": "Jammu & Kashmir",
        "andaman and nicobar": "A & N Islands",
        "andaman & nicobar": "A & N Islands",
        "dadra and nagar haveli": "Dadra & Nagar Haveli",
        "daman and diu": "Daman & Diu",
        "pondicherry": "Puducherry",
    }
    lowered = state_name.lower()
    for alias, canonical in aliases.items():
        if alias in lowered and canonical in known_states:
            return canonical
    nearest = difflib.get_close_matches(state_name, known_states, n=1, cutoff=0.4)
    return nearest[0] if nearest else known_states[0]


def get_worldbank_data() -> dict[str, float]:
    indicators = {
        "SP.POP.TOTL": "total_population",
        "SP.URB.TOTL.IN.ZS": "urbanization_rate",
        "SP.DYN.CBRT.IN": "birth_rate",
        "SP.DYN.CDRT.IN": "death_rate",
        "SP.POP.GROW": "growth_rate",
    }
    result: dict[str, float] = {}
    base = "https://api.worldbank.org/v2/country/IN/indicator"
    for code, name in indicators.items():
        try:
            resp = requests.get(f"{base}/{code}", params={"format": "json", "mrv": 3, "per_page": 3}, timeout=10)
            data = resp.json()
            if isinstance(data, list) and len(data) > 1:
                records = [row for row in data[1] if row.get("value") is not None]
                if records:
                    result[name] = float(records[0]["value"])
        except Exception:
            continue
    return result


def get_location_urban_rate(city: str, state: str) -> dict[str, Optional[float]]:
    city_rate = None
    state_rate = None
    for known_city, rate in registry.urban_city_rates.items():
        if known_city.lower() in city.lower() or city.lower() in known_city.lower():
            city_rate = rate
            break
    for known_state, rate in registry.urban_state_rates.items():
        if known_state.lower() in state.lower() or state.lower() in known_state.lower():
            state_rate = rate
            break
    return {"city_urb_rate": city_rate, "state_urb_rate": state_rate}


def generate_forest_effects_report(alert_level: int, aqi_score: float, human_score: float, forest_pct: float, ndvi: float, future_cover: float, current_cover: float) -> dict[str, Any]:
    cover_change_pct = ((future_cover - current_cover) / (current_cover + 1)) * 100
    health_risks = []
    if ndvi < 0.25:
        health_risks.append("Severe heat stress due to loss of tree cover")
    if ndvi < 0.35:
        health_risks.append("Increased dust and particulate matter")
    if alert_level >= 2:
        health_risks.append("Higher vector-borne disease risk")
    if forest_pct < 20:
        health_risks.append("Inadequate natural carbon sink")
    return {
        "air_quality": {
            "score": round(aqi_score, 1),
            "category": "Very Poor" if aqi_score > 150 else "Poor" if aqi_score > 100 else "Moderate" if aqi_score > 50 else "Good",
        },
        "health": {"score": round(human_score, 1), "risks": health_risks or ["No major health risks detected"]},
        "water": {"cover_change_pct": round(cover_change_pct, 2)},
        "climate": {"local_temp_rise_est": round(max(0, (0.3 - ndvi) * 5), 1)},
        "biodiversity": {"species_risk_level": "Critical" if alert_level == 3 else "High" if alert_level == 2 else "Moderate" if alert_level == 1 else "Low"},
    }


def infrastructure_report(infra_score: float, urban_pop_m: float, future_urban_m: float, urb_rate: float, future_urb: float) -> dict[str, Any]:
    urban_growth = future_urban_m - urban_pop_m
    return {
        "pressure_level": "CRITICAL" if infra_score > 80 else "HIGH" if infra_score > 65 else "MODERATE" if infra_score > 50 else "LOW",
        "infra_score": round(infra_score, 1),
        "new_schools": int(urban_growth * 1000 * 20),
        "new_hospitals": int(urban_growth * 1000),
        "new_road_km": int(urban_growth * 1000 * 2.5),
        "new_homes": int(urban_growth * 1e6 / 4.5),
        "water_demand_ML": round(urban_growth * 135 * 365 / 1e6, 1),
        "urb_rate_change": round(future_urb - urb_rate, 2),
        "urban_growth_M": round(urban_growth, 2),
    }


def urbanization_effects(urb_rate: float, future_urb: float, infra_score: float, growth_rate: float) -> dict[str, Any]:
    effects: dict[str, list[str]] = {
        "economic": [],
        "environment": [],
        "social": [],
        "health": [],
    }
    if future_urb > 50:
        effects["economic"].append("Major GDP boost from urban manufacturing and services")
    if future_urb > 40:
        effects["economic"].append("Rising formal employment opportunities")
    effects["economic"].append(f"Urban economy contribution estimated near {min(85, int(future_urb * 1.8))}% of GDP")

    if future_urb - urb_rate > 2:
        effects["environment"].append("Accelerated green cover loss in peri-urban areas")
    if future_urb > 45:
        effects["environment"].append("Urban heat island effect intensifying")
    if infra_score > 65:
        effects["environment"].append("Increased solid waste generation and landfill pressure")
    effects["environment"].append(f"Estimated carbon footprint rise: +{round((future_urb - urb_rate) * 0.8, 1)}%")

    if infra_score > 70:
        effects["social"].append("Housing shortage risk and slum expansion pressure")
    if infra_score > 60:
        effects["social"].append("Public transport system under strain")
    effects["social"].append("Rural-to-urban migration accelerating")
    if future_urb > 50:
        effects["social"].append("Demographic transition toward smaller urban households")

    if infra_score > 65:
        effects["health"].append("Air quality deterioration risk")
    if infra_score > 70:
        effects["health"].append("Water scarcity risk in dense urban zones")
    effects["health"].append("Healthcare access likely improves for urban residents")
    return effects


def predict_forest_models(state: str, lat: float, lon: float, current_year: int, live_ndvi: Optional[float]) -> dict[str, Any]:
    if any(
        item is None
        for item in (
            registry.forest_alert_model,
            registry.forest_ndvi_model,
            registry.forest_cover_model,
            registry.forest_aqi_model,
            registry.forest_human_model,
            registry.forest_state_encoder,
        )
    ) or not registry.forest_feature_cols:
        raise RuntimeError("Forest model assets not loaded")

    matched_state = match_forest_state(state)
    hist = registry.forest_state_data[matched_state].copy()
    ndvi_to_use = live_ndvi if live_ndvi is not None else hist["NDVI_mean"]
    ndvi_change = (live_ndvi - hist["NDVI_mean"]) if live_ndvi is not None else (hist.get("NDVI_Change_YoY", 0) or 0)
    year_delta = current_year - hist["Year"]
    forest_cover = hist["Forest_Cover_Area_SqKm"]
    forest_pct = hist["Forest_Percentage_Geographical"]
    yoy = hist.get("Forest_Change_YoY", 0) or 0
    extrap_cover = max(0, forest_cover + yoy * year_delta)
    extrap_pct = max(0, forest_pct + (hist.get("Forest_Pct_Change_YoY", 0) or 0) * year_delta)
    feat_map = {
        "Forest_Cover_Area_SqKm": extrap_cover,
        "Very_Dense_Forest_SqKm": hist.get("Very_Dense_Forest_SqKm", 0) or 0,
        "Mod_Dense_Forest_SqKm": hist.get("Mod_Dense_Forest_SqKm", 0) or 0,
        "Open_Forest_SqKm": hist.get("Open_Forest_SqKm", 0) or 0,
        "Total_Forest_Recorded_SqKm": hist.get("Total_Forest_Recorded_SqKm", 0) or 0,
        "Forest_Percentage_Geographical": extrap_pct,
        "Scrub_Area_SqKm": hist.get("Scrub_Area_SqKm", 0) or 0,
        "NDVI_mean": ndvi_to_use,
        "Total_Crop_Area_Ha": hist.get("Total_Crop_Area_Ha", 0) or 0,
        "RICE AREA (1000 ha)": hist.get("RICE AREA (1000 ha)", 0) or 0,
        "WHEAT AREA (1000 ha)": hist.get("WHEAT AREA (1000 ha)", 0) or 0,
        "SORGHUM AREA (1000 ha)": hist.get("SORGHUM AREA (1000 ha)", 0) or 0,
        "MAIZE AREA (1000 ha)": hist.get("MAIZE AREA (1000 ha)", 0) or 0,
        "GROUNDNUT AREA (1000 ha)": hist.get("GROUNDNUT AREA (1000 ha)", 0) or 0,
        "COTTON AREA (1000 ha)": hist.get("COTTON AREA (1000 ha)", 0) or 0,
        "SUGARCANE AREA (1000 ha)": hist.get("SUGARCANE AREA (1000 ha)", 0) or 0,
        "Forest_Change_YoY": yoy,
        "NDVI_Change_YoY": ndvi_change,
        "Forest_Pct_Change_YoY": hist.get("Forest_Pct_Change_YoY", 0) or 0,
        "VeryDense_Change_YoY": hist.get("VeryDense_Change_YoY", 0) or 0,
        "ModDense_Change_YoY": hist.get("ModDense_Change_YoY", 0) or 0,
        "OpenForest_Change_YoY": hist.get("OpenForest_Change_YoY", 0) or 0,
        "Crop_Change_YoY": hist.get("Crop_Change_YoY", 0) or 0,
        "Forest_Cover_Area_SqKm_3yr_avg": hist.get("Forest_Cover_Area_SqKm_3yr_avg", extrap_cover) or extrap_cover,
        "NDVI_mean_3yr_avg": hist.get("NDVI_mean_3yr_avg", ndvi_to_use) or ndvi_to_use,
        "Forest_Percentage_Geographical_3yr_avg": hist.get("Forest_Percentage_Geographical_3yr_avg", extrap_pct) or extrap_pct,
        "Forest_Cover_Area_SqKm_5yr_avg": hist.get("Forest_Cover_Area_SqKm_5yr_avg", extrap_cover) or extrap_cover,
        "NDVI_mean_5yr_avg": hist.get("NDVI_mean_5yr_avg", ndvi_to_use) or ndvi_to_use,
        "Dense_to_Total_Ratio": hist.get("Dense_to_Total_Ratio", 0) or 0,
        "Open_to_Total_Ratio": hist.get("Open_to_Total_Ratio", 0) or 0,
        "Scrub_to_Forest_Ratio": hist.get("Scrub_to_Forest_Ratio", 0) or 0,
        "Crop_to_Forest_Ratio": hist.get("Crop_to_Forest_Ratio", 0) or 0,
        "Cum_Forest_Change": hist.get("Cum_Forest_Change", 0) or 0,
        "Deforestation_Streak": hist.get("Deforestation_Streak", 0) or 0,
        "Streak_Count": hist.get("Streak_Count", 0) or 0,
        "Year": current_year,
        "State_enc": int(registry.forest_state_encoder.transform([matched_state])[0]) if matched_state in registry.forest_state_encoder.classes_ else 0,
    }
    matrix = np.array([[feat_map.get(col, 0) or 0 for col in registry.forest_feature_cols]])
    alert_pred = int(registry.forest_alert_model.predict(matrix)[0])
    alert_probs = registry.forest_alert_model.predict_proba(matrix)[0]
    future_ndvi = round(float(registry.forest_ndvi_model.predict(matrix)[0]), 3)
    future_cover = round(float(registry.forest_cover_model.predict(matrix)[0]), 1)
    aqi_score = round(float(registry.forest_aqi_model.predict(matrix)[0]), 1)
    human_score = round(float(registry.forest_human_model.predict(matrix)[0]), 1)
    label_map = registry.forest_meta.get("alert_labels", {})
    alert_label = label_map.get(str(alert_pred)) or label_map.get(alert_pred) or str(alert_pred)
    return {
        "matched_state": matched_state,
        "deforestation_alert_code": alert_pred,
        "deforestation_alert_label": alert_label,
        "deforestation_alert_confidence": round(float(alert_probs[alert_pred]) * 100, 1),
        "future_ndvi": future_ndvi,
        "future_forest_cover_sqkm": future_cover,
        "aqi_impact_score": aqi_score,
        "human_impact_score": human_score,
        "effects_report": generate_forest_effects_report(
            alert_pred,
            aqi_score,
            human_score,
            feat_map["Forest_Percentage_Geographical"],
            feat_map["NDVI_mean"],
            future_cover,
            feat_map["Forest_Cover_Area_SqKm"],
        ),
    }


def predict_urban_models(city: str, state: str, current_year: int) -> dict[str, Any]:
    if any(
        item is None
        for item in (
            registry.urban_pop_model,
            registry.urban_urb_model,
            registry.urban_infra_model,
            registry.urban_upop_model,
            registry.urban_grow_model,
        )
    ) or not registry.urban_feature_cols:
        raise RuntimeError("Urban model assets not loaded")
    national = registry.urban_national_latest
    if not national:
        raise RuntimeError("Urban metadata not loaded")
    wb = get_worldbank_data()
    loc_urb = get_location_urban_rate(city, state)
    pop = wb.get("total_population", national["India Population (Millions)"] * 1e6)
    pop_m = pop / 1e6 if pop > 1000 else pop
    urb_rate = wb.get("urbanization_rate", national["Urbanization_Rate"])
    birth_rate = wb.get("birth_rate", national["Birth Rate (per 1000)"])
    death_rate = wb.get("death_rate", national["Death Rate (per 1000)"])
    growth_rate = wb.get("growth_rate", national["India Growth Rate (%)"])
    years_extra = current_year - int(national["Year"])
    if years_extra > 0:
        pop_m = pop_m * ((1 + growth_rate / 100) ** years_extra)
        urb_rate = min(55, urb_rate + 0.4 * years_extra)
        birth_rate = max(12, birth_rate - 0.2 * years_extra)
        death_rate = max(6, death_rate - 0.05 * years_extra)
    urban_pop = pop_m * urb_rate / 100
    rural_pop = pop_m - urban_pop
    effective_urb = loc_urb["city_urb_rate"] or loc_urb["state_urb_rate"] or urb_rate
    world_pop_now = national["World Population (Millions)"] * ((1 + national["World Growth Rate (%)"] / 100) ** years_extra)
    world_growth = max(0.9, national["World Growth Rate (%)"] - 0.01 * years_extra)
    infra_pressure = (pop_m / 1500) * 40 + (effective_urb / 60) * 30 + (growth_rate / 2.5) * 20 + (urban_pop / 600) * 10
    feat_map = {
        "Years_Since_1961": current_year - 1961,
        "India Population (Millions)": pop_m,
        "India Growth Rate (%)": growth_rate,
        "Birth Rate (per 1000)": birth_rate,
        "Death Rate (per 1000)": death_rate,
        "World Population (Millions)": world_pop_now,
        "World Growth Rate (%)": world_growth,
        "Pop_Growth_Abs": pop_m * growth_rate / 100,
        "Pop_Rolling_3yr": pop_m * 0.995,
        "Pop_Rolling_5yr": pop_m * 0.988,
        "Growth_Rate_Change": growth_rate - national["India Growth Rate (%)"],
        "Natural_Increase_Rate": birth_rate - death_rate,
        "Birth_Rate_Trend": birth_rate,
        "Death_Rate_Trend": death_rate,
        "World_Pop_Ratio": pop_m / world_pop_now,
        "India_World_Growth_Diff": growth_rate - world_growth,
        "Urbanization_Rate": effective_urb,
        "Urban_Pop_Millions": urban_pop,
        "Rural_Pop_Millions": rural_pop,
        "Urban_Rural_Ratio": urban_pop / (rural_pop + 1),
        "Urb_Rate_Change": effective_urb - national["Urbanization_Rate"],
        "Infra_Pressure_Score": infra_pressure,
        "School_Pressure": (urban_pop / national["Urban_Pop_Millions"]) * 100,
        "Hospital_Pressure": (urban_pop / national["Urban_Pop_Millions"]) * 90,
        "Road_Pressure": (pop_m / national["India Population (Millions)"]) * 100,
    }
    matrix = np.array([[feat_map.get(col, 0) for col in registry.urban_feature_cols]])
    future_population = round(float(registry.urban_pop_model.predict(matrix)[0]), 2)
    future_urbanization = round(float(registry.urban_urb_model.predict(matrix)[0]), 2)
    future_infra = round(float(registry.urban_infra_model.predict(matrix)[0]), 2)
    future_urban_pop = round(float(registry.urban_upop_model.predict(matrix)[0]), 2)
    future_growth = round(float(registry.urban_grow_model.predict(matrix)[0]), 4)
    infra_needs = infrastructure_report(future_infra, feat_map["Urban_Pop_Millions"], future_urban_pop, feat_map["Urbanization_Rate"], future_urbanization)
    return {
        "future_population_millions": future_population,
        "future_urbanization_rate": future_urbanization,
        "future_urban_population_millions": future_urban_pop,
        "future_infrastructure_pressure_score": future_infra,
        "future_growth_rate": future_growth,
        "infrastructure_needs": infra_needs,
        "effects_report": urbanization_effects(feat_map["Urbanization_Rate"], future_urbanization, future_infra, future_growth),
        "location_specific_rates": loc_urb,
    }


def build_ensemble_summary(crop_result: Optional[dict[str, Any]], drought_result: Optional[dict[str, Any]], weather_result: Optional[dict[str, Any]], forest_result: Optional[dict[str, Any]], urban_result: Optional[dict[str, Any]], env: dict[str, Any]) -> dict[str, Any]:
    level = alert_score(None, env["current"]["temperature"], env["current"]["windspeed"], env["current"]["precipitation"])
    if drought_result is not None and drought_result["alert_level"] == "RED":
        level = "RED"
    elif drought_result is not None and drought_result["alert_level"] == "YELLOW" and level == "GREEN":
        level = "YELLOW"
    if weather_result is not None and weather_result["intensity_label"] == "RED":
        level = "RED"
    elif weather_result is not None and weather_result["intensity_label"] == "YELLOW" and level == "GREEN":
        level = "YELLOW"
    signals = []
    if crop_result:
        signals.append(f"crop={crop_result.get('recommended_crop')}")
    if drought_result:
        signals.append(f"drought={drought_result.get('primary_category')}")
    if weather_result:
        signals.append(f"weather={weather_result.get('primary_event')}")
    if forest_result:
        signals.append(f"forest={forest_result.get('deforestation_alert_label')}")
        if forest_result.get("deforestation_alert_code", 0) >= 2:
            level = "RED"
        elif forest_result.get("deforestation_alert_code", 0) >= 1 and level == "GREEN":
            level = "YELLOW"
    if urban_result:
        signals.append(f"urban_infra={urban_result.get('future_infrastructure_pressure_score')}")
        infra_score = urban_result.get("future_infrastructure_pressure_score", 0)
        if infra_score > 80:
            level = "RED"
        elif infra_score > 60 and level == "GREEN":
            level = "YELLOW"
    return {"overall_alert_level": level, "signals": signals}


def unified_predict(
    *,
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    temp: Optional[float] = None,
    wind: Optional[float] = None,
    precip: Optional[float] = None,
) -> dict[str, Any]:
    registry.ensure_loaded()
    # No longer default to Mumbai; use the provided location or None
    resolved_location = location.strip() if location else None
    
    if (lat is None or lon is None) and resolved_location:
        lat, lon = geocode(resolved_location)
        if lat is None or lon is None:
            raise ValueError(f"Cannot locate {resolved_location}.")
    elif lat is None or lon is None:
        # If no coordinates and no location provided, we can't proceed
        raise ValueError("Latitude and Longitude are required for analysis.")

    env = fetch_environmental_context(lat, lon, temp=temp, wind=wind, precip=precip)
    
    # NEW: Validate if location is on land and identify the state accurately
    from utils.external_apis import is_land_area
    is_land = is_land_area(lat, lon)
    state = reverse_geocode_state(lat, lon)
    
    # If Nominatim didn't return a state but it is land, we might be in a place 
    # where the 'state' field is named differently in the response.
    # Our updated reverse_geocode_state handles this by returning addr.get('country') as fallback.
    
    if not is_land or not state:
        return {
            "location": resolved_location or f"{lat}, {lon}",
            "error": "Non-land or unrecognized area detected.",
            "is_water": True, # Ensure this is always true for the error case to trigger the UI card
            "ensemble_summary": {"overall_alert_level": "GREEN", "signals": ["Location identified as water or unrecognized territory."]}
        }
        
    # month and state are now guaranteed if we reached here
    month = datetime.now().month
    predictions: dict[str, Any] = {}
    errors: list[str] = []
    crop_result = drought_result = weather_result = forest_result = urban_result = None

    try:
        crop_result = predict_crop_model(state, month, env, lat, lon)
        predictions["crop_yield_predictor"] = crop_result
    except Exception as exc:
        predictions["crop_yield_predictor"] = {"error": str(exc)}
        errors.append(f"crop_yield_predictor: {exc}")

    try:
        drought_result = predict_drought_models(lat, lon, month, env)
        predictions["drought_category_model"] = {
            "primary_category": drought_result["primary_category"],
            "top_categories": drought_result["top_categories"],
            "estimated_spei": drought_result["estimated_spei"],
            "alert_level": drought_result["alert_level"],
        }
        predictions["drought_status_model"] = {
            "drought_active": drought_result["drought_active"],
            "drought_status_code": drought_result["drought_status_code"],
            "confidence": drought_result["drought_status_confidence"],
        }
    except Exception as exc:
        predictions["drought_category_model"] = {"error": str(exc)}
        predictions["drought_status_model"] = {"error": str(exc)}
        errors.append(f"drought_models: {exc}")

    try:
        weather_result = predict_weather_models(state, month, env)
        predictions["weather_event_model"] = {
            "primary_event": weather_result["primary_event"],
            "confidence": weather_result["primary_event_confidence"],
            "top_events": weather_result["top_events"],
            "season": weather_result["season"],
            "anomalies": weather_result["anomalies"],
        }
        predictions["weather_intensity_model"] = {
            "intensity": weather_result["intensity"],
            "label": weather_result["intensity_label"],
            "mei_index": weather_result["mei_index"],
        }
    except Exception as exc:
        predictions["weather_event_model"] = {"error": str(exc)}
        predictions["weather_intensity_model"] = {"error": str(exc)}
        errors.append(f"weather_models: {exc}")

    try:
        forest_result = predict_forest_models(state, lat, lon, datetime.now().year, ndvi_proxy(lat, lon))
        predictions["forest_alert_model"] = {
            "deforestation_alert_code": forest_result["deforestation_alert_code"],
            "deforestation_alert_label": forest_result["deforestation_alert_label"],
            "confidence": forest_result["deforestation_alert_confidence"],
        }
        predictions["forest_ndvi_model"] = {"future_ndvi": forest_result["future_ndvi"]}
        predictions["forest_cover_model"] = {"future_forest_cover_sqkm": forest_result["future_forest_cover_sqkm"]}
        predictions["forest_aqi_model"] = {"aqi_impact_score": forest_result["aqi_impact_score"]}
        predictions["forest_human_model"] = {
            "human_impact_score": forest_result["human_impact_score"],
            "effects_report": forest_result["effects_report"],
        }
    except Exception as exc:
        predictions["forest_alert_model"] = {"error": str(exc)}
        predictions["forest_ndvi_model"] = {"error": str(exc)}
        predictions["forest_cover_model"] = {"error": str(exc)}
        predictions["forest_aqi_model"] = {"error": str(exc)}
        predictions["forest_human_model"] = {"error": str(exc)}
        errors.append(f"forest_models: {exc}")

    try:
        urban_result = predict_urban_models(resolved_location or f"{lat},{lon}", state, datetime.now().year)
        predictions["urban_pop_model"] = {"future_population_millions": urban_result["future_population_millions"]}
        predictions["urban_urb_model"] = {"future_urbanization_rate": urban_result["future_urbanization_rate"]}
        predictions["urban_infra_model"] = {
            "future_infrastructure_pressure_score": urban_result["future_infrastructure_pressure_score"],
            "infrastructure_needs": urban_result["infrastructure_needs"],
        }
        predictions["urban_upop_model"] = {"future_urban_population_millions": urban_result["future_urban_population_millions"]}
        predictions["urban_grow_model"] = {"future_growth_rate": urban_result["future_growth_rate"]}
    except Exception as exc:
        predictions["urban_pop_model"] = {"error": str(exc)}
        predictions["urban_urb_model"] = {"error": str(exc)}
        predictions["urban_infra_model"] = {"error": str(exc)}
        predictions["urban_upop_model"] = {"error": str(exc)}
        predictions["urban_grow_model"] = {"error": str(exc)}
        errors.append(f"urban_models: {exc}")

    ndvi = ndvi_proxy(lat, lon)
    indices = calculate_indices(env["current"]["precipitation"], ndvi, env["current"]["temperature"], env["current"]["windspeed"])
    ensemble = build_ensemble_summary(crop_result, drought_result, weather_result, forest_result, urban_result, env)
    domain_predictions = {
        "crop_intelligence": {
            "best_crop": predictions.get("crop_yield_predictor", {}),
        },
        "drought_monitoring": {
            "category": predictions.get("drought_category_model", {}),
            "status": predictions.get("drought_status_model", {}),
        },
        "weather_disaster": {
            "event": predictions.get("weather_event_model", {}),
            "intensity": predictions.get("weather_intensity_model", {}),
        },
        "forest_health": {
            "alert": predictions.get("forest_alert_model", {}),
            "future_ndvi": predictions.get("forest_ndvi_model", {}),
            "future_cover": predictions.get("forest_cover_model", {}),
            "aqi_impact": predictions.get("forest_aqi_model", {}),
            "human_impact": predictions.get("forest_human_model", {}),
        },
        "urban_growth": {
            "population": predictions.get("urban_pop_model", {}),
            "urbanization": predictions.get("urban_urb_model", {}),
            "infrastructure": predictions.get("urban_infra_model", {}),
            "urban_population": predictions.get("urban_upop_model", {}),
            "growth": predictions.get("urban_grow_model", {}),
        },
    }
    reports = {
        "effects_report": {
            "forest": forest_result.get("effects_report") if forest_result else None,
            "urban": urban_result.get("effects_report") if urban_result else None,
        },
        "infrastructure_needs": urban_result.get("infrastructure_needs") if urban_result else None,
    }
    automation = {
        "manual_input_required": 0,
        "location_resolved": {"location": resolved_location, "state": state, "lat": lat, "lon": lon},
        "features_auto_fetched": [
            "weather",
            "soil",
            "ndvi_proxy",
            "geocoding",
            "world_bank_population",
            "derived_features",
        ],
        "models_run": [name for name, result in predictions.items() if isinstance(result, dict) and "error" not in result],
    }
    return {
        "location": resolved_location,
        "coordinates": {"lat": lat, "lon": lon},
        "state": state,
        "inputs": {
            "temperature": env["current"]["temperature"],
            "wind_speed": env["current"]["windspeed"],
            "precipitation": env["current"]["precipitation"],
        },
        "indices": indices,
        "domains": domain_predictions,
        "predictions": predictions,
        "reports": reports,
        "ensemble": ensemble,
        "automation": automation,
        "weather_snapshot": env["current"],
        "errors": errors,
    }
