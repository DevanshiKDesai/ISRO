"""
Crop Prediction - Inference Pipeline
=====================================
User enters a location → APIs fill all data → Model predicts best crop.

Usage:
    python predict.py "Pune"
    python predict.py "18.52,73.85"
"""

import requests
import joblib
import numpy as np
import sys
import json
from time import sleep

# ── Load saved model & encoders ──────────────────────────────────────────────
MODEL        = joblib.load("model.joblib")
ENCODERS     = joblib.load("encoders.joblib")
TARGET_LE    = joblib.load("target_encoder.joblib")
FEATURE_COLS = joblib.load("feature_cols.joblib")


# ── STEP 1: Geocoding ─────────────────────────────────────────────────────────
def get_coordinates(location: str) -> tuple[float, float]:
    """Convert city name OR 'lat,lon' string to (lat, lon) floats."""
    location = location.strip()

    # If user typed coordinates directly, parse them
    if "," in location:
        parts = location.split(",")
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            pass  # fall through to geocoding

    # Otherwise, use Nominatim geocoding API (free, no key needed)
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "crop-predictor/1.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise ValueError(f"Location '{location}' not found. Try a different city name.")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    print(f"  Location found: {data[0]['display_name']}")
    print(f"  Coordinates: lat={lat:.4f}, lon={lon:.4f}")
    return lat, lon


# ── STEP 2: Weather & Climate API ────────────────────────────────────────────
def get_weather_data(lat: float, lon: float) -> dict:
    """Fetch temperature, rainfall, humidity, sunshine, wind from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "sunshine_duration",
        ],
        "hourly": "relativehumidity_2m",
        "timezone": "auto",
        "forecast_days": 7,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    temps_max = daily["temperature_2m_max"]
    temps_min = daily["temperature_2m_min"]
    precip     = daily["precipitation_sum"]
    wind       = daily["windspeed_10m_max"]
    sunshine   = daily["sunshine_duration"]  # seconds/day
    humidity_h = data["hourly"]["relativehumidity_2m"]

    # Compute averages, handling None values
    def avg(lst): return round(np.mean([x for x in lst if x is not None]), 2)

    temperature    = avg([(mx + mn) / 2 for mx, mn in zip(temps_max, temps_min)])
    rainfall       = round(sum(x for x in precip if x is not None) * (365 / 7), 2)  # annualise
    humidity       = avg(humidity_h)
    wind_speed     = avg(wind)
    sunshine_hours = round(avg(sunshine) / 3600, 2)  # seconds → hours/day

    return {
        "temperature":    temperature,
        "rainfall":       rainfall,
        "humidity":       humidity,
        "Wind_speed":     wind_speed,
        "Sunshine_hours": sunshine_hours,
    }


# ── STEP 3: Altitude API ──────────────────────────────────────────────────────
def get_altitude(lat: float, lon: float) -> float:
    """Fetch elevation in metres from Open-Elevation API."""
    url = "https://api.open-elevation.com/api/v1/lookup"
    payload = {"locations": [{"latitude": lat, "longitude": lon}]}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        alt = resp.json()["results"][0]["elevation"]
        return float(alt) if alt is not None else 350.0
    except Exception:
        print("  (Altitude API unavailable, using regional average)")
        return 350.0  # fallback to dataset mean


# ── STEP 4: Soil API ─────────────────────────────────────────────────────────
def get_soil_data(lat: float, lon: float) -> dict:
    """Fetch soil properties from SoilGrids REST API."""
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["phh2o", "soc", "wv0010"],
        "depth": "0-30cm",
        "value": "mean",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        layers = resp.json()["properties"]["layers"]

        result = {}
        for layer in layers:
            name = layer["name"]
            val  = layer["depths"][0]["values"]["mean"]
            if val is None:
                continue
            if name == "phh2o":
                result["pH"] = round(val / 10, 2)       # SoilGrids returns pH×10
            elif name == "soc":
                result["Organic_Carbon"] = round(val / 10, 2)  # dg/kg → %
            elif name == "wv0010":
                result["Soil_Moisture"] = round(val / 10, 2)

        # Fill any missing soil values with dataset means
        result.setdefault("pH",             6.8)
        result.setdefault("Organic_Carbon", 0.80)
        result.setdefault("Soil_Moisture",  38.0)
        return result

    except Exception:
        print("  (SoilGrids API unavailable, using regional soil averages)")
        return {"pH": 6.8, "Organic_Carbon": 0.80, "Soil_Moisture": 38.0}


# ── STEP 5: Derive state name from coordinates ───────────────────────────────
def get_state_name(lat: float, lon: float) -> str:
    """Reverse geocode to get Indian state name."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {"User-Agent": "crop-predictor/1.0"}
    try:
        sleep(1)  # Nominatim rate limit: 1 req/sec
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        address = resp.json().get("address", {})
        state = address.get("state", "Maharashtra")
        return state
    except Exception:
        return "Maharashtra"


# ── STEP 6: Determine season from month ──────────────────────────────────────
def get_season() -> str:
    """Map current month to Indian agricultural season."""
    from datetime import datetime
    month = datetime.now().month
    if month in [11, 12, 1, 2, 3]:   return "Rabi"
    elif month in [6, 7, 8, 9, 10]:  return "Kharif"
    elif month in [3, 4, 5]:         return "Summer"
    else:                             return "Kharif"


# ── STEP 7: Encode & predict ─────────────────────────────────────────────────
def encode_and_predict(features: dict) -> list[tuple[str, float]]:
    """Encode features and return top-5 crop predictions with confidence."""

    # Encode categorical columns using saved encoders
    cat_cols = ['State_Name', 'Season', 'Soil_Type', 'Irrigation_Method', 'Soil_Texture']
    for col in cat_cols:
        le = ENCODERS[col]
        val = features.get(col, le.classes_[0])
        # Handle unseen labels gracefully
        if val in le.classes_:
            features[col] = int(le.transform([val])[0])
        else:
            features[col] = 0  # fallback to first class

    # Build feature vector in correct order
    X = np.array([[features.get(col, 0) for col in FEATURE_COLS]])

    # Get probability scores for all 111 crops
    probs = MODEL.predict_proba(X)[0]

    # Top 5 predictions
    top5_idx = np.argsort(probs)[::-1][:5]
    results = []
    for idx in top5_idx:
        crop = TARGET_LE.inverse_transform([idx])[0]
        confidence = round(probs[idx] * 100, 1)
        results.append((crop, confidence))

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────
def predict_crop(location: str):
    print(f"\n{'='*55}")
    print(f"  Crop Prediction System")
    print(f"{'='*55}")
    print(f"\n[1/6] Geocoding: '{location}'...")
    lat, lon = get_coordinates(location)

    print(f"\n[2/6] Fetching weather data...")
    weather = get_weather_data(lat, lon)
    for k, v in weather.items():
        print(f"  {k}: {v}")

    print(f"\n[3/6] Fetching altitude...")
    altitude = get_altitude(lat, lon)
    print(f"  Altitude_m: {altitude}")

    print(f"\n[4/6] Fetching soil data...")
    soil = get_soil_data(lat, lon)
    for k, v in soil.items():
        print(f"  {k}: {v}")

    print(f"\n[5/6] Determining region & season...")
    state  = get_state_name(lat, lon)
    season = get_season()
    print(f"  State: {state}, Season: {season}")

    # Assemble all features
    # N, P, K — use dataset medians (APIs don't provide these; user can override)
    features = {
        "State_Name":        state,
        "Season":            season,
        "N":                 68,       # dataset median
        "P":                 53,       # dataset median
        "K":                 78,       # dataset median
        "Soil_Type":         "Neutral",
        "Irrigation_Method": "Rainfed",
        "Soil_Texture":      "Loamy",
        "Fertilizer_Used_kg": 120.0,
        "Pesticide_Usage_kg": 10.0,
        **weather,
        "Altitude_m": altitude,
        **soil,
    }

    print(f"\n[6/6] Running prediction...")
    predictions = encode_and_predict(features)

    print(f"\n{'='*55}")
    print(f"  PREDICTION RESULTS for {location}")
    print(f"{'='*55}")
    for i, (crop, conf) in enumerate(predictions, 1):
        bar = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
        print(f"  #{i}  {crop:<30} {bar} {conf}%")
    print(f"{'='*55}\n")

    return predictions


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else "Pune"
    predict_crop(location)
