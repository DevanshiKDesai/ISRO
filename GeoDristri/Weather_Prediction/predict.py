"""
Weather Event Prediction - Inference Pipeline
==============================================
User enters a location → APIs fill weather data → Model predicts:
  • Most likely weather event (Flood, Cyclone, Drought, etc.)
  • Intensity score (1–10)
  • Damage estimate (USD millions)

Usage:
    python predict.py "Mumbai"
    python predict.py "18.52,73.85"
"""

import requests
import joblib
import numpy as np
import json
import sys
from datetime import datetime
from time import sleep

# ── Load models & encoders ────────────────────────────────────────────────────
EVENT_MODEL     = joblib.load("event_model.joblib")
INTENSITY_MODEL = joblib.load("intensity_model.joblib")
STATE_LE        = joblib.load("state_encoder.joblib")
SEASON_LE       = joblib.load("season_encoder.joblib")
EVENT_LE        = joblib.load("event_encoder.joblib")
FEATURE_COLS    = joblib.load("feature_cols.joblib")

with open("metadata.json") as f:
    META = json.load(f)

COASTAL_STATES = META["coastal_states"]
HILLY_STATES   = META["hilly_states"]
DRY_STATES     = META["dry_states"]


# ── STEP 1: Geocoding ─────────────────────────────────────────────────────────
def get_coordinates(location: str) -> tuple[float, float, str]:
    """Return (lat, lon, display_name)."""
    location = location.strip()
    if "," in location:
        parts = location.split(",")
        try:
            return float(parts[0]), float(parts[1]), location
        except ValueError:
            pass

    url     = "https://nominatim.openstreetmap.org/search"
    params  = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "weather-predictor/1.0"}
    resp    = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data    = resp.json()

    if not data:
        raise ValueError(f"Location '{location}' not found.")

    return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]


# ── STEP 2: Reverse geocode → Indian state ────────────────────────────────────
def get_state(lat: float, lon: float) -> str:
    """Reverse geocode coordinates to Indian state name."""
    url     = "https://nominatim.openstreetmap.org/reverse"
    params  = {"lat": lat, "lon": lon, "format": "json"}
    headers = {"User-Agent": "weather-predictor/1.0"}
    try:
        sleep(1)
        resp  = requests.get(url, params=params, headers=headers, timeout=10)
        state = resp.json().get("address", {}).get("state", "Maharashtra")
        return state
    except Exception:
        return "Maharashtra"


# ── STEP 3: Weather anomaly data from Open-Meteo ─────────────────────────────
def get_weather_anomalies(lat: float, lon: float) -> dict:
    """
    Fetch current weather and compute anomalies vs historical normals.
    Anomaly = current value - long-term average.
    """
    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
        ],
        "timezone":      "auto",
        "forecast_days": 7,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    def avg(lst): return np.mean([x for x in lst if x is not None])

    temps  = [(mx + mn) / 2 for mx, mn in zip(daily["temperature_2m_max"], daily["temperature_2m_min"])]
    temp   = round(avg(temps), 2)
    precip = round(sum(x for x in daily["precipitation_sum"] if x is not None), 2)
    wind   = round(avg(daily["windspeed_10m_max"]), 2)

    # Historical normals (India averages)
    NORMAL_TEMP   = 25.0   # °C
    NORMAL_PRECIP = 100.0  # mm/week
    NORMAL_WIND   = 20.0   # km/h

    return {
        "temperature_anomaly_c":    round(temp   - NORMAL_TEMP,   2),
        "precipitation_anomaly_mm": round(precip - NORMAL_PRECIP, 2),
        "wind_anomaly_kmph":        round(wind   - NORMAL_WIND,   2),
        "raw_temp":   temp,
        "raw_precip": precip,
        "raw_wind":   wind,
    }


# ── STEP 4: MEI index (El Niño) from NOAA ─────────────────────────────────────
def get_mei_index() -> float:
    """Fetch latest MEI (Multivariate ENSO Index) from NOAA."""
    try:
        url  = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
        resp = requests.get(url, timeout=10)
        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        # Last data line has most recent year's values
        for line in reversed(lines):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                values = [float(v) for v in parts[1:] if v not in ['-999.00', '-999']]
                if values:
                    return round(values[-1], 2)
    except Exception:
        pass
    return 0.0  # neutral MEI as fallback


# ── STEP 5: Season from current month ────────────────────────────────────────
def get_season(month: int) -> str:
    if month in [12, 1, 2]:   return "Winter"
    elif month in [3, 4, 5]:  return "Summer"
    elif month in [6, 7, 8, 9]: return "Monsoon"
    else:                      return "Post-Monsoon"


# ── STEP 6: Rule-based event probability boost ────────────────────────────────
def apply_geographic_rules(probs: np.ndarray, state: str,
                            season: str, anomalies: dict) -> np.ndarray:
    """
    Boost probabilities based on real-world geographic & seasonal knowledge.
    This makes predictions more meaningful beyond what the model alone provides.
    """
    classes   = list(EVENT_LE.classes_)
    probs     = probs.copy()

    precip  = anomalies["precipitation_anomaly_mm"]
    wind    = anomalies["wind_anomaly_kmph"]
    temp    = anomalies["temperature_anomaly_c"]

    def boost(event, factor):
        if event in classes:
            idx = classes.index(event)
            probs[idx] *= factor

    # Coastal + Monsoon/Post-Monsoon + high wind → Cyclone
    if state in COASTAL_STATES and season in ["Monsoon","Post-Monsoon"] and wind > 10:
        boost("Cyclone", 2.5)

    # Hilly state + high precipitation → Landslide
    if state in HILLY_STATES and precip > 50:
        boost("Landslide", 2.2)

    # Dry state + Summer + high temp → Heatwave or Drought
    if state in DRY_STATES and season == "Summer":
        if temp > 3:
            boost("Heatwave", 2.0)
        if precip < -30:
            boost("Drought", 2.0)

    # High precipitation anywhere in Monsoon → Flood or Cloudburst
    if season == "Monsoon" and precip > 80:
        boost("Flood", 1.8)
        boost("Cloudburst", 1.6)

    # High wind + high precip → Thunderstorm
    if wind > 15 and precip > 30:
        boost("Thunderstorm", 1.7)

    # Renormalize to sum to 1
    total = probs.sum()
    if total > 0:
        probs = probs / total

    return probs


# ── STEP 7: Encode features & predict ─────────────────────────────────────────
def encode_and_predict(state: str, month: int, season: str,
                       anomalies: dict, mei: float,
                       duration_est: int = 5) -> dict:
    """Build feature vector and run both models."""

    is_coastal = int(state in COASTAL_STATES)
    is_hilly   = int(state in HILLY_STATES)
    is_dry     = int(state in DRY_STATES)

    # Encode categoricals safely
    state_enc  = int(STATE_LE.transform([state])[0])  if state  in STATE_LE.classes_  else 0
    season_enc = int(SEASON_LE.transform([season])[0]) if season in SEASON_LE.classes_ else 0

    feat_map = {
        "state_enc":                state_enc,
        "month":                    month,
        "season_enc":               season_enc,
        "is_coastal":               is_coastal,
        "is_hilly":                 is_hilly,
        "is_dry":                   is_dry,
        "precipitation_anomaly_mm": anomalies["precipitation_anomaly_mm"],
        "mei_index":                mei,
        "temperature_anomaly_c":    anomalies["temperature_anomaly_c"],
        "wind_anomaly_kmph":        anomalies["wind_anomaly_kmph"],
        "duration_days":            duration_est,
    }

    X = np.array([[feat_map[c] for c in FEATURE_COLS]])

    # Event type probabilities
    raw_probs = EVENT_MODEL.predict_proba(X)[0]
    adj_probs = apply_geographic_rules(raw_probs, state, season, anomalies)

    # Top 3 events
    top3_idx = np.argsort(adj_probs)[::-1][:3]
    top3 = [(EVENT_LE.inverse_transform([i])[0], round(adj_probs[i] * 100, 1))
            for i in top3_idx]

    # Intensity score
    intensity_raw = float(INTENSITY_MODEL.predict(X)[0])
    intensity     = round(np.clip(intensity_raw, 1, 10), 1)

    return {"top_events": top3, "intensity": intensity}


# ── MAIN ──────────────────────────────────────────────────────────────────────
def predict_weather(location: str):
    print(f"\n{'='*55}")
    print(f"  Weather Event Prediction System")
    print(f"{'='*55}")

    print(f"\n[1/5] Geocoding '{location}'...")
    lat, lon, display = get_coordinates(location)
    print(f"  Found: {display}")
    print(f"  Coordinates: {lat:.4f}, {lon:.4f}")

    print(f"\n[2/5] Detecting state...")
    state = get_state(lat, lon)
    print(f"  State: {state}")
    geo_tags = []
    if state in COASTAL_STATES: geo_tags.append("coastal")
    if state in HILLY_STATES:   geo_tags.append("hilly/northeast")
    if state in DRY_STATES:     geo_tags.append("arid/dry")
    if geo_tags: print(f"  Region type: {', '.join(geo_tags)}")

    print(f"\n[3/5] Fetching weather anomaly data...")
    anomalies = get_weather_anomalies(lat, lon)
    print(f"  Temperature anomaly : {anomalies['temperature_anomaly_c']:+.2f}°C")
    print(f"  Precipitation anomaly: {anomalies['precipitation_anomaly_mm']:+.2f} mm")
    print(f"  Wind anomaly         : {anomalies['wind_anomaly_kmph']:+.2f} km/h")

    print(f"\n[4/5] Fetching MEI (El Niño) index...")
    mei = get_mei_index()
    print(f"  MEI index: {mei} ({'El Niño' if mei > 0.5 else 'La Niña' if mei < -0.5 else 'Neutral'})")

    now    = datetime.now()
    month  = now.month
    season = get_season(month)
    print(f"  Current season: {season} (month {month})")

    print(f"\n[5/5] Running prediction models...")
    result = encode_and_predict(state, month, season, anomalies, mei)

    # Intensity label
    intensity = result["intensity"]
    if intensity <= 3:   severity = "Low"
    elif intensity <= 6: severity = "Moderate"
    elif intensity <= 8: severity = "High"
    else:                severity = "Extreme"

    print(f"\n{'='*55}")
    print(f"  WEATHER EVENT PREDICTIONS — {location.upper()}")
    print(f"{'='*55}")
    print(f"\n  Most likely weather events:")
    for i, (event, prob) in enumerate(result["top_events"], 1):
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"  #{i}  {event:<15} {bar} {prob}%")

    print(f"\n  Intensity score : {intensity}/10  ({severity})")
    print(f"  Season          : {season}")
    print(f"  MEI (El Niño)   : {mei}")
    print(f"{'='*55}\n")

    return result


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else "Mumbai"
    predict_weather(location)
