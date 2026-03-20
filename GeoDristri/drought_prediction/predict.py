"""
Drought Prediction - Inference Pipeline
========================================
User enters a location → APIs auto-fill all features → 3 models predict:
  1. Drought Category  (Extremely Dry → Extremely Wet)
  2. Drought Status    (Is drought happening? Yes/No)
  3. SPEI Index        (Continuous drought severity value)

Usage:
    python predict.py "Rajasthan"
    python predict.py "Kerala"
    python predict.py "28.61,77.20"
    python predict.py "Cairo"          ← works globally!
"""

import requests
import joblib
import numpy as np
import json
import sys
import math
from datetime import datetime
from time import sleep

# ── Load models ────────────────────────────────────────────────────────────────
CATEGORY_MODEL  = joblib.load("category_model.joblib")
STATUS_MODEL    = joblib.load("status_model.joblib")
SPEI_MODEL      = joblib.load("spei_model.joblib")
CATEGORY_LE     = joblib.load("category_encoder.joblib")
FEATURE_COLS    = joblib.load("feature_cols.joblib")

with open("metadata.json") as f:
    META = json.load(f)


# ── STEP 1: Geocoding ──────────────────────────────────────────────────────────
def get_coordinates(location: str) -> tuple:
    """Convert city/region name or 'lat,lon' to (lat, lon, display_name)."""
    location = location.strip()
    if "," in location:
        parts = location.split(",")
        try:
            return float(parts[0].strip()), float(parts[1].strip()), location
        except ValueError:
            pass

    url     = "https://nominatim.openstreetmap.org/search"
    params  = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "drought-predictor/1.0"}
    resp    = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data    = resp.json()

    if not data:
        raise ValueError(f"Location '{location}' not found. Try a different name.")

    return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]


# ── STEP 2: Weather data from Open-Meteo ──────────────────────────────────────
def get_weather_data(lat: float, lon: float) -> dict:
    """
    Fetch all weather features needed by the model:
    temperature, humidity, precipitation, wind, solar radiation.
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
            "shortwave_radiation_sum",   # solar radiation
        ],
        "hourly":        "relativehumidity_2m",
        "timezone":      "auto",
        "forecast_days": 7,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    daily  = data["daily"]
    hourly = data["hourly"]

    def avg(lst):
        vals = [x for x in lst if x is not None]
        return float(np.mean(vals)) if vals else 0.0

    def total(lst):
        return float(sum(x for x in lst if x is not None))

    temps_max = daily["temperature_2m_max"]
    temps_min = daily["temperature_2m_min"]
    temps_avg = [(mx + mn) / 2 for mx, mn in zip(temps_max, temps_min)
                 if mx is not None and mn is not None]

    max_temp    = round(avg(temps_max), 3)
    min_temp    = round(avg(temps_min), 3)
    avg_temp    = round(avg(temps_avg), 3)
    precip      = round(total(daily["precipitation_sum"]), 3)
    wind_kmh    = round(avg(daily["windspeed_10m_max"]), 3)
    wind_ms     = round(wind_kmh / 3.6, 3)               # km/h → m/s
    wind_bin    = round(wind_ms / 0.5) * 0.5             # bin to nearest 0.5
    solar       = round(avg(daily["shortwave_radiation_sum"]), 3)
    humidity    = round(avg(hourly["relativehumidity_2m"]), 3)

    return {
        "Relative Humidity (%)":    humidity,
        "Max Temp (°C)":            max_temp,
        "Min Temp (°C)":            min_temp,
        "Wind Speed (m/s)":         wind_ms,
        "Avg Temperature (°C)":     avg_temp,
        "Solar Radiation":          solar,
        "Precipitation (mm)":       precip,
        "Wind Speed (m/s) (bins)":  wind_bin,
    }


# ── STEP 3: Estimate SPEI from precipitation & temperature ────────────────────
def estimate_spei(precip_mm: float, avg_temp: float,
                  humidity: float, lat: float) -> float:
    """
    Estimate SPEI (Standardized Precipitation-Evapotranspiration Index).
    SPEI = (P - PET) normalized.
    Uses Thornthwaite PET approximation.
    """
    # Thornthwaite PET approximation
    # PET ≈ 16 × (10T/I)^a × correction_factor
    T = max(avg_temp, 0)
    I = max((T / 5) ** 1.514, 0.001)   # heat index approx
    a = 0.492 + 0.0179 * I - 7.71e-5 * I**2 + 6.75e-7 * I**3
    pet = 16 * ((10 * T / (I * 12)) ** a) if T > 0 else 0
    pet = max(pet, 0)

    # Adjust PET for humidity
    pet_adj = pet * (1 - humidity / 200)

    # Water balance
    water_balance = precip_mm - pet_adj

    # Normalize to SPEI-like scale (roughly -3 to +3)
    spei = water_balance / (pet_adj + 50 + 1e-6)
    spei = float(np.clip(spei, -3.5, 3.5))
    return round(spei, 4)


# ── STEP 4: Encode spatial & temporal features ─────────────────────────────────
def encode_spatiotemporal(lat: float, lon: float, month: int) -> dict:
    """Convert lat/lon/month to sin-cos encoding used by the model."""
    lat_rad   = math.radians(lat)
    lon_rad   = math.radians(lon)
    month_rad = math.radians(month * 30)   # 30° per month

    return {
        "lat_sin":   round(math.sin(lat_rad),   6),
        "lat_cos":   round(math.cos(lat_rad),   6),
        "lon_sin":   round(math.sin(lon_rad),   6),
        "lon_cos":   round(math.cos(lon_rad),   6),
        "month_sin": round(math.sin(month_rad), 6),
        "month_cos": round(math.cos(month_rad), 6),
    }


# ── STEP 5: Build full feature vector ─────────────────────────────────────────
def build_features(lat, lon, month, weather, spei):
    """Assemble all 24 features in the exact order the model expects."""
    st = encode_spatiotemporal(lat, lon, month)

    season = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}[month]

    temp_range      = weather["Max Temp (°C)"] - weather["Min Temp (°C)"]
    heat_stress     = weather["Max Temp (°C)"] * (1 - weather["Relative Humidity (%)"] / 100)
    precip_humidity = weather["Precipitation (mm)"] * weather["Relative Humidity (%)"]
    aridity_index   = weather["Solar Radiation"] / (weather["Precipitation (mm)"] + 1)
    wind_evap       = weather["Wind Speed (m/s)"] * weather["Max Temp (°C)"]

    feat_map = {
        # Raw weather
        "Relative Humidity (%)":    weather["Relative Humidity (%)"],
        "Max Temp (°C)":            weather["Max Temp (°C)"],
        "Min Temp (°C)":            weather["Min Temp (°C)"],
        "Wind Speed (m/s)":         weather["Wind Speed (m/s)"],
        "Avg Temperature (°C)":     weather["Avg Temperature (°C)"],
        "Solar Radiation":          weather["Solar Radiation"],
        "Precipitation (mm)":       weather["Precipitation (mm)"],
        "Drought Index (SPEI)":     spei,
        # Spatial
        "lat_sin":   st["lat_sin"],
        "lat_cos":   st["lat_cos"],
        "lon_sin":   st["lon_sin"],
        "lon_cos":   st["lon_cos"],
        # Temporal
        "month_sin": st["month_sin"],
        "month_cos": st["month_cos"],
        "month":     month,
        "season":    season,
        # Decoded
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        # Engineered
        "temp_range":      round(temp_range, 3),
        "heat_stress":     round(heat_stress, 3),
        "precip_humidity": round(precip_humidity, 3),
        "aridity_index":   round(aridity_index, 3),
        "wind_evap":       round(wind_evap, 3),
        "Wind Speed (m/s) (bins)": weather["Wind Speed (m/s) (bins)"],
    }

    X = np.array([[feat_map[c] for c in FEATURE_COLS]])
    return X, feat_map


# ── STEP 6: Predict ────────────────────────────────────────────────────────────
def predict(X: np.ndarray) -> dict:
    """Run all 3 models and return results."""
    # Category probabilities
    cat_probs = CATEGORY_MODEL.predict_proba(X)[0]
    top3_idx  = np.argsort(cat_probs)[::-1][:3]
    top3      = [(CATEGORY_LE.inverse_transform([i])[0], round(cat_probs[i]*100, 1))
                 for i in top3_idx]

    # Binary status
    status      = int(STATUS_MODEL.predict(X)[0])
    status_prob = round(float(STATUS_MODEL.predict_proba(X)[0][status]) * 100, 1)

    # SPEI regression
    spei_pred = round(float(SPEI_MODEL.predict(X)[0]), 3)

    return {
        "top_categories": top3,
        "status":         status,
        "status_prob":    status_prob,
        "spei_predicted": spei_pred,
    }


# ── Severity helpers ──────────────────────────────────────────────────────────
def spei_description(spei: float) -> str:
    if spei <= -2.0:   return "Extremely Dry"
    elif spei <= -1.5: return "Severely Dry"
    elif spei <= -1.0: return "Moderately Dry"
    elif spei < 1.0:   return "Near Normal"
    elif spei < 1.5:   return "Moderately Wet"
    elif spei < 2.0:   return "Severely Wet"
    else:              return "Extremely Wet"


def severity_color(category: str) -> str:
    colors = {
        "Extremely Dry": "CRITICAL",
        "Severely Dry":  "HIGH",
        "Moderately Dry":"MODERATE",
        "Near Normal":   "NORMAL",
        "Moderately Wet":"LOW",
        "Severely Wet":  "ELEVATED",
        "Extremely Wet": "ELEVATED",
    }
    return colors.get(category, "UNKNOWN")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def predict_drought(location: str):
    print(f"\n{'='*57}")
    print(f"  Drought Prediction System")
    print(f"{'='*57}")

    print(f"\n[1/5] Geocoding '{location}'...")
    lat, lon, display = get_coordinates(location)
    print(f"  Found  : {display}")
    print(f"  Coords : lat={lat:.4f}, lon={lon:.4f}")

    print(f"\n[2/5] Fetching weather data from Open-Meteo...")
    weather = get_weather_data(lat, lon)
    print(f"  Max Temp          : {weather['Max Temp (°C)']} °C")
    print(f"  Min Temp          : {weather['Min Temp (°C)']} °C")
    print(f"  Avg Temp          : {weather['Avg Temperature (°C)']} °C")
    print(f"  Relative Humidity : {weather['Relative Humidity (%)']} %")
    print(f"  Precipitation     : {weather['Precipitation (mm)']} mm")
    print(f"  Wind Speed        : {weather['Wind Speed (m/s)']} m/s")
    print(f"  Solar Radiation   : {weather['Solar Radiation']} MJ/m²")

    print(f"\n[3/5] Estimating SPEI drought index...")
    month = datetime.now().month
    spei  = estimate_spei(
        weather["Precipitation (mm)"],
        weather["Avg Temperature (°C)"],
        weather["Relative Humidity (%)"],
        lat
    )
    print(f"  SPEI estimate : {spei}")
    print(f"  Interpretation: {spei_description(spei)}")

    print(f"\n[4/5] Building feature vector ({len(FEATURE_COLS)} features)...")
    X, feat_map = build_features(lat, lon, month, weather, spei)
    print(f"  Month   : {month} | Season: {['Winter','Spring','Summer','Autumn'][feat_map['season']]}")
    print(f"  Aridity : {feat_map['aridity_index']:.2f} | Heat stress: {feat_map['heat_stress']:.2f}")

    print(f"\n[5/5] Running 3 prediction models...")
    result = predict(X)

    # ── Output ────────────────────────────────────────────────────────────────
    top_category = result["top_categories"][0][0]
    alert_level  = severity_color(top_category)

    print(f"\n{'='*57}")
    print(f"  DROUGHT PREDICTION RESULTS — {location.upper()}")
    print(f"{'='*57}")

    print(f"\n  Alert Level     : [{alert_level}]")
    print(f"  Drought Active  : {'YES — Drought conditions detected' if result['status'] == 1 else 'NO — No active drought'} ({result['status_prob']}% confidence)")
    print(f"  SPEI Index      : {result['spei_predicted']}  ({spei_description(result['spei_predicted'])})")

    print(f"\n  Drought Category (top 3 predictions):")
    for i, (cat, prob) in enumerate(result["top_categories"], 1):
        filled = int(prob / 5)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"  #{i}  {cat:<18} {bar} {prob}%")

    print(f"\n  Weather summary:")
    print(f"  Temp {weather['Min Temp (°C)']}°C – {weather['Max Temp (°C)']}°C  |  "
          f"Humidity {weather['Relative Humidity (%)']}%  |  "
          f"Rain {weather['Precipitation (mm)']}mm  |  "
          f"Wind {weather['Wind Speed (m/s)']} m/s")
    print(f"{'='*57}\n")

    return result


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else "Rajasthan"
    predict_drought(location)
