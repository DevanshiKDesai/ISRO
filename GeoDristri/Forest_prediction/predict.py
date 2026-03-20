"""
Forest & Deforestation Prediction System
=========================================
User enters area / city / state → System predicts:
  1. Deforestation Alert Level (No Alert → Critical)
  2. Future NDVI (vegetation greenness next year)
  3. Future Forest Cover (sq km next year)
  4. AQI Impact Score (air quality effect)
  5. Human Impact Score (health & livelihood risks)
  6. Full effects report on humans

Works at 3 levels:
  - State level  : "Madhya Pradesh"
  - City level   : "Bhopal"
  - Area level   : "Satpura"

Usage:
    python predict.py "Maharashtra"
    python predict.py "Bhopal"
    python predict.py "Kerala"
    python predict.py "21.16,79.09"    ← GPS coordinates
"""

import requests
import joblib
import numpy as np
import json
import sys
import math
from datetime import datetime
from time import sleep

# ── Load models & data ────────────────────────────────────────────────────────
ALERT_MODEL  = joblib.load("alert_model.joblib")
NDVI_MODEL   = joblib.load("ndvi_model.joblib")
COVER_MODEL  = joblib.load("cover_model.joblib")
AQI_MODEL    = joblib.load("aqi_model.joblib")
HUMAN_MODEL  = joblib.load("human_model.joblib")
STATE_LE     = joblib.load("state_encoder.joblib")
FEATURE_COLS = joblib.load("feature_cols.joblib")

with open("metadata.json")   as f: META       = json.load(f)
with open("state_data.json") as f: STATE_DATA = {r['State']: r for r in json.load(f)}

ALERT_LABELS = {
    0: "No Alert",
    1: "Mild Deforestation",
    2: "Severe Deforestation",
    3: "Critical Deforestation"
}
ALERT_COLORS = {0: "SAFE", 1: "WARNING", 2: "DANGER", 3: "CRITICAL"}


# ── STEP 1: Geocoding ──────────────────────────────────────────────────────────
def get_location_info(location: str) -> dict:
    """Geocode and extract state, city, area from any location string."""
    location = location.strip()

    # GPS coordinates
    if "," in location:
        parts = location.split(",")
        try:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return reverse_geocode(lat, lon)
        except ValueError:
            pass

    # Text location — forward geocode
    url     = "https://nominatim.openstreetmap.org/search"
    params  = {"q": location + " India", "format": "json",
                "addressdetails": 1, "limit": 1}
    headers = {"User-Agent": "forest-predictor/1.0"}
    resp    = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data    = resp.json()

    if not data:
        # Try without "India"
        params["q"] = location
        resp  = requests.get(url, params=params, headers=headers, timeout=10)
        data  = resp.json()

    if not data:
        raise ValueError(f"Location '{location}' not found.")

    addr = data[0].get("address", {})
    return {
        "lat":     float(data[0]["lat"]),
        "lon":     float(data[0]["lon"]),
        "display": data[0]["display_name"],
        "state":   addr.get("state", ""),
        "city":    addr.get("city", addr.get("town", addr.get("county", ""))),
        "area":    addr.get("suburb", addr.get("village", addr.get("neighbourhood", ""))),
    }


def reverse_geocode(lat: float, lon: float) -> dict:
    url     = "https://nominatim.openstreetmap.org/reverse"
    params  = {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
    headers = {"User-Agent": "forest-predictor/1.0"}
    sleep(1)
    resp    = requests.get(url, params=params, headers=headers, timeout=10)
    addr    = resp.json().get("address", {})
    return {
        "lat":     lat, "lon": lon,
        "display": resp.json().get("display_name", ""),
        "state":   addr.get("state", ""),
        "city":    addr.get("city", addr.get("town", "")),
        "area":    addr.get("suburb", addr.get("village", "")),
    }


# ── STEP 2: Match to dataset state ────────────────────────────────────────────
def match_state(state_name: str) -> str:
    """Fuzzy match the geocoded state to one in our dataset."""
    known_states = list(STATE_DATA.keys())

    # Direct match
    for s in known_states:
        if state_name.lower() == s.lower():
            return s

    # Partial match
    for s in known_states:
        if state_name.lower() in s.lower() or s.lower() in state_name.lower():
            return s

    # Common aliases
    aliases = {
        "odisha": "Orissa", "uttarakhand": "Uttarakhand",
        "jammu and kashmir": "Jammu & Kashmir",
        "andaman and nicobar": "A & N Islands",
        "andaman & nicobar": "A & N Islands",
        "dadra and nagar haveli": "Dadra & Nagar Haveli",
        "daman and diu": "Daman & Diu",
        "pondicherry": "Puducherry",
        "telangana": "Telangana",
    }
    for alias, canonical in aliases.items():
        if alias in state_name.lower():
            return canonical

    # Default fallback — use nearest by first letter
    matches = [s for s in known_states if s[0].lower() == state_name[0].lower()]
    return matches[0] if matches else known_states[0]


# ── STEP 3: Fetch live NDVI from NASA MODIS (via API) ─────────────────────────
def get_live_ndvi(lat: float, lon: float) -> float:
    """
    Fetch approximate current NDVI using Open-Meteo vegetation index proxy.
    Falls back to state average from dataset if API unavailable.
    """
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":  lat,
            "longitude": lon,
            "daily":     ["et0_fao_evapotranspiration", "precipitation_sum",
                          "temperature_2m_max"],
            "timezone":  "auto",
            "forecast_days": 7,
        }
        resp   = requests.get(url, params=params, timeout=10)
        data   = resp.json()["daily"]

        # Estimate NDVI from ET0 and precipitation
        # Higher ET0 + precipitation = more vegetation
        et0    = np.mean([x for x in data["et0_fao_evapotranspiration"] if x])
        precip = sum(x for x in data["precipitation_sum"] if x)
        temp   = np.mean([x for x in data["temperature_2m_max"] if x])

        # Proxy formula calibrated to NDVI range 0.1-0.6
        ndvi_proxy = min(0.6, max(0.14, 0.15 + (precip / 50) * 0.1 + (et0 / 5) * 0.05))
        return round(float(ndvi_proxy), 3)

    except Exception:
        return None   # will use state historical value


# ── STEP 4: Build feature vector ──────────────────────────────────────────────
def build_features(state: str, live_ndvi: float | None, current_year: int) -> np.ndarray:
    """
    Build the 37-feature vector using:
    - Historical state data from our dataset (most features)
    - Live NDVI from API (if available)
    - Current year for temporal context
    """
    hist = STATE_DATA[state].copy()

    # Override NDVI if we got a live value
    if live_ndvi is not None:
        ndvi_to_use      = live_ndvi
        ndvi_change      = live_ndvi - hist['NDVI_mean']
    else:
        ndvi_to_use      = hist['NDVI_mean']
        ndvi_change      = hist.get('NDVI_Change_YoY', 0) or 0

    year_delta           = current_year - hist['Year']
    forest_cover         = hist['Forest_Cover_Area_SqKm']
    forest_pct           = hist['Forest_Percentage_Geographical']

    # Extrapolate forest cover trend to current year
    yoy                  = hist.get('Forest_Change_YoY', 0) or 0
    extrap_cover         = max(0, forest_cover + yoy * year_delta)
    extrap_pct           = max(0, forest_pct + (hist.get('Forest_Pct_Change_YoY', 0) or 0) * year_delta)

    feat_map = {
        'Forest_Cover_Area_SqKm':             extrap_cover,
        'Very_Dense_Forest_SqKm':             hist.get('Very_Dense_Forest_SqKm', 0) or 0,
        'Mod_Dense_Forest_SqKm':              hist.get('Mod_Dense_Forest_SqKm', 0) or 0,
        'Open_Forest_SqKm':                   hist.get('Open_Forest_SqKm', 0) or 0,
        'Total_Forest_Recorded_SqKm':         hist.get('Total_Forest_Recorded_SqKm', 0) or 0,
        'Forest_Percentage_Geographical':     extrap_pct,
        'Scrub_Area_SqKm':                    hist.get('Scrub_Area_SqKm', 0) or 0,
        'NDVI_mean':                          ndvi_to_use,
        'Total_Crop_Area_Ha':                 hist.get('Total_Crop_Area_Ha', 0) or 0,
        'RICE AREA (1000 ha)':                hist.get('RICE AREA (1000 ha)', 0) or 0,
        'WHEAT AREA (1000 ha)':               hist.get('WHEAT AREA (1000 ha)', 0) or 0,
        'SORGHUM AREA (1000 ha)':             hist.get('SORGHUM AREA (1000 ha)', 0) or 0,
        'MAIZE AREA (1000 ha)':               hist.get('MAIZE AREA (1000 ha)', 0) or 0,
        'GROUNDNUT AREA (1000 ha)':           hist.get('GROUNDNUT AREA (1000 ha)', 0) or 0,
        'COTTON AREA (1000 ha)':              hist.get('COTTON AREA (1000 ha)', 0) or 0,
        'SUGARCANE AREA (1000 ha)':           hist.get('SUGARCANE AREA (1000 ha)', 0) or 0,
        'Forest_Change_YoY':                  yoy,
        'NDVI_Change_YoY':                    ndvi_change,
        'Forest_Pct_Change_YoY':              hist.get('Forest_Pct_Change_YoY', 0) or 0,
        'VeryDense_Change_YoY':               hist.get('VeryDense_Change_YoY', 0) or 0,
        'ModDense_Change_YoY':                hist.get('ModDense_Change_YoY', 0) or 0,
        'OpenForest_Change_YoY':              hist.get('OpenForest_Change_YoY', 0) or 0,
        'Crop_Change_YoY':                    hist.get('Crop_Change_YoY', 0) or 0,
        'Forest_Cover_Area_SqKm_3yr_avg':     hist.get('Forest_Cover_Area_SqKm_3yr_avg', extrap_cover) or extrap_cover,
        'NDVI_mean_3yr_avg':                  hist.get('NDVI_mean_3yr_avg', ndvi_to_use) or ndvi_to_use,
        'Forest_Percentage_Geographical_3yr_avg': hist.get('Forest_Percentage_Geographical_3yr_avg', extrap_pct) or extrap_pct,
        'Forest_Cover_Area_SqKm_5yr_avg':     hist.get('Forest_Cover_Area_SqKm_5yr_avg', extrap_cover) or extrap_cover,
        'NDVI_mean_5yr_avg':                  hist.get('NDVI_mean_5yr_avg', ndvi_to_use) or ndvi_to_use,
        'Dense_to_Total_Ratio':               hist.get('Dense_to_Total_Ratio', 0) or 0,
        'Open_to_Total_Ratio':                hist.get('Open_to_Total_Ratio', 0) or 0,
        'Scrub_to_Forest_Ratio':              hist.get('Scrub_to_Forest_Ratio', 0) or 0,
        'Crop_to_Forest_Ratio':               hist.get('Crop_to_Forest_Ratio', 0) or 0,
        'Cum_Forest_Change':                  hist.get('Cum_Forest_Change', 0) or 0,
        'Deforestation_Streak':               hist.get('Deforestation_Streak', 0) or 0,
        'Streak_Count':                       hist.get('Streak_Count', 0) or 0,
        'Year':                               current_year,
        'State_enc':                          int(STATE_LE.transform([state])[0]) if state in STATE_LE.classes_ else 0,
    }

    X = np.array([[feat_map.get(c, 0) or 0 for c in FEATURE_COLS]])
    return X, feat_map


# ── STEP 5: Generate human effects report ─────────────────────────────────────
def generate_effects_report(alert_level: int, aqi_score: float,
                             human_score: float, forest_pct: float,
                             ndvi: float, future_cover: float,
                             current_cover: float, state: str) -> dict:
    """Generate detailed human impact assessment based on predictions."""

    cover_change_pct = ((future_cover - current_cover) / (current_cover + 1)) * 100

    effects = {
        "air_quality": {},
        "health":      {},
        "water":       {},
        "climate":     {},
        "livelihood":  {},
        "biodiversity":{}
    }

    # Air Quality
    if aqi_score > 150:
        aqi_cat = "Very Poor (AQI 200-300 range)"
        aqi_msg = "Severe respiratory risks. Vulnerable groups should stay indoors."
    elif aqi_score > 100:
        aqi_cat = "Poor (AQI 150-200 range)"
        aqi_msg = "Increased asthma, bronchitis risk. Outdoor activity discouraged."
    elif aqi_score > 50:
        aqi_cat = "Moderate (AQI 100-150 range)"
        aqi_msg = "Mild respiratory irritation. Elderly and children at risk."
    else:
        aqi_cat = "Good (AQI below 100)"
        aqi_msg = "Air quality acceptable. Forests are providing clean air."
    effects["air_quality"] = {"category": aqi_cat, "message": aqi_msg, "score": round(aqi_score, 1)}

    # Health impacts
    health_risks = []
    if ndvi < 0.25: health_risks.append("Severe heat stress due to loss of tree cover")
    if ndvi < 0.35: health_risks.append("Increased dust and particulate matter")
    if alert_level >= 2: health_risks.append("Higher vector-borne disease risk (malaria, dengue)")
    if alert_level >= 3: health_risks.append("Mental health impact from degraded environment")
    if forest_pct < 20:  health_risks.append("Inadequate natural carbon sink — CO2 rising")
    effects["health"] = {"risks": health_risks or ["No major health risks detected"],
                          "score": round(human_score, 1)}

    # Water security
    water_risks = []
    if alert_level >= 2: water_risks.append("Reduced rainfall interception → flash flood risk")
    if alert_level >= 1: water_risks.append("Groundwater recharge declining")
    if ndvi < 0.3:       water_risks.append("River siltation increasing — water quality declining")
    if cover_change_pct < -5: water_risks.append("Watershed degradation accelerating")
    effects["water"] = {"risks": water_risks or ["Water security maintained"],
                         "cover_change_pct": round(cover_change_pct, 2)}

    # Climate
    temp_rise_est = max(0, (0.3 - ndvi) * 5)
    effects["climate"] = {
        "local_temp_rise_est": f"+{temp_rise_est:.1f}°C (estimated local warming)",
        "carbon_impact": "High carbon release" if alert_level >= 2 else "Moderate" if alert_level == 1 else "Stable",
        "rainfall_risk": "Disrupted rainfall patterns" if alert_level >= 2 else "Stable patterns"
    }

    # Livelihood
    livelihood_impacts = []
    if alert_level >= 1: livelihood_impacts.append("Reduced timber and non-timber forest produce")
    if alert_level >= 2: livelihood_impacts.append("Tribal & rural communities losing forest income")
    if alert_level >= 2: livelihood_impacts.append("Soil erosion reducing agricultural productivity")
    if alert_level >= 3: livelihood_impacts.append("Complete ecosystem collapse risk in affected areas")
    effects["livelihood"] = {"impacts": livelihood_impacts or ["Livelihoods not significantly impacted"]}

    # Biodiversity
    species_risk = "Critical" if alert_level == 3 else "High" if alert_level == 2 else "Moderate" if alert_level == 1 else "Low"
    effects["biodiversity"] = {
        "species_risk_level": species_risk,
        "habitat_loss": f"{abs(min(0, cover_change_pct)):.1f}% habitat predicted to be lost next year"
    }

    return effects


# ── MAIN ───────────────────────────────────────────────────────────────────────
def predict_forest(location: str, level: str = "auto"):
    """
    level: 'state', 'city', 'area', or 'auto' (auto-detects from geocoding)
    """
    current_year = datetime.now().year

    print(f"\n{'='*60}")
    print(f"  Forest & Deforestation Prediction System")
    print(f"{'='*60}")

    # ── Geocode ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Locating '{location}'...")
    loc = get_location_info(location)
    print(f"  Found   : {loc['display']}")
    print(f"  State   : {loc['state']}")
    if loc['city']: print(f"  City    : {loc['city']}")
    if loc['area']: print(f"  Area    : {loc['area']}")
    print(f"  Coords  : {loc['lat']:.4f}, {loc['lon']:.4f}")

    # ── Match state ───────────────────────────────────────────────────────────
    print(f"\n[2/5] Matching to forest dataset...")
    matched_state = match_state(loc['state'])
    print(f"  Matched state: {matched_state}")
    hist = STATE_DATA[matched_state]
    print(f"  Last recorded year: {hist['Year']}")
    print(f"  Forest cover (last): {hist['Forest_Cover_Area_SqKm']:,.1f} sq km")
    print(f"  Forest %: {hist['Forest_Percentage_Geographical']:.2f}%")
    print(f"  NDVI (last): {hist['NDVI_mean']}")

    # ── Live NDVI ─────────────────────────────────────────────────────────────
    print(f"\n[3/5] Fetching live vegetation index (NDVI proxy)...")
    live_ndvi = get_live_ndvi(loc['lat'], loc['lon'])
    if live_ndvi:
        ndvi_diff = live_ndvi - hist['NDVI_mean']
        print(f"  Live NDVI estimate : {live_ndvi}")
        print(f"  vs Historical avg  : {hist['NDVI_mean']} (change: {ndvi_diff:+.3f})")
    else:
        print(f"  Using historical NDVI: {hist['NDVI_mean']}")

    # ── Build features & predict ──────────────────────────────────────────────
    print(f"\n[4/5] Building feature vector ({len(FEATURE_COLS)} features)...")
    X, feat_map = build_features(matched_state, live_ndvi, current_year)

    print(f"\n[5/5] Running 5 prediction models...")
    alert_probs   = ALERT_MODEL.predict_proba(X)[0]
    alert_pred    = int(ALERT_MODEL.predict(X)[0])
    alert_conf    = round(float(alert_probs[alert_pred]) * 100, 1)
    future_ndvi   = round(float(NDVI_MODEL.predict(X)[0]), 3)
    future_cover  = round(float(COVER_MODEL.predict(X)[0]), 1)
    aqi_score     = round(float(AQI_MODEL.predict(X)[0]), 1)
    human_score   = round(float(HUMAN_MODEL.predict(X)[0]), 1)

    current_cover = feat_map['Forest_Cover_Area_SqKm']
    cover_delta   = future_cover - current_cover

    # ── Effects report ────────────────────────────────────────────────────────
    effects = generate_effects_report(
        alert_pred, aqi_score, human_score,
        feat_map['Forest_Percentage_Geographical'],
        live_ndvi or hist['NDVI_mean'],
        future_cover, current_cover, matched_state
    )

    # ── Alert probabilities ───────────────────────────────────────────────────
    alert_label = ALERT_LABELS[alert_pred]
    alert_color = ALERT_COLORS[alert_pred]

    # ── Print Results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {location.upper()} ({loc['state']})")
    print(f"{'='*60}")

    print(f"\n  DEFORESTATION STATUS   : [{alert_color}] {alert_label} ({alert_conf}% confidence)")
    print(f"\n  Alert breakdown:")
    for i, (label) in ALERT_LABELS.items():
        bar  = "█" * int(alert_probs[i] * 40) + "░" * (40 - int(alert_probs[i] * 40))
        print(f"    {label:<25} {bar} {alert_probs[i]*100:.1f}%")

    print(f"\n  VEGETATION (NDVI):")
    print(f"    Current  : {live_ndvi or hist['NDVI_mean']} {'(live)' if live_ndvi else '(historical)'}")
    print(f"    Predicted next year: {future_ndvi}")
    ndvi_trend = "improving" if future_ndvi > (live_ndvi or hist['NDVI_mean']) else "declining"
    print(f"    Trend    : {ndvi_trend}")

    print(f"\n  FOREST COVER:")
    print(f"    Current  : {current_cover:,.1f} sq km")
    print(f"    Predicted: {future_cover:,.1f} sq km")
    print(f"    Change   : {cover_delta:+,.1f} sq km ({'GAIN' if cover_delta >= 0 else 'LOSS'})")

    print(f"\n  IMPACT SCORES:")
    print(f"    AQI Impact    : {aqi_score}/300  — {effects['air_quality']['category']}")
    print(f"    Human Impact  : {human_score}/100 — {'High risk' if human_score > 60 else 'Moderate risk' if human_score > 40 else 'Low risk'}")

    print(f"\n  AIR QUALITY FORECAST:")
    print(f"    {effects['air_quality']['message']}")

    print(f"\n  HEALTH RISKS:")
    for r in effects['health']['risks']:
        print(f"    • {r}")

    print(f"\n  WATER SECURITY:")
    for r in effects['water']['risks']:
        print(f"    • {r}")

    print(f"\n  CLIMATE EFFECTS:")
    print(f"    • {effects['climate']['local_temp_rise_est']}")
    print(f"    • Carbon sink: {effects['climate']['carbon_impact']}")
    print(f"    • Rainfall: {effects['climate']['rainfall_risk']}")

    print(f"\n  LIVELIHOOD IMPACTS:")
    for r in effects['livelihood']['impacts']:
        print(f"    • {r}")

    print(f"\n  BIODIVERSITY:")
    print(f"    Species risk: {effects['biodiversity']['species_risk_level']}")
    print(f"    {effects['biodiversity']['habitat_loss']}")

    print(f"\n{'='*60}\n")

    return {
        "location": location,
        "state": matched_state,
        "alert_level": alert_pred,
        "alert_label": alert_label,
        "alert_confidence": alert_conf,
        "current_ndvi": live_ndvi or hist['NDVI_mean'],
        "future_ndvi": future_ndvi,
        "current_forest_cover": current_cover,
        "future_forest_cover": future_cover,
        "forest_cover_change": cover_delta,
        "aqi_impact_score": aqi_score,
        "human_impact_score": human_score,
        "effects": effects,
    }


# ── Multi-level checker ────────────────────────────────────────────────────────
def check_multilevel(location: str):
    """Check deforestation at area → city → state levels."""
    print(f"\n{'#'*60}")
    print(f"  MULTI-LEVEL DEFORESTATION CHECK")
    print(f"  Query: {location}")
    print(f"{'#'*60}")

    loc = get_location_info(location)

    levels = []
    if loc['area']:  levels.append(("Area",  f"{loc['area']}, {loc['state']}"))
    if loc['city']:  levels.append(("City",  f"{loc['city']}, {loc['state']}"))
    if loc['state']: levels.append(("State", loc['state']))

    if not levels:
        levels = [("Location", location)]

    summary = []
    for level_name, level_query in levels:
        print(f"\n{'─'*60}")
        print(f"  Checking at {level_name} level: {level_query}")
        print(f"{'─'*60}")
        result = predict_forest(level_query)
        summary.append({
            "level": level_name,
            "location": level_query,
            "alert": result['alert_label'],
            "aqi": result['aqi_impact_score'],
            "human": result['human_impact_score'],
        })

    print(f"\n{'='*60}")
    print(f"  SUMMARY ACROSS ALL LEVELS")
    print(f"{'='*60}")
    for s in summary:
        print(f"  {s['level']:<8} {s['location']:<35} Alert: {s['alert']}")
    print()

    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1:
        location = sys.argv[1]
        mode     = sys.argv[2] if len(sys.argv) > 2 else "single"
        if mode == "multi":
            check_multilevel(location)
        else:
            predict_forest(location)
    else:
        predict_forest("Maharashtra")
