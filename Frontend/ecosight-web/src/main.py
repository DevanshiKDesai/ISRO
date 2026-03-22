"""
GeoDrishti FastAPI Backend — main.py
Chatbot uses ONLY real data: your joblib models + Open-Meteo + NASA POWER APIs.
Zero hallucination policy: if data is unavailable, it says so explicitly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os

app = FastAPI(title="GeoDrishti API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
# EMAIL CONFIG — set your SMTP creds here or via env vars
# ══════════════════════════════════════════════
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER", "your_email@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_app_password")
ALERT_FROM    = os.getenv("ALERT_FROM", "geodrishti@isro.gov.in")

# ══════════════════════════════════════════════
# DATASET LOADING
# ══════════════════════════════════════════════
print("Loading GeoDrishti Datasets...")
df_pop = df_forest = df_crop = df_drought = None

try:
    df_pop = pd.read_csv("Population/india_enriched.csv", encoding='latin1')
    print("✅ Population Dataset Loaded")
except Exception as e:
    print(f"⚠️  Population dataset offline: {e}")

try:
    df_forest = pd.read_csv("Forest_prediction/New Forest.csv", encoding='latin1')
    print("✅ Forest Dataset Loaded")
except Exception as e:
    print(f"⚠️  Forest dataset offline: {e}")

try:
    df_crop = pd.read_csv("crop_predictor/enhanced_crop_yield_dataset (1).csv", encoding='latin1')
    print("✅ Crop Dataset Loaded")
except Exception as e:
    print(f"⚠️  Crop dataset offline: {e}")

try:
    df_drought = pd.read_excel("drought_prediction/Drought New.xlsx")
    print("✅ Drought Dataset Loaded")
except:
    try:
        df_drought = pd.read_csv("drought_prediction/Drought New.xlsx", encoding='latin1', on_bad_lines='skip')
        print("✅ Drought Dataset Loaded (CSV fallback)")
    except Exception as e:
        print(f"⚠️  Drought dataset offline: {e}")

# ══════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════
EVENT_MODEL = None
try:
    EVENT_MODEL = joblib.load("Weather_Prediction/event_model.joblib")
    print("✅ Disaster Event Model Loaded")
except Exception as e:
    print(f"⚠️  Disaster model offline: {e}")

# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def geocode(location: str):
    """Returns (lat, lon) for a city name using Open-Meteo geocoding."""
    try:
        r = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1",
            timeout=6
        ).json()
        if "results" not in r or not r["results"]:
            return None, None
        return r["results"][0]["latitude"], r["results"][0]["longitude"]
    except Exception:
        return None, None


def fetch_live_weather(lat: float, lon: float) -> dict:
    """Fetches real-time weather from Open-Meteo. Returns dict or raises."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=relative_humidity_2m,precipitation_probability,soil_moisture_0_1cm"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
        f"&timezone=Asia%2FKolkata&forecast_days=3"
    )
    r = requests.get(url, timeout=8).json()
    return r


def ndvi_from_coords(lat: float, lon: float) -> Optional[float]:
    """
    Approximates NDVI using NASA POWER API (surface solar radiation as proxy).
    Real NDVI requires Sentinel/Landsat — swap this for an EE call in production.
    """
    try:
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=16)).strftime("%Y%m%d")
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR"
            f"&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start}&end={end}&format=JSON"
        )
        r = requests.get(url, timeout=10).json()
        props = r["properties"]["parameter"]
        srad_vals = list(props["ALLSKY_SFC_SW_DWN"].values())
        prec_vals = list(props["PRECTOTCORR"].values())
        # Simple proxy: healthy veg correlates with moderate radiation + precipitation
        avg_srad = np.nanmean([v for v in srad_vals if v != -999])
        avg_prec = np.nanmean([v for v in prec_vals if v != -999])
        # Normalise to [0,1] range roughly matching NDVI
        ndvi_proxy = min(max((avg_prec * 0.04 + avg_srad * 0.005), 0.0), 1.0)
        return round(ndvi_proxy, 3)
    except Exception:
        return None


def alert_level_from_indices(ndvi, temp, wind, precip) -> str:
    """
    Pure logic — no model guessing.
    Returns 'green' | 'yellow' | 'orange' | 'red'
    """
    score = 0
    if ndvi is not None:
        if ndvi < 0.15:  score += 3
        elif ndvi < 0.3: score += 1
    if temp is not None:
        if temp > 45:    score += 3
        elif temp > 40:  score += 2
        elif temp > 37:  score += 1
        if temp < 5:     score += 2
    if wind is not None:
        if wind > 80:    score += 3
        elif wind > 55:  score += 2
        elif wind > 35:  score += 1
    if precip is not None:
        if precip > 80:  score += 2
        elif precip > 50: score += 1

    if score >= 6:   return "red"
    elif score >= 4: return "orange"
    elif score >= 2: return "yellow"
    return "green"


def send_alert_email(to_email: str, aoi_name: str, level: str, summary: str, coords: dict):
    """Sends an HTML alert email. Call only when alert_level != green."""
    level_colors = {"yellow": "#f59e0b", "orange": "#f97316", "red": "#ef4444"}
    color = level_colors.get(level, "#10b981")
    html = f"""
    <html><body style="font-family:sans-serif;background:#060a14;color:#e8edf5;padding:32px;">
      <div style="max-width:520px;margin:auto;background:#0d1526;border-radius:12px;
                  border:2px solid {color};padding:28px;">
        <h2 style="color:{color};margin-top:0;">⚠️ GeoDrishti Alert — {level.upper()}</h2>
        <p><strong>Area of Interest:</strong> {aoi_name}</p>
        <p><strong>Coordinates:</strong> {coords.get('lat')}, {coords.get('lng')}</p>
        <p><strong>Summary:</strong><br>{summary}</p>
        <p style="color:#6b7fa3;font-size:12px;margin-top:24px;">
          This alert was generated automatically by ISRO GeoDrishti EcoSight.<br>
          Do not reply to this email. Timestamp: {datetime.utcnow().isoformat()}Z
        </p>
      </div>
    </body></html>
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[GeoDrishti] {level.upper()} Alert — {aoi_name}"
        msg["From"]    = ALERT_FROM
        msg["To"]      = to_email
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(ALERT_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


# ══════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    history: list = []   # [{role, content}] for multi-turn

class AOIRequest(BaseModel):
    lat: float
    lng: float
    north: float
    south: float
    east: float
    west: float
    aoi_name: str = "AOI"
    user_email: Optional[str] = None

class AlertEmailRequest(BaseModel):
    to_email: str
    aoi_name: str
    level: str
    summary: str
    lat: float
    lng: float

class ExportRequest(BaseModel):
    aoi_name: str
    lat: float
    lng: float
    bounds: dict
    indices: dict
    alert_level: str
    summary: str


# ══════════════════════════════════════════════
# CHATBOT  — ZERO HALLUCINATION
# All answers come from: your datasets, your models, or verified live APIs.
# If none available → says so explicitly. Never invents numbers.
# ══════════════════════════════════════════════

INTENT_KEYWORDS = {
    "weather":    ["weather", "temperature", "rain", "forecast", "wind", "cyclone", "heatwave", "cold"],
    "disaster":   ["disaster", "flood", "drought risk", "risk", "alert", "event", "calamity"],
    "forest":     ["forest", "tree", "deforestation", "cover", "woodland", "green cover"],
    "crop":       ["crop", "yield", "agriculture", "farm", "wheat", "rice", "harvest", "soil"],
    "drought":    ["drought", "spei", "dry", "water stress", "rainfall deficit"],
    "population": ["population", "people", "census", "birth", "death", "urbanization", "growth"],
    "ndvi":       ["ndvi", "vegetation", "greenness", "satellite index", "ndwi", "ndbi"],
}

REFUSAL = (
    "I can only answer using verified data from GeoDrishti datasets or live satellite APIs. "
    "I don't have reliable information on that topic right now — please rephrase or ask about "
    "weather, forest cover, crop yield, drought, population, or live satellite indices."
)

def classify_intent(msg: str) -> str:
    msg_l = msg.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in msg_l for k in kws):
            return intent
    return "unknown"

def extract_location(msg: str) -> str:
    """Naive extraction: look for capitalised words that could be city/state names."""
    import re
    candidates = re.findall(r'\b[A-Z][a-z]{2,}\b', msg)
    indian_places = {
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kerala", "Rajasthan",
        "Maharashtra", "Gujarat", "Punjab", "Haryana", "Odisha", "Assam",
        "Telangana", "Karnataka", "Bihar", "Madhya", "Pradesh", "Uttarakhand",
        "Andhra", "Sikkim", "Meghalaya", "Manipur", "Nagaland", "Tripura"
    }
    for c in candidates:
        if c in indian_places:
            return c
    return candidates[0] if candidates else "India"

def extract_year(msg: str) -> int:
    import re
    years = re.findall(r'\b(19[0-9]{2}|20[0-2][0-9])\b', msg)
    return int(years[0]) if years else 2017

def extract_state(msg: str) -> str:
    return extract_location(msg)


def answer_weather(msg: str) -> str:
    loc = extract_location(msg)
    lat, lon = geocode(loc)
    if lat is None:
        return f"I couldn't locate **{loc}** on the map. Please specify a valid Indian city or state."
    try:
        data = fetch_live_weather(lat, lon)
        cw   = data["current_weather"]
        temp = cw["temperature"]
        wind = cw["windspeed"]
        code = cw["weathercode"]
        daily = data.get("daily", {})
        max_t = daily.get("temperature_2m_max", [None])[0]
        min_t = daily.get("temperature_2m_min", [None])[0]
        prec  = daily.get("precipitation_sum", [None])[0]

        # WMO weather code descriptions (subset)
        wmo = {0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
               45:"Foggy", 51:"Light drizzle", 61:"Slight rain", 63:"Moderate rain",
               71:"Slight snow", 80:"Rain showers", 95:"Thunderstorm"}
        condition = wmo.get(code, f"Code {code}")

        lines = [
            f"**Live weather for {loc}** (via Open-Meteo, updated hourly):",
            f"• Condition: {condition}",
            f"• Current temp: **{temp}°C** | Wind: **{wind} km/h**",
        ]
        if max_t and min_t:
            lines.append(f"• Today's range: {min_t}°C – {max_t}°C")
        if prec is not None:
            lines.append(f"• Precipitation today: {prec} mm")

        # Risk flag
        level = alert_level_from_indices(None, temp, wind, prec)
        if level == "red":
            lines.append(f"\n🔴 **CRITICAL** — Severe weather conditions. Immediate action advised.")
        elif level == "orange":
            lines.append(f"\n🟠 **WARNING** — Hazardous conditions detected. Monitor closely.")
        elif level == "yellow":
            lines.append(f"\n🟡 **HEADS UP** — Elevated risk. Stay informed.")
        else:
            lines.append(f"\n🟢 Conditions appear stable.")

        return "\n".join(lines)
    except Exception as e:
        return f"Live weather fetch failed for {loc}. The Open-Meteo API may be unreachable right now. Error: {e}"


def answer_disaster(msg: str) -> str:
    loc = extract_location(msg)
    lat, lon = geocode(loc)
    if lat is None:
        return f"Couldn't locate {loc}. Please name a valid city or state."
    try:
        data  = fetch_live_weather(lat, lon)
        cw    = data["current_weather"]
        temp  = cw["temperature"]
        wind  = cw["windspeed"]
        daily = data.get("daily", {})
        prec  = daily.get("precipitation_sum", [0])[0] or 0

        level = alert_level_from_indices(None, temp, wind, prec)
        level_txt = {"green": "🟢 Stable", "yellow": "🟡 Heads Up", "orange": "🟠 Warning", "red": "🔴 Critical"}[level]

        lines = [
            f"**Disaster risk assessment for {loc}** (live data, Open-Meteo):",
            f"• Temp: {temp}°C | Wind: {wind} km/h | Precipitation: {prec} mm",
            f"• Risk level: **{level_txt}**",
        ]
        if wind > 60:
            lines.append("• High wind speed — potential cyclonic or severe gale risk.")
        if temp > 42:
            lines.append("• Heatwave threshold exceeded — health risk for vulnerable populations.")
        if prec and prec > 50:
            lines.append("• Heavy precipitation — localised flooding possible.")
        if EVENT_MODEL:
            # Feed real values into your joblib model if feature signature matches
            try:
                feat = np.array([[temp, wind, prec]])
                pred = EVENT_MODEL.predict(feat)[0]
                lines.append(f"• **GeoDrishti ML model prediction:** {pred}")
            except Exception:
                lines.append("• (ML model prediction unavailable — feature mismatch)")
        else:
            lines.append("• ML event model not loaded — rule-based assessment used.")

        return "\n".join(lines)
    except Exception as e:
        return f"Could not fetch live data for {loc}: {e}"


def answer_forest(msg: str) -> str:
    if df_forest is None:
        return "The forest dataset is currently offline. Please verify the file path on the server."
    state = extract_state(msg)
    data = df_forest[df_forest['State'].str.contains(state, case=False, na=False)]
    if data.empty:
        avail = ", ".join(df_forest['State'].dropna().unique()[:10])
        return f"No forest data found for **{state}**. Available states include: {avail}."
    row = data.iloc[-1]
    lines = [
        f"**Forest cover data for {state}** (GeoDrishti dataset):",
        f"• Total recorded forest area: **{row['Total_Forest_Recorded_SqKm']} km²**",
        f"• Forest as % of geographical area: **{row['Forest_Percentage_Geographical']}%**",
        "• Source: GeoDrishti Forest Prediction dataset (offline/static).",
    ]
    return "\n".join(lines)


def answer_crop(msg: str) -> str:
    if df_crop is None:
        return "The crop yield dataset is offline. Please verify the file path on the server."
    state = extract_state(msg)
    data = df_crop[df_crop['State_Name'].str.contains(state, case=False, na=False)]
    if data.empty:
        avail = ", ".join(df_crop['State_Name'].dropna().unique()[:10])
        return f"No crop data found for **{state}**. Available states: {avail}."
    avg_yield   = round(data['Crop Yield (kg per hectare)'].mean(), 2)
    common_soil = data['Soil_Type'].mode()[0] if 'Soil_Type' in data.columns else "N/A"
    top_crop    = data['Crop'].mode()[0] if 'Crop' in data.columns else "N/A"
    lines = [
        f"**Crop data for {state}** (GeoDrishti dataset):",
        f"• Most common crop: **{top_crop}**",
        f"• Predominant soil type: **{common_soil}**",
        f"• Average historical yield: **{avg_yield} kg/hectare**",
        "• Source: GeoDrishti Crop Yield dataset (offline/static).",
    ]
    return "\n".join(lines)


def answer_drought(msg: str) -> str:
    if df_drought is None:
        return "The drought dataset is offline. Please verify the file path on the server."
    try:
        mean_spei = round(df_drought['Drought Index (SPEI)'].mean(), 3)
        avg_temp  = round(df_drought['Avg Temperature (°C)'].mean(), 2)
        lines = [
            "**Drought assessment** (GeoDrishti dataset):",
            f"• Average SPEI (Standardised Precipitation-Evapotranspiration Index): **{mean_spei}**",
            f"  — Interpretation: {'Below -1.0 indicates drought conditions.' if mean_spei < -1.0 else 'Near-normal moisture conditions on average.'}",
            f"• Average recorded temperature: **{avg_temp}°C**",
            "• Source: GeoDrishti Drought dataset (offline/static).",
        ]
        return "\n".join(lines)
    except KeyError as e:
        return f"Drought dataset loaded but column missing: {e}. Check your CSV headers."


def answer_population(msg: str) -> str:
    if df_pop is None:
        return "The population dataset is offline. Please verify the file path on the server."
    year = extract_year(msg)
    data = df_pop[df_pop['Year'] == year]
    if data.empty:
        avail = sorted(df_pop['Year'].dropna().unique().tolist())
        return f"No population data for **{year}**. Available years: {avail}."
    row = data.iloc[0]
    try:
        lines = [
            f"**India population data for {year}** (GeoDrishti dataset):",
            f"• Population: **{round(row['India Population (Millions)'], 2)} million**",
            f"• Birth rate: **{round(row['Birth Rate (per 1000)'], 2)} per 1,000**",
            f"• Death rate: **{round(row['Death Rate (per 1000)'], 2)} per 1,000**",
            f"• Urbanisation rate: **{round(row['Urbanization_Rate'], 2)}%**",
            "• Source: GeoDrishti Population dataset (offline/static).",
        ]
        return "\n".join(lines)
    except KeyError as e:
        return f"Population dataset loaded but missing column: {e}."


def answer_ndvi(msg: str) -> str:
    loc = extract_location(msg)
    lat, lon = geocode(loc)
    if lat is None:
        return f"Couldn't locate {loc}."
    ndvi = ndvi_from_coords(lat, lon)
    if ndvi is None:
        return "NASA POWER API is unreachable right now. NDVI proxy could not be calculated."
    interp = (
        "Dense/healthy vegetation" if ndvi > 0.6 else
        "Moderate vegetation"      if ndvi > 0.3 else
        "Sparse vegetation / stress" if ndvi > 0.1 else
        "Bare soil or urban area"
    )
    lines = [
        f"**Vegetation index estimate for {loc}** (NASA POWER 16-day composite):",
        f"• NDVI proxy: **{ndvi}** — {interp}",
        "⚠️ Note: This is a radiation-based proxy. True NDVI requires Sentinel-2/Landsat imagery via Google Earth Engine.",
        "• Source: NASA POWER API (live).",
    ]
    return "\n".join(lines)


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Routes user message to the correct real-data handler.
    Never fabricates answers — returns explicit unavailability messages if data missing.
    """
    msg    = req.message.strip()
    intent = classify_intent(msg)

    dispatch = {
        "weather":    answer_weather,
        "disaster":   answer_disaster,
        "forest":     answer_forest,
        "crop":       answer_crop,
        "drought":    answer_drought,
        "population": answer_population,
        "ndvi":       answer_ndvi,
    }

    if intent in dispatch:
        reply = dispatch[intent](msg)
    else:
        reply = REFUSAL

    return {"reply": reply, "intent": intent}


# ══════════════════════════════════════════════
# AOI ANALYSIS ENDPOINT
# ══════════════════════════════════════════════

@app.post("/aoi/analyze")
def analyze_aoi(req: AOIRequest):
    """
    Analyses an AOI using live Open-Meteo weather + NASA POWER NDVI proxy.
    Optionally emails an alert if level != green.
    """
    lat, lon = req.lat, req.lng

    # 1. Live weather
    try:
        weather = fetch_live_weather(lat, lon)
        cw      = weather["current_weather"]
        temp    = cw["temperature"]
        wind    = cw["windspeed"]
        daily   = weather.get("daily", {})
        prec    = (daily.get("precipitation_sum") or [0])[0] or 0
    except Exception as e:
        return {"error": f"Weather API unavailable: {e}"}

    # 2. NDVI proxy
    ndvi = ndvi_from_coords(lat, lon)

    # 3. Derived indices (simplified approximations)
    ndwi = round(max(-0.5, min(0.5, (prec / 200) - 0.3)), 3)  # proxy from precip
    ndbi = round(max(-0.5, min(0.5, 0.4 - (ndvi or 0.3))), 3)  # inverse of veg

    # 4. Alert level
    level   = alert_level_from_indices(ndvi, temp, wind, prec)
    level_labels = {
        "green":  "🟢 Normal — no significant change detected.",
        "yellow": "🟡 Heads Up — early signs of change. Monitor regularly.",
        "orange": "🟠 Warning — notable environmental stress. Action recommended.",
        "red":    "🔴 Critical — severe conditions. Immediate intervention needed.",
    }
    summary = level_labels[level]

    # 5. Email alert if warranted
    email_sent = False
    if req.user_email and level != "green":
        email_sent = send_alert_email(
            req.user_email, req.aoi_name, level, summary,
            {"lat": req.lat, "lng": req.lng}
        )

    return {
        "lat":         lat,
        "lng":         lon,
        "aoi_name":    req.aoi_name,
        "ndvi":        ndvi,
        "ndwi":        ndwi,
        "ndbi":        ndbi,
        "temperature": temp,
        "wind_speed":  wind,
        "precipitation": prec,
        "alert_level": level,
        "summary":     summary,
        "email_sent":  email_sent,
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "data_sources": ["Open-Meteo (live)", "NASA POWER (16-day composite)"],
    }


# ══════════════════════════════════════════════
# EMAIL ALERT ENDPOINT (manual trigger from frontend)
# ══════════════════════════════════════════════

@app.post("/alert/email")
def trigger_alert_email(req: AlertEmailRequest):
    sent = send_alert_email(
        req.to_email, req.aoi_name, req.level, req.summary,
        {"lat": req.lat, "lng": req.lng}
    )
    if sent:
        return {"success": True, "message": f"Alert email sent to {req.to_email}"}
    return {"success": False, "message": "Email failed. Check SMTP config in main.py."}


# ══════════════════════════════════════════════
# EXISTING VAPI TOOL ENDPOINTS (unchanged)
# ══════════════════════════════════════════════

@app.post("/tool/population")
def get_population(payload: dict):
    if df_pop is None:
        return {"result": "The population database is currently offline."}
    year = payload.get("year", 2017)
    data = df_pop[df_pop['Year'] == year]
    if data.empty:
        return {"result": f"No data available for the year {year}."}
    row = data.iloc[0]
    pop = round(row['India Population (Millions)'], 2)
    br  = round(row['Birth Rate (per 1000)'], 2)
    dr  = round(row['Death Rate (per 1000)'], 2)
    ur  = round(row['Urbanization_Rate'], 2)
    return {"result": f"In {year}, India's population was {pop} million. Birth rate: {br}/1000, death rate: {dr}/1000, urbanisation: {ur}%."}

@app.post("/tool/forest")
def get_forest(payload: dict):
    if df_forest is None:
        return {"result": "The forest database is currently offline."}
    state = payload.get("state", "Telangana").title()
    data  = df_forest[df_forest['State'].str.contains(state, case=False, na=False)]
    if data.empty:
        return {"result": f"No forest data found for {state}."}
    row = data.iloc[-1]
    return {"result": f"Forest area in {state}: {row['Total_Forest_Recorded_SqKm']} km² ({row['Forest_Percentage_Geographical']}% of geographical area)."}

@app.post("/tool/crop")
def get_crop(payload: dict):
    if df_crop is None:
        return {"result": "The crop yield database is currently offline."}
    state = payload.get("state", "Kerala").title()
    data  = df_crop[df_crop['State_Name'].str.contains(state, case=False, na=False)]
    if data.empty:
        return {"result": f"No crop data found for {state}."}
    avg   = round(data['Crop Yield (kg per hectare)'].mean(), 2)
    soil  = data['Soil_Type'].mode()[0] if 'Soil_Type' in data.columns else "N/A"
    crop  = data['Crop'].mode()[0] if 'Crop' in data.columns else "N/A"
    return {"result": f"In {state}: dominant crop is {crop}, soil type is {soil}, average yield is {avg} kg/ha."}

@app.post("/tool/drought")
def get_drought(payload: dict):
    if df_drought is None:
        return {"result": "The drought database is currently offline."}
    mean_spei = round(df_drought['Drought Index (SPEI)'].mean(), 2)
    avg_temp  = round(df_drought['Avg Temperature (°C)'].mean(), 2)
    return {"result": f"Average SPEI: {mean_spei}, average temperature: {avg_temp}°C."}

@app.post("/tool/disaster")
def predict_disaster(payload: dict):
    location = payload.get("location", "Mumbai").title()
    lat, lon  = geocode(location)
    if lat is None:
        return {"result": f"Cannot locate {location}."}
    try:
        data  = fetch_live_weather(lat, lon)
        cw    = data["current_weather"]
        temp  = round(cw["temperature"])
        wind  = round(cw["windspeed"])
        daily = data.get("daily", {})
        prec  = (daily.get("precipitation_sum") or [0])[0] or 0
        level = alert_level_from_indices(None, temp, wind, prec)
        label = {"green": "Stable", "yellow": "Elevated", "orange": "High", "red": "Critical"}[level]
        note  = ""
        if EVENT_MODEL:
            try:
                pred = EVENT_MODEL.predict(np.array([[temp, wind, prec]]))[0]
                note = f" ML model predicts: {pred}."
            except Exception:
                note = " (ML model feature mismatch — rule-based used.)"
        return {"result": f"{location}: {temp}°C, {wind} km/h wind, {prec} mm precip. Risk: {label}.{note}"}
    except Exception as e:
        return {"result": f"Weather fetch failed for {location}: {e}"}