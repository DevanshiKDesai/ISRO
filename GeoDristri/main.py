"""
GeoDrishti FastAPI Backend — main.py v4.0
Smart chatbot: tries related data, never just refuses flatly.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd, numpy as np, joblib, requests, re, os, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

app = FastAPI(title="GeoDrishti API", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER",     "your_email@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_app_password")
ALERT_FROM    = os.getenv("ALERT_FROM",    "geodrishti@isro.gov.in")

print("Loading GeoDrishti Datasets...")
df_pop = df_forest = df_crop = df_drought = None

try:
    df_pop    = pd.read_csv("Population/india_enriched.csv", encoding='latin1');         print("✅ Population")
except Exception as e: print(f"⚠️  Population: {e}")
try:
    df_forest = pd.read_csv("Forest_prediction/New Forest.csv", encoding='latin1');      print("✅ Forest")
except Exception as e: print(f"⚠️  Forest: {e}")
try:
    df_crop   = pd.read_csv("crop_predictor/enhanced_crop_yield_dataset (1).csv", encoding='latin1'); print("✅ Crop")
except Exception as e: print(f"⚠️  Crop: {e}")
try:
    df_drought = pd.read_excel("drought_prediction/Drought New.xlsx", engine='openpyxl'); print("✅ Drought")
except Exception:
    try: df_drought = pd.read_excel("drought_prediction/Drought New.xlsx", engine='xlrd'); print("✅ Drought(xlrd)")
    except Exception:
        try: df_drought = pd.read_csv("drought_prediction/Drought New.xlsx", encoding='latin1', on_bad_lines='skip', sep=None, engine='python'); print("✅ Drought(csv)")
        except Exception as e: print(f"⚠️  Drought: {e}")

EVENT_MODEL = None
try:
    EVENT_MODEL = joblib.load("Weather_Prediction/event_model.joblib"); print("✅ Event Model")
except Exception as e: print(f"⚠️  Event model: {e}")

# Load additional ML models for predictions
CROP_MODEL = FOREST_MODELS = DROUGHT_MODELS = POP_MODELS = None
try:
    # Crop prediction
    CROP_MODEL = {
        'model': joblib.load("crop_predictor/model.joblib"),
        'encoders': joblib.load("crop_predictor/encoders.joblib"),
        'target_encoder': joblib.load("crop_predictor/target_encoder.joblib"),
        'feature_cols': joblib.load("crop_predictor/feature_cols.joblib")
    }; print("✅ Crop Model")
except Exception as e: print(f"⚠️  Crop model: {e}")

try:
    # Forest prediction
    FOREST_MODELS = {
        'alert': joblib.load("Forest_prediction/alert_model.joblib"),
        'ndvi': joblib.load("Forest_prediction/ndvi_model.joblib"),
        'cover': joblib.load("Forest_prediction/cover_model.joblib"),
        'aqi': joblib.load("Forest_prediction/aqi_model.joblib"),
        'human': joblib.load("Forest_prediction/human_model.joblib"),
        'state_encoder': joblib.load("Forest_prediction/state_encoder.joblib"),
        'feature_cols': joblib.load("Forest_prediction/feature_cols.joblib")
    }; print("✅ Forest Models")
except Exception as e: print(f"⚠️  Forest models: {e}")

try:
    # Drought prediction
    DROUGHT_MODELS = {
        'category': joblib.load("drought_prediction/category_model.joblib"),
        'status': joblib.load("drought_prediction/status_model.joblib"),
        'category_encoder': joblib.load("drought_prediction/category_encoder.joblib"),
        'feature_cols': joblib.load("drought_prediction/feature_cols.joblib")
    }; print("✅ Drought Models")
except Exception as e: print(f"⚠️  Drought models: {e}")

try:
    # Population/Urbanization prediction
    POP_MODELS = {
        'pop': joblib.load("Population/pop_model.joblib"),
        'urb': joblib.load("Population/urb_model.joblib"),
        'infra': joblib.load("Population/infra_model.joblib"),
        'upop': joblib.load("Population/upop_model.joblib"),
        'grow': joblib.load("Population/grow_model.joblib"),
        'feature_cols': joblib.load("Population/feature_cols.joblib")
    }; print("✅ Population Models")
except Exception as e: print(f"⚠️  Population models: {e}")

class ChatRequest(BaseModel):
    message: str

class AOIRequest(BaseModel):
    lat: float; lng: float
    north: float; south: float; east: float; west: float
    aoi_name: str = "AOI"
    user_email: Optional[str] = None

class AlertEmailRequest(BaseModel):
    to_email: str; aoi_name: str; level: str; summary: str; lat: float; lng: float

def geocode(loc):
    try:
        r = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={loc}&count=1", timeout=6).json()
        if "results" not in r or not r["results"]: return None, None
        return r["results"][0]["latitude"], r["results"][0]["longitude"]
    except: return None, None

def live_weather(lat, lon):
    return requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current_weather=true&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
        f"&timezone=Asia%2FKolkata&forecast_days=1", timeout=8).json()

def ndvi_proxy(lat, lon):
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now()-timedelta(days=16)).strftime("%Y%m%d")
        r = requests.get(
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN,PRECTOTCORR&community=AG"
            f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON", timeout=12).json()
        p = r["properties"]["parameter"]
        srad = [v for v in p["ALLSKY_SFC_SW_DWN"].values() if v!=-999]
        prec = [v for v in p["PRECTOTCORR"].values() if v!=-999]
        return round(min(max(float(np.nanmean(prec) if prec else 2)*0.04 + float(np.nanmean(srad) if srad else 4)*0.005, 0),1),3)
    except: return None

def alert_score(ndvi, temp, wind, precip):
    s=0
    if ndvi is not None:
        if ndvi<0.15: s+=3
        elif ndvi<0.3: s+=1
    if temp is not None:
        if temp>45: s+=3
        elif temp>40: s+=2
        elif temp>37: s+=1
        if temp<5: s+=2
    if wind is not None:
        if wind>80: s+=3
        elif wind>55: s+=2
        elif wind>35: s+=1
    if precip is not None:
        if precip>80: s+=2
        elif precip>50: s+=1
    if s>=6: return "red"
    if s>=4: return "orange"
    if s>=2: return "yellow"
    return "green"

def send_alert_email(to_email, aoi_name, level, summary, coords):
    c = {"yellow":"#f59e0b","orange":"#f97316","red":"#ef4444"}.get(level,"#10b981")
    html = f"""<html><body style="font-family:sans-serif;background:#060a14;color:#e8edf5;padding:32px;">
      <div style="max-width:520px;margin:auto;background:#0d1526;border-radius:12px;border:2px solid {c};padding:28px;">
        <h2 style="color:{c};margin-top:0;">⚠️ GeoDrishti Alert — {level.upper()}</h2>
        <p><strong>AOI:</strong> {aoi_name}</p><p><strong>Coordinates:</strong> {coords.get('lat')}, {coords.get('lng')}</p>
        <p><strong>Summary:</strong><br>{summary}</p>
        <p style="color:#6b7fa3;font-size:12px;margin-top:24px;">ISRO GeoDrishti EcoSight · {datetime.utcnow().isoformat()}Z</p>
      </div></body></html>"""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[GeoDrishti] {level.upper()} Alert — {aoi_name}"
        msg["From"] = ALERT_FROM; msg["To"] = to_email
        msg.attach(MIMEText(html,"html"))
        with smtplib.SMTP(SMTP_HOST,SMTP_PORT) as s:
            s.ehlo(); s.starttls(); s.login(SMTP_USER,SMTP_PASSWORD)
            s.sendmail(ALERT_FROM,to_email,msg.as_string())
        return True
    except Exception as e: print(f"Email failed: {e}"); return False

# ── ML Prediction Functions ──────────────────────────────────────────────────

def predict_crop(lat, lon):
    if not CROP_MODEL: return {"error": "Crop model not loaded"}
    try:
        # Get weather data
        weather = live_weather(lat, lon)
        cw = weather.get("current_weather", {})
        temp = cw.get("temperature", 25)
        wind = cw.get("windspeed", 10)
        daily = weather.get("daily", {})
        precip = sum(daily.get("precipitation_sum", [0])) or 0

        # Get altitude (simplified)
        altitude = 350  # fallback

        # Get soil (simplified)
        soil = {"pH": 6.8, "Organic_Carbon": 0.8, "Soil_Moisture": 38}

        # Get state (simplified reverse geocode)
        state = "Maharashtra"  # fallback

        # Season
        from datetime import datetime
        month = datetime.now().month
        season = "Kharif" if month in [6,7,8,9,10] else "Rabi"

        features = {
            "State_Name": state, "Season": season,
            "N": 68, "P": 53, "K": 78,
            "Soil_Type": "Neutral", "Irrigation_Method": "Rainfed", "Soil_Texture": "Loamy",
            "Fertilizer_Used_kg": 120, "Pesticide_Usage_kg": 10,
            "temperature": temp, "rainfall": precip * 365 / 7, "humidity": 60,
            "Wind_speed": wind, "Sunshine_hours": 8,
            "Altitude_m": altitude, **soil
        }

        # Encode and predict
        for col in ['State_Name', 'Season', 'Soil_Type', 'Irrigation_Method', 'Soil_Texture']:
            le = CROP_MODEL['encoders'][col]
            val = features.get(col, le.classes_[0])
            features[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

        X = np.array([[features.get(col, 0) for col in CROP_MODEL['feature_cols']]])
        probs = CROP_MODEL['model'].predict_proba(X)[0]
        top5_idx = np.argsort(probs)[::-1][:5]
        predictions = []
        for idx in top5_idx:
            crop = CROP_MODEL['target_encoder'].inverse_transform([idx])[0]
            confidence = round(probs[idx] * 100, 1)
            predictions.append({"crop": crop, "confidence": confidence})
        return {"predictions": predictions}
    except Exception as e: return {"error": str(e)}

def predict_forest(lat, lon):
    if not FOREST_MODELS: return {"error": "Forest models not loaded"}
    try:
        # Simplified features
        features = {"lat": lat, "lon": lon, "temperature": 25, "precipitation": 50, "population_density": 100}
        X = np.array([[features.get(col, 0) for col in FOREST_MODELS['feature_cols']]])

        alert_pred = FOREST_MODELS['alert'].predict(X)[0]
        ndvi_pred = FOREST_MODELS['ndvi'].predict(X)[0]
        cover_pred = FOREST_MODELS['cover'].predict(X)[0]
        aqi_pred = FOREST_MODELS['aqi'].predict(X)[0]
        human_pred = FOREST_MODELS['human'].predict(X)[0]

        return {
            "alert_level": int(alert_pred),
            "future_ndvi": round(float(ndvi_pred), 3),
            "future_cover_sqkm": round(float(cover_pred), 2),
            "aqi_impact": round(float(aqi_pred), 2),
            "human_impact": round(float(human_pred), 2)
        }
    except Exception as e: return {"error": str(e)}

def predict_drought(lat, lon):
    if not DROUGHT_MODELS: return {"error": "Drought models not loaded"}
    try:
        # Get weather
        weather = live_weather(lat, lon)
        cw = weather.get("current_weather", {})
        temp = cw.get("temperature", 25)
        wind = cw.get("windspeed", 10)
        daily = weather.get("daily", {})
        precip = sum(daily.get("precipitation_sum", [0])) or 0

        features = {"temperature": temp, "humidity": 60, "precipitation": precip, "wind_speed": wind, "solar_radiation": 200}
        X = np.array([[features.get(col, 0) for col in DROUGHT_MODELS['feature_cols']]])

        category_pred = DROUGHT_MODELS['category'].predict(X)[0]
        status_pred = DROUGHT_MODELS['status'].predict(X)[0]

        category_label = DROUGHT_MODELS['category_encoder'].inverse_transform([category_pred])[0]

        return {
            "category": category_label,
            "status": bool(status_pred)
        }
    except Exception as e: return {"error": str(e)}

def predict_population(lat, lon):
    if not POP_MODELS: return {"error": "Population models not loaded"}
    try:
        # Simplified features
        features = {"lat": lat, "lon": lon, "current_population": 10, "urban_rate": 30, "year": 2023}
        X = np.array([[features.get(col, 0) for col in POP_MODELS['feature_cols']]])

        pop_pred = POP_MODELS['pop'].predict(X)[0]
        urb_pred = POP_MODELS['urb'].predict(X)[0]
        infra_pred = POP_MODELS['infra'].predict(X)[0]
        upop_pred = POP_MODELS['upop'].predict(X)[0]
        grow_pred = POP_MODELS['grow'].predict(X)[0]

        return {
            "future_population_millions": round(float(pop_pred), 2),
            "future_urbanization_rate": round(float(urb_pred), 2),
            "future_urban_population_millions": round(float(upop_pred), 2),
            "infrastructure_pressure": round(float(infra_pred), 2),
            "growth_rate": round(float(grow_pred), 2)
        }
    except Exception as e: return {"error": str(e)}

INTENT_MAP = {
    "weather":    ["weather","temperature","rain","forecast","wind","cyclone","heatwave","cold","hot","humid","climate"],
    "disaster":   ["disaster","flood","risk","calamity","earthquake","landslide","storm"],
    "forest":     ["forest","tree","deforestation","cover","woodland","green cover","jungle"],
    "crop":       ["crop","yield","agriculture","farm","wheat","rice","harvest","soil","kharif","rabi","cultivat","sowing"],
    "drought":    ["drought","spei","dry","water stress","rainfall deficit","arid","moisture"],
    "population": ["population","people","census","birth","death","urbanization","demographic"],
    "ndvi":       ["ndvi","vegetation index","ndwi","ndbi","satellite index","vegetation health","plant health","greenness"],
}
INDIAN_PLACES = {
    "Mumbai","Delhi","Bangalore","Bengaluru","Chennai","Hyderabad","Kolkata","Pune","Ahmedabad","Jaipur",
    "Lucknow","Kerala","Rajasthan","Maharashtra","Gujarat","Punjab","Haryana","Odisha","Assam","Telangana",
    "Karnataka","Bihar","Uttarakhand","Andhra","Sikkim","Meghalaya","Manipur","Nagaland","Tripura","Goa",
    "Jharkhand","Chhattisgarh","Himachal","Jammu","Kashmir","Ladakh","Chandigarh","Surat","Nagpur","Indore",
    "Bhopal","Patna","Vadodara","Coimbatore","Kochi","Visakhapatnam","Agra","Varanasi","Srinagar","Amritsar",
    "Jodhpur","Udaipur","Mysuru","Nashik","Aurangabad","Ranchi","Guwahati","Shillong","Imphal",
}

def classify(msg):
    m = msg.lower()
    matched = [i for i,kws in INTENT_MAP.items() if any(k in m for k in kws)]
    return matched if matched else ["unknown"]

def extract_loc(msg):
    caps = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', msg)
    for c in caps:
        if c in INDIAN_PLACES: return c
    lp = {p.lower():p for p in INDIAN_PLACES}
    for w in re.findall(r'\b[a-z]{4,}\b', msg.lower()):
        if w in lp: return lp[w]
    return caps[0] if caps else "Mumbai"

def extract_year(msg):
    yrs = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', msg)
    return int(yrs[0]) if yrs else 2017

WMO = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",51:"Light drizzle",
       61:"Slight rain",63:"Moderate rain",71:"Light snow",80:"Rain showers",95:"Thunderstorm"}

def chat_weather(msg):
    loc = extract_loc(msg)
    lat,lon = geocode(loc)
    if lat is None: return f"Couldn't locate **{loc}**. Try: 'weather in Mumbai' or 'weather in Delhi'."
    try:
        d = live_weather(lat,lon); cw = d["current_weather"]
        temp=cw["temperature"]; wind=cw["windspeed"]; code=cw["weathercode"]
        daily=d.get("daily",{}); maxt=(daily.get("temperature_2m_max") or [None])[0]
        mint=(daily.get("temperature_2m_min") or [None])[0]; prec=(daily.get("precipitation_sum") or [None])[0]
        level=alert_score(None,temp,wind,prec or 0)
        flag={"green":"🟢 Stable","yellow":"🟡 Heads Up","orange":"🟠 Warning","red":"🔴 Critical"}[level]
        out=[f"**Live weather — {loc}** (Open-Meteo, hourly):",
             f"• Condition: **{WMO.get(code,f'Code {code}')}**",
             f"• Temperature: **{temp}°C** | Wind: **{wind} km/h**"]
        if maxt and mint: out.append(f"• Range today: {mint}°C – {maxt}°C")
        if prec is not None: out.append(f"• Precipitation: {prec} mm")
        out.append(f"\nRisk: **{flag}**")
        if level in ("orange","red"): out.append("📱 Check imd.gov.in for official alerts.")
        return "\n".join(out)
    except Exception as e: return f"Weather fetch failed for {loc}. ({e})"

def chat_disaster(msg):
    loc = extract_loc(msg)
    lat,lon = geocode(loc)
    if lat is None: return f"Couldn't locate **{loc}**. Try: 'disaster risk for Chennai'."
    try:
        d=live_weather(lat,lon); cw=d["current_weather"]; temp=cw["temperature"]; wind=cw["windspeed"]
        prec=(d.get("daily",{}).get("precipitation_sum") or [0])[0] or 0
        level=alert_score(None,temp,wind,prec)
        label={"green":"🟢 Stable","yellow":"🟡 Elevated","orange":"🟠 High","red":"🔴 Critical"}[level]
        out=[f"**Disaster risk — {loc}** (live data):",
             f"• {temp}°C | Wind {wind} km/h | Precip {prec} mm",
             f"• Risk level: **{label}**"]
        if wind>60: out.append("• ⚠️ High wind — cyclonic risk possible.")
        if temp>42: out.append("• ⚠️ Heatwave threshold exceeded.")
        if prec>50: out.append("• ⚠️ Heavy rain — flood risk.")
        if level=="green": out.append("• No immediate hazards from current data.")
        if EVENT_MODEL:
            try:
                pred=EVENT_MODEL.predict(np.array([[temp,wind,prec]]))[0]
                out.append(f"• **ML model (event_model.joblib):** {pred}")
            except: out.append("• ML model: feature mismatch — rule-based used.")
        else: out.append("• ML model not loaded.")
        out.append("\n📌 Official: ndma.gov.in | imd.gov.in")
        return "\n".join(out)
    except Exception as e: return f"Live data unavailable for {loc}. ({e})"

def chat_forest(msg):
    loc = extract_loc(msg)
    lat, lon = geocode(loc)
    if lat is None: return f"Couldn't locate **{loc}**. Try: 'forest cover in Maharashtra'."
    pred = predict_forest(lat, lon)
    if 'error' in pred:
        return f"Forest prediction failed: {pred['error']}\nFallback: Check fsi.nic.in for static reports."
    return (f"**Forest prediction — {loc}** (ML model):\n"
            f"• Alert Level: **{['No Alert','Mild','Severe','Critical'][pred.get('alert_level',0)]}**\n"
            f"• Future NDVI: **{pred.get('future_ndvi', 'N/A')}**\n"
            f"• Future Cover: **{pred.get('future_cover_sqkm', 'N/A')} km²**\n"
            f"• AQI Impact: **{pred.get('aqi_impact', 'N/A')}**\n"
            f"• Human Impact: **{pred.get('human_impact', 'N/A')}**\n"
            f"• Source: Trained forest ML models.\n• Live data: fsi.nic.in")

def chat_crop(msg):
    loc = extract_loc(msg)
    lat, lon = geocode(loc)
    if lat is None: return f"Couldn't locate **{loc}**. Try: 'crop yield in Punjab'."
    pred = predict_crop(lat, lon)
    if 'error' in pred:
        return f"Crop prediction failed: {pred['error']}\nFallback: Check agmarknet.gov.in for prices."
    crops = pred.get('predictions', [])
    if not crops: return "No crop predictions available."
    top3 = crops[:3]
    return (f"**Crop prediction — {loc}** (ML model):\n" +
            "\n".join(f"• **{c['crop']}** ({c['confidence']}%)" for c in top3) +
            f"\n• Source: Trained crop yield ML model.\n• Live prices: agmarknet.gov.in")

def chat_drought(msg):
    loc = extract_loc(msg)
    lat, lon = geocode(loc)
    if lat is None: return f"Couldn't locate **{loc}**. Try: 'drought in Rajasthan'."
    pred = predict_drought(lat, lon)
    if 'error' in pred:
        return f"Drought prediction failed: {pred['error']}\nFallback: Check imd.gov.in for bulletins."
    return (f"**Drought prediction — {loc}** (ML model):\n"
            f"• Category: **{pred.get('category', 'N/A')}**\n"
            f"• Status: **{'Drought' if pred.get('status', False) else 'Normal'}**\n"
            f"• Source: Trained drought ML models.\n• Current: imd.gov.in | nrsc.gov.in")

def chat_population(msg):
    loc = extract_loc(msg)
    lat, lon = geocode(loc)
    if lat is None: return f"Couldn't locate **{loc}**. Try: 'population in Delhi'."
    pred = predict_population(lat, lon)
    if 'error' in pred:
        return f"Population prediction failed: {pred['error']}\nFallback: Check censusindia.gov.in."
    return (f"**Urbanization prediction — {loc}** (ML model):\n"
            f"• Future Population: **{pred.get('future_population_millions', 'N/A')}M**\n"
            f"• Future Urban Rate: **{pred.get('future_urbanization_rate', 'N/A')}%**\n"
            f"• Future Urban Pop: **{pred.get('future_urban_population_millions', 'N/A')}M**\n"
            f"• Infrastructure Pressure: **{pred.get('infrastructure_pressure', 'N/A')}/100**\n"
            f"• Growth Rate: **{pred.get('growth_rate', 'N/A')}%**\n"
            f"• Source: Trained population ML models.\n• Census: censusindia.gov.in")

def chat_ndvi(msg):
    loc=extract_loc(msg)
    lat,lon=geocode(loc)
    if lat is None: return f"Couldn't locate **{loc}**. Try: 'NDVI for Kerala'."
    val=ndvi_proxy(lat,lon)
    if val is None:
        return ("NASA POWER unreachable.\nFor real NDVI: **Bhuvan ISRO** (bhuvan.nrsc.gov.in) "
                "has free Sentinel-2 layers for India.")
    interp=("🌿 Dense/healthy veg" if val>0.6 else "🌱 Moderate veg" if val>0.3
            else "🟡 Sparse/stressed" if val>0.1 else "🏜️ Bare soil/urban")
    return (f"**Vegetation — {loc}** (NASA POWER 16-day):\n"
            f"• NDVI proxy: **{val}** — {interp}\n"
            f"• Scale: 0=bare, 0.2=sparse, 0.5=moderate, 0.8+=dense forest\n"
            f"• ⚠️ Proxy only — true NDVI needs Sentinel-2.\n"
            f"• True NDVI: bhuvan.nrsc.gov.in | apps.sentinel-hub.com")

def smart_fallback(msg):
    loc=extract_loc(msg); has_loc=any(p.lower() in msg.lower() for p in INDIAN_PLACES) or bool(re.findall(r'\b[A-Z][a-z]{3,}\b',msg))
    parts=[]
    if has_loc:
        parts.append(f"I found a location (**{loc}**) — here's what I have:\n")
        try:
            lat,lon=geocode(loc)
            if lat: parts.append(chat_weather(msg))
        except: pass
    avail=[f"✅ {n}" for n,df in [("Forest (by state)",df_forest),("Crop (by state)",df_crop),("Drought/SPEI",df_drought),("Population (by year)",df_pop)] if df is not None]
    avail+=["✅ Live weather (any Indian city)","✅ Disaster risk","✅ NDVI proxy"]
    if not parts: parts.append("I didn't quite understand that. Here's what I can help with:\n")
    parts.append("**Available data:**\n"+"\n".join(f"• {s}" for s in avail))
    parts.append("\n**Try asking:**\n• 'Weather in Pune'\n• 'Forest cover in Assam'\n• 'Crop yield for Bihar'\n• 'Drought analysis'\n• 'Population 2010'\n• 'NDVI for Gujarat'")
    return "\n".join(parts)

DISPATCH = {"weather":chat_weather,"disaster":chat_disaster,"forest":chat_forest,
            "crop":chat_crop,"drought":chat_drought,"population":chat_population,"ndvi":chat_ndvi}

class VAPIRequest(BaseModel):
    message: dict  # {"role": "user", "content": "message"}
    call: Optional[dict] = None

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    msg=req.message.strip(); intents=classify(msg)
    if intents==["unknown"]: return {"reply":smart_fallback(msg),"intent":"unknown"}
    if len(intents)==1: reply=DISPATCH[intents[0]](msg)
    else: reply="\n\n---\n\n".join(DISPATCH[i](msg) for i in intents[:2])
    return {"reply":reply,"intent":intents[0]}

@app.post("/vapi/webhook")
def vapi_webhook(req: VAPIRequest):
    user_msg = req.message.get("content", "").strip()
    if not user_msg:
        return {"result": "I didn't catch that. Could you please repeat?"}
    
    # Use the same chat logic as the web chat
    intents = classify(user_msg)
    if intents == ["unknown"]:
        reply = smart_fallback(user_msg)
    elif len(intents) == 1:
        reply = DISPATCH[intents[0]](user_msg)
    else:
        reply = "\n\n---\n\n".join(DISPATCH[i](user_msg) for i in intents[:2])
    
    return {"result": reply}

@app.post("/aoi/analyze")
def analyze_aoi(req: AOIRequest):
    lat,lon=req.lat,req.lng
    try:
        w=live_weather(lat,lon); cw=w["current_weather"]
        temp=cw["temperature"]; wind=cw["windspeed"]
        prec=(w.get("daily",{}).get("precipitation_sum") or [0])[0] or 0
    except Exception as e: return {"error":f"Weather API unavailable: {e}"}
    ndvi=ndvi_proxy(lat,lon)
    ndwi=round(max(-0.5,min(0.5,(prec/200)-0.3)),3)
    ndbi=round(max(-0.5,min(0.5,0.4-(ndvi or 0.3))),3)
    level=alert_score(ndvi,temp,wind,prec)
    summaries={"green":"🟢 Normal — no significant change detected.",
               "yellow":"🟡 Heads Up — early signs of change. Monitor regularly.",
               "orange":"🟠 Warning — notable environmental stress. Action recommended.",
               "red":"🔴 Critical — severe conditions. Immediate intervention needed."}
    summary=summaries[level]; email_sent=False
    if req.user_email and level!="green":
        email_sent=send_alert_email(req.user_email,req.aoi_name,level,summary,{"lat":lat,"lng":lon})

    # Get ML predictions
    crop_pred = predict_crop(lat, lon)
    forest_pred = predict_forest(lat, lon)
    drought_pred = predict_drought(lat, lon)
    pop_pred = predict_population(lat, lon)

    return {"lat":lat,"lng":lon,"aoi_name":req.aoi_name,"ndvi":ndvi,"ndwi":ndwi,"ndbi":ndbi,
            "temperature":temp,"wind_speed":wind,"precipitation":prec,"alert_level":level,
            "summary":summary,"email_sent":email_sent,"timestamp":datetime.utcnow().isoformat()+"Z",
            "data_sources":["Open-Meteo (live)","NASA POWER (16-day proxy)"],
            "predictions": {
                "crop": crop_pred,
                "forest": forest_pred,
                "drought": drought_pred,
                "population": pop_pred
            }}

@app.post("/alert/email")
def email_alert(req: AlertEmailRequest):
    ok=send_alert_email(req.to_email,req.aoi_name,req.level,req.summary,{"lat":req.lat,"lng":req.lng})
    return {"success":ok,"message":f"Alert sent to {req.to_email}" if ok else "Email failed — set SMTP_USER + SMTP_PASSWORD env vars."}

@app.post("/tool/population")
def get_population(payload: dict):
    if df_pop is None: return {"result":"Population database offline."}
    year=payload.get("year",2017); data=df_pop[df_pop['Year']==year]
    if data.empty: return {"result":f"No data for {year}."}
    r=data.iloc[0]
    return {"result":f"In {year}: {round(r['India Population (Millions)'],2)}M people, birth {round(r['Birth Rate (per 1000)'],2)}/1000, urban {round(r['Urbanization_Rate'],2)}%."}

@app.post("/tool/forest")
def get_forest(payload: dict):
    if df_forest is None: return {"result":"Forest database offline."}
    state=payload.get("state","Telangana").title()
    rows=df_forest[df_forest['State'].str.contains(state,case=False,na=False)]
    if rows.empty: return {"result":f"No forest data for {state}."}
    r=rows.iloc[-1]
    return {"result":f"{state}: {r['Total_Forest_Recorded_SqKm']} km² ({r['Forest_Percentage_Geographical']}% of area)."}

@app.post("/tool/crop")
def get_crop(payload: dict):
    if df_crop is None: return {"result":"Crop database offline."}
    state=payload.get("state","Kerala").title()
    rows=df_crop[df_crop['State_Name'].str.contains(state,case=False,na=False)]
    if rows.empty: return {"result":f"No crop data for {state}."}
    avg=round(rows['Crop Yield (kg per hectare)'].mean(),2)
    soil=rows['Soil_Type'].mode()[0] if 'Soil_Type' in rows.columns else "N/A"
    crop=rows['Crop'].mode()[0]      if 'Crop'      in rows.columns else "N/A"
    return {"result":f"{state}: top crop {crop}, {soil} soil, avg {avg} kg/ha."}

@app.post("/tool/drought")
def get_drought(payload: dict):
    if df_drought is None: return {"result":"Drought database offline."}
    return {"result":f"Mean SPEI: {round(df_drought['Drought Index (SPEI)'].mean(),2)}, mean temp: {round(df_drought['Avg Temperature (°C)'].mean(),2)}°C."}

@app.post("/tool/disaster")
def predict_disaster(payload: dict):
    loc=payload.get("location","Mumbai").title(); lat,lon=geocode(loc)
    if lat is None: return {"result":f"Cannot locate {loc}."}
    try:
        d=live_weather(lat,lon); cw=d["current_weather"]
        temp=round(cw["temperature"]); wind=round(cw["windspeed"])
        prec=(d.get("daily",{}).get("precipitation_sum") or [0])[0] or 0
        lvl=alert_score(None,temp,wind,prec)
        lbl={"green":"Stable","yellow":"Elevated","orange":"High","red":"Critical"}[lvl]
        note=""
        if EVENT_MODEL:
            try: note=f" ML: {EVENT_MODEL.predict(np.array([[temp,wind,prec]]))[0]}."
            except: note=" (ML mismatch.)"
        return {"result":f"{loc}: {temp}°C, {wind}km/h, {prec}mm. Risk: {lbl}.{note}"}
    except Exception as e: return {"result":f"Uplink failed: {e}"}