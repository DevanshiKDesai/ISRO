"""
GeoDrishti FastAPI Backend — main.py
Production build for Hugging Face Spaces (port 7860)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd, numpy as np, joblib, requests, re, os, smtplib, glob
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

app = FastAPI(title="GeoDrishti API", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER",     "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM    = os.getenv("ALERT_FROM",    "geodrishti@isro.gov.in")

print("="*50)
print("GeoDrishti: Loading datasets...")

df_pop = df_forest = df_crop = df_drought = None

try:
    df_pop = pd.read_csv("Population/india_enriched.csv", encoding='latin1')
    print(f"Population: {len(df_pop)} rows")
except Exception as e: print(f"Population FAILED: {e}")

try:
    df_forest = pd.read_csv("Forest_prediction/New Forest.csv", encoding='latin1')
    print(f"Forest: {len(df_forest)} rows")
except Exception as e: print(f"Forest FAILED: {e}")

try:
    df_crop = pd.read_csv("crop_predictor/enhanced_crop_yield_dataset (1).csv", encoding='latin1')
    print(f"Crop: {len(df_crop)} rows")
except Exception as e: print(f"Crop FAILED: {e}")

for eng in ['openpyxl','xlrd']:
    try:
        df_drought = pd.read_excel("drought_prediction/Drought New.xlsx", engine=eng)
        print(f"Drought ({eng}): {len(df_drought)} rows"); break
    except Exception: pass
if df_drought is None:
    try:
        df_drought = pd.read_csv("drought_prediction/Drought New.xlsx", encoding='latin1', on_bad_lines='skip', sep=None, engine='python')
        print(f"Drought (csv fallback): {len(df_drought)} rows")
    except Exception as e: print(f"Drought FAILED: {e}")

EVENT_MODEL = None
candidates = ["Weather_Prediction/event_model.joblib","Weather_Prediction/weather_model.joblib","Weather_Prediction/disaster_model.joblib","Weather_Prediction/model.joblib"]
for path in candidates:
    try: EVENT_MODEL = joblib.load(path); print(f"Model: {path}"); break
    except: pass
if EVENT_MODEL is None:
    found = glob.glob("Weather_Prediction/*.joblib")
    if found:
        try: EVENT_MODEL = joblib.load(found[0]); print(f"Model (glob): {found[0]}")
        except Exception as e: print(f"Model FAILED: {e}")
    else: print("No .joblib found in Weather_Prediction/")

print(f"Status: pop={df_pop is not None} forest={df_forest is not None} crop={df_crop is not None} drought={df_drought is not None} model={EVENT_MODEL is not None}")
print("="*50)

class ChatRequest(BaseModel):
    message: str

class AOIRequest(BaseModel):
    lat: float; lng: float; north: float; south: float; east: float; west: float
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
        end=datetime.now().strftime("%Y%m%d"); start=(datetime.now()-timedelta(days=16)).strftime("%Y%m%d")
        r=requests.get(f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON",timeout=12).json()
        p=r["properties"]["parameter"]
        srad=[v for v in p["ALLSKY_SFC_SW_DWN"].values() if v!=-999]
        prec=[v for v in p["PRECTOTCORR"].values() if v!=-999]
        return round(min(max(float(np.nanmean(prec) if prec else 2)*0.04+float(np.nanmean(srad) if srad else 4)*0.005,0),1),3)
    except: return None

def alert_score(ndvi,temp,wind,precip):
    s=0
    if ndvi is not None:
        if ndvi<0.15:s+=3
        elif ndvi<0.3:s+=1
    if temp is not None:
        if temp>45:s+=3
        elif temp>40:s+=2
        elif temp>37:s+=1
        if temp<5:s+=2
    if wind is not None:
        if wind>80:s+=3
        elif wind>55:s+=2
        elif wind>35:s+=1
    if precip is not None:
        if precip>80:s+=2
        elif precip>50:s+=1
    if s>=6:return"red"
    if s>=4:return"orange"
    if s>=2:return"yellow"
    return"green"

def send_alert_email(to_email,aoi_name,level,summary,coords):
    if not SMTP_USER or not SMTP_PASSWORD: print("Email skipped: secrets not set."); return False
    c={"yellow":"#f59e0b","orange":"#f97316","red":"#ef4444"}.get(level,"#10b981")
    html=f"""<html><body style="font-family:sans-serif;background:#060a14;color:#e8edf5;padding:32px;"><div style="max-width:520px;margin:auto;background:#0d1526;border-radius:12px;border:2px solid {c};padding:28px;"><h2 style="color:{c};margin-top:0;">GeoDrishti Alert - {level.upper()}</h2><p><strong>AOI:</strong> {aoi_name}</p><p><strong>Coordinates:</strong> {coords.get('lat')}, {coords.get('lng')}</p><p><strong>Summary:</strong><br>{summary}</p><p style="color:#6b7fa3;font-size:12px;margin-top:24px;">ISRO GeoDrishti EcoSight - {datetime.utcnow().isoformat()}Z</p></div></body></html>"""
    try:
        msg=MIMEMultipart("alternative"); msg["Subject"]=f"[GeoDrishti] {level.upper()} Alert - {aoi_name}"; msg["From"]=ALERT_FROM; msg["To"]=to_email
        msg.attach(MIMEText(html,"html"))
        with smtplib.SMTP(SMTP_HOST,SMTP_PORT) as s:
            s.ehlo();s.starttls();s.login(SMTP_USER,SMTP_PASSWORD);s.sendmail(ALERT_FROM,to_email,msg.as_string())
        return True
    except Exception as e: print(f"Email failed: {e}"); return False

INTENT_MAP={"weather":["weather","temperature","rain","forecast","wind","cyclone","heatwave","cold","hot","humid","climate"],"disaster":["disaster","flood","risk","calamity","earthquake","landslide","storm"],"forest":["forest","tree","deforestation","cover","woodland","green cover","jungle"],"crop":["crop","yield","agriculture","farm","wheat","rice","harvest","soil","kharif","rabi","sowing"],"drought":["drought","spei","dry","water stress","rainfall deficit","arid","moisture"],"population":["population","people","census","birth","death","urbanization","demographic"],"ndvi":["ndvi","vegetation index","ndwi","ndbi","satellite index","vegetation health","greenness"]}
INDIAN_PLACES={"Mumbai","Delhi","Bangalore","Bengaluru","Chennai","Hyderabad","Kolkata","Pune","Ahmedabad","Jaipur","Lucknow","Kerala","Rajasthan","Maharashtra","Gujarat","Punjab","Haryana","Odisha","Assam","Telangana","Karnataka","Bihar","Uttarakhand","Andhra","Sikkim","Meghalaya","Manipur","Nagaland","Tripura","Goa","Jharkhand","Chhattisgarh","Himachal","Jammu","Kashmir","Ladakh","Chandigarh","Surat","Nagpur","Indore","Bhopal","Patna","Vadodara","Coimbatore","Kochi","Visakhapatnam","Agra","Varanasi","Srinagar","Amritsar","Jodhpur","Udaipur","Mysuru","Nashik","Aurangabad","Ranchi","Guwahati","Shillong","Imphal"}
WMO={0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",51:"Light drizzle",61:"Slight rain",63:"Moderate rain",71:"Light snow",80:"Rain showers",95:"Thunderstorm"}

def classify(msg):
    m=msg.lower(); matched=[i for i,kws in INTENT_MAP.items() if any(k in m for k in kws)]; return matched if matched else ["unknown"]

def extract_loc(msg):
    caps=re.findall(r'\b[A-Z][a-zA-Z]{2,}\b',msg)
    for c in caps:
        if c in INDIAN_PLACES:return c
    lp={p.lower():p for p in INDIAN_PLACES}
    for w in re.findall(r'\b[a-z]{4,}\b',msg.lower()):
        if w in lp:return lp[w]
    return caps[0] if caps else "Mumbai"

def extract_year(msg):
    yrs=re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b',msg); return int(yrs[0]) if yrs else 2017

def chat_weather(msg):
    loc=extract_loc(msg); lat,lon=geocode(loc)
    if lat is None:return f"Couldn't locate **{loc}**. Try: 'weather in Mumbai'."
    try:
        d=live_weather(lat,lon); cw=d["current_weather"]; temp=cw["temperature"]; wind=cw["windspeed"]; code=cw["weathercode"]
        daily=d.get("daily",{}); maxt=(daily.get("temperature_2m_max") or [None])[0]; mint=(daily.get("temperature_2m_min") or [None])[0]; prec=(daily.get("precipitation_sum") or [None])[0]
        level=alert_score(None,temp,wind,prec or 0); flag={"green":"🟢 Stable","yellow":"🟡 Heads Up","orange":"🟠 Warning","red":"🔴 Critical"}[level]
        out=[f"**Live weather - {loc}** (Open-Meteo, hourly):",f"• Condition: **{WMO.get(code,f'Code {code}')}**",f"• Temperature: **{temp}°C** | Wind: **{wind} km/h**"]
        if maxt and mint:out.append(f"• Range: {mint}°C – {maxt}°C")
        if prec is not None:out.append(f"• Precipitation: {prec} mm")
        out.append(f"\nRisk: **{flag}**")
        if level in("orange","red"):out.append("📱 Check imd.gov.in for official alerts.")
        return"\n".join(out)
    except Exception as e:return f"Weather fetch failed for {loc}: {e}"

def chat_disaster(msg):
    loc=extract_loc(msg); lat,lon=geocode(loc)
    if lat is None:return f"Couldn't locate **{loc}**."
    try:
        d=live_weather(lat,lon); cw=d["current_weather"]; temp=cw["temperature"]; wind=cw["windspeed"]
        prec=(d.get("daily",{}).get("precipitation_sum") or [0])[0] or 0
        level=alert_score(None,temp,wind,prec); label={"green":"🟢 Stable","yellow":"🟡 Elevated","orange":"🟠 High","red":"🔴 Critical"}[level]
        out=[f"**Disaster risk - {loc}** (live):",f"• {temp}°C | Wind {wind} km/h | Precip {prec} mm",f"• Risk: **{label}**"]
        if wind>60:out.append("• ⚠️ High wind - cyclonic risk.")
        if temp>42:out.append("• ⚠️ Heatwave threshold exceeded.")
        if prec>50:out.append("• ⚠️ Heavy rain - flood risk.")
        if level=="green":out.append("• No immediate hazards from current data.")
        if EVENT_MODEL is not None:
            try:pred=EVENT_MODEL.predict(np.array([[temp,wind,prec]]))[0];out.append(f"• **ML model:** {pred}")
            except Exception as me:out.append(f"• ML loaded, predict failed: {me}")
        else:out.append("• ML model not loaded - rule-based risk used.")
        out.append("\n📌 Official: ndma.gov.in | imd.gov.in")
        return"\n".join(out)
    except Exception as e:return f"Live data unavailable for {loc}: {e}"

def chat_forest(msg):
    if df_forest is None:return "Forest dataset offline. Check HF Space logs.\nPublic: fsi.nic.in"
    state=extract_loc(msg); rows=df_forest[df_forest['State'].str.contains(state,case=False,na=False)]
    if rows.empty:avail=", ".join(sorted(df_forest['State'].dropna().unique()[:10]));return f"No data for **{state}**.\nAvailable: {avail}."
    row=rows.iloc[-1]
    try:pct=float(str(row['Forest_Percentage_Geographical']).replace('%','').strip());note="Above national avg (21.7%)" if pct>25 else "Below national avg" if pct<18 else "Near national avg"
    except:note=""
    return f"**Forest - {state}**:\n• Area: **{row['Total_Forest_Recorded_SqKm']} km²**\n• Coverage: **{row['Forest_Percentage_Geographical']}%** - {note}\n• Latest: fsi.nic.in"

def chat_crop(msg):
    if df_crop is None:return "Crop dataset offline. Check HF Space logs.\nLive: agmarknet.gov.in"
    state=extract_loc(msg); rows=df_crop[df_crop['State_Name'].str.contains(state,case=False,na=False)]
    if rows.empty:avail=", ".join(sorted(df_crop['State_Name'].dropna().unique()[:10]));return f"No crop data for **{state}**.\nAvailable: {avail}."
    avg=round(rows['Crop Yield (kg per hectare)'].mean(),2)
    soil=rows['Soil_Type'].mode()[0] if 'Soil_Type' in rows.columns else "N/A"
    crop=rows['Crop'].mode()[0] if 'Crop' in rows.columns else "N/A"
    top5=rows['Crop'].value_counts().head(5).index.tolist() if 'Crop' in rows.columns else []
    return f"**Crop - {state}**:\n• Top: **{crop}** | Top 5: {', '.join(top5)}\n• Soil: **{soil}**\n• Avg yield: **{avg} kg/ha**"

def chat_drought(msg):
    if df_drought is None:return "Drought dataset offline. Check HF Space logs.\nBulletins: imd.gov.in"
    try:
        spei=round(df_drought['Drought Index (SPEI)'].mean(),3); temp=round(df_drought['Avg Temperature (°C)'].mean(),2)
        interp=("🔴 Severe drought" if spei<-2 else "🟠 Moderate drought" if spei<-1 else "🟡 Mild dryness" if spei<-0.5 else "🟢 Near-normal moisture")
        return f"**Drought analysis**:\n• Mean SPEI: **{spei}** - {interp}\n• Mean temp: **{temp}°C**\n• Current: imd.gov.in | nrsc.gov.in"
    except KeyError as e:return f"Column `{e}` missing. Columns: {list(df_drought.columns)}"

def chat_population(msg):
    if df_pop is None:return "Population dataset offline. Check HF Space logs."
    year=extract_year(msg); rows=df_pop[df_pop['Year']==year]; note=""
    if rows.empty:
        avail=sorted(df_pop['Year'].dropna().astype(int).unique().tolist()); closest=min(avail,key=lambda y:abs(y-year))
        rows=df_pop[df_pop['Year']==closest]; note=f"(showing {closest})"
    row=rows.iloc[0]
    try:
        pop=round(row['India Population (Millions)'],2); br=round(row['Birth Rate (per 1000)'],2); dr=round(row['Death Rate (per 1000)'],2); ur=round(row['Urbanization_Rate'],2)
        return f"**India population - {int(row['Year'])}** {note}:\n• Population: **{pop}M**\n• Birth: **{br}/1000** | Death: **{dr}/1000**\n• Growth: **{round(br-dr,2)}/1000** | Urban: **{ur}%**"
    except KeyError as e:return f"Column `{e}` missing."

def chat_ndvi(msg):
    loc=extract_loc(msg); lat,lon=geocode(loc)
    if lat is None:return f"Couldn't locate **{loc}**."
    val=ndvi_proxy(lat,lon)
    if val is None:return "NASA POWER unreachable.\nTrue NDVI: bhuvan.nrsc.gov.in"
    interp=("🌿 Dense/healthy veg" if val>0.6 else "🌱 Moderate veg" if val>0.3 else "🟡 Sparse/stressed" if val>0.1 else "🏜️ Bare/urban")
    return f"**Vegetation - {loc}** (NASA POWER proxy):\n• NDVI proxy: **{val}** - {interp}\n• True NDVI: bhuvan.nrsc.gov.in"

def smart_fallback(msg):
    loc=extract_loc(msg); has_loc=any(p.lower() in msg.lower() for p in INDIAN_PLACES) or bool(re.findall(r'\b[A-Z][a-z]{3,}\b',msg))
    parts=[]
    if has_loc:
        parts.append(f"Found location **{loc}** - here's current weather:\n")
        try:
            lat,lon=geocode(loc)
            if lat:parts.append(chat_weather(msg))
        except:pass
    avail=[f"{'✅' if df is not None else '❌'} {n}" for n,df in [("Forest (state-wise)",df_forest),("Crop yields",df_crop),("Drought/SPEI",df_drought),("Population (by year)",df_pop)]]
    avail+=["✅ Live weather","✅ Disaster risk","✅ NDVI proxy"]
    if not parts:parts.append("Didn't understand that. Here's what I can answer:\n")
    parts.append("**Available:**\n"+"\n".join(f"• {s}" for s in avail))
    parts.append("\n**Try:**\n• 'Weather in Pune'\n• 'Forest cover Assam'\n• 'Crop yield Bihar'\n• 'Drought analysis'\n• 'Population 2010'\n• 'NDVI for Gujarat'")
    return"\n".join(parts)

DISPATCH={"weather":chat_weather,"disaster":chat_disaster,"forest":chat_forest,"crop":chat_crop,"drought":chat_drought,"population":chat_population,"ndvi":chat_ndvi}

@app.get("/")
def root():
    return {"status":"running","datasets":{"population":df_pop is not None,"forest":df_forest is not None,"crop":df_crop is not None,"drought":df_drought is not None},"ml_model":EVENT_MODEL is not None,"version":"4.0"}

@app.get("/health")
def health():return{"ok":True}

@app.post("/chat")
def chat_endpoint(req:ChatRequest):
    msg=req.message.strip(); intents=classify(msg)
    if intents==["unknown"]:return{"reply":smart_fallback(msg),"intent":"unknown"}
    reply="\n\n---\n\n".join(DISPATCH[i](msg) for i in intents[:2]) if len(intents)>1 else DISPATCH[intents[0]](msg)
    return{"reply":reply,"intent":intents[0]}

@app.post("/aoi/analyze")
def analyze_aoi(req:AOIRequest):
    lat,lon=req.lat,req.lng
    try:
        w=live_weather(lat,lon); cw=w["current_weather"]; temp=cw["temperature"]; wind=cw["windspeed"]
        prec=(w.get("daily",{}).get("precipitation_sum") or [0])[0] or 0
    except Exception as e:return{"error":f"Weather API unavailable: {e}"}
    ndvi=ndvi_proxy(lat,lon); ndwi=round(max(-0.5,min(0.5,(prec/200)-0.3)),3); ndbi=round(max(-0.5,min(0.5,0.4-(ndvi or 0.3))),3)
    level=alert_score(ndvi,temp,wind,prec)
    summaries={"green":"🟢 Normal - no significant change.","yellow":"🟡 Heads Up - early signs of change. Monitor.","orange":"🟠 Warning - environmental stress. Action recommended.","red":"🔴 Critical - severe conditions. Immediate action needed."}
    summary=summaries[level]; email_sent=False
    if req.user_email and level!="green":email_sent=send_alert_email(req.user_email,req.aoi_name,level,summary,{"lat":lat,"lng":lon})
    return{"lat":lat,"lng":lon,"aoi_name":req.aoi_name,"ndvi":ndvi,"ndwi":ndwi,"ndbi":ndbi,"temperature":temp,"wind_speed":wind,"precipitation":prec,"alert_level":level,"summary":summary,"email_sent":email_sent,"timestamp":datetime.utcnow().isoformat()+"Z","data_sources":["Open-Meteo (live)","NASA POWER (16-day proxy)"]}

@app.post("/alert/email")
def email_alert(req:AlertEmailRequest):
    ok=send_alert_email(req.to_email,req.aoi_name,req.level,req.summary,{"lat":req.lat,"lng":req.lng})
    return{"success":ok,"message":f"Alert sent to {req.to_email}" if ok else "Email failed. Set SMTP_USER + SMTP_PASSWORD in HF Space Secrets."}

@app.post("/tool/population")
def get_population(payload:dict):
    if df_pop is None:return{"result":"Population offline."}
    year=payload.get("year",2017); data=df_pop[df_pop['Year']==year]
    if data.empty:return{"result":f"No data for {year}."}
    r=data.iloc[0];return{"result":f"In {year}: {round(r['India Population (Millions)'],2)}M, birth {round(r['Birth Rate (per 1000)'],2)}/1000, urban {round(r['Urbanization_Rate'],2)}%."}

@app.post("/tool/forest")
def get_forest(payload:dict):
    if df_forest is None:return{"result":"Forest offline."}
    state=payload.get("state","Telangana").title(); rows=df_forest[df_forest['State'].str.contains(state,case=False,na=False)]
    if rows.empty:return{"result":f"No data for {state}."}
    r=rows.iloc[-1];return{"result":f"{state}: {r['Total_Forest_Recorded_SqKm']} km² ({r['Forest_Percentage_Geographical']}%)."}

@app.post("/tool/crop")
def get_crop(payload:dict):
    if df_crop is None:return{"result":"Crop offline."}
    state=payload.get("state","Kerala").title(); rows=df_crop[df_crop['State_Name'].str.contains(state,case=False,na=False)]
    if rows.empty:return{"result":f"No data for {state}."}
    avg=round(rows['Crop Yield (kg per hectare)'].mean(),2); soil=rows['Soil_Type'].mode()[0] if 'Soil_Type' in rows.columns else "N/A"; crop=rows['Crop'].mode()[0] if 'Crop' in rows.columns else "N/A"
    return{"result":f"{state}: {crop}, {soil} soil, {avg} kg/ha."}

@app.post("/tool/drought")
def get_drought(payload:dict):
    if df_drought is None:return{"result":"Drought offline."}
    try:return{"result":f"SPEI: {round(df_drought['Drought Index (SPEI)'].mean(),2)}, temp: {round(df_drought['Avg Temperature (°C)'].mean(),2)}°C."}
    except KeyError as e:return{"result":f"Column error: {e}"}

@app.post("/tool/disaster")
def predict_disaster(payload:dict):
    loc=payload.get("location","Mumbai").title(); lat,lon=geocode(loc)
    if lat is None:return{"result":f"Cannot locate {loc}."}
    try:
        d=live_weather(lat,lon); cw=d["current_weather"]; temp=round(cw["temperature"]); wind=round(cw["windspeed"])
        prec=(d.get("daily",{}).get("precipitation_sum") or [0])[0] or 0
        lvl=alert_score(None,temp,wind,prec); lbl={"green":"Stable","yellow":"Elevated","orange":"High","red":"Critical"}[lvl]; note=""
        if EVENT_MODEL is not None:
            try:note=f" ML: {EVENT_MODEL.predict(np.array([[temp,wind,prec]]))[0]}."
            except Exception as me:note=f" (ML error: {me})"
        return{"result":f"{loc}: {temp}°C, {wind}km/h, {prec}mm. Risk: {lbl}.{note}"}
    except Exception as e:return{"result":f"Failed: {e}"}