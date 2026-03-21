from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading GeoDrishti Datasets into Memory...")

# ==========================================
# 1. LOAD DATASETS (BULLETPROOF INDIVIDUAL BLOCKS)
# ==========================================
df_pop = df_forest = df_crop = df_drought = None

try:
    df_pop = pd.read_csv("Population/india_enriched.csv", encoding='latin1')
    print("✅ Population Dataset Loaded")
except Exception as e:
    print(f"⚠️ Population dataset offline: Check path 'Population/india_enriched.csv'")

try:
    df_forest = pd.read_csv("Forest_prediction/New Forest.csv", encoding='latin1')
    print("✅ Forest Dataset Loaded")
except Exception as e:
    print(f"⚠️ Forest dataset offline: Check path 'Forest_prediction/New Forest.csv'")

try:
    df_crop = pd.read_csv("crop_predictor/enhanced_crop_yield_dataset (1).csv", encoding='latin1')
    print("✅ Crop Dataset Loaded")
except Exception as e:
    print(f"⚠️ Crop dataset offline: Check path 'crop_predictor/enhanced_crop_yield_dataset (1).csv'")

try:
    # 1. Try reading it as an actual Excel file first
    df_drought = pd.read_excel("drought_prediction/Drought New.xlsx")
    print("✅ Drought Dataset Loaded (Excel Mode)")
except:
    try:
        # 2. If it's actually a CSV, force it to skip the broken row 6!
        df_drought = pd.read_csv("drought_prediction/Drought New.xlsx", encoding='latin1', on_bad_lines='skip')
        print("✅ Drought Dataset Loaded (Forced CSV Mode)")
    except Exception as e:
        print(f"⚠️ TRUE DROUGHT ERROR: {e}")


# ==========================================
# 2. LOAD MODELS (FOR DISASTERS/WEATHER)
# ==========================================
try:
    EVENT_MODEL = joblib.load("Weather_Prediction/event_model.joblib")
    print("✅ Disaster Models Loaded")
except:
    EVENT_MODEL = None
    print("⚠️ Disaster models offline (event_model.joblib not found)")

# ==========================================
# 3. VAPI TOOL ENDPOINTS (CRASH-PROOF)
# ==========================================

@app.post("/tool/population")
def get_population(payload: dict):
    if df_pop is None:
        return {"result": "The population database is currently offline."}
        
    year = payload.get("year", 2017)
    data = df_pop[df_pop['Year'] == year]
    
    if data.empty:
        return {"result": f"No data available for the year {year}."}
    
    pop_millions = round(data['India Population (Millions)'].values[0], 2)
    birth_rate = round(data['Birth Rate (per 1000)'].values[0], 2)
    death_rate = round(data['Death Rate (per 1000)'].values[0], 2)
    urban_rate = round(data['Urbanization_Rate'].values[0], 2)
    
    return {
        "result": f"In {year}, India's population was {pop_millions} million. The birth rate was {birth_rate} per thousand, the death rate was {death_rate}, and the urbanization rate was {urban_rate}%."
    }

@app.post("/tool/forest")
def get_forest(payload: dict):
    if df_forest is None:
        return {"result": "The forest database is currently offline."}
        
    state = payload.get("state", "Telangana").title()
    state_data = df_forest[df_forest['State'].str.contains(state, case=False, na=False)]
    
    if state_data.empty:
        return {"result": f"No forest data found for {state}."}
    
    latest_data = state_data.iloc[-1]
    
    return {
        "result": f"For {state}, the total recorded forest area is {latest_data['Total_Forest_Recorded_SqKm']} square kilometers, covering {latest_data['Forest_Percentage_Geographical']}% of the geographical area."
    }

@app.post("/tool/crop")
def get_crop(payload: dict):
    if df_crop is None:
        return {"result": "The crop yield database is currently offline."}
        
    state = payload.get("state", "Kerala").title()
    state_data = df_crop[df_crop['State_Name'].str.contains(state, case=False, na=False)]
    
    if state_data.empty:
        return {"result": f"No crop data found for {state}."}
    
    avg_yield = round(state_data['Crop Yield (kg per hectare)'].mean(), 2)
    common_soil = state_data['Soil_Type'].mode()[0]
    top_crop = state_data['Crop'].mode()[0]
    
    return {
        "result": f"In {state}, the predominant soil type is {common_soil}. The most common crop is {top_crop}, with an average historical yield of {avg_yield} kg per hectare."
    }

@app.post("/tool/drought")
def get_drought(payload: dict):
    if df_drought is None:
        return {"result": "The drought database is currently offline. Please verify the folder name."}
        
    mean_spei = round(df_drought['Drought Index (SPEI)'].mean(), 2)
    avg_temp = round(df_drought['Avg Temperature (°C)'].mean(), 2)
    
    return {
        "result": f"Based on our dataset analysis, the average recorded temperature is {avg_temp} degrees Celsius, and the aggregate Standardized Precipitation Evapotranspiration Index is {mean_spei}."
    }

@app.post("/tool/disaster")
def predict_disaster(payload: dict):
    location = payload.get("location", "Mumbai").title()
    
    try:
        # 1. LIVE GEOCODING: Get exact coordinates for the city
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_resp = requests.get(geo_url).json()
        
        if "results" not in geo_resp:
            return {"result": f"I cannot locate {location} on the map."}
            
        lat = geo_resp['results'][0]['latitude']
        lon = geo_resp['results'][0]['longitude']
        
        # 2. LIVE WEATHER: Fetch real-time satellite data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url).json()
        
        current = weather_resp['current_weather']
        temp = round(current['temperature'])
        wind = round(current['windspeed'])
        
        # 3. LIVE PREDICTION LOGIC
        # Here we use the live data to create a real-time risk assessment
        risk_level = "stable"
        alert = "No severe weather events predicted."
        
        if wind > 60:
            risk_level = "High"
            alert = "Warning: High risk of cyclonic activity or severe gales."
        elif temp > 42:
            risk_level = "High"
            alert = "Warning: Severe heatwave conditions detected."
        elif temp < 5:
            risk_level = "Moderate"
            alert = "Cold wave conditions are currently active."
            
        # If your EVENT_MODEL from predict.py is loaded, you can eventually pass 'temp' and 'wind' directly into it here!
            
        return {
            "result": f"Currently in {location}, it is {temp} degrees with wind speeds of {wind} kilometers per hour. The risk level is {risk_level}. {alert}"
        }
        
    except Exception as e:
        return {"result": f"Satellite uplink failed for {location}."}