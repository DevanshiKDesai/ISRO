from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting GeoDristri Omni-Backend...")

# ==========================================
# 1. LOAD ALL ML MODELS (CRASH-PROOFED)
# ==========================================

try:
    forest_model = joblib.load('Forest_prediction/cover_model.joblib')
    forest_features = joblib.load('Forest_prediction/feature_cols.joblib')
    print("✅ Forest Model Loaded")
except:
    forest_model = None
    forest_features = []

try:
    weather_model = joblib.load('Weather_prediction/weather_model.joblib')
    weather_features = joblib.load('Weather_prediction/weather_features.joblib')
    print("✅ Weather Model Loaded")
except:
    weather_model = None
    weather_features = [] # <-- This fixes your crash!

try:
    crop_model = joblib.load('Crop_prediction/crop_model.joblib')
    crop_features = joblib.load('Crop_prediction/crop_features.joblib')
    print("✅ Crop Model Loaded")
except:
    crop_model = None
    crop_features = []

try:
    drought_model = joblib.load('Drought_prediction/drought_model.joblib')
    drought_features = joblib.load('Drought_prediction/drought_features.joblib')
    print("✅ Drought Model Loaded")
except:
    drought_model = None
    drought_features = []

try:
    pop_model = joblib.load('Population_prediction/pop_model.joblib')
    pop_features = joblib.load('Population_prediction/pop_features.joblib')
    print("✅ Population Model Loaded")
except:
    pop_model = None
    pop_features = []

# ==========================================
# 2. THE DYNAMIC PREDICTION ENGINE
# ==========================================

def run_prediction(model, features, payload):
    # 1. If the model isn't built yet, stop gracefully.
    if not model:
        return {"success": False, "error": "Model offline or missing."}
        
    # 2. Dynamic Sklearn Fallback: If you forgot to save the features.joblib, 
    # the code will try to extract them directly from the model itself!
    if not features and hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        
    if not features:
        return {"success": False, "error": "Feature columns missing."}
    
    # 3. Create the empty dataframe dynamically matching your exact training columns
    df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    
    # 4. Inject payload data if available
    if payload:
        for key, value in payload.items():
            if key in df.columns:
                df[key] = value
                
    prediction = model.predict(df)[0]
    return {"success": True, "result": prediction}

# ... (Keep your @app.post routes below exactly the same)

@app.post("/predict/forest")
def predict_forest(payload: dict = None):
    res = run_prediction(forest_model, forest_features, payload)
    if res["success"]:
        return {"success": True, "predicted_cover": float(res["result"])}
    return res

@app.post("/predict/weather")
def predict_weather(payload: dict = None):
    res = run_prediction(weather_model, weather_features, payload)
    if res["success"]:
        return {"success": True, "predicted_weather": str(res["result"])}
    return res

@app.post("/predict/crop")
def predict_crop(payload: dict = None):
    res = run_prediction(crop_model, crop_features, payload)
    if res["success"]:
        return {"success": True, "optimal_crop": str(res["result"])}
    return res

@app.post("/predict/drought")
def predict_drought(payload: dict = None):
    res = run_prediction(drought_model, drought_features, payload)
    if res["success"]:
        return {"success": True, "drought_risk_index": float(res["result"])}
    return res

@app.post("/predict/population")
def predict_pop(payload: dict = None):
    res = run_prediction(pop_model, pop_features, payload)
    if res["success"]:
        return {"success": True, "population_density_shift": float(res["result"])}
    return res

# ==========================================
# 3. UNIFIED TEXT CHATBOT ROUTER (FOR REACT)
# ==========================================

@app.post("/chat")
def chat_router(payload: dict):
    msg = payload.get("message", "").lower()

    # ROUTE 1: CROP
    if any(word in msg for word in ["crop", "grow", "plant", "agriculture", "yield"]):
        res = predict_crop(payload)
        if res.get("success"):
            return {"reply": f"Based on the GeoDristri ML model, the optimal crop for these conditions is {res['optimal_crop']}."}
        return {"reply": "The Crop ML model is currently offline. Please check your .joblib files."}

    # ROUTE 2: FOREST
    elif any(word in msg for word in ["forest", "tree", "cover", "deforestation"]):
        res = predict_forest(payload)
        if res.get("success"):
            return {"reply": f"The GeoDristri continuous monitoring model predicts a forest cover of {res['predicted_cover']:.2f}%."}
        return {"reply": "The Forest ML model is currently offline. Please check your .joblib files."}

    # ROUTE 3: WEATHER
    elif any(word in msg for word in ["weather", "rain", "temperature", "climate"]):
        res = predict_weather(payload)
        if res.get("success"):
            return {"reply": f"Our predictive weather models indicate: {res['predicted_weather']}."}
        return {"reply": "The Weather ML model is currently offline. Please check your .joblib files."}

    # ROUTE 4: DROUGHT
    elif any(word in msg for word in ["drought", "dry", "water scarcity"]):
        res = predict_drought(payload)
        if res.get("success"):
            return {"reply": f"The calculated drought risk index for this region is {res['drought_risk_index']:.2f}."}
        return {"reply": "The Drought ML model is currently offline. Please check your .joblib files."}

    # ROUTE 5: POPULATION
    elif any(word in msg for word in ["population", "people", "demographic", "density"]):
        res = predict_pop(payload)
        if res.get("success"):
            return {"reply": f"The projected population density shift is {res['population_density_shift']:.2f}."}
        return {"reply": "The Population ML model is currently offline. Please check your .joblib files."}

    # FALLBACK
    else:
        return {"reply": "I am the GeoDristri Omni-AI. Ask me to run a prediction for forest cover, weather, crops, drought, or population dynamics."}