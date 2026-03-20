# Crop Prediction System
**Enter a location → Get the best crop recommendation automatically**

---

## How it works

```
User types location
        ↓
Geocoding API (Nominatim) → lat, lon
        ↓
3 APIs called automatically:
  • Open-Meteo  → temperature, rainfall, humidity, wind, sunshine
  • Open-Elevation → altitude
  • SoilGrids   → pH, organic carbon, soil moisture
        ↓
Data assembled into one feature row
        ↓
Random Forest Model predicts → Top 5 crops + confidence %
```

---

## Setup (do this once)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (do this once)
Put your dataset CSV in the same folder, then run:
```bash
python train.py
```
This creates 5 files: `model.joblib`, `encoders.joblib`, `target_encoder.joblib`,
`feature_cols.joblib`, `label_map.json`

---

## Usage

### Predict by city name
```bash
python predict.py "Pune"
python predict.py "Nashik"
python predict.py "Hyderabad"
```

### Predict by GPS coordinates
```bash
python predict.py "18.52,73.85"
python predict.py "28.61,77.20"
```

### Use in your own Python code
```python
from predict import predict_crop

results = predict_crop("Mumbai")
# Returns: [("Rice", 42.3), ("Sugarcane", 28.1), ...]

for crop, confidence in results:
    print(f"{crop}: {confidence}%")
```

---

## Dataset columns used

| Column | Source |
|--------|--------|
| temperature | Open-Meteo API |
| rainfall | Open-Meteo API |
| humidity | Open-Meteo API |
| Wind_speed | Open-Meteo API |
| Sunshine_hours | Open-Meteo API |
| Altitude_m | Open-Elevation API |
| pH | SoilGrids API |
| Organic_Carbon | SoilGrids API |
| Soil_Moisture | SoilGrids API |
| State_Name | Nominatim reverse geocode |
| Season | Auto-detected from current month |
| N, P, K | Dataset medians (override manually if known) |
| Soil_Type | Default: Neutral |
| Irrigation_Method | Default: Rainfed |
| Soil_Texture | Default: Loamy |

---

## Overriding NPK values

If you know the NPK values for a field, edit `predict.py` and change:
```python
"N": 68,   # ← change to actual value
"P": 53,   # ← change to actual value
"K": 78,   # ← change to actual value
```

---

## APIs used (all free, no key required)

| API | What it provides | URL |
|-----|-----------------|-----|
| Nominatim | Geocoding + reverse geocoding | nominatim.openstreetmap.org |
| Open-Meteo | Weather + climate | api.open-meteo.com |
| Open-Elevation | Altitude | api.open-elevation.com |
| SoilGrids | Soil properties | rest.isric.org |

---

## Model details

- Algorithm: Random Forest Classifier
- Trees: 100
- Training samples: ~82,000
- Test accuracy: **95.73%**
- Output: Top 5 crops with confidence scores
