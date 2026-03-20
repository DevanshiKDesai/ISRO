# Weather Event Prediction System
**Enter a location → Predict weather events & severity automatically**

---

## What it predicts

| Output | Description |
|--------|-------------|
| Event type | Flood, Cyclone, Drought, Heatwave, Landslide, Thunderstorm, Cloudburst, Hailstorm |
| Intensity | Severity score from 1 (mild) to 10 (extreme) |
| Top 3 events | Ranked by probability with confidence % |

---

## How it works

```
User types location
        ↓
Nominatim API → lat, lon + Indian state name
        ↓
Open-Meteo API → temperature, precipitation, wind anomalies
        ↓
NOAA API → MEI index (El Niño / La Niña signal)
        ↓
Geographic rules applied (coastal → cyclone, hilly → landslide, etc.)
        ↓
Model 1: Random Forest → Event Type (Flood, Cyclone, etc.)
Model 2: Random Forest → Intensity score (1–10)
        ↓
Results displayed
```

---

## Setup

```bash
pip install -r requirements.txt
```

### Train models (once)
```bash
python train.py
```

---

## Usage

```bash
python predict.py "Mumbai"
python predict.py "Chennai"
python predict.py "Shimla"
python predict.py "28.61,77.20"
```

### Use in your code
```python
from predict import predict_weather

result = predict_weather("Mumbai")
# Returns:
# {
#   "top_events": [("Cyclone", 38.2), ("Flood", 24.1), ("Thunderstorm", 18.5)],
#   "intensity": 6.4
# }
```

---

## APIs used (all free)

| API | Data | URL |
|-----|------|-----|
| Nominatim | Geocoding + state detection | nominatim.openstreetmap.org |
| Open-Meteo | Weather anomalies | api.open-meteo.com |
| NOAA PSL | MEI / El Niño index | psl.noaa.gov |

---

## Geographic intelligence built in

The system uses domain knowledge to boost predictions:

- **Coastal states** (Odisha, Tamil Nadu, Kerala, etc.) → higher Cyclone probability during Monsoon
- **Hilly/NE states** (Uttarakhand, Himachal, Meghalaya, etc.) → higher Landslide probability with high rainfall
- **Dry states** (Rajasthan, Haryana, Delhi, etc.) → higher Heatwave/Drought in Summer
- **Any state + Monsoon + heavy rain** → higher Flood/Cloudburst probability

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script |
| `predict.py` | Inference pipeline |
| `event_model.joblib` | Trained event type classifier |
| `intensity_model.joblib` | Trained intensity regressor |
| `state_encoder.joblib` | State label encoder |
| `season_encoder.joblib` | Season label encoder |
| `event_encoder.joblib` | Event type label encoder |
| `feature_cols.joblib` | Feature column order |
| `metadata.json` | State lists and model info |
