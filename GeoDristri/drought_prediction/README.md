# Drought Prediction System
**Enter any location → Automatic drought assessment with 99.65% accuracy**

---

## What it predicts

| Output | Description | Accuracy |
|--------|-------------|----------|
| Drought Category | Extremely Dry / Severely Dry / Moderately Dry / Near Normal / Moderately Wet / Severely Wet / Extremely Wet | **99.65%** |
| Drought Status | Is drought actively occurring? (Yes/No) | **100.00%** |
| SPEI Index | Continuous drought severity value (-3.5 to +3.5) | MAE **0.0002** |

---

## How it works

```
User types location (any city, region, or country — works globally)
        ↓
Nominatim API → lat, lon
        ↓
Open-Meteo API → Max/Min/Avg Temp, Humidity, Precipitation,
                  Wind Speed, Solar Radiation (7-day forecast)
        ↓
SPEI estimated using Thornthwaite PET formula
        ↓
Sin-Cos encoding of lat, lon, month (matches training format)
        ↓
24 features assembled (raw + engineered + spatial + temporal)
        ↓
Model 1: Random Forest → Drought Category (7 classes)
Model 2: Random Forest → Drought Status (0/1)
Model 3: Random Forest → SPEI value (regression)
        ↓
Results + alert level displayed
```

---

## Setup

```bash
pip install -r requirements.txt
```

### Train (once)
```bash
python train.py
```

---

## Usage

```bash
# Indian locations
python predict.py "Rajasthan"
python predict.py "Kerala"
python predict.py "Maharashtra"
python predict.py "Delhi"
python predict.py "Chennai"

# Global locations (works worldwide!)
python predict.py "Sahara Desert"
python predict.py "Amazon Rainforest"
python predict.py "California"

# GPS coordinates
python predict.py "28.61,77.20"
```

---

## SPEI Index reference

| SPEI Value | Category |
|------------|----------|
| ≤ -2.0 | Extremely Dry |
| -2.0 to -1.5 | Severely Dry |
| -1.5 to -1.0 | Moderately Dry |
| -1.0 to +1.0 | Near Normal |
| +1.0 to +1.5 | Moderately Wet |
| +1.5 to +2.0 | Severely Wet |
| ≥ +2.0 | Extremely Wet |

---

## APIs used (all free)

| API | Data | URL |
|-----|------|-----|
| Nominatim | Geocoding | nominatim.openstreetmap.org |
| Open-Meteo | All weather data | api.open-meteo.com |

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script |
| `predict.py` | Inference pipeline |
| `category_model.joblib` | Drought category classifier |
| `status_model.joblib` | Drought status (0/1) classifier |
| `spei_model.joblib` | SPEI regression model |
| `category_encoder.joblib` | Label encoder for categories |
| `feature_cols.joblib` | Feature column order |
| `metadata.json` | Model info and accuracies |
