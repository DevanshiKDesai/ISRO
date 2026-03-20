# Forest & Deforestation Prediction System
**Enter any area / city / state → Full deforestation & human impact report**

---

## What it predicts

| Model | Output | Accuracy |
|-------|--------|----------|
| Deforestation Alert | No Alert / Mild / Severe / Critical | **99.15%** |
| Future NDVI | Vegetation greenness next year | MAE 0.048 |
| Future Forest Cover | Forest area next year (sq km) | **R² 90.07%** |
| AQI Impact Score | Air quality degradation score | **R² 99.46%** |
| Human Impact Score | Overall human risk score 0-100 | **R² 99.23%** |

---

## What it means for locals

The system generates a full effects report covering:
- **Air Quality** — AQI forecast and respiratory risk
- **Health** — heat stress, disease vectors, CO2 levels
- **Water Security** — groundwater, flood risk, river health
- **Climate** — local temperature rise, rainfall disruption
- **Livelihood** — forest income, agriculture impact
- **Biodiversity** — species risk, habitat loss estimate

---

## 3-level checking

```bash
# State level
python predict.py "Maharashtra"

# City level (auto-resolves to state)
python predict.py "Bhopal"
python predict.py "Dehradun"

# Area / local level
python predict.py "Satpura"

# Multi-level check (area → city → state)
python predict.py "Bhopal" multi
python predict.py "Nashik" multi

# GPS coordinates
python predict.py "21.16,79.09"
```

---

## Setup

```bash
pip install -r requirements.txt
python train.py      # trains all 5 models (run once)
python predict.py "Maharashtra"
```

---

## How it works

```
User enters location (area / city / state / GPS)
        ↓
Nominatim API → lat, lon, state name
        ↓
State matched to forest dataset (36 Indian states)
        ↓
Open-Meteo API → live vegetation index (NDVI proxy)
        ↓
37 features assembled:
  • Historical forest metrics (cover, NDVI, density)
  • Year-over-year changes (deforestation trend)
  • 3-year & 5-year rolling averages
  • Crop encroachment ratios
  • Cumulative deforestation streak
  • Current year + state encoding
        ↓
5 models run in parallel → full prediction report
        ↓
Human effects report generated (air, health, water, climate, livelihood)
```

---

## APIs used (free)

| API | Data |
|-----|------|
| Nominatim | Geocoding + reverse geocoding |
| Open-Meteo | Live vegetation proxy (ET0, precipitation) |

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script |
| `predict.py` | Inference + effects report |
| `alert_model.joblib` | Deforestation alert classifier |
| `ndvi_model.joblib` | Future NDVI regressor |
| `cover_model.joblib` | Future forest cover regressor |
| `aqi_model.joblib` | AQI impact regressor |
| `human_model.joblib` | Human impact regressor |
| `state_encoder.joblib` | State label encoder |
| `feature_cols.joblib` | Feature order |
| `state_data.json` | Historical state data for inference |
| `metadata.json` | Model info and state list |
