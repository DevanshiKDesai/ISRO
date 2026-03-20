# Urbanization Prediction System
**Enter any city / state / area → 5-year urbanization forecast, fully automatic**

---

## What it predicts (5 years ahead)

| Model | Output | Accuracy |
|-------|--------|----------|
| Population | Future India population (Millions) | **R² 99.86%** |
| Urbanization Rate | % population living in cities | **R² 99.07%** |
| Urban Population | Total urban dwellers (Millions) | **R² 98.97%** |
| Infrastructure Pressure | Pressure score 0–100 | **R² 99.65%** |
| Growth Rate | Future population growth % | **R² 99.87%** |

---

## Full automatic pipeline

```
User types location
        ↓
Nominatim API → lat, lon, city, state
        ↓
World Bank API → live birth rate, death rate, growth rate,
                  urbanization rate, total population
        ↓
City/State urbanization rates (built-in lookup for 18 cities, 20 states)
        ↓
25 features built automatically
        ↓
5 models → 5-year forecast
        ↓
Full report: population, urbanization, infrastructure needs,
             economic/environmental/social/health effects
```

---

## Setup & Usage

```bash
pip install -r requirements.txt
python train.py                     # run once
python predict.py "Mumbai"
python predict.py "Rajasthan"
python predict.py "Bhopal"
python predict.py "Bangalore"
python predict.py "19.07,72.87"    # GPS coordinates
```

---

## Sample output

```
URBANIZATION FORECAST — MUMBAI
Current Year: 2025  →  Forecast Year: 2030

  METRIC                              NOW      →  2030    CHANGE
  Population (Millions)           1432.00M  →  1480.00M  +48.00M
  Urbanization Rate                 93.50%  →    95.20%   +1.70%
  Urban Population (Millions)     1339.00M  →  1409.00M  +70.00M
  Infrastructure Pressure            82.10  →    86.40    +4.30

INFRASTRUCTURE NEEDS (2025–2030):
  Pressure level  : [HIGH]  Score: 86.4/100
  ┌─ New schools needed  : 1,40,000
  ├─ New hospitals needed: 7,000
  ├─ New roads (km)      : 1,75,000
  ├─ New homes needed    : 1,55,55,556
  └─ Water demand        : 3,450 ML/year extra
```

---

## APIs used (free)

| API | Data |
|-----|------|
| Nominatim | Geocoding |
| World Bank | Live population, birth/death/growth rates |

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script |
| `predict.py` | Inference pipeline |
| `pop_model.joblib` | Population forecaster |
| `urb_model.joblib` | Urbanization rate forecaster |
| `infra_model.joblib` | Infrastructure pressure forecaster |
| `upop_model.joblib` | Urban population forecaster |
| `grow_model.joblib` | Growth rate forecaster |
| `feature_cols.joblib` | Feature order |
| `metadata.json` | City/state urbanization rates, model info |
| `india_enriched.csv` | Engineered training dataset |
