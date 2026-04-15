# 🌍 EcoSight — AI-Powered Geospatial Intelligence Platform

<div align="center">

![EcoSight Banner](https://img.shields.io/badge/EcoSight-AI%20Geospatial%20Platform-2d6a4f?style=for-the-badge&logo=leaf)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![ML Models](https://img.shields.io/badge/ML%20Models-14%20Total-FF6B35?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

**A fully automatic, end-to-end AI platform for environmental monitoring, agricultural intelligence, and urban planning — powered by real-time APIs, satellite data, and machine learning.**

[Features](#features) • [Architecture](#architecture) • [ML Models](#ml-models) • [Tech Stack](#tech-stack) • [Installation](#installation) • [Usage](#usage) • [API Reference](#api-reference)

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Machine Learning Models](#machine-learning-models)
  - [Model 1 — Crop Prediction](#model-1--crop-prediction)
  - [Model 2 — Drought Prediction](#model-2--drought-prediction)
  - [Model 3 — Forest & Deforestation](#model-3--forest--deforestation-prediction)
  - [Model 4 — Urbanization Forecasting](#model-4--urbanization-forecasting)
- [Feature Engineering](#feature-engineering)
- [Data Pipeline & Workflow](#data-pipeline--workflow)
- [AOI Auto-fill System](#aoi-auto-fill-system)
- [Tech Stack](#tech-stack)
- [APIs Used](#apis-used)
- [Dataset Details](#dataset-details)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Power BI Dashboard](#power-bi-dashboard)
- [Authentication](#authentication)
- [Model Performance](#model-performance)
- [Contributors](#contributors)

---

## 🌟 Project Overview

**EcoSight** is a full-stack geospatial AI platform built for environmental monitoring and predictive analysis across India. The user simply draws an **Area of Interest (AOI)** on an interactive map — the system automatically fetches all required real-time data through APIs, processes it through ML pipelines, and delivers instant predictions across four domains:

| Domain | What It Predicts |
|---|---|
| 🌾 **Crop Intelligence** | Best crop to grow based on soil & weather |
| 🏜️ **Drought Monitoring** | Drought category, status & SPEI index |
| 🌳 **Forest Health** | Deforestation alert, future cover & human impact |
| 🏙️ **Urban Growth** | Population, urbanization & infrastructure needs for next 5 years |

> **Zero manual input required** — AOI selection triggers the entire pipeline automatically.

---

## ✨ Key Features

- **AOI-based automatic prediction** — draw on map, get instant results
- **14 ML models** across 4 domains running simultaneously
- **Real-time API integration** — live weather, soil, population & vegetation data
- **Power BI dashboard** — interactive visual analytics
- **Multi-level geographic analysis** — area, city, and state level
- **Full effects reports** — air quality, health, water, climate, livelihood, biodiversity
- **Infrastructure needs calculator** — schools, hospitals, roads, water demand
- **Clerk authentication** — secure user login with Google/GitHub/Email
- **Fully responsive frontend** — React + TypeScript + Vite

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + TypeScript)                │
│                                                                   │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│   │  Clerk Auth  │    │ Leaflet Map  │    │  Power BI Embed  │   │
│   │  (Login/SSO) │    │ AOI Drawing  │    │   Dashboard      │   │
│   └─────────────┘    └──────┬───────┘    └──────────────────┘   │
│                              │ lat/lon                            │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (Python / FastAPI)                   │
│                                                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    API ORCHESTRATION LAYER                │  │
│   │                                                           │  │
│   │  OpenWeatherMap  SoilGrids  WorldBank  OpenMeteo  Nominatim│  │
│   │       ↓              ↓          ↓          ↓         ↓   │  │
│   │  Weather Data   Soil Data   Pop Data  NDVI Proxy  Geocode │  │
│   └──────────────────────────┬────────────────────────────────┘  │
│                               │                                   │
│   ┌──────────────────────────▼────────────────────────────────┐  │
│   │                   PREPROCESSING LAYER                     │  │
│   │   Label Encoding │ Sin-Cos Encoding │ Feature Engineering  │  │
│   │   Rolling Avg    │ YoY Calculation  │ Ratio Computation    │  │
│   └──────────────────────────┬────────────────────────────────┘  │
│                               │                                   │
│   ┌──────────────────────────▼────────────────────────────────┐  │
│   │                    ML MODEL LAYER                          │  │
│   │                                                           │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│   │  │  CROP    │  │ DROUGHT  │  │  FOREST  │  │  URBAN   │ │  │
│   │  │ 1 Model  │  │ 3 Models │  │ 5 Models │  │ 5 Models │ │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│   └──────────────────────────┬────────────────────────────────┘  │
│                               │                                   │
│   ┌──────────────────────────▼────────────────────────────────┐  │
│   │                   REPORT GENERATION                        │  │
│   │   Effects Report │ Infrastructure Needs │ Alert Levels     │  │
│   └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    JSON Response → Frontend
```

---

## 🤖 Machine Learning Models

### Overview

| Module | Models | Algorithm | Features | Best Accuracy |
|---|---|---|---|---|
| Crop Prediction | 1 | Random Forest Classifier | ~15 | High |
| Drought Prediction | 3 | RF Classifier + RF Regressor | 24 | 100% |
| Forest & Deforestation | 5 | RF Classifier + RF Regressor | 37 | 99.15% |
| Urbanization Forecasting | 5 | Gradient Boosting Regressor | 25 | R² 99.87% |
| **Total** | **14** | | | |

---

### Model 1 — Crop Prediction

**Goal:** Predict the most suitable crop for a given location based on soil and weather.

**Algorithm:** `RandomForestClassifier`

```
Input: State, Season, Soil Type, Irrigation Method, Soil Texture,
       Rainfall, Temperature, Humidity, pH, Nitrogen, Phosphorus, Potassium
          ↓
Random Forest (100 trees, class_weight='balanced')
          ↓
Output: Recommended Crop (Wheat / Rice / Cotton / etc.)
```

**Key Parameters:**
| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 100 | 100 decision trees for stable voting |
| `class_weight` | balanced | Handles rare crop classes fairly |
| `n_jobs` | -1 | Uses all CPU cores |
| `random_state` | 42 | Reproducible results |

**Preprocessing:**
- Dropped 5 derived columns (`NPK_Ratio`, `Weather_Index`, etc.) — not available in real world
- Label Encoding for 5 categorical columns
- 80/20 stratified train-test split

---

### Model 2 — Drought Prediction

**Goal:** Predict drought severity, status, and SPEI index for any location.

**Three models trained simultaneously:**

| Model | Type | Output | Accuracy |
|---|---|---|---|
| Category Classifier | RF Classifier | 7-class drought category | **99.65%** |
| Status Classifier | RF Classifier | Drought Yes/No (binary) | **100.00%** |
| SPEI Regressor | RF Regressor | Continuous SPEI value | **MAE: 0.0002** |

**SPEI (Standardized Precipitation Evapotranspiration Index):**
```
SPEI = (Precipitation - PET) / scale

PET = Thornthwaite formula:
  T = max(Avg Temperature, 0)
  I = (T/5)^1.514
  PET = 16 × (10T/I)^a

SPEI Scale:
  ≤ -2.0  → Extremely Dry
  -2.0 to -1.5 → Severely Dry
  -1.5 to -1.0 → Moderately Dry
  -1.0 to +1.0 → Near Normal
  +1.0 to +1.5 → Moderately Wet
  ≥ +2.0  → Extremely Wet
```

**Key Feature Engineering:**
```python
# Sin-Cos encoding for spatial/temporal cyclical features
lat_sin, lat_cos   = sin(lat_rad), cos(lat_rad)
lon_sin, lon_cos   = sin(lon_rad), cos(lon_rad)
month_sin, month_cos = sin(month×30°), cos(month×30°)

# Domain-driven features
temp_range      = Max Temp - Min Temp
heat_stress     = Max Temp × (1 - Humidity/100)
precip_humidity = Precipitation × Humidity
aridity_index   = Solar Radiation / (Precipitation + 1)
wind_evap       = Wind Speed × Max Temp
```

**Total Features: 24** | **Split Method: StratifiedShuffleSplit**

---

### Model 3 — Forest & Deforestation Prediction

**Goal:** Predict deforestation alert level, future vegetation, forest cover, AQI impact, and human impact.

**Five models trained:**

| Model | Type | Output | Performance |
|---|---|---|---|
| Alert Classifier | RF Classifier | 4-level deforestation alert | **99.15%** |
| NDVI Regressor | RF Regressor | Future vegetation index | High R² |
| Cover Regressor | RF Regressor | Future forest area (sq km) | **R² 90%** |
| AQI Regressor | RF Regressor | Air quality impact (0-300) | **R² 99.5%** |
| Human Regressor | RF Regressor | Human impact score (0-100) | **R² 99.2%** |

**Alert Levels:**
```
0 → No Alert           (forest stable / growing)
1 → Mild Deforestation     (loss < 500 sq km, < 1%)
2 → Severe Deforestation   (loss < 2000 sq km, < 3%)
3 → Critical Deforestation (loss > 2000 sq km, > 3%)
```

**Advanced Feature Engineering (37 Features):**

```python
# Group 1: Year-over-Year changes (7 features)
Forest_Change_YoY, NDVI_Change_YoY, Forest_Pct_Change_YoY,
VeryDense_Change_YoY, ModDense_Change_YoY, OpenForest_Change_YoY,
Crop_Change_YoY

# Group 2: Rolling averages (6 features)
Forest_Cover_3yr_avg, Forest_Cover_5yr_avg,
NDVI_3yr_avg, NDVI_5yr_avg,
Forest_Pct_3yr_avg, Forest_Pct_5yr_avg

# Group 3: Ratio features (4 features)
Dense_to_Total_Ratio  = (VeryDense + ModDense) / Total Forest
Open_to_Total_Ratio   = Open Forest / Total Forest
Scrub_to_Forest_Ratio = Scrub / Forest
Crop_to_Forest_Ratio  = Crop Area / Forest Area

# Group 4: Streak detection (3 features)
Cum_Forest_Change    = cumulative sum of all yearly changes
Deforestation_Streak = 1 if loss this year else 0
Streak_Count         = consecutive years of forest loss

# Engineered targets
AQI_Impact_Score  = (-Forest_Pct_Change × 8) + ((1-NDVI) × 40) + (Crop_Ratio × 5)
Human_Impact_Score= ((1 - Forest_Pct/100) × 40) + ((1-NDVI) × 30) + (Crop_Ratio × 3)
```

**Multi-level Prediction:**
```
Input: "Bhopal"
  → Area level  : Bhopal locality
  → City level  : Bhopal city
  → State level : Madhya Pradesh
  → Summary table comparing all three levels
```

**Effects Report covers 6 dimensions:**
`Air Quality` | `Health Risks` | `Water Security` | `Climate` | `Livelihood` | `Biodiversity`

---

### Model 4 — Urbanization Forecasting

**Goal:** Forecast population, urbanization, urban population, infrastructure pressure, and growth rate **5 years ahead**.

**Algorithm: `GradientBoostingRegressor`** — chosen over Random Forest because:
- Population is a **sequential time-series** problem
- GB builds trees **sequentially** — each corrects previous errors
- Better for precise **numerical forecasting**
- RF builds trees independently — better for classification

**Five models trained:**

| Model | Target | R² Score |
|---|---|---|
| `pop_model` | Future Population (Millions) | **99.86%** |
| `grow_model` | Future Growth Rate (%) | **99.87%** |
| `infra_model` | Infrastructure Pressure Score | **99.65%** |
| `urb_model` | Future Urbanization Rate (%) | **99.07%** |
| `upop_model` | Future Urban Population (M) | **98.97%** |

**Model Configuration:**
```python
GradientBoostingRegressor(
    n_estimators  = 300,   # 300 sequential trees
    max_depth     = 4,     # shallow to avoid overfitting
    learning_rate = 0.05,  # slow careful learning
    subsample     = 0.9,   # 90% data per tree — adds diversity
    random_state  = 42
)
```

**Feature Engineering (25 Features):**
```python
# Temporal
Years_Since_1961, Pop_Rolling_3yr, Pop_Rolling_5yr

# Demographic
Natural_Increase_Rate = Birth Rate - Death Rate
Growth_Rate_Change    = current - previous growth rate

# Global context
World_Pop_Ratio         = India Pop / World Pop
India_World_Growth_Diff = India Growth - World Growth

# Urbanization
Urban_Pop_Millions = Population × Urbanization Rate / 100
Urban_Rural_Ratio  = Urban Pop / Rural Pop

# Infrastructure composite (weighted index)
Infra_Pressure = (Pop/MaxPop × 40) + (Urb/MaxUrb × 30) +
                 (Growth/MaxGrowth × 20) + (UrbanPop/MaxUrban × 10)
```

**5-Year Future Target Creation:**
```python
# pandas shift creates future labels
df['Future_Population'] = df['India Population'].shift(-5)
# Row 2020 features → paired with → Row 2025 population value
```

**Infrastructure Needs Calculator:**
```python
new_schools   = urban_growth_M × 1000 × 20    # 20 schools per 1000 people
new_hospitals = urban_growth_M × 1000 × 1     # 1 hospital per 1000 people
new_road_km   = urban_growth_M × 1000 × 2.5   # 2.5 km road per 1000 people
new_homes     = urban_growth_M × 1M / 4.5     # avg 4.5 per household
water_demand  = urban_growth × 135L × 365 days
```

---

## 🔧 Feature Engineering Summary

| Technique | Used In | Purpose |
|---|---|---|
| **Label Encoding** | Crop | Convert text categories to numbers |
| **Sin-Cos Encoding** | Drought | Cyclical spatial/temporal features |
| **Rolling Averages** | Forest, Urban | Smooth year-to-year noise |
| **Year-over-Year Diff** | Forest | Detect change trends |
| **Ratio Features** | Forest | Relative proportions |
| **Streak Detection** | Forest | Consecutive loss patterns |
| **Composite Index** | Urban, Forest | Multi-factor pressure scores |
| **Shift(-n) Targets** | Urban | Create future prediction labels |
| **Extrapolation** | Urban | Project trends to current year |
| **Fuzzy Matching** | Forest | Handle state name inconsistencies |

---

## 🔄 Data Pipeline & Workflow

```
TRAINING PHASE (runs once)
═══════════════════════════
Raw Dataset (CSV/Excel)
        ↓
Load & Validate Data
        ↓
Feature Engineering
        ↓
Encode Categorical Variables
        ↓
Train-Test Split (80/20 Stratified)
        ↓
Train ML Models
        ↓
Evaluate (Accuracy / R² / MAE)
        ↓
Save Models to Disk (.joblib)
        ↓
Save Encoders + Metadata (.json)


PREDICTION PHASE (runs on each AOI click)
══════════════════════════════════════════
User Clicks AOI on Map
        ↓
Extract Latitude & Longitude
        ↓
Call APIs Automatically
  ├─ Nominatim     → State/City name
  ├─ Open-Meteo    → Weather + NDVI proxy
  ├─ SoilGrids     → Soil properties (Crop model)
  └─ World Bank    → Population indicators (Urban model)
        ↓
Build Feature Vector
  ├─ Apply same encoding as training
  ├─ Engineer all derived features
  └─ Arrange in exact same column order
        ↓
Load Saved Models from Disk
        ↓
Run Predictions (all models simultaneously)
        ↓
Generate Effects Report
        ↓
Return JSON → Display on Frontend
```

---

## 🗺️ AOI Auto-fill System

When a user clicks or draws an AOI on the map, the following happens automatically with **zero manual input:**

```
AOI Click
   ↓ lat, lon extracted
   ├──→ Nominatim API    → State, City, Area names
   ├──→ Open-Meteo API   → Max/Min/Avg Temp, Humidity,
   │                        Precipitation, Wind Speed, Solar Radiation
   ├──→ SoilGrids API    → Soil Type, Texture, pH, N, P, K values
   ├──→ World Bank API   → Birth Rate, Death Rate, Growth Rate,
   │                        Population, Urbanization Rate
   ├──→ System Clock     → Current Month → Season auto-detected
   └──→ Math Formulas    → Sin-Cos encoding, SPEI calculation,
                           Engineered features computed
                           
All features assembled → Models predict → Results displayed
Total time: < 3 seconds
Manual input required: 0
```

**Season Auto-detection:**
```python
if month in [11, 12, 1, 2, 3]:  → "Rabi"
if month in [6, 7, 8, 9, 10]:   → "Kharif"
if month in [3, 4, 5]:          → "Zaid / Summer"
```

---

## 💻 Tech Stack

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| React | 18+ | UI framework |
| TypeScript | 5+ | Type-safe development |
| Vite | 5+ | Fast build tool |
| Leaflet.js | Latest | Interactive map & AOI drawing |
| Clerk | Latest | Authentication (Google/GitHub/Email) |
| Power BI Embed | Latest | Dashboard integration |

### Backend / ML
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core ML language |
| scikit-learn | Latest | ML models (RF, GB) |
| pandas | Latest | Data manipulation |
| numpy | Latest | Numerical computing |
| joblib | Latest | Model serialization |
| FastAPI | Latest | REST API server |
| requests | Latest | API calls |

### ML Algorithms
| Algorithm | Used For | Why |
|---|---|---|
| `RandomForestClassifier` | Crop, Drought, Forest | Ensemble voting, handles imbalance |
| `RandomForestRegressor` | Drought SPEI, Forest | Non-linear regression |
| `GradientBoostingRegressor` | Urbanization | Sequential error correction for time-series |

### Visualization
| Tool | Purpose |
|---|---|
| Power BI Desktop | Dashboard creation |
| Power BI Service | Dashboard hosting & embed |

---

## 🌐 APIs Used

| API | Provider | Data Provided | Cost |
|---|---|---|---|
| **Open-Meteo** | Open-Meteo | Temperature, Humidity, Precipitation, Wind, Solar Radiation, ET0 | Free |
| **Nominatim** | OpenStreetMap | Geocoding, Reverse geocoding, State/City/Area names | Free |
| **SoilGrids** | ISRIC | Soil type, texture, pH, NPK values | Free |
| **World Bank API** | World Bank | India population, birth/death/growth rates, urbanization | Free |

> ✅ **All APIs are completely free — no API key required for most**

---

## 📊 Dataset Details

| Dataset | Source | Records | Coverage |
|---|---|---|---|
| `enhanced_crop_yield_dataset.csv` | Agricultural survey data | Thousands of rows | All Indian states, multiple seasons |
| `Drought_New.xlsx` | Meteorological records | Large | India, multi-year monthly data |
| `New_Forest.csv` | Forest Survey of India | State-wise yearly | 1990–2021, all states |
| `India_Population_Cleaned.csv` | Census + World Bank | 60 rows | 1961–2021, national level |

---

## ⚙️ Installation & Setup

### Prerequisites
```bash
Python 3.10+
Node.js 18+
npm 9+
```

### Backend Setup
```bash
# Clone repository
git clone https://github.com/your-username/ecosight.git
cd ecosight

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Train all models (run once)
cd crop_model    && python train.py
cd ../drought_model && python train.py
cd ../forest_model  && python train.py
cd ../urban_model   && python train.py

# Start backend server
cd ../backend
uvicorn main:app --reload --port 8000
```

### Frontend Setup
```bash
cd Frontend/ecosight-web

# Install dependencies
npm install

# Install Clerk authentication
npm install @clerk/clerk-react

# Create environment file
echo "VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_key_here" > .env

# Start development server
npm run dev
```

---

## 📁 Project Structure

```
ecosight/
│
├── 📂 crop_model/
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Inference pipeline
│   ├── enhanced_crop_yield_dataset.csv
│   ├── model.joblib                # Trained RF model
│   ├── encoders.joblib             # Label encoders
│   ├── target_encoder.joblib       # Target class encoder
│   ├── feature_cols.joblib         # Feature column order
│   └── label_map.json              # Crop name lookup
│
├── 📂 drought_model/
│   ├── train.py
│   ├── predict.py
│   ├── Drought_New.xlsx
│   ├── category_model.joblib       # 7-class drought classifier
│   ├── status_model.joblib         # Binary drought classifier
│   ├── spei_model.joblib           # SPEI regressor
│   ├── category_encoder.joblib
│   ├── feature_cols.joblib
│   └── metadata.json
│
├── 📂 forest_model/
│   ├── train.py
│   ├── predict.py
│   ├── New_Forest.csv
│   ├── alert_model.joblib          # Deforestation alert classifier
│   ├── ndvi_model.joblib           # Future NDVI regressor
│   ├── cover_model.joblib          # Future forest cover regressor
│   ├── aqi_model.joblib            # AQI impact regressor
│   ├── human_model.joblib          # Human impact regressor
│   ├── state_encoder.joblib
│   ├── feature_cols.joblib
│   ├── state_data.json             # Historical state forest data
│   └── metadata.json
│
├── 📂 urban_model/
│   ├── train.py
│   ├── predict.py
│   ├── India_Population_Cleaned.csv
│   ├── pop_model.joblib            # Future population regressor
│   ├── urb_model.joblib            # Future urbanization regressor
│   ├── infra_model.joblib          # Infrastructure pressure regressor
│   ├── upop_model.joblib           # Future urban population regressor
│   ├── grow_model.joblib           # Future growth rate regressor
│   ├── feature_cols.joblib
│   ├── india_enriched.csv
│   └── metadata.json
│
├── 📂 Frontend/
│   └── ecosight-web/
│       ├── src/
│       │   ├── App.tsx             # Main app with Clerk auth
│       │   ├── main.tsx            # ClerkProvider setup
│       │   ├── components/
│       │   │   ├── Map.tsx         # Leaflet AOI drawing
│       │   │   ├── Dashboard.tsx   # Power BI embed
│       │   │   ├── CropPanel.tsx
│       │   │   ├── DroughtPanel.tsx
│       │   │   ├── ForestPanel.tsx
│       │   │   └── UrbanPanel.tsx
│       │   └── index.css
│       ├── .env                    # Clerk publishable key
│       ├── package.json
│       └── vite.config.ts
│
├── 📂 dashboard/
│   └── EcoSight_Dashboard.pbix    # Power BI dashboard file
│
├── 📂 backend/
│   └── main.py                    # FastAPI server
│
├── requirements.txt
└── README.md
```

---

## 📈 Power BI Dashboard

The Power BI dashboard provides visual analytics of the training data and model insights.

**Dashboard includes:**
- Smart Crop Yield & Soil Health Monitor
- State-wise forest cover trends
- Drought frequency heatmaps
- Urbanization growth projections
- N, P, K distribution by state
- Irrigation method breakdown
- Seasonal crop yield comparisons

**To embed in your project:**
```html
<iframe
  src="YOUR_POWERBI_EMBED_URL"
  width="100%"
  height="700px"
  frameborder="0"
  allowFullScreen="true">
</iframe>
```

---

## 🔐 Authentication

EcoSight uses **Clerk** for authentication — providing:
- Google OAuth login
- GitHub OAuth login
- Email/Password login
- User profile management
- Protected route handling

```tsx
// Protected routes — only accessible after login
<SignedIn>
  <Dashboard />    {/* Map, predictions, Power BI */}
</SignedIn>

<SignedOut>
  <LandingPage />  {/* Sign in prompt */}
</SignedOut>
```

---

## 📊 Model Performance Summary

| Model | Task | Metric | Score |
|---|---|---|---|
| Crop Classifier | Multi-class Classification | Accuracy | High |
| Drought Category | 7-class Classification | Accuracy | **99.65%** |
| Drought Status | Binary Classification | Accuracy | **100.00%** |
| Drought SPEI | Regression | MAE | **0.0002** |
| Forest Alert | 4-class Classification | Accuracy | **99.15%** |
| Forest AQI | Regression | R² | **0.995** |
| Forest Human | Regression | R² | **0.992** |
| Forest Cover | Regression | R² | **0.90** |
| Urban Population | Regression | R² | **0.9986** |
| Urban Growth Rate | Regression | R² | **0.9987** |
| Urban Infra Pressure | Regression | R² | **0.9965** |
| Urban Urb Rate | Regression | R² | **0.9907** |
| Urban Urban Pop | Regression | R² | **0.9897** |

---

## 🤝 Contributors

| Name | Role |
|---|---|
| **Hitesh** | ML Engineering, Frontend Development, System Architecture |
| **Friend (Crop Dataset)** | Provided the agricultural dataset |

---

## 📄 License

This project was built for the **ISRO Geospatial Hackathon**.

---

## 🙏 Acknowledgements

- **ISRO** — for the hackathon platform and inspiration
- **Forest Survey of India** — for historical forest data
- **World Bank Open Data** — for population indicators
- **OpenStreetMap / Nominatim** — for free geocoding
- **Open-Meteo** — for free weather API
- **ISRIC SoilGrids** — for free soil data

---

<div align="center">

**Built with ❤️ for environmental intelligence and sustainable India**

![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat-square&logo=python)
![Made with React](https://img.shields.io/badge/Made%20with-React-61DAFB?style=flat-square&logo=react)
![For ISRO](https://img.shields.io/badge/Built%20for-ISRO%20Hackathon-orange?style=flat-square)

</div>
