"""
Urbanization Prediction - Training Script
==========================================
Trains 5 models predicting 5 years into the future:
  1. Population (Millions)        R² 99.86%
  2. Urbanization Rate (%)        R² 99.07%
  3. Infrastructure Pressure      R² 99.65%
  4. Urban Population (Millions)  R² 98.97%
  5. Growth Rate (%)              R² 99.87%

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib, json, os, warnings
warnings.filterwarnings('ignore')

DATASET_PATH  = "India_Population_Cleaned.csv"
OUTPUT_DIR    = "."
FORECAST_YEARS= 5


def engineer_features(df):
    df = df.ffill()
    df['Years_Since_1961']        = df['Year'] - 1961
    df['Pop_Growth_Abs']          = df['India Population (Millions)'].diff().fillna(0)
    df['Pop_Rolling_3yr']         = df['India Population (Millions)'].rolling(3, min_periods=1).mean()
    df['Pop_Rolling_5yr']         = df['India Population (Millions)'].rolling(5, min_periods=1).mean()
    df['Growth_Rate_Change']      = df['India Growth Rate (%)'].diff().fillna(0)
    df['Natural_Increase_Rate']   = df['Birth Rate (per 1000)'] - df['Death Rate (per 1000)']
    df['Birth_Rate_Trend']        = df['Birth Rate (per 1000)'].rolling(3, min_periods=1).mean()
    df['Death_Rate_Trend']        = df['Death Rate (per 1000)'].rolling(3, min_periods=1).mean()
    df['World_Pop_Ratio']         = df['India Population (Millions)'] / df['World Population (Millions)']
    df['India_World_Growth_Diff'] = df['India Growth Rate (%)'] - df['World Growth Rate (%)']

    urb_anchor = {1961:17.97,1971:19.91,1981:23.34,1991:25.55,
                  2001:27.81,2011:31.16,2021:35.39}
    df['Urbanization_Rate']   = np.interp(df['Year'].values,
                                           list(urb_anchor.keys()),
                                           list(urb_anchor.values()))
    df['Urban_Pop_Millions']  = df['India Population (Millions)'] * df['Urbanization_Rate'] / 100
    df['Rural_Pop_Millions']  = df['India Population (Millions)'] - df['Urban_Pop_Millions']
    df['Urban_Rural_Ratio']   = df['Urban_Pop_Millions'] / (df['Rural_Pop_Millions'] + 1)
    df['Urb_Rate_Change']     = df['Urbanization_Rate'].diff().fillna(0)

    df['Infra_Pressure_Score'] = (
        (df['India Population (Millions)'] / df['India Population (Millions)'].max()) * 40 +
        (df['Urbanization_Rate']           / df['Urbanization_Rate'].max())            * 30 +
        (df['India Growth Rate (%)']       / df['India Growth Rate (%)'].max())         * 20 +
        (df['Urban_Pop_Millions']          / df['Urban_Pop_Millions'].max())            * 10
    ).clip(0, 100)

    df['School_Pressure']   = ((df['Urban_Pop_Millions'] / df['Urban_Pop_Millions'].max()) * 100).clip(0,100)
    df['Hospital_Pressure'] = ((df['Urban_Pop_Millions'] / df['Urban_Pop_Millions'].max()) * 90).clip(0,100)
    df['Road_Pressure']     = ((df['India Population (Millions)'] / df['India Population (Millions)'].max()) * 100).clip(0,100)

    df['Future_Population']     = df['India Population (Millions)'].shift(-FORECAST_YEARS)
    df['Future_Urb_Rate']       = df['Urbanization_Rate'].shift(-FORECAST_YEARS)
    df['Future_Infra_Pressure'] = df['Infra_Pressure_Score'].shift(-FORECAST_YEARS)
    df['Future_Urban_Pop']      = df['Urban_Pop_Millions'].shift(-FORECAST_YEARS)
    df['Future_Growth_Rate']    = df['India Growth Rate (%)'].shift(-FORECAST_YEARS)
    return df, urb_anchor


FEATURE_COLS = [
    'Years_Since_1961','India Population (Millions)','India Growth Rate (%)',
    'Birth Rate (per 1000)','Death Rate (per 1000)',
    'World Population (Millions)','World Growth Rate (%)',
    'Pop_Growth_Abs','Pop_Rolling_3yr','Pop_Rolling_5yr',
    'Growth_Rate_Change','Natural_Increase_Rate',
    'Birth_Rate_Trend','Death_Rate_Trend',
    'World_Pop_Ratio','India_World_Growth_Diff',
    'Urbanization_Rate','Urban_Pop_Millions','Rural_Pop_Millions',
    'Urban_Rural_Ratio','Urb_Rate_Change','Infra_Pressure_Score',
    'School_Pressure','Hospital_Pressure','Road_Pressure',
]


def train():
    print("=" * 55)
    print("  Urbanization Prediction - Training")
    print("=" * 55)

    print(f"\n[1/4] Loading '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df)} records | {df['Year'].min()}–{df['Year'].max()}")

    print("\n[2/4] Engineering features...")
    df, urb_anchor = engineer_features(df)
    df_model = df.dropna(subset=['Future_Population','Future_Urb_Rate','Future_Infra_Pressure'])
    X = df_model[FEATURE_COLS].fillna(0)
    print(f"  Features: {len(FEATURE_COLS)} | Training rows: {len(df_model)}")

    print(f"\n[3/4] Training 5 models (forecast: +{FORECAST_YEARS} years)...")
    models  = {}
    results = {}

    targets = {
        "pop_model":   ("Future_Population",     "Population (Millions)"),
        "urb_model":   ("Future_Urb_Rate",        "Urbanization Rate (%)"),
        "infra_model": ("Future_Infra_Pressure",  "Infra Pressure Score"),
        "upop_model":  ("Future_Urban_Pop",       "Urban Population (M)"),
        "grow_model":  ("Future_Growth_Rate",     "Growth Rate (%)"),
    }

    for key, (target_col, label) in targets.items():
        y = df_model[target_col]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, random_state=42
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        mae   = mean_absolute_error(y_te, preds)
        r2    = r2_score(y_te, preds)
        print(f"  {label:<35} MAE: {mae:.4f}  R²: {r2:.4f}")
        models[key]  = model
        results[key] = {'mae': round(float(mae), 4), 'r2': round(float(r2), 4)}

    print("\n[4/4] Saving...")
    for key, model in models.items():
        joblib.dump(model, os.path.join(OUTPUT_DIR, f"{key}.joblib"), compress=3)
    joblib.dump(FEATURE_COLS, os.path.join(OUTPUT_DIR, "feature_cols.joblib"))
    df[FEATURE_COLS + ['Year']].to_csv(
        os.path.join(OUTPUT_DIR, "india_enriched.csv"), index=False)

    national_latest = {}
    for k, v in df.iloc[-1].to_dict().items():
        try: national_latest[k] = float(v)
        except: national_latest[k] = str(v)

    meta = {
        'results': results, 'feature_cols': FEATURE_COLS,
        'forecast_years': FORECAST_YEARS,
        'latest_year': int(df['Year'].max()),
        'national_latest': national_latest,
        'urbanization_known': urb_anchor,
        'city_urb_rates': {
            'Mumbai':93.5,'Delhi':97.5,'Bangalore':91.2,'Hyderabad':90.5,
            'Chennai':88.9,'Kolkata':85.3,'Pune':82.6,'Ahmedabad':88.0,
            'Jaipur':77.4,'Lucknow':72.1,'Surat':92.4,'Nagpur':74.5,
            'Bhopal':71.3,'Indore':80.2,'Patna':68.7,'Kochi':78.9,
            'Nashik':75.1,'Vadodara':83.7,
        },
        'state_urb_rates': {
            'Maharashtra':45.2,'Tamil Nadu':48.4,'Kerala':47.7,'Karnataka':38.6,
            'Gujarat':42.6,'Punjab':37.5,'Haryana':34.9,'West Bengal':31.9,
            'Andhra Pradesh':29.6,'Uttar Pradesh':22.3,'Bihar':11.3,
            'Rajasthan':24.9,'Madhya Pradesh':27.6,'Odisha':16.7,
            'Jharkhand':24.1,'Assam':14.1,'Himachal Pradesh':10.0,
            'Uttarakhand':30.6,'Telangana':38.9,'Chhattisgarh':23.2,
        }
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  All files saved!")
    print(f"  Run: python predict.py \"Mumbai\"")
    print(f"  Or : python predict.py \"Rajasthan\"\n")


if __name__ == "__main__":
    train()
