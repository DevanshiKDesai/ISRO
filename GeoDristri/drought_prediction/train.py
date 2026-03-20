"""
Drought Prediction - Model Training Script
==========================================
Trains THREE models:
  1. Drought Category Classifier  → Extremely Dry / Severely Dry / etc. (7 classes)
  2. Drought Status Classifier    → Drought (1) or No Drought (0)
  3. SPEI Regressor               → Continuous drought index value

Accuracy achieved:
  Category : 99.65%
  Status   : 100.00%
  SPEI MAE : 0.0002

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import joblib, json, os, warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "1773903002787_Drought_New.xlsx"
OUTPUT_DIR   = "."


def engineer_features(df):
    """Decode and engineer all features from raw dataset."""
    df = df.copy()
    # Decode lat/lon/month from sin-cos encoding
    df['lat']   = np.degrees(np.arctan2(df['lat_sin'],   df['lat_cos']))
    df['lon']   = np.degrees(np.arctan2(df['lon_sin'],   df['lon_cos']))
    df['month'] = (np.round(
        np.degrees(np.arctan2(df['month_sin'], df['month_cos'])) / 30
    ).astype(int) % 12 + 1)

    # Domain-driven engineered features
    df['temp_range']      = df['Max Temp (°C)'] - df['Min Temp (°C)']
    df['heat_stress']     = df['Max Temp (°C)'] * (1 - df['Relative Humidity (%)'] / 100)
    df['precip_humidity'] = df['Precipitation (mm)'] * df['Relative Humidity (%)']
    df['aridity_index']   = df['Solar Radiation'] / (df['Precipitation (mm)'] + 1)
    df['wind_evap']       = df['Wind Speed (m/s)'] * df['Max Temp (°C)']
    df['season']          = df['month'].map({
        12: 0, 1: 0, 2: 0,   # Winter
        3:  1, 4: 1, 5: 1,   # Spring
        6:  2, 7: 2, 8: 2,   # Summer
        9:  3, 10: 3, 11: 3  # Autumn
    })
    return df


FEATURE_COLS = [
    # Raw weather features
    'Relative Humidity (%)', 'Max Temp (°C)', 'Min Temp (°C)',
    'Wind Speed (m/s)', 'Avg Temperature (°C)', 'Solar Radiation',
    'Precipitation (mm)', 'Drought Index (SPEI)',
    # Spatial features (sin-cos encoded)
    'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
    # Temporal features
    'month_sin', 'month_cos', 'month', 'season',
    # Decoded coordinates
    'lat', 'lon',
    # Engineered features
    'temp_range', 'heat_stress', 'precip_humidity',
    'aridity_index', 'wind_evap',
    'Wind Speed (m/s) (bins)',
]


def train():
    print("=" * 55)
    print("  Drought Prediction - Model Training")
    print("=" * 55)

    # ── Load & engineer ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading '{DATASET_PATH}'...")
    df = pd.read_excel(DATASET_PATH)
    print(f"  {len(df):,} records | {df['Drought Category'].nunique()} categories")

    print("\n[2/5] Engineering features...")
    df = engineer_features(df)
    print(f"  Total features: {len(FEATURE_COLS)}")

    # ── Encode targets ────────────────────────────────────────────────────────
    print("\n[3/5] Preparing targets...")
    category_le = LabelEncoder()
    df['category_enc'] = category_le.fit_transform(df['Drought Category'])
    print(f"  Categories: {list(category_le.classes_)}")

    X        = df[FEATURE_COLS].values
    y_cat    = df['category_enc'].values
    y_status = df['Drought Status (0/1)'].values
    y_spei   = df['Drought Index (SPEI)'].values

    # Stratified split
    idx = np.arange(len(X))
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr_idx, te_idx in sss.split(X, y_cat):
        X_tr, X_te = X[tr_idx], X[te_idx]
        yc_tr, yc_te = y_cat[tr_idx], y_cat[te_idx]
        ys_tr, ys_te = y_status[tr_idx], y_status[te_idx]
        ysp_tr, ysp_te = y_spei[tr_idx], y_spei[te_idx]

    print(f"  Train: {len(X_tr):,} | Test: {len(X_te):,}")

    # ── Model 1: Drought Category ─────────────────────────────────────────────
    print("\n[4/5] Training models...")
    print("  Model 1: Drought Category Classifier (7 classes)...")
    rf_cat = RandomForestClassifier(
        n_estimators=150, max_depth=30, min_samples_leaf=1,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf_cat.fit(X_tr, yc_tr)
    acc_cat = accuracy_score(yc_te, rf_cat.predict(X_te))
    print(f"    Accuracy: {acc_cat*100:.2f}%")
    print(classification_report(yc_te, rf_cat.predict(X_te),
          target_names=category_le.classes_, digits=3))

    # ── Model 2: Drought Status ───────────────────────────────────────────────
    print("  Model 2: Drought Status Classifier (0/1)...")
    rf_status = RandomForestClassifier(
        n_estimators=100, max_depth=20, max_features='sqrt',
        random_state=42, n_jobs=-1
    )
    rf_status.fit(X_tr, ys_tr)
    acc_status = accuracy_score(ys_te, rf_status.predict(X_te))
    print(f"    Accuracy: {acc_status*100:.2f}%")

    # ── Model 3: SPEI Regressor ───────────────────────────────────────────────
    print("  Model 3: SPEI Regressor (continuous index)...")
    rf_spei = RandomForestRegressor(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    rf_spei.fit(X_tr, ysp_tr)
    mae_spei = mean_absolute_error(ysp_te, rf_spei.predict(X_te))
    print(f"    MAE: {mae_spei:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[5/5] Saving models...")
    joblib.dump(rf_cat,      os.path.join(OUTPUT_DIR, "category_model.joblib"),  compress=3)
    joblib.dump(rf_status,   os.path.join(OUTPUT_DIR, "status_model.joblib"),    compress=3)
    joblib.dump(rf_spei,     os.path.join(OUTPUT_DIR, "spei_model.joblib"),      compress=3)
    joblib.dump(category_le, os.path.join(OUTPUT_DIR, "category_encoder.joblib"))
    joblib.dump(FEATURE_COLS,os.path.join(OUTPUT_DIR, "feature_cols.joblib"))

    meta = {
        'feature_cols':      FEATURE_COLS,
        'category_classes':  list(category_le.classes_),
        'category_accuracy': round(float(acc_cat) * 100, 2),
        'status_accuracy':   round(float(acc_status) * 100, 2),
        'spei_mae':          round(float(mae_spei), 4),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("  Saved: category_model.joblib")
    print("  Saved: status_model.joblib")
    print("  Saved: spei_model.joblib")
    print("  Saved: category_encoder.joblib")
    print("  Saved: feature_cols.joblib")
    print("  Saved: metadata.json")

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Category accuracy : {acc_cat*100:.2f}%")
    print(f"  Status accuracy   : {acc_status*100:.2f}%")
    print(f"  SPEI MAE          : {mae_spei:.4f}")
    print(f"  Now run: python predict.py \"Your City\"")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()
