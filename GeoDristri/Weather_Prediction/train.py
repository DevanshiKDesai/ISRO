"""
Weather Event Prediction - Model Training Script
=================================================
Trains TWO models:
  1. Event Type Classifier  → predicts Flood, Cyclone, Drought etc.
  2. Intensity Regressor    → predicts severity on scale 1–10

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib, json, os

DATASET_PATH = "Weather_Events_India.csv"
OUTPUT_DIR   = "."

# States grouped by geography (domain knowledge)
COASTAL_STATES = ['Odisha','West Bengal','Tamil Nadu','Andhra Pradesh',
                  'Kerala','Karnataka','Goa','Maharashtra','Gujarat']
HILLY_STATES   = ['Himachal Pradesh','Uttarakhand','Arunachal Pradesh','Sikkim',
                  'Meghalaya','Manipur','Nagaland','Mizoram','Tripura','Assam']
DRY_STATES     = ['Rajasthan','Haryana','Punjab','Delhi','Gujarat']


def train():
    print("=" * 55)
    print("  Weather Event Prediction - Training")
    print("=" * 55)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    df['event_date'] = pd.to_datetime(df['event_date'], dayfirst=True)
    print(f"  {len(df):,} records | {df['event_type'].nunique()} event types | {df['state'].nunique()} states")

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\n[2/5] Engineering features...")
    df['month']      = df['event_date'].dt.month
    df['season']     = df['month'].map({
        12:'Winter', 1:'Winter',  2:'Winter',
        3:'Summer',  4:'Summer',  5:'Summer',
        6:'Monsoon', 7:'Monsoon', 8:'Monsoon', 9:'Monsoon',
        10:'Post-Monsoon', 11:'Post-Monsoon'
    })
    df['is_coastal'] = df['state'].isin(COASTAL_STATES).astype(int)
    df['is_hilly']   = df['state'].isin(HILLY_STATES).astype(int)
    df['is_dry']     = df['state'].isin(DRY_STATES).astype(int)

    # ── Encode ────────────────────────────────────────────────────────────────
    print("\n[3/5] Encoding categories...")
    state_le  = LabelEncoder()
    season_le = LabelEncoder()
    event_le  = LabelEncoder()
    df['state_enc']  = state_le.fit_transform(df['state'])
    df['season_enc'] = season_le.fit_transform(df['season'])
    print(f"  States: {len(state_le.classes_)} | Seasons: {len(season_le.classes_)} | Events: {df['event_type'].nunique()}")

    feature_cols = [
        'state_enc', 'month', 'season_enc',
        'is_coastal', 'is_hilly', 'is_dry',
        'precipitation_anomaly_mm', 'mei_index',
        'temperature_anomaly_c', 'wind_anomaly_kmph',
        'duration_days'
    ]
    X = df[feature_cols]

    # ── Train Model 1: Event Type ─────────────────────────────────────────────
    print("\n[4/5] Training Model 1: Event Type Classifier...")
    y_event = event_le.fit_transform(df['event_type'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_event, test_size=0.2,
                                               random_state=42, stratify=y_event)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  Event type accuracy: {acc*100:.2f}%")
    print(f"  Classes: {list(event_le.classes_)}")

    # ── Train Model 2: Intensity ───────────────────────────────────────────────
    print("\n     Training Model 2: Intensity Regressor (scale 1-10)...")
    y_int = df['intensity_scale']
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_int, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, max_depth=20,
                                 random_state=42, n_jobs=-1)
    reg.fit(X_tr2, y_tr2)
    mae = mean_absolute_error(y_te2, reg.predict(X_te2))
    print(f"  Intensity MAE: {mae:.2f} (out of scale 1-10)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[5/5] Saving models...")
    joblib.dump(clf,       os.path.join(OUTPUT_DIR, "event_model.joblib"),     compress=3)
    joblib.dump(reg,       os.path.join(OUTPUT_DIR, "intensity_model.joblib"), compress=3)
    joblib.dump(state_le,  os.path.join(OUTPUT_DIR, "state_encoder.joblib"))
    joblib.dump(season_le, os.path.join(OUTPUT_DIR, "season_encoder.joblib"))
    joblib.dump(event_le,  os.path.join(OUTPUT_DIR, "event_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_cols.joblib"))

    meta = {
        'coastal_states':  COASTAL_STATES,
        'hilly_states':    HILLY_STATES,
        'dry_states':      DRY_STATES,
        'event_classes':   list(event_le.classes_),
        'intensity_mae':   round(float(mae), 2),
        'event_accuracy':  round(float(acc), 4),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("  Saved: event_model.joblib")
    print("  Saved: intensity_model.joblib")
    print("  Saved: state_encoder.joblib")
    print("  Saved: season_encoder.joblib")
    print("  Saved: event_encoder.joblib")
    print("  Saved: feature_cols.joblib")
    print("  Saved: metadata.json")

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Now run:  python predict.py \"Your City\"")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()
