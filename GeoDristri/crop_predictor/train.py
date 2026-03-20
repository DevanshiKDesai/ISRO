"""
Crop Prediction - Model Training Script
========================================
Run this ONCE to train the model on your dataset.
It saves the model and encoders so predict.py can use them.

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = "enhanced_crop_yield_dataset__1_.csv"
OUTPUT_DIR    = "."          # saves model files in same folder
N_ESTIMATORS  = 100          # number of trees (higher = more accurate but slower)
TEST_SIZE     = 0.2          # 20% of data used for testing
RANDOM_STATE  = 42


def train():
    print("=" * 55)
    print("  Crop Prediction - Model Training")
    print("=" * 55)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Target: 'Crop' — {df['Crop'].nunique()} unique crops")

    # ── Drop engineered/derived columns (not available from APIs) ────────────
    print("\n[2/5] Selecting features...")
    drop_cols = [
        'Crop Yield (kg per hectare)',  # not needed for classification
        'Rainfall_Temperature',          # derived — we'll recompute if needed
        'Humidity_Temperature',          # derived
        'NPK_Ratio',                     # derived
        'Weather_Index',                 # derived
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"  Dropped {len(drop_cols)} derived columns")

    # ── Encode categorical features ───────────────────────────────────────────
    print("\n[3/5] Encoding categorical columns...")
    cat_cols = ['State_Name', 'Season', 'Soil_Type', 'Irrigation_Method', 'Soil_Texture']
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values")

    # Encode target column
    target_le = LabelEncoder()
    df['Crop_encoded'] = target_le.fit_transform(df['Crop'])
    print(f"  Crop (target): {len(target_le.classes_)} classes")

    # ── Split data ────────────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in ['Crop', 'Crop_encoded']]
    X = df[feature_cols]
    y = df['Crop_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Training set: {len(X_train):,} samples")
    print(f"  Test set:     {len(X_test):,} samples")

    # ── Train model ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Training Random Forest ({N_ESTIMATORS} trees)...")
    print("  This may take 1-2 minutes...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,           # use all CPU cores
        class_weight='balanced',
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Feature importances
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\n  Top 10 most important features:")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat:<25} {bar} {imp:.4f}")

    # ── Save everything ───────────────────────────────────────────────────────
    print(f"\n[5/5] Saving model and encoders...")
    joblib.dump(model,        os.path.join(OUTPUT_DIR, "model.joblib"))
    joblib.dump(encoders,     os.path.join(OUTPUT_DIR, "encoders.joblib"))
    joblib.dump(target_le,    os.path.join(OUTPUT_DIR, "target_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_cols.joblib"))

    # Save human-readable crop label map
    label_map = {int(i): str(c) for i, c in enumerate(target_le.classes_)}
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print("  Saved: model.joblib")
    print("  Saved: encoders.joblib")
    print("  Saved: target_encoder.joblib")
    print("  Saved: feature_cols.joblib")
    print("  Saved: label_map.json")

    print(f"\n{'='*55}")
    print(f"  Training complete! Accuracy: {acc*100:.2f}%")
    print(f"  Now run:  python predict.py \"Your City\"")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()
