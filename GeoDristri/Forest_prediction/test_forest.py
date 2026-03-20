import joblib
import pandas as pd
import numpy as np

print("Loading files...")

# 1. Load the cheat code (the exact columns the model expects)
features = joblib.load('feature_cols.joblib')
print(f"Model expects {len(features)} features.")

# 2. Load the actual forest cover model
model = joblib.load('cover_model.joblib')

# 3. Create perfectly sized dummy data using the exact column names
# We use a Pandas DataFrame filled with 1s just to see if the engine turns over
dummy_data = pd.DataFrame(np.ones((1, len(features))), columns=features)

# 4. Run the prediction
try:
    print("\nRunning prediction...")
    prediction = model.predict(dummy_data)
    print("✅ SUCCESS! The forest cover model predicted:")
    print(prediction)
except Exception as e:
    print("❌ The model failed. Error:")
    print(e)