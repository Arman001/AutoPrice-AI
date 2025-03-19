# src/api/services.py
import pickle
import pandas as pd
import numpy as np

# 1️⃣ Load trained model & preprocessing objects
with open("../models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/features.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("../models/numerical_cols.pkl", "rb") as f:
    numerical_cols = pickle.load(f)

with open("../models/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)  # Categorical encoding mappings

# Electric brands list for handling missing fuel types
electric_brands = ['Tesla', 'Rivian', 'Lucid', 'Polestar', 'Karma']

# 2️⃣ Preprocess input function
def preprocess_input(data):
    df = pd.DataFrame([data])  # Convert input dictionary to DataFrame

    # A. Handle Missing Fuel Type
    if pd.isna(df['fuel_type'][0]) or df['fuel_type'][0] == "":
        if df['brand'][0] in electric_brands:
            df['fuel_type'] = "Electric"
        else:
            df['fuel_type'] = mappings["fuel_type_by_brand"].get(df['brand'][0], "Unknown")

    # B. Brand Handling
    df['brand'] = df['brand'].map(mappings["brand_mapping"]).infer_objects(copy=False).fillna(-1).astype(int)


    # C. Transmission Encoding
    df['transmission'] = df['transmission'].map(mappings["transmission_mapping"]).infer_objects(copy=False).fillna(-1).astype(int)


    # D. Standardize Colors
    def map_color(color):
        """Maps color names to a standard format."""
        if isinstance(color, str):
            for key in mappings["color_mapping"]:
                if key.lower() in color.lower():
                    return mappings["color_mapping"][key]
        return 'Unknown'

    df['ext_col'] = df['ext_col'].apply(map_color)
    df['int_col'] = df['int_col'].apply(map_color)

    # E. One-Hot Encoding for Categorical Features
    df = pd.get_dummies(df, columns=['fuel_type', 'ext_col', 'int_col'], drop_first=True)

    # F. Ensure All Expected Features Exist
    for col in feature_names:
        if col not in df:
            df[col] = 0  

    # G. Convert Data Types to Float (AFTER Encoding)
    df = df.astype(float)

    return df[feature_names]

# 3️⃣ Prediction function
def predict_price(input_data):
    try:
        processed_input = preprocess_input(input_data)
        log_predicted_price = model.predict(processed_input)[0]
        predicted_price = np.exp(log_predicted_price)  # Reverse log transformation
        return round(predicted_price, 2)
    except Exception as e:
        print("❌ Prediction Error:", str(e))
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
