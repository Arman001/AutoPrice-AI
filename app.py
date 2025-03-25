import streamlit as st
import requests

# FastAPI URL
API_URL = "http://127.0.0.1:8000/api/predict"

st.title("AutoPrice AI - Car Price Predictor")

# User Inputs
year = st.number_input("Year", min_value=1990, max_value=2025, step=1)
mileage = st.number_input("Mileage", min_value=0.0, max_value=500000.0, step=1000.0)
brand = st.selectbox("Brand", ["Toyota", "Honda", "Ford", "BMW"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
engine = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, step=0.1)
ext_col = st.text_input("Exterior Color")
int_col = st.text_input("Interior Color")
accident = st.number_input("Accident Count", min_value=0, step=1)

# Predict Button
if st.button("Predict Price"):
    data = {
        "year": year,
        "mileage": mileage,
        "brand": brand,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "engine": engine,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident
    }
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        st.success(f"Predicted Price: ${response.json()['predicted_price']:.2f}")
    else:
        st.error("Prediction failed. Please check your inputs.")

