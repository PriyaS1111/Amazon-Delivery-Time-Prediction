
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Your Saved Model, Scaler, and Columns ---
try:
    model = joblib.load('amazon_delivery_time_predictor.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    numerical_features = joblib.load('numerical_features.joblib') # <-- (1) LOAD THE NEW FILE
except FileNotFoundError:
    st.error("Model, scaler, or column files not found. Please ensure all 4 .joblib files are present.")
    st.stop()

# --- App Title and Description ---
st.title("Amazon Delivery Time Predictor")
st.write("Enter the details of the order to get an estimated delivery time.")

# --- Create the Input Form ---
st.header("Order Details")

col1, col2 = st.columns(2)

with col1:
    agent_age = st.slider("Agent Age", 20, 50, 35)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.8, 0.1)
    distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=15.5, step=0.1)
    pickup_time_minutes = st.number_input("Pickup Time (minutes)", min_value=1, max_value=60, value=10)

with col2:
    order_hour = st.slider("Order Hour (24h)", 0, 23, 14)
    traffic = st.selectbox("Traffic Condition", ['Low', 'Medium', 'High', 'Jam'])
    weather = st.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Stormy', 'Sandstorms'])
    vehicle = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'van'])

# --- Prediction Logic ---
if st.button("Predict Delivery Time"):
    try:
        # 1. Create a DataFrame for prediction
        prediction_df = pd.DataFrame(columns=model_columns)
        prediction_df.loc[0] = 0

        # 2. Update with user inputs
        prediction_df.at[0, 'Agent_Age'] = agent_age
        prediction_df.at[0, 'Agent_Rating'] = agent_rating
        prediction_df.at[0, 'Distance_km'] = distance_km
        prediction_df.at[0, 'pickup_time_minutes'] = pickup_time_minutes
        prediction_df.at[0, 'order_hour'] = order_hour

        # Handle one-hot encoded features by checking if the column exists
        traffic_col = f'Traffic_{traffic}'
        if traffic_col in model_columns:
            prediction_df.at[0, traffic_col] = 1

        weather_col = f'Weather_{weather}'
        if weather_col in model_columns:
            prediction_df.at[0, weather_col] = 1
            
        vehicle_col = f'Vehicle_{vehicle}'
        if vehicle_col in model_columns:
            prediction_df.at[0, vehicle_col] = 1

        # 3. Apply scaling using the loaded numerical features list
        prediction_df[numerical_features] = scaler.transform(prediction_df[numerical_features]) # <-- (2) USE THE LOADED LIST

        # 4. Make prediction
        predicted_time = model.predict(prediction_df)

        # 5. Display result
        st.success(f"**Estimated Delivery Time: {predicted_time[0]:.0f} minutes**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
