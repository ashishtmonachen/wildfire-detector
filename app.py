# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import folium
from streamlit_folium import st_folium

# Config
st.set_page_config(page_title="Wildfire Detector", layout="wide")
st.title("🛰️ Wildfire Detection Dashboard")

# API Keys
GOOGLE_API_KEY = "AIzaSyBFJsMwO6dzcBaFNf3U51yNiGOMDz5oNeo"

# Load model once
@st.cache_resource
def load_cnn_model():
    return load_model("nasa_wildfire_cnn.h5")

# Load fire data from archived CSV
fire_data = pd.read_csv("archived_fire.csv")
selected_fire = fire_data.iloc[0]  # Select first fire for now

lat = selected_fire["latitude"]
lon = selected_fire["longitude"]

# Fetch satellite image from NASA API
def fetch_satellite_image(lat, lon):
    for days_ago in range(10):
        date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"https://api.nasa.gov/planetary/earth/imagery?lon={lon}&lat={lat}&dim=0.2&date={date}&api_key=DEMO_KEY"
        response = requests.get(url)
        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            with open("wildfire_real_time.jpg", "wb") as f:
                f.write(response.content)
            return date
    return None

st.info("Fetching satellite image...")
date_captured = fetch_satellite_image(lat, lon)

# Run CNN Prediction
def predict():
    image = load_img("wildfire_real_time.jpg", target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    model = load_cnn_model()
    prediction = model.predict(image_array)[0][0]
    return "🔥 Wildfire Confirmed!" if prediction > 0.5 else "🌿 No Wildfire Detected"

# Get nearby places from Google Places API
def get_nearby_services(lat, lon, service_type):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=50000&type={service_type}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return [place["name"] for place in data.get("results", [])][:5]  # Top 5 results

# Display Results
if os.path.exists("wildfire_real_time.jpg"):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📍 Wildfire Location")
        st.write(f"**Latitude:** {lat}, **Longitude:** {lon}")
        st.write(f"**Captured on:** {date_captured}")

        # Map
        m = folium.Map(location=[lat, lon], zoom_start=7)
        folium.Marker([lat, lon], tooltip="Detected Wildfire", icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, width=600, height=400)

    with col2:
        st.subheader("🛰️ Satellite Image")
        st.image("wildfire_real_time.jpg", use_column_width=True)

    st.subheader("🤖 Prediction Result")
    result = predict()
    st.success(result)

    st.subheader("📍 Nearby Emergency Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("🚒 **Fire Stations**")
        for station in get_nearby_services(lat, lon, "fire_station"):
            st.write(f"• {station}")

    with col2:
        st.markdown("🏥 **Hospitals**")
        for hospital in get_nearby_services(lat, lon, "hospital"):
            st.write(f"• {hospital}")

    with col3:
        st.markdown("🚔 **Police Stations**")
        for police in get_nearby_services(lat, lon, "police"):
            st.write(f"• {police}")
else:
    st.error("Satellite image could not be fetched.")
