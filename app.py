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
from streamlit_extras.colored_header import colored_header
from branca.element import MacroElement, Template, JavascriptLink

# Sci-fi Theme Setup
st.set_page_config(page_title="🛰️ Wildfire Sentinel AI", layout="wide")
colored_header("WILDFIRE DETECTION SYSTEM", description="Sentinel AI v2.7 Monitoring Active Fires in the USA", color_name="red-70")

# API Keys
GOOGLE_API_KEY = "AIzaSyBFJsMwO6dzcBaFNf3U51yNiGOMDz5oNeo"
NASA_API_KEY = "sE98DPEqgN0f7dfmi14gEpcPqE2LNeK4JCIgNk7Z"

@st.cache_resource
def load_cnn_model():
    return load_model("nasa_wildfire_cnn.h5")

# Load fire data
fire_data = pd.read_csv("archived_fire.csv")
selected_fire = fire_data.iloc[0]

try:
    lat = float(str(selected_fire["latitude"]).strip())
    lon = float(str(selected_fire["longitude"]).strip())
except ValueError:
    st.error("❌ Invalid coordinates in archived_fire.csv.")
    st.stop()

# Fetch satellite image
@st.cache_data
def fetch_satellite_image(lat, lon):
    for days_ago in range(10):
        date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"https://api.nasa.gov/planetary/earth/imagery?lon={lon}&lat={lat}&dim=0.2&date={date}&api_key={NASA_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            with open("wildfire_real_time.jpg", "wb") as f:
                f.write(response.content)
            return date
    return None

date_captured = fetch_satellite_image(lat, lon)

# Prediction
def predict():
    image = load_img("wildfire_real_time.jpg", target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    model = load_cnn_model()
    prediction = model.predict(image_array)[0][0]
    return "🔥 Wildfire Confirmed!" if prediction > 0.5 else "🌿 No Wildfire Detected"

# Nearby services
def get_nearby_services(lat, lon, service_type):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=50000&type={service_type}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    services = []
    for place in data.get("results", [])[:5]:
        name = place["name"]
        place_lat = place["geometry"]["location"]["lat"]
        place_lon = place["geometry"]["location"]["lng"]
        maps_link = f"https://www.google.com/maps/dir/?api=1&destination={place_lat},{place_lon}"
        services.append((name, maps_link, place_lat, place_lon))
    return services

# Main Display
if os.path.exists("wildfire_real_time.jpg"):
    with st.container():
        st.markdown("## 🌍 Location Intelligence")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Latitude", value=f"{lat:.4f}")
            st.metric(label="Longitude", value=f"{lon:.4f}")
            st.metric(label="Captured Date", value=date_captured)

            m = folium.Map(location=[lat, lon], zoom_start=6)
            wildfire_marker = folium.Marker([lat, lon], tooltip="Detected Wildfire", icon=folium.Icon(color="red"))
            wildfire_marker.add_to(m)

            for service_type, icon_color in zip(["fire_station", "hospital", "police"], ["blue", "green", "purple"]):
                services = get_nearby_services(lat, lon, service_type)
                for name, link, s_lat, s_lon in services:
                    folium.Marker([s_lat, s_lon], popup=name, icon=folium.Icon(color=icon_color)).add_to(m)

            # Add interactive legend
            legend_html = '''
                <div id="map-legend" style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background: rgba(0,0,0,0.6); padding: 10px; border-radius: 8px; color: white;">
                    <b onclick=\"document.querySelector('#map-legend-details').style.display=(document.querySelector('#map-legend-details').style.display=='none'?'block':'none')\" style='cursor: pointer;'>Map Legend ⮟</b>
                    <div id="map-legend-details" style="display:none; margin-top: 5px;">
                        🔴 Wildfire<br>
                        🔵 Fire Station<br>
                        🏥 Hospital<br>
                        🚔 Police
                    </div>
                </div>
            '''
            legend = MacroElement()
            legend._template = Template(f"""{{% macro html(this, kwargs) %}}{legend_html}{{% endmacro %}}""")
            m.get_root().add_child(legend)

            st_folium(m, width=600, height=400)

        with col2:
            st.markdown("### 🛰️ Satellite Image")
            st.image("wildfire_real_time.jpg", use_container_width=True, caption="Live NASA Feed")

    st.markdown("## 🧠 Model Prediction")
    result = predict()
    st.success(result)

    st.markdown(f"""
- **Fire Name**: `{selected_fire.get('fire_name', 'N/A')}`  
- **Status**: `{selected_fire.get('status', 'N/A')}`  
- **Area Burned**: `{selected_fire.get('area_burned_km2', 'N/A')} km^2`  
- **Start Date**: `{selected_fire.get('start_date', 'N/A')}`  
- **Containment**: `{selected_fire.get('containment', 'N/A')}`  
- **Wind**: `{selected_fire.get('wind_direction', 'N/A')}`  
- **Responders**: `{selected_fire.get('response_units', 'N/A')}`
""")

    st.markdown("## 🚨 Nearby Emergency Services")
    col1, col2, col3 = st.columns(3)

    for col, service_type, label in zip([col1, col2, col3], ["fire_station", "hospital", "police"], ["🚒 Fire Stations", "🏥 Hospitals", "🛡️ Police"]):
        with col:
            st.markdown(f"#### {label}")
            for name, link, _, _ in get_nearby_services(lat, lon, service_type):
                st.markdown(f"- [{name}]({link})")
else:
    st.error("❌ Satellite image could not be fetched.")
