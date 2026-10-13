import os
from datetime import datetime
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="centered")
st.title("🚲 Bike Rental Demand Prediction")
st.write("Predict hourly bike rentals using time information and automatically fetched weather.")

MODEL_PATH = "artifacts/best_bike_model.joblib"
FEATURE_PATH = "artifacts/bike_features.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run the notebook first and keep the artifacts folder beside app.py.")
    st.stop()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

@st.cache_data(ttl=600)
def fetch_weather(city):
    # Geocode city name, then fetch current weather from Open-Meteo.
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo = requests.get(geo_url, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=15)
    geo.raise_for_status()
    places = geo.json().get("results", [])
    if not places:
        raise ValueError("City not found. Try a larger nearby city.")
    place = places[0]
    weather_url = "https://api.open-meteo.com/v1/forecast"
    response = requests.get(
        weather_url,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto"
        },
        timeout=15
    )
    response.raise_for_status()
    return place, response.json()["current"]

def weather_category(code, precipitation):
    # Approximate mapping to the UCI dataset's weather situation codes:
    # 1 clear/mostly clear; 2 mist/cloudy; 3 light rain/snow; 4 severe weather.
    if precipitation >= 5 or code >= 95:
        return 4
    if precipitation > 0 or code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86]:
        return 3
    if code in [1, 2, 3, 45, 48]:
        return 2 if code in [3, 45, 48] else 1
    return 1

with st.sidebar:
    st.header("Location and time")
    city = st.text_input("City for automatic weather", value="Nagpur")
    chosen_date = st.date_input("Date", value=datetime.now().date())
    hour = st.slider("Hour of day", min_value=0, max_value=23, value=datetime.now().hour)
    season = st.selectbox("Season", options=[1, 2, 3, 4],
                          format_func=lambda x: {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}[x])
    holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    workingday = st.selectbox("Working day?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("Fetch current weather"):
    try:
        place, weather = fetch_weather(city.strip())
        st.session_state["weather_place"] = place
        st.session_state["weather_data"] = weather
        st.success(f"Weather loaded for {place['name']}, {place.get('country', '')}.")
    except Exception as e:
        st.error(f"Could not fetch weather: {e}")

weather = st.session_state.get("weather_data")
place = st.session_state.get("weather_place")

if weather:
    st.subheader("Current weather")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{weather['temperature_2m']} °C")
    c2.metric("Humidity", f"{weather['relative_humidity_2m']}%")
    c3.metric("Wind speed", f"{weather['wind_speed_10m']} km/h")
    st.caption(f"Weather location: {place['name']}, {place.get('country', '')} · Updated: {weather.get('time', 'latest')}")

    if st.button("Predict bike rentals", type="primary"):
        dt = datetime.combine(chosen_date, datetime.min.time())
        month = chosen_date.month
        weekday = chosen_date.weekday()
        # UCI season: 1 winter, 2 spring, 3 summer, 4 fall. Keep the user's season choice.
        # The year feature in the source dataset is a binary year indicator; use 1 for current-era dates.
        row = {
            "season": season,
            "yr": 1,
            "mnth": month,
            "hr": hour,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weather_category(weather.get("weather_code", 0), weather.get("precipitation", 0)),
            "temp": float(weather["temperature_2m"]) / 41.0,
            "atemp": float(weather["apparent_temperature"]) / 50.0,
            "hum": float(weather["relative_humidity_2m"]) / 100.0,
            "windspeed": float(weather["wind_speed_10m"]) / 67.0
        }
        input_df = pd.DataFrame([row], columns=features)
        prediction = max(0, float(model.predict(input_df)[0]))
        st.success(f"Estimated bike rentals for this hour: **{round(prediction):,} bikes**")
        st.caption("This is an estimate, not a guaranteed count. Weather scaling and season mapping are approximate.")
else:
    st.info("Enter a city and click **Fetch current weather** first. Then select the time details and predict.")
