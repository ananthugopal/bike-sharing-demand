import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚴",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/bike_demand_model.pkl")

model = load_model()

# -----------------------------
# App UI
# -----------------------------
st.title("🚴 Bike Sharing Demand Prediction")
st.markdown(
    "Predict the **number of bike rentals** based on weather and time features."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Winter",
            2: "Spring",
            3: "Summer",
            4: "Fall"
        }[x]
    )

    yr = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
    mnth = st.slider("Month", 1, 12)
    weekday = st.slider("Weekday (0=Sun)", 0, 6)
    workingday = st.selectbox("Working Day", [0, 1])

with col2:
    weathersit = st.selectbox(
        "Weather Situation",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Clear",
            2: "Cloudy",
            3: "Rain / Snow"
        }[x]
    )

    holiday = st.selectbox("Holiday", [0, 1])
    temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
    atemp = st.slider("Feels Like Temp (Normalized)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity", 0.0, 1.0, 0.5)
    windspeed = st.slider("Windspeed", 0.0, 1.0, 0.5)

# -----------------------------
# Prepare input dataframe
# -----------------------------
input_data = pd.DataFrame({
    "season": [season],
    "yr": [yr],
    "mnth": [mnth],
    "holiday": [holiday],
    "weekday": [weekday],
    "workingday": [workingday],
    "weathersit": [weathersit],
    "temp": [temp],
    "atemp": [atemp],
    "hum": [hum],
    "windspeed": [windspeed]
})

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🚲 Predict Bike Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"**Predicted Bike Rentals:** {int(prediction)}")

    st.caption(
        "Prediction is based on historical bike sharing data (UCI ML Repository)."
    )
