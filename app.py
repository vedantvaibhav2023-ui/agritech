import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import io
import tempfile
from fpdf import FPDF
import datetime as dt

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AgriTech AI", layout="wide")

LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta"
}

# -------------------------
# GEMINI (REST API)
# -------------------------
def gemini_generate(text, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    payload = {
        "contents": [
            {"parts": [{"text": text}]}
        ]
    }

    try:
        r = requests.post(
            f"{url}?key={api_key}",
            json=payload,
            timeout=30
        )
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"(Gemini error: {e})"


# -------------------------
# LOAD ML MODELS (Google Drive)
# -------------------------
@st.cache_resource
def load_drive_model(url):
    r = requests.get(url)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

MODEL_URLS = {
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
    "soil_encoder": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    "crop": "https://drive.google.com/uc?export=download&id=10y_phgu-8AV-gdH2K47TqOAw37L7vr-b"
}

with st.spinner("Loading ML models…"):
    crop_model = load_drive_model(MODEL_URLS["crop"])
    fert_model = load_drive_model(MODEL_URLS["fert"])
    soil_model = load_drive_model(MODEL_URLS["soil"])
    soil_encoder = load_drive_model(MODEL_URLS["soil_encoder"])
    yield_model = load_drive_model(MODEL_URLS["yield"])


# -------------------------
# WEATHER — Open Meteo
# -------------------------
def geocode(place):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": place, "count": 1})
    data = r.json()
    if not data.get("results"):
        raise ValueError("Location not found")
    d = data["results"][0]
    return d["latitude"], d["longitude"], d["name"], d["country_code"]


def get_realtime_and_daily(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 16,
        "timezone": "auto"
    })
    return r.json()


def get_seasonal(lat, lon):
    start = dt.date.today().replace(day=1)
    end = (pd.Timestamp(start) + pd.DateOffset(months=3)).date()

    r = requests.get("https://seasonal-api.open-meteo.com/v1/seasonal", params={
        "latitude": lat,
        "longitude": lon,
        "models": "ecmwf_seas5",
        "monthly": "temperature_2m_mean,precipitation_sum",
        "start_date": start,
        "end_date": end,
        "timezone": "auto"
    })
    return r.json()


# -------------------------
# PREDICTION FUNCTIONS
# -------------------------
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    return crop_model.predict(X)[0]


def predict_fertilizer(crop, N, P, K):
    feats = fert_model.feature_names_in_
    row = {f: 0 for f in feats}

    for k, v in {"N": N, "P": P, "K": K}.items():
        if k in row:
            row[k] = v

    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1

    X = pd.DataFrame([row], columns=feats)
    pred = fert_model.predict(X)[0]

    def fix(x):
        return f"Reduce by {abs(round(x,2))} kg/ha" if x < 0 else f"Add {round(x,2)} kg/ha"

    return {
        "N_action": fix(pred[0]),
        "P_action": fix(pred[1]),
        "K_action": fix(pred[2])
    }


def predict_soil(N, P, K, ph):
    pred = soil_model.predict([[N, P, K, ph]])[0]
    return soil_encoder.inverse_transform([pred])[0]


def predict_yield(crop, N, P, K, temperature, humidity, ph, rainfall):
    feats = yield_model.feature_names_in_
    row = {f: 0 for f in feats}

    for k, v in {"N": N, "P": P, "K": K,
                 "temperature": temperature,
                 "humidity": humidity,
                 "ph": ph,
                 "rainfall": rainfall}.items():
        if k in row:
            row[k] = v

    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1

    X = pd.DataFrame([row], columns=feats)
    return round(float(yield_model.predict(X)[0]), 2)


# -------------------------
# UI — SIDEBAR
# -------------------------
st.sidebar.title("AgriTech AI")
language = st.sidebar.selectbox("Language", list(LANGS.keys()))

region = st.sidebar.text_input("Region", "Pune, India")

N = st.sidebar.number_input("Nitrogen (N)", 0, 300, 90)
P = st.sidebar.number_input("Phosphorus (P)", 0, 300, 40)
K = st.sidebar.number_input("Potassium (K)", 0, 300, 40)
temperature = st.sidebar.number_input("Temperature (°C)", -5.0, 60.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 70.0)
ph = st.sidebar.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

API_KEY = st.secrets["GEMINI_API_KEY"]

st.title("🌾 AgriTech AI — Smart Agriculture Assistant")
clicked = st.button("🔍 Analyze")

if clicked:
    try:
        lat, lon, loc_name, cc = geocode(region)
        weather = get_realtime_and_daily(lat, lon)
        seasonal = get_seasonal(lat, lon)
    except Exception as e:
        st.error(f"Weather error: {e}")
        st.stop()

    crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    fert = predict_fertilizer(crop, N, P, K)
    soil_h = predict_soil(N, P, K, ph)
    yld = predict_yield(crop, N, P, K, temperature, humidity, ph, rainfall)

    st.subheader("✅ Results")
    st.write("**Recommended Crop:**", crop)
    st.write("**Soil Health:**", soil_h)
    st.write("**Yield Prediction:**", yld, "t/ha")
    st.write("**Fertilizer Actions:**", fert)

    # AI explanation
    prompt = f"""
    You are an agricultural expert. Explain the recommendations for:
    Crop: {crop}
    Soil: {soil_h}
    Yield: {yld}
    Fertilizer Advice: {fert}
    Weather: {weather['current']}
    Seasonal Forecast: {seasonal.get("monthly",{})}

    Write in {language}. Make it simple, farmer-friendly.
    """

    explanation = gemini_generate(prompt, API_KEY)
    st.write("### 🧠 AI Advisory")
    st.write(explanation)

    # -------------------------
    # PDF
    # -------------------------
    pdf = FPDF("P", "mm", "A4")
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 8, f"AgriTech AI Report - Region: {loc_name}")
    pdf.multi_cell(0, 8, f"Crop Recommendation: {crop}")
    pdf.multi_cell(0, 8, f"Soil Health: {soil_h}")
    pdf.multi_cell(0, 8, f"Yield Prediction: {yld} t/ha")
    pdf.multi_cell(0, 8, f"Fertilizer Actions: {fert}")

    pdf.ln(5)
    pdf.multi_cell(0, 8, "AI Advisory:")
    pdf.multi_cell(0, 6, explanation)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        st.download_button(
            "📄 Download PDF",
            data=tmp.read(),
            file_name="AgriTech_Report.pdf",
            mime="application/pdf"
        )
