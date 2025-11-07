# ============================================================
# AgriTech • Smart Agriculture Advisory System
# Clean UI • Gemini 1.5 Flash • A4 PDF • Seasonal Weather
# ============================================================

import os, io, tempfile, textwrap, datetime as dt
import requests, pandas as pd, numpy as np, joblib, qrcode
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fpdf import FPDF
import google.generativeai as genai


# ============================================================
# STREAMLIT CONFIG
# ============================================================
APP_NAME = "AgriTech Advisory"
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🌱")


# ============================================================
# LANGUAGE PACK
# ============================================================
LANG = {
    "English": {
        "title": "Smart Agriculture Recommendation System",
        "sidebar": "Farm Inputs",
        "region": "Region / Place",
        "n": "Nitrogen (N)",
        "p": "Phosphorus (P)",
        "k": "Potassium (K)",
        "temp": "Temperature (°C)",
        "hum": "Humidity (%)",
        "ph": "Soil pH",
        "rain": "Rainfall (mm)",
        "gen": "Generate Comprehensive Report",
        "rep": "Your Comprehensive Report",
        "wait": "Analyzing and generating report with Google Gemini…",
        "info": "Enter data in the sidebar and click 'Generate Report'."
    },

    "हिन्दी (Hindi)": {
        "title": "स्मार्ट कृषि सिफारिश प्रणाली",
        "sidebar": "खेत की जानकारी",
        "region": "क्षेत्र / स्थान",
        "n": "नाइट्रोजन (N)",
        "p": "फॉस्फोरस (P)",
        "k": "पोटैशियम (K)",
        "temp": "तापमान (°C)",
        "hum": "आर्द्रता (%)",
        "ph": "मिट्टी का pH",
        "rain": "वर्षा (mm)",
        "gen": "कृषि रिपोर्ट बनाएं",
        "rep": "आपकी कृषि रिपोर्ट",
        "wait": "Google Gemini के साथ रिपोर्ट तैयार हो रही है…",
        "info": "साइडबार में जानकारी भरें और 'Generate Report' क्लिक करें।"
    },

    "मराठी (Marathi)": {
        "title": "स्मार्ट कृषी शिफारस प्रणाली",
        "sidebar": "शेती माहिती",
        "region": "प्रदेश / ठिकाण",
        "n": "नायट्रोजन (N)",
        "p": "फॉस्फरस (P)",
        "k": "पोटॅशियम (K)",
        "temp": "तापमान (°C)",
        "hum": "आर्द्रता (%)",
        "ph": "मातीचा pH",
        "rain": "पर्जन्यमान (mm)",
        "gen": "कृषी अहवाल तयार करा",
        "rep": "तुमचा कृषी अहवाल",
        "wait": "Google Gemini सह अहवाल तयार होत आहे…",
        "info": "साइडबारमध्ये माहिती भरा आणि 'Generate Report' क्लिक करा."
    }
}


# ============================================================
# GEMINI SETUP (Stable model)
# ============================================================
GKEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

GEM = None
if GKEY:
    genai.configure(api_key=GKEY)
    GEM = genai.GenerativeModel("gemini-1.5-flash")


# Generate AI Advisory
def gemini_advisory(lang_label, user_data, preds):
    if not GEM:
        return "(Gemini not configured — add GEMINI_API_KEY in Secrets)"

    translate = "" if lang_label == "English" else \
        f"Translate final report into {lang_label} completely."

    prompt = f"""
You are an expert agronomist.
Generate a complete farmer-friendly advisory.

FARM DATA:
{user_data}

AI PREDICTIONS:
{preds}

Write sections:
1) Executive Summary
2) Soil Health Analysis
3) Crop Recommendation (why suitable)
4) Fertilizer Plan (quantities + schedule)
5) Weather & Irrigation Tips
6) Long-term Soil Improvement
7) Expected Yield & Risks

{translate}
"""

    try:
        r = GEM.generate_content(prompt)
        return (r.text or "").strip()
    except Exception as e:
        return f"(Gemini error: {e})"


# ============================================================
# MODEL DOWNLOAD (Google Drive)
# ============================================================
DRIVE = {
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
    "soil_encoder": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    "crop": "https://drive.google.com/uc?export=download&id=10y_phgu-8AV-gdH2K47TqOAw37L7vr-b"
}


@st.cache_resource
def load_drive_model(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))


# Load all ML models
with st.spinner("Loading ML models…"):
    CROP_MODEL = load_drive_model(DRIVE["crop"])
    FERT_MODEL = load_drive_model(DRIVE["fert"])
    SOIL_MODEL = load_drive_model(DRIVE["soil"])
    SOIL_ENCODER = load_drive_model(DRIVE["soil_encoder"])
    YIELD_MODEL = load_drive_model(DRIVE["yield"])


# ============================================================
# WEATHER SYSTEM (Open-Meteo)
# ============================================================
@st.cache_data
def geocode(place):
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1},
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError("Location not found.")
    d = data["results"][0]
    return d["latitude"], d["longitude"], d["name"], d["country_code"]


@st.cache_data
def get_weather(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 16,
        "timezone": "auto"
    }, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data
def get_seasonal(lat, lon):
    start = dt.date.today().replace(day=1)
    end = (pd.Timestamp(start) + pd.DateOffset(months=3)).date()

    r = requests.get("https://seasonal-api.open-meteo.com/v1/seasonal", params={
        "latitude": lat,
        "longitude": lon,
        "models": "ecmwf_seas5",
        "monthly": "temperature_2m_mean,precipitation_sum",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "auto"
    }, timeout=30)

    if r.status_code != 200:
        return None
    return r.json()


# ============================================================
# PREDICTION HELPERS
# ============================================================
def predict_crop(N,P,K,T,H,pH,R):
    return CROP_MODEL.predict([[N,P,K,T,H,pH,R]])[0]


def predict_fert(crop, N,P,K):
    feats = FERT_MODEL.feature_names_in_
    row = {f:0 for f in feats}

    for k,v in {"N":N,"P":P,"K":K}.items():
        if k in row: row[k] = v

    # one-hot for crop
    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1

    df = pd.DataFrame([row], columns=feats)
    pred = FERT_MODEL.predict(df)[0]

    return {"N": round(pred[0],2), "P": round(pred[1],2), "K": round(pred[2],2)}


def predict_soil(N,P,K,pH):
    pred = SOIL_MODEL.predict([[N,P,K,pH]])[0]
    return SOIL_ENCODER.inverse_transform([pred])[0]


def predict_yield(crop, N,P,K,T,H,pH,R):
    feats = YIELD_MODEL.feature_names_in_
    row = {f:0 for f in feats}

    for k,v in {"N":N,"P":P,"K":K,"temperature":T,
                "humidity":H,"ph":pH,"rainfall":R}.items():
        if k in row: row[k] = v

    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1

    df = pd.DataFrame([row], columns=feats)
    return round(float(YIELD_MODEL.predict(df)[0]),2)


# ============================================================
# CHART HELPERS
# ============================================================
def _save_fig(fig):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_temp(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(pd.to_datetime(df["date"]), df["tmax"], label="Tmax")
    ax.plot(pd.to_datetime(df["date"]), df["tmin"], label="Tmin")
    ax.legend()
    ax.set_title("16-Day Temperature Forecast")
    fig.autofmt_xdate()
    return _save_fig(fig)


def chart_rain(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(pd.to_datetime(df["date"]), df["precip"])
    ax.set_title("16-Day Rainfall Forecast")
    fig.autofmt_xdate()
    return _save_fig(fig)


def chart_season_temp(dfm):
    if dfm.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(pd.to_datetime(dfm["month"]), dfm["t_mean"], marker="o")
    ax.set_title("Seasonal Mean Temperature (3 Months)")
    fig.autofmt_xdate()
    return _save_fig(fig)


def chart_season_rain(dfm):
    if dfm.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(pd.to_datetime(dfm["month"]), dfm["p_sum"])
    ax.set_title("Seasonal Rainfall (3 Months)")
    fig.autofmt_xdate()
    return _save_fig(fig)


def chart_npk(n,p,k):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["N","P","K"], [n,p,k])
    ax.set_title("NPK Recommendation (kg/ha)")
    return _save_fig(fig)


# ============================================================
# PDF BUILDER (A4 Clean)
# ============================================================
def wrap_text(t, width=95):
    out = []
    for p in (t or "").splitlines():
        out.extend(textwrap.wrap(p, width=width, break_long_words=True) or [""])
    return out


class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", size=9)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "C")


def build_pdf(payload):
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # load unicode font
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    pdf.add_font("DejaVu","",font_path,uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.set_left_margin(20); pdf.set_right_margin(20)
    SAFE_W = 170  # 210 - 40 margins

    # Cover Page
    pdf.set_font("DejaVu", size=20)
    pdf.cell(SAFE_W,10, f"{APP_NAME} Report", ln=True, align="C")

    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(SAFE_W, 6, f"Region: {payload['region']}")
    pdf.multi_cell(SAFE_W, 6, f"Location: {payload['loc']}")
    pdf.multi_cell(SAFE_W, 6, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Summary
    pdf.add_page()
    pdf.set_font("DejaVu", size=16)
    pdf.cell(SAFE_W,9,"Summary", ln=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(SAFE_W,6,f"Recommended Crop: {payload['crop']}")
    pdf.multi_cell(SAFE_W,6,f"Soil Health: {payload['soil']}")
    pdf.multi_cell(SAFE_W,6,f"Predicted Yield: {payload['yield']} t/ha")

    fert = payload["fert"]
    def fert_line(x, label):
        return f"Reduce {label} by {abs(x)} kg/ha" if x<0 else f"Apply {label} {x} kg/ha"

    pdf.multi_cell(SAFE_W,6,
        f"Fertilizer Plan: {fert_line(fert['N'],'N')} | {fert_line(fert['P'],'P')} | {fert_line(fert['K'],'K')}")

    # Weather
    cur = payload["cur"]
    pdf.add_page()
    pdf.set_font("DejaVu", size=16)
    pdf.cell(SAFE_W,9,"Current Weather", ln=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(SAFE_W,6,f"Temperature: {cur.get('t')} °C")
    pdf.multi_cell(SAFE_W,6,f"Humidity: {cur.get('h')} %")
    pdf.multi_cell(SAFE_W,6,f"Precipitation: {cur.get('r')} mm")

    # 16-day charts
    pdf.add_page()
    pdf.set_font("DejaVu", size=16)
    pdf.cell(SAFE_W,9,"16-Day Forecast", ln=True)
    if payload["temp_chart"]:
        pdf.image(payload["temp_chart"], w=SAFE_W)
    if payload["rain_chart"]:
        pdf.ln(4); pdf.image(payload["rain_chart"], w=SAFE_W)

    # Seasonal Charts
    pdf.add_page()
    pdf.set_font("DejaVu", size=16)
    pdf.cell(SAFE_W,9,"3-Month Seasonal Outlook", ln=True)
    if payload["season_temp_chart"]:
        pdf.image(payload["season_temp_chart"], w=SAFE_W)
    if payload["season_rain_chart"]:
        pdf.ln(4); pdf.image(payload["season_rain_chart"], w=SAFE_W)

    # NPK chart
    pdf.add_page()
    pdf.set_font("DejaVu", size=16)
    pdf.cell(SAFE_W,9,"NPK Chart", ln=True)
    if payload["npk_chart"]:
        pdf.image(payload["npk_chart"], w=SAFE_W/2)

    # AI Advisory
    pdf.add_page()
    pdf.set_font("DejaVu", size=16)
    pdf.cell(SAFE_W,9,"AI Advisory", ln=True)
    pdf.set_font("DejaVu", size=10)
    for line in wrap_text(payload["advisory"], 100):
        pdf.multi_cell(SAFE_W,5.8,line)

    # QR Page
    pdf.add_page()
    url = "https://agritech-qktplbnzbgow5wccjg9qee.streamlit.app/"
    img = qrcode.make(url)
    qpath = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    img.save(qpath)
    pdf.image(qpath, w=60)
    pdf.ln(5)
    pdf.multi_cell(SAFE_W,6,f"Visit the App: {url}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        return tmp.read()


# ============================================================
# UI
# ============================================================
sel = st.sidebar.selectbox("Language", list(LANG.keys()))
L = LANG[sel]

st.title(L["title"])
st.caption("Unified Crop, Fertilizer, Soil, Weather & Seasonal AI Advisory")

# Sidebar inputs
st.sidebar.header(L["sidebar"])
region = st.sidebar.text_input(L["region"], "Pune, India")
N = st.sidebar.number_input(L["n"], 0, 300, 90)
P = st.sidebar.number_input(L["p"], 0, 300, 40)
K = st.sidebar.number_input(L["k"], 0, 300, 40)
T = st.sidebar.number_input(L["temp"], -10.0, 60.0, 25.0)
H = st.sidebar.number_input(L["hum"], 0.0, 100.0, 75.0)
pH = st.sidebar.number_input(L["ph"], 0.0, 14.0, 6.5)
R = st.sidebar.number_input(L["rain"], 0.0, 1000.0, 200.0)


# ============================================================
# MAIN BUTTON
# ============================================================
if st.sidebar.button(L["gen"]):

    fields = dict(N=N,P=P,K=K,temperature=T,humidity=H,ph=pH,rainfall=R)

    crop = predict_crop(N,P,K,T,H,pH,R)
    fert = predict_fert(crop,N,P,K)
    soil = predict_soil(N,P,K,pH)
    ypred = predict_yield(crop, N,P,K,T,H,pH,R)

    # Weather
    cur = {"t":None,"h":None,"r":None}
    df16 = pd.DataFrame()
    dfm = pd.DataFrame()

    try:
        lat, lon, name, cc = geocode(region)
        w = get_weather(lat, lon)

        c = w.get("current", {})
        cur = {
            "t": c.get("temperature_2m"),
            "h": c.get("relative_humidity_2m"),
            "r": c.get("precipitation"),
        }

        d = w.get("daily", {})
        df16 = pd.DataFrame({
            "date": d.get("time", []),
            "tmax": d.get("temperature_2m_max", []),
            "tmin": d.get("temperature_2m_min", []),
            "precip": d.get("precipitation_sum", []),
        })

        seas = get_seasonal(lat, lon)
        if seas and "monthly" in seas:
            m = seas["monthly"]
            dfm = pd.DataFrame({
                "month": m.get("time", []),
                "t_mean": m.get("temperature_2m_mean", []),
                "p_sum": m.get("precipitation_sum", []),
            }).head(3)

    except Exception as e:
        st.warning(f"Weather lookup issue: {e}")

    # Charts
    temp_chart = chart_temp(df16)
    rain_chart = chart_rain(df16)
    season_temp_chart = chart_season_temp(dfm)
    season_rain_chart = chart_season_rain(dfm)
    npk_chart = chart_npk(fert["N"], fert["P"], fert["K"])

    # Gemini advisory
    with st.spinner(L["wait"]):
        advisory = gemini_advisory(sel, fields, {
            "crop": crop,
            "soil": soil,
            "fert": fert,
            "yield": ypred
        })

    # Show results
    st.subheader(L["rep"])
    st.write({
        "Recommended Crop": crop,
        "Soil Health": soil,
        "Fertilizer (kg/ha)": {
            "N": fert["N"],
            "P": fert["P"],
            "K": fert["K"]
        },
        "Predicted Yield": ypred
    })

    st.markdown("### AI Advisory")
    st.markdown(advisory)

    # Build & download PDF
    pdf_bytes = build_pdf({
        "region": region,
        "loc": f"{name} ({cc})" if "name" in locals() else region,
        "crop": crop,
        "soil": soil,
        "yield": ypred,
        "fert": fert,
        "cur": cur,
        "temp_chart": temp_chart,
        "rain_chart": rain_chart,
        "season_temp_chart": season_temp_chart,
        "season_rain_chart": season_rain_chart,
        "npk_chart": npk_chart,
        "advisory": advisory
    })

    st.download_button("⬇️ Download A4 PDF",
                       data=pdf_bytes,
                       file_name="AgriTech_Report_A4.pdf",
                       mime="application/pdf")

else:
    st.info(L["info"])
