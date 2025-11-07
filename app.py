# ============================================================
# AgriTech • Smart Agriculture Advisory System
# Streamlit • Gemini REST (1.5 Flash) • A4 PDF • Seasonal Weather
# ============================================================

import os, io, json, tempfile, textwrap, datetime as dt
import requests, pandas as pd, numpy as np, joblib, qrcode
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fpdf import FPDF

# ------------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------------
APP_NAME = "AgriTech Advisory"
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🌱")

# ------------------------------------------------------------
# LANGUAGES
# ------------------------------------------------------------
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
    },
    "தமிழ் (Tamil)": {
        "title": "ஸ்மார்ட் விவசாய பரிந்துரை அமைப்பு",
        "sidebar": "பண்ணை உள்ளீடுகள்",
        "region": "பிராந்தியம் / இடம்",
        "n": "நைட்ரஜன் (N)",
        "p": "பாஸ்பரஸ் (P)",
        "k": "பொட்டாசியம் (K)",
        "temp": "வெப்பநிலை (°C)",
        "hum": "ஈரப்பதம் (%)",
        "ph": "மண் pH",
        "rain": "மழை (மிமீ)",
        "gen": "முழுமையான அறிக்கையை உருவாக்கவும்",
        "rep": "உங்கள் விரிவான விவசாய அறிக்கை",
        "wait": "Google Gemini மூலம் அறிக்கை உருவாக்கப்படுகிறது…",
        "info": "சைட்பாரில் விவரங்களை உள்ளிட்டு 'Generate Report' அழுத்தவும்."
    }
}

# ------------------------------------------------------------
# GEMINI (REST API) — NO SDK NEEDED
# ------------------------------------------------------------
GKEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

def gemini_generate_text(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Call Gemini via REST API; always works on Streamlit Cloud."""
    if not GKEY:
        return "(Gemini not configured — add GEMINI_API_KEY in Secrets)"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GKEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        try:
            return f"(Gemini error: {r.status_code} {r.text})"
        except:
            return f"(Gemini error: {e})"

def gemini_advisory(lang_label: str, user_data: dict, preds: dict) -> str:
    translate = "" if lang_label == "English" else f"Translate the entire final report into {lang_label}."
    prompt = f"""
You are an expert agronomist.
Generate a complete farmer-friendly advisory.

FARM DATA:
{json.dumps(user_data, ensure_ascii=False, indent=2)}

AI PREDICTIONS:
{json.dumps(preds, ensure_ascii=False, indent=2)}

Write sections:
1) Executive Summary
2) Soil Health Analysis
3) Crop Recommendation (why suitable)
4) Fertilizer Plan (quantities + schedule)
5) Weather & Irrigation Tips
6) Long-term Soil Improvement
7) Expected Yield & Risks

Use clear bullet points. Keep it actionable and local. {translate}
"""
    return gemini_generate_text(prompt)

# ------------------------------------------------------------
# MODELS (GOOGLE DRIVE)
# ------------------------------------------------------------
DRIVE = {
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
    "soil_encoder": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    "crop": "https://drive.google.com/uc?export=download&id=10y_phgu-8AV-gdH2K47TqOAw37L7vr-b",
}

@st.cache_resource(show_spinner=True)
def load_drive_model(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

with st.spinner("Loading ML models…"):
    CROP_MODEL = load_drive_model(DRIVE["crop"])
    FERT_MODEL = load_drive_model(DRIVE["fert"])
    SOIL_MODEL = load_drive_model(DRIVE["soil"])
    SOIL_ENCODER = load_drive_model(DRIVE["soil_encoder"])
    YIELD_MODEL = load_drive_model(DRIVE["yield"])

# ------------------------------------------------------------
# WEATHER (Open-Meteo)
# ------------------------------------------------------------
@st.cache_data
def geocode(place: str):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": place, "count": 1}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError("Location not found.")
    d = data["results"][0]
    return d["latitude"], d["longitude"], d["name"], d["country_code"]

@st.cache_data
def get_weather(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 16, "timezone": "auto"
    }, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data
def get_seasonal(lat, lon):
    start = dt.date.today().replace(day=1)
    end = (pd.Timestamp(start) + pd.DateOffset(months=3)).date()
    params = {
        "latitude": lat, "longitude": lon,
        "models": "ecmwf_seas5",
        "monthly": "temperature_2m_mean,precipitation_sum",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "auto"
    }
    r = requests.get("https://seasonal-api.open-meteo.com/v1/seasonal", params=params, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

# ------------------------------------------------------------
# ML HELPERS
# ------------------------------------------------------------
def predict_crop(N,P,K,T,H,pH,R):
    return CROP_MODEL.predict([[N,P,K,T,H,pH,R]])[0]

def predict_fert(crop, N,P,K):
    feats = getattr(FERT_MODEL, "feature_names_in_", [])
    row = {f:0 for f in feats}
    for k,v in {"N":N,"P":P,"K":K}.items():
        if k in row: row[k] = v
    for f in feats:
        if f == f"crop_{crop}": row[f] = 1
    X = pd.DataFrame([row], columns=feats)
    pred = FERT_MODEL.predict(X)[0]
    return {"N": round(pred[0],2), "P": round(pred[1],2), "K": round(pred[2],2)}

def predict_soil(N,P,K,pH):
    pred = SOIL_MODEL.predict([[N,P,K,pH]])[0]
    return SOIL_ENCODER.inverse_transform([pred])[0]

def predict_yield(crop, N,P,K,T,H,pH,R):
    feats = getattr(YIELD_MODEL, "feature_names_in_", [])
    row = {f:0 for f in feats}
    for k,v in {"N":N,"P":P,"K":K,"temperature":T,"humidity":H,"ph":pH,"rainfall":R}.items():
        if k in row: row[k] = v
    for f in feats:
        if f == f"crop_{crop}": row[f] = 1
    X = pd.DataFrame([row], columns=feats)
    return round(float(YIELD_MODEL.predict(X)[0]),2)

# ------------------------------------------------------------
# CHART HELPERS
# ------------------------------------------------------------
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
    ax.legend(); ax.set_title("16-Day Temperature Forecast"); ax.set_ylabel("°C")
    fig.autofmt_xdate()
    return _save_fig(fig)

def chart_rain(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(pd.to_datetime(df["date"]), df["precip"])
    ax.set_title("16-Day Rainfall Forecast"); ax.set_ylabel("mm")
    fig.autofmt_xdate()
    return _save_fig(fig)

def chart_season_temp(dfm):
    if dfm.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(pd.to_datetime(dfm["month"]), dfm["t_mean"], marker="o")
    ax.set_title("Seasonal Mean Temperature (3 Months)"); ax.set_ylabel("°C")
    fig.autofmt_xdate()
    return _save_fig(fig)

def chart_season_rain(dfm):
    if dfm.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(pd.to_datetime(dfm["month"]), dfm["p_sum"])
    ax.set_title("Seasonal Rainfall (3 Months)"); ax.set_ylabel("mm")
    fig.autofmt_xdate()
    return _save_fig(fig)

def chart_npk(n,p,k):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["N","P","K"], [n,p,k])
    ax.set_title("NPK Recommendation (kg/ha)")
    return _save_fig(fig)

# ------------------------------------------------------------
# PDF (A4 clean)
# ------------------------------------------------------------
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

    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    pdf.add_font("DejaVu","",font_path,uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.set_left_margin(20); pdf.set_right_margin(20)
    SAFE_W = 170  # 210 - 40 margins

    # Cover
    pdf.set_font("DejaVu", size=20)
    pdf.cell(SAFE_W,10, f"{APP_NAME} Report", ln=True, align="C")
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(SAFE_W, 6, f"Region: {payload['region']}")
    pdf.multi_cell(SAFE_W, 6, f"Location: {payload['loc']}")
    pdf.multi_cell(SAFE_W, 6, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Summary
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W,9,"Summary", ln=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(SAFE_W,6,f"Recommended Crop: {payload['crop']}")
    pdf.multi_cell(SAFE_W,6,f"Soil Health: {payload['soil']}")
    pdf.multi_cell(SAFE_W,6,f"Predicted Yield: {payload['yield']} t/ha")

    fert = payload["fert"]
    def fert_line(x, label):
        return f"Reduce {label} by {abs(x)} kg/ha" if x < 0 else f"Apply {label} {x} kg/ha"
    pdf.multi_cell(SAFE_W,6, f"Fertilizer Plan: {fert_line(fert['N'],'N')} | {fert_line(fert['P'],'P')} | {fert_line(fert['K'],'K')}")

    # Weather (Now)
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W,9,"Current Weather", ln=True)
    pdf.set_font("DejaVu", size=12)
    cur = payload["cur"]
    pdf.multi_cell(SAFE_W,6,f"Temperature: {cur.get('t')} °C")
    pdf.multi_cell(SAFE_W,6,f"Humidity: {cur.get('h')} %")
    pdf.multi_cell(SAFE_W,6,f"Precipitation: {cur.get('r')} mm")

    # 16-day charts
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W,9,"16-Day Forecast", ln=True)
    if payload["temp_chart"]: pdf.image(payload["temp_chart"], w=SAFE_W)
    if payload["rain_chart"]: pdf.ln(4); pdf.image(payload["rain_chart"], w=SAFE_W)

    # Seasonal charts
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W,9,"3-Month Seasonal Outlook", ln=True)
    if payload["season_temp_chart"]: pdf.image(payload["season_temp_chart"], w=SAFE_W)
    if payload["season_rain_chart"]: pdf.ln(4); pdf.image(payload["season_rain_chart"], w=SAFE_W)

    # NPK
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W,9,"NPK Chart", ln=True)
    if payload["npk_chart"]: pdf.image(payload["npk_chart"], w=SAFE_W/2)

    # AI Advisory
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W,9,"AI Advisory", ln=True)
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
    pdf.ln(4); pdf.multi_cell(SAFE_W,6,f"Open App: {url}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name); tmp.seek(0)
        return tmp.read()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
sel = st.sidebar.selectbox("Language", list(LANG.keys()))
L = LANG[sel]

st.title(L["title"])
st.caption("Unified Crop, Fertilizer, Soil, Weather & Seasonal AI Advisory")

st.sidebar.header(L["sidebar"])
region = st.sidebar.text_input(L["region"], "Pune, India")
N = st.sidebar.number_input(L["n"], 0, 300, 90)
P = st.sidebar.number_input(L["p"], 0, 300, 40)
K = st.sidebar.number_input(L["k"], 0, 300, 40)
T = st.sidebar.number_input(L["temp"], -10.0, 60.0, 25.0, format="%.2f")
H = st.sidebar.number_input(L["hum"], 0.0, 100.0, 75.0, format="%.2f")
pH = st.sidebar.number_input(L["ph"], 0.0, 14.0, 6.5, format="%.2f")
R = st.sidebar.number_input(L["rain"], 0.0, 1000.0, 200.0, format="%.2f")

if st.sidebar.button(L["gen"]):
    fields = {"N":N,"P":P,"K":K,"temperature":T,"humidity":H,"ph":pH,"rainfall":R}

    crop = predict_crop(N,P,K,T,H,pH,R)
    fert = predict_fert(crop, N,P,K)
    soil = predict_soil(N,P,K,pH)
    ypred = predict_yield(crop, N,P,K,T,H,pH,R)

    # Weather & seasonal
    cur = {"t": None, "h": None, "r": None}
    df16 = pd.DataFrame(); dfm = pd.DataFrame()
    name, cc = region, ""
    try:
        lat, lon, name, cc = geocode(region)
        w = get_weather(lat, lon)
        c = w.get("current", {})
        cur = {"t": c.get("temperature_2m"), "h": c.get("relative_humidity_2m"), "r": c.get("precipitation")}
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

    # Advisory (REST)
    with st.spinner(L["wait"]):
        advisory = gemini_advisory(sel, fields, {
            "crop": crop, "soil": soil, "fert": fert, "yield": ypred
        })

    # Results
    st.subheader(L["rep"])
    st.write({
        "Recommended Crop": crop,
        "Soil Health": soil,
        "Fertilizer (kg/ha)": {
            "N": f"{fert['N']} (reduce)" if fert["N"] < 0 else fert["N"],
            "P": f"{fert['P']} (reduce)" if fert["P"] < 0 else fert["P"],
            "K": f"{fert['K']} (reduce)" if fert["K"] < 0 else fert["K"],
        },
        "Predicted Yield": ypred
    })

    st.markdown("### AI Advisory")
    st.markdown(advisory)

    # PDF
    pdf_bytes = build_pdf({
        "region": region,
        "loc": f"{name} ({cc})",
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
