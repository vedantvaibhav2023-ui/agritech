# AgriIntel — Simple UI + A4 PDF + Google Drive Models + Gemini 1.5 Pro
# Deploy-ready Streamlit app

import os, io, tempfile, textwrap, datetime as dt
import requests, pandas as pd, numpy as np, joblib, qrcode
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fpdf import FPDF
import google.generativeai as genai

APP_NAME = "AgriIntel"
st.set_page_config(page_title=APP_NAME, page_icon="🌱", layout="wide")

# ---------------------------
# Languages (UI)
# ---------------------------
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
        "wait": "Analyzing and generating with Google Gemini…",
        "info": "Enter data in the sidebar and click Generate.",
    },
    "मराठी (Marathi)": {
        "title": "स्मार्ट कृषी शिफारस प्रणाली",
        "sidebar": "शेतीची माहिती",
        "region": "प्रदेश / ठिकाण",
        "n": "नायट्रोजन (N)",
        "p": "फॉस्फरस (P)",
        "k": "पोटॅशियम (K)",
        "temp": "तापमान (°C)",
        "hum": "आर्द्रता (%)",
        "ph": "मातीचा pH",
        "rain": "पर्जन्यमान (mm)",
        "gen": "अहवाल तयार करा",
        "rep": "तुमचा कृषी अहवाल",
        "wait": "Google Gemini सह विश्लेषण…",
        "info": "साइडबारमध्ये माहिती भरा आणि Generate क्लिक करा.",
    },
    "हिन्दी (Hindi)": {
        "title": "स्मार्ट कृषि सिफारिश प्रणाली",
        "sidebar": "खेत के इनपुट",
        "region": "क्षेत्र / स्थान",
        "n": "नाइट्रोजन (N)",
        "p": "फॉस्फोरस (P)",
        "k": "पोटैशियम (K)",
        "temp": "तापमान (°C)",
        "hum": "आर्द्रता (%)",
        "ph": "मिट्टी pH",
        "rain": "वर्षा (mm)",
        "gen": "विस्तृत रिपोर्ट बनाएँ",
        "rep": "आपकी व्यापक कृषि रिपोर्ट",
        "wait": "Google Gemini के साथ विश्लेषण…",
        "info": "डाटा भरें और Generate पर क्लिक करें.",
    },
}

# ---------------------------
# Gemini (1.5 Pro)
# ---------------------------
GKEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEM = None
if GKEY:
    genai.configure(api_key=GKEY)
    GEM = genai.GenerativeModel("gemini-1.5-pro")

def gemini_advisory(lang_label, user_data: dict, preds: dict) -> str:
    if not GEM:
        return "(Gemini not configured — add GEMINI_API_KEY in Secrets)"
    translate = "" if lang_label == "English" else f"Translate the final report into {lang_label} only (no code-switching)."
    prompt = f\"\"\"You are a senior agronomist. Produce a concise, farmer-friendly report.

FARM DATA:
{user_data}

AI PREDICTIONS:
{preds}

Write these sections in Markdown:
1) Executive Summary
2) Soil Health Analysis
3) Crop Recommendation (why suitable)
4) Fertilizer Plan (quantities, schedule, method)
5) Irrigation & Weather Tips
6) Long-term Soil Improvements
7) Expected Yield & Risks

Keep it actionable, clear, and localized. {translate}
\"\"\"
    try:
        r = GEM.generate_content(prompt)
        return (r.text or "").strip()
    except Exception as e:
        return f"(Gemini error: {e})"

# ---------------------------
# Google Drive model loading
# ---------------------------
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
    try:
        CROP_MODEL = load_drive_model(DRIVE["crop"])
        FERT_MODEL = load_drive_model(DRIVE["fert"])
        SOIL_MODEL = load_drive_model(DRIVE["soil"])
        SOIL_ENCODER = load_drive_model(DRIVE["soil_encoder"])
        YIELD_MODEL = load_drive_model(DRIVE["yield"])
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

# ---------------------------
# Weather helpers (Open-Meteo)
# ---------------------------
@st.cache_data
def geocode(place: str):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 1}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"): raise ValueError("Location not found")
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
    r = requests.get("https://seasonal-api.open-meteo.com/v1/seasonal", params={
        "latitude": lat, "longitude": lon,
        "models": "ecmwf_seas5",
        "monthly": "temperature_2m_mean,precipitation_sum",
        "start_date": start, "end_date": end, "timezone": "auto"
    }, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------
# Local model helpers
# ---------------------------
def predict_crop(N,P,K,temperature,humidity,ph,rainfall):
    return CROP_MODEL.predict([[N,P,K,temperature,humidity,ph,rainfall]])[0]

def predict_fert(crop, N,P,K):
    feats = getattr(FERT_MODEL, "feature_names_in_", [])
    row = {f:0 for f in feats}
    for k,v in {"N":N,"P":P,"K":K}.items():
        if k in row: row[k]=v
    for f in feats:
        if f == f"crop_{crop}": row[f]=1
    X = pd.DataFrame([row], columns=feats)
    pred = FERT_MODEL.predict(X)[0]
    return {"N": round(pred[0],2), "P": round(pred[1],2), "K": round(pred[2],2)}

def predict_soil(N,P,K,ph):
    pred = SOIL_MODEL.predict([[N,P,K,ph]])[0]
    return SOIL_ENCODER.inverse_transform([pred])[0]

def predict_yield(crop,N,P,K,temperature,humidity,ph,rainfall):
    feats = getattr(YIELD_MODEL, "feature_names_in_", [])
    row = {f:0 for f in feats}
    for k,v in {"N":N,"P":P,"K":K,"temperature":temperature,"humidity":humidity,"ph":ph,"rainfall":rainfall}.items():
        if k in row: row[k]=v
    for f in feats:
        if f == f"crop_{crop}": row[f]=1
    X = pd.DataFrame([row], columns=feats)
    return round(float(YIELD_MODEL.predict(X)[0]),2)

# ---------------------------
# Charts
# ---------------------------
def _save_fig(fig):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); return path

def chart_temp(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(7.5,3))
    ax.plot(pd.to_datetime(df["date"]), df["tmax"], label="Tmax (°C)")
    ax.plot(pd.to_datetime(df["date"]), df["tmin"], label="Tmin (°C)")
    ax.legend(); ax.set_title("16-Day Temperature"); ax.set_ylabel("°C")
    fig.autofmt_xdate()
    return _save_fig(fig)

def chart_rain(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(7.5,3))
    ax.bar(pd.to_datetime(df["date"]), df["precip"])
    ax.set_title("16-Day Rainfall (mm)"); ax.set_ylabel("mm")
    fig.autofmt_xdate()
    return _save_fig(fig)

# ---------------------------
# PDF (A4, simple & clean)
# ---------------------------
def wrap_text(t, width=95):
    out = []
    for p in (t or "").splitlines():
        out.extend(textwrap.wrap(p, width=width, break_long_words=True, break_on_hyphens=False) or [""])
    return out

class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", size=9)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "C")

def build_pdf(payload: dict) -> bytes:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    pdf.add_font("DejaVu","",font_path,uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.set_left_margin(20); pdf.set_right_margin(20)
    SAFE_W = 170  # A4 width 210 - 40 margins

    # Cover (simple A4)
    pdf.set_font("DejaVu", size=22)
    pdf.cell(SAFE_W, 12, f"{APP_NAME} Advisory Report", ln=True, align="C")
    pdf.set_font("DejaVu", size=12)
    pdf.ln(4)
    pdf.multi_cell(SAFE_W, 7, f"Region: {payload['region']}")
    pdf.multi_cell(SAFE_W, 7, f"Location: {payload['loc']}")
    pdf.multi_cell(SAFE_W, 7, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Summary
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W, 10, "Summary", ln=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(SAFE_W, 7, f"Recommended Crop: {payload['crop']}")
    pdf.multi_cell(SAFE_W, 7, f"Soil Health: {payload['soil']}")
    pdf.multi_cell(SAFE_W, 7, f"Predicted Yield: {payload['yield']} t/ha")
    fert = payload["fert"]
    p_label = f\"P={fert['P']} kg/ha (reduce)\" if fert['P'] < 0 else f\"P={fert['P']} kg/ha\"
    pdf.multi_cell(SAFE_W, 7, f\"Fertilizer Plan: N={fert['N']} kg/ha | {p_label} | K={fert['K']} kg/ha\")

    # Weather
    pdf.ln(3); pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W, 10, "Weather (Now)", ln=True)
    pdf.set_font("DejaVu", size=12)
    cur = payload["cur"]
    pdf.multi_cell(SAFE_W, 7, f"Temperature: {cur.get('t')} °C")
    pdf.multi_cell(SAFE_W, 7, f"Humidity: {cur.get('h')} %")
    pdf.multi_cell(SAFE_W, 7, f"Precipitation: {cur.get('r')} mm")

    # Charts (single page, simple A layout)
    if payload.get("temp_chart") or payload.get("rain_chart"):
        pdf.add_page()
        pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W, 10, "Forecast Charts", ln=True)
        if payload.get("temp_chart") and os.path.exists(payload["temp_chart"]):
            pdf.ln(2); pdf.image(payload["temp_chart"], w=SAFE_W)
        if payload.get("rain_chart") and os.path.exists(payload["rain_chart"]):
            pdf.ln(4); pdf.image(payload["rain_chart"], w=SAFE_W)

    # Advisory
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W, 10, "AI Advisory", ln=True)
    pdf.set_font("DejaVu", size=10)
    for line in wrap_text(payload["advisory"], 100):
        pdf.multi_cell(SAFE_W, 6, line)

    # QR
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_W, 10, "Open App", ln=True)
    url = os.getenv("APP_URL", "https://share.streamlit.io/")
    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(SAFE_W, 7, f"Scan to open: {url}")
    img = qrcode.make(url)
    qpath = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    img.save(qpath)
    pdf.ln(4); pdf.image(qpath, w=60)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name); tmp.seek(0)
        return tmp.read()

# ---------------------------
# UI (simple like reference)
# ---------------------------
sel = st.sidebar.selectbox("Language", list(LANG.keys()))
L = LANG[sel]

st.title(L["title"])
st.caption("Simple, multilingual, Gemini-powered advisory with A4 PDF.")

st.sidebar.header(L["sidebar"])
region = st.sidebar.text_input(L["region"], "Pune, India")
N = st.sidebar.number_input(L["n"], 0, 300, 90)
P = st.sidebar.number_input(L["p"], 0, 300, 40)
K = st.sidebar.number_input(L["k"], 0, 300, 40)
T = st.sidebar.number_input(L["temp"], -10.0, 60.0, 25.0, format="%.2f")
H = st.sidebar.number_input(L["hum"], 0.0, 100.0, 75.0, format="%.2f")
PH = st.sidebar.number_input(L["ph"], 0.0, 14.0, 6.5, format="%.2f")
R = st.sidebar.number_input(L["rain"], 0.0, 1000.0, 200.0, format="%.2f")

if st.sidebar.button(L["gen"]):
    # model fields only
    fields = {"N":N,"P":P,"K":K,"temperature":T,"humidity":H,"ph":PH,"rainfall":R}

    # predictions
    crop = predict_crop(**fields)
    fert = predict_fert(crop, N, P, K)
    soil = predict_soil(N, P, K, PH)
    ypred = predict_yield(crop, **fields)

    # weather
    cur = {"t": None, "h": None, "r": None}
    name, cc, lat, lon = region, "", None, None
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
    except Exception as e:
        st.warning(f"Weather lookup issue: {e}")
        df16 = pd.DataFrame(columns=["date","tmax","tmin","precip"])

    # charts
    temp_chart = chart_temp(df16)
    rain_chart = chart_rain(df16)

    # advisory
    user_data = f"N={N}, P={P}, K={K}, pH={PH}, Temp={T}, Hum={H}, Rain={R}, Region={region}"
    preds = f"Soil={soil}, Crop={crop}, Fert={fert}, Yield={ypred}"
    with st.spinner(L["wait"]):
        advisory = gemini_advisory(sel, {"inputs":user_data}, {"predictions":preds})

    # on-page
    st.subheader(L["rep"])
    st.write({
        "Crop": crop,
        "Soil Health": soil,
        "Fertilizer (kg/ha)": {"N":fert["N"], "P": f"{fert['P']} (reduce)" if fert["P"]<0 else fert["P"], "K":fert["K"]},
        "Predicted Yield (t/ha)": ypred
    })
    st.markdown("### AI Advisory")
    st.markdown(advisory)

    # pdf
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
        "advisory": advisory
    })
    st.download_button("⬇️ Download A4 PDF", data=pdf_bytes, file_name="AgriIntel_Report_A4.pdf", mime="application/pdf")

else:
    st.info(L["info"])
