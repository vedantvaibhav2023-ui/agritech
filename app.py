import os, io, json, tempfile, textwrap, datetime as dt
import requests, pandas as pd, numpy as np, joblib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fpdf import FPDF

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AgriTech AI", page_icon="🌾", layout="wide")
APP = "AgriTech AI"

LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
}

def gemini_generate_text(prompt: str, api_key: str):
    if not api_key:
        return "(Gemini not configured)"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        data = r.json()
        
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"(Gemini error: {r.status_code if 'r' in locals() else ''} {str(e)})"

# =========================
# MODELS — Google Drive
# =========================
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

# =========================
# WEATHER — Open-Meteo
# =========================
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
def get_realtime_and_daily(lat, lon):
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
    # First day of current month to ~3 months ahead (approx 92 days)
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

# =========================
# ML HELPERS
# =========================
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
    def action(x):
        return f"Reduce by {abs(round(x,2))} kg/ha" if x < 0 else f"Add {round(x,2)} kg/ha"
    return {"N": action(pred[0]), "P": action(pred[1]), "K": action(pred[2])}

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

# =========================
# CHART HELPERS (minimal)
# =========================
def _save_fig(fig):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

def chart_temp(df16):
    if df16.empty: return ""
    fig, ax = plt.subplots(figsize=(7.5,2.8))
    ax.plot(pd.to_datetime(df16["date"]), df16["tmax"], label="Tmax")
    ax.plot(pd.to_datetime(df16["date"]), df16["tmin"], label="Tmin")
    ax.set_title("16-Day Temperature"); ax.set_ylabel("°C"); ax.legend()
    fig.autofmt_xdate()
    return _save_fig(fig)

def chart_rain(df16):
    if df16.empty: return ""
    fig, ax = plt.subplots(figsize=(7.5,2.8))
    ax.bar(pd.to_datetime(df16["date"]), df16["precip"])
    ax.set_title("16-Day Rainfall"); ax.set_ylabel("mm")
    fig.autofmt_xdate()
    return _save_fig(fig)

def chart_npk(n_action, p_action, k_action):
    # Extract numeric values for simple visualization (add vs reduce)
    def val(act):
        # "Add 12.3 kg/ha" or "Reduce by 4.0 kg/ha"
        parts = act.lower().replace("kg/ha","").split()
        try:
            if "reduce" in act.lower(): return -float(parts[-1])
            return float(parts[-1])
        except:
            return 0.0
    vals = [val(n_action), val(p_action), val(k_action)]
    fig, ax = plt.subplots(figsize=(4.5,2.8))
    ax.bar(["N","P","K"], vals)
    ax.axhline(0, linewidth=1)
    ax.set_title("NPK Action (kg/ha)")
    return _save_fig(fig)

# =========================
# PDF (A4, simple)
# =========================
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

    # Unicode font (Tamil/Hindi/Marathi)
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    pdf.add_font("DejaVu","",font_path)
    pdf.set_font("DejaVu", size=12)
    SAFE_W = 190  # approx usable width with default margins

    # Header
    pdf.set_font("DejaVu", size=16)
    pdf.cell(0, 10, f"{APP} Report", align="C")
    pdf.ln(12)
    pdf.set_font("DejaVu", size=12)

    # Summary
    pdf.multi_cell(0, 7, f"Region: {payload['loc']}")
    pdf.multi_cell(0, 7, f"Recommended Crop: {payload['crop']}")
    pdf.multi_cell(0, 7, f"Soil Health: {payload['soil']}")
    pdf.multi_cell(0, 7, f"Predicted Yield: {payload['yield']} t/ha")

    fert = payload["fert"]
    pdf.multi_cell(0, 7, f"Fertilizer Actions — N: {fert['N']}, P: {fert['P']}, K: {fert['K']}")

    # Weather snapshot
    cur = payload["cur"]
    pdf.ln(4)
    pdf.multi_cell(0, 7, f"Now — Temp: {cur.get('t')} °C, Humidity: {cur.get('h')} %, Precip: {cur.get('r')} mm")

    # Charts
    if payload["t_chart"]:
        pdf.ln(3); pdf.image(payload["t_chart"], w=SAFE_W)
    if payload["r_chart"]:
        pdf.ln(3); pdf.image(payload["r_chart"], w=SAFE_W)
    if payload["npk_chart"]:
        pdf.ln(3); pdf.image(payload["npk_chart"], w=SAFE_W/2)

    # AI Advisory
    pdf.ln(3)
    pdf.set_font("DejaVu", size=13); pdf.cell(0, 8, "AI Advisory"); pdf.ln(9)
    pdf.set_font("DejaVu", size=11)
    for line in wrap_text(payload["advisory"], width=100):
        pdf.multi_cell(0, 6, line)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        return tmp.read()

# =========================
# UI
# =========================
st.title("🌾 AgriTech AI — Unified Advisory")
left, right = st.columns([2,1], vertical_alignment="top")

with right:
    language = st.selectbox("Language", list(LANGS.keys()), index=0)
    region = st.text_input("Region / Place", "Pune, India")
    N = st.number_input("Nitrogen (N)", 0, 300, 90)
    P = st.number_input("Phosphorus (P)", 0, 300, 40)
    K = st.number_input("Potassium (K)", 0, 300, 40)
    T = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0, format="%.2f")
    H = st.number_input("Humidity (%)", 0.0, 100.0, 75.0, format="%.2f")
    pH = st.number_input("Soil pH", 0.0, 14.0, 6.5, format="%.2f")
    R = st.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0, format="%.2f")
    run = st.button("🔍 Analyze & Generate")

with left:
    st.info("Enter inputs on the right, then click **Analyze & Generate**.\n\nReport includes crop, soil, fertilizer, yield, real-time weather, 16-day forecast, and a 3-month seasonal outlook (if available).")

if run:
    # Weather lookup
    loc_name, cc = region, ""
    cur = {"t": None, "h": None, "r": None}
    df16 = pd.DataFrame()
    try:
        lat, lon, loc_name, cc = geocode(region)
        w = get_realtime_and_daily(lat, lon)
        c = w.get("current", {})
        cur = {"t": c.get("temperature_2m"), "h": c.get("relative_humidity_2m"), "r": c.get("precipitation")}
        d = w.get("daily", {})
        df16 = pd.DataFrame({
            "date": d.get("time", []),
            "tmax": d.get("temperature_2m_max", []),
            "tmin": d.get("temperature_2m_min", []),
            "precip": d.get("precipitation_sum", []),
        })
        seasonal = get_seasonal(lat, lon)  # we don’t plot seasonal in simple PDF; still fetched for advisory text
    except Exception as e:
        st.warning(f"Weather lookup issue: {e}")
        seasonal = None

    # Predictions
    crop = predict_crop(N,P,K,T,H,pH,R)
    fert = predict_fert(crop, N,P,K)
    soil = predict_soil(N,P,K,pH)
    ypred = predict_yield(crop, N,P,K,T,H,pH,R)

    # Show results
    st.subheader("Results")
    colA, colB, colC = st.columns(3)
    colA.metric("🌿 Crop", crop)
    colB.metric("🧪 Soil Health", soil)
    colC.metric("📈 Yield (t/ha)", ypred)

    st.markdown("**Fertilizer Actions (kg/ha)**")
    st.write(f"N: {fert['N']}  |  P: {fert['P']}  |  K: {fert['K']}")

    # Charts for the PDF (and page)
    t_chart = chart_temp(df16)
    r_chart = chart_rain(df16)
    npk_chart = chart_npk(fert["N"], fert["P"], fert["K"])
    if t_chart: st.image(t_chart, caption="16-Day Temperature", use_column_width=True)
    if r_chart: st.image(r_chart, caption="16-Day Rainfall", use_column_width=True)

    # AI advisory
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    user_inputs = {"N":N,"P":P,"K":K,"temperature":T,"humidity":H,"ph":pH,"rainfall":R,"region":region}
    preds = {"crop":crop,"soil":soil,"fert":fert,"yield":ypred,"weather_now":cur,"seasonal":seasonal}
    with st.spinner("Generating advisory with Gemini…"):
        advisory = build_advisory(language, user_inputs, preds, api_key)
    st.markdown("### 🧠 Advisory")
    st.write(advisory)

    # Build and download PDF
    pdf_bytes = build_pdf({
        "loc": f"{loc_name} ({cc})",
        "crop": crop,
        "soil": soil,
        "yield": ypred,
        "fert": fert,
        "cur": cur,
        "t_chart": t_chart,
        "r_chart": r_chart,
        "npk_chart": npk_chart,
        "advisory": advisory
    })
    st.download_button("⬇️ Download A4 PDF", data=pdf_bytes,
                       file_name="AgriTech_Report_A4.pdf", mime="application/pdf")
