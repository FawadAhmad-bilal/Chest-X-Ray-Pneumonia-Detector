import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.models import load_model

st.set_page_config(page_title="PneumoScan · AI X-Ray Analysis", page_icon="🫁", layout="centered")

# ==================== DESIGN SYSTEM ====================
# Palette: deep clinical navy + light-box cyan glow (radiology viewing panel aesthetic)
# Display type: Space Grotesk (technical/clinical) | Body: Inter | Data: JetBrains Mono
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {
        --bg-deep: #0A0F1C;
        --panel: #111A2E;
        --panel-border: #1E2A45;
        --cyan: #4FE0D6;
        --cyan-dim: rgba(79, 224, 214, 0.15);
        --warn: #FB7185;
        --warn-dim: rgba(251, 113, 133, 0.12);
        --safe: #34D399;
        --safe-dim: rgba(52, 211, 153, 0.12);
        --text-primary: #E7ECF5;
        --text-muted: #7C8AAD;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(79, 224, 214, 0.06), transparent 40%),
            radial-gradient(circle at 85% 100%, rgba(79, 224, 214, 0.04), transparent 40%),
            var(--bg-deep);
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* ---------- Header ---------- */
    .eyebrow {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 4px;
        color: var(--cyan);
        text-transform: uppercase;
        margin-bottom: 6px;
        opacity: 0.85;
    }
    .hero-title {
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 40px;
        color: var(--text-primary);
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-title span { color: var(--cyan); }
    .hero-subtitle {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: var(--text-muted);
        font-size: 15px;
        margin-bottom: 36px;
    }

    /* ---------- Upload zone ---------- */
    div[data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1.5px dashed var(--panel-border);
        border-radius: 16px;
        padding: 22px;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--cyan);
        box-shadow: 0 0 24px rgba(79, 224, 214, 0.08);
    }
    div[data-testid="stFileUploader"] section {
        background: transparent;
    }
    div[data-testid="stFileUploader"] label p {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    /* ---------- Light-box viewer (signature element) ---------- */
    .lightbox {
        position: relative;
        margin: 28px auto 8px auto;
        max-width: 420px;
        border-radius: 4px;
        padding: 14px;
        background: #050810;
        border: 1px solid var(--panel-border);
        box-shadow:
            0 0 0 1px rgba(79, 224, 214, 0.08),
            0 0 40px rgba(79, 224, 214, 0.10),
            inset 0 0 30px rgba(0,0,0,0.6);
    }
    .lightbox img {
        width: 100%;
        display: block;
        border-radius: 2px;
        filter: contrast(1.08) brightness(1.02);
    }
    .lightbox::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cyan), transparent);
        animation: scan 2.6s ease-in-out infinite;
        opacity: 0.85;
    }
    @keyframes scan {
        0%   { top: 14px; opacity: 0; }
        10%  { opacity: 0.9; }
        90%  { opacity: 0.9; }
        100% { top: calc(100% - 14px); opacity: 0; }
    }
    .lightbox-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--text-muted);
        text-align: center;
        letter-spacing: 2px;
        margin-top: 10px;
        text-transform: uppercase;
    }

    /* ---------- Result card ---------- */
    .result-card {
        max-width: 420px;
        margin: 26px auto 0 auto;
        border-radius: 14px;
        padding: 24px 26px;
        font-family: 'Inter', sans-serif;
        animation: fadeUp 0.5s ease;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-warn { background: var(--warn-dim); border: 1px solid rgba(251,113,133,0.4); }
    .result-safe { background: var(--safe-dim); border: 1px solid rgba(52,211,153,0.4); }

    .result-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .result-verdict {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 26px;
        margin-bottom: 14px;
    }
    .result-warn .result-verdict { color: var(--warn); }
    .result-safe .result-verdict { color: var(--safe); }

    .conf-track {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.65,0,0.35,1);
    }
    .conf-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: var(--text-primary);
        margin-top: 10px;
        display: flex;
        justify-content: space-between;
    }

    /* ---------- Footer ---------- */
    .footer-note {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: var(--text-muted);
        font-size: 12px;
        margin-top: 40px;
        padding-top: 18px;
        border-top: 1px solid var(--panel-border);
        max-width: 420px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="eyebrow">CNN · Trained on Chest X-Ray Pneumonia Dataset</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Pneumo<span>Scan</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Upload a chest X-ray — the model reads it like a radiologist\'s light-box</p>', unsafe_allow_html=True)

# ==================== MODEL ====================
@st.cache_resource
def get_model():
    return load_model('pneumonia_model.h5')

model = get_model()

# ==================== UPLOAD + INFERENCE ====================
uploaded_file = st.file_uploader("Drop an X-ray image, or browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')

    # Preprocessing — matches training pipeline
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Scanning..."):
        prediction = float(model.predict(img_array)[0][0])

    # Display image inside the light-box, encoded as base64 for the styled div
    import base64
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    st.markdown(f"""
        <div class="lightbox">
            <img src="data:image/png;base64,{img_b64}" />
        </div>
        <div class="lightbox-label">{uploaded_file.name}</div>
    """, unsafe_allow_html=True)

    is_pneumonia = prediction > 0.5
    confidence = prediction * 100 if is_pneumonia else (1 - prediction) * 100
    css_class = "result-warn" if is_pneumonia else "result-safe"
    verdict = "Pneumonia detected" if is_pneumonia else "No pneumonia detected"
    bar_color = "var(--warn)" if is_pneumonia else "var(--safe)"

    st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-label">Model Verdict</div>
            <div class="result-verdict">{verdict}</div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{confidence}%; background:{bar_color};"></div>
            </div>
            <div class="conf-value">
                <span>Confidence</span>
                <span>{confidence:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="footer-note">
        ⚕️ Student ML project for educational purposes — not a substitute for
        professional medical diagnosis. Built with a custom CNN trained from scratch.
    </div>
""", unsafe_allow_html=True)