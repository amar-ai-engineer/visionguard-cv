import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import cv2

from src.analytics import ComplianceEngine
from src.visualization import (
    create_compliance_gauge,
    create_zone_chart,
    create_trend_chart,
    create_alert_table,
    create_hourly_chart,
)

st.set_page_config(
    page_title="VisionGuard AI - PPE Safety Intelligence",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
h1 { font-weight: 800 !important; color: #0f172a !important; letter-spacing: -0.025em; }
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 18px 20px;
    text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.metric-value { font-size: 2rem; font-weight: 800; color: #0ea5e9; }
.metric-label { font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }
.metric-value.danger { color: #ef4444; }
.metric-value.success { color: #22c55e; }
.section-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04); margin-bottom: 16px;
}
.violation-box {
    background: #fef2f2; border-left: 4px solid #ef4444;
    padding: 10px 14px; border-radius: 8px;
    font-size: 0.88rem; color: #991b1b; margin-top: 8px;
}
.safe-box {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 10px 14px; border-radius: 8px;
    font-size: 0.88rem; color: #14532d; margin-top: 8px;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detector():
    from src.detector import SafetyDetector
    return SafetyDetector("ppe_model.pt")


@st.cache_data
def get_compliance_data():
    engine = ComplianceEngine("data/demo_results.json")
    return {
        "summary": engine.get_summary(),
        "zones":   engine.get_zone_breakdown(),
        "trend":   engine.get_compliance_trend(),
        "alerts":  engine.get_alerts(),
        "hourly":  engine.get_hourly_detections(),
    }


def metric_card(value, label, variant=""):
    cls = f"metric-value {variant}".strip()
    return f'<div class="metric-card"><div class="{cls}">{value}</div><div class="metric-label">{label}</div></div>'


st.markdown("# VisionGuard AI")
st.markdown("*Real-time PPE compliance detection - helmet, vest, violations - powered by YOLOv8*")

tab1, tab2, tab3 = st.tabs(["Live Detection", "Compliance Dashboard", "How It Works"])


# ── Tab 1: Live Detection ─────────────────────────────────────────────────────
with tab1:
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### Controls")
        mode = st.radio("Input Source", ["Image Upload", "Video Upload", "Camera"])
        conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        st.markdown("---")
        st.markdown("**Model detects:**")
        st.markdown("- `helmet` (compliant) / `no-helmet` (violation)")
        st.markdown("- `vest` (compliant) / `no-vest` (violation)")
        st.markdown("- `person`")

    with col_main:
        if mode == "Image Upload":
            uploaded = st.file_uploader("Upload a site photo", type=["jpg", "jpeg", "png"])
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(img, caption="Original", use_container_width=True)
                with c2:
                    with st.spinner("Running PPE detection..."):
                        detector = get_detector()
                        annotated, stats = detector.detect_pil(img, conf)
                        st.image(annotated, caption="PPE Detection Result", use_container_width=True)

                # Stats row
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.markdown(metric_card(stats["persons"],    "Persons"),        unsafe_allow_html=True)
                s2.markdown(metric_card(stats["helmets"],    "Helmets", "success"), unsafe_allow_html=True)
                s3.markdown(metric_card(stats["no_helmets"], "No Helmet", "danger" if stats["no_helmets"] > 0 else ""), unsafe_allow_html=True)
                s4.markdown(metric_card(stats["vests"],      "Vests", "success"),   unsafe_allow_html=True)
                s5.markdown(metric_card(stats["no_vests"],   "No Vest", "danger" if stats["no_vests"] > 0 else ""),    unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if stats["violations"] > 0:
                    st.markdown(
                        f'<div class="violation-box"><strong>{stats["violations"]} violation(s) detected</strong> - '
                        f'{stats["no_helmets"]} missing helmet, {stats["no_vests"]} missing vest</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div class="safe-box">All visible workers are PPE compliant</div>', unsafe_allow_html=True)

            else:
                sample = "data/samples/visionguard_sample_1.png"
                if os.path.exists(sample):
                    st.image(sample, caption="Sample — upload an image to run live PPE detection", use_container_width=True)
                else:
                    st.info("Upload a construction site image to detect helmets, vests, and violations.")

        elif mode == "Video Upload":
            uploaded_vid = st.file_uploader("Upload safety footage (.mp4)", type=["mp4"])
            if uploaded_vid:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_vid.read())
                cap = cv2.VideoCapture(tfile.name)
                frame_ph  = st.empty()
                stats_ph  = st.empty()
                detector  = get_detector()
                if st.button("Process Video"):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated, stats = detector.detect(frame_rgb, conf)
                        frame_ph.image(annotated, channels="RGB", use_container_width=True)
                        viol_txt = f"{stats['violations']} violation(s)" if stats["violations"] > 0 else "Compliant"
                        stats_ph.markdown(
                            f"Persons: **{stats['persons']}** | Helmets: **{stats['helmets']}** | "
                            f"No-Helmet: **{stats['no_helmets']}** | Vests: **{stats['vests']}** | "
                            f"No-Vest: **{stats['no_vests']}** | {viol_txt}"
                        )
                    cap.release()

        else:
            st.info("Enable your webcam below for a live PPE safety check.")
            enable = st.checkbox("Enable Camera")
            snap = st.camera_input("Capture frame", disabled=not enable)
            if snap:
                img = Image.open(snap).convert("RGB")
                detector = get_detector()
                annotated, stats = detector.detect_pil(img, conf)
                st.image(annotated, caption="PPE Detection Result", use_container_width=True)
                if stats["violations"] > 0:
                    st.markdown(
                        f'<div class="violation-box">{stats["violations"]} violation(s): '
                        f'{stats["no_helmets"]} no-helmet, {stats["no_vests"]} no-vest</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div class="safe-box">All workers PPE compliant</div>', unsafe_allow_html=True)


# ── Tab 2: Compliance Dashboard ───────────────────────────────────────────────
with tab2:
    data = get_compliance_data()
    s = data["summary"]

    st.markdown("### Today's Safety Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_card(s["total_workers"], "Workers on Site"), unsafe_allow_html=True)
    c2.markdown(metric_card(f"{s['helmet_compliance_rate']*100:.0f}%", "Helmet Rate"), unsafe_allow_html=True)
    c3.markdown(metric_card(f"{s['vest_compliance_rate']*100:.0f}%", "Vest Rate"), unsafe_allow_html=True)
    c4.markdown(metric_card(f"{s['overall_compliance_score']*100:.0f}%", "Overall Compliance"), unsafe_allow_html=True)
    c5.markdown(metric_card(s["violations_today"], "Active Violations", "danger"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_gauge, col_zone = st.columns([1, 2])
    with col_gauge:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(create_compliance_gauge(s["overall_compliance_score"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_zone:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Compliance by Zone**")
        st.plotly_chart(create_zone_chart(data["zones"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_trend, col_hourly = st.columns(2)
    with col_trend:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Compliance Trend (Last 20 Working Days)**")
        st.plotly_chart(create_trend_chart(data["trend"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_hourly:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Hourly Detections & Violations (Today)**")
        st.plotly_chart(create_hourly_chart(data["hourly"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Today's Alert Log")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(create_alert_table(data["alerts"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 3: How It Works ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### How VisionGuard Works")

    with st.expander("The PPE Detection Model", expanded=True):
        st.markdown("""
VisionGuard uses a **YOLOv8n model fine-tuned on construction site PPE data** (sourced from Roboflow Universe).

The model detects 5 classes simultaneously in a single forward pass:

| Class | Meaning |
|-------|---------|
| `helmet` | Worker wearing a hard hat |
| `no-helmet` | Worker without a hard hat — **violation** |
| `vest` | Worker wearing a high-visibility vest |
| `no-vest` | Worker without a vest — **violation** |
| `person` | Person detected (no PPE label) |

Violations (`no-helmet`, `no-vest`) are highlighted in red. Compliance (`helmet`, `vest`) in green.
""")

    with st.expander("How YOLO Works"):
        st.markdown("""
**YOLO (You Only Look Once)** processes the entire image in one forward pass — making it fast enough for real-time video at 30+ FPS on CPU.

- **Backbone** — extracts feature maps at multiple scales (small, medium, large objects)
- **Neck** — fuses features from different depths for better small-object detection
- **Head** — predicts bounding boxes, confidence scores, and class probabilities per grid cell

The nano variant (~3M parameters) runs without a GPU. For 60+ FPS on a live camera feed, use the small (s) or medium (m) variant on a GPU.
""")

    with st.expander("Custom Training on Your Own Site"):
        st.markdown("""
The model included here was trained on public construction site data. To adapt it to your specific site:

1. Capture 500–1,000 images from your cameras
2. Label them with [Roboflow](https://roboflow.com) (free tier works)
3. Fine-tune:
   ```python
   from ultralytics import YOLO
   model = YOLO("ppe_model.pt")   # start from this checkpoint
   model.train(data="your_site.yaml", epochs=30, imgsz=640)
   ```
4. A site-specific model typically improves recall by 10–20% vs. a generic one

The Colab notebook in `notebooks/custom_training.ipynb` walks through this end to end.
""")

    st.markdown("---")
    st.markdown("**Colab Notebook** - fine-tuning on your own PPE dataset: `notebooks/custom_training.ipynb`")
    st.markdown("Built by **Amar Ismail** | [GitHub](https://github.com/amar-ai-engineer) | [LinkedIn](https://www.linkedin.com/in/amar-ai-engineer/)")
