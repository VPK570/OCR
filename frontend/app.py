"""
Prescription OCR Frontend - Beautiful Streamlit Application

A modern, professional UI for handwritten prescription OCR.
Run with: streamlit run frontend/app.py
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime
from io import BytesIO

import numpy as np
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from pipeline.medical.abbreviations import expand_text, get_abbreviation_count
from frontend.utils import (
    load_image_from_upload,
    create_sample_result,
    export_to_json,
    export_to_txt,
    export_to_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RxScanner Pro - Prescription OCR",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* === CSS Variables === */
    :root {
        --primary: #6366F1;
        --primary-dark: #4F46E5;
        --secondary: #8B5CF6;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-main: #F8FAFC;
        --bg-card: #FFFFFF;
        --text-main: #1E293B;
        --text-muted: #64748B;
        --border: #E2E8F0;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --radius: 12px;
    }

    /* === Dark Mode === */
    [data-theme="dark"] {
        --bg-main: #0F172A;
        --bg-card: #1E293B;
        --text-main: #F1F5F9;
        --text-muted: #94A3B8;
        --border: #334155;
    }

    /* === Typography === */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* === Upload Zone === */
    .upload-zone {
        border: 2px dashed var(--primary);
        border-radius: var(--radius);
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        border-color: var(--primary-dark);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
    }

    /* === Cards === */
    .result-card {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
    }

    /* === Metric Cards === */
    .metric-card {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* === Line Card === */
    .line-card {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .line-text {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-main);
    }

    .line-expanded {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-top: 0.5rem;
        padding-left: 1rem;
        border-left: 2px solid var(--success);
    }

    /* === Confidence Badge === */
    .conf-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .conf-high {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success);
    }

    .conf-medium {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
    }

    .conf-low {
        background: rgba(239, 68, 68, 0.15);
        color: var(--danger);
    }

    /* === Drug Highlight === */
    .drug-highlight {
        background: rgba(99, 102, 241, 0.15);
        color: var(--primary);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 600;
    }

    /* === Sidebar === */
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid var(--border);
    }

    /* === Buttons === */
    .stButton > button {
        border-radius: var(--radius);
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* === Footer === */
    .footer {
        text-align: center;
        color: var(--text-muted);
        padding: 2rem 0;
        font-size: 0.85rem;
    }

    /* === Spinner === */
    .stSpinner > div {
        border-color: var(--primary);
    }

    /* === Hide default Streamlit elements === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* === Scrollbar === */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-main);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo/Title
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2rem;">💊</span>
        <h2 style="margin: 0.5rem 0;">RxScanner</h2>
        <p style="color: var(--text-muted); font-size: 0.9rem;">Prescription OCR</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")

    selected_model = st.selectbox(
        "Recognition Model",
        options=list(config.RECOGNIZERS.keys()),
        index=0,
        help="TrOCR is recommended for best accuracy"
    )

    use_abbreviations = st.checkbox(
        "Expand Medical Abbreviations",
        value=True,
        help="Automatically expand BD, TDS, PRN, etc."
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Lines below this confidence will be flagged"
    )

    st.markdown("---")

    # History
    st.markdown("### 📋 Recent Scans")

    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"🕐 {item['time']}")
                    st.code(item['preview'][:35] + "..." if len(item['preview']) > 35 else item['preview'], language=None)
                with col2:
                    st.markdown(f"<span class='conf-badge conf-{'high' if item['confidence'] > 0.8 else 'medium' if item['confidence'] > 0.6 else 'low'}'>{item['confidence']:.0%}</span>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("No scans yet. Upload a prescription to get started!")

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Quick Stats")
    st.caption(f"Total scans: {len(st.session_state.history)}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown('<h1 class="main-title">📸 Prescription OCR Scanner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a handwritten prescription for intelligent text extraction with medical understanding</p>', unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    uploaded_file = st.file_uploader(
        "Drag & drop your prescription image here",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Supported formats: PNG, JPG, JPEG, WebP, BMP",
        label_visibility="collapsed"
    )

# ── Image Preview & Process ───────────────────────────────────────────────────
if uploaded_file is not None:
    # Load image
    img = load_image_from_upload(uploaded_file)
    st.session_state.processed_image = img

    # Display image
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Original Image")
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img,
            use_container_width=True,
        )

    with col2:
        st.markdown("### 📊 Image Info")
        h, w = img.shape[:2]
        st.write(f"**Resolution:** {w} × {h} pixels")
        st.write(f"**Format:** {uploaded_file.type}")
        st.write(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")

    st.markdown("---")

    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🔍 Process Prescription",
            type="primary",
            use_container_width=True
        )

    if process_button:
        with st.spinner("🤖 Processing prescription..."):
            from frontend.utils import run_prescription_ocr

            result = run_prescription_ocr(
                image=img,
                recognizer=selected_model,
                use_abbreviations=use_abbreviations,
                confidence_threshold=confidence_threshold,
            )

            # Store result
            st.session_state.current_result = result
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "preview": result["raw_text"][:50],
                "confidence": result["avg_confidence"],
            })

            st.success("✅ Processing complete!")

# ── Results Section ───────────────────────────────────────────────────────────
if st.session_state.current_result:
    result = st.session_state.current_result

    st.markdown("---")
    st.markdown("## 📊 Recognition Results")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result['total_lines']}</div>
            <div class="metric-label">Text Lines</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        conf_level = "high" if result['avg_confidence'] > 0.8 else "medium" if result['avg_confidence'] > 0.6 else "low"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#10B981' if conf_level=='high' else '#F59E0B' if conf_level=='medium' else '#EF4444'}">
                {result['avg_confidence']:.0%}
            </div>
            <div class="metric-label">Avg. Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result['abbreviations_found']}</div>
            <div class="metric-label">Abbreviations</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1rem;">{result['model']}</div>
            <div class="metric-label">Recognition Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Extracted lines
    st.markdown("### 📝 Extracted Text")

    for line in result["lines"]:
        conf_class = "high" if line["confidence"] > 0.8 else "medium" if line["confidence"] > 0.6 else "low"

        with st.container():
            st.markdown(f"""
            <div class="line-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 600; color: var(--text-muted);">Line {line['line_id']}</span>
                        <div class="line-text" style="margin-top: 0.5rem;">{line['text']}</div>
                        <div class="line-expanded">→ {line['expanded']}</div>
                    </div>
                    <div>
                        <span class="conf-badge conf-{conf_class}">{line['confidence']:.0%}</span>
                        {'<span style="margin-left: 0.5rem; color: var(--warning);">⚠️ Review</span>' if line['flagged'] else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Export Section ──────────────────────────────────────────────────────────
    st.markdown("### 💾 Export Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        json_data = export_to_json(result)
        st.download_button(
            "📥 JSON",
            data=json_data,
            file_name=f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    with col2:
        txt_data = export_to_txt(result)
        st.download_button(
            "📄 TXT",
            data=txt_data,
            file_name=f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col3:
        csv_data = export_to_csv(result)
        st.download_button(
            "📊 CSV",
            data=csv_data,
            file_name=f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col4:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.current_result = None
            st.rerun()

# ── Demo Mode (when no image uploaded) ───────────────────────────────────────
else:
    st.info("👆 Upload a prescription image above to get started, or try the demo below:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Run Demo (Sample Prescription)", use_container_width=True):
            # Demo result
            result = create_sample_result()
            result["model"] = selected_model
            result["abbreviations_found"] = get_abbreviation_count(result["raw_text"])

            if use_abbreviations:
                for line in result["lines"]:
                    line["expanded"] = expand_text(line["text"])

            st.session_state.current_result = result
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "preview": result["raw_text"][:50],
                "confidence": result["avg_confidence"],
            })
            st.rerun()

    with col2:
        if st.button("🔄 Reset History", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Built with Streamlit | Prescription OCR Demo</p>
    <p style="color: var(--text-muted);">Powered by TrOCR + Medical Abbreviation Understanding</p>
</div>
""", unsafe_allow_html=True)