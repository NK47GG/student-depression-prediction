"""
app.py — Streamlit App: Prediksi Depresi
Stefanus Loveniko Putra Sinory — Fast Track Bengkel Coding

Cara jalankan:
    streamlit run app.py

Pastikan 'model_depresi_final.pkl' ada di folder yang sama.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Depresi",
    page_icon="🧠",
    layout="centered",
)

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    artifact = joblib.load("model_depresi_final.pkl")
    return artifact

try:
    artifact  = load_model()
    pipeline  = artifact["pipeline"]
    conf_fn   = artifact.get("confidence_fn", None)
except FileNotFoundError:
    st.error("❌ File 'model_depresi_final.pkl' tidak ditemukan. "
             "Pastikan file ada di folder yang sama dengan app.py.")
    st.stop()


# ── Fungsi confidence level (fallback jika tidak ada di artifact) ───
def get_confidence(prob: float) -> str:
    if prob >= 0.75:
        return "Tinggi — Depresi"
    elif prob >= 0.55:
        return "Sedang — Depresi"
    elif prob >= 0.45:
        return "Borderline — Perlu Review"
    elif prob >= 0.25:
        return "Sedang — Tidak Depresi"
    else:
        return "Tinggi — Tidak Depresi"

confidence_fn = conf_fn if conf_fn is not None else get_confidence


# ── UI ─────────────────────────────────────────────────────
st.title("🧠 Prediksi Risiko Depresi")
st.caption("Model berbasis Machine Learning — Fast Track Bengkel Coding")
st.markdown("---")

st.subheader("📋 Isi Data Responden")

col1, col2 = st.columns(2)

with col1:
    age        = st.number_input("Usia", min_value=15, max_value=60, value=21)
    gender     = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    city       = st.selectbox("Kota", ["Tier 1 (Metro)", "Tier 2/3 (Non-Metro)"])
    profession = st.selectbox("Profesi", ["Student", "Working Professional"])
    degree     = st.selectbox("Jenjang Pendidikan",
                               ["B.Tech", "BSc", "BA", "BCA", "BComm",
                                "M.Tech", "MSc", "MA", "MCA", "MComm", "PhD", "Other"])
    sleep_dur  = st.selectbox("Durasi Tidur",
                               ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])

with col2:
    acad_pressure  = st.slider("Tekanan Akademis (1-5)", 1, 5, 3)
    study_sat      = st.slider("Kepuasan Studi (1-5)",   1, 5, 3)
    work_study_hrs = st.slider("Jam Kerja/Belajar per hari", 1, 12, 6)
    financial_str  = st.slider("Stres Finansial (1-5)",  1, 5, 3)
    cgpa           = st.number_input("CGPA (0=tidak ada)", min_value=0.0, max_value=10.0,
                                      value=7.5, step=0.1)
    dietary        = st.selectbox("Kebiasaan Makan", ["Healthy", "Moderate", "Unhealthy"])
    suicidal       = st.selectbox("Pernah punya pikiran suicidal?", ["No", "Yes"])
    family_hist    = st.selectbox("Riwayat keluarga gangguan mental?", ["No", "Yes"])

st.markdown("---")

if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):

    # ── Siapkan input DataFrame ───────────────────────────
    # Feature engineering sama seperti di notebook
    pressure         = acad_pressure + 0          # Work Pressure = 0 untuk Student
    role_satisfaction = study_sat + 0             # Job Satisfaction = 0 untuk Student
    cgpa_val         = np.nan if cgpa == 0.0 else cgpa

    input_data = pd.DataFrame([{
        "Age"                                    : age,
        "Gender"                                 : gender,
        "City"                                   : city,
        "Profession"                             : profession,
        "Academic Pressure"                      : float(acad_pressure),
        "CGPA"                                   : cgpa_val,
        "Study Satisfaction"                     : float(study_sat),
        "Sleep Duration"                         : sleep_dur,
        "Dietary Habits"                         : dietary,
        "Degree"                                 : degree,
        "Have you ever had suicidal thoughts ?"  : suicidal,
        "Work/Study Hours"                       : float(work_study_hrs),
        "Financial Stress"                       : float(financial_str),
        "Family History of Mental Illness"       : family_hist,
        "Pressure"                               : float(pressure),
        "Role Satisfaction"                      : float(role_satisfaction),
    }])

    # ── Prediksi ─────────────────────────────────────────
    try:
        y_prob = pipeline.predict_proba(input_data)[0, 1]
        y_pred = pipeline.predict(input_data)[0]
        conf   = confidence_fn(y_prob)

        # ── Tampilkan hasil ──────────────────────────────
        st.markdown("## 📊 Hasil Prediksi")

        # Warna berdasarkan hasil
        if y_pred == 1:
            result_color = "#e74c3c" if y_prob >= 0.55 else "#e67e22"
            result_label = "⚠️ Terindikasi Depresi"
        else:
            result_color = "#27ae60" if y_prob < 0.45 else "#f39c12"
            result_label = "✅ Tidak Terindikasi Depresi"

        st.markdown(
            f"""
            <div style="
                background:{result_color}22;
                border-left: 6px solid {result_color};
                padding: 1.2rem 1.5rem;
                border-radius: 8px;
                margin-bottom: 1rem;
            ">
                <h3 style="color:{result_color}; margin:0">{result_label}</h3>
                <p style="margin:0.4rem 0 0; font-size:0.95rem; color:#444">
                    Probabilitas Depresi: <b>{y_prob:.1%}</b> &nbsp;|&nbsp;
                    Confidence: <b>{conf}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar probabilitas
        st.markdown("**Probabilitas Depresi**")
        st.progress(float(y_prob))
        st.caption(f"{y_prob:.1%} — semakin ke kanan semakin tinggi risiko")

        # Penjelasan confidence
        st.markdown("---")
        st.markdown("#### 📌 Panduan Confidence Level")
        st.markdown("""
        | Level | Rentang Probabilitas | Arti |
        |-------|---------------------|------|
        | Tinggi — Depresi | ≥ 75% | Model sangat yakin positif |
        | Sedang — Depresi | 55–75% | Model cukup yakin positif |
        | Borderline | 45–55% | Model ragu, perlu review lebih lanjut |
        | Sedang — Tidak Depresi | 25–45% | Model cukup yakin negatif |
        | Tinggi — Tidak Depresi | < 25% | Model sangat yakin negatif |
        """)

        st.warning(
            "⚠️ **Disclaimer:** Prediksi ini bukan diagnosis medis. "
            "Selalu konsultasikan dengan profesional kesehatan mental."
        )

    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")
        st.exception(e)
