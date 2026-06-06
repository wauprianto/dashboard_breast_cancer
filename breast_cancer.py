import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Regresi Logistik (Mean)", page_icon="🧬", layout="wide")

# ==========================================
# SIDEBAR: CAPTION INFORMASI PENGEMBANG
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/6840/6840410.png", width=80)
st.sidebar.markdown("### 🧑‍💻 Informasi Pengembang")
st.sidebar.markdown("**Dibuat oleh:** \nPrianto Sanema Wau")
st.sidebar.markdown("**NIM:** \n053286867")
st.sidebar.markdown("**Program Studi:** \nStatistika")
st.sidebar.markdown("---")
# Fitur Baru: File Uploader di Sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV Baru (Opsional)", type=['csv'])
st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini disusun untuk memenuhi tugas analisis inferensi statistik menggunakan pemodelan Regresi Logistik Biner.")

# Konten Utama
st.title("📊 Dashboard Analisis Regresi Logistik (10 Variabel Mean)")
st.markdown("Aplikasi interaktif untuk menganalisis dan memprediksi sifat tumor payudara menggunakan seluruh fitur dengan metrik rata-rata (mean).")

@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv('data.csv')
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    return df

try:
    df = load_data(uploaded_file)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    fitur = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]
    
    # Menambahkan tab baru untuk fitur tambahan
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Eksplorasi Data", "📉 Ringkasan Model & Evaluasi", "🔮 Kalkulator Prediksi", "📈 Feature Importance", "🧠 Technical Insights"])
    
    # ==========================================
    # TAB 1: PENJELASAN DATASET & EKSPLORASI DATA
    # ==========================================
    with tab1:
        st.header("📋 Deskripsi Dataset")
        with st.expander("Klik untuk membaca detail 10 Variabel 'Mean'", expanded=False):
            st.markdown("*(Penjelasan variabel Anda tetap ada di sini...)*")
        
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Sampel Data Teratas")
            st.dataframe(df[['diagnosis'] + fitur].head(10), use_container_width=True)
        with col2:
            st.subheader("Distribusi Kelas")
            st.bar_chart(df['diagnosis'].value_counts(), color=["#FF4B4B"]) 

    # ==========================================
    # TAB 2: RINGKASAN MODEL STATISTIK & EVALUASI
    # ==========================================
    with tab2:
        st.header("Estimasi Parameter & Evaluasi Model")
        
        # Fit Model Statsmodels (asli Anda)
        X = sm.add_constant(df[fitur])
        y = df['target']
        model_sm = sm.Logit(y, X).fit(method='newton', disp=False)
        
        y_pred_prob = model_sm.predict(X)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Penambahan fitur: Perbandingan Model dengan Scikit-Learn
        st.subheader("Perbandingan Performa Model")
        model_choice = st.radio("Pilih Model untuk Metrik:", ["Logistic Regression (Statsmodels)", "Random Forest"], horizontal=True)
        
        if model_choice == "Random Forest":
            clf = RandomForestClassifier().fit(df[fitur], y)
            y_pred_new = clf.predict(df[fitur])
            y_prob_new = clf.predict_proba(df[fitur])[:, 1]
            akurasi = accuracy_score(y, y_pred_new)
            y_prob_final = y_prob_new
        else:
            akurasi = accuracy_score(y, y_pred)
            y_prob_final = y_pred_prob

        # Metrik
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(label="Akurasi Model", value=f"{akurasi * 100:.2f}%")
        
        # ROC Curve (Fitur Baru)
        fpr, tpr, _ = roc_curve(y, y_prob_final)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC = {auc(fpr, tpr):.2f}'))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.subheader("Tabel Summary Regresi Logistik")
        st.text_area("Statsmodels Results", str(model_sm.summary()), height=300)

    # ==========================================
    # TAB 3: KALKULATOR PREDIKSI (ASLI ANDA)
    # ==========================================
    with tab3:
        # (Kode kalkulator slider Anda tetap di sini...)
        st.header("Simulasi Prediksi Real-Time")
        st.markdown("Sesuaikan parameter di bawah ini untuk melihat bagaimana probabilitas dihitung menggunakan fungsi sigmoid:")
        st.latex(r"P(Y=1) = \frac{1}{1 + e^{-Z}} \quad \text{dimana} \quad Z = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n")
        st.write("---")
        
        # SLIDER INPUT DIBUAT 2 KOLOM AGAR RAPI
        slider_vals = {}
        col_input1, col_input2 = st.columns(2)
        
        for i, f in enumerate(fitur):
            nama_label = f.replace('_', ' ').title()
            if i < 5:
                with col_input1:
                    slider_vals[f] = st.slider(nama_label, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
            else:
                with col_input2:
                    slider_vals[f] = st.slider(nama_label, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
            
        # HITUNG LOG-ODDS (Z) OTOMATIS
        params = model_sm.params
        z_score = params['const'] + sum(params[f] * slider_vals[f] for f in fitur)
        probabilitas = 1 / (1 + np.exp(-z_score))
        
        st.markdown("---")
        
        col_hasil, col_grafik = st.columns([1, 2])
        
        with col_hasil:
            st.subheader("Hasil Diagnostik")
            st.metric(label="Nilai Log-Odds (Z)", value=f"{z_score:.4f}")
            st.metric(label="Probabilitas Ganas (P)", value=f"{probabilitas * 100:.2f}%")
            
            st.write("---")
            if probabilitas >= 0.5:
                st.error("🚨 **PREDIKSI: MALIGNANT (GANAS)**")
                st.warning("Probabilitas di atas 50%. Model mengklasifikasikan sebagai tumor ganas.")
            else:
                st.success("✅ **PREDIKSI: BENIGN (JINAK)**")
                st.info("Probabilitas di bawah 50%. Model mengklasifikasikan sebagai tumor jinak.")
                
        with col_grafik:
            # GRAFIK KURVA SIGMOID INTERAKTIF
            z_range = np.linspace(-15, 15, 200)
            p_range = 1 / (1 + np.exp(-z_range))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_range, y=p_range, mode='lines', name='Kurva Sigmoid', line=dict(color='#4B4BFF', width=3)))
            
            titik_warna = '#FF4B4B' if probabilitas >= 0.5 else '#00CC96'
            fig.add_trace(go.Scatter(x=[z_score], y=[probabilitas], mode='markers', name='Titik Pasien',
                                     marker=dict(color=titik_warna, size=16, line=dict(color='white', width=2))))
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Batas Keputusan (0.5)")
            
            fig.update_layout(
                title="Posisi Pasien pada Kurva Probabilitas",
                xaxis_title="Log-Odds (Z)",
                yaxis_title="Probabilitas (P)",
                yaxis=dict(range=[-0.05, 1.05]),
                xaxis=dict(range=[-15, 15]),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("Gagal memuat data. Pastikan file 'data.csv' sudah diunggah di folder kerja yang sama dengan app.py.")

    # ==========================================
    # TAB 4: FEATURE IMPORTANCE (BARU)
    # ==========================================
    with tab4:
        st.header("Analisis Feature Importance")
        model_rf = RandomForestClassifier().fit(df[fitur], y)
        imp = pd.DataFrame({'Feature': fitur, 'Importance': model_rf.feature_importances_}).sort_values(by='Importance')
        fig_imp, ax = plt.subplots()
        ax.barh(imp['Feature'], imp['Importance'], color='skyblue')
        st.pyplot(fig_imp)

    # ==========================================
    # TAB 5: TECHNICAL INSIGHTS (BARU)
    # ==========================================
    with tab5:
        st.header("🧠 Technical Insights")
        st.markdown("""
        * **Preprocessing:** Dilakukan drop pada kolom tidak relevan.
        * **Evaluasi:** Menggunakan AUC-ROC untuk menilai kemampuan diskriminasi model pada data medis.
        * **Insight:** Random Forest membantu mengidentifikasi fitur paling kritis tanpa asumsi linearitas.
        """)

except Exception as e:
    st.error(f"Error: {e}")
