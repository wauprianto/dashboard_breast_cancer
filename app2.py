import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Regresi Logistik Pro", page_icon="🧬", layout="wide")

# --- CSS Custom untuk Estetika ---
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR & FILE UPLOADER ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/6840/6840410.png", width=80)
st.sidebar.markdown("### 🧑‍💻 Informasi Pengembang")
st.sidebar.markdown("**Prianto Sanema Wau**\n\n**NIM:** 053286867\n**Prodi:** Statistika")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload CSV Baru (Opsional)", type=['csv'])

# Fungsi Loading Data
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
    fitur = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
             'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
    
    X = df[fitur]
    y = df['target']

    # --- TABS UTAMA ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Eksplorasi Data", "🎯 Model Comparison", "📈 Feature Importance", "🔮 Prediksi Interaktif", "🧠 Technical Insights"])

    with tab1:
        st.header("Eksplorasi Data")
        st.dataframe(df.head())
        st.bar_chart(df['diagnosis'].value_counts(), color=["#FF4B4B"])

    with tab2:
        st.header("Model Performance & Metrics")
        model_choice = st.radio("Pilih Model:", ["Logistic Regression", "Random Forest"], horizontal=True)
        
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000).fit(X, y)
        else:
            model = RandomForestClassifier().fit(X, y)
            
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Metrik
        col1, col2 = st.columns(2)
        col1.metric("Akurasi Model", f"{accuracy_score(y, y_pred)*100:.2f}%")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC = {roc_auc:.2f}'))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        st.header("Analisis Feature Importance")
        if model_choice == "Random Forest":
            imp = pd.DataFrame({'Feature': fitur, 'Importance': model.feature_importances_})
        else:
            imp = pd.DataFrame({'Feature': fitur, 'Importance': np.abs(model.coef_[0])})
        
        imp = imp.sort_values(by='Importance', ascending=True)
        fig_imp, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp['Feature'], imp['Importance'], color='skyblue')
        st.pyplot(fig_imp)

    with tab4:
        st.header("Simulasi Prediksi Real-Time")
        st.latex(r"P(Y=1) = \frac{1}{1 + e^{-Z}}")
        cols = st.columns(2)
        inputs = {}
        for i, f in enumerate(fitur):
            inputs[f] = cols[i%2].slider(f.replace('_',' ').title(), float(df[f].min()), float(df[f].max()), float(df[f].mean()))
        
        # Prediksi sederhana
        if st.button("Jalankan Prediksi"):
            input_df = pd.DataFrame([inputs])
            prob = model.predict_proba(input_df)[0][1]
            st.metric("Probabilitas Ganas", f"{prob*100:.2f}%")

    with tab5:
        st.header("Technical Insights")
        st.markdown("""
        * **Preprocessing:** Dilakukan pemetaan label M/B ke 1/0.
        * **Perbandingan:** Random Forest umumnya menangani non-linearitas lebih baik daripada Logistik.
        * **Metrik:** AUC-ROC digunakan karena lebih robust terhadap ketidakseimbangan kelas dibanding akurasi murni.
        * **Catatan:** Multikolinearitas antar variabel `mean` (radius, perimeter, area) dapat mempengaruhi kestabilan koefisien model logistik.
        """)

except Exception as e:
    st.error(f"Error: {e}")
