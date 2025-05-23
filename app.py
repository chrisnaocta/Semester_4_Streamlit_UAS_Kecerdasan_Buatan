# Import libary
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Library untuk visualisasi data
import io

from load_data import load_data

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.page_link("App.py", label="Home", icon="🏠")
st.page_link("pages/1_Preprocessing.py", label="Page Preprocessing", icon="1️⃣")
st.page_link("pages/2_Train_Test_Split.py", label="Page Split Data", icon="2️⃣")
st.page_link("pages/3_TF_IDF_WordCloud.py", label="Page TF-IDF dan Word Cloud", icon="3️⃣")
st.page_link("pages/4_Training_Model.py", label="Page Training Model dan Evaluasi", icon="4️⃣")

st.title("HOME PAGE")
st.markdown("Silakan pilih halaman di sidebar untuk eksplorasi lebih lanjut.")

if st.button("📊 Menampilkan Exploratory Data Analysis (EDA)"):
    # Title
    st.title("Analisis Sentimen Film 🎬")

    df = load_data()
    st.success("Dataset berhasil dimuat dari folder `dataset/`.")

    # EDA Section
    st.markdown("## 📊 Exploratory Data Analysis (EDA)")

    # 1. Preview Data
    st.subheader("1. Preview Data (5 Baris Pertama)")
    st.dataframe(df.head())

    # 2. Info Dataset
    st.subheader("2. Informasi Struktur Dataset")
    # Tangkap hasil df.info() ke buffer
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    # Tampilkan hasil di Streamlit dengan format code block
    st.code(info_str, language='text')

    # 3. Distribusi Label
    st.subheader("3. Distribusi Label Sentimen")
    st.write(df['label'].value_counts())

    # 4. Cek Data Null
    st.subheader("4. Jumlah Nilai Kosong (Null) per Kolom")
    st.write(df.isnull().sum())

    # 5. Pie Chart Distribusi Label
    st.subheader("5. Visualisasi Distribusi Label Sentimen (Pie Chart)")
    label_counts = df['label'].value_counts()

    def func(pct, allvalues):
        absolute = int(np.round(pct / 100. * np.sum(allvalues)))
        return f'{absolute} ({pct:.2f}%)'

    labels = [f'Label {i}: {label_counts[i]}' for i in label_counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        label_counts,
        labels=labels,
        autopct=lambda pct: func(pct, label_counts),
        startangle=140,
        colors=['#ff9999', '#66b3ff']
    )
    ax.set_title("Distribusi Label Sentimen")
    st.pyplot(fig)