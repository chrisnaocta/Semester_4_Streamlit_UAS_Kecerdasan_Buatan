import streamlit as st
import pandas as pd

from split_data import split_data
from load_data import load_data

st.set_page_config(page_title="Split Data", layout="wide")

st.page_link("App.py", label="Home", icon="🏠")
st.page_link("pages/1_Preprocessing.py", label="Page Preprocessing", icon="1️⃣")
st.page_link("pages/2_Train_Test_Split.py", label="Page Split Data", icon="2️⃣")
st.page_link("pages/3_TF_IDF_WordCloud.py", label="Page TF-IDF dan Word Cloud", icon="3️⃣")
st.page_link("pages/4_Training_Model.py", label="Page Training Model dan Evaluasi", icon="4️⃣")

# Load dataset
df = load_data()

st.title("Halaman Split Data")

# Tombol untuk mulai proses
if st.button("Split Data dan Preprocess Data"):
    X_train, X_test, y_train, y_test = split_data(df)

    st.success("Data berhasil dibagi dan dipreproses!")
    st.write(f"Jumlah data latih: {len(X_train)}")
    st.write(f"Jumlah data uji: {len(X_test)}")

    # Tampilkan beberapa contoh hasil preprocessing
    st.subheader("Contoh Hasil Preprocessing (Train)")
    st.write(X_train.head())