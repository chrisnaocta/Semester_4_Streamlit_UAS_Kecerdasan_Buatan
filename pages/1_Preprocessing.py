# Import libary
import streamlit as st
import pandas as pd
from preprocessing import preprocess_for_sentiment

st.set_page_config(page_title="Preprocessing Function", layout="wide")

st.page_link("App.py", label="Home", icon="🏠")
st.page_link("pages/1_Preprocessing.py", label="Page Preprocessing", icon="1️⃣")
st.page_link("pages/2_Train_Test_Split.py", label="Page Split Data", icon="2️⃣")
st.page_link("pages/3_TF_IDF_WordCloud.py", label="Page TF-IDF dan Word Cloud", icon="3️⃣")

st.title("Halaman Preprocessing")

# Load dataset dari folder 'dataset'
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/movie_dataset.csv")
    return df
df = load_data()

# Preview Data
st.subheader("Data 5 Baris Pertama")
st.dataframe(df.head())

st.write("Masukkan salah satu teks dari 5 baris di atas untuk dipreproses:")

text = st.text_area("Masukkan kalimat:")
if st.button("Preprocess"):
    if text:
        hasil = preprocess_for_sentiment(text)
        st.write("Hasil Preprocessing:")
        st.write(hasil)
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")