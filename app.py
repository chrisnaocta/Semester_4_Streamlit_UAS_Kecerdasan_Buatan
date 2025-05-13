# Import libary
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Library untuk visualisasi data
import seaborn as sns
from sklearn.model_selection import train_test_split #Library untuk membagi data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bs4 import BeautifulSoup
import emoji
from wordcloud import WordCloud

# Title
st.title("Analisis Sentimen Film 🎬")

# Load dataset dari folder 'dataset'
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/movie_dataset.csv")
    return df

df = load_data()
st.success("Dataset berhasil dimuat dari folder `dataset/`.")

# EDA Section
st.markdown("## 📊 Exploratory Data Analysis (EDA)")

# 1. Preview Data
st.subheader("1. Preview Data (5 Baris Pertama)")
st.dataframe(df.head())

# 2. Info Dataset
st.subheader("2. Informasi Struktur Dataset")
buffer = open('info.txt', 'w')
df.info(buf=buffer)
buffer.close()
with open("info.txt") as f:
    st.text(f.read())

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