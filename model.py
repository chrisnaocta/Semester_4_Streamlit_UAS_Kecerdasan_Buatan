import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from split_data import split_data
from load_data import load_data
import seaborn as sns
import matplotlib.pyplot as plt

from split_data import split_data
from load_data import load_data

# Penampung global model dan data
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tfidf' not in st.session_state:
    st.session_state.tfidf = None
if 'X_train_tfidf' not in st.session_state:
    st.session_state.X_train_tfidf = None
if 'X_test_tfidf' not in st.session_state:
    st.session_state.X_test_tfidf = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

def training_model():
    # Load dan split data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # TF-IDF (hanya fit ke X_train agar adil)
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 8. Training Model Random Forest
    # Inisialisasi model Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Latih model dengan data latih TF-IDF
    rf.fit(X_train_tfidf, y_train)
    # Prediksi data uji
    y_pred = rf.predict(X_test_tfidf)

    # Simpan ke session_state
    st.session_state.model = rf
    st.session_state.tfidf = tfidf
    st.session_state.X_train_tfidf = X_train_tfidf
    st.session_state.X_test_tfidf = X_test_tfidf
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred