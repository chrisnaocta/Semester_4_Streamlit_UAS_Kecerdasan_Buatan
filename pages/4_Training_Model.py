import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from split_data import split_data
from load_data import load_data
import seaborn as sns
import matplotlib.pyplot as plt

# from split_data import split_data
# from load_data import load_data
from model import training_model

st.set_page_config(page_title="Training Model & Evaluasi Model", layout="wide")

st.page_link("App.py", label="Home", icon="🏠")
st.page_link("pages/1_Preprocessing.py", label="Page Preprocessing", icon="1️⃣")
st.page_link("pages/2_Train_Test_Split.py", label="Page Split Data", icon="2️⃣")
st.page_link("pages/3_TF_IDF_WordCloud.py", label="Page TF-IDF dan Word Cloud", icon="3️⃣")
st.page_link("pages/4_Training_Model.py", label="Page Training Model dan Evaluasi", icon="4️⃣")

st.title("🧠 Training & Evaluasi Model - Random Forest")

st.write("Model akan dilatih menggunakan Algoritma **Random Forest Classifier**.")
if st.button("🚀 Mulai Training Model"):
    # # Load dan split data
    # df = load_data()
    # X_train, X_test, y_train, y_test = split_data(df)

    # # TF-IDF (hanya fit ke X_train agar adil)
    # tfidf = TfidfVectorizer()
    # X_train_tfidf = tfidf.fit_transform(X_train)
    # X_test_tfidf = tfidf.transform(X_test)

    # # Inisialisasi model
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # # Training
    # rf.fit(X_train_tfidf, y_train)
    # # Prediksi
    # y_pred = rf.predict(X_test_tfidf)

    training_model()

    st.success(f"✅ Model selesai dilatih")

# Tombol evaluasi
if st.button("📊 Evaluasi Model"):
    if st.session_state.model is None:
        st.warning("⚠️ Model belum dilatih. Silakan tekan 'Mulai Training Model' dulu.")
    else:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader("✅ Akurasi")
        st.write(f"{acc:.2%}")

        st.subheader("📋 Classification Report")
        st.text(report)

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)