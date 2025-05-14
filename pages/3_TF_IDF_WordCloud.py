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
from split_data import split_data

st.set_page_config(page_title="TF-IDF & Word Cloud", layout="wide")

st.page_link("App.py", label="Home", icon="🏠")
st.page_link("pages/1_Preprocessing.py", label="Page Preprocessing", icon="1️⃣")
st.page_link("pages/2_Train_Test_Split.py", label="Page Split Data", icon="2️⃣")
st.page_link("pages/3_TF_IDF_WordCloud.py", label="Page TF-IDF dan Word Cloud", icon="3️⃣")

st.title("📊 TF-IDF dan WordCloud")
st.write("Visualisasi komentar **positif** dan **netral** berdasarkan nilai TF-IDF dan menampilkan Word Cloud.")

if st.button("🔍 Proses TF-IDF dan Tampilkan Diagram Batang & Word Cloud"):
    # Load dan split data
    df = pd.read_csv("dataset/movie_dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Filter komentar positif & netral
    X_train_positive = X_train[y_train == 1]
    X_train_netral = X_train[y_train == 0]

    # TF-IDF POSITIF
    tfidf_pos = TfidfVectorizer()
    X_pos_tfidf = tfidf_pos.fit_transform(X_train_positive)
    words_pos = tfidf_pos.get_feature_names_out()
    mean_pos = np.asarray(X_pos_tfidf.mean(axis=0)).ravel()
    top_pos_idx = mean_pos.argsort()[::-1][:20]
    top_words_pos = words_pos[top_pos_idx]
    top_scores_pos = mean_pos[top_pos_idx]
    tfidf_dict_pos = dict(zip(words_pos, mean_pos))

    # TF-IDF NETRAL
    tfidf_net = TfidfVectorizer()
    X_net_tfidf = tfidf_net.fit_transform(X_train_netral)
    words_net = tfidf_net.get_feature_names_out()
    mean_net = np.asarray(X_net_tfidf.mean(axis=0)).ravel()
    top_net_idx = mean_net.argsort()[::-1][:20]
    top_words_net = words_net[top_net_idx]
    top_scores_net = mean_net[top_net_idx]
    tfidf_dict_net = dict(zip(words_net, mean_net))

    # Visualisasi POSITIF
    st.subheader("💚 Komentar Positif")
    fig_pos, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].barh(top_words_pos[::-1], top_scores_pos[::-1], color='green')
    ax[0].set_title("Top 20 TF-IDF Kata - Positif")
    ax[0].set_xlabel("Skor TF-IDF")
    wordcloud_pos = WordCloud(width=600, height=400, background_color='white',
                              colormap='Greens').generate_from_frequencies(tfidf_dict_pos)
    ax[1].imshow(wordcloud_pos, interpolation='bilinear')
    ax[1].axis('off')
    ax[1].set_title("WordCloud - Komentar Positif")
    st.pyplot(fig_pos)

    # Visualisasi NETRAL
    st.subheader("💙 Komentar Netral")
    fig_net, ax2 = plt.subplots(1, 2, figsize=(14, 5))
    ax2[0].barh(top_words_net[::-1], top_scores_net[::-1], color='blue')
    ax2[0].set_title("Top 20 TF-IDF Kata - Netral")
    ax2[0].set_xlabel("Skor TF-IDF")
    wordcloud_net = WordCloud(width=600, height=400, background_color='white',
                              colormap='Blues').generate_from_frequencies(tfidf_dict_net)
    ax2[1].imshow(wordcloud_net, interpolation='bilinear')
    ax2[1].axis('off')
    ax2[1].set_title("WordCloud - Komentar Netral")
    st.pyplot(fig_net)