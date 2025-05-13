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

# Preprocessing text
# NLTK = Natural Language Toolkit
nltk.download('punkt') #Modul tokenizer untuk memecah teks menjadi kalimat atau kata
nltk.download('stopwords') #Kumpulan kata-kata umum (seperti "the", "is", "and")
nltk.download('wordnet') #Korpus WordNet digunakan untuk lemmatization, yaitu mengubah kata ke bentuk dasar dengan mempertimbangkan konteksnya.
nltk.download('omw-1.4') #sumber daya tambahan untuk wordnet, mendukung banyak bahasa dan memperkaya konteks.
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english')) - {
    "not", "no", "nor", "won't", "can't", "don't", "didn't", "isn't", "wasn't",  "couldn't", "shouldn't", "wouldn't"}
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_for_sentiment(text):
    # 1. Jadikan semua uruf kecil
    text = text.lower()

    # 2. Hilangkan tag HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # 3. Convert emoji ke teks
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")

    # 4. Hapus tanda baca (kecuali tanda negasi, karena pada stopwords tanda negasi diperlukan)
    text = re.sub(r"[^\w\s']", '', text)

    # 5. Tokenisasi (memecah teks menjadi bagian-bagian yang lebih kecil)
    words = nltk.word_tokenize(text)

    # 6. Hapus kata-kata stopword tetapi pertahankan negasi yang sudah ditentukan di atas
    words = [word for word in words if word not in stop_words]

    # 7. Lemmatization (mengubah kata ke bentuk dasar yang lebih bermakna dengan mempertimbangkan konteks dan makna kata tersebut)
    words = [lemmatizer.lemmatize(word) for word in words]

    # 8. Gabung kembali teks
    return " ".join(words)