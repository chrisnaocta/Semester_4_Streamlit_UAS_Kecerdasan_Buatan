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