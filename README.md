📊 Sentiment Analysis Shopee Review - Streamlit App
Proyek ini merupakan aplikasi analisis sentimen berbasis web yang dibuat menggunakan Python dan Streamlit. Tujuan utama dari proyek ini adalah untuk mengolah dan menganalisis komentar-komentar berbahasa Inggris dari Twitter tentang film

🎯 Tujuan Proyek
- Menyediakan alat visualisasi dan pemodelan sederhana
- Menampilkan statistik dasar dari data review
- Melakukan preprocessing teks (cleansing, case folding, stopword removal, stemming)
- Menghitung dan menampilkan TF-IDF dari komentar positif dan netral
- Menampilkan Wordcloud dari kata-kata penting
- Melatih model Random Forest Classifier
- Mengevaluasi performa model dengan akurasi, classification report, dan confusion matrix

🛠️ Tools & Libraries
- Python – Bahasa pemrograman utama
- Streamlit – Untuk membangun aplikasi web interaktif
- Pandas – Untuk manipulasi dan analisis data
- NumPy – Operasi numerik dan array
- Matplotlib & Seaborn – Visualisasi data dan grafik statistik
- Scikit-learn (sklearn) – Untuk preprocessing, vektorisasi TF-IDF, pembagian data, pelatihan dan evaluasi model machine learning
- NLTK (Natural Language Toolkit) – Untuk preprocessing teks: tokenisasi, stopwords, stemming, dan lemmatization
- BeautifulSoup – Untuk membersihkan teks dari tag HTML
- emoji – Untuk mendeteksi dan menangani karakter emoji dalam teks
- WordCloud – Untuk membuat visualisasi awan kata berdasarkan frekuensi kata

📂 Struktur Halaman Streamlit
1_Load_Data.py – Memuat dan menampilkan dataset NLP Twitter (komentar tentang film-film)
2_Preprocessing.py – Proses pembersihan teks
3_TF_IDF_WordCloud.py – Analisis kata dengan TF-IDF dan Wordcloud
4_Training_Model.py – Training dan evaluasi model klasifikasi

🧪 Dataset
Dataset yang digunakan berisi kumpulan tweet berbahasa Inggris yang mengomentari film. Setiap tweet telah dilabeli dengan sentimen:
1 = Positif
0 = Netral
