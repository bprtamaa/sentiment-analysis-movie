# Analisis Sentimen Berbasis Aspek Ulasan Film Ghost in the Cell (2026) pada Website Letterboxd Menggunakan Support Vector Machine (SVM)

Repository ini berisi seluruh source code, pipeline eksperimen, dan artefak model penelitian skripsi dengan judul:
> **"Analisis Sentimen Berbasis Aspek Ulasan Film Ghost in the Cell (2026) pada Website Letterboxd Menggunakan Support Vector Machine (SVM)"**

---

## 📌 Deskripsi Penelitian

Penelitian ini bertujuan untuk menganalisis sentimen penonton terhadap film Indonesia **"Ghost in the Cell" (2026)** berdasarkan ulasan pengguna pada platform Letterboxd. Analisis dilakukan secara berbasis aspek (*Aspect-Based Sentiment Analysis* / ABSA) untuk memahami persepsi penonton terhadap elemen-elemen spesifik film, meliputi:
1. **Alur/Cerita** (Plot, ending, twist, pacing, naskah)
2. **Akting/Pemain** (Karakter, penampilan aktor/aktris, chemistry)
3. **Visual/Artistik** (Sinematografi, visual effects/gore/macabre, artistik, lighting)
4. **Komedi/Humor** (Lelucon, punchline, komedi, hiburan)
5. **Pesan/Satire** (Kritik sosial, satire politik, pesan moral, isu hukum/korupsi)

---

## 🔄 Alur Kerja & Pipeline Penelitian

Pipeline penelitian dirancang secara terstruktur dan modular ke dalam beberapa tahapan:

```
[Letterboxd]
     │
     ▼ (SeleniumBase UC Mode)
[1. Scraping Ulasan] ─────────► dataset_ghost_cell_3000.csv
     │
     ▼ (Language Detect & Cosine Similarity)
[2. Filtering Dataset] ────────► dataset_filtered.csv
     │
     ▼ (Cleansing, Slang Normalization, Stemming Sastrawi)
[3. Preprocessing] ────────────► dataset_preprocessed.csv
     │
     ▼ (Latent Dirichlet Allocation / LDA - 5 Aspek)
[4. Topic Modelling] ──────────► dataset_with_topics.csv & lda_visualization.html
     │
     ▼ (Hybrid: Rating + Leksikon Berbobot)
[5. Pelabelan Sentimen] ───────► dataset_labeled_sentiment.csv
     │
     ▼ (TF-IDF Vectorizer)
[6. Ekstraksi Fitur] ──────────► tfidf_matrix.pkl & tfidf_vectorizer.pkl
     │
     ▼ (SVM: Linear, RBF, Poly, Sigmoid + Hyperparameter Tuning)
[7. Klasifikasi & Evaluasi] ───► model_svm_terbaik.pkl & Visualisasi Evaluasi
```

---

## 📁 Struktur Direktori & Kode Sumber

| No | File | Deskripsi |
|:---:|:---|:---|
| 1 | `scraping_ghost_cell_3000.py` | Modul web scraper untuk mengumpulkan ulasan penonton di Letterboxd menggunakan SeleniumBase UC Mode (bypass Cloudflare Turnstile). |
| 2 | `filter_dataset_colab.py` | Modul filter dataset berdasarkan bahasa (bahasa Indonesia) dan deduplikasi teks (exact match & cosine similarity). |
| 3 | `preprocessing_ghost_cell_colab.py` | Modul pra-pemrosesan teks: Case folding, data cleansing, normalisasi slang/singkatan, tokenisasi, stopword removal, dan stemming Sastrawi (dengan preservasi sentimen & negasi). |
| 4 | `topic_modelling_lda_colab.py` | Pemodelan topik Latent Dirichlet Allocation (LDA) untuk mengelompokkan ulasan ke dalam 5 aspek utama dan visualisasi interaktif via pyLDAvis. |
| 5 | `sentiment_labeling_colab.py` | Pelabelan sentimen otomatis berbasis pendekatan hybrid (integrasi rating bintang Letterboxd dan kamus leksikon sentimen berbobot gradasi). |
| 6 | `extract_features_tfidf.py` | Ekstraksi fitur teks menggunakan *Term Frequency-Inverse Document Frequency* (TF-IDF). |
| 7 | `klasifikasi_svm_colab.py` | Pelatihan model Support Vector Machine (SVM) dengan evaluasi multi-kernel (Linear, RBF, Polynomial, Sigmoid), Grid Search CV, dan visualisasi performa. |
| 8 | `dataset_ghost_cell_3000.csv` | Dataset mentah hasil scraping ulasan film Ghost in the Cell dari Letterboxd. |
| 9 | `lda_visualization.html` | Visualisasi interaktif pemodelan topik pyLDAvis untuk eksplorasi aspek. |
| 10 | `tfidf_vectorizer.pkl` | Model TF-IDF Vectorizer yang telah di-fit pada korpus teks. |
| 11 | `tfidf_matrix.pkl` | Matriks representasi numerik fitur TF-IDF hasil transformasi. |
| 12 | `model_svm_terbaik.pkl` | Model Support Vector Machine (SVM) terlatih dengan performa terbaik. |

---

## 🛠️ Instalasi & Prasyarat

### Prasyarat Sistem
- Python 3.9+
- Google Colab atau Environment Lokal (Windows / Linux / macOS)

### Instalasi Dependensi
```bash
pip install pandas numpy scikit-learn nltk PySastrawi gensim pyLDAvis seleniumbase beautifulsoup4 lxml matplotlib seaborn tqdm langdetect
```

Jika menggunakan NLTK untuk pertama kali:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 🚀 Cara Menjalankan

Pipeline dapat dijalankan secara berurutan sesuai urutan eksperimen:

1. **Scraping Data:**
   ```bash
   python scraping_ghost_cell_3000.py
   ```
2. **Filtering Data:**
   ```bash
   python filter_dataset_colab.py
   ```
3. **Preprocessing Teks:**
   ```bash
   python preprocessing_ghost_cell_colab.py
   ```
4. **Topic Modelling (LDA):**
   ```bash
   python topic_modelling_lda_colab.py
   ```
5. **Pelabelan Sentimen (Hybrid):**
   ```bash
   python sentiment_labeling_colab.py
   ```
6. **Ekstraksi Fitur (TF-IDF):**
   ```bash
   python extract_features_tfidf.py
   ```
7. **Klasifikasi SVM & Evaluasi:**
   ```bash
   python klasifikasi_svm_colab.py
   ```

---

## 📊 Teknologi & Pustaka yang Digunakan

- **Bahasa Pemrograman:** Python
- **Web Scraping:** SeleniumBase UC Mode, BeautifulSoup4, LXML
- **Text Preprocessing & NLP:** NLTK, Sastrawi, Regex, Langdetect
- **Topic Modelling:** Gensim (LDA Model), pyLDAvis
- **Machine Learning & Feature Extraction:** Scikit-learn (SVM / SVC, TfidfVectorizer, Train-Test Split, GridSearchCV, Classification Metrics)
- **Visualisasi Data:** Matplotlib, Seaborn, pyLDAvis

---

## 👤 Penulis

- **Peneliti:** [bprtamaa](https://github.com/bprtamaa)
- **Topik:** Analisis Sentimen Berbasis Aspek (ABSA) / Natural Language Processing (NLP)
