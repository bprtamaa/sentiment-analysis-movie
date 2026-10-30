"""
===============================================================================
EKSTRAKSI FITUR TF-IDF ULASAN FILM "GHOST IN THE CELL (2026)"
Untuk Analisis Sentimen — Skripsi

Platform   : Google Colab
Input      : dataset_labeled_sentiment.csv (hasil pelabelan sentimen)
Output     : tfidf_matrix.pkl (matriks fitur TF-IDF)
             tfidf_vectorizer.pkl (model TF-IDF Vectorizer)
             top10_tfidf_features.png (visualisasi fitur terbobot)

Asosiasi   : Mengambil kolom 'ulasan_preprocessed' saja untuk diekstraksi.
===============================================================================
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: INSTALL DEPENDENCIES                                             ║
# ║  Jalankan cell ini di Google Colab jika belum terinstall                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Uncomment baris di bawah jika menjalankan di Colab:
# !pip install pandas numpy matplotlib seaborn scikit-learn

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: IMPORT LIBRARY                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import warnings
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("[OK] Library berhasil diimpor!")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: KONFIGURASI PARAMETER                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
INPUT_FILE = "dataset_labeled_sentiment.csv"   # Nama file input (hasil pelabelan sentimen)
TEXT_COLUMN = "ulasan_preprocessed"            # Kolom target yang akan diekstraksi
OUTPUT_MATRIX = "tfidf_matrix.pkl"             # Hasil ekspor matriks fitur
OUTPUT_VECTORIZER = "tfidf_vectorizer.pkl"     # Hasil ekspor model vectorizer
OUTPUT_PLOT = "top10_tfidf_features.png"       # Hasil ekspor plot visualisasi

# Parameter TF-IDF untuk AKURASI TINGGI
TFIDF_MAX_FEATURES = 5000                     # Maksimal jumlah fitur kata teratas
TFIDF_NGRAM_RANGE = (1, 2)                     # Menggunakan Unigram & Bigram (meningkatkan akurasi konteks)
TFIDF_SUBLINEAR_TF = True                      # Meredam pengaruh frekuensi tinggi (1 + log(tf))
TFIDF_MIN_DF = 2                               # Mengabaikan kata yang hanya muncul di 1 ulasan
TFIDF_MAX_DF = 0.90                            # Mengabaikan kata yang muncul di >90% ulasan (kebisingan kata umum)

print("[OK] Konfigurasi parameter TF-IDF dimuat:")
print(f"     Max Features : {TFIDF_MAX_FEATURES}")
print(f"     N-Gram Range : {TFIDF_NGRAM_RANGE} (Unigrams + Bigrams)")
print(f"     Sublinear TF : {TFIDF_SUBLINEAR_TF}")
print(f"     Min DF       : {TFIDF_MIN_DF}")
print(f"     Max DF       : {TFIDF_MAX_DF}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: LOAD DATASET & PREPARATION                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print(" TAHAP 1: MEMUAT DATASET HASIL PREPROCESSING")
print("=" * 70)

# Memuat dataset
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"    ✓ Dataset '{INPUT_FILE}' berhasil dimuat.")
except FileNotFoundError:
    # Fallback untuk pengujian lokal jika file tidak ada di direktori
    print(f"    [WARNING] File '{INPUT_FILE}' tidak ditemukan.")
    print("              Membuat data tiruan (dummy) untuk tujuan demonstrasi.")
    df = pd.DataFrame({
        'username': ['user1', 'user2', 'user3'],
        'tanggal': ['2026-06-01', '2026-06-02', '2026-06-03'],
        'rating': [4.5, 2.0, 3.5],
        'ulasan_asli': [
            'Film ini sangat bagus sekali, akting dimas keren.',
            'Kecewa banget alurnya lambat dan membosankan.',
            'Karakter ceritanya menarik, visualnya luar biasa indah.'
        ],
        'ulasan_preprocessed': [
            'film sangat bagus sekali akting dimas keren',
            'kecewa banget alur lambat membosankan',
            'karakter cerita menarik visual luar biasa indah'
        ]
    })

print(f"    Total data       : {len(df)} ulasan")
print(f"    Daftar kolom     : {list(df.columns)}")

# Validasi kolom ulasan preprocessed
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Kolom '{TEXT_COLUMN}' tidak ditemukan di dataset. Mohon periksa nama kolom dataset Anda.")

# Pembersihan nilai kosong (NaN) pasca preprocessing
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
df_clean = df[df[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)

print(f"    Ulasan valid     : {len(df_clean)} ulasan")
print(f"    Ulasan kosong    : {len(df) - len(df_clean)} ulasan (dihapus)")
print(f"    Kolom yang diekstrak : {TEXT_COLUMN} (hanya kolom ini yang diproses)")

# Preview ulasan preprocessed
print(f"\n    Preview 3 ulasan yang akan diekstraksi:")
for i in range(min(3, len(df_clean))):
    print(f"      [{i+1}] {df_clean[TEXT_COLUMN].iloc[i]}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: EKSTRAKSI FITUR TF-IDF                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print(" TAHAP 2: EKSTRAKSI FITUR DENGAN TF-IDF")
print("=" * 70)

# Inisialisasi TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
    sublinear_tf=TFIDF_SUBLINEAR_TF,
    min_df=TFIDF_MIN_DF,
    max_df=TFIDF_MAX_DF,
    norm='l2'
)

# Fitting dan transformasi teks
X_tfidf = vectorizer.fit_transform(df_clean[TEXT_COLUMN])

# Dapatkan informasi matriks hasil
n_docs, n_features = X_tfidf.shape
sparsity = (1.0 - X_tfidf.nnz / (n_docs * n_features)) * 100

print(f"    Matriks TF-IDF berhasil dibuat:")
print(f"      - Jumlah Dokumen (Ulasan) : {n_docs:,} ulasan")
print(f"      - Jumlah Fitur (Kosa Kata): {n_features:,} kata/frasa")
print(f"      - Kerapatan Matriks (Sparsity) : {sparsity:.2f}%")
print(f"      - N-Gram yang Terbentuk     : {TFIDF_NGRAM_RANGE[0]}-gram hingga {TFIDF_NGRAM_RANGE[1]}-gram")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: ANALISIS FITUR TERBOBOT TERTINGGI (TOP FEATURES)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print(" TAHAP 3: ANALISIS FITUR TF-IDF TERTINGGI")
print("=" * 70)

feature_names = vectorizer.get_feature_names_out()
mean_tfidf_scores = X_tfidf.mean(axis=0).A1

# Buat DataFrame untuk mempermudah manipulasi data
df_features = pd.DataFrame({
    'fitur': feature_names,
    'tfidf_rata_rata': mean_tfidf_scores
})

# Dapatkan 15 fitur dengan bobot rata-rata tertinggi
top_features = df_features.sort_values(by='tfidf_rata_rata', ascending=False).head(15).reset_index(drop=True)

print("\n  Top 15 Fitur TF-IDF dengan Bobot Rata-Rata Tertinggi:")
print(f"  {'No':<4} {'Fitur (Kata/Frasa)':<25} {'Rata-Rata TF-IDF':<15} {'Tipe Fitur'}")
print(f"  {'─' * 60}")
for idx, row in top_features.iterrows():
    words = row['fitur'].split()
    feature_type = "Bigram" if len(words) == 2 else "Unigram"
    print(f"  {idx+1:<4} \"{row['fitur']}\"{'':>{23-len(row['fitur'])}} {row['tfidf_rata_rata']:.6f}{'':8} {feature_type}")
print(f"  {'─' * 60}")

# Analisis sebaran tipe n-gram
unigrams_count = sum(1 for f in feature_names if len(f.split()) == 1)
bigrams_count = sum(1 for f in feature_names if len(f.split()) == 2)
print(f"\n  Statistik Sebaran N-Gram Fitur:")
print(f"    - Total Unigrams (1-gram): {unigrams_count:,} kata ({unigrams_count/max(n_features,1)*100:.1f}%)")
print(f"    - Total Bigrams  (2-gram): {bigrams_count:,} frasa ({bigrams_count/max(n_features,1)*100:.1f}%)")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7: VISUALISASI TOP 10 FITUR TF-IDF                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Mengambil 10 fitur teratas
top_10 = top_features.head(10)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))

bars = ax.barh(range(len(top_10)), top_10['tfidf_rata_rata'], color=colors, edgecolor='black', linewidth=0.6, height=0.6)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['fitur'], fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Rata-Rata Nilai TF-IDF', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Fitur dengan Nilai TF-IDF Tertinggi\n(Ulasan Preprocessed Film Ghost in the Cell)', fontsize=14, fontweight='bold', pad=15)

# Styling plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Tambah label nilai di ujung bar
for bar, tfidf_val in zip(bars, top_10['tfidf_rata_rata']):
    ax.text(bar.get_width() + max(top_10['tfidf_rata_rata'])*0.015, bar.get_y() + bar.get_height()/2,
            f"{tfidf_val:.5f}", va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik visualisasi disimpan: {OUTPUT_PLOT}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8: SIMPAN MODEL & MATRIKS FITUR                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print(" TAHAP 4: MENYIMPAN HASIL EKSTRAKSI FITUR")
print("=" * 70)

# Ekspor objek pkl
with open(OUTPUT_VECTORIZER, 'wb') as f:
    pickle.dump(vectorizer, f)

with open(OUTPUT_MATRIX, 'wb') as f:
    pickle.dump(X_tfidf, f)

print(f"    ✓ Model TF-IDF Vectorizer disimpan : {OUTPUT_VECTORIZER}")
print(f"    ✓ Matriks Fitur TF-IDF disimpan     : {OUTPUT_MATRIX}")
print("    Dataset hasil ekstraksi siap digunakan untuk pemodelan klasifikasi SVM.")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9: DOWNLOAD BERKAS (GOOGLE COLAB)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
try:
    from google.colab import files
    print("\n⏳ Menyiapkan unduhan berkas hasil ekstraksi...")
    
    # Download satu per satu dengan delay
    files.download(OUTPUT_PLOT)
    time.sleep(1.0)
    
    files.download(OUTPUT_VECTORIZER)
    time.sleep(1.0)
    
    files.download(OUTPUT_MATRIX)
    
    print("✓ Proses unduhan file dimulai secara otomatis.")
except ImportError:
    print("\n[INFO] Jalankan di Google Colab untuk mendownload berkas hasil ekstraksi secara otomatis.")
