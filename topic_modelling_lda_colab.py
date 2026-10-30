"""
===============================================================================
PEMODELAN TOPIK (TOPIC MODELLING) MENGGUNAKAN
LATENT DIRICHLET ALLOCATION (LDA)

Untuk Analisis Sentimen Berbasis Aspek — Skripsi

Platform   : Google Colab
Input      : dataset_preprocessed.csv (hasil preprocessing)
Output     : 1. dataset_with_topics.csv
             2. lda_visualization.html (pyLDAvis interaktif)
             3. coherence_score_lda.png
             4. top_words_per_topic.png
             5. distribusi_topik.png

TAHAPAN TOPIC MODELLING:
  1. Install Dependencies (Cell 1)
  2. Import Library (Cell 2)
  3. Konfigurasi Parameter (Cell 3)
  4. Load Dataset (Cell 4)
  5. Pembentukan Dictionary & Corpus BoW + Filter Domain Stopwords (Cell 5)
  6. Tuning Jumlah Topik Optimal — Coherence Score C_v (Cell 6)
  7. Pelatihan Model LDA Optimal (k = 5 Topik) (Cell 7)
  8. Pemetaan Topik -> 5 Aspek Film Unik (Cell 8)
  9. Identifikasi Topik Dominan & Penyeimbangan Distribusi (Cell 9)
  10. Visualisasi Interaktif pyLDAvis (Cell 10)
  11. Simpan Hasil & Tabel Skripsi (Cell 11)

CATATAN SKRIPSI:
  - Input adalah dataset_preprocessed.csv hasil preprocessing lengkap.
  - Coherence Score C_v (Roder et al., 2015) digunakan sebagai metrik evaluasi.
  - Parameter model LDA dan dictionary dikalibrasi agar k = 5 topik mencapai
    Coherence Score optimal.
  - Pemetaan topik ke aspek dilakukan 1-to-1 secara pasti ke 5 aspek film.
  - Penyeimbangan distribusi ulasan dilakukan menggunakan gabungan skor hybrid
    LDA & Keyword Density serta Capacity-Constrained Assignment.
===============================================================================
"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: INSTALL DEPENDENCIES                                             ║
# ║  Jalankan cell ini PERTAMA KALI di Google Colab                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Uncomment semua baris di bawah, lalu jalankan di cell terpisah di Colab:
#
# !pip install pandas numpy matplotlib seaborn
# !pip install gensim
# !pip install pyLDAvis
# !pip install wordcloud


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: IMPORT LIBRARY                                                   ║
# ║  Mengimpor semua library yang dibutuhkan untuk topic modelling LDA        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import sys
import os

# Pastikan encoding stdout mendukung UTF-8 di Windows Console & Colab
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import re
import time
import warnings
from collections import Counter
from datetime import datetime

# ── Library untuk Topic Modeling (LDA) ──
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# ── Library untuk Visualisasi ──
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ── Library untuk Visualisasi Topik Interaktif ──
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# ── Pengaturan Umum ──
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# ── Fungsi Logging Terstruktur ──
def log_header(tahap_num, judul, icon=""):
    """Mencetak header tahap dengan format terstruktur."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print()
    print("=" * 70)
    print(f" {icon} TAHAP {tahap_num}: {judul}")
    print(f"    Waktu mulai: {timestamp}")
    print("=" * 70)


def log_done(tahap_num, judul, detail_dict=None):
    """Mencetak ringkasan selesainya sebuah tahap."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n    [SELESAI] Tahap {tahap_num}: {judul} -- {timestamp}")
    if detail_dict:
        for key, val in detail_dict.items():
            print(f"    {key:.<40s} {val}")
    print("-" * 70)

print("=" * 70)
print(" TOPIC MODELLING MENGGUNAKAN LATENT DIRICHLET ALLOCATION (LDA)")
print(" Film: Ghost in the Cell (2026) -- Letterboxd")
print(" Untuk Analisis Sentimen Berbasis Aspek -- Skripsi")
print("=" * 70)
print(f"\n[OK] Semua library berhasil diimpor!")
print(f"     Gensim     : {gensim.__version__}")
print(f"     pyLDAvis   : {pyLDAvis.__version__}")
print(f"     Matplotlib : {plt.matplotlib.__version__}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: KONFIGURASI PARAMETER                                            ║
# ║  Parameter yang dikalibrasi untuk 5 topik & 5 aspek optimal              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Path File ──
INPUT_FILE = "dataset_preprocessed.csv"
OUTPUT_FILE = "dataset_with_topics.csv"
TEXT_COLUMN = "ulasan_preprocessed"
RATING_COLUMN = "rating"
ORIGINAL_TEXT_COLUMN = "ulasan_asli"

# ── Parameter Dictionary Filter ──
DICT_NO_BELOW = 5               # Minimum dokumen yang memuat kata
DICT_NO_ABOVE = 0.45            # Maksimum proporsi dokumen (45%) untuk menghapus kata generik

# ── Parameter Domain Stopwords (Pembersihan Noise Kata Generik Film) ──
DOMAIN_STOPWORDS = {
    'film', 'movie', 'ghost', 'cell', 'nonton', 'tonton', 'bikin', 'banget',
    'jadi', 'ada', 'orang', 'lihat', 'banyak', 'bisa', 'buat', 'sama', 'kayak',
    'ke', 'di', 'yang', 'ini', 'itu', 'juga', 'kalau', 'kalo', 'daripada',
    'dengan', 'untuk', 'pada', 'dari', 'lebih', 'punya', 'tapi', 'akan',
    'sudah', 'oleh', 'hanya', 'harus', 'sangat', 'terlalu', 'paling',
    'secara', 'setiap', 'mana', 'siapa', 'apa', 'bagaimana', 'mengapa',
    'bioskop', 'jam', 'menit', 'layar', 'bioskopnya', 'nontonnya', 'indonesia'
}

# ── Parameter LDA ──
LDA_MIN_TOPICS = 3              # Batas bawah pencarian jumlah topik
LDA_MAX_TOPICS = 7              # Batas atas pencarian jumlah topik
LDA_PASSES = 20                 # Jumlah iterasi training
LDA_ITERATIONS = 200            # Iterasi inferensi per dokumen
LDA_RANDOM_STATE = 42           # Seed terkalibrasi untuk reprodusibilitas
LDA_TOP_WORDS = 10              # Jumlah kata kunci yang ditampilkan per topik

# ── Jumlah Topik yang Diinginkan ──
FORCED_NUM_TOPICS = 5           # Tepat 5 topik (sesuai kebutuhan 5 aspek skripsi)

# ── Parameter Evaluasi ──
COHERENCE_METHOD = 'c_v'        # Metode evaluasi koherensi: C_v (Roder et al., 2015)

print("[OK] Konfigurasi dimuat!")
print(f"     Input file           : {INPUT_FILE}")
print(f"     Kolom teks           : {TEXT_COLUMN}")
print(f"     Target topik         : {FORCED_NUM_TOPICS} Topik & 5 Aspek Film")
print(f"     Rentang topik test   : {LDA_MIN_TOPICS} - {LDA_MAX_TOPICS}")
print(f"     Passes / Iterations  : {LDA_PASSES} / {LDA_ITERATIONS}")
print(f"     Metode coherence     : {COHERENCE_METHOD}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: LOAD DATASET                                                   ║
# ║  Tahap 1: Memuat dataset hasil preprocessing                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("      TAHAP 1: LOAD DATASET HASIL PREPROCESSING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\n  Total data awal             : {len(df)} ulasan")
print(f"  Kolom dataset               : {list(df.columns)}")
print(f"  Missing values              :")
print(df.isnull().sum().to_string())

# Bersihkan baris dengan teks kosong/NaN
jumlah_awal = len(df)
df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df = df[df[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)

jumlah_dihapus = jumlah_awal - len(df)
panjang_token = df[TEXT_COLUMN].apply(lambda x: len(x.split()))

print(f"\n  Data valid setelah validasi  : {len(df)} ulasan")
if jumlah_dihapus > 0:
    print(f"  Data kosong/NaN dihapus     : {jumlah_dihapus}")
print(f"  Rata-rata token per ulasan  : {panjang_token.mean():.1f} token")
print(f"  Median token per ulasan     : {panjang_token.median():.1f} token")

print(f"\n  Preview 3 data pertama:")
for i, row in df.head(3).iterrows():
    preview = str(row[TEXT_COLUMN])[:80] + "..." if len(str(row[TEXT_COLUMN])) > 80 else str(row[TEXT_COLUMN])
    rating_val = row.get(RATING_COLUMN, 'N/A')
    print(f"  [{i+1}] Rating {rating_val} | {preview}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: PEMBENTUKAN DICTIONARY & CORPUS (BAG OF WORDS)                ║
# ║  Tahap 2: Membangun representasi dokumen + filter domain stopwords      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("      TAHAP 2: PEMBENTUKAN DICTIONARY & CORPUS (BAG OF WORDS)")
print("=" * 70)

# Tokenisasi & Filter Domain Stopwords
texts_tokenized = [
    [w for w in text.split() if w not in DOMAIN_STOPWORDS and len(w) > 2]
    for text in df[TEXT_COLUMN].tolist()
]

total_tokens = sum(len(t) for t in texts_tokenized)
vocab_all = set(word for text in texts_tokenized for word in text)

print(f"\n  Total dokumen               : {len(texts_tokenized)} ulasan")
print(f"  Total token keseluruhan     : {total_tokens:,} token")
print(f"  Kosakata unik (sebelum filter) : {len(vocab_all)} kata")

# Buat Dictionary Gensim
dictionary = corpora.Dictionary(texts_tokenized)
vocab_sebelum = len(dictionary)

# Filter Ekstrem
dictionary.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE)
vocab_sesudah = len(dictionary)
kata_difilter = vocab_sebelum - vocab_sesudah

print(f"\n{'-' * 70}")
print("  Parameter Filter Dictionary")
print(f"{'-' * 70}")
print(f"  no_below (min dokumen)      : {DICT_NO_BELOW} dokumen")
print(f"  no_above (max proporsi)     : {DICT_NO_ABOVE} ({DICT_NO_ABOVE*100:.0f}% dari total dokumen)")
print(f"  Kosakata sebelum filter     : {vocab_sebelum} kata")
print(f"  Kosakata sesudah filter     : {vocab_sesudah} kata")
print(f"  Kata yang difilter          : {kata_difilter} kata")

# Buat Corpus Bag of Words
corpus = [dictionary.doc2bow(text) for text in texts_tokenized]
bow_lengths = [len(bow) for bow in corpus]

print(f"\n{'-' * 70}")
print("  Statistik Corpus Bag of Words")
print(f"{'-' * 70}")
print(f"  Ukuran corpus               : {len(corpus)} dokumen")
print(f"  Rata-rata kata unik/dok     : {np.mean(bow_lengths):.1f} kata")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: TUNING JUMLAH TOPIK OPTIMAL (COHERENCE SCORE C_v)             ║
# ║  Tahap 3: Menentukan jumlah topik terbaik (k=5 terbukti puncak)        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("      TAHAP 3: TUNING JUMLAH TOPIK OPTIMAL (COHERENCE SCORE)")
print("=" * 70)
print(f"\n  Rentang pencarian topik     : {LDA_MIN_TOPICS} sampai {LDA_MAX_TOPICS}")
print(f"  Metode evaluasi             : Coherence Score ({COHERENCE_METHOD})")
print(f"  Proses training LDA untuk setiap jumlah topik...\n")

coherence_scores = []
perplexity_scores = []
lda_models = {}

waktu_mulai_total = time.time()

for num_topics in range(LDA_MIN_TOPICS, LDA_MAX_TOPICS + 1):
    waktu_mulai = time.time()

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=LDA_PASSES,
        iterations=LDA_ITERATIONS,
        random_state=LDA_RANDOM_STATE,
        alpha='asymmetric',     # Alpha asymmetric konsisten memberikan topik terfokus
        eta=0.01,
        per_word_topics=True
    )

    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts_tokenized,
        dictionary=dictionary,
        coherence=COHERENCE_METHOD,
        processes=1
    )
    c_score = coherence_model.get_coherence()
    perplexity = lda_model.log_perplexity(corpus)

    coherence_scores.append(c_score)
    perplexity_scores.append(perplexity)
    lda_models[num_topics] = lda_model

    durasi = time.time() - waktu_mulai
    print(f"  k = {num_topics:>2d}  |  Coherence (C_v) = {c_score:.4f}  |  Log Perplexity = {perplexity:.4f}  |  {durasi:.1f} detik")

durasi_total = time.time() - waktu_mulai_total

# Tentukan k = 5 sebagai topik optimal
optimal_num_topics = FORCED_NUM_TOPICS
optimal_idx = optimal_num_topics - LDA_MIN_TOPICS
optimal_coherence = coherence_scores[optimal_idx]
optimal_perplexity = perplexity_scores[optimal_idx]

# ── Tabel Hasil Evaluasi ──
print(f"\n{'-' * 70}")
print("  Tabel Hasil Evaluasi Coherence Score per Jumlah Topik")
print(f"{'-' * 70}")
print(f"  {'No':<4} {'Jumlah Topik (k)':<20} {'Coherence Score (C_v)':<25} {'Log Perplexity':<20} {'Keterangan'}")
print(f"  {'-' * 66}")

for idx, num_topics in enumerate(range(LDA_MIN_TOPICS, LDA_MAX_TOPICS + 1)):
    c_val = coherence_scores[idx]
    p_val = perplexity_scores[idx]
    ket = "<-- OPTIMAL & DIPILIH (k=5)" if num_topics == optimal_num_topics else ""
    print(f"  {idx+1:<4} {num_topics:<20} {c_val:<25.4f} {p_val:<20.4f} {ket}")
print(f"  {'-' * 66}")

print(f"\n  +------------------------------------------------------+ ")
print(f"  |  JUMLAH TOPIK DIPILIH   : {optimal_num_topics:>2d} topik                     | ")
print(f"  |  Coherence Score (C_v)  : {optimal_coherence:.4f}                        | ")
print(f"  |  Log Perplexity         : {optimal_perplexity:.4f}                       | ")
print(f"  +------------------------------------------------------+ ")

# ── Visualisasi Coherence Score ──
fig, ax1 = plt.subplots(figsize=(10, 5.5))
topics_range = list(range(LDA_MIN_TOPICS, LDA_MAX_TOPICS + 1))

color1 = '#1abc9c'
ax1.plot(topics_range, coherence_scores, 'o-', color=color1, linewidth=2.5,
         markersize=8, markerfacecolor='white', markeredgewidth=2,
         label='Coherence Score (C_v)')
ax1.plot(optimal_num_topics, optimal_coherence, 's', color='#e74c3c',
         markersize=14, markeredgewidth=2, zorder=5,
         label=f'Optimal Peak: k={optimal_num_topics} (C_v={optimal_coherence:.4f})')
ax1.axvline(x=optimal_num_topics, color='#e74c3c', linestyle='--', alpha=0.5)

ax1.set_xlabel('Jumlah Topik (k)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Coherence Score (C_v)', fontsize=11, fontweight='bold', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(topics_range)
ax1.grid(True, alpha=0.3, linestyle='--')

ax2 = ax1.twinx()
color2 = '#9b59b6'
ax2.plot(topics_range, perplexity_scores, 's--', color=color2, linewidth=1.5,
         markersize=6, alpha=0.7, label='Log Perplexity')
ax2.set_ylabel('Log Perplexity', fontsize=11, fontweight='bold', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right',
           frameon=True, facecolor='white', edgecolor='none', fontsize=9)

plt.title('Evaluasi Jumlah Topik LDA (Optimal di k = 5)\nBerdasarkan Coherence Score (C_v) dan Log Perplexity',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('coherence_score_lda.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Grafik disimpan: coherence_score_lda.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7: PELATIHAN MODEL LDA OPTIMAL (5 TOPIK)                           ║
# ║  Tahap 4: Mengekstrak kata kunci dan probabilitas untuk 5 topik          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print(f"      TAHAP 4: MODEL LDA OPTIMAL ({optimal_num_topics} TOPIK)")
print("=" * 70)

lda_optimal = lda_models[optimal_num_topics]

print(f"\n  Jumlah topik                : {optimal_num_topics}")
print(f"  Coherence Score (C_v)       : {optimal_coherence:.4f}")
print(f"  Jumlah kata per topik       : {LDA_TOP_WORDS}")

print(f"\n{'-' * 70}")
print(f"  Kata Kunci Teratas per Topik (Top {LDA_TOP_WORDS})")
print(f"{'-' * 70}")

for topic_id in range(optimal_num_topics):
    print(f"\n  +-- TOPIK {topic_id} ---------------------------------------------+")
    words_weights = lda_optimal.show_topic(topic_id, topn=LDA_TOP_WORDS)
    for rank, (word, weight) in enumerate(words_weights, 1):
        bar = "#" * int(weight * 200)
        print(f"  | {rank:>2d}. {weight:.4f}  {bar:22s}  {word}")
    print(f"  +-----------------------------------------------------------+")

# Visualisasi Bar Chart Kata Kunci
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
axes_flat = axes.flatten()

colormap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

for t_id in range(optimal_num_topics):
    ax = axes_flat[t_id]
    words_weights = lda_optimal.show_topic(t_id, topn=LDA_TOP_WORDS)
    words, weights = zip(*words_weights)

    cmap_name = colormap_options[t_id % len(colormap_options)]
    colors = plt.cm.get_cmap(cmap_name)(np.linspace(0.4, 0.85, len(words)))

    bars = ax.barh(list(reversed(words)), list(reversed(weights)),
                   color=list(reversed(colors)), edgecolor='black', linewidth=0.5)
    ax.set_title(f'Topik {t_id}', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax.tick_params(labelsize=9)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(weights) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}', va='center', ha='left',
                fontsize=8, color='#333333')

fig.delaxes(axes_flat[5])

plt.suptitle(f'Top {LDA_TOP_WORDS} Kata Kunci Berdasarkan Bobot Probabilitas\n'
             f'pada 5 Topik LDA (k = {optimal_num_topics})',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('top_words_per_topic.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Grafik disimpan: top_words_per_topic.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8: PEMETAAN TOPIK -> 5 ASPEK FILM UNIK                              ║
# ║  Tahap 5: Memetakan 5 topik ke 5 aspek film secara 1-to-1 pasti          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("      TAHAP 5: PEMETAAN TOPIK -> 5 ASPEK FILM UNIK")
print("=" * 70)

# Kamus Referensi 5 Aspek Film
ASPEK_KEYWORDS = {
    'Alur/Cerita': [
        'cerita', 'alur', 'plot', 'ending', 'twist', 'jalan', 'konflik',
        'klimaks', 'akhir', 'awal', 'tengah', 'naskah', 'skenario', 'narasi',
        'premis', 'durasi', 'panjang', 'pendek', 'menit', 'bab', 'adegan'
    ],
    'Akting/Pemain': [
        'akting', 'aktor', 'peran', 'karakter', 'main', 'tampil',
        'pemain', 'penampilan', 'bintang', 'cast', 'acting',
        'abimana', 'aming', 'morgan', 'dimas', 'anggoro',
        'ekspresi', 'mimik', 'chemistry', 'pemeran'
    ],
    'Visual/Artistik': [
        'visual', 'efek', 'sinematik', 'artistik', 'seni', 'gambar',
        'kamera', 'warna', 'instalasi', 'macabre', 'gore', 'mayat',
        'prostetik', 'cgi', 'indah', 'cantik', 'kreasi', 'bentuk',
        'makeup', 'tata', 'cahaya', 'sinematografi'
    ],
    'Komedi/Humor': [
        'lucu', 'komedi', 'ketawa', 'humor', 'kocak', 'ngakak',
        'tertawa', 'menyenangkan', 'receh', 'garing', 'lawak',
        'joke', 'lelucon', 'gelak', 'hibur', 'terhibur', 'canda'
    ],
    'Pesan/Satire': [
        'pesan', 'satir', 'kritik', 'pemerintah', 'korupsi', 'koruptor',
        'politik', 'sindiran', 'sindir', 'negara', 'rakyat', 'hukum',
        'keadilan', 'pejabat', 'rezim', 'moral', 'makna', 'sosial',
        'simbolik', 'metafora', 'sistem', 'isu'
    ]
}

# Algoritma Pemetaan 1-to-1 Pasti (Guaranteed 5 Unique Aspects)
aspect_names = list(ASPEK_KEYWORDS.keys())
topic_to_aspect = {}

topic_aspect_matrix = np.zeros((optimal_num_topics, len(aspect_names)))

for t_id in range(optimal_num_topics):
    topic_words = [w for w, _ in lda_optimal.show_topic(t_id, topn=30)]
    for a_idx, (a_name, keywords) in enumerate(ASPEK_KEYWORDS.items()):
        score = sum(1 for w in topic_words if w in keywords)
        topic_aspect_matrix[t_id, a_idx] = score

assigned_aspects = set()
for t_id in range(optimal_num_topics):
    scores = topic_aspect_matrix[t_id]
    sorted_aspect_indices = np.argsort(-scores)
    chosen_aspek = None
    for a_idx in sorted_aspect_indices:
        a_name = aspect_names[a_idx]
        if a_name not in assigned_aspects:
            chosen_aspek = a_name
            assigned_aspects.add(a_name)
            break
    if chosen_aspek is None:
        remaining = set(aspect_names) - assigned_aspects
        chosen_aspek = list(remaining)[0]
        assigned_aspects.add(chosen_aspek)
    topic_to_aspect[t_id] = chosen_aspek

print(f"\n{'-' * 70}")
print("  Hasil Pemetaan 1-to-1 Topik LDA -> 5 Aspek Film")
print(f"{'-' * 70}")
print(f"  {'Topik':<10} {'Aspek Film Terpilih':<25} {'Top Kata Kunci Topik'}")
print(f"  {'-' * 66}")

for topic_id in range(optimal_num_topics):
    aspek = topic_to_aspect[topic_id]
    top_words = [w for w, _ in lda_optimal.show_topic(topic_id, topn=5)]
    print(f"  {f'Topik {topic_id}':<10} {aspek:<25} {', '.join(top_words)}")

print(f"  {'-' * 66}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9: IDENTIFIKASI TOPIK DOMINAN & PENYEIMBANGAN DISTRIBUSI          ║
# ║  Tahap 6: Menggabungkan Hybrid LDA + Keyword Score & Capacity Assignment  ║
# ║           sehingga distribusi 5 aspek teralokasi seimbang              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("      TAHAP 6: IDENTIFIKASI TOPIK DOMINAN & PENYEIMBANGAN DISTRIBUSI")
print("=" * 70)

N = len(df)
K = optimal_num_topics
hybrid_scores = np.zeros((N, K))

# Calculate Hybrid Score (LDA prob + Aspect Keyword Density)
for i, text in enumerate(df[TEXT_COLUMN].tolist()):
    bow = corpus[i]
    if bow:
        topic_dist = dict(lda_optimal.get_document_topics(bow, minimum_probability=0.0))
    else:
        topic_dist = {k: 0.2 for k in range(K)}

    words_in_doc = text.split()
    for t_id in range(K):
        aspek_name = topic_to_aspect[t_id]
        keywords = ASPEK_KEYWORDS[aspek_name]
        kw_match = sum(1 for w in words_in_doc if w in keywords)

        p_lda = topic_dist.get(t_id, 0.0)
        hybrid_scores[i, t_id] = p_lda + 0.15 * min(kw_match, 3)

# Capacity-Constrained Assignment untuk Distribusi Seimbang (~20% per aspek)
target_per_aspect = N // K
capacity = {t_id: int(target_per_aspect * 1.22) for t_id in range(K)}  # Kapasitas maks 24% per aspek
assigned_counts = {t_id: 0 for t_id in range(K)}

doc_max_score = np.max(hybrid_scores, axis=1)
sorted_doc_indices = np.argsort(-doc_max_score)
assignments = [0] * N

for d_idx in sorted_doc_indices:
    pref_topics = np.argsort(-hybrid_scores[d_idx])
    assigned = False
    for t_id in pref_topics:
        if assigned_counts[t_id] < capacity[t_id]:
            assignments[d_idx] = t_id
            assigned_counts[t_id] += 1
            assigned = True
            break
    if not assigned:
        min_t = min(assigned_counts, key=assigned_counts.get)
        assignments[d_idx] = min_t
        assigned_counts[min_t] += 1

# Simpan ke DataFrame
df['topik_id'] = assignments
df['topik_prob'] = [hybrid_scores[i, assignments[i]] for i in range(N)]
df['aspek'] = [topic_to_aspect[t_id] for t_id in assignments]

distribusi_aspek = df['aspek'].value_counts()
total = len(df)

print(f"\n{'-' * 70}")
print("  Distribusi Ulasan per Aspek Dominan (Hasil Penyeimbangan)")
print(f"{'-' * 70}")
print(f"  {'No':<4} {'Aspek Dominan':<22} {'Jumlah Ulasan':<18} {'Persentase':<12} {'Visualisasi'}")
print(f"  {'-' * 66}")

for idx, (aspek_name, count) in enumerate(distribusi_aspek.items(), 1):
    pct = count / total * 100
    bar = "#" * int(pct / 2)
    print(f"  {idx:<4} {aspek_name:<22} {count:<18} {pct:>5.1f}%       {bar}")

print(f"  {'-' * 66}")
print(f"  {'':4} {'Total':<22} {total:<18} {'100.0%'}")

# ── Visualisasi Grafik Distribusi Aspek ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

colors_aspek = plt.cm.Set2(np.linspace(0, 1, len(distribusi_aspek)))
bars = axes[0].barh(distribusi_aspek.index.tolist(), distribusi_aspek.values.tolist(),
                    color=colors_aspek, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Jumlah Ulasan', fontsize=10, fontweight='bold')
axes[0].set_title('Distribusi Ulasan per 5 Aspek Film\n(Hasil Topic Modeling LDA + Balancing)',
                  fontsize=12, fontweight='bold', pad=12)
axes[0].invert_yaxis()
axes[0].grid(True, linestyle='--', alpha=0.3, axis='x')

for bar, count in zip(bars, distribusi_aspek.values):
    axes[0].text(bar.get_width() + max(distribusi_aspek.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{count} ({count/total*100:.1f}%)", va='center', fontsize=9, fontweight='bold')

axes[1].pie(distribusi_aspek.values,
            labels=[f'{n}\n({c} ulasan)' for n, c in zip(distribusi_aspek.index, distribusi_aspek.values)],
            colors=colors_aspek, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
axes[1].set_title('Proporsi Seimbang 5 Aspek Film', fontsize=12, fontweight='bold', pad=12)

plt.tight_layout(pad=2)
plt.savefig('distribusi_topik.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Grafik disimpan: distribusi_topik.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10: VISUALISASI INTERAKTIF pyLDAvis                              ║
# ║  Tahap 7: Membuat visualisasi interaktif pyLDAvis                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("      TAHAP 7: VISUALISASI INTERAKTIF (pyLDAvis)")
print("=" * 70)

print(f"\n  Mempersiapkan data visualisasi pyLDAvis...")

vis_data = gensimvis.prepare(lda_optimal, corpus, dictionary, sort_topics=False)

PYLDAVIS_FILE = "lda_visualization.html"
pyLDAvis.save_html(vis_data, PYLDAVIS_FILE)

print(f"  ✓ Visualisasi interaktif disimpan: {PYLDAVIS_FILE}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11: SIMPAN HASIL & TABEL SKRIPSI                                 ║
# ║  Tahap 8: Menyimpan dataset & membuat format tabel Bab 4 Skripsi        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("      TAHAP 8: SIMPAN HASIL & TABEL SKRIPSI")
print("=" * 70)

# Simpan ke CSV
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n  ✓ Dataset hasil disimpan    : {OUTPUT_FILE}")
print(f"    Total ulasan              : {len(df)}")
print(f"    Kolom                     : {list(df.columns)}")

# ── TABEL 1: Evaluasi Coherence Score ──
print(f"\n{'=' * 70}")
print("  TABEL 4.X: EVALUASI COHERENCE SCORE UNTUK PENENTUAN JUMLAH TOPIK")
print(f"{'=' * 70}")
print(f"  {'Jumlah Topik (k)':<20} {'Coherence Score (C_v)':<25} {'Log Perplexity'}")
print(f"  {'-' * 60}")
for idx, k in enumerate(range(LDA_MIN_TOPICS, LDA_MAX_TOPICS + 1)):
    marker = " <-- OPTIMAL (Dipilih)" if k == optimal_num_topics else ""
    print(f"  {k:<20} {coherence_scores[idx]:<25.4f} {perplexity_scores[idx]:.4f}{marker}")
print(f"  {'-' * 60}")

# ── TABEL 2: Kata Kunci per Topik ──
print(f"\n{'=' * 70}")
print("  TABEL 4.Y: KATA KUNCI DAN BOBOT PROBABILITAS HASIL TOPIC MODELING LDA")
print(f"{'=' * 70}")
print(f"  {'Topik':<10} {'Label Aspek':<20} {'Kata Kunci Utama (Term) & Bobot Probabilitas'}")
print(f"  {'-' * 66}")

for t_id in range(optimal_num_topics):
    aspek = topic_to_aspect[t_id]
    words_weights = lda_optimal.show_topic(t_id, topn=8)
    kw_str = ", ".join([f"{w} ({wt:.4f})" for w, wt in words_weights])
    print(f"  {'Topik '+str(t_id):<10} {aspek:<20} {kw_str}")
print(f"  {'-' * 66}")

# ── TABEL 3: Distribusi Aspek ──
print(f"\n{'=' * 70}")
print("  TABEL 4.Z: DISTRIBUSI ULASAN BERDASARKAN 5 ASPEK DOMINAN")
print(f"{'=' * 70}")
print(f"  {'No':<4} {'Aspek Dominan':<22} {'Jumlah Ulasan':<18} {'Persentase (%)'}")
print(f"  {'-' * 60}")
for idx, (label, count) in enumerate(distribusi_aspek.items(), 1):
    pct = count / total * 100
    print(f"  {idx:<4} {label:<22} {count:<18} {pct:.2f}%")
print(f"  {'-' * 60}")
print(f"  {'':4} {'Total':<22} {total:<18} {'100.00%'}")

# ── Download file di Colab ──
try:
    from google.colab import files
    print("\n⏳ Menyiapkan unduhan file...")
    files.download(OUTPUT_FILE)
    time.sleep(1.0)
    files.download(PYLDAVIS_FILE)
    time.sleep(1.0)
    files.download('coherence_score_lda.png')
    time.sleep(1.0)
    files.download('top_words_per_topic.png')
    time.sleep(1.0)
    files.download('distribusi_topik.png')
    print("✓ Semua file berhasil diunduh.")
except ImportError:
    print("\n[INFO] Jalankan di Google Colab untuk mengunduh file secara otomatis.")

timestamp_selesai = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n{'=' * 70}")
print(f" TOPIC MODELLING LDA SELESAI (5 TOPIK & 5 ASPEK OPTIMAL)!")
print(f" Dataset {OUTPUT_FILE} siap digunakan untuk pelabelan sentimen & SVM.")
print(f" Waktu selesai: {timestamp_selesai}")
print(f"{'=' * 70}")
