"""
===============================================================================
PELABELAN SENTIMEN OTOMATIS (HYBRID: RATING & LEKSIKON BERBOBOT)
Untuk Analisis Sentimen Berbasis Aspek — Penelitian Skripsi

Platform   : Google Colab / Local
Input      : dataset_with_topics.csv (hasil topic modelling LDA)
Output     : 1. dataset_labeled_sentiment.csv (siap untuk SVM)
             2. distribusi_sentimen_hybrid.png
             3. korelasi_rating_vs_leksikon.png
             4. matriks_kebingungan_labeling.png
             5. wordcloud_sentimen.png
             6. distribusi_confidence_score.png
             7. aspek_vs_sentimen_final.png
             8. perbandingan_sebelum_sesudah_balancing.png

ALUR KERJA METODE OTOMATIS (7 TAHAP):
  Tahap 1: Load Dataset hasil Topic Modelling (LDA).
  Tahap 2: Inisialisasi Kamus Leksikon Berbobot Gradasi (Research-Grade).
  Tahap 3: Perhitungan Skor Leksikon otomatis dengan Jendela Negasi & Kata Penguat.
  Tahap 4: Pelabelan Hybrid (Integrasi Rating & Teks Berbobot) + Confidence Score.
  Tahap 5: Penentuan Label Sentimen Final & Deteksi Konflik (Informatif).
  Tahap 5.5: Penyeimbangan Distribusi Kelas Natural (Hybrid Resampling ~55.1% Positif : ~19.9% Netral : ~24.9% Negatif).
  Tahap 6: Verifikasi, Validasi & Evaluasi Statistik (Cohen's Kappa & Agreement).
  Tahap 7: Visualisasi Statistik Sentimen Akhir & Ekspor Dataset Terlabeli.

CATATAN:
  - Pelabelan dilakukan secara otomatis menggunakan formula hybrid
    yang mengintegrasikan rating bintang dan skor leksikon berbobot.
  - Penyeimbangan kelas menggunakan hybrid resampling (undersample +
    oversample) dengan proporsi natural (~55.1% Positif : ~19.9% Netral : ~24.9% Negatif).
  - Referensi: He & Garcia (2009); Chawla et al. (2002).
===============================================================================
"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: INSTALL DEPENDENCIES                                             ║
# ║  Jalankan cell ini PERTAMA KALI di Google Colab                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Menyediakan library visualisasi, pemrosesan data, dan analisis statistik:
# !pip install pandas numpy matplotlib seaborn wordcloud scikit-learn


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: IMPORT LIBRARY & UTILITIES                                       ║
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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

# Pengaturan visualisasi & reproduktivitas
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# ── Fungsi Logging Terstruktur ──
def log_header(tahap_num, judul, icon="[LOG]"):
    """Mencetak header tahap dengan format terstruktur."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print()
    print("=" * 80)
    print(f" {icon} TAHAP {tahap_num}: {judul}")
    print(f"    Waktu mulai: {timestamp}")
    print("=" * 80)


def log_done(tahap_num, judul, detail_dict=None):
    """Mencetak ringkasan selesainya sebuah tahap."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n    [SELESAI] Tahap {tahap_num}: {judul} -- {timestamp}")
    if detail_dict:
        for key, val in detail_dict.items():
            print(f"    {key:.<45s} {val}")
    print("-" * 80)


print("=" * 80)
print(" METODE OTOMATIS: PELABELAN SENTIMEN HYBRID SEIMBANG & VALIDASI STATISTIK")
print(" Film: Ghost in the Cell (2026) -- Letterboxd")
print(" Untuk Analisis Sentimen Berbasis Aspek -- Skripsi / Penelitian")
print("=" * 80)
print(f"[OK] Semua library berhasil diimpor!")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: KONFIGURASI PARAMETER                                            ║
# ║  Parameter yang mengontrol ambang batas & penyeimbangan proporsi seimbang ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Path File ──
INPUT_FILE = "dataset_with_topics.csv"
OUTPUT_FILE = "dataset_labeled_sentiment.csv"
TEXT_COLUMN = "ulasan_preprocessed"
ORIGINAL_TEXT_COLUMN = "ulasan_asli"  # atau ulasan_original jika ada
RATING_COLUMN = "rating"
ASPECT_COLUMN = "aspek"

# ── Parameter Pembagian Sentimen Berbasis Rating ──
RATING_POSITIF_MIN = 3.5      # Rating >= 3.5 dianggap Positif secara rating
RATING_NEGATIF_MAX = 2.0      # Rating <= 2.0 dianggap Negatif secara rating

# ── Parameter Bobot Hybrid Formula ──
W_RATING = 0.35
W_LEKSIKON = 0.65

# ── Parameter Ambang Batas Skor Hybrid Akhir ──
# Dikalibrasi seimbang agar pembagian kelas awal proporsional
HYBRID_POS_THRESHOLD = 0.15   # Skor hybrid >= 0.15 untuk Positif
HYBRID_NEG_THRESHOLD = -0.15  # Skor hybrid <= -0.15 untuk Negatif
# Nilai di antara [-0.15, 0.15] diklasifikasikan sebagai Netral

# ── Parameter Penyeimbangan Kelas (Natural Class Distribution) ──
# Target rasio disesuaikan proporsi natural (~55.12% Positif : ~19.94% Netral : ~24.94% Negatif)
ENABLE_BALANCING = True
TARGET_RATIO = {'positif': 0.5512, 'netral': 0.1994, 'negatif': 0.2494}
BALANCE_RANDOM_SEED = 42
MAINTAIN_TOTAL_DATA = True  # Total data dipertahankan sama dengan data awal (~3207 ulasan)

print("[OK] Konfigurasi parameter proporsi natural berhasil dimuat.")
print(f"     Rasio target NATURAL: Positif {TARGET_RATIO['positif']*100:.2f}% | Netral {TARGET_RATIO['netral']*100:.2f}% | Negatif {TARGET_RATIO['negatif']*100:.2f}%")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: LOAD DATASET (TAHAP 1)                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(1, "MEMUAT DATASET HASIL TOPIC MODELLING", "[LOAD]")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"  ✓ Berhasil memuat file: {INPUT_FILE}")
    print(f"    Total data awal: {len(df)} ulasan")
except FileNotFoundError:
    print(f"  [ERROR] File '{INPUT_FILE}' tidak ditemukan!")
    print(f"          Pastikan Anda telah menjalankan pemodelan topik LDA terlebih dahulu.")
    raise

# Validasi kolom wajib
required_cols = [TEXT_COLUMN, RATING_COLUMN, ASPECT_COLUMN]
for col in required_cols:
    if col not in df.columns:
        if col == TEXT_COLUMN and 'ulasan' in df.columns:
            TEXT_COLUMN = 'ulasan'
        else:
            raise KeyError(f"Kolom wajib '{col}' tidak ditemukan dalam dataset! Kolom yang ada: {list(df.columns)}")

if ORIGINAL_TEXT_COLUMN not in df.columns:
    if 'ulasan_original' in df.columns:
        ORIGINAL_TEXT_COLUMN = 'ulasan_original'
    elif 'ulasan_asli' in df.columns:
        ORIGINAL_TEXT_COLUMN = 'ulasan_asli'
    elif 'ulasan' in df.columns:
        ORIGINAL_TEXT_COLUMN = 'ulasan'
    else:
        ORIGINAL_TEXT_COLUMN = TEXT_COLUMN

# Bersihkan dari nilai kosong (NaN)
jumlah_awal = len(df)
df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).reset_index(drop=True)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df[ORIGINAL_TEXT_COLUMN] = df[ORIGINAL_TEXT_COLUMN].astype(str).str.strip()
df[RATING_COLUMN] = df[RATING_COLUMN].astype(float)
jumlah_dihapus = jumlah_awal - len(df)

log_done(1, "Load Dataset", {
    "Total data valid": len(df),
    "Data kosong dihapus": jumlah_dihapus,
    "Kolom teks": TEXT_COLUMN,
    "Kolom rating": RATING_COLUMN,
    "Kolom aspek": ASPECT_COLUMN
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: KAMUS LEKSIKON SENTIMEN BERBOBOT (TAHAP 2)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(2, "INISIALISASI KAMUS LEKSIKON BERBOBOT GRADASI", "[LEXICON]")

LEKSIKON_POSITIF = {
    'masterpiece': 2.0, 'mahakarya': 2.0, 'spektakuler': 2.0, 'fantastis': 2.0,
    'brilian': 2.0, 'sempurna': 2.0, 'outstanding': 2.0, 'terbaik': 2.0,
    'memukau': 2.0, 'rekomendasi': 2.0, 'recommended': 2.0, 'gokil': 2.0,
    'juara': 2.0, 'mantap': 2.0, 'mantul': 2.0, 'kece': 2.0, 'josss': 2.0,

    'bagus': 1.0, 'keren': 1.0, 'seru': 1.0, 'menarik': 1.0, 'hebat': 1.0,
    'indah': 1.0, 'cantik': 1.0, 'mengesankan': 1.0, 'apik': 1.0, 'ciamik': 1.0,
    'suka': 1.0, 'cinta': 1.0, 'senang': 1.0, 'puas': 1.0, 'terharu': 1.0,
    'menyentuh': 1.0, 'mengharukan': 1.0, 'inspiratif': 1.0, 'epik': 1.0,
    'membanggakan': 1.0, 'berkualitas': 1.0, 'oke': 1.0, 'ok': 1.0, 'good': 1.0,
    'great': 1.0, 'best': 1.0, 'nice': 1.0, 'amazing': 1.0, 'awesome': 1.0,
    'excellent': 1.0, 'superb': 1.0, 'wonderful': 1.0, 'perfect': 1.0,
    'menghibur': 1.0, 'segar': 1.0, 'fresh': 1.0, 'kreatif': 1.0, 'stabil': 1.0,
    'solid': 1.0, 'nyaman': 1.0, 'asli': 1.0, 'berani': 1.0, 'meyakinkan': 1.0,
    'efektif': 1.0, 'elegan': 1.0, 'antusias': 1.0, 'layak': 1.0, 'patut': 1.0,

    'lucu': 0.5, 'menegangkan': 0.5, 'unik': 0.5, 'emosional': 0.5, 'powerful': 0.5,
    'baper': 0.5, 'apresiasi': 0.5, 'damai': 0.5
}

LEKSIKON_NEGATIF = {
    'terburuk': -2.0, 'worst': -2.0, 'terrible': -2.0, 'sampah': -2.0,
    'busuk': -2.0, 'hancur': -2.0, 'ancur': -2.0, 'menyesal': -2.0,
    'sia-sia': -2.0, 'siasia': -2.0, 'disgusting': -2.0, 'garbage': -2.0,
    'trash': -2.0, 'memuakkan': -2.0, 'zonk': -2.0, 'flop': -2.0,

    'jelek': -1.0, 'buruk': -1.0, 'payah': -1.0, 'parah': -1.0, 'gagal': -1.0,
    'mengecewakan': -1.0, 'kecewa': -1.0, 'bosan': -1.0, 'membosankan': -1.0,
    'boring': -1.0, 'norak': -1.0, 'lebay': -1.0, 'alay': -1.0, 'receh': -1.0,
    'murahan': -1.0, 'kampungan': -1.0, 'absurd': -1.0, 'maksa': -1.0,
    'dipaksakan': -1.0, 'overrated': -1.0, 'bad': -1.0, 'awful': -1.0,
    'disappointing': -1.0, 'rugi': -1.0, 'benci': -1.0, 'marah': -1.0,
    'kesal': -1.0, 'lemah': -1.0, 'datar': -1.0, 'monoton': -1.0,
    'klise': -1.0, 'cringe': -1.0, 'garing': -1.0, 'hambar': -1.0,
    'membingungkan': -1.0, 'berantakan': -1.0, 'kaku': -1.0, 'ngantuk': -1.0,

    'aneh': -0.5, 'nanggung': -0.5, 'tanggung': -0.5, 'kurang': -0.5,
    'lambat': -0.5, 'biasa': -0.4, 'standar': -0.4, 'mediocre': -0.5,
    'lumayan': -0.3, 'cukup': -0.3, 'b_aja': -0.4, 'mid': -0.5
}

KATA_NEGASI = {
    'tidak', 'bukan', 'kurang', 'belum', 'gak', 'nggak', 'enggak', 'tanpa', 'tdk', 'gk'
}

KATA_PENGUAT = {
    'sangat': 1.5, 'sekali': 1.5, 'banget': 1.5, 'bgt': 1.5,
    'amat': 1.4, 'terlalu': 1.4, 'agak': 0.7, 'sedikit': 0.7
}

log_done(2, "Inisialisasi Kamus Leksikon", {
    "Kata Positif": len(LEKSIKON_POSITIF),
    "Kata Negatif": len(LEKSIKON_NEGATIF),
    "Kata Negasi": len(KATA_NEGASI),
    "Kata Penguat": len(KATA_PENGUAT)
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: ALGORITMA SKORING LEKSIKON CANGGIH (TAHAP 3)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(3, "PERHITUNGAN SKOR LEKSIKON TEKS (NEGASI & PENGUAT)", "[SCORE]")

def hitung_skor_leksikon_advanced(text):
    if not isinstance(text, str):
        return 0.0, [], []

    tokens = text.lower().split()
    total_score = 0.0
    n = len(tokens)

    matching_pos = []
    matching_neg = []

    negasi_window = 0
    negasi_factor = 1.0

    i = 0
    while i < n:
        token = tokens[i]

        if token in KATA_NEGASI:
            negasi_window = 3
            negasi_factor = -1.0
            i += 1
            continue

        if negasi_window > 0:
            negasi_window -= 1
        else:
            negasi_factor = 1.0

        multiplier = 1.0
        if i > 0 and tokens[i-1] in KATA_PENGUAT:
            multiplier = KATA_PENGUAT[tokens[i-1]]
        elif i < n - 1 and tokens[i+1] in KATA_PENGUAT:
            multiplier = KATA_PENGUAT[tokens[i+1]]

        if token in LEKSIKON_POSITIF:
            raw_val = LEKSIKON_POSITIF[token]
            final_val = raw_val * multiplier * negasi_factor
            total_score += final_val
            prefix = "tidak_" if negasi_factor < 0 else ""
            matching_pos.append(f"{prefix}{token}({final_val:+.1f})")

        elif token in LEKSIKON_NEGATIF:
            raw_val = LEKSIKON_NEGATIF[token]
            final_val = raw_val * multiplier * negasi_factor
            total_score += final_val
            prefix = "tidak_" if negasi_factor < 0 else ""
            matching_neg.append(f"{prefix}{token}({final_val:+.1f})")

        i += 1

    return total_score, matching_pos, matching_neg


skor_leksikon_list = []
pos_words_list = []
neg_words_list = []

for text in df[TEXT_COLUMN]:
    skor, pos_w, neg_w = hitung_skor_leksikon_advanced(text)
    skor_leksikon_list.append(skor)
    pos_words_list.append(", ".join(pos_w) if pos_w else "-")
    neg_words_list.append(", ".join(neg_w) if neg_w else "-")

df['skor_leksikon'] = skor_leksikon_list
df['kata_positif_terdeteksi'] = pos_words_list
df['kata_negatif_terdeteksi'] = neg_words_list

log_done(3, "Perhitungan Skor Leksikon Advanced", {
    "Rata-rata Skor Leksikon": f"{df['skor_leksikon'].mean():.4f}",
    "Skor Maksimum": f"{df['skor_leksikon'].max():.2f}",
    "Skor Minimum": f"{df['skor_leksikon'].min():.2f}"
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7: LABELLING HYBRID & CONFIDENCE SCORE (TAHAP 4)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(4, "PROSES LABELLING HYBRID (INTEGRASI RATING & TEKS)", "[HYBRID]")

def tentukan_label_hybrid(row):
    rating = float(row[RATING_COLUMN])
    skor_lex = float(row['skor_leksikon'])

    rating_norm = ((rating - 0.5) / 4.5) * 2 - 1.0
    lex_norm = np.tanh(skor_lex / 2.0)
    skor_hybrid = (W_RATING * rating_norm) + (W_LEKSIKON * lex_norm)

    if skor_hybrid >= HYBRID_POS_THRESHOLD:
        label = 'positif'
    elif skor_hybrid <= HYBRID_NEG_THRESHOLD:
        label = 'negatif'
    else:
        label = 'netral'

    selisih = abs(rating_norm - lex_norm)
    confidence = 1.0 - (selisih / 2.0)
    confidence = max(0.0, min(1.0, confidence))

    return pd.Series([rating_norm, lex_norm, skor_hybrid, label, confidence])


df[['rating_normalisasi', 'leksikon_normalisasi', 'skor_hybrid', 'sentimen_otomatis', 'confidence_score']] = df.apply(tentukan_label_hybrid, axis=1)

df['label_rating'] = df[RATING_COLUMN].apply(
    lambda r: 'positif' if r >= RATING_POSITIF_MIN else ('negatif' if r <= RATING_NEGATIF_MAX else 'netral')
)
df['label_leksikon'] = df['skor_leksikon'].apply(
    lambda s: 'positif' if s >= 0.5 else ('negatif' if s <= -0.5 else 'netral')
)

log_done(4, "Labelling Hybrid & Confidence Scoring", {
    "Rata-rata Skor Hybrid": f"{df['skor_hybrid'].mean():.4f}",
    "Rata-rata Confidence Score": f"{df['confidence_score'].mean()*100:.2f}%",
    "Sentimen Otomatis Positif": (df['sentimen_otomatis'] == 'positif').sum(),
    "Sentimen Otomatis Netral": (df['sentimen_otomatis'] == 'netral').sum(),
    "Sentimen Otomatis Negatif": (df['sentimen_otomatis'] == 'negatif').sum()
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8: PENENTUAN LABEL SENTIMEN FINAL & DETEKSI KONFLIK (TAHAP 5)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(5, "PENENTUAN LABEL SENTIMEN FINAL & DETEKSI KONFLIK", "[FINAL]")

df['sentimen_final'] = df['sentimen_otomatis'].copy()
df['kategori_deteksi'] = 'Konsisten'

mask_sarkasme = ((df['label_rating'] == 'positif') & (df['skor_leksikon'] <= -1.0)) | \
                ((df['label_rating'] == 'negatif') & (df['skor_leksikon'] >= 1.0))
mask_ambigu = (df['confidence_score'] < 0.55) & (~mask_sarkasme)

df.loc[mask_sarkasme, 'kategori_deteksi'] = 'Sarkasme / Konflik Ekstrem'
df.loc[mask_ambigu, 'kategori_deteksi'] = 'Ambigu / Konflik Minor'

rawan_konflik = df[df['kategori_deteksi'] != 'Konsisten'].sort_values('confidence_score')
jumlah_konsisten = (df['kategori_deteksi'] == 'Konsisten').sum()
df['sumber_labeling'] = 'Otomatis (Hybrid Rating + Leksikon)'

print(f"  [INFORMASI DETEKSI KONFLIK]")
print(f"  Total ulasan konsisten                             : {jumlah_konsisten} ({jumlah_konsisten/len(df)*100:.1f}%)")
print(f"  Total ulasan terdeteksi konflik/sarkasme           : {len(rawan_konflik)} ({len(rawan_konflik)/len(df)*100:.1f}%)")
print()

log_done(5, "Penentuan Label Sentimen Final", {
    "Total Ulasan Labeled": len(df),
    "Metode": "Otomatis (Hybrid Rating + Leksikon)",
    "Ulasan Konsisten": jumlah_konsisten,
    "Ulasan Konflik": len(rawan_konflik)
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8.5: PENYEIMBANGAN DISTRIBUSI KELAS SEIMBANG (TAHAP 5.5)           ║
# ║  Hybrid Resampling untuk Menghasilkan Proporsi Seimbang (33.3% Each)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header("5.5", "PENYEIMBANGAN DISTRIBUSI KELAS (PROPORSI NATURAL)", "[BALANCE]")

df_full = df.copy()
dist_sebelum = df['sentimen_final'].value_counts()
_labels_order = ['positif', 'netral', 'negatif']
_colors_sent = {'positif': '#2ecc71', 'netral': '#95a5a6', 'negatif': '#e74c3c'}
_color_list = [_colors_sent[l] for l in _labels_order]

print(f"  DISTRIBUSI SENTIMEN SEBELUM BALANCING:")
for label in _labels_order:
    count = dist_sebelum.get(label, 0)
    pct = count / len(df_full) * 100
    print(f"    {label.capitalize():<12}: {count:>5d} ulasan ({pct:.1f}%)")
print()

if ENABLE_BALANCING:
    count_per_class = df['sentimen_final'].value_counts().to_dict()
    total_ideal = len(df_full)

    # Hitung target per kelas sesuai proporsi natural (~55.12% Positif, ~19.94% Netral, ~24.94% Negatif)
    target_counts = {
        'positif': int(total_ideal * TARGET_RATIO['positif']),
        'netral': int(total_ideal * TARGET_RATIO['netral']),
        'negatif': int(total_ideal * TARGET_RATIO['negatif'])
    }
    # Kompensasi pembulatan integer
    diff = total_ideal - sum(target_counts.values())
    if diff != 0:
        target_counts['positif'] += diff

    print(f"  STRATEGI PENYEIMBANGAN PROPORSI NATURAL:")
    print(f"    Total Target Dataset               : {sum(target_counts.values())} ulasan")
    print(f"    Proporsi Target                    : Positif {TARGET_RATIO['positif']*100:.2f}% | Netral {TARGET_RATIO['netral']*100:.2f}% | Negatif {TARGET_RATIO['negatif']*100:.2f}%")
    print(f"    Target per kelas:")
    for label in _labels_order:
        available = count_per_class.get(label, 0)
        target = target_counts[label]
        if target > available:
            action = f"OVERSAMPLE {available} -> {target} ({target/available:.1f}x)"
        elif target < available:
            action = f"UNDERSAMPLE {available} -> {target}"
        else:
            action = "SEMUA DIPERTAHANKAN"
        print(f"      {label.capitalize():<12}: {target:>5d} ({action})")
    print()

    # Resampling per kelas dengan mempertimbangkan stratifikasi aspek
    np.random.seed(BALANCE_RANDOM_SEED)
    dfs_balanced = []

    for label in _labels_order:
        df_class = df[df['sentimen_final'] == label]
        target = target_counts[label]

        if len(df_class) == target:
            dfs_balanced.append(df_class)
        elif len(df_class) > target:
            # Undersampling terstratifikasi berdasarkan aspek
            sampled_parts = []
            for aspek_name, aspek_group in df_class.groupby(ASPECT_COLUMN):
                n_sample = max(1, int(len(aspek_group) / len(df_class) * target))
                n_sample = min(n_sample, len(aspek_group))
                sampled_parts.append(aspek_group.sample(n=n_sample, random_state=BALANCE_RANDOM_SEED))

            sampled = pd.concat(sampled_parts)
            if len(sampled) < target:
                remaining = df_class[~df_class.index.isin(sampled.index)]
                extra_n = min(target - len(sampled), len(remaining))
                if extra_n > 0:
                    extra = remaining.sample(n=extra_n, random_state=BALANCE_RANDOM_SEED)
                    sampled = pd.concat([sampled, extra])
            elif len(sampled) > target:
                sampled = sampled.sample(n=target, random_state=BALANCE_RANDOM_SEED)

            dfs_balanced.append(sampled)
        else:
            # Oversampling acak terikat untuk minoritas
            tambahan = target - len(df_class)
            oversampled = df_class.sample(n=tambahan, replace=True, random_state=BALANCE_RANDOM_SEED)
            combined = pd.concat([df_class, oversampled])
            dfs_balanced.append(combined)

    df_balanced = pd.concat(dfs_balanced).reset_index(drop=True)

    dist_sesudah = df_balanced['sentimen_final'].value_counts()
    total_balanced = len(df_balanced)
    print(f"  DISTRIBUSI SENTIMEN SESUDAH BALANCING PROPORSI NATURAL:")
    for label in _labels_order:
        count = dist_sesudah.get(label, 0)
        pct = count / total_balanced * 100
        print(f"    {label.capitalize():<12}: {count:>5d} ulasan ({pct:.2f}%)")
    print()

    # Visualisasi Perbandingan Sebelum vs Sesudah Balancing
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    counts_before = [dist_sebelum.get(l, 0) for l in _labels_order]
    bars1 = axes[0].bar(_labels_order, counts_before, color=_color_list, edgecolor='black', linewidth=0.8)
    axes[0].set_title('SEBELUM Balancing\n(Distribusi Asli)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel('Kelas Sentimen', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Jumlah Ulasan', fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    for bar, count in zip(bars1, counts_before):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_before)*0.015,
                     f"{count}\n({count/sum(counts_before)*100:.1f}%)", ha='center', va='bottom',
                     fontweight='bold', fontsize=10)

    counts_after = [dist_sesudah.get(l, 0) for l in _labels_order]
    bars2 = axes[1].bar(_labels_order, counts_after, color=_color_list, edgecolor='black', linewidth=0.8)
    axes[1].set_title(f'SESUDAH Balancing (Proporsi Natural)\n(Positif {TARGET_RATIO["positif"]*100:.2f}% | Netral {TARGET_RATIO["netral"]*100:.2f}% | Negatif {TARGET_RATIO["negatif"]*100:.2f}%)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel('Kelas Sentimen', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Jumlah Ulasan', fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    for bar, count in zip(bars2, counts_after):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_after)*0.015,
                     f"{count}\n({count/sum(counts_after)*100:.2f}%)", ha='center', va='bottom',
                     fontweight='bold', fontsize=10)

    plt.suptitle('Perbandingan Distribusi Sentimen: Sebelum vs Sesudah Balancing Natural',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('perbandingan_sebelum_sesudah_balancing.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik perbandingan disimpan: perbandingan_sebelum_sesudah_balancing.png")

    df_full.to_csv('dataset_labeled_sentiment_FULL.csv', index=False, encoding='utf-8-sig')
    df = df_balanced

log_done("5.5", "Penyeimbangan Distribusi Kelas Natural", {
    "Total Data Sebelum Balancing": len(df_full),
    "Total Data Sesudah Balancing": len(df),
    "Proporsi Akhir": f"Natural ({TARGET_RATIO['positif']*100:.2f}% Positif : {TARGET_RATIO['netral']*100:.2f}% Netral : {TARGET_RATIO['negatif']*100:.2f}% Negatif)"
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9: EVALUASI STATISTIK AGREEMENT & COHEN'S KAPPA (TAHAP 6)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(6, "EVALUASI STATISTIK & KEANDALAN INTER-RATER", "[STATS]")

kappa_score = cohen_kappa_score(df['label_rating'], df['label_leksikon'])

if kappa_score < 0:
    interpretasi = "Sangat Buruk (No Agreement)"
elif kappa_score <= 0.20:
    interpretasi = "Sangat Lemah (Slight Agreement)"
elif kappa_score <= 0.40:
    interpretasi = "Cukup (Fair Agreement)"
elif kappa_score <= 0.60:
    interpretasi = "Sedang (Moderate Agreement)"
elif kappa_score <= 0.80:
    interpretasi = "Kuat (Substantial Agreement)"
else:
    interpretasi = "Sangat Kuat (Almost Perfect Agreement)"

agreement_rate = (df['label_rating'] == df['label_leksikon']).mean() * 100
sentimen_counts = df['sentimen_final'].value_counts()
total_reviews = len(df)

print(f"  TABEL 4.S1: DISTRIBUSI KELAS SENTIMEN AKHIR (PROPORSI NATURAL)")
print(f"  {'=' * 65}")
print(f"  {'No':<4} {'Kelas Sentimen':<20} {'Jumlah Ulasan':<18} {'Persentase (%)'}")
print(f"  {'-' * 60}")
for i, (label, count) in enumerate(sentimen_counts.items(), 1):
    percentage = (count / total_reviews) * 100
    print(f"  {i:<4} {label.capitalize():<20} {count:<18} {percentage:.2f}%")
print(f"  {'-' * 60}")
print(f"  {'':4} {'Total':<20} {total_reviews:<18} 100.00%")
print(f"  {'=' * 65}")
print()

print(f"  METRIK KEANDALAN LABEL (INTER-RATER RELIABILITY):")
print(f"  - Persentase Kesepakatan (Agreement Rate)  : {agreement_rate:.2f}%")
print(f"  - Nilai Cohen's Kappa (Rating vs Leksikon) : {kappa_score:.4f}")
print(f"  - Interpretasi Kesepakatan                : {interpretasi}")
print()

log_done(6, "Evaluasi Keandalan Statistik", {
    "Cohen's Kappa": f"{kappa_score:.4f}",
    "Agreement Rate": f"{agreement_rate:.2f}%",
    "Interpretasi": interpretasi
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10: STATISTIK & VISUALISASI HASIL AKHIR (TAHAP 7)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(7, "VISUALISASI STATISTIK SENTIMEN LENGKAP", "[VISUAL]")

colors_sent = {'positif': '#2ecc71', 'netral': '#95a5a6', 'negatif': '#e74c3c'}
labels_order = ['positif', 'netral', 'negatif']
color_list = [colors_sent[l] for l in labels_order]
counts = [sentimen_counts.get(l, 0) for l in labels_order]

# Visualisasi 1: Bar & Pie Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bars = axes[0].bar(labels_order, counts, color=color_list, edgecolor='black', linewidth=0.8)
axes[0].set_title('Distribusi Kelas Sentimen Akhir (Proporsi Natural)\n(Pelabelan Otomatis Hybrid)', fontsize=12, fontweight='bold', pad=12)
axes[0].set_xlabel('Kelas Sentimen', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Jumlah Ulasan', fontsize=11, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.5)

for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(counts)*0.015),
                 f"{count}\n({count/total_reviews*100:.2f}%)", ha='center', va='bottom', fontweight='bold', fontsize=10)

axes[1].pie(counts, labels=[f'{l.capitalize()}\n({c} ulasan)' for l, c in zip(labels_order, counts)],
            colors=color_list, autopct='%1.2f%%', startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}, wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
axes[1].set_title('Proporsi Sentimen Ulasan Akhir (Natural)', fontsize=12, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('distribusi_sentimen_hybrid.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Grafik 1 disimpan: distribusi_sentimen_hybrid.png")

# Visualisasi 2: Scatter Plot Rating vs Leksikon
fig, ax = plt.subplots(figsize=(9, 6))
rating_jitter = df[RATING_COLUMN] + np.random.uniform(-0.15, 0.15, size=len(df))
sns.scatterplot(x=rating_jitter, y=df['skor_leksikon'], hue=df['sentimen_final'], 
                palette=colors_sent, alpha=0.7, edgecolor='black', linewidth=0.5, ax=ax)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Korelasi Rating Bintang Letterboxd vs Skor Leksikon Teks', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Rating Bintang (dengan jitter)', fontsize=11, fontweight='bold')
ax.set_ylabel('Skor Sentimen Leksikon', fontsize=11, fontweight='bold')
ax.legend(title='Sentimen Akhir', loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('korelasi_rating_vs_leksikon.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Grafik 2 disimpan: korelasi_rating_vs_leksikon.png")

# Visualisasi 3: Matriks Kebingungan Rating vs Leksikon
cm = confusion_matrix(df['label_rating'], df['label_leksikon'], labels=labels_order)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order,
            linewidths=1.5, linecolor='black', square=True, cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
ax.set_title('Matriks Perbandingan Klasifikasi\nRating Bintang vs Kamus Leksikon', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Prediksi Leksikon Teks', fontsize=11, fontweight='bold')
ax.set_ylabel('Prediksi Rating Bintang', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('matriks_kebingungan_labeling.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Grafik 3 disimpan: matriks_kebingungan_labeling.png")

# Visualisasi 4: Histogram Confidence Score
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['confidence_score'], bins=15, kde=True, color='skyblue', edgecolor='black', ax=ax)
ax.axvline(df['confidence_score'].mean(), color='red', linestyle='--', linewidth=1.5, 
           label=f"Rata-rata: {df['confidence_score'].mean():.2f}")
ax.set_title('Distribusi Skor Kepercayaan (Confidence Score) Pelabelan Hybrid', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Skor Kepercayaan (0.0 - 1.0)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frekuensi Ulasan', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('distribusi_confidence_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Grafik 4 disimpan: distribusi_confidence_score.png")

# Visualisasi 5: Word Cloud Sentimen (Positif, Netral & Negatif)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, label in enumerate(labels_order):
    text_sub = " ".join(df[df['sentimen_final'] == label][TEXT_COLUMN].tolist())
    cmap = 'summer' if label == 'positif' else ('Blues' if label == 'netral' else 'autumn')
    if len(text_sub.strip()) > 0:
        wc = WordCloud(background_color='white', max_words=50, colormap=cmap,
                       width=500, height=400, random_state=42, prefer_horizontal=0.85).generate(text_sub)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'Word Cloud - Sentimen {label.capitalize()}', fontsize=12, fontweight='bold', pad=10)
        axes[idx].axis('off')
    else:
        axes[idx].text(0.5, 0.5, f'Tidak ada data ulasan {label}', ha='center', va='center')
        axes[idx].axis('off')

plt.suptitle("Representasi Kata Kunci Sentimen Final (Proporsi Natural)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('wordcloud_sentimen.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Grafik 5 disimpan: wordcloud_sentimen.png")

# Visualisasi 6: Hubungan Aspek LDA vs Sentimen Akhir
crosstab_aspect = pd.crosstab(df[ASPECT_COLUMN], df['sentimen_final'], normalize='index') * 100
fig, ax = plt.subplots(figsize=(10, 6))
crosstab_aspect.loc[:, labels_order].plot(kind='barh', stacked=True, color=color_list, edgecolor='black', ax=ax)
ax.set_title('Distribusi Sentimen Akhir Natural Berdasarkan Aspek Hasil LDA', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Persentase Proporsi (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Aspek Film', fontsize=11, fontweight='bold')
ax.legend(title='Sentimen Final', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5, axis='x')

for p in ax.patches:
    width = p.get_width()
    if width > 5.0:
        x = p.get_x() + width / 2
        y = p.get_y() + p.get_height() / 2
        ax.text(x, y, f"{width:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('aspek_vs_sentimen_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Grafik 6 disimpan: aspek_vs_sentimen_final.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11: EKSPOR DATASET & RINGKASAN AKHIR                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(8, "EKSPOR DATASET TERLABELI & TABEL CROSS-TABULATION", "[EXPORT]")

df_output = df.copy()
df_output['sentimen'] = df_output['sentimen_final']

df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"  ✓ Dataset terlabeli natural berhasil disimpan ke: {OUTPUT_FILE}")
print(f"    Total data tersimpan : {len(df_output)} baris")
print()

print(f"  TABEL 4.CT: TABEL SILANG (CROSS-TABULATION) ASPEK LDA x SENTIMEN AKHIR NATURAL")
print(f"  {'=' * 75}")
crosstab_raw = pd.crosstab(
    df_output[ASPECT_COLUMN],
    df_output['sentimen'],
    margins=True,
    margins_name='Total'
)
print(crosstab_raw.to_string())
print(f"  {'=' * 75}")
print()

try:
    from google.colab import files
    print("\n⏳ Menyiapkan unduhan otomatis untuk Google Colab...")
    files.download(OUTPUT_FILE)
    time.sleep(0.5)
    files.download('distribusi_sentimen_hybrid.png')
    time.sleep(0.5)
    files.download('korelasi_rating_vs_leksikon.png')
    time.sleep(0.5)
    files.download('matriks_kebingungan_labeling.png')
    time.sleep(0.5)
    files.download('distribusi_confidence_score.png')
    time.sleep(0.5)
    files.download('wordcloud_sentimen.png')
    time.sleep(0.5)
    files.download('aspek_vs_sentimen_final.png')
    time.sleep(0.5)
    if ENABLE_BALANCING:
        files.download('perbandingan_sebelum_sesudah_balancing.png')
        time.sleep(0.5)
        files.download('dataset_labeled_sentiment_FULL.csv')
    print("  ✓ Semua berkas berhasil terunduh otomatis ke komputer lokal Anda.")
except ImportError:
    print("\n[INFO] Berjalan di lingkungan lokal. Visualisasi disimpan di direktori aktif.")

print(f"\n" + "=" * 80)
print("              RINGKASAN AKHIR PELABELAN OTOMATIS (PROPORSI NATURAL)")
print("=" * 80)
print(f"  1. Kualitas Data:")
print(f"     - Ukuran Dataset Awal (sebelum balancing) : {len(df_full)} ulasan")
print(f"     - Ukuran Dataset Akhir (sesudah balancing): {len(df_output)} ulasan")
print(f"     - Rata-rata Skor Kepercayaan              : {df_output['confidence_score'].mean()*100:.2f}%")
print(f"  2. Distribusi Sentimen Final (Proporsi Natural):")
for label, count in sentimen_counts.items():
    print(f"     - {label.capitalize():<12} : {count:>5d} ulasan ({count/total_reviews*100:.2f}%)")
print(f"  3. Penyeimbangan Kelas:")
print(f"     - Metode                        : Hybrid Resampling (Undersample + Oversample)")
print(f"     - Proporsi Target                : Natural (Positif {TARGET_RATIO['positif']*100:.2f}% : Netral {TARGET_RATIO['netral']*100:.2f}% : Negatif {TARGET_RATIO['negatif']*100:.2f}%)")
print(f"  4. Metode Pelabelan              : Otomatis (Hybrid Rating + Leksikon)")
print(f"  5. Metrik Kesepakatan (Kappa)      : {kappa_score:.4f} ({interpretasi})")
print(f"  6. File Output Dataset SVM         : {OUTPUT_FILE} (kolom target: 'sentimen')")
print("=" * 80)
print(" PROSES LABELLING SENTIMEN PROPORSI NATURAL SELESAI DENGAN SUKSES!")
print("=" * 80)
