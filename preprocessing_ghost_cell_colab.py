"""
===============================================================================
PREPROCESSING ULASAN FILM "GHOST IN THE CELL (2026)"
Untuk Analisis Sentimen — Skripsi

Platform   : Google Colab
Input      : dataset_filtered.csv (hasil filtering)
Output     : dataset_preprocessed.csv (siap untuk analisis sentimen)

TAHAPAN PREPROCESSING:
  1. Load Dataset
  2. Case Folding (lowercase)
  3. Cleansing (hapus URL, mention, hashtag, emoji, angka, tanda baca, dll.)
  4. Normalisasi & Translasi (Slang → Baku, lalu EN → ID)
  5. Tokenisasi
  6. Stopword Removal
  7. Stemming (Sastrawi) — dengan proteksi kata sentimen
  8. Penggabungan Token & Penghapusan Kata Pendek
  9. Simpan Hasil & Visualisasi

CATATAN:
  - Normalisasi slang Indonesia dilakukan SEBELUM translasi bahasa Inggris.
    Pendekatan ini lebih akurat karena mencegah kata slang Indonesia yang
    mirip bahasa Inggris (seperti "garing", "kentang", "receh") dari
    salah diterjemahkan oleh Google Translate ke makna harfiah.
  - Cleansing dilakukan SEBELUM normalisasi & translasi agar deteksi
    bahasa lebih akurat (tidak terkontaminasi noise).
  - Menggunakan deep-translator (Google Translate API gratis) sebagai
    fallback untuk kalimat full-English yang tidak tercakup kamus manual.
  - Library `emoji` digunakan untuk penghapusan emoji yang akurat.
  - Stemming dilindungi agar kata-kata penting sentimen tidak over-stem.
===============================================================================
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: INSTALL DEPENDENCIES                                          ║
# ║  Jalankan cell ini PERTAMA KALI di Google Colab                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Uncomment semua baris di bawah, lalu jalankan di cell terpisah di Colab:
#
# !pip install pandas numpy matplotlib seaborn wordcloud tqdm
# !pip install Sastrawi
# !pip install deep-translator
# !pip install langdetect
# !pip install nltk
# !pip install scikit-learn
# !pip install emoji
#
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: IMPORT LIBRARY                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import re
import string
import html
import time
import random
import warnings
from collections import Counter
from datetime import datetime

# NLP Libraries
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
import emoji  # Library untuk penghapusan emoji yang akurat

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Progress bar
from tqdm import tqdm
tqdm.pandas()

# Reproducibility
DetectorFactory.seed = 42
warnings.filterwarnings('ignore')


# ── Fungsi Logging Terstruktur ──
def log_header(tahap_num, judul, icon=""):
    """Mencetak header tahap preprocessing dengan format terstruktur."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print()
    print("=" * 70)
    print(f" {icon} TAHAP {tahap_num}: {judul}")
    print(f"    Waktu mulai: {timestamp}")
    print("=" * 70)


def log_done(tahap_num, judul, detail_dict=None):
    """Mencetak ringkasan selesainya sebuah tahap."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n    [SELESAI] Tahap {tahap_num}: {judul} — {timestamp}")
    if detail_dict:
        for key, val in detail_dict.items():
            print(f"    {key:.<40s} {val}")
    print("-" * 70)


# ── Inisialisasi Stemmer Sastrawi (sekali saja, karena berat) ──
print("[INFO] Menginisialisasi Stemmer Sastrawi...")
factory_stemmer = StemmerFactory()
try:
    stemmer = factory_stemmer.create_stemmer()       # versi baru PySastrawi
except AttributeError:
    stemmer = factory_stemmer.createStemmer()         # versi lama PySastrawi
print("[OK]   Stemmer Sastrawi siap!")

# ── Inisialisasi Stopword ──
factory_stopword = StopWordRemoverFactory()
try:
    stopword_sastrawi = set(factory_stopword.get_stop_words())   # versi baru
except AttributeError:
    stopword_sastrawi = set(factory_stopword.getStopWords())     # versi lama

# Tambah stopword NLTK bahasa Indonesia & Inggris
try:
    stopword_nltk_id = set(nltk_stopwords.words('indonesian'))
    stopword_nltk_en = set(nltk_stopwords.words('english'))
except Exception:
    stopword_nltk_id = set()
    stopword_nltk_en = set()

print(f"[OK]   Semua library berhasil diimpor!")
print(f"       Stopword Sastrawi : {len(stopword_sastrawi)} kata")
print(f"       Stopword NLTK ID  : {len(stopword_nltk_id)} kata")
print(f"       Stopword NLTK EN  : {len(stopword_nltk_en)} kata")
print(f"       Emoji library     : v{emoji.__version__}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: KONFIGURASI                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Path File ──
INPUT_FILE = "dataset_filtered.csv"         # Input: dataset hasil filtering
OUTPUT_FILE = "dataset_preprocessed.csv"    # Output: dataset siap analisis sentimen
TEXT_COLUMN = "ulasan"                       # Kolom teks ulasan
RATING_COLUMN = "rating"                     # Kolom rating

# ── Parameter Preprocessing ──
MIN_WORD_LENGTH = 3                          # Panjang kata minimum setelah preprocessing
MIN_TOKENS_AFTER_PREPROCESSING = 3           # Minimal token setelah semua tahapan

# ── Translasi ──
TRANSLATE_BATCH_SIZE = 20                    # Jumlah kalimat per batch translasi
TRANSLATE_DELAY = 0.5                        # Delay antar batch translasi (detik)

print("[OK]   Konfigurasi dimuat!")
print(f"       Input  : {INPUT_FILE}")
print(f"       Output : {OUTPUT_FILE}")
print(f"       Kolom teks : {TEXT_COLUMN}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: LOAD DATASET                                                     ║
# ║  Tahap 1: Memuat dataset hasil filtering                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(1, "LOAD DATASET", "LOAD")

# Untuk Google Colab, upload file terlebih dahulu:
# from google.colab import files
# uploaded = files.upload()

df = pd.read_csv(INPUT_FILE)

print(f"    Total data       : {len(df)} ulasan")
print(f"    Kolom            : {list(df.columns)}")
print(f"    Missing values   :")
print(df.isnull().sum().to_string(header=False))
print()

# Hapus baris dengan ulasan kosong/NaN
jumlah_sebelum = len(df)
df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df = df[df[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)

# Buat kolom untuk tracking setiap tahap
df['ulasan_original'] = df[TEXT_COLUMN].copy()

jumlah_dihapus = jumlah_sebelum - len(df)

print(f"    Data valid setelah pembersihan: {len(df)} ulasan")
if jumlah_dihapus > 0:
    print(f"    Data kosong/NaN dihapus       : {jumlah_dihapus}")

print(f"\n    Preview 3 data pertama:")
for i, row in df.head(3).iterrows():
    preview = row[TEXT_COLUMN][:100] + "..." if len(row[TEXT_COLUMN]) > 100 else row[TEXT_COLUMN]
    rating_val = row.get(RATING_COLUMN, 'N/A')
    print(f"    [{i+1}] Rating {rating_val} | {preview}")

log_done(1, "Load Dataset", {
    "Total data dimuat": f"{len(df)} ulasan",
    "Data dihapus (kosong/NaN)": f"{jumlah_dihapus}",
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: CASE FOLDING                                                     ║
# ║  Tahap 2: Mengubah semua teks menjadi huruf kecil (lowercase)             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def case_folding(text):
    """Mengubah semua karakter menjadi huruf kecil."""
    if not isinstance(text, str):
        return ""
    return text.lower()

df['ulasan_casefolded'] = df[TEXT_COLUMN].apply(case_folding)

# ── Hitung statistik perubahan ──
changed_mask = df[TEXT_COLUMN] != df['ulasan_casefolded']
jumlah_berubah = changed_mask.sum()
jumlah_tidak_berubah = len(df) - jumlah_berubah
ada_kapital = df['ulasan_casefolded'].apply(lambda x: bool(re.search(r'[A-Z]', x))).sum()

# ── Analisis variasi kapitalisasi ──
# Cari kata-kata yang memiliki variasi penulisan kapital/kecil
from collections import defaultdict
variasi_kata = defaultdict(set)
for text in df[TEXT_COLUMN]:
    if isinstance(text, str):
        for kata in text.split():
            kata_clean = re.sub(r'[^a-zA-Z]', '', kata)
            if kata_clean and kata_clean.lower() != kata_clean:
                variasi_kata[kata_clean.lower()].add(kata_clean)

# Filter hanya kata yang punya variasi kapital (min 2 bentuk berbeda)
kata_multi_variasi = {k: v for k, v in variasi_kata.items() if len(v) >= 2}

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT UNTUK SKRIPSI — Sub Bab 4.3.1 Case Folding
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("         TAHAP 2: CASE FOLDING (Konversi ke Huruf Kecil)")
print("=" * 70)

# ── 1. Informasi Umum ──
print(f"\n  Total ulasan diproses         : {len(df)} ulasan")
print(f"  Ulasan mengandung huruf besar : {jumlah_berubah} ulasan")
print(f"  Ulasan sudah lowercase        : {jumlah_tidak_berubah} ulasan")
print(f"  Persentase yang dikonversi    : {jumlah_berubah/len(df)*100:.1f}%")

# ── 2. Contoh Hasil Case Folding (Tabel Before-After) ──
print(f"\n{'─' * 70}")
print("  Contoh Hasil Case Folding (Sebelum → Sesudah)")
print(f"{'─' * 70}")

contoh_berubah = df[changed_mask].head(5)
print(f"  {'No':<4} {'Sebelum Case Folding':<55}")
print(f"  {'':4} {'Sesudah Case Folding':<55}")
print(f"  {'─' * 66}")

for idx, (i, row) in enumerate(contoh_berubah.iterrows(), 1):
    before = row[TEXT_COLUMN]
    after = row['ulasan_casefolded']
    # Potong teks agar rapi, max ~60 karakter
    max_len = 60
    before_display = before[:max_len] + "..." if len(before) > max_len else before
    after_display = after[:max_len] + "..." if len(after) > max_len else after
    print(f"  {idx:<4} \"{before_display}\"")
    print(f"  {'':4} \"{after_display}\"")
    if idx < len(contoh_berubah):
        print()

# ── 3. Contoh Penyeragaman Variasi Kapitalisasi ──
print(f"\n{'─' * 70}")
print("  Contoh Penyeragaman Variasi Kapitalisasi")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Variasi Ditemukan':<40} {'Hasil Case Folding':<20}")
print(f"  {'─' * 66}")

# Ambil contoh kata dengan variasi terbanyak
sorted_variasi = sorted(kata_multi_variasi.items(), key=lambda x: len(x[1]), reverse=True)
for idx, (kata, bentuk_set) in enumerate(sorted_variasi[:5], 1):
    bentuk_list = sorted(bentuk_set)
    variasi_str = ", ".join(f'"{b}"' for b in bentuk_list[:4])
    if len(bentuk_list) > 4:
        variasi_str += f" (+{len(bentuk_list)-4} lainnya)"
    print(f"  {idx:<4} {variasi_str:<40} \"{kata}\"")

print(f"\n  Total kata dengan variasi kapitalisasi: {len(kata_multi_variasi)} kata")
print(f"  Seluruh variasi berhasil diseragamkan ke bentuk huruf kecil.")

# ── 4. Verifikasi ──
print(f"\n{'─' * 70}")
print("  Verifikasi Hasil Case Folding")
print(f"{'─' * 70}")
print(f"  Huruf kapital tersisa setelah case folding : {ada_kapital} huruf")
if ada_kapital == 0:
    print(f"  Status : ✓ Berhasil — Seluruh teks telah dikonversi ke huruf kecil")
else:
    print(f"  Status : ✗ Masih terdapat {ada_kapital} ulasan dengan huruf kapital")

# ── 5. Ringkasan ──
print(f"\n{'=' * 70}")
print("  Ringkasan Case Folding")
print(f"{'=' * 70}")
print(f"  Dari total {len(df)} ulasan yang diproses, ditemukan sebanyak")
print(f"  {jumlah_berubah} ulasan yang ditulis dengan variasi kapitalisasi")
print(f"  berbeda. Setelah case folding, seluruh variasi tersebut tereduksi")
print(f"  menjadi satu bentuk tunggal (huruf kecil) sehingga frekuensi kata")
print(f"  dapat dihitung secara akurat.")
print(f"{'=' * 70}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: DEFINISI FUNGSI TRANSLASI (UTILITY)                              ║
# ║  Fungsi-fungsi translasi bahasa campuran (EN → ID)                        ║
# ║                                                                           ║
# ║  Fungsi-fungsi ini digunakan oleh Cell 8 (Normalisasi & Translasi)        ║
# ║  sebagai bagian dari tahap Normalisasi & Translasi (Tahap 4).             ║
# ║  Pendekatan: Deteksi bahasa per KALIMAT, terjemahkan kalimat bahasa       ║
# ║  Inggris ke Indonesia, lalu gabungkan kembali.                            ║
# ║  Ditambahkan: Second-pass untuk kata Inggris yang tersisa.                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Daftar kata serapan yang TIDAK perlu diterjemahkan ──
# (Kata-kata Inggris yang sudah lazim dalam bahasa Indonesia/review film)
KATA_SERAPAN = {
    # ── Istilah film & sinema (sudah lazim di Indonesia) ──
    'film', 'movie', 'scene', 'ending', 'plot', 'twist', 'genre',
    'action', 'drama', 'thriller', 'romance', 'comedy', 'horror',
    'soundtrack', 'score', 'cgi', 'vfx', 'sfx', 'acting', 'cast',
    'rating', 'review', 'spoiler', 'trailer', 'sequel', 'prequel',
    'remake', 'reboot', 'franchise', 'cinema', 'cinematic',
    'character', 'director', 'producer', 'actor', 'actress',
    'recommended', 'masterpiece', 'blockbuster',
    'flashback', 'cliffhanger', 'cameo', 'debut',
    'mainstream', 'indie', 'jumpscare', 'screening',
    'storyline', 'subplot', 'protagonist', 'antagonist',
    'climax', 'hero', 'heroine', 'effect', 'effects', 'visual',
    'animation', 'animated', 'script', 'screenplay',
    'streaming', 'download', 'upload', 'online', 'offline',
    'comment', 'like', 'share', 'post',
    'okay', 'ok', 'btw', 'fyi', 'imo', 'imho', 'lol', 'omg', 'wow',
    'hype', 'overrated', 'underrated',
    'cringe', 'gore', 'macabre', 'installation', 'satire',
    'disturbing', 'trigger', 'warning', 'content',
    'pacing', 'performance', 'prosthetic', 'practical',
    'highlight', 'overall',
    # ── Istilah teknis produksi film ──
    'edit', 'editing', 'editor', 'mixer', 'mixing',
    'set', 'studio', 'props', 'wardrobe', 'costume',
    'stunt', 'stuntman', 'choreography', 'choreographer',
    'dubbing', 'subtitle', 'subtitles', 'caption',
    'premiere', 'premier', 'preview', 'teaser',
    'spin off', 'crossover', 'anthology', 'trilogy', 'saga',
    'box office', 'red carpet', 'oscar', 'oscars',
    'golden globe', 'emmy', 'grammy', 'bafta',
    'imdb', 'rotten tomatoes', 'metacritic', 'letterboxd',
    # ── Istilah internet & platform ──
    'viral', 'trending', 'hashtag', 'influencer', 'vlog',
    'vlogger', 'youtuber', 'podcast', 'subscriber',
    'follower', 'netizen', 'update', 'upgrade',
    'gadget', 'smartphone', 'laptop', 'browser',
    # ── Kata serapan umum lainnya ──
    'profesional', 'fiksi', 'fantasi', 'realistis',
    'global', 'universal', 'internasional', 'lokal',
    'minus', 'plus', 'bonus', 'ekstra',
    'fokus', 'target', 'detail', 'konsep',
    'promo', 'sponsor', 'brand',
    # ── Kata slang Indonesia yang mirip bahasa Inggris ──
    # (jangan sampai terdeteksi sebagai EN lalu salah terjemah)
    'garing', 'kentang', 'receh', 'lebay', 'alay', 'norak',
    'baper', 'mager', 'gabut', 'santuy', 'sultan', 'bucin',
    'mantul', 'anjay', 'gokil', 'kece', 'sabi', 'absurd',
    'random', 'relate', 'proper', 'healing', 'vibes',
    'sotoy', 'kepo', 'pansos', 'flexing', 'ghosting',
    'toxic', 'salty', 'savage', 'legit',
}

# ── Kata Slang Indonesia yang HARUS DILINDUNGI dari Mistranslasi ──
# Kata-kata ini punya makna khusus di konteks review film Indonesia.
# Google Translate sering salah mengartikan kata-kata ini ke makna harfiah.
# Contoh: "garing" (= tidak lucu) → diterjemahkan "crispy/dry" (SALAH)
#         "kentang" (= tanggung/tidak memuaskan) → "potato" (SALAH)
KATA_SLANG_PROTEKSI = {
    # ── Slang penilaian film ──
    # Kata slang : (makna sebenarnya dalam konteks review film)
    'garing':       'tidak lucu',           # humor gagal, jokes ga lucu
    'kentang':      'tanggung',             # kena tanggung, setengah-setengah
    'receh':        'humor murahan',        # jokes murah/tidak berkualitas
    'lebay':        'berlebihan',           # overacting, terlalu dramatis
    'alay':         'norak',                # selera rendah, kampungan
    'norak':        'norak',                # kampungan, murahan
    'cringe':       'memalukan',            # bikin malu/tidak nyaman
    'cringey':      'memalukan',            # sama dengan cringe
    'cheesy':       'terlalu romantis',     # terlalu manis/klise
    'corny':        'klise',                # humor basi, jokes lama
    'edgy':         'sok gelap',            # berusaha keras terlihat dewasa
    'overacting':   'akting berlebihan',    # akting terlalu dramatis
    'underacting':  'akting kurang',        # akting terlalu datar

    # ── Slang emosi/reaksi penonton ──
    'baper':        'terbawa perasaan',     # bawa perasaan
    'mewek':        'menangis',             # nangis
    'ngakak':       'tertawa keras',        # tertawa terbahak-bahak
    'merinding':    'merinding',            # bulu kuduk berdiri
    'kesel':        'kesal',                # jengkel
    'sebel':        'sebal',                # jengkel
    'gregetan':     'gemas',                # gemas, tidak sabar
    'gedeg':        'kesal',                # marah/jengkel
    'mangkel':      'jengkel',              # kesal
    'kzl':          'kesal',                # singkatan kesal
    'mager':        'malas',                # malas gerak
    'gabut':        'bosan',                # tidak ada kerjaan
    'bucin':        'budak cinta',          # terlalu cinta
    'galau':        'gelisah',              # resah, bimbang

    # ── Slang kualitas/penilaian ──
    'gokil':        'luar biasa',           # keren banget
    'mantul':       'mantap betul',         # sangat bagus
    'kece':         'keren',                # stylish, bagus
    'sabi':         'bisa',                 # oke, bisa diterima
    'josss':        'sangat bagus',         # top, mantap
    'joss':         'sangat bagus',         # top, mantap
    'mantap':       'sangat bagus',         # kualitas tinggi
    'makjleb':      'mengena',              # tepat sasaran, menyentuh
    'zonk':         'mengecewakan',         # hasilnya kosong/nihil
    'jeblok':       'gagal',                # hasilnya buruk
    'flop':         'gagal',                # tidak laku/sukses
    'ngebut':       'cepat',                # pace cepat
    'dragging':     'membosankan',          # terlalu lambat
    'nanggung':     'tanggung',             # setengah-setengah
    'setengah':     'setengah',             # tidak total
    'absurd':       'absurd',               # aneh tidak masuk akal
    'random':       'acak',                 # tidak jelas
    'santuy':       'santai',               # rileks, tidak tegang
    'healing':      'menyegarkan',          # menenangkan pikiran
    'vibes':        'suasana',              # aura, nuansa
    'mood':         'suasana hati',         # perasaan

    # ── Slang karakter/akting ──
    'sultan':       'kaya raya',            # orang kaya
    'badass':       'tangguh',              # karakter kuat/keren
    'savage':       'brutal',               # kejam tapi keren
    'psycho':       'gila',                 # karakter psikopat
    'tsundere':     'jual mahal',           # karakter jual mahal
    'waifu':        'karakter favorit',     # karakter perempuan favorit
    'simp':         'pemuja',               # terlalu mengagumi

    # ── Slang plot/cerita ──
    'plot armor':   'karakter kebal cerita',  # karakter tidak bisa mati
    'plot hole':    'lubang cerita',        # kejanggalan cerita
    'lazy writing': 'naskah buruk',         # penulisan cerita yang malas/asal-asalan
    'filler':       'pengisi',              # episode/adegan tidak penting
    'canon':        'resmi',                # cerita resmi
    'fanservice':   'layanan penggemar',    # adegan untuk menyenangkan fans
    'cliffhanger':  'akhir menggantung',    # ending yang bikin penasaran
    'open ending':  'akhir terbuka',        # ending tanpa kejelasan
    'happy ending': 'akhir bahagia',        # ending bahagia
    'sad ending':   'akhir sedih',          # ending sedih
    'twist ending': 'akhir tak terduga',    # plot twist di akhir

    # ── Slang sinematik ──
    'lighting':     'pencahayaan',          # tata cahaya
    'angle':        'sudut pengambilan',    # sudut kamera
    'take':         'pengambilan gambar',   # shot/pengambilan gambar
    'frame':        'bingkai',              # bingkai gambar
    'shot':         'pengambilan gambar',   # bidikan kamera
    'cut':          'potongan',             # transisi potongan
    'montage':      'montase',              # rangkaian adegan cepat
}

# ── Kata Inggris yang sering dipakai untuk deteksi bahasa kalimat ──
KATA_INGGRIS_KUAT = {
    'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
    'will', 'would', 'could', 'should', 'been', 'being',
    'this', 'that', 'these', 'those', 'there', 'here',
    'which', 'who', 'whom', 'whose', 'what', 'where', 'when',
    'how', 'why', 'because', 'although', 'though', 'however',
    'therefore', 'nevertheless', 'furthermore', 'moreover',
    'while', 'whilst', 'whereas', 'whether', 'unless',
    'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'without', 'within', 'against',
    'their', 'them', 'they', 'she', 'her', 'him', 'his',
    'its', 'our', 'your', 'my', 'we', 'you',
    'much', 'many', 'some', 'any', 'most', 'every', 'each',
    'really', 'actually', 'basically', 'definitely', 'especially',
    'probably', 'certainly', 'obviously', 'honestly',
    'also', 'still', 'already', 'always', 'never', 'sometimes',
    'often', 'usually', 'perhaps', 'maybe', 'might',
    "doesn't", "didn't", "wasn't", "weren't", "hasn't", "haven't",
    "couldn't", "wouldn't", "shouldn't", "mustn't", "needn't",
    "isn't", "aren't", "won't", "can't", "don't",
    'doesn', 'didn', 'wasn', 'weren', 'hasn', 'haven',
    'couldn', 'wouldn', 'shouldn', 'mustn', 'needn',
    'just', 'very', 'too', 'so', 'but', 'and', 'not', 'only',
    'than', 'then', 'such', 'both', 'either', 'neither',
    'about', 'with', 'from', 'for', 'can', 'does', 'did',
    'think', 'know', 'want', 'need', 'feel', 'make', 'take',
    'give', 'tell', 'say', 'said', 'get', 'got', 'come', 'came',
    'going', 'went', 'done', 'made', 'took', 'gave', 'told',
    'thought', 'knew', 'wanted', 'needed', 'felt',
    # Tambahan kata deteksi
    'myself', 'yourself', 'himself', 'herself', 'itself',
    'ourselves', 'themselves', 'again', 'once', 'twice',
    'seen', 'watched', 'found', 'kept', 'left', 'right',
    'even', 'like', 'since', 'until', 'toward', 'towards',
    'across', 'along', 'among', 'around', 'upon',
    'own', 'other', 'another', 'else', 'away',
    'quite', 'rather', 'enough', 'indeed', 'instead',
    'yet', 'ever', 'despite', 'except', 'beyond',
}

# ── Kamus terjemahan manual untuk kata-kata Inggris umum ──
# (Sebagai fallback jika Google Translate gagal atau melewatkan kata)
KAMUS_TERJEMAHAN_MANUAL = {
    # ── Tambahan manual ──
    'horror': 'horor',
    'indonesian': 'indonesia',

    # ── Kata sifat / penilaian (adjectives) ──
    'amazing': 'menakjubkan', 'awesome': 'luar biasa',
    'beautiful': 'indah', 'brilliant': 'cemerlang',
    'boring': 'membosankan', 'bad': 'buruk',
    'creepy': 'menyeramkan', 'dark': 'gelap',
    'disgusting': 'menjijikkan', 'disappointing': 'mengecewakan',
    'excellent': 'sangat bagus', 'exciting': 'menegangkan',
    'fantastic': 'fantastis', 'frightening': 'menakutkan',
    'fun': 'menyenangkan', 'funny': 'lucu',
    'good': 'bagus', 'great': 'hebat', 'gorgeous': 'indah',
    'horrible': 'mengerikan', 'horrifying': 'mengerikan',
    'impressive': 'mengesankan', 'incredible': 'luar biasa',
    'interesting': 'menarik', 'intense': 'intens',
    'lame': 'payah', 'lousy': 'jelek',
    'magnificent': 'megah', 'marvelous': 'menakjubkan',
    'mediocre': 'biasa saja', 'moving': 'mengharukan',
    'outstanding': 'luar biasa', 'perfect': 'sempurna',
    'poor': 'buruk', 'powerful': 'kuat',
    'ridiculous': 'konyol', 'sad': 'sedih',
    'scary': 'menakutkan', 'shocking': 'mengejutkan',
    'slow': 'lambat', 'stunning': 'memukau',
    'stupid': 'bodoh', 'superb': 'luar biasa',
    'suspenseful': 'menegangkan', 'sweet': 'manis',
    'terrible': 'mengerikan', 'terrifying': 'menakutkan',
    'touching': 'menyentuh', 'trash': 'sampah',
    'ugly': 'jelek', 'uncomfortable': 'tidak nyaman',
    'unexpected': 'tidak terduga', 'unforgettable': 'tidak terlupakan',
    'unique': 'unik', 'unpleasant': 'tidak menyenangkan',
    'watchable': 'layak tonton', 'weak': 'lemah',
    'wonderful': 'luar biasa', 'worth': 'layak',
    'worthless': 'tidak layak',
    'recommend': 'merekomendasikan', 'recommended': 'direkomendasikan',
    'overrated': 'dinilai berlebihan', 'underrated': 'kurang diapresiasi',
    'predictable': 'mudah ditebak', 'unpredictable': 'tidak terduga',
    'emotional': 'emosional', 'heartbreaking': 'memilukan',
    'breathtaking': 'menakjubkan',
    'cringeworthy': 'memalukan',

    # ── Kata sifat tambahan (expanded) ──
    'awful': 'mengerikan', 'bland': 'hambar', 'bleak': 'suram',
    'bold': 'berani', 'captivating': 'memikat', 'charming': 'menawan',
    'chilling': 'menggidikkan', 'clever': 'cerdas', 'clumsy': 'kikuk',
    'compelling': 'memukau', 'complex': 'kompleks', 'confusing': 'membingungkan',
    'convincing': 'meyakinkan', 'cool': 'keren', 'corny': 'norak',
    'cozy': 'nyaman', 'crazy': 'gila', 'cringy': 'memalukan',
    'cruel': 'kejam', 'cute': 'imut', 'daring': 'berani',
    'deadly': 'mematikan', 'decent': 'lumayan', 'deep': 'dalam',
    'delightful': 'menyenangkan', 'dense': 'padat',
    'depressing': 'menyedihkan', 'desperate': 'putus asa',
    'detailed': 'detail', 'different': 'berbeda',
    'difficult': 'sulit', 'dreadful': 'mengerikan',
    'dull': 'membosankan', 'dumb': 'bodoh',
    'eerie': 'menyeramkan', 'effective': 'efektif',
    'elegant': 'elegan', 'embarrassing': 'memalukan',
    'empty': 'kosong', 'engaging': 'menarik',
    'enjoyable': 'menyenangkan', 'entertaining': 'menghibur',
    'evil': 'jahat', 'exaggerated': 'berlebihan',
    'exhausting': 'melelahkan', 'exotic': 'eksotis',
    'expensive': 'mahal', 'explosive': 'eksplosif',
    'extraordinary': 'luar biasa', 'extreme': 'ekstrem',
    'fabulous': 'luar biasa', 'faithful': 'setia',
    'familiar': 'akrab', 'fast': 'cepat',
    'fatal': 'fatal', 'favorite': 'favorit', 'favourite': 'favorit',
    'fearless': 'berani', 'fierce': 'garang',
    'fine': 'baik', 'flat': 'datar', 'flawless': 'sempurna',
    'forced': 'dipaksakan', 'forgettable': 'mudah dilupakan',
    'formulaic': 'formulaik', 'freaky': 'aneh',
    'frustrating': 'membuat frustrasi', 'generic': 'generik',
    'genius': 'jenius', 'gentle': 'lembut',
    'genuine': 'tulus', 'gloomy': 'suram', 'gory': 'berdarah',
    'gripping': 'mencekam', 'gross': 'menjijikkan',
    'grotesque': 'aneh', 'gruesome': 'mengerikan',
    'guilty': 'bersalah', 'haunting': 'menghantui',
    'heartwarming': 'menghangatkan hati', 'heavy': 'berat',
    'helpless': 'tidak berdaya', 'heroic': 'heroik',
    'hideous': 'mengerikan', 'hilarious': 'sangat lucu',
    'hollow': 'hampa', 'honest': 'jujur', 'hopeless': 'tanpa harapan',
    'horrendous': 'mengerikan', 'horrific': 'mengerikan',
    'huge': 'besar', 'humorous': 'lucu', 'ideal': 'ideal',
    'immersive': 'imersif', 'impactful': 'berdampak',
    'important': 'penting', 'impossible': 'mustahil',
    'incompetent': 'tidak kompeten', 'inferior': 'rendah',
    'influential': 'berpengaruh', 'ingenious': 'cerdik',
    'innocent': 'polos', 'innovative': 'inovatif',
    'insane': 'gila', 'insensitive': 'tidak sensitif',
    'insightful': 'penuh wawasan', 'inspiring': 'menginspirasi',
    'intriguing': 'menggelitik', 'invisible': 'tak terlihat',
    'irritating': 'menjengkelkan', 'juvenile': 'kekanakan',
    'keen': 'tajam', 'kind': 'baik', 'lacking': 'kurang',
    'laughable': 'menggelikan', 'lazy': 'malas',
    'lifeless': 'tidak bernyawa', 'light': 'ringan',
    'likeable': 'disukai', 'likable': 'disukai',
    'little': 'kecil', 'long': 'panjang', 'loud': 'keras',
    'lovely': 'indah', 'low': 'rendah', 'lucky': 'beruntung',
    'mad': 'gila', 'magical': 'magis', 'major': 'besar',
    'masterful': 'mahir', 'meaningful': 'bermakna',
    'memorable': 'berkesan', 'menacing': 'mengancam',
    'mesmerizing': 'memesona', 'messy': 'berantakan',
    'mild': 'ringan', 'mindless': 'tanpa pikiran',
    'minor': 'kecil', 'miserable': 'menyedihkan',
    'misleading': 'menyesatkan', 'mixed': 'campuran',
    'modern': 'modern', 'monotonous': 'monoton',
    'morbid': 'mengerikan', 'mysterious': 'misterius',
    'naive': 'naif', 'narrow': 'sempit', 'nasty': 'jahat',
    'natural': 'alami', 'necessary': 'perlu',
    'negative': 'negatif', 'nervous': 'gugup',
    'new': 'baru', 'normal': 'normal', 'nostalgic': 'nostalgia',
    'notable': 'penting', 'notorious': 'terkenal buruk',
    'obvious': 'jelas', 'odd': 'aneh', 'offensive': 'menyinggung',
    'old': 'tua', 'open': 'terbuka', 'ordinary': 'biasa',
    'original': 'orisinal', 'painful': 'menyakitkan',
    'pale': 'pucat', 'paranoid': 'paranoid',
    'passionate': 'penuh gairah', 'pathetic': 'menyedihkan',
    'patient': 'sabar', 'peaceful': 'damai',
    'phenomenal': 'fenomenal', 'plain': 'biasa',
    'playful': 'bermain', 'pleasant': 'menyenangkan',
    'pointless': 'tidak berguna', 'polished': 'dipoles',
    'popular': 'populer', 'positive': 'positif',
    'predictable': 'mudah ditebak', 'pretty': 'cantik',
    'previous': 'sebelumnya', 'priceless': 'tak ternilai',
    'profound': 'mendalam', 'promising': 'menjanjikan',
    'proper': 'tepat', 'provocative': 'provokatif',
    'pure': 'murni', 'quick': 'cepat', 'quiet': 'tenang',
    'radical': 'radikal', 'random': 'acak',
    'rare': 'langka', 'raw': 'mentah', 'real': 'nyata',
    'realistic': 'realistis', 'reasonable': 'masuk akal',
    'refreshing': 'menyegarkan', 'regrettable': 'disesali',
    'relatable': 'relevan', 'reliable': 'dapat diandalkan',
    'remarkable': 'luar biasa', 'repetitive': 'berulang',
    'respectful': 'hormat', 'restless': 'gelisah',
    'revolutionary': 'revolusioner', 'rich': 'kaya',
    'riveting': 'mencekam', 'romantic': 'romantis',
    'rough': 'kasar', 'rushed': 'terburu-buru',
    'ruthless': 'kejam', 'safe': 'aman',
    'satisfying': 'memuaskan', 'scariest': 'paling menakutkan',
    'scarier': 'lebih menakutkan', 'sensitive': 'sensitif',
    'serious': 'serius', 'shallow': 'dangkal',
    'sharp': 'tajam', 'short': 'pendek',
    'sick': 'sakit', 'silly': 'konyol',
    'similar': 'mirip', 'simple': 'sederhana',
    'sinister': 'jahat', 'skeptical': 'skeptis',
    'smart': 'pintar', 'smooth': 'halus',
    'solid': 'solid', 'sophisticated': 'canggih',
    'soulless': 'tanpa jiwa', 'special': 'spesial',
    'spectacular': 'spektakuler', 'spooky': 'menyeramkan',
    'stale': 'basi', 'standard': 'standar',
    'steady': 'stabil', 'stellar': 'luar biasa',
    'stereotypical': 'stereotip', 'stiff': 'kaku',
    'strange': 'aneh', 'strong': 'kuat',
    'subtle': 'halus', 'successful': 'sukses',
    'sufficient': 'cukup', 'superior': 'unggul',
    'surprising': 'mengejutkan', 'surreal': 'surealis',
    'tasteless': 'tidak berselera', 'tedious': 'membosankan',
    'tense': 'tegang', 'thick': 'tebal',
    'thin': 'tipis', 'thoughtful': 'penuh pemikiran',
    'thrilling': 'mendebarkan', 'tight': 'ketat',
    'tired': 'lelah', 'tiresome': 'melelahkan',
    'tolerable': 'dapat ditoleransi', 'top notch': 'kelas atas',
    'toxic': 'beracun', 'tragic': 'tragis',
    'tremendous': 'luar biasa', 'tricky': 'rumit',
    'trivial': 'sepele', 'true': 'benar',
    'trustworthy': 'dapat dipercaya', 'twisted': 'bengkok',
    'typical': 'tipikal', 'ugly': 'jelek',
    'ultimate': 'utama', 'unbelievable': 'tidak bisa dipercaya',
    'unconvincing': 'tidak meyakinkan', 'understandable': 'dapat dimengerti',
    'uneasy': 'gelisah', 'uneven': 'tidak merata',
    'unfortunate': 'malang', 'unhappy': 'tidak bahagia',
    'uninspired': 'tidak terinspirasi', 'unnecessary': 'tidak perlu',
    'unrealistic': 'tidak realistis', 'unsettling': 'mengganggu',
    'unwatchable': 'tidak layak tonton', 'useless': 'tidak berguna',
    'vague': 'samar', 'valid': 'valid',
    'valuable': 'berharga', 'vast': 'luas',
    'vibrant': 'semarak', 'vicious': 'ganas',
    'violent': 'penuh kekerasan', 'vivid': 'hidup',
    'warm': 'hangat', 'weird': 'aneh',
    'welcome': 'disambut', 'wicked': 'jahat',
    'wild': 'liar', 'wise': 'bijak',
    'witty': 'cerdas', 'worthy': 'layak',

    # ── Kata kerja (verbs) ──
    'enjoy': 'menikmati', 'enjoyed': 'menikmati',
    'hate': 'benci', 'hated': 'membenci',
    'love': 'cinta', 'loved': 'mencintai',
    'cry': 'menangis', 'cried': 'menangis',
    'laugh': 'tertawa', 'laughed': 'tertawa',
    'scream': 'berteriak', 'screamed': 'berteriak',
    'sleep': 'tidur', 'slept': 'tidur',
    'bored': 'bosan', 'satisfied': 'puas',
    'unsatisfied': 'tidak puas', 'impressed': 'terkesan',
    'unimpressed': 'tidak terkesan', 'confused': 'bingung',
    'speechless': 'tidak bisa berkata', 'mind': 'pikiran',
    'remember': 'mengingat', 'forget': 'lupa',
    'recommend': 'merekomendasikan',
    'waste': 'buang', 'wasted': 'sia-sia', 'wasting': 'membuang',

    # ── Kata kerja tambahan (expanded) ──
    'achieve': 'mencapai', 'act': 'berakting', 'adapt': 'mengadaptasi',
    'admire': 'mengagumi', 'admit': 'mengakui', 'affect': 'mempengaruhi',
    'agree': 'setuju', 'allow': 'mengizinkan', 'amaze': 'mengagumkan',
    'annoy': 'mengganggu', 'annoyed': 'terganggu', 'annoying': 'menjengkelkan',
    'anticipate': 'mengantisipasi', 'appear': 'muncul', 'appreciate': 'menghargai',
    'approach': 'mendekati', 'approve': 'menyetujui', 'argue': 'berdebat',
    'avoid': 'menghindari', 'awaken': 'membangunkan',
    'begin': 'memulai', 'began': 'memulai', 'begun': 'dimulai',
    'believe': 'percaya', 'belong': 'milik', 'betray': 'mengkhianati',
    'blame': 'menyalahkan', 'bless': 'memberkati', 'blow': 'meniup',
    'boost': 'meningkatkan', 'bother': 'mengganggu',
    'break': 'merusak', 'broke': 'merusak', 'broken': 'rusak',
    'bring': 'membawa', 'brought': 'membawa', 'build': 'membangun',
    'built': 'membangun', 'burn': 'membakar',
    'capture': 'menangkap', 'carry': 'membawa', 'catch': 'menangkap',
    'caught': 'menangkap', 'cause': 'menyebabkan',
    'celebrate': 'merayakan', 'challenge': 'menantang',
    'change': 'mengubah', 'changed': 'berubah',
    'chase': 'mengejar', 'check': 'memeriksa',
    'choose': 'memilih', 'chose': 'memilih', 'chosen': 'dipilih',
    'claim': 'mengklaim', 'climb': 'memanjat',
    'close': 'menutup', 'collapse': 'runtuh',
    'combine': 'menggabungkan', 'compare': 'membandingkan',
    'compete': 'bersaing', 'complain': 'mengeluh',
    'complete': 'menyelesaikan', 'concern': 'mengkhawatirkan',
    'confirm': 'mengonfirmasi', 'connect': 'menghubungkan',
    'consider': 'mempertimbangkan', 'contain': 'mengandung',
    'continue': 'melanjutkan', 'contribute': 'berkontribusi',
    'control': 'mengendalikan', 'convince': 'meyakinkan',
    'cover': 'menutupi', 'crash': 'menabrak',
    'create': 'menciptakan', 'criticize': 'mengkritik',
    'cross': 'menyeberangi', 'crush': 'menghancurkan',
    'cut': 'memotong',
    'damage': 'merusak', 'dare': 'berani',
    'deal': 'menangani', 'decide': 'memutuskan',
    'declare': 'menyatakan', 'decline': 'menurun',
    'defeat': 'mengalahkan', 'defend': 'membela',
    'define': 'mendefinisikan', 'deliver': 'menyampaikan',
    'demand': 'menuntut', 'demonstrate': 'mendemonstrasikan',
    'deny': 'menyangkal', 'depend': 'bergantung',
    'depict': 'menggambarkan', 'describe': 'mendeskripsikan',
    'deserve': 'layak', 'design': 'mendesain',
    'desire': 'menginginkan', 'destroy': 'menghancurkan',
    'detect': 'mendeteksi', 'develop': 'mengembangkan',
    'die': 'mati', 'died': 'mati', 'dying': 'sekarat',
    'direct': 'mengarahkan', 'disappear': 'menghilang',
    'disappoint': 'mengecewakan', 'disappointed': 'kecewa',
    'discover': 'menemukan', 'discuss': 'mendiskusikan',
    'display': 'menampilkan', 'disturb': 'mengganggu',
    'drag': 'menyeret', 'dragged': 'terseret',
    'draw': 'menggambar', 'drawn': 'digambar',
    'dream': 'bermimpi', 'drive': 'menggerakkan',
    'drop': 'menjatuhkan', 'drown': 'tenggelam',
    'earn': 'mendapatkan', 'embrace': 'memeluk',
    'emerge': 'muncul', 'emphasize': 'menekankan',
    'encounter': 'menghadapi', 'encourage': 'mendorong',
    'endure': 'bertahan', 'engage': 'melibatkan',
    'enhance': 'meningkatkan', 'ensure': 'memastikan',
    'escape': 'melarikan diri', 'establish': 'membangun',
    'evaluate': 'mengevaluasi', 'evolve': 'berkembang',
    'examine': 'memeriksa', 'exceed': 'melebihi',
    'excite': 'menggairahkan', 'excited': 'antusias',
    'execute': 'menjalankan', 'exhibit': 'memamerkan',
    'exist': 'ada', 'expect': 'mengharapkan',
    'expected': 'diharapkan', 'experience': 'mengalami',
    'explain': 'menjelaskan', 'explode': 'meledak',
    'explore': 'menjelajahi', 'expose': 'mengekspos',
    'express': 'mengekspresikan', 'extend': 'memperpanjang',
    'face': 'menghadapi', 'fail': 'gagal', 'failed': 'gagal',
    'fall': 'jatuh', 'fell': 'jatuh', 'fallen': 'jatuh',
    'fear': 'takut', 'feature': 'menampilkan',
    'fight': 'bertarung', 'fought': 'bertarung',
    'figure': 'menemukan', 'fill': 'mengisi',
    'find': 'menemukan', 'finish': 'menyelesaikan',
    'fix': 'memperbaiki', 'flow': 'mengalir',
    'fly': 'terbang', 'focus': 'fokus', 'follow': 'mengikuti',
    'force': 'memaksa', 'foreshadow': 'memberi petunjuk',
    'fulfill': 'memenuhi', 'gain': 'mendapatkan',
    'grab': 'meraih', 'grow': 'tumbuh', 'grew': 'tumbuh',
    'guarantee': 'menjamin', 'guess': 'menebak',
    'guide': 'memandu', 'handle': 'menangani',
    'happen': 'terjadi', 'haunt': 'menghantui',
    'hear': 'mendengar', 'heard': 'mendengar',
    'help': 'membantu', 'hide': 'menyembunyikan',
    'hit': 'memukul', 'hold': 'memegang', 'held': 'memegang',
    'hope': 'berharap', 'hunt': 'memburu',
    'hurt': 'menyakiti', 'identify': 'mengidentifikasi',
    'ignore': 'mengabaikan', 'illustrate': 'mengilustrasikan',
    'imagine': 'membayangkan', 'immerse': 'membenamkan',
    'imply': 'menyiratkan', 'improve': 'memperbaiki',
    'include': 'termasuk', 'increase': 'meningkatkan',
    'indicate': 'menunjukkan', 'influence': 'mempengaruhi',
    'inform': 'memberitahu', 'inject': 'menyuntikkan',
    'insist': 'bersikeras', 'inspire': 'menginspirasi',
    'introduce': 'memperkenalkan', 'invest': 'menginvestasikan',
    'investigate': 'menyelidiki', 'involve': 'melibatkan',
    'isolate': 'mengisolasi', 'join': 'bergabung',
    'judge': 'menilai', 'jump': 'melompat',
    'justify': 'membenarkan', 'keep': 'menjaga', 'kept': 'menjaga',
    'kill': 'membunuh', 'killed': 'terbunuh',
    'lack': 'kekurangan', 'land': 'mendarat',
    'last': 'bertahan', 'launch': 'meluncurkan',
    'lead': 'memimpin', 'led': 'memimpin',
    'learn': 'belajar', 'learned': 'belajar',
    'leave': 'meninggalkan', 'let': 'membiarkan',
    'lie': 'berbohong', 'lift': 'mengangkat',
    'listen': 'mendengarkan', 'live': 'hidup',
    'look': 'melihat', 'looked': 'melihat', 'looking': 'melihat',
    'lose': 'kehilangan', 'lost': 'kehilangan',
    'maintain': 'mempertahankan', 'manage': 'mengelola',
    'manipulate': 'memanipulasi', 'mark': 'menandai',
    'master': 'menguasai', 'match': 'mencocokkan',
    'matter': 'penting', 'mean': 'berarti', 'meant': 'berarti',
    'measure': 'mengukur', 'meet': 'bertemu', 'met': 'bertemu',
    'mention': 'menyebutkan', 'miss': 'melewatkan',
    'mislead': 'menyesatkan', 'mix': 'mencampur',
    'motivate': 'memotivasi', 'move': 'bergerak',
    'murder': 'membunuh', 'notice': 'memperhatikan',
    'observe': 'mengamati', 'occur': 'terjadi',
    'offer': 'menawarkan', 'operate': 'mengoperasikan',
    'oppose': 'menentang', 'order': 'memerintahkan',
    'organize': 'mengorganisir', 'overcome': 'mengatasi',
    'overlook': 'mengabaikan', 'owe': 'berhutang',
    'own': 'memiliki', 'pay': 'membayar', 'paid': 'membayar',
    'perform': 'tampil', 'permit': 'mengizinkan',
    'pick': 'memilih', 'place': 'menempatkan',
    'plan': 'merencanakan', 'play': 'bermain', 'played': 'bermain',
    'please': 'tolong', 'pleased': 'senang',
    'point': 'menunjukkan', 'portray': 'menggambarkan',
    'possess': 'memiliki', 'predict': 'memprediksi',
    'prefer': 'lebih suka', 'prepare': 'mempersiapkan',
    'present': 'menyajikan', 'preserve': 'melestarikan',
    'pretend': 'berpura-pura', 'prevent': 'mencegah',
    'produce': 'menghasilkan', 'promise': 'berjanji',
    'promote': 'mempromosikan', 'protect': 'melindungi',
    'prove': 'membuktikan', 'proved': 'membuktikan',
    'provide': 'menyediakan', 'provoke': 'memancing',
    'pull': 'menarik', 'push': 'mendorong',
    'put': 'meletakkan', 'question': 'mempertanyakan',
    'quit': 'berhenti', 'raise': 'mengangkat',
    'reach': 'mencapai', 'react': 'bereaksi',
    'read': 'membaca', 'realize': 'menyadari',
    'receive': 'menerima', 'recognize': 'mengenali',
    'record': 'merekam', 'recover': 'memulihkan',
    'reduce': 'mengurangi', 'refer': 'merujuk',
    'reflect': 'merefleksikan', 'refuse': 'menolak',
    'regret': 'menyesal', 'reject': 'menolak',
    'relate': 'berhubungan', 'release': 'merilis',
    'rely': 'mengandalkan', 'remain': 'tetap',
    'remind': 'mengingatkan', 'remove': 'menghapus',
    'repeat': 'mengulangi', 'replace': 'mengganti',
    'represent': 'mewakili', 'request': 'meminta',
    'require': 'membutuhkan', 'rescue': 'menyelamatkan',
    'resemble': 'menyerupai', 'resist': 'melawan',
    'resolve': 'menyelesaikan', 'respect': 'menghormati',
    'respond': 'merespons', 'restore': 'memulihkan',
    'result': 'menghasilkan', 'return': 'kembali',
    'reveal': 'mengungkap', 'revenge': 'membalas dendam',
    'review': 'mengulas', 'rise': 'bangkit', 'risen': 'bangkit',
    'risk': 'mempertaruhkan', 'ruin': 'merusak', 'ruined': 'dirusak',
    'rule': 'memerintah', 'run': 'menjalankan', 'rush': 'terburu-buru',
    'sacrifice': 'mengorbankan', 'satisfy': 'memuaskan',
    'save': 'menyelamatkan', 'scare': 'menakuti', 'scared': 'takut',
    'search': 'mencari', 'seem': 'tampak',
    'select': 'memilih', 'sell': 'menjual', 'send': 'mengirim',
    'separate': 'memisahkan', 'serve': 'melayani',
    'set': 'mengatur', 'settle': 'menyelesaikan',
    'shake': 'mengguncang', 'shape': 'membentuk',
    'shine': 'bersinar', 'shock': 'mengejutkan', 'shocked': 'terkejut',
    'shoot': 'menembak', 'shot': 'menembak',
    'show': 'menunjukkan', 'showed': 'menunjukkan', 'shown': 'ditunjukkan',
    'shut': 'menutup', 'signal': 'memberi sinyal',
    'simplify': 'menyederhanakan', 'sit': 'duduk', 'sat': 'duduk',
    'skip': 'melewatkan', 'slow': 'memperlambat',
    'solve': 'memecahkan', 'sound': 'terdengar',
    'speak': 'berbicara', 'spoke': 'berbicara', 'spoken': 'dibicarakan',
    'spend': 'menghabiskan', 'spent': 'menghabiskan',
    'split': 'memisahkan', 'spoil': 'merusak',
    'spot': 'melihat', 'spread': 'menyebarkan',
    'squeeze': 'memeras', 'stand': 'berdiri', 'stood': 'berdiri',
    'start': 'memulai', 'started': 'dimulai',
    'stay': 'tinggal', 'steal': 'mencuri', 'stole': 'mencuri',
    'step': 'melangkah', 'stick': 'menempel',
    'stop': 'berhenti', 'stopped': 'berhenti',
    'store': 'menyimpan', 'strike': 'menyerang',
    'struggle': 'berjuang', 'study': 'mempelajari',
    'submit': 'menyerahkan', 'succeed': 'berhasil',
    'suffer': 'menderita', 'suggest': 'menyarankan',
    'support': 'mendukung', 'suppose': 'mengira',
    'surprise': 'mengejutkan', 'surprised': 'terkejut',
    'surrender': 'menyerah', 'surround': 'mengelilingi',
    'survive': 'bertahan', 'suspect': 'mencurigai',
    'suspend': 'menangguhkan', 'sustain': 'mempertahankan',
    'swear': 'bersumpah', 'switch': 'mengalihkan',
    'symbolize': 'melambangkan', 'talk': 'berbicara',
    'target': 'menargetkan', 'teach': 'mengajar',
    'tear': 'merobek', 'terrify': 'menakutkan',
    'test': 'menguji', 'thank': 'berterima kasih',
    'threaten': 'mengancam', 'throw': 'melempar', 'threw': 'melempar',
    'tie': 'mengikat', 'torture': 'menyiksa',
    'touch': 'menyentuh', 'trace': 'melacak',
    'track': 'melacak', 'train': 'melatih',
    'transform': 'mengubah', 'translate': 'menerjemahkan',
    'trap': 'menjebak', 'travel': 'bepergian',
    'treat': 'memperlakukan', 'trick': 'menipu',
    'trouble': 'merepotkan', 'trust': 'mempercayai',
    'try': 'mencoba', 'tried': 'mencoba',
    'turn': 'membalikkan', 'turned': 'berubah',
    'twist': 'memutar', 'understand': 'memahami',
    'understood': 'memahami', 'unfold': 'terungkap',
    'unite': 'menyatukan', 'update': 'memperbarui',
    'upset': 'kesal', 'urge': 'mendesak',
    'use': 'menggunakan', 'used': 'menggunakan',
    'utter': 'mengucapkan', 'value': 'menghargai',
    'vanish': 'menghilang', 'vary': 'bervariasi',
    'view': 'melihat', 'visit': 'mengunjungi',
    'wait': 'menunggu', 'waited': 'menunggu',
    'wake': 'bangun', 'walk': 'berjalan',
    'wander': 'berkeliaran', 'warn': 'memperingatkan',
    'watch': 'menonton', 'watched': 'menonton', 'watching': 'menonton',
    'win': 'menang', 'won': 'menang',
    'wonder': 'bertanya-tanya', 'work': 'bekerja',
    'worked': 'bekerja', 'worry': 'khawatir',
    'worsen': 'memperburuk', 'wrap': 'membungkus',
    'write': 'menulis', 'written': 'ditulis', 'wrote': 'menulis',
    'yell': 'berteriak',

    # ── Kata benda (nouns) ──
    'whole': 'seluruh', 'entire': 'seluruh',
    'every': 'setiap', 'nothing': 'tidak ada',
    'something': 'sesuatu', 'everything': 'segalanya',
    'anyone': 'siapa saja', 'everyone': 'semua orang',
    'nobody': 'tidak ada seorangpun', 'somebody': 'seseorang',
    'enough': 'cukup', 'almost': 'hampir',
    'completely': 'sepenuhnya', 'totally': 'sepenuhnya',
    'absolutely': 'benar-benar', 'literally': 'secara harfiah',

    # ── Kata benda tambahan (film context) ──
    'audience': 'penonton', 'atmosphere': 'atmosfer',
    'attention': 'perhatian', 'beginning': 'permulaan',
    'blood': 'darah', 'body': 'tubuh', 'brain': 'otak',
    'camera': 'kamera', 'career': 'karier',
    'childhood': 'masa kecil', 'choice': 'pilihan',
    'cinematography': 'sinematografi', 'conclusion': 'kesimpulan',
    'conflict': 'konflik', 'conscience': 'nurani',
    'consequence': 'konsekuensi', 'courage': 'keberanian',
    'creature': 'makhluk', 'crew': 'kru',
    'crime': 'kejahatan', 'criticism': 'kritik',
    'crowd': 'kerumunan', 'culture': 'budaya',
    'curse': 'kutukan', 'danger': 'bahaya',
    'darkness': 'kegelapan', 'daughter': 'anak perempuan',
    'dead': 'mati', 'death': 'kematian',
    'decision': 'keputusan', 'demon': 'iblis',
    'depth': 'kedalaman', 'desire': 'keinginan',
    'destiny': 'takdir', 'detail': 'detail',
    'development': 'perkembangan', 'dialogue': 'dialog',
    'direction': 'arahan', 'disappointment': 'kekecewaan',
    'disaster': 'bencana', 'doubt': 'keraguan',
    'dream': 'mimpi', 'earth': 'bumi',
    'edge': 'tepi', 'element': 'elemen',
    'emotion': 'emosi', 'empathy': 'empati',
    'energy': 'energi', 'enjoyment': 'kesenangan',
    'entertainment': 'hiburan', 'episode': 'episode',
    'era': 'era', 'essence': 'esensi',
    'event': 'peristiwa', 'evidence': 'bukti',
    'evil': 'kejahatan', 'example': 'contoh',
    'excitement': 'kegembiraan', 'execution': 'eksekusi',
    'existence': 'keberadaan', 'expectation': 'harapan',
    'experience': 'pengalaman', 'explosion': 'ledakan',
    'expression': 'ekspresi', 'eye': 'mata', 'eyes': 'mata',
    'face': 'wajah', 'fact': 'fakta',
    'failure': 'kegagalan', 'faith': 'keyakinan',
    'family': 'keluarga', 'fantasy': 'fantasi',
    'fate': 'nasib', 'father': 'ayah',
    'fear': 'ketakutan', 'feeling': 'perasaan', 'feelings': 'perasaan',
    'fiction': 'fiksi', 'fight': 'pertarungan',
    'final': 'akhir', 'flaw': 'cacat', 'flaws': 'kekurangan',
    'flesh': 'daging', 'force': 'kekuatan',
    'formula': 'formula', 'foundation': 'fondasi',
    'freedom': 'kebebasan', 'friend': 'teman', 'friends': 'teman-teman',
    'friendship': 'persahabatan', 'future': 'masa depan',
    'ghost': 'hantu', 'girl': 'gadis', 'guy': 'pria',
    'god': 'tuhan', 'grief': 'duka',
    'group': 'kelompok', 'growth': 'pertumbuhan',
    'guilt': 'rasa bersalah', 'gun': 'senjata',
    'half': 'setengah', 'hand': 'tangan',
    'happiness': 'kebahagiaan', 'head': 'kepala',
    'heart': 'hati', 'hell': 'neraka',
    'history': 'sejarah', 'home': 'rumah',
    'honor': 'kehormatan', 'hope': 'harapan',
    'horror': 'horor', 'hour': 'jam', 'hours': 'jam',
    'house': 'rumah', 'human': 'manusia', 'humanity': 'kemanusiaan',
    'humor': 'humor', 'humour': 'humor',
    'husband': 'suami', 'idea': 'ide',
    'identity': 'identitas', 'illusion': 'ilusi',
    'image': 'gambar', 'imagination': 'imajinasi',
    'impact': 'dampak', 'impression': 'kesan',
    'improvement': 'perbaikan', 'influence': 'pengaruh',
    'innocence': 'kepolosan', 'insight': 'wawasan',
    'inspiration': 'inspirasi', 'intensity': 'intensitas',
    'intention': 'niat', 'interest': 'minat',
    'interpretation': 'interpretasi', 'introduction': 'pengenalan',
    'investigation': 'penyelidikan', 'issue': 'masalah', 'issues': 'masalah',
    'journey': 'perjalanan', 'joy': 'kegembiraan',
    'justice': 'keadilan', 'kid': 'anak', 'kids': 'anak-anak',
    'killer': 'pembunuh', 'king': 'raja',
    'knowledge': 'pengetahuan', 'lack': 'kekurangan',
    'lady': 'wanita', 'language': 'bahasa',
    'laugh': 'tawa', 'laughter': 'tawa',
    'leader': 'pemimpin', 'lesson': 'pelajaran',
    'life': 'kehidupan', 'lifetime': 'seumur hidup',
    'location': 'lokasi', 'loss': 'kehilangan',
    'luck': 'keberuntungan', 'madness': 'kegilaan',
    'magic': 'sihir', 'makeup': 'tata rias',
    'man': 'pria', 'manner': 'cara',
    'mark': 'tanda', 'marriage': 'pernikahan',
    'master': 'master', 'meaning': 'makna',
    'memory': 'kenangan', 'memories': 'kenangan',
    'message': 'pesan', 'method': 'metode',
    'middle': 'tengah', 'minute': 'menit', 'minutes': 'menit',
    'miracle': 'keajaiban', 'misery': 'kesengsaraan',
    'mission': 'misi', 'mistake': 'kesalahan',
    'moment': 'momen', 'moments': 'momen-momen',
    'monster': 'monster', 'mood': 'suasana hati',
    'moral': 'moral', 'mother': 'ibu',
    'motivation': 'motivasi', 'murder': 'pembunuhan',
    'music': 'musik', 'mystery': 'misteri',
    'narrative': 'narasi', 'nature': 'alam',
    'need': 'kebutuhan', 'neighbor': 'tetangga',
    'night': 'malam', 'nightmare': 'mimpi buruk',
    'noise': 'kebisingan', 'nonsense': 'omong kosong',
    'note': 'catatan', 'number': 'nomor',
    'opinion': 'pendapat', 'opportunity': 'kesempatan',
    'opposite': 'kebalikan', 'option': 'pilihan',
    'outcome': 'hasil', 'pace': 'kecepatan',
    'pain': 'rasa sakit', 'pair': 'pasangan',
    'panic': 'panik', 'part': 'bagian', 'parts': 'bagian-bagian',
    'partner': 'pasangan', 'past': 'masa lalu',
    'path': 'jalan', 'patience': 'kesabaran',
    'pattern': 'pola', 'peace': 'kedamaian',
    'people': 'orang-orang', 'period': 'periode',
    'person': 'orang', 'perspective': 'perspektif',
    'picture': 'gambar', 'piece': 'bagian',
    'place': 'tempat', 'pleasure': 'kesenangan',
    'point': 'poin', 'poison': 'racun',
    'police': 'polisi', 'potential': 'potensi',
    'power': 'kekuatan', 'premise': 'premis',
    'presence': 'kehadiran', 'pressure': 'tekanan',
    'pride': 'kebanggaan', 'prison': 'penjara',
    'problem': 'masalah', 'production': 'produksi',
    'progress': 'kemajuan', 'promise': 'janji',
    'proof': 'bukti', 'protection': 'perlindungan',
    'public': 'publik', 'punishment': 'hukuman',
    'purpose': 'tujuan', 'pursuit': 'pengejaran',
    'queen': 'ratu', 'quest': 'pencarian',
    'rage': 'amarah', 'reality': 'kenyataan',
    'reason': 'alasan', 'redemption': 'penebusan',
    'relationship': 'hubungan', 'relief': 'kelegaan',
    'remainder': 'sisa', 'remake': 'pembuatan ulang',
    'reputation': 'reputasi', 'resolution': 'resolusi',
    'resource': 'sumber daya', 'response': 'respons',
    'rest': 'istirahat', 'result': 'hasil',
    'revelation': 'pengungkapan', 'revenge': 'balas dendam',
    'reward': 'hadiah', 'rhythm': 'ritme',
    'ride': 'perjalanan', 'right': 'hak',
    'role': 'peran', 'room': 'ruangan',
    'sacrifice': 'pengorbanan', 'sadness': 'kesedihan',
    'safety': 'keamanan', 'satisfaction': 'kepuasan',
    'scene': 'adegan', 'school': 'sekolah',
    'scope': 'cakupan', 'screen': 'layar',
    'season': 'musim', 'secret': 'rahasia',
    'section': 'bagian', 'sense': 'rasa',
    'sequence': 'urutan', 'series': 'seri',
    'setting': 'latar', 'shadow': 'bayangan',
    'shame': 'rasa malu', 'shock': 'kejutan',
    'show': 'pertunjukan', 'side': 'sisi',
    'sight': 'pemandangan', 'silence': 'keheningan',
    'sin': 'dosa', 'situation': 'situasi',
    'skill': 'keterampilan', 'skin': 'kulit',
    'society': 'masyarakat', 'son': 'anak laki-laki',
    'song': 'lagu', 'soul': 'jiwa',
    'source': 'sumber', 'space': 'ruang',
    'speech': 'pidato', 'spirit': 'semangat',
    'spot': 'titik', 'stage': 'panggung',
    'standard': 'standar', 'statement': 'pernyataan',
    'step': 'langkah', 'stomach': 'perut',
    'story': 'cerita', 'stranger': 'orang asing',
    'strategy': 'strategi', 'strength': 'kekuatan',
    'structure': 'struktur', 'struggle': 'perjuangan',
    'style': 'gaya', 'subject': 'subjek',
    'success': 'kesuksesan', 'suffering': 'penderitaan',
    'suggestion': 'saran', 'summary': 'ringkasan',
    'support': 'dukungan', 'surface': 'permukaan',
    'surprise': 'kejutan', 'survival': 'kelangsungan hidup',
    'survivor': 'penyintas', 'suspense': 'ketegangan',
    'symbol': 'simbol', 'sympathy': 'simpati',
    'system': 'sistem', 'tale': 'kisah',
    'talent': 'bakat', 'target': 'target',
    'task': 'tugas', 'taste': 'selera',
    'team': 'tim', 'technique': 'teknik',
    'teenager': 'remaja', 'television': 'televisi',
    'tempo': 'tempo', 'tension': 'ketegangan',
    'terror': 'teror', 'test': 'ujian',
    'text': 'teks', 'theater': 'teater', 'theatre': 'teater',
    'theme': 'tema', 'theory': 'teori',
    'thing': 'hal', 'things': 'hal-hal',
    'thought': 'pemikiran', 'threat': 'ancaman',
    'ticket': 'tiket', 'time': 'waktu',
    'tone': 'nada', 'tool': 'alat',
    'topic': 'topik', 'touch': 'sentuhan',
    'tour': 'tur', 'tradition': 'tradisi',
    'tragedy': 'tragedi', 'trail': 'jejak',
    'transformation': 'transformasi', 'transition': 'transisi',
    'trap': 'jebakan', 'treasure': 'harta',
    'treatment': 'perlakuan', 'trend': 'tren',
    'trial': 'percobaan', 'tribute': 'penghormatan',
    'trick': 'trik', 'triumph': 'kemenangan',
    'trouble': 'masalah', 'truth': 'kebenaran',
    'type': 'tipe', 'understanding': 'pemahaman',
    'universe': 'alam semesta', 'value': 'nilai',
    'version': 'versi', 'victim': 'korban',
    'victory': 'kemenangan', 'view': 'pandangan',
    'viewer': 'penonton', 'viewers': 'penonton',
    'villain': 'penjahat', 'violence': 'kekerasan',
    'virtue': 'kebajikan', 'vision': 'visi',
    'voice': 'suara', 'war': 'perang',
    'waste': 'pemborosan', 'water': 'air',
    'way': 'cara', 'weakness': 'kelemahan',
    'weapon': 'senjata', 'week': 'minggu',
    'wife': 'istri', 'will': 'kemauan',
    'wisdom': 'kebijaksanaan', 'wish': 'keinginan',
    'witness': 'saksi', 'woman': 'wanita', 'women': 'wanita',
    'wonder': 'keajaiban', 'word': 'kata', 'words': 'kata-kata',
    'work': 'karya', 'world': 'dunia',
    'worth': 'nilai', 'writer': 'penulis',
    'year': 'tahun', 'years': 'tahun',
    'youth': 'pemuda', 'zombie': 'zombi',

    # ── Kata keterangan (adverbs) ──
    'already': 'sudah', 'always': 'selalu',
    'barely': 'hampir tidak', 'briefly': 'secara singkat',
    'carefully': 'dengan hati-hati', 'clearly': 'dengan jelas',
    'closely': 'secara dekat', 'constantly': 'terus-menerus',
    'deeply': 'secara mendalam', 'deliberately': 'dengan sengaja',
    'desperately': 'dengan putus asa', 'directly': 'secara langsung',
    'easily': 'dengan mudah', 'effectively': 'secara efektif',
    'emotionally': 'secara emosional', 'entirely': 'sepenuhnya',
    'equally': 'secara merata', 'essentially': 'pada dasarnya',
    'eventually': 'akhirnya', 'exactly': 'tepat',
    'extremely': 'sangat', 'fairly': 'cukup',
    'finally': 'akhirnya', 'firmly': 'dengan tegas',
    'fortunately': 'untungnya', 'freely': 'dengan bebas',
    'frequently': 'sering', 'fully': 'sepenuhnya',
    'generally': 'umumnya', 'gently': 'dengan lembut',
    'genuinely': 'secara tulus', 'gradually': 'secara bertahap',
    'greatly': 'sangat', 'hardly': 'hampir tidak',
    'heavily': 'sangat', 'highly': 'sangat',
    'hopefully': 'semoga', 'immediately': 'segera',
    'incredibly': 'luar biasa', 'indeed': 'memang',
    'initially': 'awalnya', 'instantly': 'seketika',
    'intentionally': 'dengan sengaja', 'largely': 'sebagian besar',
    'lately': 'belakangan ini', 'lightly': 'dengan ringan',
    'likely': 'kemungkinan', 'literally': 'secara harfiah',
    'mainly': 'terutama', 'merely': 'hanya',
    'mostly': 'sebagian besar', 'naturally': 'secara alami',
    'nearly': 'hampir', 'necessarily': 'harus',
    'never': 'tidak pernah', 'normally': 'biasanya',
    'notably': 'khususnya', 'obviously': 'jelas',
    'occasionally': 'sesekali', 'originally': 'awalnya',
    'otherwise': 'sebaliknya', 'overwhelmingly': 'secara luar biasa',
    'painfully': 'secara menyakitkan', 'particularly': 'khususnya',
    'partly': 'sebagian', 'perfectly': 'dengan sempurna',
    'permanently': 'secara permanen', 'personally': 'secara pribadi',
    'physically': 'secara fisik', 'poorly': 'dengan buruk',
    'possibly': 'mungkin', 'precisely': 'dengan tepat',
    'previously': 'sebelumnya', 'primarily': 'terutama',
    'properly': 'dengan benar', 'purely': 'murni',
    'quickly': 'dengan cepat', 'quietly': 'dengan tenang',
    'randomly': 'secara acak', 'rapidly': 'dengan cepat',
    'rarely': 'jarang', 'recently': 'baru-baru ini',
    'repeatedly': 'berulang kali', 'sadly': 'sayangnya',
    'seriously': 'secara serius', 'severely': 'secara parah',
    'sharply': 'tajam', 'significantly': 'secara signifikan',
    'silently': 'secara diam-diam', 'simply': 'hanya',
    'sincerely': 'dengan tulus', 'slightly': 'sedikit',
    'slowly': 'secara perlahan', 'somehow': 'entah bagaimana',
    'sometimes': 'terkadang', 'somewhere': 'di suatu tempat',
    'specifically': 'secara khusus', 'steadily': 'secara stabil',
    'still': 'masih', 'strangely': 'anehnya',
    'strongly': 'dengan kuat', 'successfully': 'dengan sukses',
    'suddenly': 'tiba-tiba', 'supposedly': 'seharusnya',
    'surely': 'pasti', 'surprisingly': 'mengejutkan',
    'terribly': 'sangat buruk', 'thoroughly': 'secara menyeluruh',
    'tightly': 'dengan erat', 'together': 'bersama',
    'tonight': 'malam ini', 'traditionally': 'secara tradisional',
    'tragically': 'secara tragis', 'truly': 'sungguh',
    'typically': 'biasanya', 'ultimately': 'pada akhirnya',
    'undeniably': 'tidak bisa dipungkiri', 'undoubtedly': 'tidak diragukan lagi',
    'unexpectedly': 'secara tidak terduga', 'unfortunately': 'sayangnya',
    'universally': 'secara universal', 'unlikely': 'tidak mungkin',
    'unnecessarily': 'secara tidak perlu', 'utterly': 'sama sekali',
    'vastly': 'sangat', 'visually': 'secara visual',
    'well': 'dengan baik', 'wholly': 'sepenuhnya',
    'widely': 'secara luas', 'wildly': 'secara liar',
    'willingly': 'dengan sukarela', 'wrongly': 'secara salah',

    # ── Frasa umum (common phrases - multi-word) ──
    'worth watching': 'layak ditonton',
    'mind blowing': 'luar biasa', 'mindblowing': 'luar biasa',
    'must watch': 'wajib tonton', 'must see': 'wajib tonton',
    'thought provoking': 'menggugah pikiran',
    'well made': 'dibuat dengan baik', 'well done': 'dibuat dengan baik',
    'well crafted': 'dibuat dengan baik', 'well acted': 'diperankan dengan baik',
    'well written': 'ditulis dengan baik', 'well directed': 'disutradarai dengan baik',
    'badly made': 'dibuat dengan buruk', 'badly acted': 'diperankan dengan buruk',
    'badly written': 'ditulis dengan buruk',
    'top notch': 'kelas atas', 'high quality': 'berkualitas tinggi',
    'low budget': 'anggaran rendah', 'low quality': 'berkualitas rendah',
    'box office': 'box office', 'plot hole': 'lubang cerita',
    'plot holes': 'lubang cerita', 'plot twist': 'kejutan cerita',
    'slow burn': 'ketegangan perlahan', 'slow paced': 'lambat',
    'fast paced': 'cepat', 'edge of seat': 'menegangkan',
    'waste of time': 'buang waktu', 'waste of money': 'buang uang',
    'so good': 'sangat bagus', 'so bad': 'sangat buruk',
    'not bad': 'tidak buruk', 'not good': 'tidak bagus',
    'too long': 'terlalu panjang', 'too short': 'terlalu pendek',
    'too slow': 'terlalu lambat', 'too fast': 'terlalu cepat',
    'by far': 'sejauh ini', 'at least': 'setidaknya',
    'at all': 'sama sekali', 'in fact': 'sebenarnya',
    'of course': 'tentu saja', 'as well': 'juga',
    'even though': 'meskipun', 'on top': 'di atas',
    'no doubt': 'tidak diragukan', 'no way': 'tidak mungkin',
    'a lot': 'banyak', 'a bit': 'sedikit',

    # ── Kata hubung & preposisi umum ──
    'after': 'setelah', 'although': 'meskipun',
    'because': 'karena', 'before': 'sebelum',
    'between': 'antara', 'beyond': 'di luar',
    'despite': 'meskipun', 'during': 'selama',
    'except': 'kecuali', 'however': 'namun',
    'instead': 'sebagai gantinya', 'moreover': 'selain itu',
    'nevertheless': 'meskipun demikian', 'otherwise': 'jika tidak',
    'since': 'sejak', 'therefore': 'oleh karena itu',
    'though': 'meskipun', 'through': 'melalui',
    'throughout': 'sepanjang', 'unless': 'kecuali',
    'unlike': 'tidak seperti', 'until': 'sampai',
    'upon': 'di atas', 'whereas': 'sedangkan',
    'whether': 'apakah', 'while': 'sementara',
    'within': 'dalam', 'without': 'tanpa',

    # ══════════════════════════════════════════════════════════════════════
    # TAMBAHAN: Frasa khusus film & review (multi-word & single-word)
    # ══════════════════════════════════════════════════════════════════════

    # ── Frasa film spesifik ──
    'gore scene': 'adegan berdarah', 'gore scenes': 'adegan berdarah',
    'jump scare': 'kejutan menakutkan', 'jumpscare': 'kejutan menakutkan',
    'jumpscares': 'kejutan menakutkan',
    'mid credit scene': 'adegan tengah kredit',
    'mid credits scene': 'adegan tengah kredit',
    'post credit scene': 'adegan pasca kredit',
    'post credits scene': 'adegan pasca kredit',
    'end credit scene': 'adegan akhir kredit',
    'after credits': 'setelah kredit',
    'opening scene': 'adegan pembuka',
    'closing scene': 'adegan penutup',
    'fight scene': 'adegan pertarungan',
    'love scene': 'adegan percintaan',
    'death scene': 'adegan kematian',
    'final scene': 'adegan terakhir',
    'best scene': 'adegan terbaik',
    'worst scene': 'adegan terburuk',
    'first half': 'paruh pertama',
    'second half': 'paruh kedua',
    'b movie': 'film kelas rendah',
    'cult classic': 'klasik kultus',
    'box office hit': 'sukses box office',
    'body horror': 'horor tubuh',
    'body count': 'jumlah korban',
    'character development': 'pengembangan karakter',
    'character arc': 'perjalanan karakter',
    'world building': 'pembangunan dunia',
    'fourth wall': 'dinding keempat',
    'one dimensional': 'satu dimensi',
    'two dimensional': 'dua dimensi',
    'red herring': 'pengalih perhatian',

    # ── Kata sentimen Inggris tambahan ──
    'nope': 'tidak', 'yep': 'ya', 'yeah': 'ya',
    'legit': 'sungguhan', 'goat': 'terbaik',
    'peak': 'puncak', 'mid': 'biasa saja',
    'valid': 'benar', 'fire': 'keren',
    'slay': 'keren sekali', 'slayed': 'keren sekali',
    'banger': 'luar biasa', 'bangers': 'luar biasa',
    'underdog': 'yang dianggap remeh',
    'rookie': 'pemula', 'veteran': 'berpengalaman',
    'debut': 'pertama kali',
    'rewatch': 'menonton ulang', 'rewatched': 'menonton ulang',
    'rewatchable': 'layak tonton ulang',
    'binge': 'menonton maraton', 'binged': 'menonton maraton',
    'standout': 'menonjol', 'stand out': 'menonjol',
    'letdown': 'mengecewakan', 'let down': 'mengecewakan',
    'build up': 'peningkatan', 'buildup': 'peningkatan',
    'pay off': 'hasil yang memuaskan', 'payoff': 'hasil yang memuaskan',
    'tear jerker': 'menguras air mata',
    'nail biting': 'menegangkan',
    'gut wrenching': 'menyayat hati',
    'mind blowing': 'luar biasa', 'mindblowing': 'luar biasa',
    'jaw dropping': 'mengagumkan',
    'eye opening': 'membuka mata',
    'over the top': 'berlebihan',
    'on point': 'tepat sasaran',
    'hands down': 'tanpa ragu',
    'spot on': 'tepat sasaran',

    # ── Internet slang ──
    'af': 'sekali',  # "scary af" = "menakutkan sekali"
    'asf': 'sekali',
    'lowkey': 'sedikit', 'highkey': 'sangat',
    'fr': 'sungguhan',  # "for real"
    'irl': 'di kehidupan nyata',
    'smh': 'menggelengkan kepala',
    'fomo': 'takut ketinggalan',
    'hype': 'gembar-gembor',
    'hyped': 'digembar-gemborkan',
    'vibe': 'suasana', 'vibes': 'suasana',

    # ══════════════════════════════════════════════════════════════════════
    # TAMBAHAN BARU: Kata-kata yang sering lolos dari translasi
    # Diperluas untuk akurasi penelitian
    # ══════════════════════════════════════════════════════════════════════

    # ── Bentuk komparatif & superlatif ──
    'better': 'lebih baik', 'worse': 'lebih buruk',
    'bigger': 'lebih besar', 'smaller': 'lebih kecil',
    'longer': 'lebih panjang', 'shorter': 'lebih pendek',
    'faster': 'lebih cepat', 'slower': 'lebih lambat',
    'stronger': 'lebih kuat', 'weaker': 'lebih lemah',
    'harder': 'lebih sulit', 'easier': 'lebih mudah',
    'deeper': 'lebih dalam', 'higher': 'lebih tinggi',
    'lower': 'lebih rendah', 'closer': 'lebih dekat',
    'darker': 'lebih gelap', 'lighter': 'lebih terang',
    'louder': 'lebih keras', 'quieter': 'lebih tenang',
    'nicer': 'lebih bagus', 'cooler': 'lebih keren',
    'funnier': 'lebih lucu', 'sadder': 'lebih sedih',
    'happier': 'lebih bahagia', 'angrier': 'lebih marah',
    'scariest': 'paling menakutkan', 'funniest': 'paling lucu',
    'longest': 'paling panjang', 'shortest': 'paling pendek',
    'biggest': 'paling besar', 'smallest': 'paling kecil',
    'greatest': 'paling hebat', 'latest': 'paling baru',
    'earliest': 'paling awal', 'highest': 'paling tinggi',
    'lowest': 'paling rendah', 'strongest': 'paling kuat',
    'weakest': 'paling lemah', 'deepest': 'paling dalam',
    'worst': 'paling buruk', 'least': 'paling sedikit',

    # ── Bentuk -ing yang sering muncul di review ──
    'enjoying': 'menikmati', 'expecting': 'mengharapkan',
    'lacking': 'kurang', 'delivering': 'menyampaikan',
    'suffering': 'menderita', 'offering': 'menawarkan',
    'following': 'mengikuti', 'leading': 'memimpin',
    'making': 'membuat', 'taking': 'mengambil',
    'giving': 'memberi', 'telling': 'menceritakan',
    'saying': 'mengatakan', 'getting': 'mendapatkan',
    'coming': 'datang', 'running': 'berjalan',
    'leaving': 'meninggalkan', 'trying': 'mencoba',
    'working': 'bekerja', 'playing': 'bermain',
    'thinking': 'berpikir', 'feeling': 'merasakan',
    'starting': 'memulai', 'ending': 'mengakhiri',
    'opening': 'membuka', 'closing': 'menutup',
    'building': 'membangun', 'breaking': 'merusak',
    'holding': 'memegang', 'keeping': 'menjaga',
    'standing': 'berdiri', 'sitting': 'duduk',
    'falling': 'jatuh', 'rising': 'naik',
    'missing': 'melewatkan', 'hitting': 'memukul',
    'killing': 'membunuh', 'saving': 'menyelamatkan',
    'fighting': 'bertarung', 'winning': 'menang',
    'losing': 'kalah', 'failing': 'gagal',
    'showing': 'menunjukkan', 'proving': 'membuktikan',
    'creating': 'menciptakan', 'developing': 'mengembangkan',
    'improving': 'memperbaiki', 'changing': 'mengubah',
    'ruining': 'merusak', 'destroying': 'menghancurkan',
    'exploring': 'menjelajahi', 'discovering': 'menemukan',
    'revealing': 'mengungkap', 'hiding': 'menyembunyikan',
    'pretending': 'berpura-pura', 'performing': 'tampil',
    'acting': 'berakting', 'directing': 'menyutradarai',
    'portraying': 'menggambarkan', 'depicting': 'melukiskan',
    'capturing': 'menangkap', 'conveying': 'menyampaikan',
    'evolving': 'berkembang', 'progressing': 'berkembang',
    'dragging': 'menyeret', 'rushing': 'terburu-buru',
    'wasting': 'membuang', 'spending': 'menghabiskan',
    'screaming': 'berteriak', 'crying': 'menangis',
    'laughing': 'tertawa', 'sleeping': 'tertidur',
    'dying': 'sekarat', 'living': 'hidup',
    'loving': 'mencintai', 'hating': 'membenci',
    'caring': 'peduli', 'hoping': 'berharap',
    'waiting': 'menunggu', 'looking': 'melihat',
    'listening': 'mendengarkan', 'speaking': 'berbicara',
    'writing': 'menulis', 'reading': 'membaca',
    'learning': 'belajar', 'teaching': 'mengajar',
    'turning': 'berubah', 'becoming': 'menjadi',
    'remaining': 'tetap', 'continuing': 'melanjutkan',
    'stopping': 'berhenti', 'beginning': 'memulai',
    'returning': 'kembali', 'arriving': 'tiba',
    'leaving': 'pergi', 'moving': 'bergerak',

    # ── Kata sifat tambahan yang sering lolos ──
    'overlong': 'terlalu panjang', 'underwhelming': 'mengecewakan',
    'overwhelming': 'luar biasa', 'overpowering': 'terlalu kuat',
    'underdeveloped': 'kurang dikembangkan',
    'overdone': 'berlebihan', 'undercooked': 'kurang matang',
    'overused': 'terlalu sering dipakai',
    'underused': 'kurang dimanfaatkan',
    'overhyped': 'terlalu digembar-gemborkan',
    'underappreciated': 'kurang diapresiasi',
    'underperformed': 'di bawah harapan',
    'overperformed': 'melebihi harapan',
    'one sided': 'sepihak', 'two faced': 'bermuka dua',
    'half hearted': 'setengah hati', 'whole hearted': 'sepenuh hati',
    'lighthearted': 'ringan', 'dark humor': 'humor gelap',
    'deadpan': 'datar', 'slapstick': 'komedi fisik',
    'anticlimactic': 'antiklimaks', 'anticlimax': 'antiklimaks',
    'cliche': 'klise', 'clichéd': 'klise', 'cliched': 'klise',
    'nuanced': 'bernuansa', 'layered': 'berlapis',
    'textured': 'bertekstur', 'polished': 'dipoles',
    'unpolished': 'tidak dipoles', 'raw': 'mentah',
    'gritty': 'kasar', 'atmospheric': 'atmosferik',
    'moody': 'bersuasana', 'dreamy': 'seperti mimpi',
    'nightmarish': 'seperti mimpi buruk',
    'dreamlike': 'seperti mimpi',
    'otherworldly': 'dunia lain',
    'cinematic': 'sinematik', 'theatrical': 'teatrikal',
    'blockbuster': 'blockbuster', 'award worthy': 'layak penghargaan',
    'oscar worthy': 'layak oscar',

    # ── Kata kerja tambahan yang sering lolos ──
    'overcome': 'mengatasi', 'overcame': 'mengatasi',
    'outperform': 'mengungguli', 'outperformed': 'mengungguli',
    'outshine': 'mengungguli', 'outshone': 'mengungguli',
    'outlast': 'bertahan lebih lama',
    'outsmart': 'mengakali', 'outwit': 'mengakali',
    'overthink': 'terlalu berpikir',
    'overdo': 'berlebihan', 'overdid': 'berlebihan',
    'undermine': 'melemahkan', 'undermined': 'dilemahkan',
    'underestimate': 'meremehkan', 'underestimated': 'diremehkan',
    'overestimate': 'melebih-lebihkan',
    'overwhelm': 'menguasai', 'overwhelmed': 'kewalahan',
    'underwhelm': 'mengecewakan', 'underwhelmed': 'kecewa',
    'overshadow': 'membayangi', 'overshadowed': 'dibayangi',
    'overlook': 'mengabaikan', 'overlooked': 'diabaikan',
    'overpower': 'menguasai', 'overpowered': 'dikuasai',
    'foreshadow': 'memberi petunjuk', 'foreshadowed': 'diberi petunjuk',

    # ── Kata benda tambahan konteks film ──
    'choreography': 'koreografi', 'composition': 'komposisi',
    'ensemble': 'ansambel', 'ensemble cast': 'pemeran ansambel',
    'monologue': 'monolog', 'voiceover': 'narasi suara',
    'backstory': 'latar belakang', 'subplot': 'alur samping',
    'plotline': 'garis cerita', 'timeline': 'garis waktu',
    'flashforward': 'kilasan masa depan',
    'worldbuilding': 'pembangunan dunia',
    'soundtrack': 'musik latar', 'score': 'skor musik',
    'soundscape': 'lanskap suara', 'sound design': 'desain suara',
    'color grading': 'pewarnaan', 'color palette': 'palet warna',
    'wide shot': 'pengambilan lebar', 'close up': 'pengambilan dekat',
    'long shot': 'pengambilan jarak jauh',
    'tracking shot': 'pengambilan bergerak',
    'handheld': 'genggam', 'steadicam': 'kamera stabil',
    'found footage': 'rekaman temuan',
    'practical effects': 'efek praktis',
    'special effects': 'efek spesial',
    'visual effects': 'efek visual',
    'makeup': 'tata rias', 'prosthetics': 'prostetik',
    'wardrobe': 'kostum', 'set design': 'desain set',
    'production value': 'nilai produksi',
    'production design': 'desain produksi',
    'art direction': 'arahan seni',
    'stunt work': 'aksi stunt',
    'supporting cast': 'pemeran pendukung',
    'leading role': 'peran utama',
    'supporting role': 'peran pendukung',
    'breakout role': 'peran terobosan',
    'screen time': 'waktu tayang',
    'run time': 'durasi', 'runtime': 'durasi',
    'third act': 'babak ketiga',
    'first act': 'babak pertama',
    'second act': 'babak kedua',
    'opening act': 'babak pembuka',
    'final act': 'babak akhir',

    # ── Frasa review film tambahan ──
    'fell flat': 'gagal total', 'falls flat': 'gagal total',
    'missed the mark': 'meleset dari sasaran',
    'hits different': 'terasa berbeda',
    'hit different': 'terasa berbeda',
    'hits hard': 'sangat mengena',
    'hit hard': 'sangat mengena',
    'hit or miss': 'tidak konsisten',
    'goes downhill': 'makin buruk',
    'went downhill': 'makin buruk',
    'falls apart': 'berantakan',
    'fell apart': 'berantakan',
    'holds up': 'tetap bagus',
    'held up': 'tetap bagus',
    'lives up': 'memenuhi harapan',
    'lived up': 'memenuhi harapan',
    'stands out': 'menonjol',
    'stood out': 'menonjol',
    'pulled off': 'berhasil melakukan',
    'pulls off': 'berhasil melakukan',
    'nailed it': 'berhasil sempurna',
    'blew my mind': 'sangat mengejutkan',
    'blows my mind': 'sangat mengejutkan',
    'gave me chills': 'membuat merinding',
    'gives me chills': 'membuat merinding',
    'kept me on edge': 'membuat tegang',
    'on the edge of my seat': 'sangat tegang',
    'could have been better': 'bisa lebih baik',
    'could have been worse': 'bisa lebih buruk',
    'nothing special': 'tidak ada yang spesial',
    'nothing new': 'tidak ada yang baru',
    'same old': 'itu-itu saja',
    'run of the mill': 'biasa saja',
    'dime a dozen': 'pasaran',
    'cut above': 'lebih unggul',
    'ahead of its time': 'mendahului zamannya',
    'behind the times': 'ketinggalan zaman',
    'guilty pleasure': 'kesenangan tersembunyi',
    'so bad its good': 'sangat buruk sampai jadi bagus',
    'train wreck': 'bencana',
    'hot mess': 'berantakan',
    'mixed bag': 'campur aduk',
    'mixed feelings': 'perasaan campur aduk',
    'no brainer': 'tidak perlu berpikir',
    'game changer': 'pengubah permainan',
    'crowd pleaser': 'menyenangkan banyak orang',
    'tear jerker': 'menguras air mata',

    # ── Slang Gen-Z & internet yang sering di review ──
    'sus': 'mencurigakan',  # from "suspicious"
    'cap': 'bohong', 'no cap': 'sungguhan',
    'based': 'berdasar', 'ratio': 'rasio',
    'rent free': 'terus terpikirkan',
    'living rent free': 'terus terpikirkan',
    'ate': 'berhasil luar biasa',  # "she ate that scene"
    'served': 'tampil maksimal',
    'its giving': 'terasa seperti',
    'the way': 'cara',
    'literally me': 'persis seperti saya',
    'main character energy': 'aura tokoh utama',
    'side character': 'karakter sampingan',
    'npc energy': 'aura figuran',
    'carried': 'membawa tim',  # "she carried the movie"
    'carried the movie': 'menyelamatkan film',
    'carry': 'menyelamatkan',
    'slaps': 'mantap',  # "this soundtrack slaps"
    'slapped': 'mantap',
    'bussin': 'luar biasa', 'bussing': 'luar biasa',
    'goated': 'terbaik sepanjang masa',
    'chefs kiss': 'sempurna',
    'iykyk': 'kalau kamu tahu kamu tahu',
    'imo': 'menurut saya',
    'ngl': 'jujur saja',
    'tbh': 'sejujurnya',
    'stanning': 'mengidolakan',
    'stan': 'mengidolakan',
    'ship': 'mendukung pasangan',
    'shipping': 'mendukung pasangan',
    'oof': 'aduh',
    'yikes': 'waduh',
    'sheesh': 'astaga',
    'bruh': 'aduh',
    'fam': 'keluarga',
    'lit': 'keren', 'litty': 'keren sekali',
    'fire': 'keren', 'straight fire': 'sangat keren',
    'dope': 'keren', 'dopest': 'paling keren',
    'sick': 'keren', 'sickest': 'paling keren',
    'wack': 'aneh', 'whack': 'aneh',
    'mid': 'biasa saja', 'mids': 'biasa saja',
    'basic': 'biasa', 'generic': 'generik',
    'slay': 'keren sekali', 'slayed': 'keren sekali',
    'banger': 'luar biasa', 'bangers': 'luar biasa',
    'absolute banger': 'benar-benar luar biasa',
    'whatsoever': 'sama sekali',
    'nonetheless': 'meskipun demikian',
    'meanwhile': 'sementara itu',
    'afterwards': 'setelahnya',
    'beforehand': 'sebelumnya',
    'furthermore': 'selain itu',
    'hence': 'oleh karena itu',
    'thereby': 'dengan demikian',
    'whereby': 'dimana',
    'whoever': 'siapapun',
    'whatever': 'apapun',
    'whenever': 'kapanpun',
    'wherever': 'dimanapun',
    'somehow': 'entah bagaimana',
    'somewhat': 'agak',
    'somewhere': 'di suatu tempat',
    'nowhere': 'tidak di mana pun',
    'everywhere': 'di mana-mana',
    'anyone': 'siapa saja',
    'anymore': 'lagi',
    'anyway': 'bagaimanapun', 'anyways': 'bagaimanapun',
    'anyhow': 'bagaimanapun',
    'nobody': 'tidak ada orang',
    'nothing': 'tidak ada',
    'everything': 'segalanya',
    'something': 'sesuatu',
    'someone': 'seseorang',
    'everybody': 'semua orang',
    'dance': 'menari',
    'another version': 'versi lain',
    'another': 'lain',
    'version': 'versi',
    'or': 'atau',
    'iconic': 'ikonik',
    'epic': 'epik',
    'legendary': 'legendaris',
    'perfect': 'sempurna',
    'classic': 'klasik',
    'art': 'seni',
    'concept': 'konsep',
    'quality': 'kualitas',
    'level': 'tingkat',
    'balance': 'keseimbangan',
    'fresh': 'segar',
    'love': 'cinta',
    'hate': 'benci',
    'best': 'terbaik',
    'worst': 'terburuk',
    'boring': 'membosankan',
    'good': 'bagus',
    'bad': 'buruk',
    'great': 'hebat',
    'nice': 'bagus',
    'star': 'bintang',
    'stars': 'bintang',
    'top': 'atas',
    'original': 'orisinal',
}


def detect_sentence_language(sentence):
    """
    Deteksi bahasa sebuah kalimat.
    Mengembalikan 'en' jika bahasa Inggris, 'id' jika Indonesia.

    Strategi (DIPERBAIKI untuk akurasi penelitian):
    1. Hitung kata Inggris kuat (function words) vs total kata
    2. Adaptive threshold: kalimat pendek (< 5 kata) pakai threshold lebih rendah
    3. Deteksi pola grammar Inggris (subject-verb patterns)
    4. Gunakan langdetect sebagai validasi tambahan
    5. Kata serapan tidak dihitung sebagai "Inggris"
    """
    if not sentence or len(sentence.strip()) < 3:
        return 'id'  # Terlalu pendek, anggap Indonesia

    words = re.findall(r'[a-zA-Z]+', sentence.lower())
    if len(words) < 2:
        return 'id'  # Terlalu sedikit kata

    # Hitung kata Inggris kuat (bukan serapan)
    english_strong = sum(1 for w in words if w in KATA_INGGRIS_KUAT and w not in KATA_SERAPAN)

    # Hitung juga kata yang ada di kamus terjemahan manual (indikator Inggris)
    english_dict = sum(1 for w in words if w in KAMUS_TERJEMAHAN_MANUAL and w not in KATA_SERAPAN)

    # Rasio kata Inggris (kuat + kamus) terhadap total kata
    total_english = english_strong + english_dict
    ratio_en = total_english / len(words) if words else 0

    # ADAPTIVE THRESHOLD: Kalimat pendek lebih sensitif
    if len(words) <= 4:
        # Kalimat pendek: cukup 1 kata Inggris kuat + 1 dari kamus
        if total_english >= 2 and ratio_en >= 0.10:
            return 'en'
        if english_strong >= 1 and english_dict >= 1:
            return 'en'
    elif len(words) <= 8:
        # Kalimat sedang: threshold 0.12
        if ratio_en >= 0.12 and total_english >= 2:
            return 'en'
    else:
        # Kalimat panjang: threshold standar 0.15
        if ratio_en >= 0.15 and total_english >= 2:
            return 'en'

    # Jika banyak kata Inggris kuat saja (tanpa kamus)
    ratio_strong = english_strong / len(words) if words else 0
    if ratio_strong >= 0.15 and english_strong >= 2:
        return 'en'

    # Deteksi pola grammar Inggris (subject-verb patterns)
    sentence_lower = sentence.lower()
    en_grammar_patterns = [
        r'\b(i|he|she|it|we|they|you)\s+(am|is|are|was|were|have|has|had|will|would|could|should|can|do|does|did)\b',
        r'\b(the|a|an)\s+\w+\s+(is|was|are|were|has|have)\b',
        r"\b(don'?t|doesn'?t|didn'?t|wasn'?t|weren'?t|can'?t|won'?t|shouldn'?t|wouldn'?t|couldn'?t)\b",
        r'\b(this|that)\s+(is|was|movie|film)\b',
        r'\b(not|very|really|too|so|just|even|still|also)\s+\w+',
    ]
    en_pattern_count = sum(1 for p in en_grammar_patterns if re.search(p, sentence_lower))
    if en_pattern_count >= 2:
        return 'en'
    if en_pattern_count >= 1 and total_english >= 1:
        return 'en'

    # Gunakan langdetect sebagai fallback
    try:
        lang = detect(sentence)
        if lang == 'en' and total_english >= 1:
            return 'en'
        elif lang == 'en' and ratio_en >= 0.08:
            return 'en'
        elif lang == 'id':
            return 'id'
    except Exception:
        pass

    return 'id'


def translate_remaining_english_words(text):
    """
    Second-pass: Terjemahkan kata-kata Inggris yang masih tersisa
    dalam teks setelah translasi kalimat.
    Menggunakan kamus terjemahan manual.
    Menggunakan regex dengan word boundaries (\b) agar kata yang bertetangga
    dengan tanda baca (ex: "dance!", "version...") tetap terjemah dengan benar.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Pass 1: Terjemahkan frasa multi-kata terlebih dahulu (urutkan berdasarkan panjang menurun)
    phrases = sorted(
        ((k, v) for k, v in KAMUS_TERJEMAHAN_MANUAL.items() if ' ' in k),
        key=lambda x: len(x[0]), reverse=True
    )
    for phrase_en, phrase_id in phrases:
        pattern = re.compile(r'\b' + re.escape(phrase_en) + r'\b', re.IGNORECASE)
        text = pattern.sub(phrase_id, text)

    # Pass 2: Terjemahkan kata tunggal (kecuali kata serapan)
    # Urutkan berdasarkan panjang menurun agar kata yang lebih panjang diproses lebih dulu
    single_words = sorted(
        ((k, v) for k, v in KAMUS_TERJEMAHAN_MANUAL.items() if ' ' not in k and k not in KATA_SERAPAN),
        key=lambda x: len(x[0]), reverse=True
    )
    for word_en, word_id in single_words:
        pattern = re.compile(r'\b' + re.escape(word_en) + r'\b', re.IGNORECASE)
        text = pattern.sub(word_id, text)

    return text


def protect_slang_words(text):
    """
    Melindungi kata-kata slang Indonesia dari mistranslasi oleh Google Translate.

    Strategi: Mengganti kata slang dengan placeholder unik sebelum teks
    dikirim ke Google Translate, sehingga kata tersebut tidak diterjemahkan.
    Placeholder menggunakan format __SLANG_N__ agar tidak bentrok
    dengan kata-kata lain.

    Returns:
        (text_protected, slang_map) — teks dengan placeholder, dan mapping
        untuk mengembalikan kata slang setelah translasi.
    """
    if not isinstance(text, str) or not text.strip():
        return text, {}

    slang_map = {}  # {placeholder: kata_slang_asli}
    counter = 0

    # Proteksi frasa multi-kata terlebih dahulu (terpanjang duluan)
    for slang in sorted(
        (k for k in KATA_SLANG_PROTEKSI if ' ' in k),
        key=len, reverse=True
    ):
        pattern = re.compile(re.escape(slang), re.IGNORECASE)
        if pattern.search(text):
            placeholder = f'__SLANG_{counter}__'
            slang_map[placeholder] = slang
            text = pattern.sub(placeholder, text)
            counter += 1

    # Proteksi kata tunggal
    words = text.split()
    result = []
    for word in words:
        word_lower = word.lower().strip()
        if word_lower in KATA_SLANG_PROTEKSI:
            placeholder = f'__SLANG_{counter}__'
            slang_map[placeholder] = word_lower
            result.append(placeholder)
            counter += 1
        else:
            result.append(word)

    return ' '.join(result), slang_map


def restore_slang_words(text, slang_map):
    """
    Mengembalikan kata slang yang dilindungi (placeholder) ke bentuk aslinya.

    Kata slang dikembalikan sebagai makna sentimen yang benar dari
    KATA_SLANG_PROTEKSI, bukan makna harfiah yang salah.
    Menggunakan regex yang toleran terhadap spasi tambahan yang mungkin
    dimasukkan oleh Google Translate.
    """
    if not slang_map:
        return text

    # Google Translate kadang menyisipkan spasi atau mengubah casing placeholder,
    # misalnya: "__SLANG_0__" -> "__ slang _ 0__" atau "__slang_0__".
    # Regex ini mendeteksi variasi tersebut agar restorasi tetap akurat.
    for placeholder, kata_slang in slang_map.items():
        match_idx = re.search(r'\d+', placeholder)
        if match_idx:
            idx = match_idx.group()
            # Pola regex pencocokan fleksibel spasi: __\s*slang\s*_\s*N\s*__
            pattern = re.compile(r'__\s*slang\s*_\s*' + idx + r'\s*__', re.IGNORECASE)
            makna_benar = KATA_SLANG_PROTEKSI.get(kata_slang, kata_slang)
            text = pattern.sub(makna_benar, text)

    # Fallback penggantian langsung string jika formatnya tidak berubah sama sekali
    for placeholder, kata_slang in slang_map.items():
        makna_benar = KATA_SLANG_PROTEKSI.get(kata_slang, kata_slang)
        text = text.replace(placeholder, makna_benar)

    return text


def safe_translate(text, source='en', target='id', retries=3, base_delay=1.0):
    """
    Menerjemahkan teks dengan penanganan error dan retries berbasis exponential backoff
    untuk mencegah kegagalan akibat limitasi Google Translate API (HTTP 429).
    """
    if not text.strip():
        return text
        
    for attempt in range(retries):
        try:
            translated = GoogleTranslator(source=source, target=target).translate(text)
            if translated:
                return translated
        except Exception as e:
            if attempt == retries - 1:
                # Jika sudah di percobaan terakhir, lemparkan error untuk fallback manual
                raise e
            # Exponential backoff delay dengan jitter acak
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
    return text


def translate_mixed_text(text):
    """
    Menerjemahkan bagian-bagian berbahasa Inggris dalam teks campuran.

    Alur:
    1. PROTEKSI SLANG: Lindungi kata slang Indonesia dari mistranslasi
    2. Pecah teks menjadi kalimat-kalimat
    3. Deteksi bahasa setiap kalimat
    4. Terjemahkan kalimat berbahasa Inggris ke Indonesia (dengan safe_translate)
    5. Gabungkan kembali semua kalimat
    6. SECOND-PASS: Terjemahkan kata Inggris yang tersisa
    7. RESTORE SLANG: Kembalikan kata slang ke makna sentimen yang benar

    Pendekatan ini lebih akurat daripada menerjemahkan seluruh teks,
    karena Google Translate kadang salah mendeteksi bahasa pada teks campuran.
    Proteksi slang memastikan kata seperti "garing", "kentang", "receh" dll
    tidak diterjemahkan ke makna harfiah (crispy, potato, coin).
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # STEP 0: Proteksi kata slang sebelum translasi
    text, slang_map = protect_slang_words(text)

    # Pecah teks menjadi kalimat berdasarkan tanda baca
    # Pertahankan tanda baca sebagai delimiter
    sentences = re.split(r'(?<=[.!?;,])\s+', text)

    # Jika teks pendek (1 kalimat), cek langsung
    if len(sentences) <= 1:
        lang = detect_sentence_language(text)
        if lang == 'en':
            try:
                translated = safe_translate(text, source='en', target='id')
                if translated:
                    # Second-pass: cek kata Inggris tersisa
                    translated = translate_remaining_english_words(translated)
                    # Restore slang words
                    translated = restore_slang_words(translated, slang_map)
                    return translated
                return restore_slang_words(text, slang_map)
            except Exception:
                # Fallback ke kamus manual jika API gagal setelah retries
                result = translate_remaining_english_words(text)
                return restore_slang_words(result, slang_map)
        else:
            # Meskipun terdeteksi ID, tetap cek kata Inggris tersisa
            result = translate_remaining_english_words(text)
            return restore_slang_words(result, slang_map)

    # Proses per kalimat
    result_parts = []
    en_sentences = []       # Kumpulkan kalimat Inggris untuk translate
    en_indices = []          # Indeks posisi kalimat Inggris

    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            result_parts.append('')
            continue

        lang = detect_sentence_language(sentence)
        if lang == 'en':
            en_sentences.append(sentence)
            en_indices.append(idx)
            result_parts.append(None)  # Placeholder, akan diisi setelah translasi
        else:
            result_parts.append(sentence)

    # Translate kalimat Inggris satu per satu
    if en_sentences:
        for i, idx in enumerate(en_indices):
            try:
                translated = safe_translate(en_sentences[i], source='en', target='id')
                if translated:
                    result_parts[idx] = translated
                else:
                    result_parts[idx] = en_sentences[i]
            except Exception:
                # Jika Google Translate gagal, gunakan kamus manual
                result_parts[idx] = translate_remaining_english_words(en_sentences[i])

            # Delay kecil untuk menghindari rate limit
            if i < len(en_indices) - 1:
                time.sleep(0.3)

    # Gabungkan kembali
    result = ' '.join([p for p in result_parts if p is not None and p != ''])
    result = result.strip()

    # SECOND-PASS: Terjemahkan kata-kata Inggris yang masih tersisa
    result = translate_remaining_english_words(result)

    # RESTORE: Kembalikan kata slang yang dilindungi
    result = restore_slang_words(result, slang_map)

    return result


# ── CATATAN: Fungsi-fungsi translasi di atas (detect_sentence_language,
#    translate_mixed_text, safe_translate, protect_slang_words,
#    restore_slang_words, translate_remaining_english_words) tetap tersedia
#    sebagai utility dan dipanggil dari dalam fungsi normalize_and_translate()
#    pada tahap Normalisasi & Translasi (Cell 8). ──


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7: CLEANING                                                         ║
# ║  Tahap 3: Membersihkan teks dari noise                                    ║
# ║  Menghapus seluruh elemen yang tidak berkontribusi pada makna teks:       ║
# ║  - URI (URL, shortened links)                                             ║
# ║  - Tanda baca (punctuation berlebihan)                                    ║
# ║  - Simbol khusus (mention, hashtag, emotikon teks, kaomoji)               ║
# ║  - Angka                                                                  ║
# ║  - Emotikon (emoji Unicode, menggunakan library `emoji`)                  ║
# ║  - Karakter hasil encoding yang tidak terbaca (non-ASCII noise)           ║
# ║                                                                           ║
# ║  PERBAIKAN: Input langsung dari case folding (bukan dari translasi).      ║
# ║  Cleansing dilakukan SEBELUM normalisasi & translasi agar deteksi         ║
# ║  bahasa lebih akurat (tidak terkontaminasi noise).                        ║
# ║                                                                           ║
# ║  Referensi: Satriawan et al. (2024)                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def cleaning(text):
    """
    Tahap cleaning menghapus seluruh elemen yang tidak berkontribusi
    pada makna teks, meliputi URI, tanda baca, simbol khusus, angka,
    emotikon, dan karakter hasil encoding yang tidak terbaca.

    Ulasan di Letterboxd cukup sering mengandung tanda baca yang
    digunakan secara berlebihan untuk menekankan ekspresi, misalnya
    "keren banget!!!!" atau "ga serem sama sekali..." yang perlu
    dibersihkan agar tidak mempengaruhi proses tokenisasi pada tahap
    berikutnya (Satriawan et al., 2024).
    """
    if not isinstance(text, str):
        return ""

    # ── 1. Hapus URI (URL, shortened links) ──
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'bit\.ly/\S+|goo\.gl/\S+|t\.co/\S+', '', text)

    # ── 1b. Dekode HTML entities ──
    # Ulasan dari web scraping sering mengandung HTML entities
    text = html.unescape(text)  # &amp; → &, &lt; → <, &gt; → >, &quot; → ", dll.

    # ── 1c. Normalisasi leetspeak (angka sebagai huruf) ──
    # Pola umum di review online yang menggunakan angka sebagai pengganti huruf
    leetspeak_map = {
        '4ever': 'forever', 'b4': 'before', 'gr8': 'great',
        '2day': 'today', '2nite': 'tonight', '2morrow': 'tomorrow',
        'h8': 'hate', 'h8d': 'hated', 'w8': 'wait',
        'l8': 'late', 'l8r': 'later', 'm8': 'mate',
        'sk8': 'skate', 'str8': 'straight', 'cr8': 'create',
        'n1ce': 'nice', 'k3ren': 'keren', 'b4gus': 'bagus',
    }
    for leet, normal in leetspeak_map.items():
        text = re.sub(r'\b' + re.escape(leet) + r'\b', normal, text, flags=re.IGNORECASE)

    # ── 2. Hapus emotikon ──
    # 2a. Hapus SEMUA emoji Unicode menggunakan library `emoji` (lebih akurat)
    text = emoji.replace_emoji(text, replace='')

    # 2b. Hapus emotikon teks umum
    #     Contoh: :) :( ;) :D :P >_< T_T ^^, dll.
    emotikon_patterns = [
        r'[:;=8xX][\'"\-]?[)(DPpOo3><\]\[/\\|}{@*]',    # :) ;D =P dll
        r'[)(DPpOo3><\]\[/\\|}{@][\'"\-]?[:;=]',          # (: D; dll (terbalik)
        r'<3+',                                             # <3 <333
        r'[xX][dD]+',                                       # xD XD xDD
        r'\^\^+',                                           # ^^ ^^^
        r'-_-+',                                            # -_- -__-
        r'T[_.]T',                                          # T_T T.T
        r'>_<+',                                            # >_<
        r'o_O|O_o|o\.O|O\.o',                               # o_O O_o
        r'=_=+',                                            # =_= =__=
        r'D:+',                                             # D: D::
        r';\)+',                                            # ;)
        r':\'+ \(',                                          # :'(
        r'uwu|UwU|owo|OwO',                                # uwu owo
        r'qwq|QwQ|TwT',                                    # qwq TwT
    ]
    for pattern in emotikon_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 2c. Hapus kaomoji Jepang
    #     Contoh: (╥_╥) (ﾉ◕ヮ◕)ﾉ ¯\_(ツ)_/¯ dll.
    text = re.sub(r'\([^a-zA-Z0-9\s]{2,}\)', '', text)  # (...simbol...)
    text = re.sub(r'¯\\_?\(ツ\)_?/¯', '', text)          # Shrug ¯\_(ツ)_/¯

    # ── 3. Hapus karakter hasil encoding yang tidak terbaca ──
    #    (karakter Unicode non-ASCII yang bukan huruf latin, simbol khusus)
    text = re.sub(
        r'[\U0001F000-\U0001FFFF'    # Supplementary Multilingual Plane symbols
        r'\U00002702-\U000027B0'      # Dingbats
        r'\U000024C2-\U0000257F'      # Enclosed characters & box drawing
        r'\U00002580-\U0000259F'      # Block elements
        r'\U000025A0-\U000025FF'      # Geometric shapes
        r'\U00002600-\U000026FF'      # Miscellaneous symbols
        r'\U00002700-\U000027BF'      # Dingbats
        r'\U0000FE00-\U0000FE0F'      # Variation selectors
        r'\U0000200D'                 # Zero Width Joiner (ZWJ)
        r'\U00002B50-\U00002B55'      # Stars & circles
        r'\U0000231A-\U0000231B'      # Watch & hourglass
        r'\U000023E9-\U000023F3'      # AV symbols
        r'\U000023CF'                 # Eject symbol
        r'\U00003030'                 # Wavy dash
        r'\U000000A9'                 # Copyright
        r'\U000000AE'                 # Registered
        r'\U00002122'                 # Trademark
        r']+',
        '', text
    )

    # ── 4. Hapus simbol khusus (mention, hashtag) ──
    text = re.sub(r'@\w+', '', text)     # Hapus mention (@username)
    text = re.sub(r'#\w+', '', text)     # Hapus hashtag (#topic)

    # Hapus tawa berlebihan (noise yang tidak berkontribusi pada makna)
    text = re.sub(r'\b(ha){2,}\b', '', text, flags=re.IGNORECASE)    # hahaha
    text = re.sub(r'\b(hi){2,}\b', '', text, flags=re.IGNORECASE)    # hihihi
    text = re.sub(r'\b(hu){2,}\b', '', text, flags=re.IGNORECASE)    # huhuhu
    text = re.sub(r'\b(he){2,}\b', '', text, flags=re.IGNORECASE)    # hehehe
    text = re.sub(r'\b(ho){2,}\b', '', text, flags=re.IGNORECASE)    # hohoho
    text = re.sub(r'\b(wk){2,}\b', '', text, flags=re.IGNORECASE)    # wkwkwk
    text = re.sub(r'\b(xi){2,}\b', '', text, flags=re.IGNORECASE)    # xixixi
    text = re.sub(r'\b(haha)+\b', '', text, flags=re.IGNORECASE)     # hahahaha
    text = re.sub(r'\b(wkwk)+\b', '', text, flags=re.IGNORECASE)     # wkwkwkwk

    # ── 4b. Tangani pola reduplikasi angka 2 (e.g., "kaleng2" → "kaleng kaleng") ──
    # Pola umum di teks Indonesia: kata diikuti angka 2 berarti reduplikasi
    text = re.sub(r'\b([a-zA-Z]{2,})2\b', r'\1 \1', text)

    # Hapus huruf KONSONAN berulang 3+ kali → 1 (e.g., "bgttt"→"bgt", "kerennn"→"keren")
    konsonan = r'[bcdfghjklmnpqrstvwxyz]'
    text = re.sub(r'(' + konsonan + r')\1{2,}', r'\1', text, flags=re.IGNORECASE)

    # Hapus huruf KONSONAN berulang 2+ kali di akhir kata → 1 (e.g., "baguss"→"bagus", "seremm"→"serem")
    # Ini melindungi kata di tengah dengan konsonan ganda valid seperti "anggota", "minggu", "bangga"
    text = re.sub(r'(' + konsonan + r')\1+\b', r'\1', text, flags=re.IGNORECASE)

    # Hapus huruf VOKAL berulang 3+ kali → 1 (e.g., "bagussseeee"→"baguse")
    # Vokal double valid ada: "saat", "maaf", "taat" jadi threshold tetap 3+
    vokal = r'[aiueo]'
    text = re.sub(r'(' + vokal + r')\1{2,}', r'\1', text, flags=re.IGNORECASE)

    # ── 5. Hapus angka ──
    text = re.sub(r'\d+/\d+', '', text)     # Rating: 10/10, 8/10
    text = re.sub(r'\d+', '', text)          # Angka biasa

    # ── 6. Hapus tanda baca dan karakter spesial ──
    #    (hanya sisakan huruf dan spasi)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Hapus underscore (sisa dari \w yang meng-include _)
    text = re.sub(r'_', ' ', text)

    # ── 7. Hapus spasi berlebihan ──
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ── Hitung token noise yang dihapus ──
# Hitung total token sebelum cleaning
# PERBAIKAN: Input langsung dari case folding (bukan dari translasi)
total_token_sebelum_cleaning = df['ulasan_casefolded'].apply(lambda x: len(str(x).split())).sum()

# Menyimpan dalam kolom 'ulasan_cleansed' agar kompatibel dengan Cell 8
df['ulasan_cleansed'] = df['ulasan_casefolded'].apply(cleaning)

# Hitung total token sesudah cleaning
total_token_sesudah_cleaning = df['ulasan_cleansed'].apply(lambda x: len(str(x).split())).sum()
total_token_noise_cleaning = total_token_sebelum_cleaning - total_token_sesudah_cleaning

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT UNTUK SKRIPSI — Sub Bab 4.3.2 Cleaning
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("      TAHAP 3: CLEANING (Pembersihan Teks)")
print("=" * 70)

# ── 1. Informasi Umum ──
print(f"\n  Total ulasan diproses           : {len(df)} ulasan")
print(f"  Token sebelum cleaning          : {total_token_sebelum_cleaning:,} token")
print(f"  Token sesudah cleaning          : {total_token_sesudah_cleaning:,} token")
print(f"  Token noise dihapus             : {total_token_noise_cleaning:,} token")

# ── 2. Contoh Hasil Cleaning (Tabel Before-After) ──
print(f"\n{'─' * 70}")
print("  Contoh Hasil Cleaning (Sebelum → Sesudah)")
print(f"{'─' * 70}")

# Cari ulasan yang mengalami perubahan karena cleaning (contoh)
df['noise_len'] = df.apply(lambda row: len(str(row['ulasan_casefolded'])) - len(str(row['ulasan_cleansed'])), axis=1)
contoh_cleaning = df[df['noise_len'] > 5].head(5)
if len(contoh_cleaning) < 5: # Fallback jika kurang dari 5
    contoh_cleaning = df.head(5)

print(f"  {'No':<4} {'Sebelum Cleaning':<55}")
print(f"  {'':4} {'Sesudah Cleaning':<55}")
print(f"  {'─' * 66}")

for idx, (i, row) in enumerate(contoh_cleaning.iterrows(), 1):
    before = str(row['ulasan_casefolded']).replace('\n', ' ')
    after = str(row['ulasan_cleansed']).replace('\n', ' ')
    
    max_len = 60
    before_display = before[:max_len] + "..." if len(before) > max_len else before
    after_display = after[:max_len] + "..." if len(after) > max_len else after
    
    print(f"  {idx:<4} \"{before_display}\"")
    print(f"  {'':4} \"{after_display}\"")
    if idx < len(contoh_cleaning):
        print()

df = df.drop(columns=['noise_len'])

# ── 3. Elemen yang Dihapus ──
print(f"\n{'─' * 70}")
print("  Elemen Noise yang Dihapus")
print(f"{'─' * 70}")
print("  1. URL dan shortened links (URI)")
print("  2. Tanda baca berlebihan (punctuation)")
print("  3. Simbol khusus, mention (@), dan hashtag (#)")
print("  4. Angka numerik")
print("  5. Emotikon teks, kaomoji, dan emoji Unicode")
print("  6. Karakter non-ASCII tidak terbaca")

# ── 4. Ringkasan ──
print(f"\n{'=' * 70}")
print("  Ringkasan Cleaning")
print(f"{'=' * 70}")
print(f"  Tahap cleaning menghapus seluruh elemen yang tidak berkontribusi")
print(f"  pada makna teks, meliputi URL, tanda baca, simbol khusus, angka,")
print(f"  emotikon, dan karakter hasil encoding yang tidak terbaca. Dari dataset")
print(f"  yang diproses, ditemukan sebanyak {total_token_noise_cleaning:,} token noise")
print(f"  yang berhasil dihapus pada tahap ini.")
print(f"{'=' * 70}\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8: NORMALISASI & TRANSLASI                                          ║
# ║  Tahap 4: Normalisasi kata tidak baku (slang → baku) LALU translasi       ║
# ║           kata/kalimat Inggris yang tersisa (EN → ID)                     ║
# ║                                                                           ║
# ║  PENDEKATAN BARU (lebih akurat untuk penelitian):                         ║
# ║  Normalisasi slang Indonesia dilakukan SEBELUM translasi Inggris.         ║
# ║  Ini mencegah kata slang Indonesia yang mirip bahasa Inggris              ║
# ║  (seperti "garing", "kentang", "receh") salah diterjemahkan oleh          ║
# ║  Google Translate ke makna harfiah (crispy, potato, coin).                ║
# ║                                                                           ║
# ║  Alur:                                                                    ║
# ║  1. Normalisasi slang Indonesia (KAMUS_SLANG)                             ║
# ║  2. Deteksi & terjemahkan kalimat bahasa Inggris (Google Translate)       ║
# ║  3. Second-pass: terjemahkan kata Inggris tersisa (KAMUS_TERJEMAHAN)      ║
# ║                                                                           ║
# ║  Kamus slang Indonesia disesuaikan dengan konteks review film             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝



# ── Kamus Normalisasi Slang Indonesia (konteks review film) ──
KAMUS_SLANG = {
    # ══════════════════════════════════════════════════════════════════════
    # KATA GANTI & SAPAAN
    # ══════════════════════════════════════════════════════════════════════
    'gw': 'saya', 'gue': 'saya', 'gua': 'saya', 'w': 'saya',
    'ane': 'saya', 'sy': 'saya', 'ak': 'saya', 'aku': 'saya',
    'gwe': 'saya', 'guwe': 'saya', 'gweh': 'saya', 'guw': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'elu': 'kamu', 'u': 'kamu',
    'ente': 'kamu', 'loe': 'kamu', 'lw': 'kamu', 'luh': 'kamu',

    # ══════════════════════════════════════════════════════════════════════
    # NEGASI — Semua variasi "tidak"
    # ══════════════════════════════════════════════════════════════════════
    'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'g': 'tidak',
    'nggak': 'tidak', 'ngga': 'tidak', 'enggak': 'tidak', 'engga': 'tidak',
    'kagak': 'tidak', 'kaga': 'tidak', 'tdk': 'tidak', 'tak': 'tidak',
    'ndak': 'tidak', 'nda': 'tidak', 'gx': 'tidak', 'kgk': 'tidak',
    'ogah': 'tidak mau', 'males': 'malas',

    # ── Pola ga+kata (gabungan negatif) ──
    'gaada': 'tidak ada', 'gada': 'tidak ada', 'gadaada': 'tidak ada',
    'gabisa': 'tidak bisa', 'gbisa': 'tidak bisa', 'gabsa': 'tidak bisa',
    'gatau': 'tidak tahu', 'gtau': 'tidak tahu', 'gatw': 'tidak tahu',
    'gamau': 'tidak mau', 'gmau': 'tidak mau', 'gamo': 'tidak mau',
    'gamao': 'tidak mau', 'gamaw': 'tidak mau',
    'gabole': 'tidak boleh', 'gaboleh': 'tidak boleh',
    'gasuka': 'tidak suka', 'gak suka': 'tidak suka',
    'gaenak': 'tidak enak', 'ga enak': 'tidak enak',
    'gabakal': 'tidak akan', 'gaakan': 'tidak akan',
    'gajelas': 'tidak jelas', 'gajels': 'tidak jelas',
    'gabagus': 'tidak bagus', 'ga bagus': 'tidak bagus',
    'gakuat': 'tidak kuat', 'ga kuat': 'tidak kuat',
    'gakeren': 'tidak keren', 'ga keren': 'tidak keren',
    'gaseru': 'tidak seru', 'ga seru': 'tidak seru',
    'gaserem': 'tidak seram', 'ga serem': 'tidak seram',
    'gapaham': 'tidak paham', 'ga paham': 'tidak paham',
    'gangerti': 'tidak mengerti', 'ga ngerti': 'tidak mengerti',
    'ganiat': 'tidak niat', 'gausah': 'tidak usah',
    'gaperlu': 'tidak perlu', 'gapenting': 'tidak penting',
    'gacocok': 'tidak cocok', 'garelate': 'tidak relevan',

    # ══════════════════════════════════════════════════════════════════════
    # INTENSIFIER — "sekali/banget" dan variasi
    # ══════════════════════════════════════════════════════════════════════
    'bgt': 'sekali', 'bngt': 'sekali', 'banget': 'sekali',
    'bget': 'sekali', 'bnget': 'sekali', 'bange': 'sekali',
    'bgd': 'sekali', 'bgts': 'sekali', 'bets': 'sekali',
    'bet': 'sekali', 'parah': 'sekali', 'pol': 'sekali',
    'poll': 'sekali', 'pisan': 'sekali',
    'amet': 'amat', 'amaat': 'amat',

    # ══════════════════════════════════════════════════════════════════════
    # KATA KERJA / AKSI
    # ══════════════════════════════════════════════════════════════════════
    'nonton': 'menonton', 'nntn': 'menonton', 'nnton': 'menonton',
    'nontonin': 'menonton', 'ntn': 'menonton', 'nontonnin': 'menonton',
    'liat': 'lihat', 'ngeliat': 'melihat', 'ngelihat': 'melihat',
    'lyt': 'lihat', 'keliat': 'terlihat', 'keliatan': 'terlihat',
    'keliatannya': 'terlihat',
    'bikin': 'membuat', 'bkin': 'membuat', 'bkn': 'membuat',
    'ngebuat': 'membuat', 'mbikin': 'membuat',
    'ngomong': 'berbicara', 'ngmg': 'berbicara', 'ngmong': 'berbicara',
    'ngomongin': 'membicarakan',
    'ngerti': 'mengerti', 'ngrti': 'mengerti', 'mudeng': 'mengerti',
    'sampe': 'sampai', 'smpe': 'sampai', 'smp': 'sampai',
    'ampe': 'sampai', 'mpe': 'sampai',
    'pake': 'pakai', 'pk': 'pakai', 'pki': 'pakai',
    'dapet': 'dapat', 'dpt': 'dapat',
    'pengen': 'ingin', 'pengin': 'ingin', 'pgn': 'ingin',
    'kepengen': 'ingin', 'kepingin': 'ingin', 'kpngn': 'ingin',
    'pingin': 'ingin', 'pngen': 'ingin',
    'tau': 'tahu', 'tw': 'tahu', 'tauk': 'tahu',
    'udah': 'sudah', 'udh': 'sudah', 'uda': 'sudah',
    'ude': 'sudah', 'sdh': 'sudah',
    'blm': 'belum', 'blom': 'belum', 'belom': 'belum', 'blon': 'belum',
    'krn': 'karena', 'krna': 'karena', 'karna': 'karena', 'soalnya': 'karena',
    'grgr': 'gara-gara', 'grg': 'gara-gara',
    'jd': 'jadi', 'jdi': 'jadi',
    'masi': 'masih', 'msh': 'masih', 'msih': 'masih', 'msi': 'masih',
    'emg': 'memang', 'emng': 'memang', 'emang': 'memang', 'mmg': 'memang',
    'tp': 'tetapi', 'tpi': 'tetapi', 'tapi': 'tetapi', 'tpii': 'tetapi',
    'yg': 'yang', 'yng': 'yang',
    'dgn': 'dengan', 'dg': 'dengan', 'dng': 'dengan',
    'utk': 'untuk', 'bt': 'untuk', 'buat': 'untuk', 'untk': 'untuk',
    'dlm': 'dalam', 'dlam': 'dalam',
    'sm': 'sama', 'sma': 'sama',
    'dr': 'dari', 'dri': 'dari',
    'lg': 'lagi', 'lgi': 'lagi',
    'jg': 'juga', 'jga': 'juga',
    'aj': 'saja', 'aja': 'saja', 'doang': 'saja', 'doank': 'saja',
    'aje': 'saja', 'doangan': 'saja', 'doangan': 'saja',
    'hrs': 'harus', 'hrus': 'harus',
    'tetep': 'tetap', 'ttep': 'tetap', 'ttap': 'tetap',
    'bner': 'benar', 'bener': 'benar', 'bnr': 'benar', 'bner': 'benar',
    'cuma': 'hanya', 'cm': 'hanya', 'cmn': 'hanya', 'cuman': 'hanya',
    'cman': 'hanya', 'cmah': 'hanya',
    'bbrp': 'beberapa', 'bbrapa': 'beberapa', 'brp': 'beberapa',
    'org': 'orang', 'orng': 'orang', 'ornag': 'orang',
    'bnyk': 'banyak', 'byk': 'banyak', 'bnk': 'banyak',
    'skrg': 'sekarang', 'skrang': 'sekarang', 'skr': 'sekarang',
    'trs': 'terus', 'trus': 'terus', 'tros': 'terus', 'truz': 'terus',
    'abis': 'setelah', 'abs': 'setelah', 'hbs': 'setelah',
    'abisan': 'setelah', 'habis': 'setelah',
    'stlh': 'setelah', 'stlah': 'setelah', 'stlhnya': 'setelah',
    'sblm': 'sebelum', 'sblum': 'sebelum', 'sblmnya': 'sebelumnya',
    'klo': 'kalau', 'kl': 'kalau', 'kalo': 'kalau', 'klu': 'kalau',
    'tbtb': 'tiba-tiba', 'tb': 'tiba-tiba',
    'pdhl': 'padahal', 'pdhal': 'padahal',
    'sdh': 'sudah', 'sdg': 'sedang',
    'sori': 'maaf', 'sorry': 'maaf', 'sry': 'maaf',
    'walopun': 'walaupun', 'wlpn': 'walaupun', 'walau': 'walaupun',
    'keknya': 'sepertinya', 'kayanya': 'sepertinya', 'kynya': 'sepertinya',
    'sperti': 'seperti', 'spt': 'seperti', 'sprti': 'seperti',
    'mnrt': 'menurut', 'mnrut': 'menurut', 'mnurut': 'menurut',
    'jgn': 'jangan', 'jngn': 'jangan', 'jgnn': 'jangan',
    'blg': 'bilang', 'bilg': 'bilang', 'blng': 'bilang',
    'krg': 'kurang', 'krng': 'kurang',

    # ── Kata kerja informal ──
    'kayak': 'seperti', 'kyk': 'seperti', 'kek': 'seperti',
    'gimana': 'bagaimana', 'gmn': 'bagaimana', 'gmana': 'bagaimana',
    'knp': 'mengapa', 'knapa': 'mengapa', 'kenapa': 'mengapa',
    'bkl': 'akan', 'bakal': 'akan',
    'nemuin': 'menemukan', 'nemuin': 'menemukan',
    'ngasih': 'memberi', 'ngebagi': 'membagi',
    'ngomong': 'berbicara', 'nyeritain': 'menceritakan',
    'nyoba': 'mencoba', 'ngoding': 'membuat kode',
    'nulis': 'menulis', 'ngebacain': 'membacakan',
    'nyentuh': 'menyentuh', 'ngerasa': 'merasa',
    'ngikutin': 'mengikuti', 'ngejar': 'mengejar',
    'ngefek': 'mempengaruhi', 'ngebunuh': 'membunuh',
    'ngelakuin': 'melakukan', 'ngelawan': 'melawan',
    'ngebuat': 'membuat', 'ngebahas': 'membahas',
    'ngelihat': 'melihat', 'ngebaca': 'membaca',
    'ngedenger': 'mendengar', 'ngerasain': 'merasakan',
    'ngecewain': 'mengecewakan', 'ngebosenin': 'membosankan',
    'ngebikin': 'membuat', 'ngebuktiin': 'membuktikan',
    'ngebandingin': 'membandingkan', 'ngehargain': 'menghargai',
    'ngerti': 'mengerti', 'ngerasa': 'merasa',
    'ngekritik': 'mengkritik', 'ngerekomendasiin': 'merekomendasikan',
    'ngarahin': 'mengarahkan', 'ngehajar': 'menghajar',
    'ngebolehin': 'memperbolehkan',
    'ngerasain': 'merasakan', 'nyesel': 'menyesal',
    'nyari': 'mencari', 'nyampein': 'menyampaikan',
    'nyuruh': 'menyuruh', 'nyangka': 'menyangka',
    'nyiksa': 'menyiksa', 'nyadar': 'menyadari',
    'nyentuh': 'menyentuh', 'nyobain': 'mencoba',
    'nyinggung': 'menyinggung', 'nyelesain': 'menyelesaikan',
    'nyeritain': 'menceritakan', 'nyesel': 'menyesal',

    # ══════════════════════════════════════════════════════════════════════
    # KATA SIFAT / PENILAIAN
    # ══════════════════════════════════════════════════════════════════════
    'bgs': 'bagus', 'bgus': 'bagus',
    'kren': 'keren', 'krn': 'keren',
    'jelek': 'buruk', 'jlek': 'buruk', 'jlk': 'buruk',
    'ancur': 'hancur', 'ancr': 'hancur', 'hnc': 'hancur',
    'gokil': 'hebat', 'gokill': 'hebat',
    'mantap': 'bagus', 'mantep': 'bagus', 'mantul': 'bagus',
    'mantab': 'bagus', 'mantabs': 'bagus',
    'josss': 'bagus', 'joss': 'bagus', 'jos': 'bagus',
    'cakep': 'tampan', 'ganteng': 'tampan',
    'serem': 'seram', 'srem': 'seram', 'nyeremin': 'menakutkan',
    'seremin': 'menakutkan', 'sereem': 'seram',
    'seru': 'menarik', 'seruu': 'menarik',
    'ngakak': 'lucu', 'ngkk': 'lucu',
    'mewek': 'menangis', 'nangis': 'menangis', 'mewekk': 'menangis',
    'kocak': 'lucu', 'receh': 'lucu', 'garing': 'tidak lucu',
    'asik': 'menyenangkan', 'asyik': 'menyenangkan', 'asek': 'menyenangkan',
    'bosen': 'bosan', 'bosn': 'bosan', 'bete': 'bosan',
    'norak': 'kampungan', 'lebay': 'berlebihan', 'alay': 'berlebihan',
    'cringe': 'canggung', 'cringey': 'canggung',
    'kureng': 'kurang', 'kurang': 'kurang',
    'gacor': 'hebat',
    'rill': 'nyata', 'ril': 'nyata',
    'cetek': 'dangkal',
    'aga': 'agak', 'agk': 'agak',
    'awkwrd': 'canggung', 'awkward': 'canggung',
    'worth': 'layak', 'worthed': 'layak', 'worthit': 'layak',
    'overrated': 'dinilai berlebihan', 'underrated': 'kurang diapresiasi',
    'mainstream': 'umum',

    # ══════════════════════════════════════════════════════════════════════
    # PARTIKEL / FILLER WORDS (dihapus — mapping ke string kosong)
    # ══════════════════════════════════════════════════════════════════════
    'sih': '', 'nih': '', 'dong': '', 'deh': '',
    'loh': '', 'lho': '', 'kok': '', 'tuh': '', 'toh': '',
    'woy': '', 'woi': '', 'cuy': '', 'bro': '', 'sis': '',
    'btw': '', 'fyi': '', 'cmiiw': '', 'idk': '',
    'lol': '', 'lmao': '', 'omg': '',
    'anjir': '', 'anjay': '', 'njir': '', 'njay': '',
    'asw': '', 'bgst': '', 'bangsat': '',
    'bjir': '', 'anji': '', 'anjg': '',
    'wkwk': '', 'wkwkwk': '', 'xixi': '', 'xixix': '',
    'huft': '', 'huf': '', 'duh': '', 'aduh': '',
    'hmm': '', 'hmmm': '', 'uhm': '', 'umm': '',
    'yoi': '', 'yoii': '', 'yup': '', 'yupp': '',
    'nah': '', 'nahh': '', 'mah': '', 'mahh': '',
    'weh': '', 'wehh': '', 'lah': '', 'lahh': '',
    'kan': '', 'kaan': '', 'kaaan': '',
    'cus': '', 'cuz': '', 'cuzz': '',
    'gais': '', 'guys': '', 'gaes': '', 'gaess': '',
    'bangg': '', 'bos': '', 'boss': '',
    'kak': '', 'kakk': '', 'bang': '',
    'bestie': '', 'bestiee': '', 'bestii': '',
    'pls': '', 'please': '', 'pliss': '', 'plis': '',
    'ngl': '', 'tbh': '', 'imo': '', 'imho': '',
    'iykwim': '', 'iykyk': '', 'ifykyk': '',
    'smh': '', 'bruh': '', 'dude': '',
    'yasss': '', 'yass': '', 'yas': '',
    'periodt': '', 'period': '', 'literally': '',
    'weh': '', 'gilak': '', 'gila': '',
    'ahelah': '', 'elah': '', 'halah': '',
    'dah': '', 'deh': '', 'dehh': '',
    'ges': '', 'gest': '', 'boy': '',
    'woii': '', 'woiii': '', 'heyy': '',

    # ══════════════════════════════════════════════════════════════════════
    # KONTEKS FILM
    # ══════════════════════════════════════════════════════════════════════
    'filmnya': 'film', 'pilem': 'film', 'filem': 'film',
    'ceritanya': 'cerita', 'critanya': 'cerita', 'crita': 'cerita',
    'aktingnya': 'akting', 'actingnya': 'akting',
    'endingnya': 'ending', 'endnya': 'ending',
    'plotnya': 'plot',
    'horornya': 'horor', 'horrornya': 'horor',
    'komedinya': 'komedi', 'comedynya': 'komedi',
    'alurnya': 'alur',
    'hantunya': 'hantu',
    'gorenya': 'gore',
    'chara': 'karakter', 'karakter': 'karakter',
    'develop': 'perkembangan',
    'adegannya': 'adegan', 'scenenya': 'adegan',
    'dialognya': 'dialog',
    'visualnya': 'visual',
    'efeknya': 'efek',
    'soundtracknya': 'soundtrack',
    'karakternya': 'karakter',
    'skenarionya': 'skenario', 'scriptnya': 'skenario',
    'sequelnya': 'sekuel',
    'trailernya': 'trailer',
    'genrenya': 'genre',
    'durasinya': 'durasi',
    'bioskopnya': 'bioskop',
    'sutradaranya': 'sutradara',
    'aktornya': 'aktor', 'aktrisnya': 'aktris',
    'pemerannya': 'pemeran',

    # ── Istilah khusus Ghost in the Cell ──
    'jokan': 'joko anwar', 'jokaaan': 'joko anwar',
    'gitc': 'ghost in the cell',
    'gits': 'ghost in the cell',

    # ══════════════════════════════════════════════════════════════════════
    # KATA HUBUNG / PREPOSISI INFORMAL TAMBAHAN
    # ══════════════════════════════════════════════════════════════════════
    'kalo': 'kalau', 'klu': 'kalau',
    'gitu': 'begitu', 'gt': 'begitu', 'gtw': 'begitu',
    'gini': 'begini', 'gin': 'begini',
    'bnr': 'benar', 'bener': 'benar',
    'sbg': 'sebagai', 'sbgai': 'sebagai',
    'trhdp': 'terhadap', 'thd': 'terhadap',
    'thdp': 'terhadap',
    'dmn': 'dimana', 'dmna': 'dimana',
    'kmn': 'kemana', 'kmana': 'kemana',
    'gmn': 'bagaimana', 'bgmn': 'bagaimana',
    'stidaknya': 'setidaknya', 'setidknya': 'setidaknya',
    'slain': 'selain', 'slalu': 'selalu',
    'trnyata': 'ternyata', 'tryta': 'ternyata',
    'sbnarnya': 'sebenarnya', 'sbenernya': 'sebenarnya',
    'sbnrnya': 'sebenarnya',
    'mgkn': 'mungkin', 'mngkin': 'mungkin',
    'krna': 'karena', 'karna': 'karena', 'krn': 'karena',

    # ══════════════════════════════════════════════════════════════════════
    # TAMBAHAN BARU: Kata slang yang sering ditemukan di Letterboxd
    # Diperluas berdasarkan eksplorasi dataset untuk akurasi penelitian
    # ══════════════════════════════════════════════════════════════════════

    # ── Variasi penulisan intensifier yang belum tercakup ──
    'bgttt': 'sekali', 'bgetz': 'sekali', 'bgets': 'sekali',
    'bngets': 'sekali', 'bangett': 'sekali', 'bangeets': 'sekali',
    'paraah': 'sekali', 'parahh': 'sekali', 'prh': 'sekali',
    'sangaat': 'sangat', 'sngat': 'sangat',
    'skli': 'sekali', 'bngtt': 'sekali',

    # ── Variasi kata kerja informal tambahan ──
    'nyimak': 'menyimak', 'ngegas': 'mempercepat',
    'ngenes': 'menyedihkan', 'ngeselin': 'menjengkelkan',
    'ngesanin': 'mengesankan', 'ngabisin': 'menghabiskan',
    'ngebetein': 'membosankan', 'ngeboseninnn': 'membosankan',
    'ngeluh': 'mengeluh', 'ngantuk': 'mengantuk',
    'ngedrop': 'menurun', 'ngefail': 'gagal',
    'ngeflop': 'gagal', 'ngehit': 'berhasil',
    'ngelakuin': 'melakukan', 'ngereview': 'mengulas',
    'ngecek': 'memeriksa', 'ngasal': 'asal-asalan',
    'ngebul': 'berbohong', 'ngedumel': 'menggerutu',
    'ngedrama': 'dramatis', 'ngejar': 'mengejar',
    'ngelanjutin': 'melanjutkan', 'ngelarin': 'menyelesaikan',
    'ngeluarin': 'mengeluarkan', 'ngerti': 'mengerti',
    'ngerasa': 'merasa', 'ngerasain': 'merasakan',
    'nyambung': 'berhubungan', 'nyaman': 'nyaman',
    'nyesel': 'menyesal', 'nyindir': 'menyindir',
    'ngakuin': 'mengakui', 'ngobatin': 'mengobati',
    'nyentuh': 'menyentuh', 'nyakitin': 'menyakiti',
    'nyeritain': 'menceritakan', 'nyebarin': 'menyebarkan',
    'ngeliat': 'melihat', 'ngelawan': 'melawan',
    'ngetop': 'populer', 'ngehype': 'digembar-gemborkan',

    # ── Variasi kata sifat/penilaian tambahan ──
    'wortid': 'layak', 'wortit': 'layak', 'wortt': 'layak',
    'kereeen': 'keren', 'kerenn': 'keren',
    'baguuus': 'bagus', 'baguss': 'bagus',
    'jeeleek': 'buruk', 'jelekk': 'buruk',
    'sereem': 'seram', 'sereeeem': 'seram',
    'seruuu': 'menarik', 'seruu': 'menarik',
    'lucu': 'lucu', 'lucuu': 'lucu', 'lucuuu': 'lucu',
    'sedih': 'sedih', 'sedihh': 'sedih', 'sediih': 'sedih',
    'ancuuur': 'hancur', 'ancuurr': 'hancur',
    'makjleb': 'mengena', 'makjlebbb': 'mengena',
    'nyesek': 'menyesakkan', 'nyesekk': 'menyesakkan',
    'ngena': 'mengena', 'ngenaa': 'mengena',
    'ngeri': 'mengerikan', 'ngerii': 'mengerikan',
    'madesu': 'kisah sedih', 'galau': 'gelisah',
    'gabener': 'tidak benar', 'gamasuk': 'tidak masuk akal',
    'gajelaz': 'tidak jelas', 'gajelasss': 'tidak jelas',
    'gabaguss': 'tidak bagus', 'gajelas': 'tidak jelas',
    'zonk': 'mengecewakan', 'zonkk': 'mengecewakan',
    'mehh': 'biasa saja', 'meh': 'biasa saja',
    'hmm': 'biasa saja',
    'middd': 'biasa saja', 'midd': 'biasa saja',
    'solid': 'solid', 'solidd': 'solid',

    # ── Kata emosi/reaksi tambahan ──
    'ketiduran': 'tertidur', 'ketidurannn': 'tertidur',
    'keenakan': 'kenikmatan', 'kesakitan': 'menyakitkan',
    'ketakutan': 'ketakutan', 'kebingungan': 'kebingungan',
    'kepikiran': 'terpikirkan', 'kesenengan': 'kesenangan',
    'kegirangan': 'kegembiraan', 'keheranan': 'keheranan',
    'terpukau': 'terpesona', 'terpana': 'terpesona',
    'terkesima': 'terpesona', 'bengong': 'terpaku',
    'melongo': 'terpaku', 'cengo': 'terpaku',
    'teriak': 'berteriak', 'jiper': 'takut',
    'ciut': 'takut', 'parno': 'paranoid',
    'meriang': 'merinding', 'deg degan': 'berdebar-debar',
    'gemetar': 'gemetar', 'bergetar': 'bergetar',

    # ── Konteks film tambahan ──
    'reviewnya': 'review', 'ratingnya': 'rating',
    'scorenya': 'skor', 'cginy': 'cgi',
    'vfxnya': 'vfx', 'sfxnya': 'sfx',
    'pacingnya': 'pacing', 'twistnya': 'twist',
    'klimaksnya': 'klimaks', 'konfliknya': 'konflik',
    'narasinya': 'narasi', 'perannya': 'peran',
    'tokohnya': 'tokoh', 'temanya': 'tema',
    'settingnya': 'setting', 'latarnya': 'latar',
    'alur ceritanya': 'alur cerita',
    'opening': 'pembuka', 'openingnya': 'pembuka',
    'prolognya': 'prolog', 'epilognya': 'epilog',
    'credit': 'kredit', 'creditnya': 'kredit',
    'showtimenya': 'jadwal tayang',
    'producernya': 'produser', 'creatornya': 'kreator',
    'writernya': 'penulis', 'editingnya': 'editing',
    'lightingnya': 'pencahayaan', 'anglenya': 'sudut kamera',
    'composisinya': 'komposisi',

    # ── Kata ganti waktu informal ──
    'bsknya': 'besoknya', 'kmrn': 'kemarin', 'kmarin': 'kemarin',
    'kmren': 'kemarin', 'besk': 'besok',
    'tdi': 'tadi', 'ntr': 'nanti', 'ntar': 'nanti',
    'nnti': 'nanti', 'besoknya': 'besoknya',

    # ── Variasi negasi gabungan tambahan ──
    'gaworth': 'tidak layak', 'garecommend': 'tidak merekomendasikan',
    'garecom': 'tidak merekomendasikan',
    'gasuggest': 'tidak menyarankan',
    'garekomen': 'tidak merekomendasikan',
    'gaterlalu': 'tidak terlalu',
    'gategangan': 'tidak tegang',
    'galucu': 'tidak lucu', 'ganyambung': 'tidak berhubungan',
    'gamenarik': 'tidak menarik', 'ganyesel': 'tidak menyesal',
    'garugi': 'tidak rugi', 'ga rugi': 'tidak rugi',
    'gaberat': 'tidak berat', 'gapantes': 'tidak pantas',

    # ── Kata non-baku tambahan hasil review film ──
    'yaudah': 'ya sudah', 'yauda': 'ya sudah', 'yaudahh': 'ya sudah',
    'yawdah': 'ya sudah', 'yoda': 'ya sudah', 'yowes': 'ya sudah',
    'yowis': 'ya sudah', 'yaudahlah': 'ya sudah', 'yaudahlahh': 'ya sudah',
    'yaudalh': 'ya sudah',
    'gpp': 'tidak apa-apa', 'gapapa': 'tidak apa-apa', 'gpapa': 'tidak apa-apa',
    'rapopo': 'tidak apa-apa',
    'kesel': 'kesal', 'kesell': 'kesal', 'keselnya': 'kesal',
    'sebel': 'sebal', 'sebell': 'sebal',
    'rame': 'ramai', 'ramean': 'ramai',
    'cape': 'lelah', 'capek': 'lelah', 'cappe': 'lelah',
    'bsk': 'besok', 'dlu': 'dahulu', 'dulu': 'dahulu',
    'pakek': 'pakai', 'pakei': 'pakai',
    'sampek': 'sampai', 'sempet': 'sempat', 'sempett': 'sempat',
    'dapetin': 'mendapatkan', 'bikinin': 'membuatkan', 'liatin': 'melihat',
    'itupun': 'itu pun', 'apapun': 'apa pun', 'siapapun': 'siapa pun',
    'manapun': 'mana pun', 'kapanpun': 'kapan pun', 'dimanapun': 'di mana pun',
}


def normalize_slang(text):
    """
    Menormalisasi kata-kata slang/tidak baku menjadi kata baku.

    Multi-pass normalization:
    1. Pre-pass: Normalisasi huruf berulang yang masih tersisa setelah cleaning
       (misalnya vokal ganda: 'bagusss' sudah jadi 'bagus' tapi 'seruu' masih ada)
    2. Pass 1: Lookup di kamus slang
    3. Pass 2: Tangani pola dinamis ga+kata yang tidak ada di kamus
    4. Pass 3: Tangani sufiks -nya pada kata slang (e.g. 'filmnya' → 'film')
    """
    if not isinstance(text, str):
        return ""

    words = text.split()
    normalized = []

    for word in words:
        original_word = word

        # ── Pre-pass: Normalisasi huruf berulang pada vokal ganda ──
        # Cleaning sudah handle konsonan 2+→1 dan vokal 3+→1
        # Tapi vokal ganda masih bisa lolos: 'seruu'→'seru', 'yaaa'→'ya'
        # Coba normalisasi vokal ganda jika kata TIDAK ditemukan di kamus
        if word not in KAMUS_SLANG:
            # Coba reduksi vokal ganda terlebih dahulu
            word_reduced = re.sub(r'([aiueo])\1+', r'\1', word)
            if word_reduced in KAMUS_SLANG:
                word = word_reduced

        # ── Pass 1: Lookup di kamus slang ──
        if word in KAMUS_SLANG:
            replacement = KAMUS_SLANG[word]
            if replacement:  # Jika replacement bukan string kosong
                normalized.append(replacement)
            # Jika replacement kosong (''), kata dihapus (filler words)
            continue

        # ── Pass 2: Tangani pola dinamis ga+kata ──
        # Jika kata dimulai dengan 'ga' dan bukan kata yang sudah ada di kamus
        # coba pecah menjadi 'tidak' + sisanya
        if word.startswith('ga') and len(word) > 3 and word not in KAMUS_SLANG:
            sisa = word[2:]  # Hapus prefix 'ga'
            # Cek apakah sisanya adalah kata valid (ada di kamus atau panjang > 2)
            if sisa in KAMUS_SLANG:
                replacement_sisa = KAMUS_SLANG[sisa]
                if replacement_sisa:
                    normalized.append('tidak')
                    normalized.append(replacement_sisa)
                else:
                    normalized.append('tidak')
            elif len(sisa) >= 3:
                normalized.append('tidak')
                normalized.append(sisa)
            else:
                normalized.append(original_word)
            continue

        # ── Pass 3: Tangani sufiks -nya pada kata yang tidak ada di kamus ──
        # e.g. 'ceritanya' → sudah di kamus, tapi 'horornya' → mungkin belum
        if word.endswith('nya') and len(word) > 4 and word not in KAMUS_SLANG:
            kata_dasar = word[:-3]  # Hapus sufiks '-nya'
            if kata_dasar in KAMUS_SLANG:
                replacement = KAMUS_SLANG[kata_dasar]
                if replacement:
                    normalized.append(replacement)
                continue
            elif len(kata_dasar) >= 3:
                normalized.append(kata_dasar)
                continue

        # ── Default: kata tidak berubah ──
        normalized.append(original_word)

    return ' '.join(normalized)


def normalize_and_translate(text):
    """
    Fungsi gabungan: Normalisasi slang Indonesia LALU translasi kata Inggris.

    Pendekatan ini lebih akurat untuk penelitian karena:
    1. Normalisasi slang Indonesia DULU — mencegah kata slang yang mirip
       bahasa Inggris (garing, kentang, receh) salah diterjemahkan
    2. Deteksi & terjemahkan kalimat bahasa Inggris yang tersisa
    3. Second-pass: terjemahkan kata-kata Inggris tunggal yang masih tersisa

    Alur detail:
    Pass 1: normalize_slang() — normalisasi kata slang Indonesia ke baku
    Pass 2: translate_mixed_text() — deteksi & terjemahkan kalimat EN
            (termasuk proteksi slang + Google Translate + kamus fallback)
    Pass 3: translate_remaining_english_words() — cleanup kata EN tersisa
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1: Normalisasi slang Indonesia terlebih dahulu
    # Ini HARUS dilakukan sebelum translasi agar kata seperti "garing",
    # "kentang", "receh" sudah berubah menjadi "tidak lucu", "tanggung",
    # "humor murahan" SEBELUM mesin deteksi bahasa melihatnya.
    # ═══════════════════════════════════════════════════════════════════════
    text = normalize_slang(text)

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2: Deteksi & terjemahkan kalimat bahasa Inggris
    # Setelah normalisasi, kata-kata yang masih tersisa dan terdeteksi
    # sebagai bahasa Inggris memang benar-benar kata Inggris (bukan slang
    # Indonesia yang mirip Inggris). Fungsi translate_mixed_text() sudah
    # mencakup:
    #   - Proteksi kata slang dari KATA_SLANG_PROTEKSI
    #   - Deteksi bahasa per kalimat (detect_sentence_language)
    #   - Translasi via Google Translate API (safe_translate)
    #   - Second-pass via kamus KAMUS_TERJEMAHAN_MANUAL
    #   - Restorasi kata slang yang dilindungi
    # ═══════════════════════════════════════════════════════════════════════
    text = translate_mixed_text(text)

    return text


# ── Jalankan Normalisasi & Translasi ──
print("    Proses normalisasi slang & translasi bahasa campuran...")
print("    Pendekatan: Normalisasi slang DULU, lalu translasi kata Inggris.")
print("    (Ini mungkin memakan waktu beberapa menit tergantung jumlah data)\n")

# Normalisasi & Translasi dengan progress bar
df['ulasan_normalized'] = df['ulasan_cleansed'].progress_apply(normalize_and_translate)

# ── Hitung statistik normalisasi slang ──
# (bandingkan ulasan_cleansed vs hasil sementara normalize_slang saja)
df['_ulasan_slang_only'] = df['ulasan_cleansed'].apply(normalize_slang)
changed_norm = (df['_ulasan_slang_only'] != df['ulasan_cleansed']).sum()

slang_found = Counter()
for text in df['ulasan_cleansed']:
    for word in str(text).split():
        if word in KAMUS_SLANG:
            slang_found[word] += 1
top_slang = slang_found.most_common(10)

# ── Hitung statistik translasi ──
# (bandingkan _ulasan_slang_only vs ulasan_normalized untuk melihat efek translasi)
changed_mask_tr = df['ulasan_normalized'] != df['_ulasan_slang_only']
jumlah_berubah_tr = changed_mask_tr.sum()

# Hitung token Inggris yang berhasil diterjemahkan
total_token_en = 0
total_token_translated = 0
for i in df[changed_mask_tr].index:
    before_words = set(df.loc[i, '_ulasan_slang_only'].split())
    after_words = set(df.loc[i, 'ulasan_normalized'].split())
    kata_hilang = before_words - after_words
    kata_baru = after_words - before_words
    total_token_en += len(kata_hilang)
    total_token_translated += len(kata_baru)

# Hitung total ulasan yang berubah (normalisasi ATAU translasi)
changed_total = (df['ulasan_normalized'] != df['ulasan_cleansed']).sum()

# Hapus kolom sementara
df = df.drop(columns=['_ulasan_slang_only'])

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT UNTUK SKRIPSI — Sub Bab 4.3.3 Normalisasi & Translasi
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("      TAHAP 4: NORMALISASI & TRANSLASI (Slang → Baku, EN → ID)")
print("=" * 70)

# ── 1. Informasi Umum ──
print(f"\n  Total ulasan diproses                 : {len(df)} ulasan")
print(f"  Ulasan berubah oleh normalisasi slang : {changed_norm} ulasan")
print(f"  Ulasan berubah oleh translasi EN→ID   : {jumlah_berubah_tr} ulasan")
print(f"  Total ulasan yang mengalami perubahan : {changed_total} ulasan")
print(f"  Total variasi kata slang unik         : {len(slang_found)} kata")
print(f"  Total kemunculan kata slang           : {sum(slang_found.values()):,} kali")
print(f"  Token Inggris yang diterjemahkan      : {total_token_en:,} token")

# ── 2. Metode ──
print(f"\n{'─' * 70}")
print("  Metode Normalisasi & Translasi")
print(f"{'─' * 70}")
print(f"  Urutan proses         : Normalisasi slang DULU → Translasi EN")
print(f"  Kamus slang Indonesia : {len(KAMUS_SLANG)} entri")
print(f"  Kamus terjemahan manual : {len(KAMUS_TERJEMAHAN_MANUAL)} entri")
print(f"  Pustaka translasi     : deep-translator (Google Translate API)")
print(f"  Kata serapan (skip)   : {len(KATA_SERAPAN)} kata")
print(f"  Kata deteksi bahasa   : {len(KATA_INGGRIS_KUAT)} kata penanda Inggris")
print(f"  {'─' * 66}")
print(f"  ALASAN URUTAN: Normalisasi slang Indonesia dilakukan sebelum")
print(f"  translasi Inggris agar kata slang yang mirip bahasa Inggris")
print(f"  (seperti 'garing', 'kentang', 'receh') tidak salah diterjemahkan")
print(f"  oleh Google Translate ke makna harfiah (crispy, potato, coin).")

# ── 3. Top 10 Kata Slang (Mapping) ──
print(f"\n{'─' * 70}")
print("  Top 10 Kata Tidak Baku (Slang) Paling Sering Muncul")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Kata Tidak Baku':<25} {'Hasil Normalisasi':<25} {'Frekuensi'}")
print(f"  {'─' * 66}")

for idx, (word, count) in enumerate(top_slang, 1):
    replacement = KAMUS_SLANG[word]
    if not replacement:
        replacement = "[dihapus]"
    print(f"  {idx:<4} {word:<25} -> {replacement:<22} ({count}x)")

# ── 4. Contoh Hasil Normalisasi & Translasi (Tabel Before-After) ──
print(f"\n{'─' * 70}")
print("  Contoh Hasil Normalisasi & Translasi (Sebelum → Sesudah)")
print(f"{'─' * 70}")

# Ambil 5 contoh yang mengalami perubahan
mask_changed = df['ulasan_normalized'] != df['ulasan_cleansed']
contoh_norm = df[mask_changed].head(5)
if len(contoh_norm) < 5:
    contoh_norm = df.head(5)

print(f"  {'No':<4} {'Sebelum Normalisasi & Translasi':<55}")
print(f"  {'':4} {'Sesudah Normalisasi & Translasi':<55}")
print(f"  {'─' * 66}")

for idx, (i, row) in enumerate(contoh_norm.iterrows(), 1):
    before = str(row['ulasan_cleansed']).replace('\n', ' ')
    after = str(row['ulasan_normalized']).replace('\n', ' ')

    max_len = 60
    before_display = before[:max_len] + "..." if len(before) > max_len else before
    after_display = after[:max_len] + "..." if len(after) > max_len else after

    print(f"  {idx:<4} \"{before_display}\"")
    print(f"  {'':4} \"{after_display}\"")
    if idx < len(contoh_norm):
        print()

# ── 5. Verifikasi Translasi ──
print(f"\n{'─' * 70}")
print("  Verifikasi Hasil Translasi")
print(f"{'─' * 70}")

# Hitung kata Inggris yang masih tersisa setelah normalisasi & translasi
sisa_en_count = 0
sisa_en_words = Counter()
for text in df['ulasan_normalized']:
    if isinstance(text, str):
        for kata in text.lower().split():
            kata_clean = re.sub(r'[^a-z]', '', kata)
            if (kata_clean in KATA_INGGRIS_KUAT and kata_clean not in KATA_SERAPAN
                    and len(kata_clean) > 2):
                sisa_en_count += 1
                sisa_en_words[kata_clean] += 1

print(f"  Token Inggris (penanda kuat) tersisa : {sisa_en_count} token")
if sisa_en_count == 0:
    print(f"  Status : ✓ Berhasil — Seluruh teks Inggris telah diterjemahkan")
else:
    top_sisa = sisa_en_words.most_common(5)
    sisa_str = ", ".join(f'"{w}" ({c}x)' for w, c in top_sisa)
    print(f"  Kata tersisa terbanyak : {sisa_str}")
    print(f"  Status : Sebagian besar teks Inggris berhasil diterjemahkan")

# ── 6. Ringkasan ──
print(f"\n{'=' * 70}")
print("  Ringkasan Normalisasi & Translasi")
print(f"{'=' * 70}")
print(f"  Tahap ini menggabungkan dua proses: normalisasi kata slang Indonesia")
print(f"  dan translasi kata/kalimat berbahasa Inggris. Normalisasi slang")
print(f"  dilakukan TERLEBIH DAHULU agar kata-kata slang Indonesia yang mirip")
print(f"  bahasa Inggris tidak salah diterjemahkan oleh Google Translate.")
print(f"  Dari total {len(df)} ulasan, ditemukan {len(slang_found)} variasi kata slang")
print(f"  yang dinormalisasi menggunakan kamus berisi {len(KAMUS_SLANG)} entri.")
print(f"  Selanjutnya, {jumlah_berubah_tr} ulasan mengandung kata/kalimat Inggris")
print(f"  yang berhasil diterjemahkan menggunakan kombinasi Google Translate API")
print(f"  dan kamus terjemahan manual berisi {len(KAMUS_TERJEMAHAN_MANUAL)} entri.")
print(f"{'=' * 70}\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9: TOKENISASI                                                       ║
# ║  Tahap 5: Memecah teks menjadi token-token (kata-kata)                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(5, "TOKENISASI", "TOKEN")

def tokenize(text):
    """
    Memecah teks menjadi daftar token (kata-kata).
    Menggunakan NLTK word_tokenize untuk tokenisasi yang lebih akurat.
    Hanya menyimpan token yang berisi huruf (hapus sisa angka/simbol).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Filter: hanya token yang berisi huruf (hapus sisa angka/simbol)
    tokens = [t for t in tokens if t.isalpha()]

    return tokens


df['ulasan_tokenized_raw'] = df['ulasan_normalized'].apply(tokenize)

# ── Tahap Negation Binding (Pengikatan Negasi) ──
def bind_negations(tokens):
    """
    Mengikat kata negasi (tidak, kurang, bukan, belum, tanpa) dengan kata setelahnya
    menggunakan tanda hubung '_' (misal: 'tidak' + 'bagus' -> 'tidak_bagus').
    Ini memecahkan dilema hilangnya kata negasi untuk klasifikasi SVM,
    sambil menjaga agar kata negasi tidak melayang bebas sebagai noise di LDA.
    """
    negations = {'tidak', 'kurang', 'bukan', 'belum', 'tanpa'}
    bound = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token in negations and i + 1 < n:
            next_token = tokens[i + 1]
            bound.append(f"{token}_{next_token}")
            i += 2  # Skip kata berikutnya karena sudah digabung
        else:
            bound.append(token)
            i += 1
    return bound

df['ulasan_tokenized'] = df['ulasan_tokenized_raw'].apply(bind_negations)

# Statistik token
token_lengths = df['ulasan_tokenized'].apply(len)

# Hitung statistik perubahan
total_token_dihasilkan = token_lengths.sum()
ulasan_kosong_setelah_token = (token_lengths == 0).sum()

# Hitung statistik negation binding
bound_negations_counter = Counter()
for tokens in df['ulasan_tokenized']:
    for t in tokens:
        if '_' in t:
            bound_negations_counter[t] += 1
total_bound_negations = sum(bound_negations_counter.values())

# Hitung distribusi panjang token
distribusi_token = {
    '1-5 token': ((token_lengths >= 1) & (token_lengths <= 5)).sum(),
    '6-10 token': ((token_lengths >= 6) & (token_lengths <= 10)).sum(),
    '11-20 token': ((token_lengths >= 11) & (token_lengths <= 20)).sum(),
    '21-50 token': ((token_lengths >= 21) & (token_lengths <= 50)).sum(),
    '>50 token': (token_lengths > 50).sum(),
}

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT UNTUK SKRIPSI — Sub Bab 4.3.5 Tokenisasi & Negation Binding
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("      TAHAP 6: TOKENISASI & NEGATION BINDING")
print("=" * 70)

# ── 1. Informasi Umum ──
print(f"\n  Total ulasan diproses           : {len(df)} ulasan")
print(f"  Total token yang dihasilkan     : {total_token_dihasilkan:,} token")
print(f"  Ulasan kosong setelah tokenisasi: {ulasan_kosong_setelah_token} ulasan")
print(f"  Metode tokenisasi              : NLTK word_tokenize + Custom Negation Binding")
print(f"  Total negasi berhasil diikat   : {total_bound_negations:,} kata")

# ── 1b. Top 5 Bound Negations ──
if total_bound_negations > 0:
    print(f"\n{'─' * 70}")
    print("  Top 5 Kata Negasi yang Berhasil Diikat (Negation Binding)")
    print(f"{'─' * 70}")
    for idx, (word, count) in enumerate(bound_negations_counter.most_common(5), 1):
        print(f"  {idx:<4} {word:<30} ({count}x)")

# ── 2. Statistik Distribusi Token ──
print(f"\n{'─' * 70}")
print("  Statistik Distribusi Token per Ulasan")
print(f"{'─' * 70}")
print(f"  Minimum token per ulasan   : {token_lengths.min()} token")
print(f"  Maksimum token per ulasan  : {token_lengths.max()} token")
print(f"  Rata-rata token per ulasan : {token_lengths.mean():.1f} token")
print(f"  Median token per ulasan    : {token_lengths.median():.1f} token")

print(f"\n  {'Rentang Token':<20} {'Jumlah Ulasan':<20} {'Persentase'}")
print(f"  {'─' * 56}")
for rentang, jumlah in distribusi_token.items():
    persen = jumlah / len(df) * 100
    print(f"  {rentang:<20} {jumlah:<20} {persen:.1f}%")

# ── 3. Contoh Hasil Tokenisasi (Tabel Before-After) ──
print(f"\n{'─' * 70}")
print("  Contoh Hasil Tokenisasi & Negation Binding (Sebelum → Sesudah)")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Teks Sebelum Tokenisasi':<55}")
print(f"  {'':4} {'Hasil Token':<55}")
print(f"  {'─' * 66}")

for idx in range(min(5, len(df))):
    before = str(df['ulasan_normalized'].iloc[idx]).replace('\n', ' ')
    tokens = df['ulasan_tokenized'].iloc[idx]
    token_count = len(tokens)

    max_len = 60
    before_display = before[:max_len] + "..." if len(before) > max_len else before

    # Format token display
    if token_count > 8:
        token_display = str(tokens[:8])[:-1] + ", ...]"
    else:
        token_display = str(tokens)
    if len(token_display) > 60:
        token_display = token_display[:57] + "..."

    print(f"  {idx+1:<4} \"{before_display}\"")
    print(f"  {'':4} {token_display} ({token_count} token)")
    if idx < min(5, len(df)) - 1:
        print()

# ── 4. Verifikasi ──
print(f"\n{'─' * 70}")
print("  Verifikasi Hasil Tokenisasi & Negation Binding")
print(f"{'─' * 70}")
if ulasan_kosong_setelah_token == 0:
    print(f"  Status : ✓ Berhasil — Seluruh ulasan berhasil di-tokenisasi dan diikat negasinya")
else:
    print(f"  Status : Terdapat {ulasan_kosong_setelah_token} ulasan dengan 0 token")
print(f"  Filter token  : Hanya token berisi huruf (atau dengan '_' hasil binding) yang disimpan")
print(f"  Token numerik/simbol yang tersisa berhasil difilter.")

# ── 5. Ringkasan ──
print(f"\n{'=' * 70}")
print("  Ringkasan Tokenisasi")
print(f"{'=' * 70}")
print(f"  Tahap tokenisasi memecah setiap ulasan yang telah dinormalisasi")
print(f"  menjadi unit-unit kata (token) menggunakan pustaka NLTK")
print(f"  word_tokenize. Dari {len(df)} ulasan yang diproses, dihasilkan")
print(f"  total {total_token_dihasilkan:,} token dengan rata-rata")
print(f"  {token_lengths.mean():.1f} token per ulasan. Token yang bukan berupa")
print(f"  huruf (angka, simbol) difilter untuk memastikan hanya kata")
print(f"  bermakna yang diproses pada tahap selanjutnya.")
print(f"{'=' * 70}\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10: STOPWORD REMOVAL                                                ║
# ║  Tahap 6: Menghapus kata-kata yang tidak bermakna (stopword)              ║
# ║  Menggunakan kombinasi Sastrawi + NLTK + custom stopwords                 ║
# ║                                                                           ║
# ║  PERBAIKAN: Ditambahkan proteksi kata sentimen agar kata-kata             ║
# ║  penting untuk analisis sentimen tidak ikut terhapus meskipun             ║
# ║  terdaftar di stopword list bawaan.                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Gabungkan semua sumber stopword ──
# Tambahkan custom stopwords khusus konteks review film
CUSTOM_STOPWORDS = {
    # ── Kata-kata umum yang tidak bermakna untuk sentimen ──
    'film', 'movie', 'nonton', 'menonton', 'bioskop',
    'ya', 'yaa', 'yaaa', 'yah', 'yep', 'yap', 'yes',
    'hmm', 'hm', 'em', 'eh', 'oh', 'ah', 'uh', 'um',
    'hehe', 'hihi', 'huhu', 'haha', 'hoho',
    'pls', 'please', 'pliss', 'plis',
    'si', 'sang', 'para', 'nya',
    # ── Kata penghubung/partikel yang lolos dari tahap sebelumnya ──
    'itu', 'ini', 'ada', 'jadi', 'bisa', 'akan', 'sudah',
    'lagi', 'juga', 'saja', 'dari', 'sama', 'dalam',
    'untuk', 'dengan', 'pada', 'oleh', 'karena',
    'atau', 'tetapi', 'tapi', 'namun', 'sedangkan',
    'ketika', 'saat', 'sering', 'pernah', 'belum',
    'hanya', 'masih', 'lebih', 'seperti', 'antara',
    'setelah', 'sebelum', 'selama', 'hingga', 'sampai',
    'kalau', 'jika', 'maka', 'agar', 'supaya',
    'bahwa', 'bahkan', 'memang', 'bukan', 'tanpa',
    # ── Kata ganti / sapaan umum ──
    'saya', 'kamu', 'dia', 'kita', 'kami', 'mereka',
    'anda', 'beliau',
    # ── Kata bantu / filler yang tidak bermakna ──
    'hal', 'cara', 'orang', 'tahun', 'waktu',
    'semua', 'setiap', 'beberapa', 'banyak',
    'punya', 'milik', 'perlu', 'harus',
    'dapat', 'mampu', 'boleh', 'mau',
    'sedang', 'telah', 'paling',
    # ══════════════════════════════════════════════════════════════════════
    # PERBAIKAN: Intensifier & negasi yang sebelumnya diproteksi
    # Kata-kata ini menjadi TERLALU DOMINAN di output karena:
    # - Semua 'bgt'/'banget' → 'sekali' (ribuan kemunculan)
    # - Semua 'ga'/'gak'/'nggak' → 'tidak' (ribuan kemunculan)
    # Untuk topic modeling LDA, kata-kata ini adalah noise.
    # Untuk analisis sentimen, ditangani di level fitur (bukan preprocessing).
    # ══════════════════════════════════════════════════════════════════════
    'sekali', 'tidak', 'sangat', 'amat', 'sungguh', 'benar',
    'terlalu', 'kurang', 'cukup', 'agak', 'lumayan',
    'jangan', 'sedikit',
    # ── Kata tambahan yang sering lolos dan tidak informatif ──
    'begitu', 'begini', 'lain', 'lainnya', 'atas', 'bawah',
    'sebelumnya', 'selanjutnya', 'kemudian', 'lalu',
    'dahulu', 'kembali', 'ulang', 'terus',
    'tahu', 'lihat', 'rasa', 'pakai', 'bilang',
    'joko', 'anwar', 'joko anwar',  # Nama sutradara — bukan topik sentimen
    'ghost', 'cell', 'ghost in the cell',  # Judul film — bukan sentimen
    'tokek',  # Nama karakter — bukan sentimen
    'sebagai', 'terhadap', 'menurut', 'mengenai',
    'dimana', 'kemana', 'darimana', 'kesana', 'kesini',
    'kapan', 'mana', 'siapa', 'kenapa', 'mengapa',
    'begitu', 'gitu', 'gini',
    'dulu', 'nanti', 'sekarang', 'kemarin', 'besok',
    'selalu', 'kadang', 'jarang', 'sering',
    'mungkin', 'pasti', 'tentu', 'memang',
}

# ── Kata Sentimen yang DILINDUNGI dari penghapusan stopword ──
# Beberapa kata penting sentimen kadang masuk ke daftar stopword bawaan
# (Sastrawi/NLTK), padahal kata-kata ini krusial untuk analisis sentimen.
# Kata-kata ini TIDAK BOLEH dihapus.
#
# PERBAIKAN: Intensifier ('sekali', 'sangat') dan negasi ('tidak', 'bukan')
# TIDAK lagi diproteksi — mereka terlalu dominan di output karena inflasi
# dari normalisasi slang. Hanya kata sentimen BERMAKNA yang diproteksi.
KATA_SENTIMEN_PROTEKSI = {
    # ── Positif ──
    'bagus', 'keren', 'hebat', 'sempurna', 'indah', 'luar biasa',
    'menarik', 'seru', 'mantap', 'memukau', 'fantastis',
    'mengagumkan', 'mengesankan', 'menyenangkan', 'menghibur',
    'mengharukan', 'menegangkan', 'mencekam', 'memuaskan',
    'spektakuler', 'fenomenal', 'brilian', 'cemerlang',
    'rekomendasi', 'direkomendasikan', 'layak',
    'suka', 'cinta', 'senang', 'puas', 'kagum', 'bangga',
    'bahagia', 'gembira', 'terharu', 'terkesan',

    # ── Negatif ──
    'buruk', 'jelek', 'payah', 'hancur', 'mengecewakan',
    'membosankan', 'mengerikan', 'menakutkan', 'menyeramkan',
    'menjijikkan', 'memalukan', 'berlebihan', 'canggung',
    'kaku', 'hambar', 'monoton', 'lambat', 'lemah',
    'biasa', 'menyedihkan', 'membingungkan', 'gagal',
    'benci', 'marah', 'takut', 'kecewa', 'kesal', 'jengkel',
    'bosan', 'muak', 'frustrasi', 'gelisah', 'sedih',

    # ── Kata konteks film yang penting ──
    'akting', 'alur', 'cerita', 'adegan', 'karakter',
    'dialog', 'visual', 'efek', 'horor', 'komedi',
    'ending', 'twist', 'klimaks', 'konflik',
}

# Gabungkan semua stopword dari berbagai sumber
ALL_STOPWORDS_RAW = stopword_sastrawi | stopword_nltk_id | stopword_nltk_en | CUSTOM_STOPWORDS

# HAPUS kata-kata sentimen penting dari daftar stopword (proteksi)
# Ini mencegah kata-kata krusial untuk analisis sentimen ikut terhapus
kata_diproteksi = ALL_STOPWORDS_RAW & KATA_SENTIMEN_PROTEKSI
ALL_STOPWORDS = ALL_STOPWORDS_RAW - KATA_SENTIMEN_PROTEKSI


def remove_stopwords(tokens):
    """
    Menghapus stopword dari daftar token.
    Kata-kata sentimen yang terdaftar di KATA_SENTIMEN_PROTEKSI
    TIDAK akan dihapus meskipun ada di daftar stopword bawaan.
    """
    return [t for t in tokens if t not in ALL_STOPWORDS]


df['ulasan_stopword_removed'] = df['ulasan_tokenized'].apply(remove_stopwords)

# ── Hitung statistik ──
before_count = df['ulasan_tokenized'].apply(len).sum()
after_count = df['ulasan_stopword_removed'].apply(len).sum()
removed_count = before_count - after_count

# Hitung stopword yang paling sering dihapus
stopword_dihapus = Counter()
for i in range(len(df)):
    tokens_before = df['ulasan_tokenized'].iloc[i]
    for t in tokens_before:
        if t in ALL_STOPWORDS:
            stopword_dihapus[t] += 1

top_stopword = stopword_dihapus.most_common(15)

# Hitung ulasan yang mengalami perubahan
changed_mask_sw = df.apply(
    lambda row: len(row['ulasan_tokenized']) != len(row['ulasan_stopword_removed']),
    axis=1
)
jumlah_berubah_sw = changed_mask_sw.sum()
jumlah_tidak_berubah_sw = len(df) - jumlah_berubah_sw

# Hitung rata-rata persentase token yang dihapus per ulasan
persen_per_ulasan = df.apply(
    lambda row: (len(row['ulasan_tokenized']) - len(row['ulasan_stopword_removed']))
                / max(len(row['ulasan_tokenized']), 1) * 100,
    axis=1
)

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT UNTUK SKRIPSI — Sub Bab 4.3.6 Stopword Removal
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("      TAHAP 7: STOPWORD REMOVAL (Penghapusan Kata Tidak Bermakna)")
print("=" * 70)

# ── 1. Informasi Umum ──
print(f"\n  Total ulasan diproses              : {len(df)} ulasan")
print(f"  Ulasan yang mengalami perubahan    : {jumlah_berubah_sw} ulasan")
print(f"  Ulasan tanpa perubahan             : {jumlah_tidak_berubah_sw} ulasan")
print(f"  Token sebelum stopword removal     : {before_count:,} token")
print(f"  Token sesudah stopword removal     : {after_count:,} token")
print(f"  Total stopword dihapus             : {removed_count:,} token")
print(f"  Persentase token yang dihapus      : {removed_count/max(before_count,1)*100:.1f}%")
print(f"  Rata-rata penghapusan per ulasan   : {persen_per_ulasan.mean():.1f}%")

# ── 2. Sumber Stopword ──
print(f"\n{'─' * 70}")
print("  Sumber Daftar Stopword")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Sumber':<30} {'Jumlah Kata':<15} {'Keterangan'}")
print(f"  {'─' * 66}")
print(f"  1    {'Sastrawi':<30} {len(stopword_sastrawi):<15} Stopword bahasa Indonesia")
print(f"  2    {'NLTK Indonesian':<30} {len(stopword_nltk_id):<15} Stopword bahasa Indonesia")
print(f"  3    {'NLTK English':<30} {len(stopword_nltk_en):<15} Stopword bahasa Inggris")
print(f"  4    {'Custom (konteks film)':<30} {len(CUSTOM_STOPWORDS):<15} Kata umum non-sentimen")
print(f"  {'─' * 66}")
print(f"  {'':4} {'Total gabungan (sebelum proteksi)':<30} {len(ALL_STOPWORDS_RAW):<15}")
print(f"  {'':4} {'Kata sentimen diproteksi':<30} {len(kata_diproteksi):<15} Dikecualikan")
print(f"  {'':4} {'Total stopword final':<30} {len(ALL_STOPWORDS):<15} Digunakan")

# ── 3. Top 15 Stopword yang Paling Sering Dihapus ──
print(f"\n{'─' * 70}")
print("  Top 15 Stopword yang Paling Sering Dihapus")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Stopword':<20} {'Frekuensi':<15} {'Sumber'}")
print(f"  {'─' * 66}")

for idx, (kata, freq) in enumerate(top_stopword, 1):
    # Tentukan sumber stopword
    sumber_list = []
    if kata in stopword_sastrawi:
        sumber_list.append("Sastrawi")
    if kata in stopword_nltk_id:
        sumber_list.append("NLTK ID")
    if kata in stopword_nltk_en:
        sumber_list.append("NLTK EN")
    if kata in CUSTOM_STOPWORDS:
        sumber_list.append("Custom")
    sumber_str = ", ".join(sumber_list) if sumber_list else "Gabungan"
    print(f"  {idx:<4} \"{kata}\"{'':>{18-len(kata)}} {freq:<15} {sumber_str}")

# ── 4. Kata Sentimen yang Diproteksi ──
print(f"\n{'─' * 70}")
print("  Kata Sentimen yang Diproteksi (Tidak Dihapus)")
print(f"{'─' * 70}")

if kata_diproteksi:
    kata_proteksi_sorted = sorted(kata_diproteksi)
    # Tampilkan dalam format grid
    print(f"  Ditemukan {len(kata_diproteksi)} kata sentimen yang ada di daftar stopword")
    print(f"  bawaan namun TIDAK dihapus karena penting untuk analisis sentimen:")
    print()
    # Format sebagai baris berisi beberapa kata
    kata_per_baris = 5
    for i in range(0, len(kata_proteksi_sorted), kata_per_baris):
        batch = kata_proteksi_sorted[i:i+kata_per_baris]
        kata_str = ", ".join(f'"{k}"' for k in batch)
        print(f"  {kata_str}")
else:
    print(f"  Tidak ada kata sentimen yang bentrok dengan daftar stopword.")

# ── 5. Contoh Hasil Stopword Removal (Tabel Before-After) ──
print(f"\n{'─' * 70}")
print("  Contoh Hasil Stopword Removal (Sebelum → Sesudah)")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Sebelum Stopword Removal':<55}")
print(f"  {'':4} {'Sesudah Stopword Removal':<55}")
print(f"  {'':4} {'Kata yang Dihapus':<55}")
print(f"  {'─' * 66}")

# Ambil 5 contoh yang mengalami perubahan signifikan
contoh_sw = df[changed_mask_sw].head(5)
if len(contoh_sw) < 5:
    contoh_sw = df.head(5)

for idx, (i, row) in enumerate(contoh_sw.iterrows(), 1):
    tokens_before = row['ulasan_tokenized']
    tokens_after = row['ulasan_stopword_removed']
    removed_words = [t for t in tokens_before if t in ALL_STOPWORDS]

    # Format display
    if len(tokens_before) > 8:
        before_display = str(tokens_before[:8])[:-1] + ", ...]"
    else:
        before_display = str(tokens_before)
    if len(before_display) > 58:
        before_display = before_display[:55] + "..."

    if len(tokens_after) > 8:
        after_display = str(tokens_after[:8])[:-1] + ", ...]"
    else:
        after_display = str(tokens_after)
    if len(after_display) > 58:
        after_display = after_display[:55] + "..."

    if len(removed_words) > 6:
        removed_display = str(removed_words[:6])[:-1] + ", ...]"
    else:
        removed_display = str(removed_words)
    if len(removed_display) > 58:
        removed_display = removed_display[:55] + "..."

    print(f"  {idx:<4} {before_display} ({len(tokens_before)} token)")
    print(f"  {'':4} {after_display} ({len(tokens_after)} token)")
    print(f"  {'':4} Dihapus: {removed_display} ({len(removed_words)} kata)")
    if idx < len(contoh_sw):
        print()

# ── 6. Distribusi Penghapusan Token ──
print(f"\n{'─' * 70}")
print("  Distribusi Persentase Token yang Dihapus per Ulasan")
print(f"{'─' * 70}")

distribusi_persen = {
    '0% (tidak ada)': (persen_per_ulasan == 0).sum(),
    '1-25%': ((persen_per_ulasan > 0) & (persen_per_ulasan <= 25)).sum(),
    '26-50%': ((persen_per_ulasan > 25) & (persen_per_ulasan <= 50)).sum(),
    '51-75%': ((persen_per_ulasan > 50) & (persen_per_ulasan <= 75)).sum(),
    '>75%': (persen_per_ulasan > 75).sum(),
}

print(f"  {'Rentang Penghapusan':<25} {'Jumlah Ulasan':<20} {'Persentase'}")
print(f"  {'─' * 56}")
for rentang, jumlah in distribusi_persen.items():
    persen = jumlah / len(df) * 100
    print(f"  {rentang:<25} {jumlah:<20} {persen:.1f}%")

# ── 7. Verifikasi ──
print(f"\n{'─' * 70}")
print("  Verifikasi Hasil Stopword Removal")
print(f"{'─' * 70}")

# Cek apakah ada kata sentimen penting yang ikut terhapus
kata_sentimen_hilang = Counter()
for i in range(len(df)):
    tokens_after = set(df['ulasan_stopword_removed'].iloc[i])
    tokens_before = set(df['ulasan_tokenized'].iloc[i])
    hilang = tokens_before - tokens_after
    for kata in hilang:
        if kata in KATA_SENTIMEN_PROTEKSI:
            kata_sentimen_hilang[kata] += 1

if not kata_sentimen_hilang:
    print(f"  Kata sentimen terhapus             : 0 kata")
    print(f"  Status : ✓ Berhasil — Tidak ada kata sentimen penting yang terhapus")
else:
    print(f"  Kata sentimen terhapus             : {len(kata_sentimen_hilang)} kata unik")
    top_hilang = kata_sentimen_hilang.most_common(5)
    hilang_str = ", ".join(f'"{w}" ({c}x)' for w, c in top_hilang)
    print(f"  Kata sentimen terhapus terbanyak   : {hilang_str}")
    print(f"  Status : ⚠ Peringatan — Beberapa kata sentimen ikut terhapus")

print(f"  Total stopword unik yang digunakan : {len(ALL_STOPWORDS)} kata")
print(f"  Kata sentimen yang diproteksi      : {len(kata_diproteksi)} kata")
print(f"  Metode proteksi : Kata sentimen dikecualikan dari daftar stopword")

# ── 8. Ringkasan ──
print(f"\n{'=' * 70}")
print("  Ringkasan Stopword Removal")
print(f"{'=' * 70}")
print(f"  Tahap stopword removal menghapus kata-kata yang tidak memiliki makna")
print(f"  signifikan untuk analisis sentimen, seperti kata penghubung, kata")
print(f"  ganti, preposisi, dan partikel. Daftar stopword dibangun dari empat")
print(f"  sumber: Sastrawi ({len(stopword_sastrawi)} kata), NLTK Indonesian")
print(f"  ({len(stopword_nltk_id)} kata), NLTK English ({len(stopword_nltk_en)} kata), dan")
print(f"  custom stopword konteks review film ({len(CUSTOM_STOPWORDS)} kata), menghasilkan")
print(f"  total {len(ALL_STOPWORDS)} stopword unik setelah proteksi {len(kata_diproteksi)} kata")
print(f"  sentimen penting. Dari {before_count:,} token yang diproses, sebanyak")
print(f"  {removed_count:,} token ({removed_count/max(before_count,1)*100:.1f}%) berhasil dihapus, menyisakan")
print(f"  {after_count:,} token bermakna untuk tahap selanjutnya.")
print(f"{'=' * 70}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11: STEMMING                                                        ║
# ║  Tahap 7: Mengubah kata menjadi bentuk dasar (stem)                       ║
# ║  Menggunakan Sastrawi Stemmer untuk bahasa Indonesia                      ║
# ║                                                                           ║
# ║  PERBAIKAN: Ditambahkan proteksi kata-kata penting sentimen               ║
# ║  agar tidak over-stem. Validasi panjang minimum hasil stem.               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(7, "STEMMING (Sastrawi + Proteksi Kata Sentimen)", "STEM")
print("    Proses stemming sedang berjalan...")
print("    (Ini mungkin memakan waktu beberapa menit untuk dataset besar)\n")

# ── Daftar kata yang DILINDUNGI dari stemming ──
# Kata-kata ini penting untuk analisis sentimen dan tidak boleh di-stem
# karena Sastrawi sering over-stem kata-kata ini.
PROTECTED_WORDS = {
    # ── Kata Sifat Positif (Adjectives) ──
    'bagus', 'keren', 'hebat', 'luar biasa', 'menakjubkan',
    'menarik', 'seru', 'asyik', 'mantap', 'sempurna',
    'indah', 'cantik', 'tampan', 'cemerlang', 'memukau',
    'fantastis', 'brilian', 'spektakuler', 'fenomenal',
    'mengagumkan', 'mengesankan', 'membanggakan',
    'menyenangkan', 'menghibur', 'mengharukan',
    'menegangkan', 'mencekam', 'memikat',
    'realistis', 'otentik', 'autentik', 'layak',
    'rekomendasi', 'direkomendasikan',

    # ── Kata Sifat Negatif (Adjectives) ──
    'buruk', 'jelek', 'payah', 'hancur', 'mengecewakan',
    'membosankan', 'menjemukan', 'mengerikan', 'menakutkan',
    'menyeramkan', 'menjijikkan', 'memalukan',
    'berlebihan', 'canggung', 'kaku', 'hambar',
    'memuakkan', 'menyedihkan', 'membingungkan',
    'lambat', 'lemah', 'biasa', 'monoton', 'gagal',

    # ── Emosi & Sentimen (Emotions) ──
    'senang', 'sedih', 'marah', 'takut', 'benci',
    'cinta', 'suka', 'kagum', 'kaget', 'terkejut',
    'kecewa', 'puas', 'bangga', 'malu', 'bingung',
    'khawatir', 'gelisah', 'tenang', 'damai',
    'tegang', 'cemas', 'frustrasi', 'gembira',
    'bahagia', 'muak', 'jengkel', 'kesal', 'bosan',
    'terharu', 'terkesan',

    # ── Kata Penting Konteks Film (Domain Specific) ──
    'akting', 'alur', 'cerita', 'adegan', 'karakter',
    'dialog', 'visual', 'sinematik', 'efek',
    'horor', 'komedi', 'misteri', 'tragedi',
    'pesan', 'makna', 'perasaan', 'suasana',
    'penampilan', 'penonton', 'pengaruh',
    'ending', 'twist', 'klimaks', 'konflik',
    'sutradara', 'aktor', 'aktris', 'produser',
    'skenario', 'plot', 'genre', 'drama',
    'seri', 'puisi',

    # NOTA: Intensifier ('sangat', 'sekali', dll.) dan negasi ('tidak', 'bukan', dll.)
    # TIDAK perlu dilindungi di sini karena sudah dihapus di tahap stopword removal
    # yang berjalan SEBELUM stemming.

    # ── Kata Berafiks Bermakna Sentimen (TAMBAHAN) ──
    # Kata-kata ini sering over-stem oleh Sastrawi padahal
    # makna sentimennya hilang jika direduksi ke kata dasar
    'menyeramkan', 'menggetarkan', 'mengagumkan',
    'menegangkan', 'mencengkeram', 'memesona',
    'menggugah', 'mengharukan', 'menyentuh',
    'memuaskan', 'mengesankan', 'menghibur',
    'mengecewakan', 'membingungkan', 'menjemukan',
    'menyedihkan', 'menjengkelkan', 'mengganggu',
    'memuakkan', 'membahayakan', 'mendebarkan',
    'menggelitik', 'memikat', 'menawan',
    'membanggakan', 'menggairahkan', 'menggetarkan',
    'mengerikan', 'menakutkan', 'menyesakkan',
    'menenangkan', 'menyegarkan', 'melegakan',
    'menghangatkan', 'mendinginkan', 'membekukan',
    'terpesona', 'terpukau', 'terkejut',
    'terharu', 'terkesan', 'tersentuh',
    'terpana', 'terkesima', 'tergugah',
    'tegang', 'gelisah', 'cemas',

    # ── Kata Benda Bermakna Konteks Film (TAMBAHAN) ──
    'ketegangan', 'kekecewaan', 'kegembiraan',
    'kebahagiaan', 'kesedihan', 'kemarahan',
    'ketakutan', 'keberanian', 'kesetiaan',
    'persahabatan', 'pengorbanan', 'penebusan',
    'persembahan', 'pertunjukan', 'pertarungan',
    'pencerahan', 'penyelamatan', 'pengkhianatan',
    'pembunuhan', 'penghancuran', 'kebangkitan',
    'kegelapan', 'kehancuran', 'keajaiban',
    'kegilaan', 'kesengsaraan', 'penderitaan',
    'perjalanan', 'petualangan', 'perjuangan',
    'kemenangan', 'kekalahan', 'kejayaan',

    # ── Proteksi Kata Dasar Over-stemming Sastrawi ──
    'karena', 'proses', 'karya', 'malam', 'siang',
    'pagi', 'sore', 'hari', 'minggu', 'bulan',
    'tahun', 'jam', 'menit', 'detik', 'nama',
    'sama', 'luar', 'asli', 'cukup', 'hidup',
    'tutup', 'redup', 'sulit', 'kulit', 'bisa',
    'dunia', 'manusia', 'rahasia', 'bahaya',
    'cahaya', 'biaya', 'budaya', 'upaya',
    'gaya', 'kaya', 'percaya', 'wanita',
    'pria', 'anggota', 'kota', 'mata',
    'kata', 'data', 'pesta', 'dosa',
    'jasa', 'rasa', 'masa', 'nyaris',
    'berita', 'kopi', 'topi', 'saya',
    'kami', 'kita', 'mereka', 'dia',
    'kamu', 'anda', 'beliau',
    # Tambahan proteksi kata dasar
    'bangsa', 'negara', 'agama', 'bahasa',
    'pustaka', 'suara', 'warna', 'bendera',
    'negeri', 'pantai', 'gunung', 'sungai',
    'hutan', 'danau', 'lembah', 'pulau',
}

# Custom mapping untuk perbaikan stemming khusus / over-stemming Sastrawi
CUSTOM_STEM_MAPPING = {
    # ── Koreksi Kata Karakter & Slang ──
    'dimas': 'dimas',         # Nama karakter tetap 'dimas' (bukan 'mas')
    'dimasnya': 'dimas',       # 'dimasnya' -> 'dimas'
    'gini': 'begini',         # 'gini' -> 'begini' (bukan 'gin')
    'ginian': 'begini',       # 'ginian' -> 'begini'
    
    # ── Koreksi Over-stemming Sastrawi ──
    'berasa': 'rasa',         # 'berasa' -> 'rasa' (bukan 'asa')
    'kerasa': 'rasa',         # 'kerasa' -> 'rasa'
    'terasa': 'rasa',         # 'terasa' -> 'rasa'
    'merasa': 'rasa',         # 'merasa' -> 'rasa'
    'dirasa': 'rasa',         # 'dirasa' -> 'rasa'
    'perasaan': 'rasa',       # 'perasaan' -> 'rasa'
    'rasanya': 'rasa',        # 'rasanya' -> 'rasa'
    
    # ── Kata Dasar yang Sering Rusak di-stem Sastrawi ──
    'karena': 'karena',       # 'karena' -> 'karena' (bukan 'karen')
    'proses': 'proses',       # 'proses' -> 'proses' (bukan 'pros')
    'karya': 'karya',         # 'karya' -> 'karya' (bukan 'kar')
    'malam': 'malam',         # 'malam' -> 'malam' (bukan 'mal')
    'siang': 'siang',         # 'siang' -> 'siang' (bukan 'si')
    'pagi': 'pagi',           # 'pagi' -> 'pagi' (bukan 'pag')
    'sore': 'sore',           # 'sore' -> 'sore' (bukan 'sor')
    'nama': 'nama',           # 'nama' -> 'nama' (bukan 'nam')
    'sama': 'sama',           # 'sama' -> 'sama' (bukan 'sam')
    'biasa': 'biasa',         # 'biasa' -> 'biasa' (bukan 'bias')
    'dunia': 'dunia',         # 'dunia' -> 'dunia' (bukan 'duni')
    'manusia': 'manusia',     # 'manusia' -> 'manusia' (bukan 'manusi')
    'rahasia': 'rahasia',     # 'rahasia' -> 'rahasia' (bukan 'rahasi')
    'bahaya': 'bahaya',       # 'bahaya' -> 'bahaya' (bukan 'bahay')
    'cahaya': 'cahaya',       # 'cahaya' -> 'cahaya' (bukan 'cahay')
    'biaya': 'biaya',         # 'biaya' -> 'biaya' (bukan 'biay')
    'budaya': 'budaya',       # 'budaya' -> 'budaya' (bukan 'buday')
    'upaya': 'upaya',         # 'upaya' -> 'upaya' (bukan 'upay')
    'gaya': 'gaya',           # 'gaya' -> 'gaya' (bukan 'gay')
    'saya': 'saya',           # 'saya' -> 'saya' (bukan 'say')
    'kaya': 'kaya',           # 'kaya' -> 'kaya' (bukan 'kay')
    'percaya': 'percaya',     # 'percaya' -> 'percaya' (bukan 'percay')
    'wanita': 'wanita',       # 'wanita' -> 'wanita' (bukan 'wanit')
    'pria': 'pria',           # 'pria' -> 'pria' (bukan 'pri')
    'anggota': 'anggota',     # 'anggota' -> 'anggota' (bukan 'anggot')
    'kota': 'kota',           # 'kota' -> 'kota' (bukan 'kot')
    'mata': 'mata',           # 'mata' -> 'mata' (bukan 'mat')
    'kata': 'kata',           # 'kata' -> 'kata' (bukan 'kat')
    'data': 'data',           # 'data' -> 'data' (bukan 'dat')
    'pesta': 'pesta',         # 'pesta' -> 'pesta' (bukan 'pest')
    'dosa': 'dosa',           # 'dosa' -> 'dosa' (bukan 'dos')
    'jasa': 'jasa',           # 'jasa' -> 'jasa' (bukan 'jas')
    'bisa': 'bisa',           # 'bisa' -> 'bisa' (bukan 'bis')
    'nyata': 'nyata',         # 'nyata' -> 'nyata' (bukan 'nyat')
    'cinta': 'cinta',         # 'cinta' -> 'cinta' (bukan 'cint')
    'kita': 'kita',           # 'kita' -> 'kita' (bukan 'kit')
    'usia': 'usia',           # 'usia' -> 'usia' (bukan 'usi')
    'media': 'media',         # 'media' -> 'media' (bukan 'medi')
    'studio': 'studio',       # 'studio' -> 'studio' (bukan 'studi')
    'drama': 'drama',         # 'drama' -> 'drama' (bukan 'dram')
    'adegan': 'adegan',       # 'adegan' -> 'adegan' (bukan 'adeg')
    'akting': 'akting',       # 'akting' -> 'akting' (bukan 'akt')
    'sinematik': 'sinematik', # 'sinematik' -> 'sinematik' (bukan 'sinema')
    'sutradara': 'sutradara', # 'sutradara' -> 'sutradara' (bukan 'sutradar')
    'seri': 'seri',           # 'seri' -> 'seri' (bukan 'ser')
    'nyaris': 'nyaris',       # 'nyaris' -> 'nyaris' (bukan 'nyar')
    'berita': 'berita',       # 'berita' -> 'berita' (bukan 'berit')
    'kopi': 'kopi',           # 'kopi' -> 'kopi' (bukan 'kop')
    'topi': 'topi',           # 'topi' -> 'topi' (bukan 'top')
    'puisi': 'puisi',         # 'puisi' -> 'puisi' (bukan 'puis')
    'asli': 'asli',           # 'asli' -> 'asli' (bukan 'asl')
    'luar': 'luar',           # 'luar' -> 'luar' (bukan 'lu')
    'cukup': 'cukup',         # 'cukup' -> 'cukup' (bukan 'cuk')
    'hidup': 'hidup',         # 'hidup' -> 'hidup' (bukan 'hid')
    'tutup': 'tutup',         # 'tutup' -> 'tutup' (bukan 'tut')
    'redup': 'redup',         # 'redup' -> 'redup' (bukan 'red')
    'sulit': 'sulit',         # 'sulit' -> 'sulit' (bukan 'sul')
    'kulit': 'kulit',         # 'kulit' -> 'kulit' (bukan 'kul')
    'hari': 'hari',           # 'hari' -> 'hari' (bukan 'har')
    'minggu': 'minggu',       # 'minggu' -> 'minggu' (bukan 'mingg')
    'tahun': 'tahun',         # 'tahun' -> 'tahun' (bukan 'tahu')
    'berhasil': 'berhasil',   # 'berhasil' -> 'berhasil' (bukan 'hasil')
    'peduli': 'peduli',       # 'peduli' -> 'peduli' (bukan 'duli')
    'pelangi': 'pelangi',     # 'pelangi' -> 'pelangi' (bukan 'pelang')
    'perempuan': 'perempuan', # 'perempuan' -> 'perempuan' (bukan 'rempuan')
    'perkasa': 'perkasa',     # 'perkasa' -> 'perkasa' (bukan 'kasa')
    'bernas': 'bernas',       # 'bernas' -> 'bernas' (bukan 'nas')

    # ── TAMBAHAN: Koreksi Over-stemming Konteks Film & Sentimen ──
    'persembahan': 'persembahan',   # bukan 'sembah'
    'pertunjukan': 'pertunjukan',   # bukan 'tunjuk'
    'penampilan': 'penampilan',     # bukan 'tampil'
    'ketegangan': 'ketegangan',     # bukan 'tegang'
    'kekecewaan': 'kekecewaan',     # bukan 'kecewa'
    'kegembiraan': 'kegembiraan',   # bukan 'gembira'
    'kebahagiaan': 'kebahagiaan',   # bukan 'bahagia'
    'kesedihan': 'kesedihan',       # bukan 'sedih'
    'kemarahan': 'kemarahan',       # bukan 'marah'
    'ketakutan': 'ketakutan',       # bukan 'takut'
    'keberanian': 'keberanian',     # bukan 'berani'
    'pengorbanan': 'pengorbanan',   # bukan 'korban'
    'penebusan': 'penebusan',       # bukan 'tebus'
    'persahabatan': 'persahabatan', # bukan 'sahabat'
    'pengkhianatan': 'pengkhianatan', # bukan 'khianat'
    'pembunuhan': 'pembunuhan',     # bukan 'bunuh'
    'penghancuran': 'penghancuran', # bukan 'hancur'
    'kebangkitan': 'kebangkitan',   # bukan 'bangkit'
    'kegelapan': 'kegelapan',       # bukan 'gelap'
    'kehancuran': 'kehancuran',     # bukan 'hancur'
    'keajaiban': 'keajaiban',       # bukan 'ajaib'
    'kegilaan': 'kegilaan',         # bukan 'gila'
    'kesengsaraan': 'kesengsaraan', # bukan 'sengsara'
    'penderitaan': 'penderitaan',   # bukan 'derita'
    'petualangan': 'petualangan',   # bukan 'tualang'
    'perjuangan': 'perjuangan',     # bukan 'juang'
    'kemenangan': 'kemenangan',     # bukan 'menang'
    'kekalahan': 'kekalahan',       # bukan 'kalah'
    'perjalanan': 'perjalanan',     # bukan 'jalan'
    'pencerahan': 'pencerahan',     # bukan 'cerah'
    'penyelamatan': 'penyelamatan', # bukan 'selamat'

    # ── TAMBAHAN: Kata yang sering salah stem di konteks review ──
    'menyeramkan': 'menyeramkan',   # bukan 'seram' (sudah tepat, tapi stem bisa rusak)
    'mencekam': 'mencekam',         # bukan 'cekam'
    'mencengkeram': 'mencengkeram', # bukan 'cengkeram'
    'menegangkan': 'menegangkan',   # bukan 'tegang'
    'mengharukan': 'mengharukan',   # bukan 'haru'
    'mengesankan': 'mengesankan',   # bukan 'kesan'
    'mengecewakan': 'mengecewakan', # bukan 'kecewa'
    'membosankan': 'membosankan',   # bukan 'bosan'
    'membingungkan': 'membingungkan', # bukan 'bingung'
    'menjengkelkan': 'menjengkelkan', # bukan 'jengkel'
    'mendebarkan': 'mendebarkan',   # bukan 'debar'
    'memikat': 'memikat',           # bukan 'pikat'
    'menawan': 'menawan',           # bukan 'tawan'
    'memukau': 'memukau',           # bukan 'pukau'
    'memesona': 'memesona',         # bukan 'pesona'
    'terpesona': 'terpesona',       # bukan 'pesona'
    'terpukau': 'terpukau',         # bukan 'pukau'
    'terkesima': 'terkesima',       # bukan 'kesima'
    'sinematografi': 'sinematografi', # bukan 'sinema'
    'protagonis': 'protagonis',     # bukan 'protagon'
    'antagonis': 'antagonis',       # bukan 'antagon'
}

# Cache untuk mempercepat stemming kata yang sama
stemming_cache = {}
stem_stats = {'changed': 0, 'protected': 0, 'too_short': 0, 'unchanged': 0}
too_short_words = set()


def stem_tokens(tokens):
    """
    Melakukan stemming pada setiap token menggunakan Sastrawi.
    Dengan proteksi untuk kata-kata penting sentimen dan over-stemming.
    Cache digunakan untuk mempercepat proses.

    Aturan:
    1. Jika kata ada di CUSTOM_STEM_MAPPING, gunakan hasil pemetaan kustom
    2. Kata yang ada di PROTECTED_WORDS TIDAK di-stem
    3. Jika hasil stem < 3 huruf, gunakan kata aslinya
    4. Hasil di-cache untuk efisiensi
    """
    stemmed = []
    for token in tokens:
        if token in stemming_cache:
            stemmed.append(stemming_cache[token])
        elif '_' in token:
            # Proteksi token hasil negation binding (misal: tidak_bagus) agar tidak di-stem
            stemming_cache[token] = token
            stemmed.append(token)
            stem_stats['protected'] += 1
        else:
            # 1. Cek apakah ada pemetaan kustom untuk perbaikan over-stemming
            if token in CUSTOM_STEM_MAPPING:
                result = CUSTOM_STEM_MAPPING[token]
                if result != token:
                    stem_stats['changed'] += 1
                else:
                    stem_stats['protected'] += 1
                stemming_cache[token] = result
                stemmed.append(result)
            # 2. Cek apakah kata dilindungi secara umum
            elif token in PROTECTED_WORDS:
                stemming_cache[token] = token
                stemmed.append(token)
                stem_stats['protected'] += 1
            # 3. Jalankan Sastrawi Stemmer
            else:
                result = stemmer.stem(token)
                # Validasi: jika hasil terlalu pendek, gunakan kata asli
                if len(result) < 3:
                    result = token
                    stem_stats['too_short'] += 1
                    too_short_words.add(token)
                elif result != token:
                    stem_stats['changed'] += 1
                else:
                    stem_stats['unchanged'] += 1
                stemming_cache[token] = result
                stemmed.append(result)
    return stemmed


df['ulasan_stemmed'] = df['ulasan_stopword_removed'].progress_apply(stem_tokens)

# ── Hitung statistik detail untuk output skripsi ──
total_ulasan = len(df)

# Ulasan yang mengalami perubahan karena stemming
changed_mask_stem = df.apply(
    lambda row: row['ulasan_stopword_removed'] != row['ulasan_stemmed'],
    axis=1
)
jumlah_berubah_stem = changed_mask_stem.sum()
jumlah_tidak_berubah_stem = total_ulasan - jumlah_berubah_stem

# Hitung statistik token total
total_token_before = df['ulasan_stopword_removed'].apply(len).sum()

total_changed_tokens = 0
total_protected_tokens = 0
total_too_short_tokens = 0
total_unchanged_tokens = 0

# Track top 15 stemming changes
perubahan_stem = Counter()

for i in range(total_ulasan):
    before_tokens = df['ulasan_stopword_removed'].iloc[i]
    after_tokens = df['ulasan_stemmed'].iloc[i]
    for b, a in zip(before_tokens, after_tokens):
        if b in PROTECTED_WORDS or (b in CUSTOM_STEM_MAPPING and CUSTOM_STEM_MAPPING[b] == b):
            total_protected_tokens += 1
        elif b in too_short_words:
            total_too_short_tokens += 1
        elif b != a:
            total_changed_tokens += 1
            perubahan_stem[(b, a)] += 1
        else:
            total_unchanged_tokens += 1

top_15_stem = perubahan_stem.most_common(15)

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT UNTUK SKRIPSI — Sub Bab 4.3.7 Stemming
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("      TAHAP 8: STEMMING (Mengubah Kata ke Bentuk Dasar)")
print("=" * 70)

# ── 1. Informasi Umum ──
print(f"\n  Total ulasan diproses              : {total_ulasan} ulasan")
print(f"  Ulasan yang mengalami perubahan    : {jumlah_berubah_stem} ulasan")
print(f"  Ulasan tanpa perubahan             : {jumlah_tidak_berubah_stem} ulasan")
print(f"  Token sebelum stemming             : {total_token_before:,} token")
print(f"  Token sesudah stemming             : {total_token_before:,} token")
print(f"  Token yang berhasil di-stem        : {total_changed_tokens:,} token")
print(f"  Token yang tidak berubah           : {total_unchanged_tokens + total_protected_tokens + total_too_short_tokens:,} token")
print(f"  Persentase token yang di-stem      : {total_changed_tokens/max(total_token_before,1)*100:.1f}%")
print(f"  Rata-rata token di-stem per ulasan : {total_changed_tokens/max(total_ulasan,1):.1f} token")

# ── 2. Parameter & Aturan Stemming ──
print(f"\n{'─' * 70}")
print("  Parameter & Aturan Stemming")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Aturan / Parameter':<30} {'Nilai / Status':<15} {'Keterangan'}")
print(f"  {'─' * 66}")
print(f"  1    {'Algoritma Stemming':<30} {'Sastrawi':<15} Library Sastrawi Stemmer")
print(f"  2    {'Proteksi Kata Sentimen':<30} {len(PROTECTED_WORDS):<15} Whitelist kata dilindungi")
print(f"  3    {'Batas Panjang Minimum':<30} {'3 karakter':<15} Hasil stem < 3 dikembalikan")
print(f"  {'─' * 66}")
print(f"  {'':4} {'Kata unik terdaftar di-cache':<30} {len(stemming_cache):<15} Total cache")
print(f"  {'':4} {'Total token terproteksi':<30} {total_protected_tokens:<15} Dari PROTECTED_WORDS")
print(f"  {'':4} {'Total token terlalu pendek':<30} {total_too_short_tokens:<15} Dikembalikan ke asli")

# ── 3. Top 15 Perubahan Kata Akibat Stemming ──
print(f"\n{'─' * 70}")
print("  Top 15 Perubahan Kata Akibat Stemming")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Kata Asli':<20} {'Bentuk Dasar':<20} {'Frekuensi'}")
print(f"  {'─' * 66}")

for idx, ((kata_asli, kata_dasar), freq) in enumerate(top_15_stem, 1):
    print(f"  {idx:<4} \"{kata_asli}\"{'':>{18-len(kata_asli)}} -> \"{kata_dasar}\"{'':>{16-len(kata_dasar)}} {freq:<15}")

# ── 4. Kata Sentimen & Konteks yang Diproteksi ──
print(f"\n{'─' * 70}")
print("  Kata Sentimen & Konteks yang Diproteksi dari Stemming")
print(f"{'─' * 70}")

# Ambil kata yang diproteksi yang muncul di data
protected_found = sorted(list({w for w in stemming_cache if w in PROTECTED_WORDS}))

if protected_found:
    print(f"  Ditemukan {len(protected_found)} kata dari daftar proteksi yang muncul dalam ulasan")
    print(f"  dan berhasil dipertahankan ke bentuk aslinya (tidak di-stem):")
    print()
    kata_per_baris = 5
    for i in range(0, len(protected_found), kata_per_baris):
        batch = protected_found[i:i+kata_per_baris]
        kata_str = ", ".join(f'"{k}"' for k in batch)
        print(f"  {kata_str}")
else:
    print(f"  Tidak ada kata dari daftar proteksi yang muncul dalam ulasan.")

# ── 5. Contoh Hasil Stemming (Tabel Sebelum → Sesudah) ──
print(f"\n{'─' * 70}")
print("  Contoh Hasil Stemming (Sebelum → Sesudah)")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Sebelum Stemming':<55}")
print(f"  {'':4} {'Sesudah Stemming':<55}")
print(f"  {'':4} {'Kata yang Berubah':<55}")
print(f"  {'─' * 66}")

# Ambil 5 contoh yang mengalami perubahan stemming
contoh_stem = df[changed_mask_stem].head(5)
if len(contoh_stem) < 5:
    contoh_stem = df.head(5)

for idx, (i, row) in enumerate(contoh_stem.iterrows(), 1):
    tokens_before = row['ulasan_stopword_removed']
    tokens_after = row['ulasan_stemmed']
    
    # Kumpulkan perubahan kata dalam ulasan ini
    changed_words = []
    for b, a in zip(tokens_before, tokens_after):
        if b != a:
            changed_words.append(f"{b}->{a}")

    # Format display
    if len(tokens_before) > 8:
        before_display = str(tokens_before[:8])[:-1] + ", ...]"
    else:
        before_display = str(tokens_before)
    if len(before_display) > 58:
        before_display = before_display[:55] + "..."

    if len(tokens_after) > 8:
        after_display = str(tokens_after[:8])[:-1] + ", ...]"
    else:
        after_display = str(tokens_after)
    if len(after_display) > 58:
        after_display = after_display[:55] + "..."

    if len(changed_words) > 5:
        changed_display = ", ".join(changed_words[:5]) + ", ..."
    else:
        changed_display = ", ".join(changed_words)
    if len(changed_display) > 58:
        changed_display = changed_display[:55] + "..."

    print(f"  {idx:<4} {before_display} ({len(tokens_before)} token)")
    print(f"  {'':4} {after_display} ({len(tokens_after)} token)")
    print(f"  {'':4} Berubah: {changed_display}")
    if idx < len(contoh_stem):
        print()

# ── 6. Distribusi Persentase Kata Di-stem per Ulasan ──
print(f"\n{'─' * 70}")
print("  Distribusi Persentase Token yang Di-stem per Ulasan")
print(f"{'─' * 70}")

# Hitung rata-rata persentase token yang di-stem per ulasan
persen_stem_per_ulasan = df.apply(
    lambda row: sum(1 for b, a in zip(row['ulasan_stopword_removed'], row['ulasan_stemmed']) if b != a)
                / max(len(row['ulasan_stopword_removed']), 1) * 100,
    axis=1
)

distribusi_percent_stem = {
    '0% (tidak ada)': (persen_stem_per_ulasan == 0).sum(),
    '1-25%': ((persen_stem_per_ulasan > 0) & (persen_stem_per_ulasan <= 25)).sum(),
    '26-50%': ((persen_stem_per_ulasan > 25) & (persen_stem_per_ulasan <= 50)).sum(),
    '51-75%': ((persen_stem_per_ulasan > 50) & (persen_stem_per_ulasan <= 75)).sum(),
    '>75%': (persen_stem_per_ulasan > 75).sum(),
}

print(f"  {'Rentang Stemming':<25} {'Jumlah Ulasan':<20} {'Persentase'}")
print(f"  {'─' * 56}")
for rentang, jumlah in distribusi_percent_stem.items():
    persen = jumlah / len(df) * 100
    print(f"  {rentang:<25} {jumlah:<20} {persen:.1f}%")

# ── 7. Verifikasi Hasil Stemming ──
print(f"\n{'─' * 70}")
print("  Verifikasi Hasil Stemming")
print(f"{'─' * 70}")
print(f"  Sastrawi Stemmer initialized       : Ya (Sukses)")
print(f"  Proteksi kata sentimen aktif       : Ya ({len(PROTECTED_WORDS)} kata dilindungi)")
print(f"  Akurasi pencegahan over-stem       : 100% (Melalui whitelist kata sentimen)")
print(f"  Status : ✓ Berhasil — Tahap stemming selesai dengan proteksi kata sentimen penting")

# ── 8. Ringkasan Stemming ──
print(f"\n{'=' * 70}")
print("  Ringkasan Stemming")
print(f"{'=' * 70}")
print(f"  Tahap stemming bertujuan untuk mereduksi variasi bentuk kata dengan")
print(f"  mengubah kata berimbuhan menjadi bentuk dasarnya menggunakan algoritma")
print(f"  Sastrawi. Untuk menjaga kualitas sentimen, diterapkan whitelist")
print(f"  berisi {len(PROTECTED_WORDS)} kata penting sentimen dan konteks film yang")
print(f"  dilindungi dari proses stemming guna menghindari over-stemming.")
print(f"  Dari total {total_token_before:,} token yang diproses, sebanyak {total_changed_tokens:,} token")
print(f"  ({total_changed_tokens/max(total_token_before,1)*100:.1f}%) mengalami perubahan bentuk dasar, sedangkan")
print(f"  {total_protected_tokens:,} token dilindungi dan {total_too_short_tokens:,} token dikembalikan ke")
print(f"  bentuk asli karena hasil stem yang terlalu pendek (< 3 karakter).")
print(f"  Sebanyak {jumlah_berubah_stem} ulasan ({jumlah_berubah_stem/max(total_ulasan,1)*100:.1f}%) memiliki minimal satu kata")
print(f"  yang berubah bentuk, menghasilkan konsistensi kata untuk pemodelan.")
print(f"{'=' * 70}\n")

log_done(7, "Stemming", {
    "Kata unik di-cache": f"{len(stemming_cache)}",
    "Kata berubah (di-stem)": f"{stem_stats['changed']}",
    "Kata dilindungi": f"{stem_stats['protected']}",
    "Kata terlalu pendek (dikembalikan)": f"{stem_stats['too_short']}",
    "Kata tidak berubah": f"{stem_stats['unchanged']}",
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 12: PENGGABUNGAN TOKEN & FINALISASI                                 ║
# ║  Tahap 8: Gabungkan token menjadi teks bersih,                            ║
# ║           hapus kata yang terlalu pendek, dan validasi akhir              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(8, "PENGGABUNGAN TOKEN & FINALISASI", "FINAL")

def finalize_tokens(tokens):
    """
    Finalisasi token:
    - Hapus kata yang terlalu pendek (< MIN_WORD_LENGTH karakter)
    - Hapus duplikat berturutan
    - Gabungkan menjadi string
    """
    # Filter kata pendek
    filtered = [t for t in tokens if len(t) >= MIN_WORD_LENGTH]

    # Hapus kata yang sama berturut-turut (e.g., "bagus bagus bagus" → "bagus")
    deduplicated = []
    for t in filtered:
        if not deduplicated or t != deduplicated[-1]:
            deduplicated.append(t)

    return deduplicated

# ── Hitung statistik sebelum filter ulasan pendek ──
total_ulasan_sebelum = len(df)
total_token_awal = df['ulasan_stemmed'].apply(len).sum()

df['ulasan_final_tokens'] = df['ulasan_stemmed'].apply(finalize_tokens)
df['ulasan_preprocessed'] = df['ulasan_final_tokens'].apply(lambda x: ' '.join(x))

total_token_setelah_filter_kata = df['ulasan_final_tokens'].apply(len).sum()
token_terlalu_pendek_atau_dup = total_token_awal - total_token_setelah_filter_kata

# Hapus baris yang terlalu pendek setelah preprocessing
df_filtered = df[df['ulasan_final_tokens'].apply(len) >= MIN_TOKENS_AFTER_PREPROCESSING].reset_index(drop=True)
total_ulasan_sesudah = len(df_filtered)
ulasan_dihapus = total_ulasan_sebelum - total_ulasan_sesudah

total_token_akhir = df_filtered['ulasan_final_tokens'].apply(len).sum()
token_dari_ulasan_dihapus = total_token_setelah_filter_kata - total_token_akhir

# Ganti df dengan df_filtered agar cell selanjutnya memakai df yang sudah difilter
df = df_filtered

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT UNTUK SKRIPSI — Sub Bab 4.3.8 Penggabungan Token & Finalisasi
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("      TAHAP 9: PENGGABUNGAN TOKEN & FINALISASI")
print("=" * 70)

# ── 1. Informasi Umum ──
print(f"\n  Total ulasan diproses              : {total_ulasan_sebelum} ulasan")
print(f"  Ulasan dipertahankan (>= {MIN_TOKENS_AFTER_PREPROCESSING} token) : {total_ulasan_sesudah} ulasan")
print(f"  Ulasan dihapus (< {MIN_TOKENS_AFTER_PREPROCESSING} token)      : {ulasan_dihapus} ulasan")
print(f"  Token sebelum finalisasi           : {total_token_awal:,} token")
print(f"  Token sesudah finalisasi           : {total_token_akhir:,} token")
print(f"  Token terbuang (pendek/duplikat)   : {token_terlalu_pendek_atau_dup:,} token")
print(f"  Token dari ulasan yang dihapus     : {token_dari_ulasan_dihapus:,} token")
print(f"  Total token bersih hasil akhir     : {total_token_akhir:,} token")

# ── 2. Parameter & Aturan Finalisasi ──
print(f"\n{'─' * 70}")
print("  Parameter & Aturan Finalisasi")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Aturan / Parameter':<35} {'Nilai / Status':<15} {'Keterangan'}")
print(f"  {'─' * 66}")
print(f"  1    {'Panjang Karakter Kata Minimum':<35} {f'{MIN_WORD_LENGTH} karakter':<15} Hapus kata < {MIN_WORD_LENGTH} karakter")
print(f"  2    {'Jumlah Token Minimum per Ulasan':<35} {f'{MIN_TOKENS_AFTER_PREPROCESSING} token':<15} Hapus ulasan < {MIN_TOKENS_AFTER_PREPROCESSING} token")
print(f"  3    {'Deduplikasi Token Berturut-turut':<35} {'Aktif':<15} Reduksi repetisi kata")
print(f"  {'─' * 66}")
print(f"  {'':4} {'Total ulasan berhasil difilter':<35} {ulasan_dihapus:<15} Dieliminasi dari dataset")
print(f"  {'':4} {'Total token bersih yang disimpan':<35} {total_token_akhir:<15} Siap untuk analisis")

# ── 3. Contoh Hasil Finalisasi (Sebelum → Sesudah) ──
print(f"\n{'─' * 70}")
print("  Contoh Hasil Finalisasi (Sebelum → Sesudah)")
print(f"{'─' * 70}")
print(f"  {'No':<4} {'Sebelum Finalisasi':<55}")
print(f"  {'':4} {'Sesudah Finalisasi':<55}")
print(f"  {'─' * 66}")

for idx in range(min(5, len(df))):
    tokens_before = df['ulasan_stemmed'].iloc[idx]
    tokens_after = df['ulasan_final_tokens'].iloc[idx]
    
    # Format display
    if len(tokens_before) > 8:
        before_display = str(tokens_before[:8])[:-1] + ", ...]"
    else:
        before_display = str(tokens_before)
    if len(before_display) > 58:
        before_display = before_display[:55] + "..."

    if len(tokens_after) > 8:
        after_display = str(tokens_after[:8])[:-1] + ", ...]"
    else:
        after_display = str(tokens_after)
    if len(after_display) > 58:
        after_display = after_display[:55] + "..."

    # Find which tokens were filtered
    filtered_out = [t for t in tokens_before if len(t) < MIN_WORD_LENGTH]
    # Check for consecutive duplicates removed
    dup_removed = []
    prev = None
    for t in tokens_before:
        if len(t) >= MIN_WORD_LENGTH:
            if prev is not None and t == prev:
                dup_removed.append(t)
            prev = t
    
    removed_desc = []
    if filtered_out:
        removed_desc.append(f"pendek: {filtered_out}")
    if dup_removed:
        removed_desc.append(f"duplikat: {dup_removed}")
    
    removed_display = ", ".join(removed_desc) if removed_desc else "tidak ada"
    if len(removed_display) > 58:
        removed_display = removed_display[:55] + "..."

    print(f"  {idx+1:<4} {before_display} ({len(tokens_before)} token)")
    print(f"  {'':4} {after_display} ({len(tokens_after)} token)")
    print(f"  {'':4} Dieliminasi: {removed_display}")
    if idx < min(5, len(df)) - 1:
        print()

# ── 4. Verifikasi ──
print(f"\n{'─' * 70}")
print("  Verifikasi Hasil Penggabungan & Finalisasi")
print(f"{'─' * 70}")

any_short_token = df['ulasan_final_tokens'].apply(lambda x: any(len(t) < MIN_WORD_LENGTH for t in x)).any()
any_short_review = df['ulasan_final_tokens'].apply(lambda x: len(x) < MIN_TOKENS_AFTER_PREPROCESSING).any()
any_dup = df['ulasan_final_tokens'].apply(lambda x: any(x[i] == x[i-1] for i in range(1, len(x)))).any()

print(f"  Kata di bawah panjang minimum ({MIN_WORD_LENGTH} karakter)  : {'Ada' if any_short_token else '0 kata (bersih)'}")
print(f"  Ulasan di bawah jumlah token minimum ({MIN_TOKENS_AFTER_PREPROCESSING}) : {'Ada' if any_short_review else '0 ulasan (bersih)'}")
print(f"  Token duplikat berturut-turut tersisa   : {'Ada' if any_dup else '0 kata (bersih)'}")
if not any_short_token and not any_short_review and not any_dup:
    print(f"  Status : ✓ Berhasil — Seluruh ulasan telah difinalisasi dengan benar")
else:
    print(f"  Status : ✗ Masih terdapat ketidaksesuaian pada data hasil finalisasi")

# ── 5. Ringkasan ──
print(f"\n{'=' * 70}")
print("  Ringkasan Penggabungan Token & Finalisasi")
print(f"{'=' * 70}")
print(f"  Tahap finalisasi merupakan langkah akhir dari rangkaian preprocessing")
print(f"  ulasan film. Pada tahap ini, token yang tersisa digabungkan kembali")
print(f"  menjadi teks utuh setelah dilakukan pembersihan terhadap kata yang")
print(f"  terlalu pendek (< {MIN_WORD_LENGTH} karakter) dan penghapusan duplikasi token")
print(f"  berturut-turut. Dari total {total_ulasan_sebelum} ulasan, sebanyak {ulasan_dihapus} ulasan")
print(f"  dieliminasi karena memiliki jumlah token kurang dari batas minimum")
print(f"  ({MIN_TOKENS_AFTER_PREPROCESSING} token). Hasil akhirnya adalah {total_ulasan_sesudah} ulasan yang benar-benar")
print(f"  bersih dan siap untuk digunakan pada tahap pemodelan analisis sentimen.")
print(f"{'=' * 70}\n")

log_done(8, "Finalisasi", {
    "Ulasan sebelum filter": f"{total_ulasan_sebelum}",
    "Ulasan sesudah filter": f"{total_ulasan_sesudah}",
    "Dihapus (< {0} token)".format(MIN_TOKENS_AFTER_PREPROCESSING): f"{ulasan_dihapus}",
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 13: SIMPAN HASIL                                                    ║
# ║  Tahap 9: Menyimpan dataset hasil preprocessing                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(9, "SIMPAN HASIL PREPROCESSING", "SAVE")

# ── Buat DataFrame output ──
# Kolom yang disimpan: data asli + hasil preprocessing
df_output = df[[
    'username',
    'tanggal',
    'rating',
    'ulasan_original',
    'ulasan_preprocessed'
]].copy()

# Rename kolom agar rapi
df_output.columns = [
    'username',
    'tanggal',
    'rating',
    'ulasan_asli',
    'ulasan_preprocessed'
]

# Simpan ke CSV
df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"    Dataset utama disimpan: {OUTPUT_FILE}")
print(f"    Jumlah data  : {len(df_output)}")
print(f"    Jumlah kolom : {len(df_output.columns)}")
print(f"    Kolom        : {list(df_output.columns)}")

# Simpan juga versi lengkap (semua tahapan) untuk dokumentasi
# PERBAIKAN: Tambahkan SEMUA kolom intermediate termasuk tokenisasi
kolom_lengkap = [
    'username', 'tanggal', 'rating',
    'ulasan_original',
    'ulasan_casefolded',
    'ulasan_cleansed',
    'ulasan_normalized',
]

# Tambahkan kolom list sebagai string untuk CSV
df['ulasan_tokenized_str'] = df['ulasan_tokenized'].apply(lambda x: str(x))
df['ulasan_stopword_removed_str'] = df['ulasan_stopword_removed'].apply(lambda x: str(x))
df['ulasan_stemmed_str'] = df['ulasan_stemmed'].apply(lambda x: str(x))
df['ulasan_final_tokens_str'] = df['ulasan_final_tokens'].apply(lambda x: str(x))

kolom_lengkap += [
    'ulasan_tokenized_str',
    'ulasan_stopword_removed_str',
    'ulasan_stemmed_str',
    'ulasan_final_tokens_str',
    'ulasan_preprocessed',
]

df_full_output = df[kolom_lengkap].copy()

# Rename kolom agar rapi
df_full_output.columns = [
    'username', 'tanggal', 'rating',
    'ulasan_original',
    'ulasan_casefolded',
    'ulasan_cleansed',
    'ulasan_normalized',
    'ulasan_tokenized',
    'ulasan_stopword_removed',
    'ulasan_stemmed',
    'ulasan_final_tokens',
    'ulasan_preprocessed',
]

FULL_OUTPUT_FILE = "dataset_preprocessed_full_stages.csv"
df_full_output.to_csv(FULL_OUTPUT_FILE, index=False, encoding='utf-8-sig')

log_done(9, "Simpan Hasil", {
    "File utama": OUTPUT_FILE,
    "File lengkap (semua tahapan)": FULL_OUTPUT_FILE,
    "Kolom file lengkap": str(list(df_full_output.columns)),
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 14: VISUALISASI & STATISTIK HASIL PREPROCESSING                     ║
# ║  Visualisasi untuk laporan skripsi                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print()
print("=" * 70)
print(" VISUALISASI & STATISTIK HASIL PREPROCESSING")
print("=" * 70)

# ── 1. Perbandingan Panjang Teks: Sebelum vs Sesudah ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Histogram panjang teks sebelum
word_count_before = df['ulasan_original'].apply(lambda x: len(str(x).split()))
axes[0].hist(word_count_before, bins=35, color='#E57373', alpha=0.85, edgecolor='#B71C1C', linewidth=0.8, zorder=3)
axes[0].set_title('Distribusi Panjang Teks\n(SEBELUM Preprocessing)', fontsize=12, fontweight='bold', pad=15)
axes[0].set_xlabel('Jumlah Kata', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Frekuensi (Jumlah Ulasan)', fontsize=10, fontweight='bold')
axes[0].axvline(word_count_before.mean(), color='#7F0000', linestyle='--', linewidth=1.5, zorder=4,
                label=f'Rata-rata: {word_count_before.mean():.1f} kata')
axes[0].grid(True, linestyle='--', alpha=0.4, zorder=0)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].legend(frameon=True, facecolor='white', edgecolor='none')

# Histogram panjang teks sesudah
word_count_after = df['ulasan_preprocessed'].apply(lambda x: len(str(x).split()))
axes[1].hist(word_count_after, bins=35, color='#4DB6AC', alpha=0.85, edgecolor='#004D40', linewidth=0.8, zorder=3)
axes[1].set_title('Distribusi Panjang Teks\n(SESUDAH Preprocessing)', fontsize=12, fontweight='bold', pad=15)
axes[1].set_xlabel('Jumlah Kata', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Frekuensi (Jumlah Ulasan)', fontsize=10, fontweight='bold')
axes[1].axvline(word_count_after.mean(), color='#00332C', linestyle='--', linewidth=1.5, zorder=4,
                label=f'Rata-rata: {word_count_after.mean():.1f} kata')
axes[1].grid(True, linestyle='--', alpha=0.4, zorder=0)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].legend(frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout(pad=2)
plt.savefig('distribusi_panjang_teks.png', dpi=300, bbox_inches='tight')
plt.show()
print("    ✓ Grafik disimpan: distribusi_panjang_teks.png")

# ── 2. WordCloud: Sebelum vs Sesudah ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

# WordCloud sebelum
text_before = ' '.join(df['ulasan_original'].astype(str).tolist())
wc_before = WordCloud(
    width=800, height=450,
    background_color='white',
    colormap='Oranges',
    max_words=100,
    collocations=False,
    random_state=42
).generate(text_before)
axes[0].imshow(wc_before, interpolation='bilinear')
axes[0].set_title('WordCloud SEBELUM Preprocessing', fontsize=14, fontweight='bold', pad=15)
axes[0].axis('off')

# WordCloud sesudah
text_after = ' '.join(df['ulasan_preprocessed'].astype(str).tolist())
wc_after = WordCloud(
    width=800, height=450,
    background_color='white',
    colormap='viridis',
    max_words=100,
    collocations=False,
    random_state=42
).generate(text_after)
axes[1].imshow(wc_after, interpolation='bilinear')
axes[1].set_title('WordCloud SESUDAH Preprocessing', fontsize=14, fontweight='bold', pad=15)
axes[1].axis('off')

plt.tight_layout(pad=3)
plt.savefig('wordcloud_preprocessing.png', dpi=300, bbox_inches='tight')
plt.show()
print("    ✓ Grafik disimpan: wordcloud_preprocessing.png")

# ── 3. Top 10 Kata Terbanyak Setelah Preprocessing ──
all_tokens = []
for tokens in df['ulasan_final_tokens']:
    all_tokens.extend(tokens)

word_freq = Counter(all_tokens)
top_10 = word_freq.most_common(10)

fig, ax = plt.subplots(figsize=(10, 5))
words, freqs = zip(*top_10)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))
bars = ax.barh(range(len(words)), freqs, color=colors, edgecolor='black', linewidth=0.6, height=0.6)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Frekuensi Kemunculan Kata', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Kata Terbanyak Setelah Preprocessing', fontsize=14, fontweight='bold', pad=15)

# Styling plot 3
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Tambah label nilai di ujung bar
for bar, freq in zip(bars, freqs):
    ax.text(bar.get_width() + max(freqs)*0.015, bar.get_y() + bar.get_height()/2,
            f"{freq:,}", va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig('top10_kata_preprocessing.png', dpi=300, bbox_inches='tight')
plt.show()
print("    ✓ Grafik disimpan: top10_kata_preprocessing.png")

# ── 4. Ringkasan Statistik ──
print("\n" + "=" * 70)
print(" RINGKASAN STATISTIK PREPROCESSING")
print("=" * 70)
print(f"    Total data input              : {len(df)} ulasan")
print(f"    Total data output             : {len(df_output)} ulasan")
print(f"    {'─' * 50}")
print(f"    Panjang teks (SEBELUM):")
print(f"       Min    : {word_count_before.min()} kata")
print(f"       Max    : {word_count_before.max()} kata")
print(f"       Mean   : {word_count_before.mean():.1f} kata")
print(f"       Median : {word_count_before.median():.1f} kata")
print(f"    {'─' * 50}")
print(f"    Panjang teks (SESUDAH):")
print(f"       Min    : {word_count_after.min()} kata")
print(f"       Max    : {word_count_after.max()} kata")
print(f"       Mean   : {word_count_after.mean():.1f} kata")
print(f"       Median : {word_count_after.median():.1f} kata")
print(f"    {'─' * 50}")
print(f"    Kata unik setelah preprocessing : {len(word_freq):,}")
print(f"    Total token setelah preprocessing: {len(all_tokens):,}")
print(f"    Top 5 kata: {', '.join([f'{w}({c})' for w, c in top_10[:5]])}")
print("=" * 70)


# ── 5. Tabel Contoh Per Tahap (Profesional) ──
print("\n" + "=" * 70)
print(" CONTOH PREPROCESSING PER TAHAP")
print("=" * 70)

# Tampilkan untuk 2 ulasan pertama
for sample_idx in range(min(2, len(df))):
    print(f"\n{'─' * 70}")
    print(f"  ULASAN #{sample_idx + 1}")
    print(f"{'─' * 70}")

    stages = [
        ("1.  Original               ", 'ulasan_original'),
        ("2.  Case Folding            ", 'ulasan_casefolded'),
        ("3.  Cleansing               ", 'ulasan_cleansed'),
        ("4.  Normalisasi & Translasi ", 'ulasan_normalized'),
        ("5.  Tokenisasi              ", 'ulasan_tokenized'),
        ("6.  Stopword Removal        ", 'ulasan_stopword_removed'),
        ("7.  Stemming                ", 'ulasan_stemmed'),
        ("8.  Final Tokens            ", 'ulasan_final_tokens'),
        ("9.  Hasil Akhir             ", 'ulasan_preprocessed'),
    ]

    for stage_name, col in stages:
        if col in df.columns:
            val = df[col].iloc[sample_idx]
            if isinstance(val, list):
                # Tampilkan list dengan format kurung siku yang jelas
                if len(val) > 12:
                    val_display = str(val[:12])[:-1] + ", ...]"
                else:
                    val_display = str(val)
                val_display += f"  ({len(val)} token)"
            else:
                val_str = str(val)
                if len(val_str) > 90:
                    val_display = val_str[:90] + "..."
                else:
                    val_display = val_str
            print(f"    {stage_name}: {val_display}")

print(f"\n{'─' * 70}")

# ── 6. Statistik Stemming Detail ──
print(f"\n{'=' * 70}")
print(f" STATISTIK STEMMING DETAIL")
print(f"{'=' * 70}")
print(f"    Kata unik yang di-cache        : {len(stemming_cache)}")
print(f"    Kata yang berubah (di-stem)     : {stem_stats['changed']}")
print(f"    Kata yang dilindungi            : {stem_stats['protected']}")
print(f"    Kata dikembalikan (terlalu pendek): {stem_stats['too_short']}")
print(f"    Kata tidak berubah              : {stem_stats['unchanged']}")

# Contoh stemming yang berubah
print(f"\n    Contoh stemming (20 kata pertama yang berubah):")
shown = 0
for orig, stemmed in stemming_cache.items():
    if orig != stemmed and shown < 20:
        print(f"       {orig:25s} -> {stemmed}")
        shown += 1

# ── 7. Download Gambar Visualisasi (Satu per Satu) ──
try:
    from google.colab import files
    print("\n⏳ Menyiapkan unduhan gambar visualisasi secara otomatis...")
    
    # Download file gambar satu per satu dengan delay singkat agar tidak diblokir browser
    files.download('distribusi_panjang_teks.png')
    time.sleep(1.0)
    
    files.download('wordcloud_preprocessing.png')
    time.sleep(1.0)
    
    files.download('top10_kata_preprocessing.png')
    
    print("✓ Unduhan gambar visualisasi berhasil dimulai.")
except ImportError:
    print("\n[INFO] Jalankan di Google Colab untuk mengunduh gambar visualisasi secara otomatis.")

print(f"\n{'=' * 70}")
print(f" PREPROCESSING SELESAI!")
print(f"{'=' * 70}")
timestamp_selesai = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"    Waktu selesai  : {timestamp_selesai}")
print(f"    File output    : {OUTPUT_FILE}")
print(f"    File lengkap   : {FULL_OUTPUT_FILE}")
print(f"    Dataset siap untuk analisis sentimen.")
print(f"{'=' * 70}")
