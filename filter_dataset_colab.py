# ============================================================
# CELL 1: Install dependencies (jalankan di cell terpisah)
# ============================================================
# !pip install langdetect scikit-learn pandas numpy tqdm

# ============================================================
# CELL 2: Filtering Dataset Ulasan Film
# Kriteria Inklusi & Eksklusi
# ============================================================

import re
import pandas as pd
import numpy as np
from collections import Counter
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Reproducibility untuk langdetect
DetectorFactory.seed = 42

# ────────────────────────────────────────────────────────────
# KONFIGURASI - Sesuaikan path file dataset Anda
# ────────────────────────────────────────────────────────────
INPUT_FILE = "dataset_ghost_cell_3000.csv"       # <-- Ganti dengan nama file Anda
TEXT_COLUMN = "ulasan"                  # <-- Ganti dengan nama kolom teks ulasan
RATING_COLUMN = "rating"               # <-- Ganti dengan nama kolom rating bintang
OUTPUT_FILE = "dataset_filtered.csv"    # Nama file output
COSINE_THRESHOLD = 0.85                 # Threshold cosine similarity untuk duplikat

# ────────────────────────────────────────────────────────────
# LOAD DATASET
# ────────────────────────────────────────────────────────────
print("=" * 65)
print("📂 MEMUAT DATASET")
print("=" * 65)

# Uncomment salah satu sesuai format file Anda:
df = pd.read_csv(INPUT_FILE)
# df = pd.read_excel(INPUT_FILE)

print(f"   Total data awal: {len(df)} ulasan")
print(f"   Kolom tersedia : {list(df.columns)}")
print()

# Simpan jumlah awal
total_awal = len(df)

# Pastikan kolom teks dan rating ada
assert TEXT_COLUMN in df.columns, f"Kolom '{TEXT_COLUMN}' tidak ditemukan! Kolom yang ada: {list(df.columns)}"
assert RATING_COLUMN in df.columns, f"Kolom '{RATING_COLUMN}' tidak ditemukan! Kolom yang ada: {list(df.columns)}"

# Bersihkan data: hapus baris dengan teks kosong/NaN
df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df = df[df[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)

# Tracking alasan eksklusi
df['_alasan_eksklusi'] = ''

# ────────────────────────────────────────────────────────────
# FUNGSI-FUNGSI FILTERING
# ────────────────────────────────────────────────────────────

# --- Daftar kata sentimen Indonesia ---
KATA_POSITIF = {
    'bagus', 'keren', 'mantap', 'seru', 'menarik', 'luar biasa', 'hebat',
    'indah', 'cantik', 'memukau', 'spektakuler', 'fantastis', 'brilian',
    'mengesankan', 'terbaik', 'sempurna', 'apik', 'ciamik', 'josss',
    'recommended', 'rekomendasi', 'suka', 'cinta', 'senang', 'puas',
    'terharu', 'menyentuh', 'mengharukan', 'inspiratif', 'epik',
    'masterpiece', 'mahakarya', 'membanggakan', 'berkualitas', 'wow',
    'gokil', 'top', 'oke', 'ok', 'good', 'great', 'best', 'nice',
    'amazing', 'awesome', 'excellent', 'outstanding', 'superb',
    'phenomenal', 'magnificent', 'wonderful', 'brilliant', 'perfect',
    'lucu', 'menghibur', 'menegangkan', 'segar', 'unik', 'kreatif',
    'emosional', 'powerful', 'compelling', 'engaging', 'captivating'
}

KATA_NEGATIF = {
    'jelek', 'buruk', 'payah', 'busuk', 'parah', 'ancur', 'hancur',
    'sampah', 'gagal', 'mengecewakan', 'kecewa', 'bosan', 'membosankan',
    'boring', 'norak', 'lebay', 'alay', 'receh', 'murahan', 'kampungan',
    'aneh', 'gak jelas', 'tidak jelas', 'absurd', 'maksa', 'dipaksakan',
    'overrated', 'overhype', 'overacting', 'worst', 'bad', 'terrible',
    'awful', 'horrible', 'disgusting', 'disappointing', 'trash', 'garbage',
    'rugi', 'menyesal', 'buang waktu', 'sia-sia', 'zonk', 'tidak suka',
    'benci', 'marah', 'kesal', 'menyedihkan', 'menjijikkan', 'memuakkan',
    'gak masuk akal', 'tidak masuk akal', 'plot hole', 'lemah', 'datar',
    'monoton', 'klise', 'pasaran', 'biasa aja', 'biasa saja',
    'tidak menarik', 'gak menarik', 'gak seru', 'tidak seru'
}

# Kata-kata generik yang TIDAK menunjukkan penilaian spesifik
FRASA_GENERIK = [
    r'\bfilm\s+bagus\b', r'\bfilm\s+keren\b', r'\bwajib\s+ditonton\b',
    r'\bwajib\s+nonton\b', r'\brecommended\b', r'\bharus\s+nonton\b',
    r'\bbagus\s+banget\b', r'\bkeren\s+banget\b', r'\bmantap\b',
    r'\btop\b', r'\bthe\s+best\b', r'\bseru\s+banget\b',
    r'\bfilm\s+terbaik\b', r'\bpaling\s+bagus\b', r'\bjoss\b',
    r'\bbiasa\s+aja\b', r'\bbiasa\s+saja\b', r'\bjelek\s+banget\b',
    r'\bfilm\s+jelek\b', r'\b(very\s+)?good\b', r'\bnice\b',
    r'\bgak\s+bagus\b', r'\btidak\s+bagus\b'
]

# Kata kunci aspek film yang menunjukkan penilaian spesifik
ASPEK_FILM = [
    r'\b(akting|acting|peran|karakter|character|pemeran)\b',
    r'\b(cerita|plot|alur|storyline|narasi|skenario|script|naskah)\b',
    r'\b(sinematografi|cinematography|visual|gambar|shot|kamera|angle)\b',
    r'\b(musik|soundtrack|score|sound|suara|audio|efek\s+suara)\b',
    r'\b(sutradara|director|arahan|directing)\b',
    r'\b(ending|akhir|klimaks|twist|konflik|adegan|scene)\b',
    r'\b(dialog|dialogue|percakapan)\b',
    r'\b(CGI|VFX|efek|effect|animasi|animation)\b',
    r'\b(kostum|costume|makeup|setting|latar|lokasi)\b',
    r'\b(genre|horor|horror|komedi|comedy|drama|action|thriller|romance)\b',
    r'\b(penonton|audience|bioskop|cinema|layar)\b',
    r'\b(pesan|moral|makna|tema|theme|filosofi|pelajaran)\b',
    r'\b(emosi|perasaan|feel|mood|atmosfer|suasana|nuansa)\b',
    r'\b(durasi|panjang|tempo|pace|pacing|ritme)\b',
    r'\b(sequel|prequel|franchise|adaptasi|remake|reboot)\b',
]


def hitung_kata(teks):
    """Menghitung jumlah kata dalam teks (hanya alfanumerik)."""
    kata = re.findall(r'\b[a-zA-Z0-9]+\b', teks)
    return len(kata)


def hanya_emotikon_atau_simbol(teks):
    """
    EKSKLUSI 1: Cek apakah ulasan hanya berisi emotikon/karakter tidak bermakna.
    """
    # Hapus emoji, emoticon, simbol, whitespace
    teks_bersih = re.sub(
        r'[\U00010000-\U0010ffff]'  # Emoji
        r'|[\u2600-\u27BF]'          # Misc symbols
        r'|[\uFE00-\uFE0F]'         # Variation selectors
        r'|[\u200d]'                 # Zero width joiner
        r'|[^\w\s]'                  # Non-alphanumeric (selain whitespace)
        r'|\d',                      # Angka saja
        '', teks
    ).strip()
    # Jika setelah dibersihkan tidak ada teks bermakna
    return len(teks_bersih) < 3


def deteksi_bahasa_indonesia(teks):
    """
    INKLUSI 1: Deteksi apakah ulasan ditulis dalam Bahasa Indonesia.
    Menggunakan langdetect + validasi tambahan.
    """
    try:
        bahasa = detect(teks)
        return bahasa == 'id'
    except Exception:
        # Fallback: cek kata-kata umum bahasa Indonesia
        kata_id = [
            'yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk',
            'dari', 'pada', 'adalah', 'tidak', 'akan', 'juga',
            'sudah', 'bisa', 'ada', 'atau', 'ke', 'saya', 'aku',
            'nya', 'sangat', 'sekali', 'film', 'nonton', 'tapi',
            'terlalu', 'banget', 'gak', 'gue', 'gw', 'udah', 'bgt',
            'emang', 'sih', 'dong', 'nih', 'loh', 'lah', 'deh',
            'filmnya', 'ceritanya', 'aktingnya', 'endingnya'
        ]
        teks_lower = teks.lower()
        kata_dalam_teks = teks_lower.split()
        cocok = sum(1 for k in kata_dalam_teks if k in kata_id)
        rasio = cocok / max(len(kata_dalam_teks), 1)
        return rasio >= 0.15


def cek_campuran_bahasa(teks):
    """
    EKSKLUSI 2: Cek apakah ulasan dominan bercampur bahasa asing.
    Return True jika campuran bahasa asing terlalu dominan (>50%).
    """
    # Kata-kata bahasa Inggris umum (yang bukan serapan Indonesia)
    kata_inggris = {
        'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
        'will', 'would', 'could', 'should', 'can', 'may', 'might',
        'this', 'that', 'these', 'those', 'there', 'here', 'where',
        'when', 'what', 'which', 'who', 'whom', 'how', 'why',
        'not', 'but', 'and', 'or', 'because', 'although', 'though',
        'if', 'then', 'than', 'very', 'really', 'just', 'also',
        'about', 'after', 'before', 'between', 'into', 'through',
        'during', 'without', 'within', 'along', 'following',
        'like', 'love', 'hate', 'think', 'know', 'want', 'need',
        'make', 'take', 'give', 'tell', 'say', 'said', 'come',
        'going', 'went', 'been', 'being', 'their', 'them', 'they',
        'she', 'her', 'him', 'his', 'its', 'our', 'your', 'my',
        'we', 'you', 'it', 'he', 'me', 'much', 'many', 'some',
        'any', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'such', 'only', 'same', 'so', 'no',
        'well', 'even', 'still', 'already', 'always', 'never',
        'ever', 'often', 'sometimes', 'usually', 'too', 'enough',
        'movie', 'watch', 'watched', 'watching', 'seen', 'story',
        'character', 'scene', 'plot', 'acting', 'actor', 'actress'
    }
    # Kata yang bisa dianggap serapan/umum dipakai di Indonesia
    kata_serapan_wajar = {
        'film', 'rating', 'review', 'genre', 'plot', 'scene',
        'acting', 'character', 'soundtrack', 'ending', 'twist',
        'score', 'sequel', 'prequel', 'remake', 'franchise',
        'action', 'horror', 'thriller', 'drama', 'comedy',
        'romance', 'fantasy', 'animation', 'director', 'cinema',
        'box office', 'CGI', 'VFX', 'recommended', 'overrated',
        'masterpiece', 'epic', 'classic', 'original', 'cast',
        'trailer', 'spoiler', 'hype', 'boring', 'good', 'bad',
        'best', 'worst', 'top', 'nice', 'cool', 'great', 'ok',
        'so', 'no', 'yes'
    }

    kata_teks = re.findall(r'\b[a-zA-Z]+\b', teks.lower())
    if len(kata_teks) < 5:
        return False

    jumlah_asing = 0
    for kata in kata_teks:
        if kata in kata_inggris and kata not in kata_serapan_wajar:
            jumlah_asing += 1

    rasio_asing = jumlah_asing / len(kata_teks)
    return rasio_asing > 0.50


def punya_penilaian_spesifik(teks):
    """
    INKLUSI 2: Cek apakah ulasan memuat penilaian spesifik terhadap elemen film,
    bukan hanya pernyataan umum generik.
    """
    teks_lower = teks.lower()

    # Cek apakah ada aspek film yang disebutkan
    ada_aspek = any(re.search(pola, teks_lower) for pola in ASPEK_FILM)
    if ada_aspek:
        return True

    # Cek apakah ulasan hanya berisi frasa generik
    teks_tanpa_generik = teks_lower
    for pola in FRASA_GENERIK:
        teks_tanpa_generik = re.sub(pola, '', teks_tanpa_generik).strip()

    # Jika setelah hapus frasa generik masih ada konten substansial (>5 kata)
    kata_sisa = re.findall(r'\b[a-zA-Z]+\b', teks_tanpa_generik)
    if len(kata_sisa) >= 5:
        return True

    return False


def hitung_skor_sentimen(teks):
    """
    Hitung skor sentimen sederhana berbasis kata kunci Indonesia.
    Return: (skor_positif, skor_negatif, sentimen_label)
    """
    teks_lower = teks.lower()
    kata_teks = set(re.findall(r'\b[a-zA-Z]+\b', teks_lower))

    # Cek juga bigram
    kata_list = re.findall(r'\b[a-zA-Z]+\b', teks_lower)
    bigrams = {f"{kata_list[i]} {kata_list[i+1]}" for i in range(len(kata_list)-1)} if len(kata_list) > 1 else set()

    semua = kata_teks | bigrams

    skor_pos = len(semua & KATA_POSITIF)
    skor_neg = len(semua & KATA_NEGATIF)

    # Deteksi negasi
    pola_negasi = [r'\btidak\s+\w+', r'\bbukan\s+\w+', r'\bgak\s+\w+',
                   r'\bnggak\s+\w+', r'\bga\s+\w+', r'\btdk\s+\w+']
    for pola in pola_negasi:
        negasi = re.findall(pola, teks_lower)
        for n in negasi:
            kata_setelah = n.split()[-1]
            if kata_setelah in KATA_POSITIF:
                skor_pos -= 1
                skor_neg += 1
            elif kata_setelah in KATA_NEGATIF:
                skor_neg -= 1
                skor_pos += 1

    skor_pos = max(skor_pos, 0)
    skor_neg = max(skor_neg, 0)

    if skor_pos > skor_neg:
        label = 'positif'
    elif skor_neg > skor_pos:
        label = 'negatif'
    else:
        label = 'netral'

    return skor_pos, skor_neg, label


def cek_inkonsistensi_rating_sentimen(teks, rating, max_rating=5):
    """
    EKSKLUSI 4: Cek inkonsistensi ekstrem antara rating bintang dan sentimen teks.
    Return True jika inkonsisten (harus dieksklusi).
    """
    skor_pos, skor_neg, sentimen = hitung_skor_sentimen(teks)

    # Jika tidak ada sinyal sentimen yang jelas, jangan eksklusi
    total_sinyal = skor_pos + skor_neg
    if total_sinyal < 2:
        return False

    # Rating tinggi (4-5) tapi sentimen sangat negatif
    if rating >= (max_rating - 1):
        if sentimen == 'negatif' and skor_neg >= 3 and skor_pos == 0:
            return True

    # Rating rendah (1-2) tapi sentimen sangat positif
    if rating <= 2:
        if sentimen == 'positif' and skor_pos >= 3 and skor_neg == 0:
            return True

    return False


# ────────────────────────────────────────────────────────────
# PROSES FILTERING BERTAHAP
# ────────────────────────────────────────────────────────────
log_eksklusi = {
    'emotikon_simbol': 0,
    'bukan_bahasa_id': 0,
    'campuran_bahasa': 0,
    'kurang_15_kata': 0,
    'tidak_spesifik': 0,
    'duplikat_cosine': 0,
    'inkonsistensi_rating': 0
}

print("=" * 65)
print("🔍 TAHAP 1: Eksklusi ulasan emotikon/karakter tidak bermakna")
print("=" * 65)
mask_emotikon = df[TEXT_COLUMN].apply(hanya_emotikon_atau_simbol)
log_eksklusi['emotikon_simbol'] = mask_emotikon.sum()
df.loc[mask_emotikon, '_alasan_eksklusi'] = 'emotikon/simbol saja'
df_valid = df[~mask_emotikon].reset_index(drop=True)
print(f"   ❌ Dieksklusi: {log_eksklusi['emotikon_simbol']} ulasan")
print(f"   ✅ Tersisa   : {len(df_valid)} ulasan\n")

print("=" * 65)
print("🔍 TAHAP 2: Inklusi - Bahasa Indonesia saja")
print("=" * 65)
mask_bahasa = []
for i, teks in enumerate(tqdm(df_valid[TEXT_COLUMN], desc="   Deteksi bahasa")):
    mask_bahasa.append(deteksi_bahasa_indonesia(teks))
mask_bahasa = pd.Series(mask_bahasa)
log_eksklusi['bukan_bahasa_id'] = (~mask_bahasa).sum()
df_valid = df_valid[mask_bahasa.values].reset_index(drop=True)
print(f"   ❌ Dieksklusi: {log_eksklusi['bukan_bahasa_id']} ulasan (bukan bahasa Indonesia)")
print(f"   ✅ Tersisa   : {len(df_valid)} ulasan\n")

print("=" * 65)
print("🔍 TAHAP 3: Eksklusi - Campuran bahasa asing dominan")
print("=" * 65)
mask_campuran = df_valid[TEXT_COLUMN].apply(cek_campuran_bahasa)
log_eksklusi['campuran_bahasa'] = mask_campuran.sum()
df_valid = df_valid[~mask_campuran].reset_index(drop=True)
print(f"   ❌ Dieksklusi: {log_eksklusi['campuran_bahasa']} ulasan (campuran bahasa)")
print(f"   ✅ Tersisa   : {len(df_valid)} ulasan\n")

print("=" * 65)
print("🔍 TAHAP 4: Inklusi - Minimal 10 kata")
print("=" * 65)
df_valid['_jumlah_kata'] = df_valid[TEXT_COLUMN].apply(hitung_kata)
mask_kata = df_valid['_jumlah_kata'] >= 10
log_eksklusi['kurang_15_kata'] = (~mask_kata).sum()
df_valid = df_valid[mask_kata].reset_index(drop=True)
print(f"   ❌ Dieksklusi: {log_eksklusi['kurang_15_kata']} ulasan (< 10 kata)")
print(f"   ✅ Tersisa   : {len(df_valid)} ulasan\n")

print("=" * 65)
print("🔍 TAHAP 5: Inklusi - Penilaian spesifik terhadap elemen film")
print("=" * 65)
mask_spesifik = df_valid[TEXT_COLUMN].apply(punya_penilaian_spesifik)
log_eksklusi['tidak_spesifik'] = (~mask_spesifik).sum()
df_valid = df_valid[mask_spesifik].reset_index(drop=True)
print(f"   ❌ Dieksklusi: {log_eksklusi['tidak_spesifik']} ulasan (tidak spesifik)")
print(f"   ✅ Tersisa   : {len(df_valid)} ulasan\n")

print("=" * 65)
print("🔍 TAHAP 6: Eksklusi - Duplikat (Cosine Similarity > 0.85)")
print("=" * 65)
print("   ⏳ Menghitung TF-IDF dan cosine similarity...")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df_valid[TEXT_COLUMN])

# Hitung cosine similarity secara batch untuk efisiensi memori
duplikat_idx = set()
batch_size = 500
n = tfidf_matrix.shape[0]

for start in tqdm(range(0, n, batch_size), desc="   Cek duplikat"):
    end = min(start + batch_size, n)
    sim_batch = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)

    for i in range(start, end):
        if i in duplikat_idx:
            continue
        for j in range(i + 1, n):
            if j in duplikat_idx:
                continue
            if sim_batch[i - start, j] > COSINE_THRESHOLD:
                duplikat_idx.add(j)

log_eksklusi['duplikat_cosine'] = len(duplikat_idx)
mask_duplikat = ~df_valid.index.isin(duplikat_idx)
df_valid = df_valid[mask_duplikat].reset_index(drop=True)
print(f"   ❌ Dieksklusi: {log_eksklusi['duplikat_cosine']} ulasan (duplikat)")
print(f"   ✅ Tersisa   : {len(df_valid)} ulasan\n")

print("=" * 65)
print("🔍 TAHAP 7: Eksklusi - Inkonsistensi rating vs sentimen")
print("=" * 65)
mask_inkonsisten = df_valid.apply(
    lambda row: cek_inkonsistensi_rating_sentimen(row[TEXT_COLUMN], row[RATING_COLUMN]),
    axis=1
)
log_eksklusi['inkonsistensi_rating'] = mask_inkonsisten.sum()
df_valid = df_valid[~mask_inkonsisten].reset_index(drop=True)
print(f"   ❌ Dieksklusi: {log_eksklusi['inkonsistensi_rating']} ulasan (inkonsisten)")
print(f"   ✅ Tersisa   : {len(df_valid)} ulasan\n")

# ────────────────────────────────────────────────────────────
# RINGKASAN HASIL FILTERING
# ────────────────────────────────────────────────────────────
# Hapus kolom internal
df_final = df_valid.drop(columns=['_alasan_eksklusi', '_jumlah_kata'], errors='ignore')

total_akhir = len(df_final)
total_eksklusi = total_awal - total_akhir

print("=" * 65)
print("📊 RINGKASAN HASIL FILTERING")
print("=" * 65)
print(f"   📌 Total data awal          : {total_awal}")
print(f"   ─────────────────────────────────────────────")
print(f"   ❌ Emotikon/simbol saja      : {log_eksklusi['emotikon_simbol']}")
print(f"   ❌ Bukan bahasa Indonesia    : {log_eksklusi['bukan_bahasa_id']}")
print(f"   ❌ Campuran bahasa dominan   : {log_eksklusi['campuran_bahasa']}")
print(f"   ❌ Kurang dari 10 kata       : {log_eksklusi['kurang_15_kata']}")
print(f"   ❌ Tidak ada penilaian spesifik: {log_eksklusi['tidak_spesifik']}")
print(f"   ❌ Duplikat (cosine > {COSINE_THRESHOLD})   : {log_eksklusi['duplikat_cosine']}")
print(f"   ❌ Inkonsistensi rating-sentimen: {log_eksklusi['inkonsistensi_rating']}")
print(f"   ─────────────────────────────────────────────")
print(f"   ❌ Total dieksklusi          : {total_eksklusi} ({total_eksklusi/total_awal*100:.1f}%)")
print(f"   ✅ Total data akhir          : {total_akhir} ({total_akhir/total_awal*100:.1f}%)")
print()

# ────────────────────────────────────────────────────────────
# SIMPAN DATASET HASIL FILTERING
# ────────────────────────────────────────────────────────────
df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"💾 Dataset tersaring disimpan ke: {OUTPUT_FILE}")
print(f"   Jumlah kolom: {len(df_final.columns)}")
print(f"   Jumlah baris: {len(df_final)}")
print()

# Preview beberapa data
print("=" * 65)
print("👀 PREVIEW 5 DATA PERTAMA HASIL FILTERING")
print("=" * 65)
for i, row in df_final.head(5).iterrows():
    teks = row[TEXT_COLUMN]
    rating = row[RATING_COLUMN]
    preview = teks[:120] + "..." if len(teks) > 120 else teks
    print(f"\n   [{i+1}] ⭐ Rating: {rating}")
    print(f"       📝 {preview}")

print("\n✅ Proses filtering selesai!")
