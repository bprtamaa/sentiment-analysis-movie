"""
Letterboxd Review Scraper — Ghost in the Cell (2026)
Optimized for Google Colab

Target   : 3000+ ulasan bahasa Indonesia
Periode  : 16 April 2026 – 5 Juni 2026
Output   : dataset_ghost_cell_3000.csv
Platform : Google Colab (SeleniumBase UC Mode)

FILTER:
  - Ulasan ditulis dalam bahasa Indonesia
  - Ulasan duplikat tidak dimasukkan (exact match + cosine similarity > 0.85)

SOLUSI CLOUDFLARE:
  - Pakai SeleniumBase UC Mode (uc=True) + xvfb
  - uc_open_with_reconnect() untuk bypass Cloudflare Turnstile

FITUR RESUME:
  - Progress disimpan di progress_ghost_cell.json
  - Saat dijalankan ulang, lanjut dari strategi & halaman terakhir
  - Data CSV juga di-resume (tidak scraping ulang)

CELL COLAB:
  Cell 1: Install dependencies
  Cell 2: Jalankan script ini
"""

# ══════════════════════════════════════════════════════════
#  CELL 1: INSTALL DEPENDENCIES (jalankan terpisah di Colab)
# ══════════════════════════════════════════════════════════
# !sudo apt update && sudo apt upgrade -y
# !curl -o google-chrome-stable_current_amd64.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# !sudo apt install ./google-chrome-stable_current_amd64.deb -y
# !rm google-chrome-stable_current_amd64.deb
# !apt-get install -y xvfb
# !pip install seleniumbase -U
# !pip install beautifulsoup4 lxml
# !pip install scikit-learn    # untuk cosine similarity deduplication

# ══════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════
import sys
import time
import random
import csv
import re
import os
import json
from datetime import datetime, date

try:
    from seleniumbase import SB
    print("✓ seleniumbase tersedia")
except ImportError:
    raise RuntimeError(
        "\n\nseleniumbase belum diinstall!\n"
        "Jalankan cell berikut dulu:\n"
        "  !pip install seleniumbase -U\n"
        "  !apt-get install -y xvfb\n"
    )

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    SKLEARN_AVAILABLE = True
    print("✓ scikit-learn tersedia (cosine similarity aktif)")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ scikit-learn belum diinstall, cosine similarity dedup dinonaktifkan")
    print("  Install: !pip install scikit-learn")

from selenium.webdriver.common.by import By
from urllib3.exceptions import ReadTimeoutError
from selenium.common.exceptions import TimeoutException, WebDriverException

# ══════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════
MAX_PAGE       = 256        # maksimal halaman per strategi (limit Letterboxd)
TARGET_TOTAL   = 4000       # target scraping (buffer untuk post-processing)
FINAL_TARGET   = 3500       # target akhir setelah filtering
OUTPUT_FILE    = "dataset_ghost_cell_3000.csv"
PROGRESS_FILE  = "progress_ghost_cell.json"  # file progress untuk resume
AUTOSAVE_EVERY = 10         # auto-save tiap N data baru

# Retry / reconnect settings
MAX_RETRIES_PER_PAGE   = 5   # max retry per halaman sebelum skip
MAX_CONSECUTIVE_FAIL   = 5   # max gagal berturut sebelum restart browser
BROWSER_RESTART_LIMIT  = 10  # max restart browser sebelum berhenti total
RETRY_BACKOFF_BASE     = 8   # base delay (seconds) untuk exponential backoff
RETRY_BACKOFF_MAX      = 90  # max delay (seconds) untuk backoff

# Timeout settings (penting untuk Colab yang koneksinya lebih lambat)
PAGE_LOAD_TIMEOUT      = 90  # timeout load halaman (detik) — dinaikkan untuk Colab
UC_RECONNECT_TIME      = 12  # waktu reconnect untuk uc_open_with_reconnect
UC_RECONNECT_TIME_LONG = 20  # waktu reconnect lebih lama (untuk retry)

# Periode tanggal ulasan
DATE_START = date(2026, 4, 16)
DATE_END   = date(2026, 6, 5)

# Film target
FILM_SLUG  = "ghost-in-the-cell"
FILM_TITLE = "Ghost in the Cell (2026)"

# Cosine similarity threshold untuk deduplication
COSINE_SIM_THRESHOLD = 0.85

# Path sistem Letterboxd yang BUKAN username
SKIP_PATHS = {
    "film", "list", "reviews", "diary", "watchlist", "films",
    "likes", "members", "activity", "settings", "sign-in", "create-account",
    "journal", "news", "pro", "about", "contact", "legal",
}

# ══════════════════════════════════════════════════════════
#  KATA-KATA BAHASA INDONESIA (untuk deteksi bahasa)
# ══════════════════════════════════════════════════════════
INDO_WORDS = {
    # Kata umum
    "yang","dan","di","ke","dari","ini","itu","ada","tidak","bisa",
    "dengan","untuk","sangat","banget","aku","saya","gue","gw",
    "nggak","enggak","gak","ga","udah","sudah","belum","lagi","juga",
    "lebih","mau","harus","jadi","sama","tapi","atau","buat","punya",
    "bikin","kayak","gimana","emang","memang","sih","nih","dong","kok",
    "kalo","kalau","keren","bagus","jelek","seru","serem","seram","ngeri",
    # Kata terkait film/review
    "horor","cerita","karakter","alur","nonton","penonton","ending",
    "adegan","nuansa","visual","musik","biasa","lumayan","mantap","parah",
    "wah","eh","lho","lah","pun","pula","bahkan","apalagi","malah",
    "justru","ternyata","jadinya","karena","soalnya","bgt",
    "filmnya","bagusnya","jeleknya","plotnya","aktingnya","horornya",
    "menurut","banyak","sekali","paling","cukup","kurang",
    # Kata tambahan
    "akan","bukan","hanya","saja","masih","sedang","telah","mereka",
    "kita","kami","dia","ia","nya","mu","ku","se","ter","ber","men",
    "pada","oleh","tentang","antara","setelah","sebelum","ketika",
    "namun","walau","meski","agar","supaya","sehingga","bahwa",
    "seperti","bagai","daripada","hingga","sampai","tanpa","selain",
    "suka","senang","takut","sedih","marah","kecewa","puas","luar",
    "dapet","gila","anjir","asli","mantul",
    "kaget","tegang","deg","degan","ngakak","lucu","terharu",
    "recommended","nontonin","ditonton","menonton","kesel","sebel",
    "penasaran","sumpah","cakep","ganteng","cantik","kece",
    "anjay","baper","gabisa","gatau","gaada","gamau","gabole",
    "pengen","kepengen","kepingin","pingin","pengin",
    "terlalu","agak","rada","sedikit","dikit","doang","aja","cuma",
    "deh","yah","yaa","yaaa","gitu","gini","begitu","begini",
    "emosi","perasaan","suasana","atmosfer","menegangkan","mencekam",
    "menakutkan","menyeramkan","mengharukan","menyentuh","membosankan",
    "bagian","scene","akting","aktor","aktris","sutradara","sinematografi",
    "adalah","merupakan","menjadi","sebagai","secara","terhadap","melalui",
    "sebuah","suatu","beberapa","semua","setiap","tiap","seluruh",
    "sangatlah","sekali","amat","benar","betul","sungguh",
    "boleh","perlu","wajib","layak","patut","pantas",
    "berasa","rasanya","terasa","kesan","pesan","keseluruhan",
    "menurut","menurutku","menurutnya","menurut saya",
    "pokoknya","intinya","kesimpulannya","overall",
}

# ══════════════════════════════════════════════════════════
#  KATA SERAPAN / LOAN WORDS (tidak dihitung sebagai asing)
#  Kata-kata Inggris yang lazim dalam diskusi film Indonesia
# ══════════════════════════════════════════════════════════
LOAN_WORDS = {
    # Istilah perfilman
    "film", "movie", "scene", "ending", "plot", "twist", "genre",
    "horror", "action", "drama", "thriller", "romance", "comedy",
    "soundtrack", "score", "cgi", "vfx", "acting", "cast",
    "rating", "review", "spoiler", "trailer", "sequel", "prequel",
    "remake", "reboot", "franchise", "cinema", "cinematic",
    "director", "producer", "actor", "actress",
    "recommended", "masterpiece", "blockbuster", "box", "office",
    "flashback", "cliffhanger", "cameo", "debut",
    "mainstream", "indie", "independent",
    "jump", "scare", "jumpscare",
    "screening", "premiere", "premier",
    "storyline", "plotline", "subplot",
    "character", "protagonist", "antagonist",
    "climax", "anti", "hero", "heroine",
    "special", "effect", "effects", "visual",
    "animation", "animated", "live",
    "script", "screenplay",
    # Istilah umum yang sering dipakai
    "online", "streaming", "download", "upload",
    "comment", "like", "share", "post",
    "good", "bad", "best", "worst", "great", "nice",
    "okay", "ok", "so", "but", "and", "the", "of", "in",
    "no", "yes", "not", "very", "really", "much",
    "love", "hate", "feel", "think",
    "worth", "watching", "watch",
    "btw", "fyi", "imo", "imho", "lol", "omg", "wow",
    "level", "quality", "concept", "classic",
    "perfect", "epic", "legendary", "iconic",
    "hype", "overrated", "underrated",
    "star", "stars",
}



# ══════════════════════════════════════════════════════════
#  BULAN (untuk parsing tanggal)
# ══════════════════════════════════════════════════════════
MONTH_MAP = {
    "jan": 1, "january": 1, "januari": 1,
    "feb": 2, "february": 2, "februari": 2,
    "mar": 3, "march": 3, "maret": 3,
    "apr": 4, "april": 4,
    "may": 5, "mei": 5,
    "jun": 6, "june": 6, "juni": 6,
    "jul": 7, "july": 7, "juli": 7,
    "aug": 8, "august": 8, "agustus": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "oktober": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12, "desember": 12,
}


# ══════════════════════════════════════════════════════════
#  FUNGSI UTILITAS DASAR
# ══════════════════════════════════════════════════════════
def count_words(text: str) -> int:
    """Hitung jumlah kata dalam teks."""
    t = text.strip()
    if not t:
        return 0
    return len(t.split())


def tokenize(text: str) -> list:
    """Tokenisasi teks: lowercase, hapus tanda baca, split."""
    t = text.lower()
    for ch in ".,;:!?()'\"\\/-—[]{}@#$%^&*_+=~`|<>""''…":
        t = t.replace(ch, " ")
    return [w for w in t.split() if w]


def clean_review(text: str) -> str:
    """Bersihkan noise dari teks review."""
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r'^(\d+|No)\s+likes?', ln, re.IGNORECASE):
            continue
        if re.match(r'^Report this review$', ln, re.IGNORECASE):
            continue
        if re.match(r'^\d+ comments?$', ln, re.IGNORECASE):
            continue
        lines.append(ln)
    result = " ".join(lines).strip()
    result = re.sub(r'\s*(\d+|No)\s+likes?\.*\s*$', '', result, flags=re.IGNORECASE).strip()
    result = re.sub(r'\s+', ' ', result)
    return result


# ══════════════════════════════════════════════════════════
#  PROGRESS FILE — UNTUK RESUME
# ══════════════════════════════════════════════════════════
def save_progress(strategy_index, page, total_data, filename=PROGRESS_FILE):
    """Simpan progress scraping ke file JSON agar bisa di-resume."""
    progress = {
        "strategy_index": strategy_index,
        "page": page,
        "total_data": total_data,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ⚠ Gagal simpan progress: {e}")


def load_progress(filename=PROGRESS_FILE):
    """
    Load progress terakhir.
    Mengembalikan (strategy_index, page) atau (0, 1) jika tidak ada.
    """
    if not os.path.exists(filename):
        return 0, 1

    try:
        with open(filename, "r", encoding="utf-8") as f:
            progress = json.load(f)
        si = progress.get("strategy_index", 0)
        pg = progress.get("page", 1)
        ts = progress.get("timestamp", "?")
        td = progress.get("total_data", "?")
        print(f"  📌 Progress ditemukan: strategi #{si}, halaman {pg}")
        print(f"     Data saat itu: {td} | Waktu: {ts}")
        return si, pg
    except Exception as e:
        print(f"  ⚠ Gagal load progress: {e}")
        return 0, 1


def clear_progress(filename=PROGRESS_FILE):
    """Hapus file progress setelah scraping selesai."""
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
#  FUNGSI PARSING & FILTER TANGGAL
# ══════════════════════════════════════════════════════════
def parse_review_date(date_str: str):
    """
    Parse string tanggal dari Letterboxd ke objek date.
    Format yang didukung:
      - ISO: "2026-04-20", "2026-04-20T12:30:00"
      - Teks: "20 Apr 2026", "Apr 20, 2026", "May 1, 2026"
      - Pendek: "20 Apr" (tanpa tahun, asumsi 2026)
    Mengembalikan date atau None jika gagal.
    """
    if not date_str or not date_str.strip():
        return None

    s = date_str.strip()

    # Format ISO: 2026-04-20 atau 2026-04-20T...
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})', s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Format: "20 Apr 2026" atau "20 April 2026"
    m = re.match(r'^(\d{1,2})\s+(\w+)\s+(\d{4})', s)
    if m:
        day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if month_str in MONTH_MAP:
            try:
                return date(year, MONTH_MAP[month_str], day)
            except ValueError:
                pass

    # Format: "Apr 20, 2026" atau "April 20, 2026"
    m = re.match(r'^(\w+)\s+(\d{1,2}),?\s+(\d{4})', s)
    if m:
        month_str, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        if month_str in MONTH_MAP:
            try:
                return date(year, MONTH_MAP[month_str], day)
            except ValueError:
                pass

    # Format pendek: "20 Apr" (tanpa tahun, asumsi 2026)
    m = re.match(r'^(\d{1,2})\s+(\w+)$', s)
    if m:
        day, month_str = int(m.group(1)), m.group(2).lower()
        if month_str in MONTH_MAP:
            try:
                return date(2026, MONTH_MAP[month_str], day)
            except ValueError:
                pass

    # Format pendek: "Apr 20" (tanpa tahun)
    m = re.match(r'^(\w+)\s+(\d{1,2})$', s)
    if m:
        month_str, day = m.group(1).lower(), int(m.group(2))
        if month_str in MONTH_MAP:
            try:
                return date(2026, MONTH_MAP[month_str], day)
            except ValueError:
                pass

    return None


def is_in_date_range(review_date, start=DATE_START, end=DATE_END) -> bool:
    """Cek apakah tanggal ulasan berada dalam periode yang diinginkan."""
    if review_date is None:
        return False
    return start <= review_date <= end


# ══════════════════════════════════════════════════════════
#  DETEKSI BAHASA INDONESIA
# ══════════════════════════════════════════════════════════
def is_indonesian(text: str) -> bool:
    """
    Deteksi apakah teks berbahasa Indonesia.
    Untuk teks ≥10 kata, minimal 3 kata Indo harus ditemukan.
    """
    words = set(tokenize(text))
    hit = len(words & INDO_WORDS)
    # Untuk teks yang lebih panjang, butuh lebih banyak kata Indo
    min_hits = 3 if len(words) >= 10 else 2
    return hit >= min_hits





# ══════════════════════════════════════════════════════════
#  DEDUPLICATION — Cosine Similarity
# ══════════════════════════════════════════════════════════
def remove_near_duplicates(data_list, threshold=COSINE_SIM_THRESHOLD):
    """
    Hapus ulasan duplikat/near-duplicate menggunakan cosine similarity
    pada TF-IDF vectors (Foswanto et al., 2024).
    Simpan ulasan yang muncul pertama, hapus yang datang setelahnya.
    Mengembalikan (filtered_list, jumlah_dihapus).
    """
    if not data_list or len(data_list) < 2 or not SKLEARN_AVAILABLE:
        return data_list, 0

    print(f"\n  Menghitung cosine similarity untuk {len(data_list)} ulasan...")
    texts = [d["ulasan"] for d in data_list]

    try:
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=0.95,
            max_features=15000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    except Exception as e:
        print(f"  ⚠ TF-IDF gagal: {e}")
        return data_list, 0

    to_remove = set()
    n = len(texts)

    # Proses per baris untuk efisiensi memori
    for i in range(n - 1):
        if i in to_remove:
            continue
        if i % 500 == 0 and i > 0:
            print(f"    ... diproses {i}/{n} ({len(to_remove)} duplikat ditemukan)")

        # Bandingkan dokumen i dengan semua dokumen setelahnya
        sim_scores = sk_cosine_similarity(
            tfidf_matrix[i:i+1],
            tfidf_matrix[i+1:]
        )[0]

        for j_offset, score in enumerate(sim_scores):
            j = i + 1 + j_offset
            if j not in to_remove and score > threshold:
                to_remove.add(j)

    removed_count = len(to_remove)
    filtered = [d for idx, d in enumerate(data_list) if idx not in to_remove]

    print(f"  ✓ Cosine similarity selesai: {removed_count} duplikat dihapus "
          f"(threshold={threshold})")

    return filtered, removed_count


# ══════════════════════════════════════════════════════════
#  BALANCING RATING DISTRIBUTION (55% >= 3.5, 45% < 3.5)
# ══════════════════════════════════════════════════════════
def balance_rating_distribution(data_list, target_total=FINAL_TARGET,
                                 high_ratio=0.55, low_ratio=0.45):
    """
    Seimbangkan distribusi rating:
      - 55% ulasan rating >= 3.5 (high rating)
      - 45% ulasan rating < 3.5  (low rating)
    Ulasan tanpa rating dimasukkan ke grup low.
    Mengembalikan (balanced_list, stats_dict).
    """
    if not data_list:
        return data_list, {}

    # Pisahkan berdasarkan rating
    high_group = []  # rating >= 3.5
    low_group = []   # rating < 3.5 atau tanpa rating

    for d in data_list:
        rating_str = d.get("rating", "").strip()
        if rating_str:
            try:
                rating_val = float(rating_str)
                if rating_val >= 3.5:
                    high_group.append(d)
                else:
                    low_group.append(d)
            except ValueError:
                low_group.append(d)
        else:
            low_group.append(d)

    print(f"\n  [BALANCE] Rating >= 3.5 : {len(high_group)} ulasan")
    print(f"  [BALANCE] Rating < 3.5  : {len(low_group)} ulasan")

    # Hitung kuota ideal
    high_quota = int(target_total * high_ratio)
    low_quota = target_total - high_quota  # sisanya untuk low

    print(f"  [BALANCE] Kuota target  : high={high_quota} ({high_ratio*100:.0f}%), "
          f"low={low_quota} ({low_ratio*100:.0f}%)")

    # Sampling: ambil sesuai kuota, jika kurang ambil semua
    random.seed(42)  # reproducible

    if len(high_group) >= high_quota:
        sampled_high = random.sample(high_group, high_quota)
    else:
        sampled_high = high_group[:]
        print(f"  [BALANCE] ⚠ High rating kurang: {len(high_group)}/{high_quota}")

    if len(low_group) >= low_quota:
        sampled_low = random.sample(low_group, low_quota)
    else:
        sampled_low = low_group[:]
        print(f"  [BALANCE] ⚠ Low rating kurang: {len(low_group)}/{low_quota}")

    # Jika salah satu grup kurang, isi kekurangan dari grup lain
    total_sampled = len(sampled_high) + len(sampled_low)
    if total_sampled < target_total:
        deficit = target_total - total_sampled
        if len(sampled_high) < high_quota:
            # High kurang, ambil tambahan dari low
            remaining_low = [d for d in low_group if d not in sampled_low]
            extra = remaining_low[:deficit]
            sampled_low.extend(extra)
            print(f"  [BALANCE] +{len(extra)} tambahan dari low untuk kompensasi")
        elif len(sampled_low) < low_quota:
            # Low kurang, ambil tambahan dari high
            remaining_high = [d for d in high_group if d not in sampled_high]
            extra = remaining_high[:deficit]
            sampled_high.extend(extra)
            print(f"  [BALANCE] +{len(extra)} tambahan dari high untuk kompensasi")

    # Gabungkan dan shuffle
    balanced = sampled_high + sampled_low
    random.shuffle(balanced)

    # Statistik
    final_high = len(sampled_high)
    final_low = len(sampled_low)
    final_total = len(balanced)
    stats = {
        "high_count": final_high,
        "low_count": final_low,
        "total": final_total,
        "high_pct": (final_high / final_total * 100) if final_total > 0 else 0,
        "low_pct": (final_low / final_total * 100) if final_total > 0 else 0,
    }

    print(f"  [BALANCE] Hasil akhir   : {final_high} high ({stats['high_pct']:.1f}%) + "
          f"{final_low} low ({stats['low_pct']:.1f}%) = {final_total} total")

    return balanced, stats


# ══════════════════════════════════════════════════════════
#  POST-PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════
def post_process_pipeline(data_list):
    """
    Jalankan post-processing:
      1. Deduplication via cosine similarity
      2. Balancing rating distribution (55% >= 3.5, 45% < 3.5)
    Mengembalikan list yang sudah difilter.
    """
    print("\n" + "=" * 60)
    print("  POST-PROCESSING PIPELINE")
    print("=" * 60)
    print(f"  Data masuk: {len(data_list)} ulasan\n")

    if not data_list:
        return data_list

    # ── Cosine similarity dedup ──────────────────────────
    data_list, removed_dup = remove_near_duplicates(data_list)
    print(f"  [DEDUP] Cosine sim duplikat : -{removed_dup:>4} | sisa: {len(data_list)}")

    # ── Balancing rating distribution ────────────────────
    data_list, balance_stats = balance_rating_distribution(data_list)

    # ── Ringkasan ────────────────────────────────────────
    print(f"\n  Total dibuang (dedup)   : {removed_dup}")
    if balance_stats:
        print(f"  Rating >= 3.5 (55%)    : {balance_stats['high_count']} "
              f"({balance_stats['high_pct']:.1f}%)")
        print(f"  Rating < 3.5  (45%)    : {balance_stats['low_count']} "
              f"({balance_stats['low_pct']:.1f}%)")
    print(f"  Data akhir             : {len(data_list)} ulasan")
    print("=" * 60)

    return data_list


# ══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS — SELENIUM / SCRAPING
# ══════════════════════════════════════════════════════════
def rand_sleep(lo=2.0, hi=5.0):
    time.sleep(random.uniform(lo, hi))


def human_scroll(sb):
    """Scroll halaman secara manusiawi via SeleniumBase."""
    try:
        total_h = sb.execute_script("return document.body.scrollHeight")
        step = random.randint(300, 500)
        pos = 0
        while pos < total_h:
            pos += step
            sb.execute_script(f"window.scrollTo(0, {pos});")
            time.sleep(random.uniform(0.1, 0.3))
        sb.execute_script(f"window.scrollTo(0, {total_h - random.randint(100, 300)});")
        time.sleep(0.3)
    except Exception:
        pass


def extract_rating(el) -> str:
    """Ambil rating dari elemen."""
    cls = el.get_attribute("class") or ""
    m = re.search(r'rated-(\d+)', cls)
    if m:
        return str(int(m.group(1)) / 2)
    txt = el.text.strip()
    stars = txt.count('★') + (0.5 if '½' in txt else 0)
    return str(stars) if stars > 0 else ""


def extract_username_from_href(href: str) -> str:
    """Ambil username dari URL Letterboxd (support relative & absolute URL)."""
    if not href:
        return ""
    # Absolute URL: https://letterboxd.com/username/...
    m = re.search(r'letterboxd\.com/([^/?#]+)', href)
    if m:
        candidate = m.group(1)
        if candidate not in SKIP_PATHS and len(candidate) >= 2:
            return candidate
    # Relative URL: /username/ atau /username/film/...
    m = re.match(r'^/([^/?#]+)', href)
    if m:
        candidate = m.group(1)
        if candidate not in SKIP_PATHS and len(candidate) >= 2:
            return candidate
    return ""


def is_blocked(sb) -> bool:
    """Cek apakah halaman diblokir Cloudflare."""
    try:
        title = sb.get_title().lower()
        src = sb.get_page_source()[:5000].lower()
    except Exception:
        return True

    block_signals = [
        "just a moment", "cf-browser-verification", "enable javascript and cookies",
        "checking your browser", "ray id", "security check", "please wait",
        "verify you are human",
    ]
    if any(sig in title for sig in ["just a moment", "attention required", "verify"]):
        return True
    cf_count = sum(1 for sig in block_signals if sig in src)
    return cf_count >= 2


def wait_for_cf(sb, timeout=30) -> bool:
    """Tunggu sampai Cloudflare challenge selesai."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_blocked(sb):
            return True
        time.sleep(2)
    return False


def safe_open_url(sb, url, reconnect_time=None):
    """
    Buka URL dengan bypass Cloudflare.
    Pakai uc_open_with_reconnect untuk bypass Turnstile challenge.
    Mengembalikan True jika berhasil, False jika gagal.

    Khusus ReadTimeoutError: coba beberapa kali dengan reconnect_time
    yang semakin besar, karena Colab sering lambat koneksinya.
    Strategi fallback bertingkat:
      1. uc_open_with_reconnect dengan rt semakin besar (4 attempts)
      2. driver.refresh() — kadang koneksi sudah setengah jalan
      3. uc_open (tanpa reconnect) — lebih simpel, kadang lebih stabil
      4. sb.open() — fallback terakhir tanpa UC mode
    """
    if reconnect_time is None:
        reconnect_time = UC_RECONNECT_TIME

    # Daftar reconnect_time yang akan dicoba jika ReadTimeoutError
    # Dinaikkan secara bertahap: 12 → 20 → 30 → 45 detik
    reconnect_attempts = [
        reconnect_time,
        UC_RECONNECT_TIME_LONG,
        UC_RECONNECT_TIME_LONG + 10,
        UC_RECONNECT_TIME_LONG + 25,
    ]

    last_err = None
    for attempt_idx, rt in enumerate(reconnect_attempts):
        try:
            sb.uc_open_with_reconnect(url, rt)
            last_err = None
            break  # berhasil, keluar dari loop
        except (ReadTimeoutError, TimeoutException) as e:
            last_err = e
            err_name = type(e).__name__
            print(f"[uc_open {err_name}, rt={rt}s, attempt {attempt_idx+1}/{len(reconnect_attempts)}]", end=" ")
            if attempt_idx < len(reconnect_attempts) - 1:
                # Delay lebih panjang + jitter sebelum retry
                wait = random.uniform(8, 15)
                print(f"[wait {wait:.0f}s]", end=" ", flush=True)
                time.sleep(wait)

                # Coba refresh driver — kadang koneksi sudah setengah jalan
                if attempt_idx >= 1:
                    try:
                        sb.driver.refresh()
                        time.sleep(3)
                    except Exception:
                        pass
            continue
        except WebDriverException as e:
            err_msg = str(e)[:100]
            print(f"[WebDriver err: {err_msg}]", end=" ")
            last_err = e
            # Coba recovery: refresh + delay
            try:
                sb.driver.refresh()
                time.sleep(3)
            except Exception:
                pass
            break
        except Exception as e:
            print(f"[uc_open err: {type(e).__name__}]", end=" ")
            last_err = e
            break

    # Jika semua uc_open_with_reconnect gagal, coba fallback bertingkat
    if last_err is not None:
        # ── Fallback 1: uc_open tanpa reconnect (lebih simpel) ──
        try:
            print("[trying uc_open...]", end=" ", flush=True)
            sb.uc_open(url)
            time.sleep(random.uniform(3, 5))
            # Cek apakah halaman benar-benar ter-load
            page_src = sb.get_page_source()
            if page_src and len(page_src) > 500:
                print("[uc_open OK]", end=" ")
                last_err = None
        except Exception as e_uc:
            print(f"[uc_open err: {type(e_uc).__name__}]", end=" ")

        # ── Fallback 2: sb.open() biasa ──
        if last_err is not None:
            try:
                # Set timeout lebih besar untuk fallback
                try:
                    sb.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT + 60)
                except Exception:
                    pass
                print("[trying sb.open...]", end=" ", flush=True)
                sb.open(url)
                print("[fallback open OK]", end=" ")
                last_err = None
            except Exception as e2:
                print(f"[fallback err: {type(e2).__name__}]", end=" ")

                # ── Fallback 3: refresh sebagai upaya terakhir ──
                try:
                    print("[trying refresh...]", end=" ", flush=True)
                    sb.driver.refresh()
                    time.sleep(5)
                    page_src = sb.get_page_source()
                    if page_src and len(page_src) > 500 and not is_blocked(sb):
                        print("[refresh OK]", end=" ")
                        last_err = None
                except Exception:
                    pass
            finally:
                # Kembalikan timeout ke normal
                try:
                    sb.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
                except Exception:
                    pass

        if last_err is not None:
            return False

    time.sleep(random.uniform(2, 4))

    # Cek apakah kena Cloudflare
    if is_blocked(sb):
        print("[CF terdeteksi, coba bypass...]", end=" ")

        # Coba klik captcha Turnstile jika ada
        try:
            sb.uc_gui_click_captcha()
            time.sleep(3)
        except Exception:
            pass

        # Tunggu CF selesai (timeout lebih lama untuk Colab)
        ok = wait_for_cf(sb, timeout=60)
        if not ok:
            # Retry sekali lagi: refresh + tunggu
            try:
                sb.driver.refresh()
                time.sleep(5)
                ok = wait_for_cf(sb, timeout=30)
            except Exception:
                pass

        if not ok:
            print("[GAGAL bypass CF!]", end=" ")
            return False
        print("[CF OK]", end=" ")
        time.sleep(2)

    return True


# ══════════════════════════════════════════════════════════
#  CSV I/O
# ══════════════════════════════════════════════════════════
def save_data_raw(data_list, filename):
    """Simpan data mentah (list of dicts) ke CSV — format 4 kolom."""
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["username", "tanggal", "rating", "ulasan"])
        for d in data_list:
            w.writerow([d["username"], d["tanggal"], d["rating"], d["ulasan"]])


def save_data_rows(rows, filename):
    """Simpan data (list of lists) ke CSV — format 4 kolom."""
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["username", "tanggal", "rating", "ulasan"])
        w.writerows(rows)


def load_existing_csv(filename):
    """
    Load data dari CSV yang sudah ada untuk resume.
    Menangani format lama (5 kolom: film,username,ulasan,rating,tanggal)
    dan format baru (4 kolom: username,tanggal,rating,ulasan).
    Mengembalikan (list_of_dicts, set_of_seen_keys).
    """
    data = []
    seen = set()

    if not os.path.exists(filename):
        return data, seen

    try:
        with open(filename, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header:
                return data, seen

            # Deteksi format berdasarkan header
            header_lower = [h.strip().lower() for h in header]

            if "film" in header_lower:
                # Format lama: film, username, ulasan, rating, tanggal
                print("  (format CSV lama terdeteksi, mengkonversi...)")
                for row in reader:
                    if len(row) >= 3:
                        username = row[1] if len(row) > 1 else ""
                        ulasan = row[2] if len(row) > 2 else ""
                        rating = row[3] if len(row) > 3 else ""
                        tanggal = row[4] if len(row) > 4 else ""
                        d = {
                            "username": username,
                            "tanggal": tanggal,
                            "rating": rating,
                            "ulasan": ulasan,
                        }
                        data.append(d)
                        seen.add(ulasan[:150])
            else:
                # Format baru: username, tanggal, rating, ulasan
                for row in reader:
                    if len(row) >= 4:
                        d = {
                            "username": row[0],
                            "tanggal": row[1],
                            "rating": row[2],
                            "ulasan": row[3],
                        }
                        data.append(d)
                        seen.add(row[3][:150])

    except Exception as e:
        print(f"  ⚠ Gagal load CSV ({type(e).__name__}: {e}), mulai dari awal")
        return [], set()

    return data, seen


# ══════════════════════════════════════════════════════════
#  SCRAPE SATU HALAMAN (dengan retry bawaan)
# ══════════════════════════════════════════════════════════
def scrape_page(sb, url: str, debug_first=False) -> tuple:
    """
    Scrape satu halaman review Letterboxd.
    Mengembalikan tuple (list_of_dict, out_of_range_count).
    Hanya mengembalikan ulasan yang lolos filter real-time:
      - Tanggal dalam range
      - Bahasa Indonesia
    """
    ok = safe_open_url(sb, url, reconnect_time=UC_RECONNECT_TIME)
    if not ok:
        return [], 0

    # Scroll halaman
    human_scroll(sb)
    time.sleep(random.uniform(1, 2))

    # Cek page source
    try:
        page_src = sb.get_page_source()
        src_len = len(page_src)
    except Exception:
        return [], 0

    # Ambil driver Selenium untuk find_elements
    driver = sb.driver

    # ── Cari container review ─────────────────────────────
    containers = []
    for sel in [
        "li.film-detail",
        "li[class*='film-detail']",
        "section.viewings-list li",
        "ul.js-list-entries li",
        "article.production-viewing",
        "div[class*='viewing-review']",
    ]:
        try:
            found = driver.find_elements(By.CSS_SELECTOR, sel)
            if found:
                containers = found
                print(f"[sel={sel!r} n={len(found)}]", end=" ")
                break
        except Exception:
            continue

    if not containers:
        try:
            body_text = driver.find_element(By.TAG_NAME, "body").text
            preview = body_text[:200].replace('\n', ' ')
            print(f"[no-container src={src_len} preview={preview!r}]", end=" ")
        except Exception:
            print(f"[no-container src={src_len}]", end=" ")
        return [], 0

    _debug_printed = not debug_first
    results = []
    out_of_range = 0

    for item in containers:
        try:
            # DEBUG: print HTML item pertama
            if not _debug_printed:
                try:
                    html_snippet = driver.execute_script(
                        "return arguments[0].outerHTML;", item
                    )[:1200]
                    print(f"\n[DEBUG HTML]:\n{html_snippet}\n", flush=True)
                    _debug_printed = True
                except Exception:
                    _debug_printed = True

            # ── USERNAME ──────────────────────────────────
            username = ""
            for usel in [
                "a.avatar",
                "a.context",
                "p.attribution a.name",
                "strong.name a",
                "span.name a",
                "a.name",
                ".attribution a",
            ]:
                try:
                    anchors = item.find_elements(By.CSS_SELECTOR, usel)
                    for anchor in anchors:
                        href = anchor.get_attribute("href") or ""
                        uname = extract_username_from_href(href)
                        if uname:
                            username = uname
                            break
                    if username:
                        break
                except Exception:
                    continue

            # Fallback: semua href via JS
            if not username:
                try:
                    all_hrefs = driver.execute_script(
                        "return Array.from(arguments[0].querySelectorAll('a[href]'))"
                        ".map(a => a.getAttribute('href'));",
                        item
                    )
                    for href in (all_hrefs or []):
                        uname = extract_username_from_href(href)
                        if uname:
                            username = uname
                            break
                except Exception:
                    pass

            # ── TANGGAL ───────────────────────────────────
            review_date_str = ""
            for dsel in [
                "span.date time",
                "time[datetime]",
                "span.date",
            ]:
                try:
                    date_els = item.find_elements(By.CSS_SELECTOR, dsel)
                    if date_els:
                        dt_attr = date_els[0].get_attribute("datetime") or ""
                        if dt_attr:
                            review_date_str = dt_attr.strip()
                            break
                        dt_text = date_els[0].text.strip()
                        if dt_text:
                            review_date_str = dt_text
                            break
                except Exception:
                    continue

            # Filter tanggal — hanya ambil ulasan dalam periode
            parsed_date = parse_review_date(review_date_str)
            if not is_in_date_range(parsed_date):
                out_of_range += 1
                continue

            # Format tanggal untuk output
            display_date = parsed_date.isoformat() if parsed_date else review_date_str

            # ── RATING ────────────────────────────────────
            rating = ""
            for rsel in ["span.rating", "span[class*='rated-']", "span[class*='rating']"]:
                try:
                    rels = item.find_elements(By.CSS_SELECTOR, rsel)
                    if rels:
                        rating = extract_rating(rels[0])
                        if rating:
                            break
                except Exception:
                    continue

            # ── TEKS REVIEW ───────────────────────────────
            review_text = ""

            # Cek apakah ada "read more" → klik expand
            try:
                expand_links = item.find_elements(By.CSS_SELECTOR, "a.reveal, a[data-reveal]")
                for link in expand_links:
                    try:
                        driver.execute_script("arguments[0].click();", link)
                        time.sleep(0.5)
                    except Exception:
                        pass
            except Exception:
                pass

            for tsel in [
                "div.body-text p",
                "div.body-text.-prose p",
                "div[class*='body-text'] p",
                "div.js-review-body p",
                "div.js-review p",
                "div.body-text",
                "div.js-review-body",
                "div.js-review",
                "div[class*='review-content']",
            ]:
                try:
                    tels = item.find_elements(By.CSS_SELECTOR, tsel)
                    if tels:
                        parts = [el.text.strip() for el in tels if el.text.strip()]
                        if parts:
                            review_text = clean_review(" ".join(parts))
                            break
                except Exception:
                    continue

            # Fallback: teks mentah keseluruhan item
            if not review_text:
                raw = item.text.strip()
                if raw:
                    lines = [l.strip() for l in raw.splitlines() if l.strip()]
                    body = lines[1:] if len(lines) > 1 else lines
                    review_text = clean_review(" ".join(body))

            # ══ FILTER REAL-TIME ══════════════════════════

            # Bahasa Indonesia
            if not is_indonesian(review_text):
                continue

            results.append({
                "username": username,
                "tanggal":  display_date,
                "rating":   rating,
                "ulasan":   review_text,
            })

        except Exception:
            continue

    return results, out_of_range


# ══════════════════════════════════════════════════════════
#  FUNGSI RETRY DENGAN BACKOFF
# ══════════════════════════════════════════════════════════
def retry_scrape_page(sb, url, max_retries=MAX_RETRIES_PER_PAGE, debug_first=False):
    """
    Scrape halaman dengan retry dan exponential backoff.
    Mengembalikan (entries, out_of_range_count, is_connection_dead).
    is_connection_dead = True jika semua retry gagal total (browser mungkin crash).
    """
    for attempt in range(1, max_retries + 1):
        try:
            entries, out_of_range_count = scrape_page(sb, url, debug_first=debug_first)
            # Jika dapat data (bahkan kosong tapi berhasil), return
            return entries, out_of_range_count, False
        except Exception as e:
            err_name = type(e).__name__
            print(f"[err#{attempt}: {err_name}]", end=" ")

            if attempt < max_retries:
                # Exponential backoff: 5, 10, 20, 40... max 60s
                delay = min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
                print(f"[backoff {delay:.0f}s]", end=" ", flush=True)
                time.sleep(delay)
            else:
                # Semua retry habis
                print(f"[GAGAL setelah {max_retries} retry]", end=" ")
                return [], 0, True

    return [], 0, True


# ══════════════════════════════════════════════════════════
#  BANGUN DAFTAR STRATEGI SCRAPING
# ══════════════════════════════════════════════════════════
def build_strategies():
    """Bangun daftar semua strategi scraping."""
    # Daftar semua rating filter di Letterboxd
    ALL_RATINGS = [
        "rated/5", "rated/4.5", "rated/4", "rated/3.5", "rated/3",
        "rated/2.5", "rated/2", "rated/1.5", "rated/1", "rated/0.5",
        "rated/none",
    ]

    # Daftar sort yang menghasilkan urutan berbeda
    ALL_SORTS = [
        ("by/added",              "Terbaru"),
        ("by/added-earliest",     "Terlama"),
        ("by/popular",            "Populer"),
        ("by/activity",           "Aktivitas"),
    ]

    # Rating label untuk display
    RATING_LABELS = {
        "rated/5":    "★★★★★",
        "rated/4.5":  "★★★★½",
        "rated/4":    "★★★★",
        "rated/3.5":  "★★★½",
        "rated/3":    "★★★",
        "rated/2.5":  "★★½",
        "rated/2":    "★★",
        "rated/1.5":  "★½",
        "rated/1":    "★",
        "rated/0.5":  "½",
        "rated/none": "Tanpa Rating",
    }

    SCRAPE_STRATEGIES = []

    # ── Fase 1: Tanpa filter rating × semua sort ─────
    for sort_path, sort_label in ALL_SORTS:
        SCRAPE_STRATEGIES.append(
            ("", sort_path, f"{sort_label} (semua rating)")
        )

    # ── Fase 2: Per rating × by/added ────────────────
    PRIORITY_RATINGS = [
        "rated/4", "rated/3.5", "rated/3", "rated/4.5", "rated/5",
        "rated/2.5", "rated/2", "rated/none",
        "rated/1.5", "rated/1", "rated/0.5",
    ]
    for rf in PRIORITY_RATINGS:
        SCRAPE_STRATEGIES.append(
            (rf, "by/added", f"{RATING_LABELS[rf]} Terbaru")
        )

    # ── Fase 3: Per rating × by/added-earliest ───────
    for rf in PRIORITY_RATINGS:
        SCRAPE_STRATEGIES.append(
            (rf, "by/added-earliest", f"{RATING_LABELS[rf]} Terlama")
        )

    # ── Fase 4: Per rating × by/popular ──────────────
    for rf in PRIORITY_RATINGS:
        SCRAPE_STRATEGIES.append(
            (rf, "by/popular", f"{RATING_LABELS[rf]} Populer")
        )

    # ── Fase 5: Per rating × by/activity ─────────────
    for rf in PRIORITY_RATINGS:
        SCRAPE_STRATEGIES.append(
            (rf, "by/activity", f"{RATING_LABELS[rf]} Aktivitas")
        )

    # ── Fase 6: Sorting global tambahan ──────────────
    SCRAPE_STRATEGIES.extend([
        ("",         "by/entry-rating",        "Rating Tertinggi"),
        ("",         "by/entry-rating-lowest", "Rating Terendah"),
    ])

    # ── Fase 7: Per rating × entry-rating sorts ──────
    for rf in ["rated/5", "rated/4.5", "rated/4", "rated/3.5",
               "rated/3", "rated/2.5", "rated/2"]:
        SCRAPE_STRATEGIES.append(
            (rf, "by/entry-rating", f"{RATING_LABELS[rf]} Rating↑")
        )
        SCRAPE_STRATEGIES.append(
            (rf, "by/entry-rating-lowest", f"{RATING_LABELS[rf]} Rating↓")
        )

    return SCRAPE_STRATEGIES


# ══════════════════════════════════════════════════════════
#  MAIN — JALANKAN SCRAPER
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  Letterboxd Scraper — Ghost in the Cell (2026)")
print("  Google Colab Edition (SeleniumBase UC Mode)")
print("  ✨ DENGAN FITUR RESUME — bisa pause & lanjutkan!")
print("=" * 60)
print(f"  Film     : {FILM_TITLE}")
print(f"  Slug     : /film/{FILM_SLUG}/reviews/")
print(f"  Periode  : {DATE_START.isoformat()} s.d. {DATE_END.isoformat()}")
print(f"  Target   : {FINAL_TARGET}+ ulasan (scrape buffer: {TARGET_TOTAL})")
print(f"  Output   : {OUTPUT_FILE}")
print(f"  Progress : {PROGRESS_FILE}")
print(f"  Kolom    : username | tanggal | rating | ulasan")
print("=" * 60)
print()
print("  Filter:")
print("    - Bahasa Indonesia")
print("    - Duplikat dihapus (exact match + cosine similarity > 0.85)")
print("=" * 60 + "\n")

# Deteksi environment
IS_COLAB = False
try:
    import google.colab
    IS_COLAB = True
    print("✓ Berjalan di Google Colab")
except ImportError:
    print("ℹ Berjalan di luar Colab (local)")

# ── Auto-install dependencies di Colab ────────────────────
if IS_COLAB:
    import subprocess
    import shutil

    def _run_cmd(cmd, label=""):
        """Jalankan shell command, print status."""
        print(f"  {label}... ", end="", flush=True)
        result = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"GAGAL (exit {result.returncode})")
            if result.stderr:
                print(f"    stderr: {result.stderr[:200]}")
        return result.returncode == 0

    # Cek apakah Chrome sudah terinstall
    chrome_path = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if not chrome_path:
        print("\n⚠ Google Chrome belum terinstall, menginstall otomatis...")
        _run_cmd("apt-get update -qq", "Update apt")
        _run_cmd(
            "curl -sS -o /tmp/chrome.deb "
            "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb",
            "Download Chrome"
        )
        _run_cmd("apt-get install -y -qq /tmp/chrome.deb", "Install Chrome")
        _run_cmd("rm -f /tmp/chrome.deb", "Cleanup")

        # Verifikasi
        chrome_path = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
        if chrome_path:
            print(f"✓ Chrome terinstall di: {chrome_path}\n")
        else:
            print("✗ Chrome gagal diinstall! Coba jalankan manual:")
            print("  !curl -o chrome.deb https://dl.google.com/linux/direct/"
                  "google-chrome-stable_current_amd64.deb")
            print("  !apt install ./chrome.deb -y\n")
    else:
        print(f"✓ Chrome ditemukan di: {chrome_path}")

    # Cek xvfb
    if not shutil.which("Xvfb"):
        print("  Installing xvfb...", end=" ", flush=True)
        _run_cmd("apt-get install -y -qq xvfb", "Install xvfb")
    else:
        print("✓ xvfb tersedia")

    print()

# ── Resume dari CSV yang sudah ada ────────────────────────
data, seen = load_existing_csv(OUTPUT_FILE)
unsaved_count = 0

if data:
    print(f"✓ Resume dari '{OUTPUT_FILE}': {len(data)} data sudah ada")
    print(f"  Dedup keys di-load: {len(seen)} (duplikat akan di-skip otomatis)")

# ── Load progress (strategi & halaman terakhir) ───────────
resume_strategy_idx, resume_page = load_progress()
if resume_strategy_idx > 0 or resume_page > 1:
    print(f"  🔄 Akan melanjutkan dari strategi #{resume_strategy_idx}, "
          f"halaman {resume_page}")
else:
    if data:
        print(f"\n  ⚠ PERHATIAN: File progress ({PROGRESS_FILE}) tidak ditemukan/reset,")
        print(f"    tapi CSV sudah ada {len(data)} data.")
        print(f"    Script akan scan dari awal tapi SKIP semua duplikat.")
        print(f"    Tip: Jika tahu posisi terakhir, edit {PROGRESS_FILE} manual:")
        print(f'    {{"strategy_index": <nomor>, "page": <halaman>}}')
print()

# ── Bangun strategi ───────────────────────────────────────
SCRAPE_STRATEGIES = build_strategies()
print(f"  Total strategi scraping: {len(SCRAPE_STRATEGIES)} kombinasi")
print(f"  Potensi halaman max: {len(SCRAPE_STRATEGIES) * MAX_PAGE}\n")

# ── Cek apakah target sudah tercapai ─────────────────────
if len(data) >= TARGET_TOTAL:
    print(f"✓ Data sudah cukup ({len(data)} >= {TARGET_TOTAL}), "
          f"langsung ke post-processing...\n")
else:
    # ══════════════════════════════════════════════════════
    #  LOOP SCRAPING DENGAN AUTO-RESTART BROWSER
    # ══════════════════════════════════════════════════════
    browser_restart_count = 0
    current_strategy_idx = resume_strategy_idx
    current_start_page = resume_page

    # ── Variabel live tracking (untuk pause/resume yang akurat) ──
    # Variabel ini diupdate setiap halaman agar KeyboardInterrupt
    # dan Exception handler selalu menyimpan posisi terkini
    _live_si   = current_strategy_idx
    _live_page = current_start_page

    while (len(data) < TARGET_TOTAL
           and current_strategy_idx < len(SCRAPE_STRATEGIES)
           and browser_restart_count <= BROWSER_RESTART_LIMIT):

        # ── SeleniumBase Context Manager ──────────────────
        sb_kwargs = {
            "uc": True,
            "headless": True,
            "xvfb": IS_COLAB,
            "page_load_strategy": "eager",  # Tidak menunggu semua resource selesai load
            "agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }

        need_restart = False

        try:
            with SB(**sb_kwargs) as sb:
                # Set page load timeout agar tidak hang terlalu lama
                try:
                    sb.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
                except Exception:
                    pass

                if browser_restart_count == 0:
                    print("✓ SeleniumBase driver siap")
                    print(f"  (page_load_timeout={PAGE_LOAD_TIMEOUT}s, "
                          f"reconnect_time={UC_RECONNECT_TIME}s, "
                          f"strategy=eager)\n")
                else:
                    print(f"✓ Browser restart #{browser_restart_count} berhasil\n")

                # ── Warmup: buka homepage dulu ────────────────
                print("Warming up (membuka Letterboxd homepage)...")
                try:
                    ok = safe_open_url(sb, "https://letterboxd.com/", reconnect_time=UC_RECONNECT_TIME)
                    if ok:
                        human_scroll(sb)
                        rand_sleep(3, 5)
                        print("  OK\n")
                    else:
                        print("  ⚠ Homepage gagal, tapi lanjut coba review page...\n")
                except Exception as e:
                    print(f"  Warmup error ({type(e).__name__}), lanjut...\n")
                    time.sleep(3)

                # ── Loop strategi ─────────────────────────────
                consecutive_total_fail = 0  # gagal total berturut (connection dead)

                for si in range(current_strategy_idx, len(SCRAPE_STRATEGIES)):
                    if len(data) >= TARGET_TOTAL:
                        break
                    if need_restart:
                        break

                    rating_filter, sort_path, strategy_label = SCRAPE_STRATEGIES[si]

                    # Bangun base URL
                    if rating_filter:
                        base_url = (f"https://letterboxd.com/film/{FILM_SLUG}/"
                                    f"reviews/{rating_filter}/{sort_path}")
                    else:
                        base_url = (f"https://letterboxd.com/film/{FILM_SLUG}/"
                                    f"reviews/{sort_path}")

                    print(f"\n{'='*60}")
                    print(f"  Strategi #{si}: {strategy_label}")
                    print(f"  URL base: {base_url}")
                    print(f"  Progress: {len(data)}/{TARGET_TOTAL} "
                          f"(target akhir: {FINAL_TARGET}+)")
                    print(f"{'='*60}")

                    consecutive_empty = 0
                    consecutive_dup = 0
                    consecutive_out_of_range = 0

                    # Tentukan halaman awal (resume atau 1)
                    start_page = current_start_page if si == current_strategy_idx else 1

                    for page in range(start_page, MAX_PAGE + 1):
                        if len(data) >= TARGET_TOTAL:
                            print(f"\n✓ TARGET SCRAPING {TARGET_TOTAL} TERCAPAI!")
                            break

                        url = f"{base_url}/page/{page}/"
                        print(f"  [{strategy_label}] Hal {page:>3} ... ",
                              end="", flush=True)

                        # Update live tracking (untuk pause/resume)
                        _live_si = si
                        _live_page = page

                        # Simpan progress setiap halaman
                        save_progress(si, page, len(data))

                        # Scrape dengan retry dan backoff
                        entries, out_of_range_count, connection_dead = \
                            retry_scrape_page(sb, url,
                                              debug_first=(page == 1 and not data))

                        if connection_dead:
                            consecutive_total_fail += 1
                            print(f"[KONEKSI MATI #{consecutive_total_fail}]")

                            if consecutive_total_fail >= MAX_CONSECUTIVE_FAIL:
                                print(f"\n  ⚠ {MAX_CONSECUTIVE_FAIL}x gagal berturut! "
                                      f"Restart browser...")
                                # Simpan data & progress sebelum restart
                                if data:
                                    save_data_raw(data, OUTPUT_FILE)
                                    print(f"  [saved {len(data)} data sebelum restart]")
                                save_progress(si, page, len(data))
                                current_strategy_idx = si
                                current_start_page = page
                                need_restart = True
                                break

                            # Delay panjang sebelum coba lagi
                            delay = min(30 + (consecutive_total_fail * 10), 90)
                            print(f"  [delay panjang {delay}s...]")
                            time.sleep(delay)
                            continue

                        # Reset counter gagal jika berhasil koneksi
                        consecutive_total_fail = 0

                        if not entries:
                            consecutive_empty += 1
                            if out_of_range_count > 0:
                                consecutive_out_of_range += 1
                                print(f"kosong (di luar periode: {out_of_range_count}, "
                                      f"streak: {consecutive_out_of_range}x)")
                                if consecutive_out_of_range >= 15:
                                    print(f"  => 15 halaman berturut semua di luar "
                                          f"periode, pindah strategi")
                                    break
                            else:
                                print(f"kosong ({consecutive_empty}x)")

                            if consecutive_empty >= 10:
                                print(f"  => Habis/limit tercapai, pindah strategi "
                                      f"berikutnya")
                                break
                            rand_sleep(5, 8)
                            continue

                        consecutive_empty = 0
                        consecutive_out_of_range = 0
                        added = 0
                        skipped_dup = 0

                        for e in entries:
                            if len(data) >= TARGET_TOTAL:
                                break
                            key = e["ulasan"][:150]
                            if key in seen:
                                skipped_dup += 1
                                continue
                            seen.add(key)
                            data.append(e)
                            added += 1
                            unsaved_count += 1

                        print(f"dapat={len(entries)} | +baru={added} | "
                              f"dup={skipped_dup} | diluar_tgl={out_of_range_count} "
                              f"| total={len(data)}/{TARGET_TOTAL}")

                        # Cek apakah strategi ini masih menghasilkan data baru
                        if added == 0:
                            consecutive_dup += 1
                        else:
                            consecutive_dup = 0

                        # Auto-save
                        if unsaved_count >= AUTOSAVE_EVERY:
                            save_data_raw(data, OUTPUT_FILE)
                            unsaved_count = 0
                            print(f"  [auto-saved {len(data)} data]")

                        # Delay antar halaman
                        rand_sleep(3.0, 6.0)

                        # Delay lebih lama tiap 20 halaman
                        if page % 20 == 0:
                            delay = random.uniform(10, 20)
                            print(f"  [jeda panjang {delay:.1f}s untuk "
                                  f"anti-rate-limit...]")
                            time.sleep(delay)

                    # Jika strategi ini selesai normal (bukan restart),
                    # reset start_page untuk strategi berikutnya
                    if not need_restart:
                        current_start_page = 1

                # Update current_strategy_idx jika loop selesai normal
                if not need_restart:
                    current_strategy_idx = len(SCRAPE_STRATEGIES)  # selesai semua

        except KeyboardInterrupt:
            print("\n\n⏸ PAUSE — menyimpan data & progress...")
            if data:
                save_data_raw(data, OUTPUT_FILE)
                print(f"  ✓ {len(data)} data tersimpan ke '{OUTPUT_FILE}'")
            # PENTING: gunakan _live_si dan _live_page, BUKAN
            # current_strategy_idx/current_start_page yang bisa basi!
            save_progress(_live_si, _live_page + 1, len(data))
            print(f"  ✓ Progress tersimpan (strategi #{_live_si}, "
                  f"hal {_live_page} → resume dari hal {_live_page + 1})")
            print(f"\n  💡 Jalankan ulang script untuk MELANJUTKAN dari posisi ini!")
            print(f"     Tidak perlu scraping ulang.\n")
            sys.exit(0)

        except Exception as e:
            print(f"\n\n❌ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\nMenyimpan data yang sudah terkumpul...")
            if data:
                save_data_raw(data, OUTPUT_FILE)
                print(f"  ✓ {len(data)} data tersimpan")
            # Gunakan live tracking untuk progress yang akurat
            save_progress(_live_si, _live_page + 1, len(data))

        # Jika perlu restart browser
        if need_restart:
            browser_restart_count += 1
            if browser_restart_count > BROWSER_RESTART_LIMIT:
                print(f"\n❌ Browser sudah di-restart {BROWSER_RESTART_LIMIT}x, "
                      f"berhenti.")
                print(f"   Data tersimpan: {len(data)} ulasan")
                break
            print(f"\n🔄 Restart browser #{browser_restart_count}...")
            # Delay sebelum restart
            restart_delay = random.uniform(15, 30)
            print(f"   Delay {restart_delay:.0f}s sebelum restart...")
            time.sleep(restart_delay)
            need_restart = False
            continue
        else:
            # Selesai normal (target tercapai atau semua strategi habis)
            break

# ══════════════════════════════════════════════════════════
#  SIMPAN DATA MENTAH (sebelum post-processing)
# ══════════════════════════════════════════════════════════
raw_count = len(data)
if data:
    save_data_raw(data, OUTPUT_FILE)
    print(f"\n{'='*60}")
    print(f"  SCRAPING SELESAI — {raw_count} ulasan mentah tersimpan")
    print(f"{'='*60}")

# ══════════════════════════════════════════════════════════
#  POST-PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════
if data:
    data = post_process_pipeline(data)
    save_data_raw(data, OUTPUT_FILE)

# ══════════════════════════════════════════════════════════
#  HAPUS PROGRESS FILE (scraping selesai)
# ══════════════════════════════════════════════════════════
clear_progress()

# ══════════════════════════════════════════════════════════
#  LAPORAN FINAL
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  HASIL AKHIR — {len(data)} ulasan tersimpan ke '{OUTPUT_FILE}'")
if len(data) >= FINAL_TARGET:
    print(f"  ✓ Target {FINAL_TARGET}+ data TERCAPAI! ({len(data)} ulasan)")
else:
    print(f"  ⚠ Target {FINAL_TARGET} data belum tercapai (hanya {len(data)} data)")
print(f"  Sebelum post-processing: {raw_count} ulasan")
print(f"  Setelah post-processing: {len(data)} ulasan")
print(f"  Periode: {DATE_START.isoformat()} s.d. {DATE_END.isoformat()}")
print("=" * 60)

if data:
    print(f"\n  Kolom    : username | tanggal | rating | ulasan")
    filled_user = sum(1 for d in data if d["username"])
    print(f"  Username terisi : {filled_user}/{len(data)} "
          f"({filled_user/len(data)*100:.1f}%)")
    filled_date = sum(1 for d in data if d["tanggal"])
    print(f"  Tanggal terisi  : {filled_date}/{len(data)} "
          f"({filled_date/len(data)*100:.1f}%)")
    filled_rating = sum(1 for d in data if d["rating"])
    print(f"  Rating terisi   : {filled_rating}/{len(data)} "
          f"({filled_rating/len(data)*100:.1f}%)")
    word_counts = [count_words(d["ulasan"]) for d in data]
    print(f"  Rata-rata kata  : {sum(word_counts)/len(word_counts):.1f}")
    print(f"  Min kata        : {min(word_counts)}")
    print(f"  Max kata        : {max(word_counts)}")

    # Statistik rating
    ratings = [float(d["rating"]) for d in data if d["rating"]]
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        print(f"  Rata-rata rating: {avg_rating:.2f}")

    # Distribusi tanggal
    date_counts = {}
    for d in data:
        pd = parse_review_date(d["tanggal"])
        if pd:
            month_key = pd.strftime("%Y-%m")
            date_counts[month_key] = date_counts.get(month_key, 0) + 1
    if date_counts:
        print(f"\n  Distribusi per bulan:")
        for month in sorted(date_counts.keys()):
            print(f"    {month}: {date_counts[month]} ulasan")

else:
    print("\n[!] Tidak ada data. Kemungkinan:")
    print("    1. Film belum punya cukup review di Letterboxd")
    print("    2. Slug salah — cek manual: letterboxd.com/film/ghost-in-the-cell/reviews/")
    print("    3. Cloudflare masih memblokir — coba:")
    print("       a. Restart runtime Colab (dapat IP baru)")
    print("       b. Ganti runtime type (GPU/TPU lalu balik)")
    print("       c. Jalankan ulang script")
    print("    4. Tidak cukup review Indonesia dalam periode")
    print(f"       {DATE_START.isoformat()} s.d. {DATE_END.isoformat()}")

# Download file di Colab
try:
    from google.colab import files
    if data:
        files.download(OUTPUT_FILE)
        print(f"\n  📥 File '{OUTPUT_FILE}' siap di-download!")
except ImportError:
    print(f"\n  File tersimpan di: {os.path.abspath(OUTPUT_FILE)}")
except Exception as e:
    print(f"\n  ⚠ Auto-download gagal. Download manual dari panel Files (📁).")
