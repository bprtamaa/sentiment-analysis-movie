"""
===============================================================================
KLASIFIKASI SENTIMEN MENGGUNAKAN SUPPORT VECTOR MACHINE (SVM)
Untuk Analisis Sentimen Berbasis Aspek — Skripsi

Platform   : Google Colab
Input      : dataset_labeled_sentiment.csv (hasil pelabelan sentimen)
             tfidf_vectorizer.pkl (model TF-IDF Vectorizer)
             tfidf_matrix.pkl (matriks fitur TF-IDF)
Output     : 1. model_svm_terbaik.pkl (model SVM terlatih terbaik)
             2. hasil_klasifikasi_svm.csv (dataset + prediksi SVM)
             3. confusion_matrix_svm.png
             4. classification_report_svm.png
             5. perbandingan_kernel_svm.png
             6. learning_curve_svm.png
             7. distribusi_prediksi_svm.png
             8. evaluasi_metrik_svm.png
             9. radar_metrik_per_kelas_svm.png
             10. cross_validation_per_fold_svm.png

ALUR KERJA KLASIFIKASI SVM (10 TAHAP):
  Tahap 1  : Load Dataset dan Artefak TF-IDF
  Tahap 2  : Eksplorasi Distribusi Kelas Sentimen
  Tahap 3  : Pembagian Data 80:20 (Train/Test Split) dengan Stratified Sampling
  Tahap 4  : Pelatihan Baseline SVM (Kernel Linear, RBF, Poly, Sigmoid)
  Tahap 5  : Hyperparameter Tuning dengan GridSearchCV + Stratified K-Fold
  Tahap 6  : Evaluasi Model Terbaik (Accuracy, Precision, Recall, F1-Score)
  Tahap 7  : Confusion Matrix & Classification Report Detail
  Tahap 8  : Visualisasi Metrik Evaluasi (Bar Chart, Radar, Per-Fold CV)
  Tahap 9  : Learning Curve (Analisis Bias-Varians)
  Tahap 10 : Simpan Model, Ekspor Hasil, & Ringkasan Akhir

CATATAN:
  - Input utama adalah matriks TF-IDF (tfidf_matrix.pkl) sebagai fitur X
    dan kolom 'sentimen' dari dataset_labeled_sentiment.csv sebagai label y.
  - Menggunakan Stratified K-Fold Cross-Validation agar setiap fold
    memiliki proporsi kelas yang sama (penting untuk data tidak seimbang).
  - GridSearchCV digunakan untuk mencari kombinasi hyperparameter terbaik.
  - Semua grafik disimpan dalam resolusi 300 DPI untuk standar publikasi.
  - Referensi interpretasi metrik:
      * Landis & Koch (1977) untuk interpretasi Cohen's Kappa
      * Sokolova & Lapalme (2009) untuk interpretasi metrik klasifikasi
===============================================================================
"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: INSTALL DEPENDENCIES                                             ║
# ║  Jalankan cell ini PERTAMA KALI di Google Colab                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Uncomment baris di bawah jika menjalankan di Colab:
# !pip install pandas numpy matplotlib seaborn scikit-learn


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: IMPORT LIBRARY & UTILITIES                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import pickle
import time
import warnings
from datetime import datetime

# Visualisasi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Scikit-learn: SVM & Evaluasi
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# Pengaturan visual & reproduktivitas
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Konfigurasi font untuk visualisasi riset
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# ── Fungsi Logging Terstruktur (Konsisten dengan modul-modul sebelumnya) ──
def log_header(tahap_num, judul, icon="🚀"):
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
    print(f"\n    [SELESAI] Tahap {tahap_num}: {judul} — {timestamp}")
    if detail_dict:
        for key, val in detail_dict.items():
            print(f"    {key:.<50s} {val}")
    print("-" * 80)


print("=" * 80)
print(" KLASIFIKASI SENTIMEN MENGGUNAKAN SUPPORT VECTOR MACHINE (SVM)")
print(" Film: Ghost in the Cell (2026) — Letterboxd")
print(" Untuk Analisis Sentimen Berbasis Aspek — Skripsi / Penelitian")
print("=" * 80)
print(f"[OK] Semua library berhasil diimpor!")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: KONFIGURASI PARAMETER                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Path File Input ──
INPUT_DATASET = "dataset_labeled_sentiment.csv"   # Dataset dengan label sentimen
INPUT_TFIDF_MATRIX = "tfidf_matrix.pkl"           # Matriks fitur TF-IDF (sparse matrix)
INPUT_TFIDF_VECTORIZER = "tfidf_vectorizer.pkl"   # Model TF-IDF Vectorizer

# ── Kolom Target ──
TEXT_COLUMN = "ulasan_preprocessed"                # Kolom teks (untuk referensi)
LABEL_COLUMN = "sentimen"                          # Kolom label sentimen target
# Catatan: Jika kolom 'sentimen' tidak ada, script akan mencoba 'sentimen_final'

# ── Parameter Pembagian Data (80% Training : 20% Testing) ──
TEST_SIZE = 0.20                                   # 20% data untuk pengujian (80:20 split)
RANDOM_STATE = 42                                  # Seed untuk reproduktivitas

# ── Parameter Cross-Validation ──
CV_FOLDS = 5                                       # Jumlah fold Stratified K-Fold
CV_SCORING = 'accuracy'                            # Metrik evaluasi utama (Accuracy)

# ── Parameter Grid Search SVM ──
# Hyperparameter yang akan di-tuning
PARAM_GRID = {
    'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],       # Regularization (tanpa C=100 untuk cegah overfitting)
    'kernel': ['linear', 'rbf'],                   # Kernel functions
    'gamma': ['scale', 'auto', 0.1, 0.01],         # Kernel coefficient (untuk RBF) — lebih granular
    'class_weight': ['balanced']                    # Selalu balanced (data sudah di-resampling)
}

# ── Path File Output ──
OUTPUT_MODEL = "model_svm_terbaik.pkl"
OUTPUT_CSV = "hasil_klasifikasi_svm.csv"
OUTPUT_CONFUSION_MATRIX = "confusion_matrix_svm.png"
OUTPUT_CLASSIFICATION_REPORT = "classification_report_svm.png"
OUTPUT_KERNEL_COMPARISON = "perbandingan_kernel_svm.png"
OUTPUT_LEARNING_CURVE = "learning_curve_svm.png"
OUTPUT_DISTRIBUSI_PREDIKSI = "distribusi_prediksi_svm.png"
OUTPUT_EVALUASI_METRIK = "evaluasi_metrik_svm.png"
OUTPUT_RADAR_METRIK = "radar_metrik_per_kelas_svm.png"
OUTPUT_CV_PER_FOLD = "cross_validation_per_fold_svm.png"

print("[OK] Konfigurasi parameter berhasil dimuat.")
print(f"     Input Dataset      : {INPUT_DATASET}")
print(f"     Input TF-IDF       : {INPUT_TFIDF_MATRIX}")
print(f"     Split Ratio        : {(1-TEST_SIZE)*100:.0f}% Training : {TEST_SIZE*100:.0f}% Testing (80:20)")
print(f"     Cross-Validation   : {CV_FOLDS}-Fold Stratified")
print(f"     Metrik Evaluasi    : {CV_SCORING}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: LOAD DATASET & ARTEFAK TF-IDF (TAHAP 1)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(1, "MEMUAT DATASET & ARTEFAK TF-IDF", "📂")

# ── 1.1 Load Dataset CSV ──
try:
    df = pd.read_csv(INPUT_DATASET)
    print(f"  ✓ Dataset '{INPUT_DATASET}' berhasil dimuat.")
    print(f"    Total data       : {len(df)} ulasan")
    print(f"    Daftar kolom     : {list(df.columns)}")
except FileNotFoundError:
    print(f"  [ERROR] File '{INPUT_DATASET}' tidak ditemukan!")
    print(f"          Pastikan Anda telah menjalankan tahap pelabelan sentimen terlebih dahulu.")
    raise

# ── 1.2 Deteksi Kolom Label Sentimen ──
if LABEL_COLUMN not in df.columns:
    if 'sentimen_final' in df.columns:
        LABEL_COLUMN = 'sentimen_final'
        print(f"  [INFO] Kolom 'sentimen' tidak ditemukan. Menggunakan kolom '{LABEL_COLUMN}' sebagai label.")
    else:
        raise KeyError(f"Kolom label '{LABEL_COLUMN}' tidak ditemukan! Kolom yang ada: {list(df.columns)}")

# ── 1.3 Load Matriks TF-IDF ──
try:
    with open(INPUT_TFIDF_MATRIX, 'rb') as f:
        X_tfidf = pickle.load(f)
    print(f"\n  ✓ Matriks TF-IDF '{INPUT_TFIDF_MATRIX}' berhasil dimuat.")
    print(f"    Dimensi matriks  : {X_tfidf.shape[0]} dokumen × {X_tfidf.shape[1]} fitur")
except FileNotFoundError:
    print(f"  [ERROR] File '{INPUT_TFIDF_MATRIX}' tidak ditemukan!")
    print(f"          Pastikan Anda telah menjalankan tahap ekstraksi fitur TF-IDF terlebih dahulu.")
    raise

# ── 1.4 Load TF-IDF Vectorizer ──
try:
    with open(INPUT_TFIDF_VECTORIZER, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"  ✓ Vectorizer '{INPUT_TFIDF_VECTORIZER}' berhasil dimuat.")
    print(f"    Jumlah kosa kata : {len(vectorizer.get_feature_names_out())} kata/frasa")
except FileNotFoundError:
    print(f"  [WARNING] File '{INPUT_TFIDF_VECTORIZER}' tidak ditemukan. Vectorizer tidak dimuat.")
    vectorizer = None

# ── 1.5 Pembersihan & Validasi Data ──
# Hapus baris dengan label sentimen NaN
jumlah_awal = len(df)
df = df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
jumlah_dihapus = jumlah_awal - len(df)

# Pastikan jumlah baris dataset == jumlah baris matriks TF-IDF
if len(df) != X_tfidf.shape[0]:
    print(f"\n  [WARNING] Jumlah baris dataset ({len(df)}) ≠ baris TF-IDF ({X_tfidf.shape[0]}).")
    print(f"            Menyesuaikan ke jumlah terkecil...")
    min_rows = min(len(df), X_tfidf.shape[0])
    df = df.iloc[:min_rows].reset_index(drop=True)
    X_tfidf = X_tfidf[:min_rows]

# Ambil label sentimen
y = df[LABEL_COLUMN].values

log_done(1, "Load Dataset & Artefak TF-IDF", {
    "Total data valid": f"{len(df)} ulasan",
    "Data NaN dihapus": str(jumlah_dihapus),
    "Dimensi fitur TF-IDF (X)": f"{X_tfidf.shape}",
    "Jumlah label sentimen (y)": f"{len(y)}",
    "Kelas sentimen unik": f"{list(np.unique(y))}"
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: EKSPLORASI DISTRIBUSI KELAS SENTIMEN (TAHAP 2)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(2, "EKSPLORASI DISTRIBUSI KELAS SENTIMEN", "📊")

# ── 2.1 Hitung distribusi kelas ──
label_counts = pd.Series(y).value_counts()
total_data = len(y)

# Urutan kelas standar
labels_order = ['positif', 'netral', 'negatif']
labels_order = [l for l in labels_order if l in label_counts.index]
# Tambahkan label yang mungkin tidak standar
for l in label_counts.index:
    if l not in labels_order:
        labels_order.append(l)

print(f"\n  TABEL 4.SVM1: DISTRIBUSI KELAS SENTIMEN UNTUK KLASIFIKASI SVM")
print(f"  {'━' * 65}")
print(f"  {'No':<4} {'Kelas Sentimen':<20} {'Jumlah Ulasan':<18} {'Persentase (%)'}")
print(f"  {'─' * 60}")
for i, label in enumerate(labels_order, 1):
    count = label_counts.get(label, 0)
    pct = (count / total_data) * 100
    print(f"  {i:<4} {label.capitalize():<20} {count:<18} {pct:.2f}%")
print(f"  {'─' * 60}")
print(f"  {'':4} {'Total':<20} {total_data:<18} 100.00%")
print(f"  {'━' * 65}")

# ── 2.2 Cek rasio ketidakseimbangan kelas ──
max_count = label_counts.max()
min_count = label_counts.min()
imbalance_ratio = max_count / max(min_count, 1)

print(f"\n  Analisis Keseimbangan Kelas:")
print(f"    Kelas terbanyak    : {label_counts.idxmax().capitalize()} ({max_count} ulasan)")
print(f"    Kelas tersedikit   : {label_counts.idxmin().capitalize()} ({min_count} ulasan)")
print(f"    Rasio Ketidakseimbangan : {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print(f"    ⚠ Kelas TIDAK SEIMBANG (rasio > 3:1). Menggunakan class_weight='balanced'")
    print(f"      pada SVM untuk mengatasi ketidakseimbangan ini.")
elif imbalance_ratio > 1.5:
    print(f"    ⚠ Kelas SEDIKIT tidak seimbang. class_weight='balanced' tetap dicoba di GridSearch.")
else:
    print(f"    ✓ Kelas SEIMBANG. Tidak diperlukan penanganan khusus.")

log_done(2, "Eksplorasi Distribusi Kelas", {
    "Total data": str(total_data),
    "Jumlah kelas": str(len(labels_order)),
    "Rasio ketidakseimbangan": f"{imbalance_ratio:.2f}:1"
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: PEMBAGIAN DATA TRAIN/TEST 80:20 (TAHAP 3)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(3, "PEMBAGIAN DATA 80:20 (TRAIN/TEST SPLIT) — STRATIFIED SAMPLING", "✂️")

# ── 3.1 Encode Label Sentimen ke Numerik ──
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

print(f"  Encoding Label Sentimen:")
print(f"  {'─' * 40}")
for idx, name in enumerate(class_names):
    print(f"    {name.capitalize():<15} → Kode: {idx}")

# ── 3.2 Stratified Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded       # Stratified sampling: proporsi kelas tetap terjaga
)

print(f"\n  Hasil Pembagian Data 80:20 (Stratified Sampling):")
print(f"  {'━' * 70}")
print(f"  {'Partisi':<12} {'Jumlah':<12} {'Persentase':<15}", end="")
for name in class_names:
    print(f" {name.capitalize():<12}", end="")
print()
print(f"  {'─' * 70}")

# Data Training
train_counts = pd.Series(y_train).value_counts().sort_index()
print(f"  {'Training':<12} {len(y_train):<12} {len(y_train)/total_data*100:.1f}%{'':10}", end="")
for idx in range(len(class_names)):
    count = train_counts.get(idx, 0)
    print(f" {count:<12}", end="")
print()

# Data Testing
test_counts = pd.Series(y_test).value_counts().sort_index()
print(f"  {'Testing':<12} {len(y_test):<12} {len(y_test)/total_data*100:.1f}%{'':10}", end="")
for idx in range(len(class_names)):
    count = test_counts.get(idx, 0)
    print(f" {count:<12}", end="")
print()

print(f"  {'─' * 70}")

# Verifikasi proporsionalitas
print(f"\n  Verifikasi Proporsi Stratified Sampling:")
for idx, name in enumerate(class_names):
    train_pct = train_counts.get(idx, 0) / len(y_train) * 100
    test_pct = test_counts.get(idx, 0) / len(y_test) * 100
    overall_pct = (label_counts.get(name, 0) / total_data) * 100
    print(f"    {name.capitalize():<12}: Keseluruhan={overall_pct:.1f}% | Training={train_pct:.1f}% | Testing={test_pct:.1f}%")

print(f"\n    ✓ Proporsi kelas sentimen TERJAGA pada kedua partisi 80:20 (Stratified Sampling).")

log_done(3, "Pembagian Data Train/Test 80:20", {
    "Rasio Split": "80% Training : 20% Testing",
    "Data Training": f"{len(y_train)} ulasan ({len(y_train)/total_data*100:.1f}%)",
    "Data Testing": f"{len(y_test)} ulasan ({len(y_test)/total_data*100:.1f}%)",
    "Metode Sampling": "Stratified (proporsi kelas terjaga)",
    "Random State": str(RANDOM_STATE)
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7: PELATIHAN BASELINE SVM — PERBANDINGAN KERNEL (TAHAP 4)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(4, "PELATIHAN BASELINE SVM — PERBANDINGAN KERNEL", "⚙️")

# ── 4.1 Definisi kernel yang akan dibandingkan ──
kernels_to_compare = {
    'Linear': SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=RANDOM_STATE),
    'RBF': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=RANDOM_STATE),
    'Polynomial': SVC(kernel='poly', C=1.0, degree=3, gamma='scale', class_weight='balanced', random_state=RANDOM_STATE),
    'Sigmoid': SVC(kernel='sigmoid', C=1.0, gamma='scale', class_weight='balanced', random_state=RANDOM_STATE),
}

# ── 4.2 Latih dan evaluasi setiap kernel dengan metrik lengkap ──
baseline_results = {}
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"\n  Perbandingan Performa Kernel SVM (Baseline, C=1.0, class_weight='balanced'):")
print(f"  {CV_FOLDS}-Fold Stratified Cross-Validation pada Data Training")
print(f"  {'━' * 85}")
print(f"  {'No':<4} {'Kernel':<15} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<14} {'Waktu'}")
print(f"  {'─' * 85}")

for i, (kernel_name, model) in enumerate(kernels_to_compare.items(), 1):
    start_time = time.time()

    # Cross-validation untuk semua metrik utama
    cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_prec = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision_weighted')
    cv_rec = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall_weighted')
    cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')

    elapsed = time.time() - start_time

    baseline_results[kernel_name] = {
        'accuracy_mean': cv_acc.mean(), 'accuracy_std': cv_acc.std(),
        'precision_mean': cv_prec.mean(), 'precision_std': cv_prec.std(),
        'recall_mean': cv_rec.mean(), 'recall_std': cv_rec.std(),
        'f1_mean': cv_f1.mean(), 'f1_std': cv_f1.std(),
        'time': elapsed
    }

    print(f"  {i:<4} {kernel_name:<15} {cv_acc.mean():.4f}±{cv_acc.std():.3f}  "
          f"{cv_prec.mean():.4f}±{cv_prec.std():.3f}  "
          f"{cv_rec.mean():.4f}±{cv_rec.std():.3f}  "
          f"{cv_f1.mean():.4f}±{cv_f1.std():.3f}  {elapsed:.1f}s")

print(f"  {'─' * 85}")

# ── 4.3 Tentukan kernel baseline terbaik ──
best_baseline_kernel = max(baseline_results, key=lambda k: baseline_results[k]['accuracy_mean'])
best_baseline_acc = baseline_results[best_baseline_kernel]['accuracy_mean']

print(f"\n  Hasil Analisis Baseline:")
print(f"    Kernel Terbaik (Baseline)  : {best_baseline_kernel}")
print(f"    Accuracy Terbaik (Baseline): {best_baseline_acc:.4f}")
print(f"    → Kernel '{best_baseline_kernel}' akan digunakan sebagai basis untuk Hyperparameter Tuning.")

log_done(4, "Perbandingan Kernel Baseline SVM", {
    "Kernel Terbaik": best_baseline_kernel,
    "Accuracy Terbaik": f"{best_baseline_acc:.4f}",
    "Jumlah Kernel Diuji": str(len(kernels_to_compare))
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8: HYPERPARAMETER TUNING — GRIDSEARCHCV (TAHAP 5)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(5, "HYPERPARAMETER TUNING — GRIDSEARCHCV + STRATIFIED K-FOLD", "🔧")

print(f"  Parameter Grid yang akan dicari:")
print(f"  {'─' * 60}")
for param, values in PARAM_GRID.items():
    print(f"    {param:<18} : {values}")

total_combinations = 1
for v in PARAM_GRID.values():
    total_combinations *= len(v)
total_fits = total_combinations * CV_FOLDS

print(f"  {'─' * 60}")
print(f"    Total kombinasi hyperparameter : {total_combinations}")
print(f"    Total model yang akan dilatih  : {total_fits} ({total_combinations} × {CV_FOLDS} folds)")
print(f"\n  ⏳ Memulai GridSearchCV... (proses ini mungkin memakan beberapa menit)")

# ── 5.1 Jalankan GridSearchCV ──
start_time = time.time()

grid_search = GridSearchCV(
    estimator=SVC(random_state=RANDOM_STATE),
    param_grid=PARAM_GRID,
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring=CV_SCORING,
    refit=True,               # Latih ulang model terbaik pada seluruh data training
    n_jobs=-1,                # Gunakan semua core CPU
    verbose=1,                # Tampilkan progres
    return_train_score=True   # Simpan skor training juga (untuk analisis overfitting)
)

grid_search.fit(X_train, y_train)
tuning_time = time.time() - start_time

# ── 5.2 Tampilkan Hasil GridSearchCV ──
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print(f"\n  {'━' * 70}")
print(f"  HASIL HYPERPARAMETER TUNING (GridSearchCV)")
print(f"  {'━' * 70}")
print(f"    Waktu Pencarian          : {tuning_time:.2f} detik")
print(f"    Total Model Dilatih      : {total_fits}")
print(f"    Metrik Evaluasi (Scoring) : {CV_SCORING}")
print(f"  {'─' * 70}")
print(f"    HYPERPARAMETER TERBAIK:")
for param, value in best_params.items():
    print(f"      {param:<18} : {value}")
print(f"  {'─' * 70}")
print(f"    Skor CV Terbaik ({CV_SCORING}) : {best_cv_score:.4f}")
print(f"  {'━' * 70}")

# ── 5.3 Tampilkan Top-5 Kombinasi Terbaik ──
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results_sorted = cv_results.sort_values('rank_test_score').head(5)

print(f"\n  Top 5 Kombinasi Hyperparameter Terbaik:")
print(f"  {'─' * 90}")
print(f"  {'Rank':<6} {'C':<8} {'Kernel':<10} {'Gamma':<8} {'Class Weight':<15} {'F1-Score (Mean±Std)':<22} {'Train Score'}")
print(f"  {'─' * 90}")

for _, row in cv_results_sorted.iterrows():
    params = row['params']
    rank = int(row['rank_test_score'])
    mean_test = row['mean_test_score']
    std_test = row['std_test_score']
    mean_train = row['mean_train_score']
    c_val = params.get('C', '-')
    kernel = params.get('kernel', '-')
    gamma = params.get('gamma', '-')
    cw = params.get('class_weight', '-')
    marker = " ★" if rank == 1 else ""
    print(f"  {rank:<6} {str(c_val):<8} {kernel:<10} {str(gamma):<8} {str(cw):<15} "
          f"{mean_test:.4f}±{std_test:.4f}{'':<5} {mean_train:.4f}{marker}")

print(f"  {'─' * 90}")

# ── 5.4 Ambil model terbaik ──
model_terbaik = grid_search.best_estimator_

log_done(5, "Hyperparameter Tuning GridSearchCV", {
    "Waktu Pencarian": f"{tuning_time:.2f} detik",
    "Kernel Terbaik": best_params.get('kernel', '-'),
    "C Terbaik": str(best_params.get('C', '-')),
    "Gamma Terbaik": str(best_params.get('gamma', '-')),
    "Class Weight": str(best_params.get('class_weight', '-')),
    f"Skor CV Terbaik ({CV_SCORING})": f"{best_cv_score:.4f}"
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9: EVALUASI MODEL TERBAIK PADA DATA UJI (TAHAP 6)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(6, "EVALUASI MODEL TERBAIK PADA DATA UJI (TEST SET)", "🎯")

# ── 6.1 Prediksi pada data uji ──
y_pred = model_terbaik.predict(X_test)

# ── 6.1b Deteksi Overfitting: bandingkan akurasi training vs testing ──
y_train_pred = model_terbaik.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)

# ── 6.2 Hitung SEMUA metrik evaluasi (Weighted & Macro) ──
accuracy = accuracy_score(y_test, y_pred)

precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# ── 6.2b Hitung gap overfitting ──
overfit_gap_acc = train_accuracy - accuracy
overfit_gap_f1 = train_f1 - f1_w

precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

# ── 6.3 Tabel Evaluasi Lengkap (format standar skripsi Indonesia) ──
print(f"\n  ┌────────────────────────────────────────────────────────────────────────┐")
print(f"  │        TABEL 4.SVM2: HASIL EVALUASI MODEL SVM TERBAIK                │")
print(f"  │        Kernel: {best_params.get('kernel','?').upper():<8} | C: {str(best_params.get('C','?')):<6} | "
      f"Gamma: {str(best_params.get('gamma','?')):<6} | "
      f"CW: {str(best_params.get('class_weight','?')):<8}│")
print(f"  ├────────────────────────────────────────────────────────────────────────┤")
print(f"  │  {'Metrik Evaluasi':<25} {'Weighted':>12} {'Macro':>12} {'Micro':>12}     │")
print(f"  ├────────────────────────────────────────────────────────────────────────┤")
print(f"  │  {'Accuracy':<25} {accuracy:>12.4f} {accuracy:>12.4f} {accuracy:>12.4f}     │")
print(f"  │  {'Precision':<25} {precision_w:>12.4f} {precision_macro:>12.4f} {precision_micro:>12.4f}     │")
print(f"  │  {'Recall (Sensitivity)':<25} {recall_w:>12.4f} {recall_macro:>12.4f} {recall_micro:>12.4f}     │")
print(f"  │  {'F1-Score':<25} {f1_w:>12.4f} {f1_macro:>12.4f} {f1_micro:>12.4f}     │")
print(f"  ├────────────────────────────────────────────────────────────────────────┤")
print(f"  │  Data Training: {len(y_train)} | Data Uji: {len(y_test)} | CV Score: {best_cv_score:.4f}{'':>14}│")
print(f"  └────────────────────────────────────────────────────────────────────────┘")

# ── 6.4 Penjelasan Metrik untuk Skripsi ──
print(f"\n  PENJELASAN METRIK EVALUASI (Ref: Sokolova & Lapalme, 2009):")
print(f"  {'─' * 70}")
print(f"    • Accuracy   : Proporsi prediksi benar dari seluruh data uji.")
print(f"    • Precision  : Ketepatan model saat memprediksi suatu kelas.")
print(f"                   (Dari semua yang diprediksi kelas X, berapa yang benar?)")
print(f"    • Recall     : Kemampuan model menemukan seluruh sampel suatu kelas.")
print(f"                   (Dari semua data aktual kelas X, berapa yang terdeteksi?)")
print(f"    • F1-Score   : Rata-rata harmonik Precision dan Recall.")
print(f"                   (Mengukur keseimbangan antara Precision dan Recall)")
print(f"  {'─' * 70}")
print(f"    • Weighted   : Rata-rata berbobot berdasarkan jumlah sampel per kelas.")
print(f"    • Macro      : Rata-rata tanpa bobot (setiap kelas bernilai sama).")
print(f"    • Micro      : Menghitung TP, FP, FN secara global lalu menghitung metrik.")
print(f"  {'─' * 70}")

# ── 6.5 Cross-Validation Score per Fold ──
skf_final = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_acc_folds = cross_val_score(model_terbaik, X_tfidf, y_encoded, cv=skf_final, scoring='accuracy')
cv_prec_folds = cross_val_score(model_terbaik, X_tfidf, y_encoded, cv=skf_final, scoring='precision_weighted')
cv_rec_folds = cross_val_score(model_terbaik, X_tfidf, y_encoded, cv=skf_final, scoring='recall_weighted')
cv_f1_folds = cross_val_score(model_terbaik, X_tfidf, y_encoded, cv=skf_final, scoring='f1_weighted')

print(f"\n  TABEL 4.SVM3: HASIL {CV_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION (MODEL TERBAIK)")
print(f"  {'━' * 80}")
print(f"  {'Fold':<8} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<14}")
print(f"  {'─' * 80}")
for fold_i in range(CV_FOLDS):
    print(f"  Fold {fold_i+1:<3} {cv_acc_folds[fold_i]:<14.4f} {cv_prec_folds[fold_i]:<14.4f} "
          f"{cv_rec_folds[fold_i]:<14.4f} {cv_f1_folds[fold_i]:<14.4f}")
print(f"  {'─' * 80}")
print(f"  {'Mean':<8} {cv_acc_folds.mean():<14.4f} {cv_prec_folds.mean():<14.4f} "
      f"{cv_rec_folds.mean():<14.4f} {cv_f1_folds.mean():<14.4f}")
print(f"  {'Std':<8} {cv_acc_folds.std():<14.4f} {cv_prec_folds.std():<14.4f} "
      f"{cv_rec_folds.std():<14.4f} {cv_f1_folds.std():<14.4f}")
print(f"  {'━' * 80}")

# Interpretasi performa (Ref: Landis & Koch, 1977 — diadaptasi untuk klasifikasi)
if f1_w >= 0.90:
    interpretasi_performa = "Sangat Baik / Excellent (F1 ≥ 0.90)"
elif f1_w >= 0.80:
    interpretasi_performa = "Baik / Good (0.80 ≤ F1 < 0.90)"
elif f1_w >= 0.70:
    interpretasi_performa = "Cukup Baik / Fair (0.70 ≤ F1 < 0.80)"
elif f1_w >= 0.60:
    interpretasi_performa = "Cukup / Moderate (0.60 ≤ F1 < 0.70)"
elif f1_w >= 0.50:
    interpretasi_performa = "Kurang / Poor (0.50 ≤ F1 < 0.60)"
else:
    interpretasi_performa = "Sangat Kurang / Very Poor (F1 < 0.50)"

print(f"\n  Interpretasi Performa Model:")
print(f"    F1-Score Weighted  : {f1_w:.4f}")
print(f"    Kategori           : {interpretasi_performa}")

log_done(6, "Evaluasi Model pada Data Uji", {
    "Accuracy": f"{accuracy:.4f} ({accuracy*100:.2f}%)",
    "Precision (Weighted)": f"{precision_w:.4f}",
    "Recall (Weighted)": f"{recall_w:.4f}",
    "F1-Score (Weighted)": f"{f1_w:.4f}",
    "F1-Score (Macro)": f"{f1_macro:.4f}",
    "CV Accuracy (Mean ± Std)": f"{cv_acc_folds.mean():.4f} ± {cv_acc_folds.std():.4f}",
    "Interpretasi": interpretasi_performa
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10: CONFUSION MATRIX & CLASSIFICATION REPORT DETAIL (TAHAP 7)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(7, "CONFUSION MATRIX & CLASSIFICATION REPORT DETAIL", "📋")

# ── 7.1 Classification Report (teks lengkap) ──
report_text = classification_report(
    y_test, y_pred,
    target_names=[name.capitalize() for name in class_names],
    zero_division=0
)

print(f"\n  TABEL 4.SVM4: CLASSIFICATION REPORT (PER KELAS SENTIMEN)")
print(f"  {'━' * 65}")
for line in report_text.split('\n'):
    print(f"  {line}")
print(f"  {'━' * 65}")

# ── 7.2 Confusion Matrix (teks dengan TP/TN/FP/FN detail) ──
cm = confusion_matrix(y_test, y_pred)
cm_labels = [name.capitalize() for name in class_names]

print(f"\n  TABEL 4.SVM5: CONFUSION MATRIX (MATRIKS KEBINGUNGAN)")
print(f"  {'━' * 60}")
print(f"  {'':20} Prediksi Model SVM")
print(f"  {'':20}", end="")
for lbl in cm_labels:
    print(f" {lbl:<12}", end="")
print()
print(f"  {'─' * 60}")
for i, lbl in enumerate(cm_labels):
    print(f"  Aktual {lbl:<11}", end="")
    for j in range(len(cm_labels)):
        marker = " ✓" if i == j else ""
        print(f" {cm[i][j]:<10}{marker:>2}", end="")
    print()
print(f"  {'─' * 60}")

# ── 7.3 Analisis Detail Per Kelas: TP, TN, FP, FN, Sensitivity, Specificity ──
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

print(f"\n  TABEL 4.SVM6: ANALISIS DETAIL PER KELAS SENTIMEN (TP/TN/FP/FN)")
print(f"  {'━' * 90}")
print(f"  {'Kelas':<12} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} "
      f"{'F1-Score':>10} {'Specificity':>12} {'Support':>8}")
print(f"  {'─' * 90}")

detail_per_kelas = {}
for idx_k, name in enumerate(class_names):
    # Hitung TP, TN, FP, FN untuk setiap kelas (One-vs-Rest)
    tp = cm[idx_k][idx_k]
    fn = cm[idx_k].sum() - tp                                 # Baris aktual - TP
    fp = cm[:, idx_k].sum() - tp                               # Kolom prediksi - TP
    tn = cm.sum() - tp - fp - fn                               # Sisanya

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)                                 # Recall = Sensitivity
    spec = tn / max(tn + fp, 1)                                # Specificity
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    support = int(report_dict[name]['support'])

    detail_per_kelas[name] = {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Precision': prec, 'Recall': rec, 'Specificity': spec, 'F1': f1,
        'Support': support
    }

    print(f"  {name.capitalize():<12} {tp:>6} {tn:>6} {fp:>6} {fn:>6} {prec:>10.4f} {rec:>10.4f} "
          f"{f1:>10.4f} {spec:>12.4f} {support:>8}")

print(f"  {'─' * 90}")

# Hitung rata-rata (macro) Specificity
avg_specificity = np.mean([v['Specificity'] for v in detail_per_kelas.values()])
avg_sensitivity = np.mean([v['Recall'] for v in detail_per_kelas.values()])

print(f"\n  Ringkasan Metrik Tambahan:")
print(f"    Rata-rata Sensitivity (Macro) : {avg_sensitivity:.4f}")
print(f"    Rata-rata Specificity (Macro) : {avg_specificity:.4f}")
print(f"\n  KETERANGAN:")
print(f"    TP (True Positive)  = Prediksi benar untuk kelas tersebut")
print(f"    TN (True Negative)  = Prediksi benar untuk BUKAN kelas tersebut")
print(f"    FP (False Positive) = Salah prediksi sebagai kelas tersebut (Type I Error)")
print(f"    FN (False Negative) = Gagal mendeteksi kelas tersebut (Type II Error)")
print(f"    Sensitivity/Recall  = TP / (TP + FN) — Kemampuan mendeteksi kelas")
print(f"    Specificity         = TN / (TN + FP) — Kemampuan menolak kelas lain")

# ── 7.4 Visualisasi Confusion Matrix (Absolut + Normalisasi) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix (angka absolut)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels,
            linewidths=2, linecolor='black', square=True, cbar=False,
            annot_kws={'size': 16, 'weight': 'bold'}, ax=axes[0])
axes[0].set_title('Confusion Matrix (Absolut)\nModel SVM — Data Uji', fontsize=13, fontweight='bold', pad=15)
axes[0].set_xlabel('Prediksi Model SVM', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Label Aktual (Ground Truth)', fontsize=11, fontweight='bold')

# Confusion Matrix (persentase / normalisasi per baris)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlOrRd', xticklabels=cm_labels, yticklabels=cm_labels,
            linewidths=2, linecolor='black', square=True, cbar=True,
            annot_kws={'size': 14, 'weight': 'bold'}, ax=axes[1],
            cbar_kws={'label': 'Persentase (%)', 'shrink': 0.8})
axes[1].set_title('Confusion Matrix (Normalisasi %)\nModel SVM — Data Uji', fontsize=13, fontweight='bold', pad=15)
axes[1].set_xlabel('Prediksi Model SVM', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Label Aktual (Ground Truth)', fontsize=11, fontweight='bold')

# Tambahkan tanda persen pada annotasi
for text in axes[1].texts:
    text.set_text(text.get_text() + '%')

plt.tight_layout()
plt.savefig(OUTPUT_CONFUSION_MATRIX, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik Confusion Matrix disimpan: {OUTPUT_CONFUSION_MATRIX}")

# ── 7.5 Visualisasi Classification Report sebagai Heatmap ──
report_df = pd.DataFrame(report_dict).T
report_metrics = report_df.loc[class_names, ['precision', 'recall', 'f1-score']].astype(float)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(report_metrics, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=2, linecolor='black',
            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax,
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=[name.capitalize() for name in class_names])
ax.set_title(f'Classification Report Heatmap — Model SVM\n'
             f'(Kernel: {best_params.get("kernel","?").upper()}, C={best_params.get("C","?")}, '
             f'Accuracy: {accuracy:.4f})',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Metrik Evaluasi', fontsize=11, fontweight='bold')
ax.set_ylabel('Kelas Sentimen', fontsize=11, fontweight='bold')

# Tambahkan baris summary
if 'weighted avg' in report_dict:
    wa = report_dict['weighted avg']
    ax.text(0.5, -0.18, f"Weighted Avg → Precision: {wa['precision']:.4f} | "
            f"Recall: {wa['recall']:.4f} | F1-Score: {wa['f1-score']:.4f} | "
            f"Accuracy: {accuracy:.4f}",
            transform=ax.transAxes, ha='center', fontsize=10, fontstyle='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(OUTPUT_CLASSIFICATION_REPORT, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik Classification Report disimpan: {OUTPUT_CLASSIFICATION_REPORT}")

log_done(7, "Confusion Matrix & Classification Report", {
    "Prediksi Benar (TP Total)": str(np.trace(cm)),
    "Prediksi Salah": str(np.sum(cm) - np.trace(cm)),
    "Akurasi Keseluruhan": f"{accuracy:.4f} ({accuracy*100:.2f}%)",
    "Rata-rata Specificity": f"{avg_specificity:.4f}",
    "Rata-rata Sensitivity": f"{avg_sensitivity:.4f}"
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11: VISUALISASI METRIK EVALUASI — BAR CHART, RADAR, CV (TAHAP 8)    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(8, "VISUALISASI METRIK EVALUASI (BAR CHART, RADAR, CV PER FOLD)", "📊")

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISASI 1: BAR CHART — PERBANDINGAN METRIK EVALUASI UTAMA
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── 8.1a Bar Chart: Metrik Keseluruhan (Weighted vs Macro) ──
metrik_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
weighted_values = [accuracy, precision_w, recall_w, f1_w]
macro_values = [accuracy, precision_macro, recall_macro, f1_macro]

x_pos = np.arange(len(metrik_names))
width = 0.32

colors_w = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
colors_m = ['#27ae60', '#2980b9', '#d35400', '#c0392b']

bars_w = axes[0].bar(x_pos - width/2, weighted_values, width, label='Weighted Average',
                      color=colors_w, edgecolor='black', linewidth=0.8)
bars_m = axes[0].bar(x_pos + width/2, macro_values, width, label='Macro Average',
                      color=colors_m, edgecolor='black', linewidth=0.8, alpha=0.75)

axes[0].set_title('Perbandingan Metrik Evaluasi Model SVM\n(Weighted vs Macro Average)',
                   fontsize=12, fontweight='bold', pad=15)
axes[0].set_xlabel('Metrik Evaluasi', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Skor (0.0 — 1.0)', fontsize=11, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(metrik_names, fontweight='bold')
axes[0].set_ylim(0, 1.18)
axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
axes[0].grid(True, linestyle='--', alpha=0.4, axis='y')
axes[0].axhline(y=0.80, color='green', linestyle=':', alpha=0.5, linewidth=1.2)
axes[0].text(len(metrik_names)-0.5, 0.81, 'Target 80%', fontsize=8, color='green', fontstyle='italic')

# Label nilai di atas bar
for bar in bars_w:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
for bar in bars_m:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8,
                 color='#555555')

# ── 8.1b Bar Chart: Metrik Per Kelas Sentimen ──
kelas_names_cap = [n.capitalize() for n in class_names]
precision_per_class = [report_dict[n]['precision'] for n in class_names]
recall_per_class = [report_dict[n]['recall'] for n in class_names]
f1_per_class = [report_dict[n]['f1-score'] for n in class_names]

x_pos2 = np.arange(len(class_names))
width2 = 0.25

bars_p = axes[1].bar(x_pos2 - width2, precision_per_class, width2, label='Precision',
                      color='#3498db', edgecolor='black', linewidth=0.8)
bars_r = axes[1].bar(x_pos2, recall_per_class, width2, label='Recall',
                      color='#e67e22', edgecolor='black', linewidth=0.8)
bars_f = axes[1].bar(x_pos2 + width2, f1_per_class, width2, label='F1-Score',
                      color='#e74c3c', edgecolor='black', linewidth=0.8)

axes[1].set_title('Metrik Evaluasi Per Kelas Sentimen\n(Precision, Recall, F1-Score)',
                   fontsize=12, fontweight='bold', pad=15)
axes[1].set_xlabel('Kelas Sentimen', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Skor (0.0 — 1.0)', fontsize=11, fontweight='bold')
axes[1].set_xticks(x_pos2)
axes[1].set_xticklabels(kelas_names_cap, fontweight='bold')
axes[1].set_ylim(0, 1.18)
axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
axes[1].grid(True, linestyle='--', alpha=0.4, axis='y')

# Label nilai
for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_EVALUASI_METRIK, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik Evaluasi Metrik disimpan: {OUTPUT_EVALUASI_METRIK}")

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISASI 2: RADAR CHART — METRIK PER KELAS SENTIMEN
# ═════════════════════════════════════════════════════════════════════════════

# Siapkan data radar
radar_metrics = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity']
n_metrics = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # Tutup lingkaran

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

colors_radar = {'positif': '#2ecc71', 'netral': '#3498db', 'negatif': '#e74c3c'}
# Fallback jika ada kelas non-standar
for name in class_names:
    if name not in colors_radar:
        colors_radar[name] = '#888888'

for name in class_names:
    d = detail_per_kelas[name]
    values = [d['Precision'], d['Recall'], d['F1'], d['Specificity'], d['Recall']]  # Sensitivity = Recall
    values += values[:1]
    ax.fill(angles, values, alpha=0.15, color=colors_radar.get(name, '#888888'))
    ax.plot(angles, values, 'o-', linewidth=2, markersize=6, color=colors_radar.get(name, '#888888'),
            label=f'{name.capitalize()} (F1={d["F1"]:.3f})')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.set_title(f'Radar Chart Metrik Evaluasi Per Kelas\nModel SVM ({best_params.get("kernel","?").upper()}, '
             f'C={best_params.get("C","?")})',
             fontsize=13, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_RADAR_METRIK, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik Radar Metrik Per Kelas disimpan: {OUTPUT_RADAR_METRIK}")

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISASI 3: CROSS-VALIDATION PER FOLD (BAR CHART)
# ═════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))

fold_labels = [f'Fold {i+1}' for i in range(CV_FOLDS)]
x_fold = np.arange(CV_FOLDS)
width_fold = 0.18

bars_acc_f = ax.bar(x_fold - 1.5*width_fold, cv_acc_folds, width_fold, label='Accuracy',
                     color='#2ecc71', edgecolor='black', linewidth=0.7)
bars_prec_f = ax.bar(x_fold - 0.5*width_fold, cv_prec_folds, width_fold, label='Precision',
                      color='#3498db', edgecolor='black', linewidth=0.7)
bars_rec_f = ax.bar(x_fold + 0.5*width_fold, cv_rec_folds, width_fold, label='Recall',
                     color='#e67e22', edgecolor='black', linewidth=0.7)
bars_f1_f = ax.bar(x_fold + 1.5*width_fold, cv_f1_folds, width_fold, label='F1-Score',
                    color='#e74c3c', edgecolor='black', linewidth=0.7)

# Garis rata-rata
ax.axhline(y=cv_acc_folds.mean(), color='#2ecc71', linestyle='--', alpha=0.6, linewidth=1)
ax.axhline(y=cv_f1_folds.mean(), color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1)

ax.set_title(f'{CV_FOLDS}-Fold Stratified Cross-Validation — Seluruh Metrik Per Fold\n'
             f'Model SVM Terbaik (Kernel: {best_params.get("kernel","?").upper()}, C={best_params.get("C","?")})',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Fold Cross-Validation', fontsize=11, fontweight='bold')
ax.set_ylabel('Skor Metrik', fontsize=11, fontweight='bold')
ax.set_xticks(x_fold)
ax.set_xticklabels(fold_labels, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')

# Label nilai per bar
for bars in [bars_acc_f, bars_prec_f, bars_rec_f, bars_f1_f]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                rotation=90)

# Tambahkan summary text di bawah grafik
summary_text = (f"Mean ± Std → Accuracy: {cv_acc_folds.mean():.4f}±{cv_acc_folds.std():.4f} | "
                f"Precision: {cv_prec_folds.mean():.4f}±{cv_prec_folds.std():.4f} | "
                f"Recall: {cv_rec_folds.mean():.4f}±{cv_rec_folds.std():.4f} | "
                f"F1: {cv_f1_folds.mean():.4f}±{cv_f1_folds.std():.4f}")
ax.text(0.5, -0.12, summary_text, transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(OUTPUT_CV_PER_FOLD, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik CV Per Fold disimpan: {OUTPUT_CV_PER_FOLD}")

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISASI 4: PERBANDINGAN KERNEL BASELINE SVM
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

kernel_names_list = list(baseline_results.keys())
acc_means = [baseline_results[k]['accuracy_mean'] for k in kernel_names_list]
acc_stds = [baseline_results[k]['accuracy_std'] for k in kernel_names_list]
f1_means = [baseline_results[k]['f1_mean'] for k in kernel_names_list]
f1_stds = [baseline_results[k]['f1_std'] for k in kernel_names_list]

colors_kernel = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# Chart 1: Accuracy
bars_acc = axes[0].bar(kernel_names_list, acc_means, yerr=acc_stds, color=colors_kernel,
                        edgecolor='black', linewidth=0.8, capsize=5, error_kw={'linewidth': 1.5})
axes[0].set_title('Perbandingan Accuracy per Kernel SVM\n(Baseline, 5-Fold CV)', fontsize=12, fontweight='bold', pad=12)
axes[0].set_xlabel('Kernel SVM', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Accuracy (Mean ± Std)', fontsize=11, fontweight='bold')
axes[0].set_ylim(0, 1.15)
axes[0].grid(True, linestyle='--', alpha=0.5)

for bar, mean_val in zip(bars_acc, acc_means):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Chart 2: F1-Score
bars_f1 = axes[1].bar(kernel_names_list, f1_means, yerr=f1_stds, color=colors_kernel,
                       edgecolor='black', linewidth=0.8, capsize=5, error_kw={'linewidth': 1.5})
axes[1].set_title('Perbandingan F1-Score (Weighted) per Kernel SVM\n(Baseline, 5-Fold CV)', fontsize=12, fontweight='bold', pad=12)
axes[1].set_xlabel('Kernel SVM', fontsize=11, fontweight='bold')
axes[1].set_ylabel('F1-Score Weighted (Mean ± Std)', fontsize=11, fontweight='bold')
axes[1].set_ylim(0, 1.15)
axes[1].grid(True, linestyle='--', alpha=0.5)

for bar, mean_val in zip(bars_f1, f1_means):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_KERNEL_COMPARISON, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik Perbandingan Kernel disimpan: {OUTPUT_KERNEL_COMPARISON}")

log_done(8, "Visualisasi Metrik Evaluasi Lengkap", {
    "Grafik Evaluasi Metrik": OUTPUT_EVALUASI_METRIK,
    "Grafik Radar Metrik": OUTPUT_RADAR_METRIK,
    "Grafik CV Per Fold": OUTPUT_CV_PER_FOLD,
    "Grafik Perbandingan Kernel": OUTPUT_KERNEL_COMPARISON
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 12: LEARNING CURVE — ANALISIS BIAS-VARIANS (TAHAP 9)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(9, "LEARNING CURVE — ANALISIS BIAS-VARIANS MODEL SVM", "📈")

print(f"  Menghitung Learning Curve untuk model SVM terbaik...")
print(f"  (Kernel: {best_params.get('kernel','?').upper()}, C={best_params.get('C','?')})")
print(f"  Menggunakan {CV_FOLDS}-Fold Stratified Cross-Validation...\n")

# ── 9.1 Hitung Learning Curve ──
train_sizes_abs, train_scores, val_scores = learning_curve(
    model_terbaik, X_tfidf, y_encoded,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring='accuracy',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
val_scores_mean = val_scores.mean(axis=1)
val_scores_std = val_scores.std(axis=1)

# ── 9.2 Tampilkan tabel learning curve ──
print(f"  TABEL 4.SVM7: LEARNING CURVE DATA")
print(f"  {'━' * 75}")
print(f"  {'No':<4} {'Train Size':<14} {'Train Acc (Mean±Std)':<25} {'Val Acc (Mean±Std)':<25} {'Gap'}")
print(f"  {'─' * 75}")
for i in range(len(train_sizes_abs)):
    gap = train_scores_mean[i] - val_scores_mean[i]
    gap_marker = "⚠" if gap > 0.10 else "✓"
    print(f"  {i+1:<4} {train_sizes_abs[i]:<14} "
          f"{train_scores_mean[i]:.4f}±{train_scores_std[i]:.4f}{'':>8} "
          f"{val_scores_mean[i]:.4f}±{val_scores_std[i]:.4f}{'':>8} "
          f"{gap:.4f} {gap_marker}")
print(f"  {'─' * 75}")

# Analisis bias-varians
final_gap = train_scores_mean[-1] - val_scores_mean[-1]
if final_gap > 0.15:
    bias_varians = "High Variance (Overfitting)"
    saran = "Pertimbangkan regularisasi lebih kuat (C lebih kecil) atau tambah data."
elif final_gap > 0.05:
    bias_varians = "Sedikit Overfitting (Normal)"
    saran = "Model cukup baik. Gap kecil antara training dan validasi."
elif val_scores_mean[-1] < 0.60:
    bias_varians = "High Bias (Underfitting)"
    saran = "Pertimbangkan fitur tambahan atau model yang lebih kompleks."
else:
    bias_varians = "Good Fit (Seimbang)"
    saran = "Model memiliki keseimbangan bias-varians yang baik."

print(f"\n  Analisis Bias-Varians:")
print(f"    Gap Training-Validasi Akhir : {final_gap:.4f}")
print(f"    Diagnosis                   : {bias_varians}")
print(f"    Saran                       : {saran}")

# ── 9.3 Visualisasi Learning Curve ──
fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.15, color='#3498db')
ax.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                val_scores_mean + val_scores_std, alpha=0.15, color='#e74c3c')

ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='#3498db', linewidth=2.5,
        markersize=8, label=f'Training Score (Final: {train_scores_mean[-1]:.4f})')
ax.plot(train_sizes_abs, val_scores_mean, 's-', color='#e74c3c', linewidth=2.5,
        markersize=8, label=f'Validation Score (Final: {val_scores_mean[-1]:.4f})')

ax.set_title(f'Learning Curve — Model SVM\n'
             f'(Kernel: {best_params.get("kernel","?").upper()}, C={best_params.get("C","?")}, '
             f'{CV_FOLDS}-Fold CV)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Jumlah Data Training', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)

# Tambahkan anotasi gap
ax.annotate(f'Gap: {final_gap:.4f}\n({bias_varians})',
            xy=(train_sizes_abs[-1], (train_scores_mean[-1] + val_scores_mean[-1]) / 2),
            xytext=(-120, 30), textcoords='offset points',
            fontsize=10, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(OUTPUT_LEARNING_CURVE, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik Learning Curve disimpan: {OUTPUT_LEARNING_CURVE}")

log_done(9, "Learning Curve & Analisis Bias-Varians", {
    "Training Score Akhir": f"{train_scores_mean[-1]:.4f}",
    "Validation Score Akhir": f"{val_scores_mean[-1]:.4f}",
    "Gap Akhir": f"{final_gap:.4f}",
    "Diagnosis": bias_varians
})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 13: VISUALISASI DISTRIBUSI PREDIKSI (TAHAP 9 LANJUTAN)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Decode prediksi ke label asli ──
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# ── Visualisasi Distribusi: Aktual vs Prediksi ──
colors_sent = {'positif': '#2ecc71', 'netral': '#95a5a6', 'negatif': '#e74c3c'}
labels_display = [l for l in labels_order if l in class_names]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Chart 1: Distribusi Label Aktual vs Prediksi (Grouped Bar)
actual_counts = pd.Series(y_test_labels).value_counts()
pred_counts = pd.Series(y_pred_labels).value_counts()

x_pos = np.arange(len(labels_display))
width = 0.35

bars1 = axes[0].bar(x_pos - width/2, [actual_counts.get(l, 0) for l in labels_display],
                     width, label='Aktual (Ground Truth)', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = axes[0].bar(x_pos + width/2, [pred_counts.get(l, 0) for l in labels_display],
                     width, label='Prediksi SVM', color='#e67e22', edgecolor='black', linewidth=0.8)

axes[0].set_title('Perbandingan Distribusi Aktual vs Prediksi SVM\n(Data Uji)', fontsize=12, fontweight='bold', pad=12)
axes[0].set_xlabel('Kelas Sentimen', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Jumlah Ulasan', fontsize=11, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([l.capitalize() for l in labels_display])
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)

# Tambah label angka
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold', fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold', fontsize=9)

# Chart 2: Pie Chart Prediksi SVM
pred_counts_ordered = [pred_counts.get(l, 0) for l in labels_display]
colors_ordered = [colors_sent.get(l, '#888888') for l in labels_display]

axes[1].pie(pred_counts_ordered,
            labels=[f'{l.capitalize()}\n({c} ulasan)' for l, c in zip(labels_display, pred_counts_ordered)],
            colors=colors_ordered, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
axes[1].set_title('Proporsi Prediksi Sentimen SVM\n(Data Uji)', fontsize=12, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_DISTRIBUSI_PREDIKSI, dpi=300, bbox_inches='tight')
plt.show()
print(f"    ✓ Grafik Distribusi Prediksi disimpan: {OUTPUT_DISTRIBUSI_PREDIKSI}")

# ── Hitung statistik prediksi ──
correct_mask = y_test == y_pred
wrong_mask = ~correct_mask


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 14: SIMPAN MODEL, EKSPOR HASIL, & RINGKASAN AKHIR (TAHAP 10)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

log_header(10, "SIMPAN MODEL & EKSPOR HASIL KLASIFIKASI", "💾")

# ── 10.1 Simpan Model SVM Terbaik ──
with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump({
        'model': model_terbaik,
        'label_encoder': le,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'accuracy': accuracy,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'specificity_macro': avg_specificity,
        'class_names': list(class_names),
        'detail_per_kelas': detail_per_kelas,
        'cv_scores': {
            'accuracy': cv_acc_folds.tolist(),
            'precision': cv_prec_folds.tolist(),
            'recall': cv_rec_folds.tolist(),
            'f1': cv_f1_folds.tolist()
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, f)
print(f"  ✓ Model SVM terbaik disimpan: {OUTPUT_MODEL}")
print(f"    Berisi: model terlatih, label encoder, hyperparameter, semua metrik evaluasi")

# ── 10.2 Buat DataFrame Hasil Prediksi (Seluruh Data) ──
# Prediksi pada seluruh dataset untuk ekspor
y_all_pred = model_terbaik.predict(X_tfidf)
y_all_pred_labels = le.inverse_transform(y_all_pred)

df_output = df.copy()
df_output['prediksi_svm'] = y_all_pred_labels

# Tandai apakah prediksi benar/salah (jika label asli tersedia)
if LABEL_COLUMN in df_output.columns:
    df_output['status_prediksi'] = np.where(
        df_output[LABEL_COLUMN] == df_output['prediksi_svm'],
        'Benar', 'Salah'
    )
    akurasi_all = (df_output['status_prediksi'] == 'Benar').mean()
    print(f"\n  Akurasi pada seluruh dataset: {akurasi_all:.4f} ({akurasi_all*100:.2f}%)")

# Ekspor CSV
df_output.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"  ✓ Hasil klasifikasi disimpan: {OUTPUT_CSV}")
print(f"    Total baris tersimpan: {len(df_output)}")

# ── 10.3 Tabel Cross-Tabulation Sentimen Aktual vs Prediksi SVM ──
if LABEL_COLUMN in df_output.columns:
    print(f"\n  TABEL 4.SVM8: CROSS-TABULATION SENTIMEN AKTUAL × PREDIKSI SVM (SELURUH DATA)")
    print(f"  {'━' * 70}")
    crosstab = pd.crosstab(
        df_output[LABEL_COLUMN],
        df_output['prediksi_svm'],
        margins=True,
        margins_name='Total'
    )
    print(crosstab.to_string())
    print(f"  {'━' * 70}")

# ── 10.4 Download otomatis di Google Colab ──
try:
    from google.colab import files
    print(f"\n  ⏳ Menyiapkan unduhan otomatis untuk Google Colab...")

    download_files = [
        OUTPUT_MODEL,
        OUTPUT_CSV,
        OUTPUT_CONFUSION_MATRIX,
        OUTPUT_CLASSIFICATION_REPORT,
        OUTPUT_KERNEL_COMPARISON,
        OUTPUT_LEARNING_CURVE,
        OUTPUT_DISTRIBUSI_PREDIKSI,
        OUTPUT_EVALUASI_METRIK,
        OUTPUT_RADAR_METRIK,
        OUTPUT_CV_PER_FOLD
    ]

    for fpath in download_files:
        try:
            files.download(fpath)
            time.sleep(0.5)
        except Exception as e:
            print(f"    [WARNING] Gagal mengunduh {fpath}: {e}")

    print(f"  ✓ Semua berkas berhasil terunduh otomatis ke komputer lokal Anda.")
except ImportError:
    print(f"\n  [INFO] Berjalan di lingkungan lokal. File disimpan di direktori aktif.")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RINGKASAN AKHIR KLASIFIKASI SVM                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n" + "=" * 80)
print("         RINGKASAN AKHIR KLASIFIKASI SENTIMEN SVM")
print("         (Untuk Bab 4 Hasil Penelitian — Skripsi)")
print("=" * 80)

print(f"\n  1. DATA PENELITIAN:")
print(f"     - Ukuran Dataset              : {len(df)} ulasan")
print(f"     - Jumlah Fitur TF-IDF         : {X_tfidf.shape[1]} fitur")
print(f"     - Rasio Pembagian Data        : 80% Training : 20% Testing")
print(f"     - Data Training               : {len(y_train)} ulasan ({len(y_train)/total_data*100:.1f}%)")
print(f"     - Data Testing                : {len(y_test)} ulasan ({len(y_test)/total_data*100:.1f}%)")

print(f"\n  2. PERBANDINGAN KERNEL BASELINE SVM:")
print(f"     {'Kernel':<15} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<14}")
print(f"     {'─' * 65}")
for kname, res in baseline_results.items():
    marker = " ★" if kname == best_baseline_kernel else ""
    print(f"     {kname:<15} {res['accuracy_mean']:.4f}{'':>8} {res['precision_mean']:.4f}{'':>8} "
          f"{res['recall_mean']:.4f}{'':>8} {res['f1_mean']:.4f}{marker}")

print(f"\n  3. MODEL TERBAIK (Setelah GridSearchCV):")
print(f"     - Kernel                      : {best_params.get('kernel', '?').upper()}")
print(f"     - C (Regularization)          : {best_params.get('C', '?')}")
print(f"     - Gamma                       : {best_params.get('gamma', '?')}")
print(f"     - Class Weight                : {best_params.get('class_weight', '?')}")
print(f"     - CV Score ({CV_SCORING})   : {best_cv_score:.4f}")

print(f"\n  4. HASIL EVALUASI PADA DATA UJI:")
print(f"     ┌────────────────────────────────────────────────────┐")
print(f"     │ Metrik              Weighted    Macro      Micro   │")
print(f"     ├────────────────────────────────────────────────────┤")
print(f"     │ Accuracy          {accuracy:>10.4f} {accuracy:>10.4f} {accuracy:>10.4f} │")
print(f"     │ Precision         {precision_w:>10.4f} {precision_macro:>10.4f} {precision_micro:>10.4f} │")
print(f"     │ Recall            {recall_w:>10.4f} {recall_macro:>10.4f} {recall_micro:>10.4f} │")
print(f"     │ F1-Score          {f1_w:>10.4f} {f1_macro:>10.4f} {f1_micro:>10.4f} │")
print(f"     └────────────────────────────────────────────────────┘")
print(f"     - Interpretasi Performa       : {interpretasi_performa}")

print(f"\n  5. METRIK DETAIL PER KELAS:")
print(f"     {'Kelas':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Specificity':>12} {'Support':>8}")
print(f"     {'─' * 62}")
for name in class_names:
    d = detail_per_kelas[name]
    print(f"     {name.capitalize():<12} {d['Precision']:>10.4f} {d['Recall']:>10.4f} "
          f"{d['F1']:>10.4f} {d['Specificity']:>12.4f} {d['Support']:>8}")
print(f"     {'─' * 62}")
print(f"     Rata-rata Sensitivity (Macro) : {avg_sensitivity:.4f}")
print(f"     Rata-rata Specificity (Macro) : {avg_specificity:.4f}")

print(f"\n  6. CROSS-VALIDATION ({CV_FOLDS}-FOLD):")
print(f"     - Accuracy  Mean ± Std        : {cv_acc_folds.mean():.4f} ± {cv_acc_folds.std():.4f}")
print(f"     - Precision Mean ± Std        : {cv_prec_folds.mean():.4f} ± {cv_prec_folds.std():.4f}")
print(f"     - Recall    Mean ± Std        : {cv_rec_folds.mean():.4f} ± {cv_rec_folds.std():.4f}")
print(f"     - F1-Score  Mean ± Std        : {cv_f1_folds.mean():.4f} ± {cv_f1_folds.std():.4f}")

print(f"\n  7. ANALISIS BIAS-VARIANS:")
print(f"     - Gap Training-Validasi       : {final_gap:.4f}")
print(f"     - Diagnosis                   : {bias_varians}")

print(f"\n  8. FILE OUTPUT ({len([OUTPUT_MODEL, OUTPUT_CSV, OUTPUT_CONFUSION_MATRIX, OUTPUT_CLASSIFICATION_REPORT, OUTPUT_KERNEL_COMPARISON, OUTPUT_LEARNING_CURVE, OUTPUT_DISTRIBUSI_PREDIKSI, OUTPUT_EVALUASI_METRIK, OUTPUT_RADAR_METRIK, OUTPUT_CV_PER_FOLD])} file):")
print(f"     - Model SVM                   : {OUTPUT_MODEL}")
print(f"     - Hasil Klasifikasi (CSV)     : {OUTPUT_CSV}")
print(f"     - Confusion Matrix            : {OUTPUT_CONFUSION_MATRIX}")
print(f"     - Classification Report       : {OUTPUT_CLASSIFICATION_REPORT}")
print(f"     - Evaluasi Metrik (Bar Chart) : {OUTPUT_EVALUASI_METRIK}")
print(f"     - Radar Metrik Per Kelas      : {OUTPUT_RADAR_METRIK}")
print(f"     - CV Per Fold                 : {OUTPUT_CV_PER_FOLD}")
print(f"     - Perbandingan Kernel         : {OUTPUT_KERNEL_COMPARISON}")
print(f"     - Learning Curve              : {OUTPUT_LEARNING_CURVE}")
print(f"     - Distribusi Prediksi         : {OUTPUT_DISTRIBUSI_PREDIKSI}")
print("=" * 80)
print(" PROSES KLASIFIKASI SVM SELESAI DENGAN SUKSES!")
print("=" * 80)
