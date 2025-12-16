# Laporan Penilaian Tugas Deep Learning (AlexNet & iFood 2019)

## 1. Ringkasan
Secara umum, implementasi kode mahasiswa sudah terstruktur dengan baik dan mengikuti standar praktik *Deep Learning* menggunakan PyTorch. Mahasiswa berhasil mengimplementasikan arsitektur AlexNet dasar beserta modifikasinya (Batch Normalization dan LeakyReLU). Struktur repositori rapi dan modular. Namun, terdapat beberapa aspek yang belum lengkap sesuai dengan spesifikasi tugas, terutama terkait penanganan *class imbalance* dan dokumen laporan analisis.

## 2. Validasi Fungsionalitas
Berikut adalah hasil validasi yang telah dilakukan terhadap kode:

*   **Instalasi Dependensi**: `requirements.txt` tidak tersedia, namun kode berjalan lancar dengan dependensi standar (`torch`, `torchvision`, `pandas`, `wandb`, `pillow`).
*   **Generasi Data Mock**: Script `create_mock_data.py` berjalan dengan baik dan berhasil membuat struktur direktori yang dibutuhkan (`train`, `val`, `test`).
*   **Model Instantiation**: Script `test_models.py` berhasil dijalankan. Semua varian model (`baseline`, `mod1`, `mod2`, `combined`) dapat diinisialisasi dan melakukan *forward pass* tanpa error.
*   **Training Loop**: Script `train.py` berhasil dijalankan (uji coba 1 epoch). Integrasi dengan Weights & Biases (`wandb`) tersedia.
*   **Evaluasi**: Script `evaluate.py` berjalan normal dan menghasilkan metrik akurasi serta matriks kebingungan (confusion matrix).

## 3. Analisis Pemenuhan Syarat (Compliance Checklist)

| No | Syarat | Status | Catatan |
|----|--------|--------|---------|
| 1 | Ringkasan Paper AlexNet | ❌ Tidak Ditemukan | Tidak ada file (PDF/MD) yang berisi ringkasan paper. |
| 2 | Implementasi Baseline AlexNet | ✅ Terpenuhi | Terimplementasi di `src/models/alexnet.py`. |
| 3 | Modifikasi Arsitektur (2 jenis) | ✅ Terpenuhi | Menggunakan **Batch Normalization** dan **LeakyReLU**. |
| 4 | Eksperimen (A, B, C, D) | ✅ Terpenuhi | Kode mendukung pemilihan model lewat argumen `--model_name`. |
| 5 | Train/Val/Test Split | ✅ Terpenuhi | `data_loader.py` menangani split sesuai file CSV. |
| 6 | Penanganan Class Imbalance | ⚠️ Parsial | Augmentasi data ada, tetapi *Class Weighting* belum diimplementasikan (hanya komentar `# For now, we use standard CrossEntropyLoss`). |
| 7 | Data Augmentation | ✅ Terpenuhi | RandomCrop dan RandomHorizontalFlip diterapkan. |
| 8 | Hyperparameter Tuning | ✅ Terpenuhi | Argumen untuk LR, Batch Size, Epochs tersedia. |
| 9 | Logging (WandB) | ✅ Terpenuhi | Kode integrasi `wandb` sudah ada. |
| 10 | Dokumentasi/Laporan | ⚠️ Kurang | README berisi soal tugas, bukan laporan mahasiswa. Tidak ada analisis hasil eksperimen. |

## 4. Evaluasi Kode (Code Quality)

*   **Struktur Direktori**: Sangat baik. Pemisahan antara `models`, `data_loader`, dan script eksekusi (`train.py`, `evaluate.py`) membuat kode mudah dibaca dan dikelola.
*   **Kualitas Kode**:
    *   Penggunaan `argparse` memudahkan eksperimen.
    *   Kelas `AlexNetCustom` ditulis dengan cukup fleksibel namun sedikit repetitif pada blok `if self.modification == ...`. Bisa dioptimalkan dengan helper function untuk *block building*.
    *   Penanganan error pada `data_loader` (jika gambar tidak ditemukan) cukup baik untuk mencegah *crash* saat training.
*   **Reproducibility**: Penggunaan seed belum eksplisit di set, namun struktur kode mendukung reproduktifitas jika argumen yang sama digunakan.

## 5. Kekurangan & Rekomendasi Perbaikan

### Kekurangan Utama:
1.  **Laporan Ringkasan & Analisis**: Mahasiswa belum menyertakan ringkasan paper AlexNet dan analisis hasil eksperimen (perbandingan akurasi model A, B, C, D).
2.  **Class Imbalance Handling**: Kode untuk *weighted loss* masih berupa komentar. Seharusnya hitung frekuensi kelas dari data latih dan gunakan sebagai bobot di `CrossEntropyLoss`.

### Rekomendasi untuk Mahasiswa:
1.  **Buat Laporan**: Tambahkan file `REPORT.md` yang berisi ringkasan paper, tabel hasil eksperimen, dan analisis mengapa modifikasi tertentu memberikan hasil lebih baik/buruk.
2.  **Implementasi Class Weighting**:
    ```python
    # Contoh implementasi di train.py
    # Hitung bobot berdasarkan frekuensi kelas di dataset
    class_counts = ... # logic hitung jumlah sampel per kelas
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    ```
3.  **Requirements File**: Tambahkan `requirements.txt` agar memudahkan *reviewer* menginstal library yang tepat.

## 6. Kesimpulan Penilaian

Berdasarkan rubrik singkat:
*   **Ringkasan Paper**: 0/10 (Tidak ada)
*   **Implementasi Baseline**: 20/20 (Sangat baik)
*   **Eksperimen & Analisis**: 15/30 (Kode eksperimen ada, tapi laporan analisis & hasil tidak ada)
*   **Best Practices**: 15/20 (Imbalance handling belum lengkap, dokumentasi kurang)
*   **Kode/Teknis**: 20/20 (Kode berjalan baik, struktur bagus)

**Estimasi Nilai Sementara: 70/100**
*Mahasiswa disarankan untuk segera melengkapi laporan analisis dan ringkasan paper serta mengaktifkan fitur class weighting untuk mendapatkan nilai maksimal.*
