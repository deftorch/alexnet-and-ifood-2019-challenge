# Deep Learning: AlexNet Implementation for iFood 2019

Repositori ini berisi implementasi AlexNet dan variannya untuk klasifikasi gambar pada dataset iFood 2019, sebagai bagian dari tugas mata kuliah Kecerdasan Buatan.

## 📁 Struktur Direktori
- `src/models/`: Definisi arsitektur model (AlexNet Baseline & Modifikasi).
- `src/data_loader.py`: Logika loading dataset dan augmentasi.
- `src/train.py`: Script utama untuk pelatihan model.
- `src/evaluate.py`: Script untuk evaluasi model pada set validasi/test.
- `src/create_mock_data.py`: Script bantu untuk membuat data dummy (untuk debugging).

## 🚀 Cara Menjalankan

### 1. Instalasi
Pastikan Python 3.8+ terinstal, lalu jalankan:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.
```

### 2. Persiapan Data
Unduh dataset iFood 2019 dan ekstrak ke folder `data/` (atau sesuai konfigurasi).
Struktur folder yang diharapkan:
```
data/
  train_images/
  val_images/
  test_images/
  train_info.csv
  val_info.csv
  test_info.csv
  class_list.txt
```

*Jika ingin mencoba dengan data dummy, jalankan:*
```bash
python src/create_mock_data.py
```

### 3. Training
Gunakan `src/train.py` untuk melatih model. Anda bisa memilih varian model dengan argumen `--model_name`.

**Pilihan Model:**
- `alexnet_baseline`: AlexNet standar.
- `alexnet_mod1`: AlexNet + Batch Normalization.
- `alexnet_mod2`: AlexNet + LeakyReLU.
- `alexnet_combined`: AlexNet + Batch Normalization + LeakyReLU.

**Contoh Command:**
```bash
# Latih Baseline
python src/train.py --data_dir data_mock --model_name alexnet_baseline --num_epochs 10

# Latih Model Modifikasi (Combined) dengan Logging WandB
python src/train.py --data_dir data_mock --model_name alexnet_combined --use_wandb
```

### 4. Evaluasi
Setelah training, file model (`.pth`) akan tersimpan. Gunakan `src/evaluate.py` untuk mengukur akurasi.

```bash
python src/evaluate.py --data_dir data_mock --model_path model_alexnet_combined.pth --model_name alexnet_combined
```

## 📊 Laporan & Analisis
Ringkasan paper AlexNet dan analisis hasil eksperimen dapat dilihat di file [REPORT.md](REPORT.md).
