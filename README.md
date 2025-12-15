# Tugas Mata Kuliah Kecerdasan Buatan  
## Deep Learningm for Image Classification — AlexNet & iFood 2019 Challenge

### 📌 Pengantar  

Paper *"ImageNet Classification with Deep Convolutional Neural Networks"* (AlexNet) oleh Krizhevsky, Sutskever, dan Hinton (2012) merupakan tonggak revolusi dalam bidang **Artificial Intelligence** dan **Deep Learning**.  
Paper ini menunjukkan bahwa **deep convolutional networks** dapat mengungguli sistem vision tradisional dalam klasifikasi gambar skala besar (ImageNet).  
Keberhasilan arsitektur AlexNet mengawali era kemajuan pesat dalam **komputer vision**, **GPU‑accelerated training**, **CNN modern**, hingga aplikasi AI dalam kehidupan sehari‑hari.

Paper dapat diakses di link berikut:  
[https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

---

### 🎯 Deskripsi Proyek  

#### 1️⃣ Ringkasan Paper  
- Membuat ringkasan paper *“ImageNet Classification with Deep CNN”*  
  Mencakup:  
  - Motivasi  
  - Arsitektur AlexNet  
  - Teknik baru (ReLU, dropout, multi‑GPU, dll.)  
  - Hasil eksperimen & dampak penelitian  

#### 2️⃣ Implementasi Baseline AlexNet  
- Implementasi arsitektur AlexNet sebagai baseline
- Dataset challenge: **iFood 2019**  
  - Link resmi challenge: https://www.kaggle.com/c/ifood-2019-fgvc6
  - Link dataset: https://github.com/karansikka1/iFood_2019
  - Jenis: **Fine‑grained food classification**
  - Jumlah kelas: 251  
  - Tantangan: variasi makanan, kondisi pengambilan gambar, class imbalance  

#### 3️⃣ Modifikasi Arsitektur  
Lakukan **dua** modifikasi berbeda terhadap AlexNet, contoh:  
- Mengganti aktivasi (misal: ReLU → LeakyReLU / GELU)  
- Menambahkan **Batch Normalization**
- Mengubah konfigurasi pooling atau fully‑connected layer
- Menambahkan regularisasi tambahan

#### 4️⃣ Eksperimen Performansi  
Lakukan eksperimen berikut:

| Eksperimen | Model | Modifikasi |
|---|---|---|
| A | AlexNet baseline | - |
| B | AlexNet Modified 1 | 1 modifikasi |
| C | AlexNet Modified 2 | 1 modifikasi |
| D | AlexNet Modified (1+2) | 2 modifikasi |

Output eksperimen:  
- Train & Validation Accuracy
- Test Accuracy / Confusion Matrix
- Analisis peningkatan performa

---

### 🔍 Best Practices Machine Learning (WAJIB)
 Harus menerapkan:
- Train/Validation/Test split seperti yang sudah disediakan di web iFood 2019.
- Pemeriksaan **class imbalance**, solusi:  
  - Augmentasi data  
  - Class weighting / oversampling bila perlu
- Data Augmentation (rotation, flip, color jitter, dll.)
- Hyperparameter tuning (learning rate, batch size)
- Logging metrik pelatihan. Gunakan tools seperti [Weights & Biases](https://wandb.ai/site/).
- Dokumentasi kode yang jelas

---

### 🏁 Penilaian (Rubrik Singkat)
| Aspek | Penilaian |
|---|---|
| Ringkasan paper | Pemahaman & ketepatan isi |
| Implementasi baseline | Kebenaran & kelengkapan |
| Eksperimen & analisis | Kualitas eksperimen, evaluasi, dan kesimpulan |
| Dokumentasi | Reproducibility, struktur repo, laporan |
| Presentasi | Jelas, padat, komunikatif |

---

### 🚀 Catatan Tambahan
1. Gunakan Google Colab / Kaggle Kernels untuk eksperimen jika tidak memiliki GPU lokal. 
2. Dataset harus disimpan di google drive agar tidak mengupload ulang setiap kali runtime di-restart.
3. Pastikan semua dependensi tercantum di `requirements.txt` atau `environment.yml`.
4. Manfaatkan pre-trained weights AlexNet dari PyTorch untuk transfer learning jika diperlukan.
5. Jangan lupa untuk melakukan commit dan push secara berkala ke repository GitHub Anda.

---
### 📚 Referensi
- Paper asli AlexNet: [https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Kaggle iFood 2019 Challenge: https://www.kaggle.com/c/ifood-2019-fgvc6
- iFood 2019 Dataset: https://github.com/karansikka1/iFood_2019
- PyTorch AlexNet Documentation: https://docs.pytorch.org/vision/main/models/alexnet.html
- Weights & Biases for Experiment Tracking: https://wandb.ai/site/
- Data Augmentation Techniques: https://docs.pytorch.org/vision/stable/transforms.html
- Transfer Learning Guide: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---
### 🎉 Selamat mengerjakan tugas!
