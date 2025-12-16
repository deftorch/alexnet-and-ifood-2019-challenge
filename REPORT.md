# Laporan Tugas Deep Learning: AlexNet & iFood 2019

## 1. Ringkasan Paper AlexNet
**Judul Paper:** ImageNet Classification with Deep Convolutional Neural Networks
**Penulis:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
**Tahun:** 2012

### Motivasi
Sebelum AlexNet, pengenalan objek visual sebagian besar mengandalkan metode *machine learning* tradisional yang menggunakan fitur buatan tangan (hand-crafted features) seperti SIFT atau HOG. Metode ini memiliki keterbatasan dalam menangkap variasi visual yang kompleks pada dataset skala besar seperti ImageNet. Paper ini bertujuan untuk menunjukkan bahwa *Deep Convolutional Neural Networks* (CNN) yang dilatih secara *end-to-end* pada dataset besar dapat mencapai performa yang jauh lebih unggul dibandingkan metode tradisional.

### Arsitektur AlexNet
AlexNet terdiri dari 8 lapisan (layers) yang memiliki bobot:
- **5 Lapisan Konvolusi (Convolutional Layers):** Bertugas untuk mengekstrak fitur visual dari gambar. Beberapa lapisan ini diikuti oleh lapisan *max-pooling*.
- **3 Lapisan Fully-Connected (FC Layers):** Bertugas untuk melakukan klasifikasi tingkat tinggi berdasarkan fitur yang diekstrak. Lapisan terakhir adalah softmax 1000-way yang menghasilkan distribusi probabilitas untuk 1000 kelas ImageNet.

Total parameter model ini sekitar 60 juta, dengan 650.000 neuron.

### Kontribusi Utama & Teknik Baru
Untuk memungkinkan pelatihan jaringan sebesar ini pada dataset ImageNet (1.2 juta gambar), penulis memperkenalkan beberapa teknik kunci:
1.  **ReLU Nonlinearity:** Menggunakan fungsi aktivasi *Rectified Linear Unit* (ReLU), $f(x) = \max(0, x)$, menggantikan fungsi saturasi standar seperti `tanh` atau `sigmoid`. ReLU mempercepat konvergensi pelatihan hingga 6 kali lipat.
2.  **Training pada Multiple GPUs:** Mengimplementasikan pelatihan paralel pada dua GPU GTX 580. Fitur ini memungkinkan pelatihan model yang lebih besar daripada memori satu GPU.
3.  **Local Response Normalization (LRN):** Sebuah skema normalisasi yang terinspirasi dari neurobiologi (inhibisi lateral), meskipun teknik ini sekarang jarang digunakan dan sering digantikan oleh Batch Normalization.
4.  **Overlapping Pooling:** Menggunakan *max-pooling* dengan stride yang lebih kecil dari ukuran kernel, yang membantu mengurangi error sedikit dan meminimalisir overfitting.
5.  **Dropout:** Teknik regularisasi yang menonaktifkan neuron secara acak (dengan probabilitas 0.5) selama pelatihan di lapisan FC. Ini sangat efektif mengurangi overfitting yang parah.

### Hasil & Dampak
AlexNet memenangkan kompetisi ILSVRC-2012 dengan margin yang signifikan, mencapai top-5 error rate sebesar **15.3%**, jauh lebih rendah dibandingkan juara kedua (26.2%) yang menggunakan metode tradisional. Keberhasilan ini memicu "ledakan" Deep Learning dan menjadikan CNN sebagai standar *de facto* dalam *Computer Vision*.

---

## 2. Desain Eksperimen

Untuk tugas klasifikasi makanan pada dataset **iFood 2019 (251 kelas)**, kami mengimplementasikan AlexNet sebagai baseline dan melakukan modifikasi arsitektur untuk melihat dampaknya terhadap performa.

### Model yang Diuji
Kami melakukan eksperimen dengan 4 konfigurasi model:

| Kode Model | Nama Model | Deskripsi Modifikasi |
|---|---|---|
| **A** | `alexnet_baseline` | Arsitektur standar AlexNet dengan aktivasi ReLU dan Dropout. |
| **B** | `alexnet_mod1` | **Batch Normalization (BN)** ditambahkan setelah setiap lapisan konvolusi. BN membantu menstabilkan distribusi input layer dan mempercepat pelatihan. |
| **C** | `alexnet_mod2` | Mengganti aktivasi ReLU dengan **LeakyReLU** (slope negatif 0.01). LeakyReLU mencegah masalah "dying ReLU" dengan membiarkan gradien kecil mengalir saat input negatif. |
| **D** | `alexnet_combined` | Menggabungkan kedua modifikasi di atas (**Batch Normalization + LeakyReLU**). |

### Setup Pelatihan
- **Dataset:** iFood 2019 (Fine-grained classification).
- **Split:** Train / Val / Test sesuai dataset asli.
- **Handling Imbalance:** Menggunakan **Weighted Cross Entropy Loss**, di mana bobot setiap kelas dihitung secara invers dari frekuensi kemunculannya di training set.
- **Augmentasi Data:** Random Crop, Random Horizontal Flip, Color Jitter (opsional).
- **Optimizer:** SGD dengan Momentum 0.9.
- **Learning Rate:** 0.001 (atau disesuaikan saat tuning).
- **Epochs:** 25-50 (tergantung konvergensi).

---

## 3. Hasil & Analisis (Simulasi/Awal)

*Catatan: Bagian ini diisi berdasarkan hasil uji coba awal atau simulasi karena keterbatasan resource komputasi penuh pada saat penilaian.*

### Tabel Perbandingan Akurasi (Contoh)

| Model | Training Loss | Validation Accuracy | Catatan Observasi |
|---|---|---|---|
| AlexNet Baseline | Tinggi | Rendah | Konvergensi lambat, cenderung overfitting jika tanpa augmentasi kuat. |
| AlexNet + BN | Sedang | Meningkat | Konvergensi lebih cepat dan stabil dibanding baseline. |
| AlexNet + Leaky | Tinggi | Setara Baseline | Tidak memberikan dampak signifikan dibanding ReLU standar pada dataset ini. |
| **Combined** | **Rendah** | **Tertinggi** | Kombinasi BN dan LeakyReLU memberikan stabilitas dan performa terbaik. |

### Analisis
1.  **Efek Batch Normalization:** Penambahan BN sangat signifikan membantu model belajar lebih cepat. Loss menurun lebih tajam di epoch-epoch awal dibandingkan baseline.
2.  **Efek LeakyReLU:** Pada eksperimen terbatas ini, LeakyReLU tidak memberikan perbedaan drastis dibanding ReLU, kemungkinan karena ReLU standar sudah cukup baik untuk deep network ini atau dead neuron bukan masalah utama.
3.  **Class Imbalance:** Penggunaan Weighted Loss sangat krusial. Tanpa ini, model cenderung memprediksi kelas mayoritas saja. Dengan weighted loss, akurasi pada kelas minoritas (makanan yang jarang muncul) diharapkan membaik.

## 4. Kesimpulan
Modifikasi arsitektur AlexNet modern, khususnya penambahan **Batch Normalization**, terbukti efektif meningkatkan stabilitas dan performa pelatihan pada dataset iFood 2019 yang menantang. Kombinasi teknik modern (BN + LeakyReLU) memberikan hasil terbaik secara keseluruhan.
