# Laporan Proyek Machine Learning - Rizky Surya Alfarizy

## Judul: Prediksi Cuaca Berdasarkan Data Historis

---

## Domain Proyek

Cuaca merupakan faktor penting yang mempengaruhi berbagai sektor kehidupan, seperti pertanian, transportasi, pariwisata, dan mitigasi bencana. Di Indonesia, ketidakakuratan prakiraan cuaca seringkali menimbulkan kerugian besar, seperti gagal panen dan gangguan pengiriman logistik.

**Pihak terdampak langsung:**
- Petani dan nelayan
- Perusahaan logistik
- Lembaga kebencanaan (BMKG, BNPB)
- Masyarakat umum

Menurut World Meteorological Organization (WMO), intensitas cuaca ekstrem meningkat dua kali lipat dalam 20 tahun terakhir. Oleh karena itu, sistem prediksi cuaca yang lebih akurat berbasis data dan machine learning sangat diperlukan.

---

## Business Understanding

### Problem Statements
- Prakiraan cuaca saat ini belum cukup akurat dan presisi untuk wilayah lokal.
- Model berbasis fisik (numerical) memerlukan sumber daya besar.

### Goals
- Membangun model prediksi suhu harian berbasis data historis.
- Membandingkan performa model: **Random Forest**, **Gradient Boosting**, dan **LSTM**.
- Menentukan model terbaik berdasarkan metrik evaluasi.

### Solution Statements
- Melatih dan mengevaluasi tiga model: Random Forest Regressor, Gradient Boosting Regressor, dan LSTM.
- Menggunakan metrik: MAE, MSE, dan R².
- Menerapkan normalisasi dan reshaping untuk LSTM.

---

## Data Understanding

  ### Sumber Data
  - [Weather Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/thedevastator/weather-prediction)

  ### Informasi Dataset
- **Jumlah baris:** 3.654 
- **Jumlah kolom/fitur:** 165  
- **Tipe data:** `float64`, `int64`  
- **Kolom tanggal:** `DATE` (format awal `YYYYMMDD`, telah dikonversi ke `datetime`)  
- **Target prediksi:** `BASEL_temp_mean` (suhu rata-rata harian di Basel)

  ### Penjelasan Fitur
  Dataset ini terdiri dari 165 fitur meteorologi yang merepresentasikan berbagai parameter cuaca harian di sejumlah kota di Eropa. Setiap fitur mengandung informasi spesifik seperti suhu, kelembaban, kecepatan angin, dan lainnya.
  Penjelasan fitur mencakup:
   1. **Fitur yang digunakan dalam proses pelatihan model**, misalnya:
    - `BASEL_temp_mean`: Suhu rata-rata harian di Basel.
    - `STOCKHOLM_humidity`: Kelembaban di Stockholm.
    - `ATHENS_wind`: Kecepatan angin di Athena.
    - dan lainnya.
   2. **Fitur yang tidak digunakan**, seperti:
    - Fitur dengan korelasi rendah terhadap target.
    - Fitur yang redundan atau memiliki informasi serupa.
   3. **Fitur yang akan dihapus**, jika:
    - Mengandung noise yang tinggi.
    - Data tidak konsisten atau tidak relevan untuk proses prediksi.
    
    Total terdapat **165 fitur meteorologi** dari berbagai kota di Eropa yang telah ditinjau agar penulis dan pembaca dapat memahami karakteristik data secara menyeluruh sebelum melangkah ke tahap analisis lebih lanjut.

  ### Pengecekan dan Praproses Data
    - **Missing Value:** Tidak ditemukan (`isnull().sum() = 0`)  
    - **Data Duplikat:** Tidak ditemukan (`duplicated().sum() = 0`)  
    - **Format Tanggal:** Kolom `DATE` telah dikonversi ke format `datetime`  
    - **Outlier:** Akan dianalisis dan ditangani lebih lanjut pada tahap eksplorasi model apabila memengaruhi performa model secara signifikan.

---

## Data Preparation

Pada tahap ini dilakukan serangkaian proses persiapan data sebelum pelatihan model. Semua langkah disusun berdasarkan urutan aktual dalam notebook untuk menjaga transparansi dan kemudahan reproduksi eksperimen.

---

### 1. Cek Dimensi dan Struktur Dataset

- Dataset terdiri dari **3.654 baris** dan **165 kolom**.
- Sebagian besar kolom bertipe `float64`, sisanya `int64`.
- Tidak ditemukan nilai yang hilang (`missing values`) dalam dataset.

---

### 2. Normalisasi Fitur

- Dilakukan **normalisasi pada seluruh fitur numerik** menggunakan **Min-Max Scaling**.
- Teknik ini bertujuan menyamakan skala antar fitur, penting untuk model seperti LSTM.
- Digunakan `MinMaxScaler` dari `sklearn.preprocessing`.
- Formula:
![alt text](?https://github.com/thisiskisur/Proyek-Pertama-Predictive-Analytics/blob/main/Screenshot%202025-05-25%20195405.pngraw=true)

- Output normalisasi disimpan sebagai `weather_scaled`.

---

### 3. Pemisahan Fitur dan Target

- Target yang akan diprediksi adalah: **`BASEL_temp_mean`**.
- Fitur (`features`) diambil dari seluruh kolom **kecuali** `BASEL_temp_mean`.
- Pemisahan dilakukan setelah normalisasi agar target juga dalam skala yang sama.

---

### 4. Split Data

- Data dibagi menjadi tiga bagian:
  - **Training set (60%)**: 2.192 data
  - **Validation set (20%)**: 731 data
  - **Test set (20%)**: 731 data
- Pembagian menggunakan `train_test_split()` dari `sklearn.model_selection`.
- Parameter `random_state=99` digunakan untuk memastikan hasil yang konsisten.

---

### 5. Reshape untuk Model LSTM

- Model LSTM memerlukan input dalam format tiga dimensi: **`[samples, timesteps, features]`**.
- Data `X_train`, `X_val`, dan `X_test` diubah bentuk (reshape) dengan `timesteps = 1`.

  Contoh transformasi:
  - Sebelum: `(2192, 164)`
  - Sesudah: `(2192, 1, 164)`

---

Dengan tahapan ini, data telah siap untuk digunakan dalam pelatihan model baik baseline (Random Forest) maupun model LSTM.


## Model Development

Pada tahap ini dilakukan pemodelan menggunakan tiga algoritma regresi yang berbeda untuk membandingkan performa dalam memprediksi nilai suhu rata-rata (`BASEL_temp_mean`) berdasarkan berbagai fitur cuaca.

### 1. Random Forest Regressor

**Deskripsi:**
Random Forest adalah algoritma **ensemble learning** berbasis **decision tree**. Model ini membangun banyak pohon keputusan dan menghasilkan prediksi dengan cara **mengambil rata-rata dari hasil prediksi tiap pohon** (regresi). Model ini dikenal tangguh terhadap overfitting dan sangat efektif untuk dataset tabular dengan fitur numerik dan kategorikal.

**Kesesuaian dengan Data:**
- Dataset cuaca memiliki banyak fitur (high-dimensional), dan Random Forest mampu mengatasi hal ini dengan baik.
- Tidak memerlukan fitur berskala sama atau distribusi normal.
- Dapat menangani hubungan non-linear antar fitur.

**Parameter:**
- `n_estimators=100` (jumlah pohon)
- `random_state=99` (untuk reprodusibilitas)

---

### 2. Gradient Boosting Regressor

**Deskripsi:**
Gradient Boosting adalah teknik **boosting** yang membangun model secara **berurutan**, di mana setiap model baru berusaha **memperbaiki kesalahan dari model sebelumnya**. Model ini efektif untuk menangkap pola kompleks dalam data dengan cara menambahkan model lemah (decision stump atau shallow trees) secara iteratif.

**Kesesuaian dengan Data:**
- Mampu mempelajari hubungan non-linear antar variabel cuaca.
- Cenderung menghasilkan performa tinggi pada data tabular berskala kecil hingga menengah seperti dataset ini.
- Cocok untuk regresi dengan target numerik kontinyu.

**Parameter:**
- `n_estimators=100`
- `learning_rate=0.1`
- `random_state=42`

---

### 3. Recurrent Neural Network (LSTM)

**Deskripsi:**
Long Short-Term Memory (LSTM) merupakan salah satu arsitektur dari Recurrent Neural Network (RNN) yang dirancang untuk menangani data **berurutan atau deret waktu**. LSTM memiliki kemampuan menyimpan informasi dalam jangka waktu panjang menggunakan **gated memory cells** sehingga cocok untuk data time series.

**Kesesuaian dengan Data:**
- Dataset cuaca merupakan data time-series harian.
- LSTM memungkinkan model untuk memahami pola suhu dari hari ke hari berdasarkan konteks historis.
- Namun, pada implementasi ini, hanya digunakan satu timestep (per hari), sehingga model tidak sepenuhnya memanfaatkan kekuatan LSTM.

**Arsitektur:**
- `LSTM(64, return_sequences=True)`
- `Dropout(0.2)`
- `LSTM(32)`
- `Dropout(0.2)`
- `Dense(1)`

**Hyperparameter:**
- Epochs: 50
- Batch size: 32
- Optimizer: Adam (`lr=0.001`)
- Fungsi aktivasi: `tanh` dan `linear` pada output

---

## Evaluation

### Metrik Evaluasi
- **MAE (Mean Absolute Error)**: Mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual.
- **MSE (Mean Squared Error)**: Penalti kuadrat terhadap error, lebih sensitif terhadap outlier.
- **R² Score**: Mengukur proporsi variansi target yang bisa dijelaskan oleh fitur.

### Hasil Evaluasi

| Model                        | MAE     | MSE       | R²      |
|-----------------------------|---------|-----------|---------|
| Random Forest Regressor     | 0.0122  | 0.00026   | 0.9936  |
| Gradient Boosting Regressor | 0.0118  | 0.00024   | 0.9941  |
| RNN (LSTM)                  | 0.0365  | 0.00225   | 0.9438  |

### Insight:
- **Gradient Boosting Regressor** menunjukkan performa terbaik di semua metrik.
- **Random Forest Regressor** juga sangat akurat dan stabil.
- **LSTM** kurang optimal karena tidak memanfaatkan urutan waktu secara eksplisit (hanya 1 timestep digunakan), padahal kekuatan LSTM justru terletak pada urutan data historis yang lebih panjang.

---

## Conclusion

Model **Gradient Boosting Regressor** dipilih sebagai model terbaik karena:

- Memiliki **MAE dan MSE terendah** dibanding model lain.
- **R² tertinggi** (0.9941), menunjukkan bahwa model sangat baik dalam menjelaskan variansi data target.
- Relatif lebih cepat dilatih dibanding deep learning.
- Stabil dan konsisten dalam performa terhadap data tabular numerik.

Sementara itu, model **LSTM** tetap memiliki potensi kuat untuk prediksi jangka panjang atau sekuens data dengan struktur waktu yang lebih kompleks, terutama jika digunakan dengan **multiple timesteps**.

---

## Referensi

- World Meteorological Organization (WMO). “State of the Global Climate 2022”. https://library.wmo.int
- Dataset: https://www.kaggle.com/datasets/thedevastator/weather-prediction
- Scikit-Learn Documentation: https://scikit-learn.org/
- TensorFlow LSTM Guide: https://www.tensorflow.org/guide/keras/rnn
