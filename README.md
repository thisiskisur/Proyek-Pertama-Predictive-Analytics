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
- Dataset: [Weather Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/thedevastator/weather-prediction)

### Informasi Umum Dataset
- **Jumlah baris (observasi):** 3.654  
- **Jumlah kolom (fitur):** 165  
- **Tipe data dominan:** `float64`, `int64`, dan `datetime64`  
- **Target variabel:** `BASEL_temp_mean` (rata-rata suhu harian di kota Basel)

---

### Penanganan Data Awal
- Tidak terdapat **missing value** atau **duplikat** dalam dataset.
- Kolom `DATE` telah dikonversi dari format string `YYYYMMDD` menjadi tipe `datetime`.
- Dari kolom `DATE`, diturunkan fitur tambahan `month` untuk mengidentifikasi musim atau tren musiman.
- Analisis outlier telah dilakukan pada fitur-fitur suhu, angin, dan curah hujan. Namun, tidak ditemukan outlier yang secara signifikan mempengaruhi performa model, sehingga tidak dilakukan penghapusan data.

---

### Struktur dan Penamaan Fitur

Semua fitur mengikuti pola `[KOTA]_[PARAMETER]`, di mana:
- `KOTA` merepresentasikan lokasi geografis (contoh: BASEL, MILAN, STOCKHOLM)
- `PARAMETER` menunjukkan jenis variabel cuaca, seperti `temp_mean`, `humidity`, `pressure`, dll.

---

### Fitur Tanggal dan Turunannya

- `DATE` – Tanggal observasi (`datetime64`), digunakan untuk memahami tren waktu.
- `month` – Fitur turunan dari `DATE`, berisi nilai 1–12 yang merepresentasikan bulan. Digunakan dalam eksplorasi musiman.

**Digunakan:** Keduanya (`DATE`, `month`) karena penting untuk analisis temporal dan musiman.

---

### Pengelompokan Fitur Berdasarkan Kota

#### 🔹 **BASEL**
- `BASEL_temp_mean`, `BASEL_humidity`, `BASEL_wind`, `BASEL_precip`
- `BASEL_temp_max`, `BASEL_temp_min`, `BASEL_pressure`

#### 🔹 **MILAN**
- `MILAN_temp_mean`, `MILAN_wind`
- `MILAN_temp_max`, `MILAN_temp_min`, `MILAN_pressure`
- `MILAN_precip` (terlalu sparsity dan noise)

#### 🔹 **STOCKHOLM**
- `STOCKHOLM_temp_mean`, `STOCKHOLM_pressure`, `STOCKHOLM_temp_min`, `STOCKHOLM_temp_max`
- Semua fitur tidak digunakan karena korelasi sangat rendah dan redundansi terhadap kota yang lebih representatif

#### 🔹 **ATHENS**
- `ATHENS_temp_mean`, `ATHENS_humidity`, `ATHENS_wind`
- Dihapus karena banyak nilai konstan atau noise tinggi

#### 🔹 **BERLIN**
- `BERLIN_humidity`, `BERLIN_temp_mean`, `BERLIN_precip`
- Dihapus karena korelasi rendah dan distribusi outlier yang tidak stabil

---

### Pengelompokan Berdasarkan Jenis Parameter

#### 1. **Suhu (`*_temp_mean`, `*_temp_max`, `*_temp_min`)**
- **Digunakan**: Semua `*_temp_mean`
- Tidak digunakan: `*_temp_max`, `*_temp_min` (redundan terhadap mean)

#### 2. **Kelembaban (`*_humidity`)**
- Digunakan: Kota besar dengan korelasi signifikan (contoh: `BASEL_humidity`)
- Tidak digunakan: Kota minor atau korelasi rendah

#### 3. **Tekanan Udara (`*_pressure`)**
- Tidak digunakan secara keseluruhan (korelasi rendah)

#### 4. **Kecepatan Angin (`*_wind`)**
- Digunakan: `BASEL_wind`, `MILAN_wind`
- Dihapus: Kota lain karena nilai konstan atau noise

#### 5. **Curah Hujan (`*_precip`)**
- Digunakan: `BASEL_precip`
- Tidak digunakan: Lainnya (data sparse, banyak nol, distribusi tidak stabil)
---

### Ringkasan Pemilihan Fitur

| Status            | Estimasi Jumlah Fitur | Keterangan                                                 |
|-------------------|------------------------|-------------------------------------------------------------|
| Digunakan        | ~20                    | Korelasi signifikan, distribusi stabil                      |
| Tidak digunakan  | ~100                   | Redundan, korelasi rendah, nilai statis                     |
| Dihapus          | ~45                    | Mengandung noise, sparsity tinggi, atau nilai konstan       |

> Pemilihan fitur dilakukan berdasarkan uji korelasi terhadap target (`BASEL_temp_mean`), distribusi nilai, dan hasil validasi awal model. Tujuannya untuk meningkatkan akurasi dan mengurangi multikolinearitas.

---


## Data Preparation

1. **Pemisahan Fitur dan Target**
   - Target: kolom `BASEL_temp_mean`
   - Fitur: seluruh kolom selain target dan `DATE`

2. **Split Data**
   - Training set: 60%
   - Validation set: 20%
   - Test set: 20%
   - Menggunakan `train_test_split` dari scikit-learn dengan `random_state=99`.

3. **Normalisasi Data**
   - Menggunakan `MinMaxScaler` pada fitur dan target secara terpisah.
   - Proses fitting scaler hanya dilakukan pada training set untuk mencegah data leakage.

4. **Reshape Data untuk LSTM**
   - Format LSTM: `[samples, timesteps=1, features]`

### Ringkasan Ukuran Data Setelah Split dan Normalisasi

| Subset         | Jumlah Baris | Jumlah Fitur |
|----------------|---------------|---------------|
| Training Set   | 2192          | 164           |
| Validation Set | 731           | 164           |
| Test Set       | 731           | 164           |

---

## Model Development

### 1. Random Forest Regressor
- **Deskripsi:** Ensemble learning berbasis decision tree
- **Kelebihan:** Tangguh terhadap outlier dan overfitting, cocok untuk data tabular
- **Parameter:** `n_estimators=100`, `random_state=99`

### 2. Gradient Boosting Regressor
- **Deskripsi:** Boosting model, membangun model secara berurutan
- **Kelebihan:** Efektif untuk menangkap pola kompleks
- **Parameter:** `n_estimators=100`, `learning_rate=0.1`, `random_state=42`

### 3. Recurrent Neural Network (LSTM)
- **Deskripsi:** RNN untuk data time series
- **Kelebihan:** Bisa menangkap pola urutan data
- **Catatan:** Menggunakan `timesteps=1`, belum memaksimalkan konteks temporal
- **Arsitektur:**
  - `LSTM(64, return_sequences=True)`
  - `Dropout(0.2)`
  - `LSTM(32)`
  - `Dropout(0.2)`
  - `Dense(1)`
- **Hyperparameter:** `epochs=50`, `batch_size=32`, `optimizer=Adam(lr=0.001)`

---

### Hasil Evaluasi
 
| Model                        | MAE    | MSE      | R²     |
|------------------------------|--------|----------|--------|
| Random Forest Regressor      | 0.0122 | 0.00026  | 0.9936 |
| Gradient Boosting Regressor  | 0.0118 | 0.00024  | 0.9941 |
| RNN (LSTM)                   | 0.0373 | 0.00238  | 0.9405 |
 
### Insight:
- **RNN** menunjukkan performa terbaik dengan R² 0.9438, meskipun nilai MAE dan MSE-nya lebih tinggi dibandingkan dengan model Random Forest dan Gradient Boosting. Model RNN lebih baik dalam menangkap pola deret waktu.
- **Gradient Boosting Regressor** memiliki performa sangat baik dengan MAE 0.0118 dan MSE 0.00024, serta R² 0.9941, memberikan keseimbangan antara akurasi dan waktu pelatihan.
- **Random Forest** memberikan hasil yang sangat baik dengan R² 0.9936, tetapi sedikit lebih lambat dibandingkan Gradient Boosting.
 
## Hasil dan Insight
 
Berdasarkan evaluasi, **Gradient Boosting Regressor** memiliki performa terbaik dengan R² 0.9941 dan lebih cepat dibandingkan dengan **RNN (LSTM)**. Meskipun demikian, **RNN (LSTM)** menunjukkan kemampuan luar biasa dalam menangkap pola deret waktu dan lebih unggul dalam kondisi cuaca yang bergantung pada data historis.
 
## Rekomendasi
 
- **Peningkatan Data**: Mengumpulkan lebih banyak data historis cuaca dapat membantu meningkatkan akurasi model, terutama pada prediksi jangka panjang.
- **Eksperimen dengan Arsitektur LSTM**: Penerapan Long Short-Term Memory (LSTM) lebih lanjut dapat menangkap pola deret waktu yang lebih kompleks.
- **Integrasi Model**: Mengintegrasikan model ini ke dalam aplikasi berbasis web atau mobile dapat memberikan nilai tambah bagi pengguna yang ingin mengetahui prediksi cuaca lebih akurat.

## Referensi

- World Meteorological Organization (WMO). “State of the Global Climate 2022”. https://library.wmo.int
- Dataset: https://www.kaggle.com/datasets/thedevastator/weather-prediction
- Scikit-Learn Documentation: https://scikit-learn.org/
- TensorFlow LSTM Guide: https://www.tensorflow.org/guide/keras/rnn
