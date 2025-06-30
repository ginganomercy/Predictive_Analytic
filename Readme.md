# Laporan Proyek Machine Learning Terapan - Rafly Ashraffi Rachmat

## Domain Proyek: Kesehatan

Kanker merupakan penyakit akibat pertumbuhan sel abnormal yang tidak terkendali dan bisa menyebar ke jaringan tubuh lainnya. Kondisi ini menjadi penyebab kematian kedua tertinggi di dunia karena sering kali tidak menunjukkan gejala pada tahap awal. Oleh karena itu, deteksi dini sangat penting.

Teknologi *machine learning* telah terbukti mampu membantu diagnosis dini berbagai penyakit termasuk kanker. Proyek ini bertujuan untuk memanfaatkan *machine learning* dalam mendeteksi risiko kanker berdasarkan data kesehatan pribadi. Dataset digunakan dari [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) yang berisi 1500 entri dan 9 fitur.

---

## Business Understanding

### Problem Statements:

1. Apa saja faktor yang paling berpengaruh dalam prediksi risiko kanker?
2. Bagaimana mengetahui risiko kanker seseorang berdasarkan riwayat kesehatan dan gaya hidupnya?

### Goals:

1. Mengidentifikasi fitur yang paling berpengaruh dalam diagnosis kanker.
2. Menemukan model prediksi terbaik berdasarkan evaluasi performa.

### Solution Statement:

Proyek ini akan membandingkan performa enam algoritma *machine learning*:

* XGBoost
* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)
* Random Forest
* CatBoost

Model akan dievaluasi menggunakan metrik: Accuracy, Precision, Recall, dan F1-Score. Model terbaik dipilih berdasarkan kinerja metrik yang paling seimbang.

---

## Data Understanding

### Deskripsi Dataset:

* Jumlah entri: 1500
* Jumlah fitur: 9 (termasuk target)
* Sumber: [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)
* Tidak ditemukan missing value maupun duplikasi
* Outlier ekstrem tidak ditemukan pada distribusi fitur numerik

![Gambar Dataset](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/data_info.png?raw=true)

### Daftar Fitur:

| Nama             | Deskripsi                                        |
| ---------------- | ------------------------------------------------ |
| Age              | Usia                                             |
| Gender           | Jenis kelamin (0: wanita, 1: pria)               |
| BMI              | Indeks massa tubuh                               |
| Smoking          | Kebiasaan merokok (0: tidak, 1: ya)              |
| GeneticRisk      | Risiko genetik (0: rendah, 1: sedang, 2: tinggi) |
| PhysicalActivity | Aktivitas fisik harian                           |
| AlcoholIntake    | Konsumsi alkohol                                 |
| CancerHistory    | Riwayat keluarga terkena kanker                  |
| Diagnosis        | Target (0: tidak kanker, 1: kanker)              |

---

## Exploratory Data Analysis (EDA)

### Univariate Analysis:

* Mayoritas responden adalah pria, tidak merokok, risiko genetik rendah, dan tidak memiliki riwayat kanker.
* Fitur numerik (`Age`, `BMI`, `PhysicalActivity`, `AlcoholIntake`) memiliki distribusi yang merata.

![Distribusi Kategori](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/boxplot1.png?raw=true)

### Multivariate Analysis:

* Perempuan lebih banyak terdiagnosis kanker dibanding pria.
  ![Per Gender](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/jumlahdiagnosisberdasarkankelamin.png?raw=true)
* Risiko kanker meningkat pada usia di atas 40 tahun.
  ![Per Age](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/plotstripdiagnosis.png?raw=true)
* Penderita kanker cenderung perokok.
  ![Smoking](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/plotbarbertumpuk.png?raw=true)
* BMI sedikit lebih tinggi pada penderita kanker.
  ![BMI](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/distribusi_bmi.png?raw=true)
* Risiko genetik tinggi dominan pada kelompok kanker.
  ![Genetik](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/plotbarbertumpukdiagnosiskanker.png?raw=true)

---

## Data Preparation

### Tahapan:

1. **Penghapusan Duplikat**: Dicek dan tidak ditemukan data duplikat.
2. **Missing Value**: Tidak ditemukan.
3. **Encoding Fitur Kategorikal**:

   * `Gender`, `Smoking`, `CancerHistory` menggunakan *OrdinalEncoder*.
   * `GeneticRisk` menggunakan *OneHotEncoder* karena memiliki 3 kategori nominal.
4. **Scaling Fitur Numerik**:

   * Digunakan **MinMaxScaler** untuk mengubah skala data ke rentang \[0, 1] (**normalisasi**, bukan standarisasi).
   * Fitur: `Age`, `BMI`, `PhysicalActivity`, `AlcoholIntake`.
5. **Pembagian Dataset**:

   * Split data ke *training* dan *testing* dengan rasio 70:30 secara *stratified*.
6. **Penanganan Outlier**:

   * Tidak dilakukan karena distribusi tidak menunjukkan ekstrem.

---

## Model Development

### Algoritma dan Cara Kerja Singkat:

1. **Logistic Regression**: Model linier yang memetakan probabilitas hasil biner.
2. **Decision Tree**: Pemisahan data berdasarkan fitur paling informatif.
3. **K-Nearest Neighbors (KNN)**: Klasifikasi berdasarkan kedekatan fitur ke tetangga terdekat.
4. **Random Forest**: Ensembling dari banyak decision tree untuk hasil stabil.
5. **XGBoost**: Model boosting berbasis pohon yang dioptimasi untuk kecepatan dan akurasi.
6. **CatBoost**: Boosting model khusus yang bagus untuk fitur kategorikal dan menangani overfitting.

### Hyperparameter Tuning (Jika Ada):

Dilakukan tuning sederhana (default vs manual) khusus untuk:

* `n_estimators`, `max_depth` (Random Forest)
* `learning_rate`, `iterations` (CatBoost)
* `n_neighbors` (KNN)

---

## Evaluation

### Metrik yang Digunakan:

* **Accuracy**: Proporsi prediksi benar
* **Precision**: Proporsi positif yang benar-benar positif
* **Recall**: Seberapa banyak kasus positif terdeteksi
* **F1-Score**: Harmonik dari Precision dan Recall

### Hasil Evaluasi Model:

| Model               | Accuracy   | Precision | Recall   | F1-Score |
| ------------------- | ---------- | --------- | -------- | -------- |
| Logistic Regression | 81.11%     | 0.80      | 0.81     | 0.80     |
| Decision Tree       | 81.56%     | 0.81      | 0.82     | 0.81     |
| KNN                 | 68.22%     | 0.67      | 0.69     | 0.68     |
| Random Forest       | 90.67%     | 0.91      | 0.90     | 0.90     |
| XGBoost             | 85.33%     | 0.85      | 0.85     | 0.85     |
| **CatBoost**        | **92.44%** | **0.92**  | **0.92** | **0.92** |

![Perbandingan Akurasi](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/perbandinganakurasi.png?raw=true)

### Feature Importance (CatBoost):

Fitur penting dalam diagnosis:

* `Age`
* `PhysicalActivity`
* `BMI`
* `AlcoholIntake`

![Feature Importance](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/feature_importance.png?raw=true)

### Keterkaitan dengan Business Understanding:

* Model **CatBoost** dengan akurasi dan F1 tertinggi dapat digunakan untuk membantu skrining awal pasien berisiko.
* Dapat digunakan sebagai sistem pendukung keputusan medis untuk mendeteksi kemungkinan kanker.
* Solusi menjawab semua *problem statement* dan mencapai *goals* yang ditetapkan.

---

## Kesimpulan:

1. Faktor penting dalam risiko kanker meliputi usia, aktivitas fisik, BMI, dan konsumsi alkohol.
2. Model CatBoost dipilih sebagai model terbaik dengan akurasi 92% dan F1-Score tertinggi.
3. Model memiliki potensi signifikan dalam membantu deteksi dini kanker berbasis data pasien.

---

## Referensi:

1. [Alodokter - Penyakit Kanker](https://www.alodokter.com/penyakit-kanker)
2. [Kaggle - Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)
