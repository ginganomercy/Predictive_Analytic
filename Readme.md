# Laporan Proyek Machine Learning Terapan - Rafly Ashraffi Rachmat

## Domain Proyek: Kesehatan

Kanker merupakan penyakit akibat pertumbuhan sel abnormal yang tidak terkendali dan bisa menyebar ke jaringan tubuh lainnya. Kondisi ini menjadi penyebab kematian kedua tertinggi di dunia karena sering kali tidak menunjukkan gejala pada tahap awal. Oleh karena itu, deteksi dini sangat penting.

Dalam era modern, teknologi seperti *machine learning* digunakan untuk membantu diagnosis dini berbagai penyakit termasuk kanker. Proyek ini bertujuan untuk memanfaatkan *machine learning* dalam mendeteksi kemungkinan seseorang mengidap kanker berdasarkan data kesehatan pribadi. Dataset diambil dari [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) dengan 1500 entri dan 9 fitur.

## Business Understanding

### Problem Statements:

1. Apa saja faktor yang paling berpengaruh dalam prediksi risiko kanker?
2. Bagaimana mengetahui risiko kanker seseorang berdasarkan riwayat kesehatan dan gaya hidupnya?

### Goals:

1. Mengidentifikasi fitur yang paling berpengaruh dalam diagnosis kanker.
2. Menemukan model prediksi terbaik berdasarkan evaluasi performa.

### Solution Statement:

* Menggunakan 6 algoritma ML: XGBoost, Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Random Forest, dan CatBoost.
* Menentukan model terbaik dengan metrik evaluasi akurasi dan F1-score.

## Data Understanding

Dataset berisi informasi demografis dan gaya hidup:

* Jumlah baris: 1500, Kolom: 9
* Tidak terdapat data duplikat dan missing value.
* Tidak ditemukan outlier ekstrem pada fitur numerik.

![Gambar Deskripsi Dataset](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/data_info.png?raw=true)

### Variabel:

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

## Exploratory Data Analysis

### Univariate Analysis:

* Mayoritas responden: pria, tidak merokok, risiko genetik rendah, tidak punya riwayat kanker, tidak terdiagnosis kanker.
* Distribusi `Age`, `BMI`, `PhysicalActivity`, dan `AlcoholIntake` tersebar cukup merata.

![Barplot Kategorikal](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/barplot_kategorikal.png?raw=true)

### Multivariate Analysis:

* Perempuan lebih banyak terdiagnosis kanker dibanding pria.
  ![Perbandingan Gender](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/barplot_gender_diagnosis.png?raw=true)

* Kanker lebih umum pada usia >40 tahun.
  ![Perbandingan Usia](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/stripplot_usia.png?raw=true)

* Perokok lebih banyak ditemukan di kelompok penderita kanker.
  ![Perbandingan Merokok](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/histplot_smoking.png?raw=true)

* BMI pada penderita kanker sedikit lebih tinggi.
  ![Perbandingan BMI](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/boxplot_bmi.png?raw=true)

* Risiko genetik tinggi dominan pada kelompok dengan kanker.
  ![Perbandingan Genetik](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/histplot_geneticrisk.png?raw=true)

## Data Preparation

* Encoding kategori dilakukan menggunakan OrdinalEncoder dan OneHotEncoder.
* Pembagian data training dan testing: 70:30.
* Standarisasi dilakukan untuk fitur numerik menggunakan MinMaxScaler.
* Tidak perlu penanganan khusus untuk outlier karena distribusi data baik.

## Modeling

### Model yang digunakan:

1. **XGBoost**: Akurasi ≈ 85.33%
2. **Logistic Regression**: Akurasi ≈ 81.11%
3. **Decision Tree**: Akurasi ≈ 81.56%
4. **KNN**: Akurasi ≈ 68.22%
5. **Random Forest**: Akurasi ≈ 90.67%
6. **CatBoost**: Akurasi ≈ 92.44%

![Perbandingan Akurasi](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/perbandingan_akurasi.png?raw=true)

### Evaluasi:

* Metode evaluasi: Confusion Matrix, Accuracy, Precision, Recall, F1-Score.
* CatBoost memiliki F1-score tertinggi dan kesalahan klasifikasi terendah.

### Feature Importance (CatBoost):

Fitur yang paling berpengaruh: `Age`, `PhysicalActivity`, `BMI`, `AlcoholIntake`.

![Feature Importance](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/feature_importance.png?raw=true)

## Kesimpulan:

1. Faktor penting dalam risiko kanker meliputi usia, aktivitas fisik, BMI, dan konsumsi alkohol.
2. Model CatBoost dipilih sebagai model terbaik dengan akurasi tertinggi (92%) dan kesalahan klasifikasi yang rendah.

## Referensi:

1. [https://www.alodokter.com/penyakit-kanker](https://www.alodokter.com/penyakit-kanker)
2. [https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)
