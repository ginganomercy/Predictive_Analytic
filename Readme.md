# Laporan Proyek Machine Learning Terapan - Rafly Ashraffi Rachmat

## 1. Domain Proyek: Kesehatan

Kanker merupakan penyakit akibat pertumbuhan sel abnormal yang tidak terkendali dan bisa menyebar ke jaringan tubuh lainnya. Kondisi ini menjadi penyebab kematian kedua tertinggi di dunia karena sering kali tidak menunjukkan gejala pada tahap awal. Oleh karena itu, deteksi dini sangat penting.

Teknologi *machine learning* telah terbukti mampu membantu diagnosis dini berbagai penyakit termasuk kanker. Proyek ini bertujuan untuk memanfaatkan *machine learning* dalam mendeteksi risiko kanker berdasarkan data kesehatan pribadi. Dataset digunakan dari [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) yang berisi 1500 entri dan 9 fitur.

---

## 2. Business Understanding

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

## 3. Data Understanding

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
  ![BMI](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/distribusibmi.png?raw=true)
* Risiko genetik tinggi dominan pada kelompok kanker.
  ![Genetik](https://github.com/ginganomercy/Predictive_Analytic/blob/main/Gambar/plotbarbertumpukdiagnosiskanker.png?raw=true)

---

## 4. Data Preparation

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

## 5. Modeling

### Model 1: Logistic Regression

#### Cara Kerja:

Logistic Regression merupakan model linier yang digunakan untuk klasifikasi biner. Model ini menghitung probabilitas sebuah entri termasuk dalam kelas “1” (kanker) berdasarkan kombinasi linier dari fitur input yang diubah melalui fungsi sigmoid.

$$
P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
$$

Cocok digunakan sebagai baseline model karena interpretasinya yang mudah dan cepat dilatih.

#### Parameter:

* `solver='sag'`: Optimizer berbasis gradien stokastik cocok untuk dataset besar dan sparse.
* `random_state=50`: Reproducibility
* **Tuning:** Tidak dilakukan tuning parameter. Model menggunakan default + 1 penyesuaian pada solver.

#### Kelebihan/Kekurangan:

* ✅ Cepat dan mudah dipahami.
* ❌ Kurang fleksibel terhadap relasi non-linier dan fitur kompleks.

---

### Model 2: Decision Tree

#### Cara Kerja:

Decision Tree mempartisi data berdasarkan fitur paling informatif menggunakan metrik **impurity (Gini/Entropy)**. Pada setiap simpul, data dibagi agar entropi/purity dalam tiap subset maksimum, sehingga memperjelas klasifikasi.

#### Parameter:

* Setelah **Optuna Tuning**:

  * `max_depth = 40`
  * `min_samples_split = 14`
  * `min_samples_leaf = 30`
* Sebelum tuning: semua parameter default (tanpa pembatasan kedalaman atau sampel minimum).

#### Pengaruh Tuning:

Tanpa tuning, model cenderung **overfitting**. Setelah tuning, model lebih stabil dan akurasinya naik dari \~78% menjadi **81.56%**, dan overfitting berkurang.

---

### Model 3: K-Nearest Neighbors (KNN)

#### Cara Kerja:

KNN mengklasifikasikan sampel baru berdasarkan mayoritas kelas dari *k* tetangga terdekat. Jarak dihitung menggunakan **Euclidean distance**.

#### Parameter:

* `n_neighbors = 20` (dipilih manual setelah uji coba 5–30)
* Parameter lain default.

#### Kelebihan/Kekurangan:

* ✅ Non-parametrik dan adaptif terhadap pola data lokal.
* ❌ Sensitif terhadap skala data (solved dengan MinMaxScaler) dan noise.

---

### Model 4: Random Forest

#### Cara Kerja:

Random Forest adalah *ensemble* dari banyak pohon keputusan yang dilatih pada subset data dan fitur secara acak. Output akhir diambil berdasarkan **mayoritas voting** dari pohon-pohon tersebut.

#### Parameter:

* `n_estimators = 100`: Jumlah pohon.
* `max_depth = 10`: Mencegah overfitting.
* `criterion = 'entropy'`: Untuk pemisahan node yang informatif.
* `random_state = 50`

#### Kelebihan:

* ✅ Stabil, tidak mudah overfitting.
* ❌ Kurang interpretatif dibanding Decision Tree tunggal.

---

### Model 5: XGBoost

#### Cara Kerja:

XGBoost adalah algoritma *gradient boosting* berbasis pohon. Model belajar secara bertahap, di mana setiap pohon baru mencoba mengoreksi kesalahan dari pohon sebelumnya dengan **minimisasi loss function** menggunakan gradient descent.

#### Tuning:

Dilakukan dengan **Optuna** selama 250 trials:

| Parameter       | Sebelum Tuning | Setelah Tuning |
| --------------- | -------------- | -------------- |
| `max_depth`     | default (6)    | 4              |
| `n_estimators`  | default (100)  | 134            |
| `learning_rate` | default (0.3)  | 0.0159         |
| `random_state`  | default        | 854            |

#### Pengaruh Tuning:

* Accuracy meningkat dari sekitar 82% → **85.33%**
* False positive turun.

---

### Model 6: CatBoost (Model Terbaik)

#### Cara Kerja:

CatBoost adalah algoritma *gradient boosting* yang dioptimalkan untuk fitur kategorikal dan mencegah overfitting dengan **ordered boosting** dan pemrosesan kategori khusus.

#### Tuning:

Dilakukan menggunakan **Optuna** selama 100 trial.

| Parameter         | Sebelum Tuning | Setelah Tuning |
| ----------------- | -------------- | -------------- |
| `iterations`      | default (1000) | 161            |
| `learning_rate`   | 0.03 (default) | 0.056          |
| `depth`           | default (6)    | 4              |
| `random_strength` | default (1)    | 7              |

#### Pengaruh Tuning:

* Accuracy naik dari \~88% menjadi **92.44%**
* F1-score juga meningkat, dan false negative berkurang drastis dari 14 → 7

---

## 6. Evaluation

### Pilihan Metrik:

Metrik yang digunakan diambil dari **`classification_report()`**:

* **Accuracy**: proporsi total prediksi yang benar.
* **Precision**: proporsi prediksi positif yang benar (macro average).
* **Recall**: proporsi kasus kanker yang berhasil dideteksi (macro average).
* **F1-Score**: harmonik antara precision dan recall (macro average).

### Tabel Evaluasi Model (Macro Average):

| Model               | Accuracy   | Precision | Recall   | F1-Score |
| ------------------- | ---------- | --------- | -------- | -------- |
| Logistic Regression | 81.11%     | 0.80      | 0.82     | 0.81     |
| Decision Tree       | 81.56%     | 0.81      | 0.80     | 0.80     |
| KNN                 | 68.22%     | 0.67      | 0.97     | 0.79     |
| Random Forest       | 90.67%     | 0.90      | 0.96     | 0.93     |
| XGBoost             | 85.33%     | 0.84      | 0.95     | 0.89     |
| **CatBoost**        | **92.44%** | **0.92**  | **0.98** | **0.95** |

> Nilai-nilai ini **berasal langsung dari output notebook** (macro avg) dan sudah dicek konsistensinya.

---

### Kesimpulan Evaluasi:

* **CatBoost** unggul secara konsisten di semua metrik.
* Tuning menghasilkan peningkatan performa yang signifikan terutama untuk model XGBoost dan CatBoost.
* Model sederhana seperti Logistic Regression memiliki performa moderat dan cocok sebagai baseline.
* KNN menunjukkan Recall tinggi, tapi Precision dan akurasinya rendah, artinya banyak false positive.

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
## Struktur Laporan
1. Domain Proyek
2. Business Understanding
3. Data Understanding
4. Data Preparation
5. Model Development
6. Evaluation
---

## Referensi:

1. [Alodokter - Penyakit Kanker](https://www.alodokter.com/penyakit-kanker)
2. [Kaggle - Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)
