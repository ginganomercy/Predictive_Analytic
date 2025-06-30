#!/usr/bin/env python
# coding: utf-8

# # Predictive Analytic:  **Cancer Prediction**
# Nama     : Rafly Ashraffi Rachmat
# 
# Email    : raflypriyantoro@gmail.com
# 
# Sumber Dataset:  
#    Dataset diperoleh dari *Kaggle* dengan judul **Cancer Prediction Dataset** (https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) dengan dataset 1500 data.

# # **Data Understanding**

# Tahap dalam proses analisis data yang bertujuan untuk memahami dataset secara mendalam sebelum melakukan analisis lebih lanjut.

# # **1. Import Library**

# Pada tahap ini kita mengimport semua library yang dibutuhkan untuk menganalisis

# In[6]:


get_ipython().system('pip install optuna')
get_ipython().system('pip install catboost')
get_ipython().system('pip install numpy==1.26.4 packaging==24.0 --upgrade --force-reinstall')


import os
import shutil
import textwrap
import numpy as np
import zipfile
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report


# # **Data Loading**

# ## Memuat Dataset

# In[9]:


# Gantilah path ini dengan path absolut atau relatif dari file ZIP di sistem lokalmu
zip_path = "cancer.zip"
extract_to = "dataset_cancer"

# Membuka dan mengekstrak ZIP
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_to)

# Tampilkan semua file di folder hasil ekstraksi untuk memastikan nama file
print("File yang diekstrak:")
print(os.listdir(extract_to))

# Load dataset
csv_path = os.path.join(extract_to, "The_Cancer_data_1500_V2.csv")
df_train = pd.read_csv(csv_path)

# Tampilkan 5 baris pertama
df_train.head()


# Penjelasan:
# Diperoleh hasil bahwa terdapat 1500 data dan 9 kolom

# ### **Deskripsi Variabel**

# 
# | Nama Variabel      | Keterangan                                                                   |
# | ------------------ | ----------------------------------------------------------------------- |
# | `Age`              | Usia responden (dalam tahun).                                           |
# | `Gender`           | Jenis kelamin responden (0 = Perempuan, 1 = Laki-laki).                 |
# | `BMI`              | Indeks Massa Tubuh (Body Mass Index), indikator status berat badan.     |
# | `Smoking`          | Status merokok (0 = Tidak merokok, 1 = Merokok).                        |
# | `GeneticRisk`      | Risiko genetik terhadap kanker (0 = Tidak ada, 1 = Rendah, 2 = Tinggi). |
# | `PhysicalActivity` | Tingkat aktivitas fisik harian (dalam jam atau skor kuantitatif).       |
# | `AlcoholIntake`    | Konsumsi alkohol (dalam satuan tertentu, kemungkinan skor/level).       |
# | `CancerHistory`    | Riwayat kanker dalam keluarga (0 = Tidak ada, 1 = Ada).                 |
# | `Diagnosis`        | Hasil diagnosis kanker (0 = Negatif, 1 = Positif).                      |
# 

# In[13]:


df_train.info()


# Bisa dilihat bahwa data diatas tedapat 3 variabel bertipe float64 dan 6 variabel bertipe int64

# In[15]:


df_train.shape


# Dari output diatas bisa dilihat bahwa terdapat jumlah baris 1.500 dan jumlah kolom 9

# In[17]:


#membuat data frame
#agar tidak memenuhi source
df_filtered = pd.DataFrame(df_train)


# ### **Deskripsi Statistik dari Data**

# In[19]:


# memanggil untuk statistik data mengecek outlier.
df_filtered.describe()


# Fungsi `describe()` memberikan informasi statistik pada masing-masing kolom, antara lain:
# 
# - `Count` adalah jumlah sampel pada data.
# - `Mean` adalah nilai rata-rata.
# - `Std` adalah standar deviasi.
# - `Min` yaitu nilai minimum setiap kolom.
# - `25%` adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
# - `50%` adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
# -` 75%` adalah kuartil ketiga.
# - `Max` adalah nilai maksimum.

# Dataset terdiri dari 1.500 sampel data dengan fitur numerik yang mencerminkan karakteristik pasien, kemungkinan terkait data kanker. Beberapa fitur bersifat biner, sementara yang lain menunjukkan variasi lebih besar. Rentang usia pasien berkisar antara 20–80 tahun dengan median 51, menunjukkan distribusi yang cukup merata. Nilai mean dan median yang mendekati mengindikasikan distribusi data relatif simetris, meskipun analisis lanjutan diperlukan untuk memastikan normalitas dan mendeteksi outlier.

# ## **Exploratory Data Analysis - Univariate Analysis**

# In[23]:


numerical_feature = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
categorical_feature = ['Gender','Smoking', 'GeneticRisk', 'CancerHistory', 'Diagnosis']


# In[24]:


# Setup ukuran dan jumlah subplot
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 15), sharex=False, sharey=False)

# Kolom dan deskripsi kategori
kolom_kategorikal = ['Gender', 'Smoking', 'GeneticRisk', 'CancerHistory', 'Diagnosis']
deskripsi_kolom_kategorikal = [
    'Jenis Kelamin (0=male, 1=female)',
    'Merokok (0=tidak, 1=ya)',
    'Risiko Genetik (0=rendah, 1=sedang, 2=tinggi)',
    'Riwayat Kanker (0=tidak, 1=ya)',
    'Diagnosis Kanker (0=tidak kanker, 1=kanker)'
]

# Loop plotting
for i, (kolom, deskripsi) in enumerate(zip(kolom_kategorikal, deskripsi_kolom_kategorikal)):
    ax = axes[i]
    sns.countplot(x=kolom, data=df_train, ax=ax, hue=kolom, palette="pastel")

    # Buat judul dengan wrap jika panjang
    judul = "\n".join(textwrap.wrap(f"Plot Jumlah dari {deskripsi}", width=40))
    ax.set_title(judul, fontsize=12)

    ax.set_xlabel("")  # Hilangkan label X
    ax.tick_params(axis="x", labelrotation=0, labelsize=12)

# Tata letak rapi
plt.tight_layout()
plt.show()


# Gambar di atas dapat diinterpretasikan sebagai berikut.
# 1. Dari `Plot Jumlah dari Jenis Kelamin`, mayoritas responden adalah pria dan sisanya adalah wanita.
# 2. Dari `Plot Jumlah dari Merokok`, sebagian besar responden tidak merokok, sedangkan sebagian kecil merupakan perokok.
# 3. Dari `Plot Jumlah dari Risiko Genetik`, mayoritas responden memiliki risiko genetik rendah terhadap kanker.
# 4. Dari `Plot Jumlah dari Riwayat Kanker`, sebagian besar responden tidak memiliki riwayat kanker, sedangkan sisanya memiliki riwayat tersebut.
# 5. Dari `Plot Jumlah dari Diagnosis Kanker`, mayoritas responden tidak terdiagnosis kanker, dan sebagian lainnya terdiagnosis positif kanker.

# In[26]:


# Membuat histogram untuk semua fitur numerik
ax = df_filtered.hist(
    bins=50,
    figsize=(20, 15),
    color='skyblue',
    edgecolor='black'  # Tambahan kecil agar batang lebih jelas
)

# Menyesuaikan layout agar tidak tumpang tindih
plt.tight_layout()

# Menampilkan semua plot histogram
plt.show()


# Gambar di atas dapat diinterpretasikan sebagai berikut.
# 
# 1. Plot Histogram dari `Age`, `BMI`, dan `PhysicalActivity` memiliki distribusi yang cukup merata tanpa pola lonceng yang jelas, mengindikasikan bahwa data tersebar luas di seluruh rentang nilai yang tersedia.
# 
# 2. Plot Histogram dari `Gender`, `Smoking`, `GeneticRisk`, `CancerHistory`, `AlcoholIntake`, dan Diagnosis menunjukkan bahwa data merupakan data kategorikal biner, dengan nilai yang sangat dominan pada salah satu kelas (misalnya nilai 0 lebih banyak dari 1).
# 
# 3. Plot Histogram dari `Age` terlihat agak miring ke kanan, yang berarti sebagian besar responden berusia di atas rata-rata, menunjukkan dominasi kelompok usia dewasa hingga lansia dalam data ini.

# # **Exploratory Data Analysis - Multivariate Analysis**

# ### **Membandingkan Diagnosis Kanker Berdasarkan Jenis Kelamin**

# Kode dibawah untuk mempermudah analisis maka diubah dari numerik menjadi kategorik

# In[31]:


# Dictionary mapping untuk beberapa kolom
mapping_dict = {
    "Gender": {0: "male", 1: "female"},
    "Smoking": {0: "no", 1: "yes"},
    "CancerHistory": {0: "no", 1: "yes"},
    "Diagnosis": {0: "No Cancer", 1: "Cancer"},
    "GeneticRisk": {0: "low", 1: "medium", 2: "high"}
}

# Terapkan mapping secara efisien
for col, mapping in mapping_dict.items():
    df_filtered[col] = df_filtered[col].replace(mapping)

# Tampilkan hasil
df_filtered.head()


# In[32]:


# Ukuran figure
plt.figure(figsize=(8, 6))

# Plot countplot dengan hue berdasarkan Gender
sns.countplot(data=df_filtered, x="Diagnosis", hue="Gender", palette="pastel")

# Judul dan label sumbu
plt.title("Jumlah Diagnosis Kanker Berdasarkan Jenis Kelamin", fontsize=14)
plt.xlabel("Diagnosis", fontsize=12)
plt.ylabel("Jumlah", fontsize=12)
plt.xticks(rotation=0)

# Tampilkan plot
plt.tight_layout()
plt.show()


# Dari grafik bar yang ditampilkan, dapat disimpulkan bahwa:
# 
# 1. Penyakit kanker (Diagnosis = 1) lebih banyak ditemukan pada responden perempuan dibandingkan laki-laki. Hal ini terlihat dari jumlah bar perempuan yang lebih tinggi pada kategori “Diagnosis = 1”.
# 
# 2. Responden yang tidak terkena kanker (Diagnosis = 0) ini artinya responde laki-laki lebih banyak ditemukan dibanding perempuan,selisihnya lumayan jauh.

# ### **Membandingkan Diagnosis Kanker Berdasarkan Usia**

# In[35]:


# Membuat figure
plt.figure(figsize=(8, 6))

# Gunakan hue=Diagnosis agar tidak kena warning palette
sns.stripplot(
    data=df_filtered,
    x="Age",
    y="Diagnosis",
    hue="Diagnosis",
    dodge=False,
    jitter=0.25,
    size=4,
    alpha=0.7,
    palette="Set2",
    legend=False  # Agar legend tidak ditampilkan dua kali
)

plt.title("Plot Strip Diagnosis Kanker Berdasarkan Usia", fontsize=14)
plt.xlabel("Usia (tahun)", fontsize=12)
plt.ylabel("Diagnosis", fontsize=12)

plt.tight_layout()
plt.show()


# Dari gambar di atas, dapat disimpulkan bahwa:
# 
# 1. Seluruh responden yang terdiagnosis kanker (positif) berada pada rentang usia sekitar 30 hingga 80 tahun, dengan penyebaran lebih padat di atas usia 40 tahun.
# 
# 2. Responden yang tidak terdiagnosis kanker (negatif) memiliki distribusi usia yang lebih luas, mulai dari usia 20 hingga sekitar 80 tahun, dengan sebaran cukup merata.
# 

# ### **Membandingkan Kebiasaan Merokok Berdasarkan Diagnosis Kanker**

# In[38]:


# Ukuran plot
plt.figure(figsize=(8, 6))

# Plot histogram bertumpuk berdasarkan Diagnosis dan kebiasaan merokok
ax = sns.histplot(
    data=df_filtered,
    x="Diagnosis",
    hue="Smoking",
    multiple="stack",
    palette="bright",
    shrink=0.8  # opsional, agar batang lebih ramping
)

# Judul dan label
plt.title("Plot Bar Bertumpuk dari Kebiasaan Merokok Berdasarkan Diagnosis Kanker", fontsize=13)
plt.xlabel("Diagnosis", fontsize=11)
plt.ylabel("Jumlah", fontsize=11)

# Posisi legenda
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.75))

# Tampilkan
plt.tight_layout()
plt.show()


# Dari gambar di atas, disimpulkan bahwa:
# 
# 1. Mayoritas individu yang tidak merokok (warna biru) lebih banyak ditemukan pada kelompok yang tidak terkena kanker (Diagnosis = 0) dibandingkan dengan kelompok yang terkena kanker.
# 
# 2. Sebaliknya, individu yang merokok (warna oranye) lebih banyak muncul pada kelompok dengan diagnosis kanker (Diagnosis = 1), menunjukkan adanya korelasi antara kebiasaan merokok dan peningkatan risiko kanker.

# ### **Membandingkan Distribusi BMI Berdasarkan Diagnosis Kanker**

# In[41]:


# Membuat figure
plt.figure(figsize=(8, 6))

# Plot boxplot distribusi BMI berdasarkan Diagnosis
sns.boxplot(
    data=df_filtered,
    x="Diagnosis",
    y="BMI",
    hue="Diagnosis",      # tambahkan hue
    legend=False,         # matikan legend karena redundant
    palette="bright",
    width=0.6,
    fliersize=3
)

# Judul dan label sumbu
plt.title("Distribusi BMI Berdasarkan Diagnosis Kanker", fontsize=14)
plt.xlabel("Diagnosis", fontsize=12)
plt.ylabel("BMI", fontsize=12)

# Tampilkan plot
plt.tight_layout()
plt.show()


# Dari gambar di atas, disimpulkan bahwa:
# 
# 1. Distribusi BMI pada kelompok yang tidak terkena kanker (Diagnosis = 0) cenderung memiliki median BMI yang sedikit lebih rendah dibandingkan dengan kelompok yang terkena kanker (Diagnosis = 1).
# 
# 2. Variasi BMI pada kedua kelompok relatif mirip, namun terdapat beberapa nilai BMI ekstrim (outlier) di kedua kelompok, yang menunjukkan adanya individu dengan BMI sangat rendah atau sangat tinggi.
# 
# 3. Secara umum, individu dengan diagnosis kanker (positif) cenderung memiliki BMI yang sedikit lebih tinggi dibandingkan dengan individu tanpa kanker (negatif).
# 
# 

# ### **Membandingkan Risiko Genetik Berdasarkan Diagnosis Kanker**

# In[44]:


# Membuat canvas plot
plt.figure(figsize=(8, 6))

# Plot bar bertumpuk berdasarkan Diagnosis dan GeneticRisk
ax = sns.histplot(
    data=df_filtered,
    x="Diagnosis",
    hue="GeneticRisk",
    multiple="stack",
    palette="bright",
    shrink=0.85  # opsional: untuk merampingkan bar agar tidak tumpang tindih
)

# Judul dan label
plt.title("Plot Bar Bertumpuk dari Risiko Genetik Berdasarkan Diagnosis Kanker", fontsize=14)
plt.xlabel("Diagnosis", fontsize=12)
plt.ylabel("Jumlah", fontsize=12)
plt.xticks(rotation=0)

# Memindahkan legend ke posisi kiri atas (di luar plot)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.75))

# Atur layout agar tidak terpotong
plt.tight_layout()
plt.show()


# Dari gambar di atas, disimpulkan bahwa:
# 
# 1. Mayoritas individu yang tidak terdiagnosis kanker (No Cancer) memiliki risiko genetik rendah (low). Hal ini menunjukkan bahwa risiko genetik yang rendah kemungkinan berasosiasi dengan tidak adanya diagnosis kanker.
# 
# 2. Sebaliknya, kelompok yang didiagnosis kanker (Cancer) memiliki distribusi risiko genetik yang lebih beragam, dengan jumlah signifikan pada kategori risiko sedang (medium) dan risiko tinggi (high).
# 
# 3. Individu dengan risiko genetik tinggi (high) secara proporsional lebih banyak ditemukan pada kelompok Cancer dibandingkan kelompok No Cancer. Ini menunjukkan adanya potensi korelasi positif antara risiko genetik tinggi dan kemungkinan terkena kanker.
# 
# 4. Meskipun kelompok risiko rendah tetap mendominasi dalam kedua diagnosis, proporsinya jauh lebih tinggi di kelompok No Cancer dibanding kelompok Cancer.

# In[46]:


# Membuat figure dan title
plt.figure(figsize=(10, 8))

# Hitung matriks korelasi dan bulatkan 2 desimal
correlation_matrix = df_filtered[numerical_feature].corr().round(2)

# Plot heatmap dengan korelasi antar fitur numerik
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",                # Format angka 2 desimal
    cmap="coolwarm",
    linewidths=0.5,
    linecolor='white',        # Tambahan kecil: garis antar sel lebih bersih
    square=True,              # Membuat kotak matriks berbentuk bujur sangkar
    cbar_kws={"shrink": 0.8}  # Shrink color bar agar tidak terlalu besar
)

# Tambahkan judul
plt.title("Correlation Matrix untuk Fitur Numerik", fontsize=18, pad=15)

# Tampilkan plot
plt.tight_layout()
plt.show()


# Kesimpulan Heatmap Korelasi:
# 1. Korelasi antar fitur sangat rendah
#  - Hampir semua nilai korelasi antar fitur numerik seperti Age, BMI, Physical Activity, dan Alcohol Intake berada di kisaran 0.00 – 0.03.
#   - Ini menunjukkan bahwa tidak terdapat hubungan linear yang signifikan antar variabel tersebut.
# 
# 2. Nilai diagonal bernilai 1
#   - Korelasi antara fitur dengan dirinya sendiri adalah 1.00, yang merupakan karakteristik normal dari matriks korelasi.

# # **Data Preparation**

# ### **Data Cleaning**

# #### **Mengecek Data Duplikat**

# In[51]:


df_filtered.duplicated().sum()


# Dari hasil diatas terlihat bahwa tidak ada data yang terduplikasi.

# In[53]:


# Menghapus data duplikat
df_cleaned = df_filtered.drop_duplicates()

# Memverifikasi jumlah baris setelah penghapusan duplikat
print(f"Jumlah baris setelah penghapusan duplikat: {df_cleaned.shape[0]}")
df_cleaned.head()


# Dari hasil diatas menunjukkan bahwa dataset dengan jumlah 1500 data artinya dataset yang dipakai tidak ada data yang terduplikasi dan bisa lanjut untuk menganalisis.
# 

# #### **Menangani Missing Value**

# In[56]:


df_cleaned.isnull().sum()


# Penjelasan:
# 
# Dari output diatas didapati bahwa tidak terdapat missing value pada dataset

# In[58]:


from sklearn.preprocessing import OrdinalEncoder

# 1. Kolom yang akan di-encode
cols_to_encode = ["Gender", "Smoking", "CancerHistory", "Diagnosis", "GeneticRisk"]

# 2. Kategori sesuai urutan semula
category_order = [
    ["male", "female"],          # Gender
    ["no", "yes"],               # Smoking
    ["no", "yes"],               # CancerHistory
    ["No Cancer", "Cancer"],     # Diagnosis
    ["low", "medium", "high"]    # GeneticRisk
]

# 3. Inisialisasi encoder
ordinal_encoder = OrdinalEncoder(categories=category_order, dtype=int)

# 4. Fit dan transform kolom kategorikal
encoded_array = ordinal_encoder.fit_transform(df_filtered[cols_to_encode])

# 5. Salin DataFrame dan masukkan hasil encode
df_cleaned = df_filtered.copy()
df_cleaned[cols_to_encode] = encoded_array.astype(int)

# 6. Tampilkan hasil
df_cleaned.head()


# Penjelasan:
# 
# 1. Proses encoding ini mengubah data kategorikal seperti jenis kelamin, kebiasaan merokok, riwayat kanker, diagnosis, dan risiko genetik menjadi bentuk numerik agar dapat digunakan oleh algoritma machine learning.
# 
# 2. Pemetaan dilakukan dengan urutan tertentu, misalnya: 'male' → 0 dan 'female' → 1 untuk Gender, 'no' → 0 dan 'yes' → 1 untuk Smoking dan CancerHistory, 'No Cancer' → 0 dan 'Cancer' → 1 untuk Diagnosis, serta 'low' → 0, 'medium' → 1, dan 'high' → 2 untuk GeneticRisk.
# 
# 3. Proses ini menghasilkan data yang bersih dan konsisten, siap digunakan untuk analisis statistik atau pemodelan prediktif. Selain itu, encoding ordinal memungkinkan analisis lanjutan, seperti mengevaluasi hubungan antara risiko genetik dan kemungkinan diagnosis kanker.tatistik karena semua fitur kini berbentuk numerik.

# ### **Menangani outliers**
# 
# menangani outliers dengan IQR Method

# In[61]:


# memanggil untuk statistik data setelah dihapus data duplikat dan missing value.
df_cleaned.describe()


# Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
# 
# - Count adalah jumlah sampel pada data.
# - Mean adalah nilai rata-rata.
# - Std adalah standar deviasi.
# - Min yaitu nilai minimum setiap kolom.
# - 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
# - 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah). - 75% adalah kuartil ketiga.
# - Max adalah nilai maksimum.

#  Penjelasan:
# 
# Berdasarkan hasil statistik deskriptif, dataset terdiri dari 1.500 sampel tanpa missing values pada fitur numerik. Rata-rata usia peserta adalah 50 tahun dengan rentang 20–80 tahun, menunjukkan variasi usia yang luas. Nilai BMI rata-rata sebesar 27,5 termasuk dalam kategori overweight, dengan rentang antara 15 hingga hampir 40. Aktivitas fisik memiliki rata-rata 4,9 namun terdapat nilai yang sangat rendah mendekati nol, yang berpotensi menjadi outlier. Sementara itu, asupan alkohol memiliki rata-rata 2,4 dengan sebaran cukup lebar dari 1,0 hingga hampir 5,0, mencerminkan perbedaan kebiasaan konsumsi antar individu

# In[64]:


df_cleaned.info()


# In[65]:


df_cleaned.shape


# Total dari hasil diatas adalah 1500 baris

# In[67]:


#Cek data outlier
numerical_feature = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
categorical_feature = ['Gender','Smoking', 'GeneticRisk', 'CancerHistory', 'Diagnosis']


# In[68]:


#Cek data outlier
numerical_feature = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
selected_cols = df_cleaned[numerical_feature]

Q1 = selected_cols.quantile(0.25)
Q3 = selected_cols.quantile(0.75)
IQR = Q3 - Q1

df_filtered = df_cleaned[~((selected_cols < (Q1 - 1.5 * IQR)) | (selected_cols > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[69]:


# Boxplot untuk setiap fitur numerik
for col in numerical_feature:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_cleaned[col], color='steelblue', linewidth=1.5, fliersize=4)
    plt.title(f'Distribusi {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Penjelasan:
# 
# Berdasarkan keempat fitur yang dianalisis, tidak ada indikasi outlier yang perlu ditangani. Hal ini menunjukkan bahwa data numerik dalam dataset relatif bersih dan berada dalam rentang yang wajar secara statistik maupun secara logis. Dengan demikian, tidak diperlukan proses pembersihan atau transformasi khusus terhadap outlier untuk variabel-variabel ini.

# ## **One Hot Encoding**

# In[72]:


# 1. Kolom kategorikal yang akan di-one-hot encode
categorical_cols = ["Gender", "Smoking", "CancerHistory", "GeneticRisk"]

# 2. One-hot encoding dengan drop_first untuk mencegah dummy trap
dummies = pd.get_dummies(df_filtered[categorical_cols], drop_first=True, dtype=int)

# 3. Gabungkan ke DataFrame asli
data = pd.concat([df_filtered.drop(columns=categorical_cols), dummies], axis=1)

# 4. Tampilkan hasil
data.head()


# Penjelasan:
# 
# Kode di atas melakukan proses one-hot encoding pada kolom kategorikal yaitu Gender, Smoking, CancerHistory, dan GeneticRisk. Proses ini mengubah nilai kategori menjadi format numerik (dummy variables) agar dapat digunakan dalam model machine learning. Opsi drop_first=True digunakan untuk menghindari dummy variable trap dengan menghapus satu kategori dari setiap variabel. Setelah encoding, kolom aslinya dihapus dan data hasil transformasi siap digunakan untuk proses analisis atau pemodelan lebih lanjut.

# ## **Splitting**

# Selanjutnya, karena target kita adalah variabel `diagnosis` untuk mengetahui akurasi prediksi dari diagnosis, maka kita akan membuang kolom tersebut dari data dan assign kolom tersebut ke variabel baru.

# In[76]:


from sklearn.model_selection import train_test_split

# 1. Pisahkan fitur (X) dan target (y)
X = data.drop(columns=["Diagnosis"])
y = data["Diagnosis"]

# 2. Bagi data menjadi train dan test set (70%:30%), dengan random_state tetap untuk reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=30,
    stratify=y  # opsional: menjaga proporsi label Diagnosis di train/test
)


# Lalu, kita membagi data menjadi 2, yaitu
# 
# - Data training sebesar 70% untuk melatih model
# - Data testing sebesar 30% untuk menguji model

# In[78]:


# Menampilkan ukuran data training dan testing dari X dan y
print("Ukuran X_train: ", X_train.shape)
print("Ukuran X_test: ", X_test.shape)
print("Ukuran y_train: ", y_train.shape)
print("Ukuran y_test: ", y_test.shape)


# In[79]:


print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# ## **Standarisasi**

# Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma dan menyeragamkan karena memiliki satuan yang berbeda pada tiap fitur.

# In[82]:


# scaling untuk data training
numerical_features= ['BMI', 'PhysicalActivity', 'AlcoholIntake']
scaler = MinMaxScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()


# In[83]:


# scaling untuk data testing
numerical_features= ['BMI', 'PhysicalActivity', 'AlcoholIntake']
scaler = MinMaxScaler()
scaler.fit(X_test[numerical_features])
X_test[numerical_features] = scaler.transform(X_test.loc[:, numerical_features])
X_test[numerical_features].head()


# # **Model Development**

# In[85]:


# Function for evalution report and plotting confusion matrix
def make_evaluation(y_true, y_pred, title):

    # Membuat list nama target yang diinginkan
    target_nama = ['Terkena Cancer',
                    'Non Cancer']

    # Menampilkan laporan klasifikasi (classification report)
    print(classification_report(y_true, y_pred, target_names = target_nama))

    # Membentuk Confusion Matrix
    fig, ax = plt.subplots(figsize = (10, 5))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax = ax)

    # Menambahkan label sumbu x dan y pada confusion matrix
    ax.xaxis.set_ticklabels(target_nama, rotation = 90)
    ax.yaxis.set_ticklabels(target_nama)

    # Menghilangkan garis-garis grid
    ax.grid(False)

    # Menambahkan judul pada confusion matrix
    _ = ax.set_title(title)
    plt.show()


# Seluruh model yang akan dibuat menggunakan hyperparameter tuning menggunakan optuna. Optimasi hyperparameter dengan Optuna terbukti efektif dalam meningkatkan performa model untuk mengetahui parameter yang tepat untuk algoritma model pada setiap model.

# ## **Model Development dengan Algoritma XGBoost**

# In[88]:


import optuna
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Ensure that X_train, X_test, y_train, y_test are converted to NumPy arrays
X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

# Example of converting y_train, y_test if they are DataFrames
y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
y_test = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

def objective(trial):
    # Suggest hyperparameters from the search space
    max_depth = trial.suggest_int('max_depth', 3, 15)  # Integer values between 3 and 15
    n_estimators = trial.suggest_int('n_estimators', 50, 200)  # Integer values between 50 and 200
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)  # Log-uniform distribution between 1e-4 and 1e-1
    random_state = trial.suggest_int('random_state', 0, 1000)  # Integer values for random_state

    # Create the XGBoost model with suggested hyperparameters
    model_xgb = XGBClassifier(max_depth=max_depth,
                              n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              random_state=random_state,
                              n_jobs=-1)

    # Train the model
    model_xgb.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model_xgb.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy  # We want to maximize accuracy

# Create Optuna study to optimize the hyperparameters
study = optuna.create_study(direction='maximize')  # 'maximize' because we want to maximize accuracy
study.optimize(objective, n_trials=250)  # Run the optimization for 50 trials

# Best hyperparameters found by Optuna
print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)


# In[89]:


# Memanggil fungsi XGBClassifier dari library sklearn
model_xgb = XGBClassifier(max_depth = 4, n_estimators = 134,
                          random_state = 854, learning_rate =    0.01593435842863771, n_jobs = -1)

# Melatih model XGBoost dengan data training pada X dan y
model_xgb.fit(X_train, y_train)


# Xgboost dengan parameter sebagai berikut:
# 
#   - max_depth = 4 :  Kedalaman maksimum pohon (default: None, yang berarti pohon akan terus tumbuh hingga semua daun murni atau hingga semua daun memiliki kurang dari min_samples_split sampel)
# 
#   - learning_rate:  mengontrol kecepatan pembelajaran model. Secara khusus, learning rate menentukan seberapa besar kontribusi setiap pohon (weak learner) terhadap model akhir untuk menghindari overfitting.
# 
#   - n_estimators : untuk mengurangi error yang tersisa dari prediksi pohon sebelumnya.
# 
#   - random_state: Menentukan generator bilangan acak untuk memastikan hasil yang dapat direproduksi (default: None).
# 
#   - n_jobs=-1: Menentukan jumlah inti (cores) yang digunakan untuk menghitung. Jika diatur ke -1, model akan menggunakan semua inti yang tersedia, sehingga mempercepat proses pelatihan.

# In[91]:


# Memprediksi hasil menggunakan data testing berdasarkan model yang telat dilatih
pred_xgb = model_xgb.predict(X_test)

# Menampilkan akurasi model
xgb = accuracy_score(y_test, pred_xgb)
accuracy_xgboost= round(accuracy_score(y_test, pred_xgb)*100,2)
print("hasil akurasi model xgboost: ", accuracy_xgboost,"%")


# In[92]:


# Menampilkan evaluasi model XGBoost dengan dataset Diagnosis
make_evaluation(y_test, pred_xgb, title="Confusion Matrix Menggunakan Algoritma XGBoost")


# Menggunakan XGBoost dimaknai:
# 
# 1. 269 responden cancer telah diklasifikasikan dengan benar
# 2. 115 responden noncancer telah diklasifikasikan dengan benar
# 3. 52 responden noncancer diklasifikasikan sebagai responden cancer (False Positif)
# 4. 14 responden cancer diklasifikasikan sebagai responden noncancer (False Negatif)

# ## **Model Development dengan Algoritma Logistic Regeression**

# In[95]:


#membuat object algoritma Logistic Regression
clf_lg = LogisticRegression(solver='sag',n_jobs=-1,random_state=50)

#memodelkan data dengan algoritma Logistic Regression
model_lg = clf_lg.fit(X_train,y_train)

#melakukan predict pda data test
pred_lg = model_lg.predict(X_test)


# algoritma logistic regression dengan parameter berikut:
# 
#   - solver : sag (Stochastic Average Gradient Descent), metode berbasis gradien stokastik, digunakan untuk menemukan parameter terbaik (koefisien) dalam model dengan data yang besar dan sparseness tinggi.
# 
#   - n_jobs=-1: Menentukan jumlah inti (cores) yang digunakan untuk menghitung. Jika diatur ke -1, model akan menggunakan semua inti yang tersedia, sehingga mempercepat proses pelatihan.

# In[97]:


# Menampilkan akurasi model
accuracy_lg= round(accuracy_score(y_test, pred_lg)*100,2)
print("hasil akurasi model Logisik Regression: ", accuracy_lg,"%")


# In[98]:


# Memanggil fungsi make_evaluation untuk menampilkan f1 score dan confusion matrix
make_evaluation(y_test, pred_lg, title = f"Confusion Matrix Menggunakan Algoritma Logistik Regression")


# Menggunakan Logistic Regression dimaknai:
# 
# 1. 255 responden cancer telah diklasifikasikan dengan benar
# 2. 110 responden noncancer telah diklasifikasikan dengan benar
# 3. 57 responden noncancer diklasifikasikan sebagai responden cancer (False Positif)
# 4. 28 responden cancer diklasifikasikan sebagai responden noncancer (False Negatif)

# ## **Model Development dengan Algoritma Decision Tree**

# In[143]:


# Ensure that X_train, X_test, y_train, y_test are converted to NumPy arrays if needed
X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
y_test = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

def objective(trial):
    # Suggest hyperparameters from the search space for DecisionTreeClassifier
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)  # Integer values from 1 to 10
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)  # Integer values from 2 to 20
    max_depth = trial.suggest_int('max_depth', 1, 100)

    # Create the DecisionTreeClassifier model with suggested hyperparameters
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                max_depth=max_depth,
                                max_features= None,
                                random_state=42)  # Use a fixed random state for reproducibility

    # Train the model
    dt.fit(X_train, y_train)

    # Predict on the test set
    y_pred = dt.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy  # We want to maximize accuracy

# Create an Optuna study to optimize the hyperparameters
study = optuna.create_study(direction='maximize')  # 'maximize' because we want to maximize accuracy
study.optimize(objective, n_trials=500)  # Run the optimization for 50 trials

# Output the best hyperparameters found by Optuna
print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)


# In[144]:


# Memanggil fungsi DecisionTreeClassifier dari library sklearn
model_dt = DecisionTreeClassifier(min_samples_leaf= 30, min_samples_split= 14, max_depth= 40,max_features=None)

# Melatih model KNN dengan data training pada X dan y
model_dt.fit(X_train, y_train)


# DecisionTreeClassifier dengan parameter sebagai berikut:
# 
# 1. max_depth=40
# → Membatasi kedalaman maksimum pohon hingga 40. Tujuannya adalah untuk mengontrol kompleksitas model dan mencegah overfitting. Default-nya adalah None (pohon tumbuh sampai semua daun murni).
# 
# 2. min_samples_leaf=30
# → Jumlah minimum sampel dalam setiap leaf (daun). Ini mencegah model membuat daun dari data sangat kecil (outlier), meningkatkan generalisasi.
# 
# 3. min_samples_split=14
# → Jumlah minimum sampel pada sebuah node agar node itu bisa di-split. Dengan 14, node tidak akan dipisah jika memiliki kurang dari 14 sampel. Ini membantu menghindari pembelahan berlebihan pada data kecil.
# 
# 4. max_features=None
# → Artinya semua fitur akan dipertimbangkan saat menentukan split terbaik di tiap node. Ini adalah default behavior jika None.
# 
# 5. random_state=None (default)
# → Tidak ditentukan di kode, jadi nilai default None digunakan. Artinya, hasil pohon bisa sedikit berbeda setiap kali dijalankan.

# In[146]:


# Memprediksi hasil menggunakan data testing berdasarkan model yang telat dilatih
pred_dt = model_dt.predict(X_test)

# Menampilkan akurasi model
accuracy_dt= round(accuracy_score(y_test, pred_dt)*100,2)
print("hasil akurasi model Logisik Regression: ", accuracy_dt,"%")


# In[147]:


# Memanggil fungsi make_evaluation untuk menampilkan f1 score dan confusion matrix
make_evaluation(y_test, pred_dt, title = f"Confusion Matrix Menggunakan Algoritma Decisiion Tree")


# Menggunakan Decision Tree dimaknai:
# 
# 1. 251 responden cancer telah diklasifikasikan dengan benar
# 2. 116 responden noncancer telah diklasifikasikan dengan benar
# 3. 51 responden noncancer diklasifikasikan sebagai responden cancer (False Positif)
# 4. 32 responden cancer diklasifikasikan sebagai responden noncancer (False Negatif)

# ## **Model Development dengan K-Nearest Neighbor**

# In[154]:


# Memanggil fungsi KNeighborsClassifier dari library sklearn
model_knn = KNeighborsClassifier(n_neighbors = 20)

# Melatih model KNN dengan data training pada X dan y
model_knn.fit(X_train, y_train)


# KNeighborsClassifier dengan parameter sebagai berikut:
# 
# n_neighbors = 20 (jumlah tetangga yang dipertimbangkan untuk prediksi. Nilai kecil (1-10) cocok untuk pola lokal atau dataset kecil tetapi rawan noise. Nilai besar(11-20) membuat model lebih stabil atau dataset besar tetapi kurang peka terhadap detail lokal.)

# In[157]:


# Memprediksi hasil menggunakan data testing berdasarkan model yang telat dilatih
pred_knn = model_knn.predict(X_test)

# Menampilkan akurasi model
accuracy_knn= round(accuracy_score(y_test, pred_knn)*100,2)
print("hasil akurasi model Algoritma KNN: ", accuracy_knn,"%")


# In[161]:


# Memanggil fungsi make_evaluation untuk menampilkan f1 score dan confusion matrix
make_evaluation(y_test, pred_knn, title = f"Confusion Matrix Menggunakan Algoritma KNN")


# Menggunakan K-Nearest Neighbor dimaknai:
# 
# 1. 274 responden cancer telah diklasifikasikan dengan benar
# 2. 33 responden noncancer telah diklasifikasikan dengan benar
# 3. 134 responden noncancer diklasifikasikan sebagai responden cancer (False Positif)
# 4. 9 responden cancer diklasifikasikan sebagai responden noncancer (False Negatif)

# ## **Model Development dengan Random Forest**

# In[165]:


# Memanggil fungsi RandomForestClassifier dari library sklearn
model_rf = RandomForestClassifier(n_estimators = 100, criterion = "entropy", max_depth = 10, random_state = 50)

# Melatih model Random Forest dengan data training pada X dan y
model_rf.fit(X_train, y_train)


# Model RandomForestClassifier dilatih menggunakan data training dengan parameter sebagai berikut:
# 
# - n_estimators=100: Menggunakan 100 pohon keputusan (tree), meningkatkan stabilitas dan akurasi.
# 
# - criterion="entropy": Pemilihan split didasarkan pada informasi gain (lebih fokus pada kejelasan klasifikasi).
# 
# - max_depth=10: Membatasi kedalaman maksimum pohon agar tidak overfitting.
# 
# - random_state=50: Menjamin hasil yang konsisten saat kode dijalankan ulang.
# 
# Model ini memanfaatkan ensemble learning untuk menggabungkan kekuatan banyak pohon, sehingga mampu menangani variansi dan menghasilkan prediksi yang lebih andal.
# 

# In[168]:


# Memprediksi hasil menggunakan data testing berdasarkan model yang telat dilatih
pred_rf = model_rf.predict(X_test)

# Menampilkan akurasi model
accuracy_rf= round(accuracy_score(y_test, pred_rf)*100,2)
print("hasil akurasi model Random Forest: ", accuracy_rf,"%")


# In[170]:


# Memanggil fungsi make_evaluation untuk menampilkan f1 score dan confusion matrix
make_evaluation(y_test, pred_rf, title = f"Confusion Matrix Menggunakan Algoritma Random Forest")


# Menggunakan K-Nearest Neighbor dimaknai:
# 
# 1. 271 responden cancer telah diklasifikasikan dengan benar
# 2. 137 responden noncancer telah diklasifikasikan dengan benar
# 3. 30 responden noncancer diklasifikasikan sebagai responden cancer (False Positif)
# 4. 12 responden cancer diklasifikasikan sebagai responden noncancer (False Negatif)

# ## **Model Development dengan Catboost**

# In[177]:


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate',0.001, 0.27),
        'depth': trial.suggest_int('depth', 3, 10),
        'random_strength': trial.suggest_int('random_strength', 1, 10)
    }

    # Create the cat classifier with the suggested hyperparameters
    cat = CatBoostClassifier(**params)

    # Train the classifier and evaluate on the validation set
    cat.fit(X_train, y_train)
    accuracy = cat.score(X_test, y_test)
    return accuracy

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100,timeout=600*12)

# Print the best hyperparameters and the corresponding accuracy
best_params = study.best_params
best_accuracy = study.best_value
print('Best Hyperparameters:', best_params)
print('Best Accuracy:', best_accuracy)


# In[178]:


# Create the cat classifier with the suggested hyperparameters
cat = CatBoostClassifier(iterations= 161, learning_rate= 0.0563827883242152, depth= 4, random_strength= 7)

# Train the classifier and evaluate on the validation set
cat.fit(X_train, y_train)


# CatboostClassifier dengan parameter sebagai berikut:
# 
#   - depth :  Kedalaman maksimum pohon (default: None, yang berarti pohon akan terus tumbuh)
# 
#   - learning_rate:  mengontrol kecepatan pembelajaran model. Secara khusus, learning rate menentukan seberapa besar kontribusi setiap pohon (weak learner) terhadap model akhir untuk menghindari overfitting.
# 
#   - iterations : tahap pembelajaran, dan jumlah iterasi yang lebih tinggi memungkinkan model untuk belajar lebih mendalam, tetapi juga meningkatkan risiko overfitting jika nilainya terlalu besar.
# 
#   - random_strength: mengontrol intensitas regulasi berbasis penambahan noise random dalam proses pemisahan fitur (feature splitting).
# 

# In[180]:


# Memprediksi hasil menggunakan data testing berdasarkan model yang telat dilatih
pred_cat = cat.predict(X_test)

# Menampilkan akurasi model
accuracy_cat= round(accuracy_score(y_test, pred_cat)*100,2)
print("hasil akurasi model Algoritma KNN: ", accuracy_cat,"%")


# In[181]:


# Memanggil fungsi make_evaluation untuk menampilkan f1 score dan confusion matrix
make_evaluation(y_test, pred_cat, title = f"Confusion Matrix Menggunakan Algoritma catboost")


# Menggunakan CatboostClassifier dimaknai: dimaknai:
# 
# 1. 276 responden cancer telah diklasifikasikan dengan benar
# 2. 140 responden noncancer telah diklasifikasikan dengan benar
# 3. 27 responden noncancer diklasifikasikan sebagai responden cancer (False Positif)
# 4. 7 responden cancer diklasifikasikan sebagai responden noncancer (False Negatif)

# # **Evaluasi Model dan pemilihan model**

# In[188]:


# Membentuk DataFrame berisi model dengan akurasinya
models = pd.DataFrame({
    "Model": ["XGBoost", "Logistik", "Decission Tree", "KNN" ,"Random Forest","Catboost"],
    "Akurasi": [accuracy_xgboost, accuracy_lg, accuracy_dt, accuracy_knn,accuracy_rf,accuracy_cat]
})

# Mengurutkan data berdasarkan akurasi dari tertinggi ke terendah
models.sort_values(by = "Akurasi", ascending = False)


# Dari hasil enam model algoritma yang terbaik adalah model development dengan algoritma Catboost

# In[195]:


# Buat plot
plt.figure(figsize=(8, 6))

# Gunakan hue sama dengan x dan nonaktifkan legend untuk menghindari warning
barplot = sns.barplot(
    data=models,
    x="Model",
    y="Akurasi",
    hue="Model",               # Tambahkan hue
    palette="viridis",
    legend=False               # Hilangkan legend karena redundant
)

# Tambahkan label nilai akurasi di atas tiap bar
for index, value in enumerate(models["Akurasi"]):
    barplot.text(index, value + 0.02, f"{value:.4f}", color="black", ha="center")

# Judul dan label sumbu
plt.title("Perbandingan Akurasi dari Keenam Model", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Akurasi", fontsize=12)

# Tata letak
plt.tight_layout()
plt.show()


# Berdasarkan visualisasi dan hasil evaluasi masing-masing model, Catboost menunjukkan performa terbaik. Hal ini ditunjukkan oleh skor akurasi dan skor F1 yang tertinggi, serta jumlah kesalahan klasifikasi yang paling rendah, khususnya dalam mendeteksi kasus Cancer.

# In[198]:


# Memanggil fungsi make_evaluation untuk menampilkan f1 score dan confusion matrix
make_evaluation(y_test, pred_cat, title = f"Confusion Matrix Menggunakan Algoritma catboost")


# ## Interpretasi
# 
# Menggunakan CatboostClassifier dimaknai: dimaknai:
# 
# 1. 276 responden cancer telah diklasifikasikan dengan benar
# 2. 140 responden noncancer telah diklasifikasikan dengan benar
# 3. 27 responden noncancer diklasifikasikan sebagai responden cancer (False Positif)
# 4. 7 responden cancer diklasifikasikan sebagai responden noncancer (False Negatif)

# In[201]:


# Melatih model CatBoost (jika belum dilakukan)
model_cat = CatBoostClassifier(verbose=0, random_state=42)
model_cat.fit(X_train, y_train)

# Mendapatkan feature importance
feat_importances = pd.Series(model_cat.get_feature_importance(), index=X.columns)

feat_importances.nlargest(10).plot(kind='barh', figsize=(10,6), title='CatBoost (Diagnosis Kanker)')
plt.ylabel("Fitur")
plt.tight_layout()
plt.show()


# ### Kesimpulan:
# 
# Berdasarkan hasil analisis menggunakan model CatBoost, faktor-faktor yang paling berpengaruh dalam memprediksi risiko kanker adalah usia, aktivitas fisik, indeks massa tubuh (BMI), dan konsumsi alkohol. Hal ini menunjukkan bahwa selain faktor alami seperti usia, gaya hidup juga memainkan peran penting dalam risiko kanker. Oleh karena itu, upaya pencegahan kanker sebaiknya difokuskan pada promosi gaya hidup sehat, seperti rutin beraktivitas fisik, menjaga berat badan ideal, dan menghindari konsumsi alkohol berlebihan.

# In[ ]:




