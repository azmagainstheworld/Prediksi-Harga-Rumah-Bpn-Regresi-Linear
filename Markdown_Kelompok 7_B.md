# Prediksi Harga Rumah di Balikpapan Menggunakan Metode Regresi Linear

**Authors:** Adji Zahra Mahdiyyah, Muhammad Azka Yunastio, Muhammad Fachrudy Al-Ghifari, Naufal Faza Adhitya, Oktiara Azzahra Rahmadina

## Abstrak

Abstract Prediksi harga rumah di Balikpapan
Proyek ini membangun sistem prediksi harga rumah di Balikpapan menggunakan Regresi Linear berbasis data listing rumah. Fitur yang digunakan meliputi Tanah, Luas Bangunan, Jumlah Kamar Tidur, Jumlah Kamar Mandi, serta Kecamatan.

Tahapan utama proyek meliputi pembersihan data harga (konversi satuan Juta/Miliar menjadi IDR numerik), pembersihan atribut lokasi, penyaringan nilai hilang, serta penghapusan outlier menggunakan metode IQR dan batas manual. Model dilatih menggunakan LinearRegression dan dilakukan penyetelan sederhana menggunakan GridSearchCV. Model terbaik disimpan dalam format pickle` dan diintegrasikan ke aplikasi web menggunakan Flask agar pengguna dapat memasukkan data rumah dan memperoleh estimasi harga beserta rentang prediksi sederhana berbasis deviasi residual.

## Methods

### Regresi Linear

Regresi Linear adalah metode statistik dan machine learning yang digunakan untuk memodelkan hubungan antara satu variabel dependen (target) dengan satu atau lebih variabel independen (fitur) melalui hubungan linier. Tujuan utama regresi linear adalah menemukan persamaan matematis yang paling sesuai untuk merepresentasikan hubungan tersebut sehingga dapat digunakan untuk melakukan prediksi pada data baru.

Dalam penelitian ini, regresi linear digunakan untuk memprediksi **harga rumah di Kota Balikpapan** berdasarkan karakteristik properti dan lokasi. Pendekatan ini sejalan dengan konsep *hedonic pricing model* yang umum digunakan dalam analisis ekonomi properti.

Secara matematis, regresi linear multivariat dirumuskan sebagai:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon$$


**Keterangan:**
* $y$: variabel target (harga rumah)
* $x_1, x_2, \dots, x_n$: variabel fitur
* $\beta_0$: intercept (bias)
* $\beta_1, \dots, \beta_n$: koefisien regresi
* $\varepsilon$: error term (residual)

Setiap koefisien $\beta_i$ merepresentasikan perubahan rata-rata pada harga rumah akibat perubahan satu satuan pada fitur $x_i$, dengan asumsi fitur lainnya konstan.

---

### 1. Data Loading dan Preprocessing

Dataset dibaca dari berkas `data_final_bersih.csv` menggunakan pustaka **pandas**. Dataset terdiri dari **309 observasi** dan **10 kolom**, yang meliputi:

- **Fitur numerik**:
  - Luas Tanah ($m^2$)
  - Luas Bangunan ($m^2$)
  - Jumlah Kamar Tidur
  - Jumlah Kamar Mandi

- **Fitur kategorikal lokasi** (One-Hot Encoding):
  - Balikpapan Barat, Kota, Selatan, Tengah, Timur, dan Utara.

Variabel target pada penelitian ini adalah **Harga (IDR)**.

---

### 2. Baseline Model (Regresi Linear)

Sebagai baseline, dibangun model regresi linear menggunakan fitur asli tanpa transformasi target maupun feature engineering tambahan.

#### 2.1 Data Splitting
Dataset dibagi menjadi **80% data latih** dan **20% data uji** dengan `random_state = 42` untuk konsistensi hasil.

#### 2.2 Evaluasi Baseline
Model baseline dievaluasi menggunakan koefisien determinasi ($R^2$):

$$R^2 = 0.507$$

Nilai ini menunjukkan bahwa sekitar **50.7% variasi harga rumah** dapat dijelaskan oleh model regresi linear dasar.

---

### 3. Transformasi Variabel Target (Log Transform)

Distribusi harga rumah umumnya bersifat **right-skewed**. Untuk memperbaiki performa model dan mengurangi efek outlier, dilakukan transformasi logaritmik:

$$y_{\text{log}} = \ln(y)$$

Persamaan model menjadi:

$$\ln(y) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon$$


Evaluasi model dengan log transform menghasilkan peningkatan akurasi:

$$R^2 = 0.517$$

---

### 4. Feature Engineering dan Model Akhir

Dilakukan penambahan fitur turunan untuk meningkatkan korelasi antar variabel.

#### 4.1 Fitur Turunan
1. **Rasio luas bangunan terhadap luas tanah (`rasio_lb_lt`)**:

   $$\text{rasioLBLT} = \frac{\text{Luas Bangunan}}{\text{Luas Tanah}}$$

#### 4.2 Model Akhir
Model akhir dilatih menggunakan fitur numerik, fitur lokasi (OHE), dan fitur `rasio_lb_lt` dengan target logaritmik. Hasilnya menunjukkan peningkatan signifikan:

$$R^2 = 0.677$$

---

### 5. Evaluasi pada Skala Harga Asli

Nilai prediksi dikembalikan ke skala asli (Rupiah) menggunakan fungsi eksponensial:

$$\hat{y} = e^{\hat{y}_{\text{log}}}$$

#### 5.1 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} | y_{\text{true},i} - y_{\text{pred},i} |$$
$$\text{MAE} = 473,813,412 \text{ IDR}$$

#### 5.2 Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{ \frac{1}{m} \sum_{i=1}^{m} ( y_{\text{true},i} - y_{\text{pred},i} )^2 }$$
$$\text{RMSE} = 917,210,564 \text{ IDR}$$

---

### 6. Model Persistence
Model final disimpan menggunakan `joblib` dalam file `final_model.pkl` agar dapat digunakan langsung untuk proses deployment tanpa training ulang.

## Struktur Project

```
Prediksi-Harga-Rumah-BPN-Regresi-Linear-Main
├── env/
│   ├── Lib\site-packages
│   ├── Scripts
│   ├── share
│   └──pyvenv.cfg
│
├── model/
│   ├── data_collection.ipynb
│   ├── data_final_bersih.csv
│   ├── harga_rumah_balikpapan_tubes.csv
│   ├── kolom_fitur_model.pkl
│   ├── model_regresi_linear_harga_rumah.pkl
│   ├── model_training.ipynb
│   
│
├── templates/
│  └── index.html
│
├── app.py
└── README.md
```

## Implementation

  File terdiri dari:
1. data_collection.ipynb (membangun data_final_bersih.csv)  
2. model_training.ipynb (melatih model dan menyimpan
model_regresi_linear_harga_rumah.pkl)  
3. app.py dan index.html (aplikasi Flask untuk demo prediksi)

model/data_collection.ipynb

```python
1     # import sys
# !{sys.executable} -m pip install pandas requests beautifulsoup4 matplotlib
```
Penjelasan :
Cell 1: Digunakan pada awal notebook untuk memastikan semua library yang dibutuhkan sudah terinstall sebelum menjalankan code selanjutnya

```python
2    import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# DATA INPUT 
csv_content =   """daerah,harga,kamar_tidur,kamar_mandi,luas_tanah_m²,luas_bangunan_m²
"Balikpapan Kota,","7,90M",5,5,264,306
"Balikpapan Selatan,","1,12M",1,1,250,250
"Balikpapan Tengah,",925Jt,2,1,90,60
"Balikpapan Selatan,",575Jt,2,2,100,68
"Balikpapan Tengah,","695,70Jt",3,2,92,72
"Balikpapan Selatan,","1,65M",3,2,144,280
"Balikpapan Selatan,",795Jt,3,1,213,90
"Balikpapan Utara,","5,50M",6,7,400,350
"Balikpapan Timur,",Rp 3M,5,4,420,400
"Balikpapan Selatan,",Rp 5M,5,3,256,211
"Balikpapan Selatan,","1,25M",4,2,120,145
"Balikpapan Selatan,","1,40M",3,2,120,250
"Balikpapan Utara,","1,65M",4,2,165,220
"Balikpapan Utara,","1,85M",4,4,180,320
"Balikpapan Selatan,","2,50M",3,4,320,90
"Balikpapan Selatan,","5,50M",4,4,374,90
"Balikpapan Selatan,","1,79M",3,3,120,120
"Balikpapan Selatan,","4,50M",4,4,204,190
"Balikpapan Utara,","2,60M",3,3,165,300
"Balikpapan Utara,","1,15M",3,2,179,160
"Balikpapan Selatan,","1,70M",3,2,120,120
"Balikpapan Utara,",475Jt,3,1,124,60
"Balikpapan Selatan,","1,70M",4,3,160,100
"Balikpapan Utara,",450Jt,2,1,121,45
"Balikpapan Selatan,","1,80M",4,2,155,82
"Balikpapan Selatan,","1,40M",2,1,120,45
"Balikpapan Selatan,","1,30M",4,3,168,158
"Balikpapan Utara,","2,60M",3,3,165,330
"Balikpapan Utara,",Rp 2M,3,2,160,96
"Balikpapan Selatan,","3,25M",7,3,234,293
"Balikpapan Utara,","2,50M",4,2,307,250
"Balikpapan Utara,","1,90M",4,4,204,250
"Balikpapan Selatan,",Rp 4M,4,4,260,420
"Balikpapan Selatan,","6,37M",3,3,500,350
"Balikpapan Utara,","1,75M",4,2,216,170
"Balikpapan Utara,",450Jt,2,1,120,120
"Balikpapan Utara,",330Jt,2,1,84,36
"Balikpapan Utara,",450Jt,2,1,96,40
"Balikpapan Utara,",470Jt,2,1,140,45
"Balikpapan Utara,",550Jt,2,1,96,45
"Balikpapan Selatan,","2,50M",4,3,105,105
"Balikpapan Utara,",760Jt,4,2,156,156
"Balikpapan Selatan,",Rp 5M,5,4,276,276
"Balikpapan Selatan,",650Jt,2,1,84,84
"Balikpapan Selatan,","1,90M",3,2,72,72
"Balikpapan Selatan,",950Jt,3,2,198,198
"Balikpapan Kota,",Rp 12M,6,6,780,780
"Balikpapan Selatan,",Rp 5M,6,4,276,276
"Balikpapan Selatan,","6,60M",6,6,400,400
"Balikpapan Selatan,","1,60M",3,3,216,216
"Balikpapan Utara,","1,30M",3,3,135,135
"Balikpapan Selatan,","1,20M",5,3,260,260
"Balikpapan Utara,",750Jt,2,1,120,120
"Balikpapan Selatan,","2,85M",5,3,210,210
"Balikpapan Selatan,","2,30M",4,2,332,332
"Balikpapan Utara,",450Jt,2,1,120,45
"Balikpapan Selatan,","2,80M",5,3,296,296
"Balikpapan Selatan,","1,30M",4,3,200,200
"Balikpapan Selatan,","4,80M",6,5,395,395
"Balikpapan Selatan,","2,70M",5,4,180,180
"Balikpapan Selatan,",700Jt,2,1,104,50
"Balikpapan Utara,",380Jt,2,1,96,36
"Balikpapan Timur,",Rp 6M,2,1,720,90
"Balikpapan Utara,",Rp 3M,4,3,586,219
"Balikpapan Utara,",Rp 3M,3,3,318,200
"Balikpapan Selatan,","2,99M",3,3,60,220
"Balikpapan Utara,","2,15M",5,5,67,375
"Balikpapan Utara,","2,50M",3,3,27,105
"Balikpapan Utara,","1,75M",3,2,6,150
"Balikpapan Selatan,",795Jt,2,1,38,84
"Balikpapan Selatan,","2,10M",3,3,89,209
"Balikpapan Utara,","1,80M",6,3,4,204
"Balikpapan Selatan,","1,88M",3,2,18,98
"Balikpapan Utara,","1,80M",3,1,120,103
"Balikpapan Selatan,","1,60M",3,3,225,225
"Balikpapan Tengah,",880Jt,3,3,125,135
"Balikpapan Tengah,","4,50M",5,5,674,355
"Balikpapan Tengah,","5,20M",5,3,496,300
"Balikpapan Selatan,","5,50M",6,4,374,90
"Balikpapan Selatan,","1,50M",3,2,96,104
"Balikpapan Utara,","5,60M",2,3,870,260
"Balikpapan Tengah,","3,20M",5,4,210,245
"Balikpapan Timur,","1,95M",3,2,119,91
"Balikpapan Timur,","2,70M",3,2,105,119
"Balikpapan Selatan,","1,05M",3,2,145,120
"Balikpapan Tengah,","3,35M",5,4,220,340
"Balikpapan Selatan,","1,22M",3,2,72,50
"Balikpapan Selatan,","1,80M",5,2,150,230
"Balikpapan Selatan,","2,70M",3,2,150,219
"Balikpapan Selatan,","1,25M",3,2,120,140
"Balikpapan Tengah,",750Jt,2,1,105,56
"Balikpapan Selatan,",720Jt,2,1,96,36
"Balikpapan Kota,","1,55M",4,4,180,250
"Balikpapan Utara,","1,20M",3,2,157,111
"Balikpapan Selatan,","1,73M",3,4,205,180
"Balikpapan Selatan,","2,50M",8,8,340,200
"Balikpapan Utara,",950Jt,2,2,78,47
"Balikpapan Tengah,","1,25M",6,3,144,207
"Balikpapan Utara,","1,50M",3,2,160,120
"Balikpapan Utara,","3,20M",7,6,347,430
"Balikpapan Selatan,","1,70M",4,3,152,100
"Balikpapan Tengah,","6,65M",3,3,514,300
"Balikpapan Selatan,",Rp 2M,3,2,150,104
"Balikpapan Utara,","2,50M",4,3,200,105
"Balikpapan Tengah,","2,75M",4,3,240,110
"Balikpapan Timur,","2,75M",5,4,145,162
"Balikpapan Timur,","2,50M",3,3,555,250
"Balikpapan Tengah,","3,36M",6,4,495,320
"Balikpapan Timur,",885Jt,3,3,90,138
"Balikpapan Utara,","1,15M",2,2,78,47
"Balikpapan Selatan,","1,18M",4,2,120,150
"Balikpapan Timur,","4,50M",2,3,501,285
"Balikpapan Timur,","2,70M",4,3,311,225
"Balikpapan Selatan,",850Jt,2,2,90,54
"Balikpapan Selatan,","1,73M",3,2,205,180
"Balikpapan Selatan,",850Jt,3,1,95,60
"Balikpapan Utara,","5,60M",2,3,870,260
"Balikpapan Selatan,",900Jt,4,2,120,130
"Balikpapan Tengah,","2,70M",4,4,243,180
"Balikpapan Tengah,","1,35M",2,1,112,45
"Balikpapan Selatan,","1,85M",4,2,252,204
"Balikpapan Kota,","1,15M",3,2,90,90
"Balikpapan Timur,",800Jt,3,2,110,157
"Balikpapan Selatan,","1,25M",3,2,180,120
"Balikpapan Selatan,",515Jt,2,1,150,100
"Balikpapan Selatan,",Rp 10M,4,3,421,400
"Balikpapan Selatan,",Rp 7M,4,5,486,332
"Balikpapan Kota,","3,50M",2,2,940,500
"Balikpapan Kota,",950Jt,2,1,90,40
"Balikpapan Utara,","1,45M",4,3,130,190
"Balikpapan Selatan,","2,49M",3,2,295,120
"Balikpapan Utara,",850Jt,3,2,115,80
"Balikpapan Utara,",975Jt,3,2,105,100
"Balikpapan Utara,",850Jt,3,2,116,80
"Balikpapan Utara,","2,70M",4,3,318,318
"Balikpapan Selatan,","1,70M",3,2,126,126
"Balikpapan Utara,","4,67M",5,4,180,180
"Balikpapan Selatan,","2,70M",5,4,180,180
"Balikpapan Timur,",800Jt,2,3,104,104
"Balikpapan Selatan,",330Jt,2,1,140,140
"Balikpapan Selatan,",980Jt,3,1,160,100
"Balikpapan Selatan,",Rp 2M,2,1,324,120
"Balikpapan Selatan,",650Jt,2,1,90,45
"Balikpapan Selatan,",800Jt,3,3,162,90
"Balikpapan Selatan,",Rp 5M,4,2,330,200
"Balikpapan Selatan,","2,50M",4,3,340,200
"Balikpapan Selatan,","2,20M",5,4,200,200
"Balikpapan Selatan,","5,80M",6,4,625,500
"Balikpapan Selatan,","2,50M",3,2,135,135
"Balikpapan Selatan,","4,80M",6,5,395,250
"Balikpapan Selatan,","1,85M",3,2,204,142
"Balikpapan Utara,","1,50M",4,3,320,180
"Balikpapan Kota,","5,50M",9,3,668,120
"Balikpapan Selatan,","4,50M",4,5,554,355
"Balikpapan Tengah,","1,40M",5,4,294,250
"Balikpapan Tengah,",800Jt,4,4,300,150
"Balikpapan Selatan,","1,40M",4,4,153,258
"Balikpapan Utara,",450Jt,4,4,96,176
"Balikpapan Selatan,",930Jt,2,1,145,275
"Balikpapan Utara,",Rp 2M,9,9,170,320
"Balikpapan Tengah,",850Jt,3,3,130,280
"Balikpapan Kota,","2,80M",3,1,238,250
"Balikpapan Selatan,","1,65M",3,2,135,226
"Balikpapan Utara,","1,25M",3,2,72,142
"Balikpapan Utara,",950Jt,3,3,78,135
"Balikpapan Selatan,",700Jt,2,1,90,90
"Balikpapan Selatan,","1,30M",2,2,60,60
"Balikpapan Utara,","1,50M",3,2,340,410
"Balikpapan Utara,","9,80M",4,3,700,700
"Balikpapan Utara,","1,35M",4,2,150,150
"Balikpapan Utara,",870Jt,7,7,150,222
"Balikpapan Utara,","2,30M",3,2,294,294
"Balikpapan Selatan,","2,35M",4,4,135,170
"Balikpapan Selatan,","2,35M",3,3,135,170
"Balikpapan Tengah,",850Jt,3,2,213,180
"Balikpapan Selatan,","2,50M",3,2,105,210
"Balikpapan Selatan,",Rp 2M,3,2,204,142
"Balikpapan Selatan,",850Jt,3,1,160,100
"Balikpapan Selatan,","2,50M",6,4,150,234
"Balikpapan Selatan,","1,85M",3,3,140,190
"Balikpapan Selatan,","1,40M",2,2,78,80
"Balikpapan Tengah,",Rp 3M,5,2,427,220
"Balikpapan Selatan,","3,50M",4,2,240,320
"Balikpapan Selatan,","1,65M",4,2,175,250
"Balikpapan Kota,",900Jt,2,1,160,177
"Balikpapan Selatan,","1,55M",4,3,152,150
"Balikpapan Utara,",277Jt,2,1,72,36
"Balikpapan Selatan,","2,60M",4,3,220,120
"Balikpapan Timur,","1,70M",3,3,135,145
"Balikpapan Selatan,","1,10M",5,3,180,140
"Balikpapan Selatan,",690Jt,2,1,90,45
"Balikpapan Timur,",690Jt,2,1,90,45
"Balikpapan Selatan,","2,10M",3,3,128,117
"Balikpapan Barat,",695Jt,2,1,200,65
"Balikpapan Kota,",700Jt,4,2,149,100
"Balikpapan Utara,","1,35M",2,1,271,90
"Balikpapan Selatan,","1,20M",3,2,200,75
"Balikpapan Kota,",380Jt,2,1,122,70
"Balikpapan Selatan,","1,10M",3,2,150,102
"Balikpapan Selatan,",780Jt,5,3,120,140
"Balikpapan Selatan,","2,50M",4,4,315,210
"Balikpapan Selatan,",950Jt,2,1,96,60
"Balikpapan Selatan,",650Jt,3,1,81,81
"Balikpapan Utara,","3,20M",5,4,225,185
"Balikpapan Selatan,","4,20M",3,2,375,160
"Balikpapan Selatan,",950Jt,2,2,78,47
"Balikpapan Selatan,","4,25M",3,3,325,300
"Balikpapan Selatan,",680Jt,2,1,104,52
"Balikpapan Utara,","2,50M",3,2,294,150
"Balikpapan Selatan,","2,60M",4,3,105,100
"Balikpapan Selatan,",Rp 2M,5,5,332,200
"Balikpapan Utara,","2,80M",3,3,228,185
"Balikpapan Selatan,",550Jt,2,1,90,30
"Balikpapan Utara,","1,75M",3,2,180,90
"Balikpapan Selatan,",590Jt,2,2,150,100
"Balikpapan Selatan,",Rp 2M,3,2,160,104
"Balikpapan Selatan,",Rp 2M,3,2,204,140
"Balikpapan Utara,",875Jt,4,2,270,80
"Balikpapan Selatan,",675Jt,2,1,90,60
"Balikpapan Selatan,",380Jt,3,1,90,90
"Balikpapan Selatan,",Rp 5M,5,5,150,187
"Balikpapan Utara,",550Jt,2,1,196,87
"Balikpapan Selatan,",590Jt,2,2,109,80
"Balikpapan Selatan,","2,60M",4,4,165,300
"Balikpapan Selatan,",Rp 5M,4,3,400,380
"Balikpapan Selatan,","6,50M",3,3,673,500
"Balikpapan Selatan,","3,50M",3,3,363,130
"Balikpapan Tengah,",Rp 1M,3,2,319,150
"Balikpapan Selatan,","1,20M",4,4,150,180
"Balikpapan Tengah,","1,30M",6,3,209,339
"Balikpapan Selatan,","1,20M",4,2,120,144
"Balikpapan Selatan,",450Jt,2,1,96,42
"Balikpapan Timur,",880Jt,3,2,90,70
"Balikpapan Selatan,","1,50M",3,3,150,200
"Balikpapan Selatan,","1,50M",3,3,160,160
"Balikpapan Kota,","1,50M",4,2,105,96
"Balikpapan Kota,","4,50M",5,4,400,236
"Balikpapan Utara,","1,60M",3,3,90,120
"Balikpapan Utara,","1,40M",3,2,96,104
"Balikpapan Utara,","1,10M",3,2,100,60
"Balikpapan Utara,","2,10M",4,3,200,220
"Balikpapan Selatan,",700Jt,3,2,150,80
"Balikpapan Utara,","1,60M",7,5,224,72
"Balikpapan Selatan,",500Jt,2,1,84,45
"Balikpapan Kota,",730Jt,2,1,120,70
"Balikpapan Timur,","1,90M",5,5,180,230
"Balikpapan Selatan,",850Jt,3,2,198,198
"Balikpapan Selatan,","2,50M",15,14,200,460
"Balikpapan Selatan,","2,50M",3,3,150,200
"Balikpapan Kota,","1,90M",4,4,287,125
"Balikpapan Timur,",760Jt,2,1,119,89
"Balikpapan Utara,",900Jt,2,2,150,110
"Balikpapan Selatan,",730Jt,3,2,175,75
"Balikpapan Barat,",570Jt,2,1,200,61
"Balikpapan Selatan,",800Jt,3,2,145,120
"Balikpapan Kota,","1,20M",4,4,150,120
"Balikpapan Selatan,",980Jt,2,1,102,100
"Balikpapan Utara,","1,55M",3,2,150,145
"Balikpapan Selatan,","6,50M",4,3,363,270
"Balikpapan Selatan,",850Jt,3,4,72,90
"Balikpapan Selatan,","3,50M",3,2,180,90
"Balikpapan Selatan,","2,50M",2,2,180,90
"Balikpapan Selatan,",Rp 2M,3,2,102,70
"Balikpapan Utara,",600Jt,3,2,105,70
"Balikpapan Selatan,","2,10M",4,3,180,180
"Balikpapan Selatan,","3,80M",4,3,220,250
"Balikpapan Selatan,","1,30M",3,2,165,80
"Balikpapan Utara,",Rp 2M,4,3,117,150
"Balikpapan Utara,","3,50M",3,4,252,117
"Balikpapan Selatan,",Rp 2M,3,2,216,150
"Balikpapan Utara,","4,80M",5,4,396,400
"Balikpapan Selatan,","1,30M",2,1,131,57
"Balikpapan Utara,","1,70M",3,2,154,150
"Balikpapan Utara,","1,70M",3,3,90,114
"Balikpapan Utara,",830Jt,3,2,72,66
"Balikpapan Timur,",930Jt,3,2,120,100
"Balikpapan Kota,",Rp 5M,4,3,585,250
"Balikpapan Utara,",820Jt,3,2,72,66
"Balikpapan Selatan,","8,50M",4,6,190,380
"Balikpapan Selatan,",650Jt,2,2,120,52
"Balikpapan Selatan,",Rp 11M,8,6,358,458
"Balikpapan Utara,","4,50M",5,4,350,450
"Balikpapan Selatan,",680Jt,2,1,72,40
"Balikpapan Selatan,",Rp 7M,5,4,395,200
"Balikpapan Utara,","3,50M",3,4,252,117
"Balikpapan Selatan,",675Jt,3,2,120,100
"Balikpapan Selatan,","2,30M",5,4,300,5
"Balikpapan Selatan,","1,90M",3,4,112,144
"Balikpapan Selatan,","1,20M",2,2,90,72
"Balikpapan Kota,",800Jt,2,2,153,75
"Balikpapan Kota,",850Jt,3,2,331,102
"Balikpapan Selatan,",Rp 3M,3,2,276,150
"Balikpapan Timur,","1,20M",2,2,162,70
"Balikpapan Timur,","1,60M",3,3,98,140
"Balikpapan Selatan,","4,50M",3,3,465,265
"Balikpapan Tengah,",695Jt,2,1,200,65
"Balikpapan Utara,",700Jt,4,2,149,100
"Balikpapan Selatan,","2,70M",4,2,332,120
"Balikpapan Selatan,",850Jt,2,2,60,120
"Balikpapan Selatan,","3,60M",4,4,280,225
"Balikpapan Selatan,","2,15M",3,3,126,88
"Balikpapan Utara,","1,60M",5,4,180,220
"Balikpapan Selatan,","3,80M",4,4,280,225
"Balikpapan Selatan,",Rp 5M,4,2,330,200
"Balikpapan Selatan,",Rp 1M,4,2,127,107
"Balikpapan Selatan,","1,90M",4,2,180,150
"Balikpapan Selatan,","2,50M",3,3,222,160
"Balikpapan Selatan,","3,40M",5,4,190,250
"Balikpapan Selatan,","2,75M",3,2,95,160
"Balikpapan Timur,",600Jt,2,1,60,30
"Balikpapan Utara,",550Jt,3,3,77,90
"Balikpapan Selatan,","4,50M",3,3,490,330
"Balikpapan Selatan,","2,10M",4,4,204,250
"Balikpapan Timur,","1,20M",5,3,285,106
"Balikpapan Selatan,","2,50M",2,2,300,70
"Balikpapan Selatan,",875Jt,4,2,117,85
"Balikpapan Timur,","1,30M",3,3,140,98
"Balikpapan Utara,",950Jt,3,2,163,100
"Balikpapan Utara,","1,40M",5,4,180,220
"Balikpapan Utara,","1,60M",3,3,91,175
"Balikpapan Utara,","1,70M",2,1,112,56
"Balikpapan Selatan,","2,60M",5,3,300,200
"Balikpapan Selatan,",Rp 17M,5,4,660,770
"Balikpapan Selatan,","2,50M",3,2,300,140
"Balikpapan Kota,","1,60M",4,3,248,182
"Balikpapan Timur,","1,60M",6,4,200,183
"Balikpapan Timur,",880Jt,2,2,156,72
"Balikpapan Timur,","1,20M",3,2,105,91
"Balikpapan Timur,","1,50M",3,2,135,85
"Balikpapan Selatan,","2,80M",3,2,126,135
"Balikpapan Selatan,","2,25M",3,3,120,150
"Balikpapan Utara,","3,40M",5,4,190,300
"Balikpapan Kota,","2,20M",4,4,350,250
"Balikpapan Selatan,","2,50M",4,3,340,200
"Balikpapan Kota,",Rp 15M,67,6,660,750
"Balikpapan Selatan,","2,90M",4,2,220,200
"Balikpapan Selatan,","1,60M",3,3,135,200
"Balikpapan Selatan,","2,55M",4,3,165,250
"Balikpapan Utara,","3,50M",10,6,418,600
"Balikpapan Selatan,","3,50M",4,4,246,300
"Balikpapan Utara,",455Jt,2,1,97,39
"Balikpapan Tengah,",895Jt,4,2,325,240
"Balikpapan Selatan,","2,50M",4,3,419,250
"Balikpapan Utara,",385Jt,2,1,80,45
"Balikpapan Utara,",495Jt,2,1,110,42
"Balikpapan Utara,",550Jt,3,1,120,50
"Balikpapan Utara,","1,10M",2,2,450,114
"Balikpapan Utara,",275Jt,2,1,121,36
"Balikpapan Utara,","1,50M",4,3,170,120
"Balikpapan Utara,",430Jt,2,1,87,36
"Balikpapan Utara,",395Jt,3,1,120,50
"Balikpapan Utara,",500Jt,2,1,78,45
"Balikpapan Selatan,",500Jt,2,1,78,45
"Balikpapan Utara,",395Jt,2,1,96,36
"Balikpapan Selatan,","2,50M",11,10,375,260
"Balikpapan Selatan,","459,25Jt",2,1,102,36
"Balikpapan Utara,","383,68Jt",2,1,107,36
"Balikpapan Selatan,",690Jt,3,2,90,75
"Balikpapan Utara,",475Jt,2,1,120,45
"Balikpapan Timur,",385Jt,2,1,84,36
"Balikpapan Utara,",250Jt,2,1,160,36
"Balikpapan Selatan,",650Jt,2,1,119,119
"Balikpapan Selatan,","2,70M",4,3,180,210
"Balikpapan Selatan,","4,20M",3,4,550,550
"Balikpapan Selatan,","1,60M",3,2,216,100
"Balikpapan Tengah,",Rp 2M,3,2,763,280
"Balikpapan Selatan,",950Jt,3,2,150,120
"Balikpapan Kota,","1,50M",3,2,462,235
"Balikpapan Selatan,","1,10M",2,2,90,90
"Balikpapan Selatan,","3,30M",4,3,147,240
"Balikpapan Selatan,","1,15M",2,2,78,47
"Balikpapan Selatan,",950Jt,3,2,127,120
"""

df = pd.read_csv(io.StringIO(csv_content), sep=',', quotechar='"')

# PEMBERSIHAN DATA

# 1. Pembersihan Harga
def clean_price(price_str):
    if pd.isna(price_str) or price_str is None:
        return None
    
    # Menghapus 'Rp' dan spasi
    price_str = str(price_str).strip().replace('Rp', '').replace(' ', '')
    
    # Penanganan Miliar (M)
    if 'M' in price_str:
        num_part = price_str.replace('M', '').replace(',', '.') # Mengganti koma desimal
        try: return float(num_part) * 10**9
        except ValueError: return None
        
    # Penanganan Juta (Jt)
    elif 'Jt' in price_str:
        num_part = price_str.replace('Jt', '').replace(',', '.')
        try: return float(num_part) * 10**6
        except ValueError: return None
    
    return None

df['Harga (IDR)'] = df['harga'].apply(clean_price)

# 2. Pembersihan Daerah
# Menghapus koma di akhir nama daerah dan mengambil subdistrict
df['subdistrict'] = df['daerah'].str.replace(',', '').str.strip()

# 3. Konversi Kolom Numerik dan Filtering NaN
# Konversi kolom-kolom numerik yang sudah bersih
df['luas_tanah_m²'] = pd.to_numeric(df['luas_tanah_m²'], errors='coerce')
df['luas_bangunan_m²'] = pd.to_numeric(df['luas_bangunan_m²'], errors='coerce')
df['kamar_tidur'] = pd.to_numeric(df['kamar_tidur'], errors='coerce')
df['kamar_mandi'] = pd.to_numeric(df['kamar_mandi'], errors='coerce')

# Filter baris dengan nilai NaN pada Luas Tanah (m²)
df_filtered = df.dropna(subset=['luas_tanah_m²'])

# PENGELOMPOKAN DATA
print("\n" + "="*70)
print("## 🔢 Hasil Pengelompokan Data per Kecamatan 🏠")
print("="*70)

# A. Jumlah Data per Kecamatan
count_by_subdistrict = df_filtered['subdistrict'].value_counts().rename('Jumlah Data').to_frame()
print("### A. Jumlah Data")
print(count_by_subdistrict.to_string()) 

print("\n" + "-"*70)

# B. Rata-rata Harga per Kecamatan (IDR Numerik Murni)
avg_price_by_subdistrict = df_filtered.groupby('subdistrict')['Harga (IDR)'].mean().sort_values(ascending=False).to_frame()
avg_price_by_subdistrict.columns = ['Harga Rata-rata (IDR)']

# Format harga menjadi integer murni dengan pemisah ribuan (titik)
avg_price_by_subdistrict['Harga Rata-rata (IDR) Numeric'] = (
    avg_price_by_subdistrict['Harga Rata-rata (IDR)']
    .round()
    .astype('Int64')
    .apply(lambda x: f'{x:,}'.replace(',', '.'))
)

print("### B. Harga Rata-rata (IDR Numerik Murni)")
     print(avg_price_by_subdistrict[['Harga Rata-rata (IDR) Numeric']].to_string())
```
Penjelasan:

Cell 2: Install Library
Import library eksternal (pandas, numpy, requests, beautifulsoup4, matplotlib) dan modul internal (io) untuk functionality data processing, http requests, HTML parsing, dan visualisasi.

```python
3    import pandas as pd
import numpy as np
import io

# =============================================================
# 1. DATA INPUT (Konten CSV Bersih Anda)
# =============================================================
csv_content = """daerah,harga,kamar_tidur,kamar_mandi,luas_tanah_m²,luas_bangunan_m²
"Balikpapan Kota,","7,90M",5,5,264,306
"Balikpapan Selatan,","1,12M",1,1,250,250
"Balikpapan Tengah,",925Jt,2,1,90,60
"Balikpapan Selatan,",575Jt,2,2,100,68
"Balikpapan Tengah,","695,70Jt",3,2,92,72
"Balikpapan Selatan,","1,65M",3,2,144,280
"Balikpapan Selatan,",795Jt,3,1,213,90
"Balikpapan Utara,","5,50M",6,7,400,350
"Balikpapan Timur,",Rp 3M,5,4,420,400
"Balikpapan Selatan,",Rp 5M,5,3,256,211
"Balikpapan Selatan,","1,25M",4,2,120,145
"Balikpapan Selatan,","1,40M",3,2,120,250
"Balikpapan Utara,","1,65M",4,2,165,220
"Balikpapan Utara,","1,85M",4,4,180,320
"Balikpapan Selatan,","2,50M",3,4,320,90
"Balikpapan Selatan,","5,50M",4,4,374,90
"Balikpapan Selatan,","1,79M",3,3,120,120
"Balikpapan Selatan,","4,50M",4,4,204,190
"Balikpapan Utara,","2,60M",3,3,165,300
"Balikpapan Utara,","1,15M",3,2,179,160
"Balikpapan Selatan,","1,70M",3,2,120,120
"Balikpapan Utara,",475Jt,3,1,124,60
"Balikpapan Selatan,","1,70M",4,3,160,100
"Balikpapan Utara,",450Jt,2,1,121,45
"Balikpapan Selatan,","1,80M",4,2,155,82
"Balikpapan Selatan,","1,40M",2,1,120,45
"Balikpapan Selatan,","1,30M",4,3,168,158
"Balikpapan Utara,","2,60M",3,3,165,330
"Balikpapan Utara,",Rp 2M,3,2,160,96
"Balikpapan Selatan,","3,25M",7,3,234,293
"Balikpapan Utara,","2,50M",4,2,307,250
"Balikpapan Utara,","1,90M",4,4,204,250
"Balikpapan Selatan,",Rp 4M,4,4,260,420
"Balikpapan Selatan,","6,37M",3,3,500,350
"Balikpapan Utara,","1,75M",4,2,216,170
"Balikpapan Utara,",450Jt,2,1,120,120
"Balikpapan Utara,",330Jt,2,1,84,36
"Balikpapan Utara,",450Jt,2,1,96,40
"Balikpapan Utara,",470Jt,2,1,140,45
"Balikpapan Utara,",550Jt,2,1,96,45
"Balikpapan Selatan,","2,50M",4,3,105,105
"Balikpapan Utara,",760Jt,4,2,156,156
"Balikpapan Selatan,",Rp 5M,5,4,276,276
"Balikpapan Selatan,",650Jt,2,1,84,84
"Balikpapan Selatan,","1,90M",3,2,72,72
"Balikpapan Selatan,",950Jt,3,2,198,198
"Balikpapan Kota,",Rp 12M,6,6,780,780
"Balikpapan Selatan,",Rp 5M,6,4,276,276
"Balikpapan Selatan,","6,60M",6,6,400,400
"Balikpapan Selatan,","1,60M",3,3,216,216
"Balikpapan Utara,","1,30M",3,3,135,135
"Balikpapan Selatan,","1,20M",5,3,260,260
"Balikpapan Utara,",750Jt,2,1,120,120
"Balikpapan Selatan,","2,85M",5,3,210,210
"Balikpapan Selatan,","2,30M",4,2,332,332
"Balikpapan Utara,",450Jt,2,1,120,45
"Balikpapan Selatan,","2,80M",5,3,296,296
"Balikpapan Selatan,","1,30M",4,3,200,200
"Balikpapan Selatan,","4,80M",6,5,395,395
"Balikpapan Selatan,","2,70M",5,4,180,180
"Balikpapan Selatan,",700Jt,2,1,104,50
"Balikpapan Utara,",380Jt,2,1,96,36
"Balikpapan Timur,",Rp 6M,2,1,720,90
"Balikpapan Utara,",Rp 3M,4,3,586,219
"Balikpapan Utara,",Rp 3M,3,3,318,200
"Balikpapan Selatan,","2,99M",3,3,60,220
"Balikpapan Utara,","2,15M",5,5,67,375
"Balikpapan Utara,","2,50M",3,3,27,105
"Balikpapan Utara,","1,75M",3,2,6,150
"Balikpapan Selatan,",795Jt,2,1,38,84
"Balikpapan Selatan,","2,10M",3,3,89,209
"Balikpapan Utara,","1,80M",6,3,4,204
"Balikpapan Selatan,","1,88M",3,2,18,98
"Balikpapan Utara,","1,80M",3,1,120,103
"Balikpapan Selatan,","1,60M",3,3,225,225
"Balikpapan Tengah,",880Jt,3,3,125,135
"Balikpapan Tengah,","4,50M",5,5,674,355
"Balikpapan Tengah,","5,20M",5,3,496,300
"Balikpapan Selatan,","5,50M",6,4,374,90
"Balikpapan Selatan,","1,50M",3,2,96,104
"Balikpapan Utara,","5,60M",2,3,870,260
"Balikpapan Tengah,","3,20M",5,4,210,245
"Balikpapan Timur,","1,95M",3,2,119,91
"Balikpapan Timur,","2,70M",3,2,105,119
"Balikpapan Selatan,","1,05M",3,2,145,120
"Balikpapan Tengah,","3,35M",5,4,220,340
"Balikpapan Selatan,","1,22M",3,2,72,50
"Balikpapan Selatan,","1,80M",5,2,150,230
"Balikpapan Selatan,","2,70M",3,2,150,219
"Balikpapan Selatan,","1,25M",3,2,120,140
"Balikpapan Tengah,",750Jt,2,1,105,56
"Balikpapan Selatan,",720Jt,2,1,96,36
"Balikpapan Kota,","1,55M",4,4,180,250
"Balikpapan Utara,","1,20M",3,2,157,111
"Balikpapan Selatan,","1,73M",3,4,205,180
"Balikpapan Selatan,","2,50M",8,8,340,200
"Balikpapan Utara,",950Jt,2,2,78,47
"Balikpapan Tengah,","1,25M",6,3,144,207
"Balikpapan Utara,","1,50M",3,2,160,120
"Balikpapan Utara,","3,20M",7,6,347,430
"Balikpapan Selatan,","1,70M",4,3,152,100
"Balikpapan Tengah,","6,65M",3,3,514,300
"Balikpapan Selatan,",Rp 2M,3,2,150,104
"Balikpapan Utara,","2,50M",4,3,200,105
"Balikpapan Tengah,","2,75M",4,3,240,110
"Balikpapan Timur,","2,75M",5,4,145,162
"Balikpapan Timur,","2,50M",3,3,555,250
"Balikpapan Tengah,","3,36M",6,4,495,320
"Balikpapan Timur,",885Jt,3,3,90,138
"Balikpapan Utara,","1,15M",2,2,78,47
"Balikpapan Selatan,","1,18M",4,2,120,150
"Balikpapan Timur,","4,50M",2,3,501,285
"Balikpapan Timur,","2,70M",4,3,311,225
"Balikpapan Selatan,",850Jt,2,2,90,54
"Balikpapan Selatan,","1,73M",3,2,205,180
"Balikpapan Selatan,",850Jt,3,1,95,60
"Balikpapan Utara,","5,60M",2,3,870,260
"Balikpapan Selatan,",900Jt,4,2,120,130
"Balikpapan Tengah,","2,70M",4,4,243,180
"Balikpapan Tengah,","1,35M",2,1,112,45
"Balikpapan Selatan,","1,85M",4,2,252,204
"Balikpapan Kota,","1,15M",3,2,90,90
"Balikpapan Timur,",800Jt,3,2,110,157
"Balikpapan Selatan,","1,25M",3,2,180,120
"Balikpapan Selatan,",515Jt,2,1,150,100
"Balikpapan Selatan,",Rp 10M,4,3,421,400
"Balikpapan Selatan,",Rp 7M,4,5,486,332
"Balikpapan Kota,","3,50M",2,2,940,500
"Balikpapan Kota,",950Jt,2,1,90,40
"Balikpapan Utara,","1,45M",4,3,130,190
"Balikpapan Selatan,","2,49M",3,2,295,120
"Balikpapan Utara,",850Jt,3,2,115,80
"Balikpapan Utara,",975Jt,3,2,105,100
"Balikpapan Utara,",850Jt,3,2,116,80
"Balikpapan Utara,","2,70M",4,3,318,318
"Balikpapan Selatan,","1,70M",3,2,126,126
"Balikpapan Utara,","4,67M",5,4,180,180
"Balikpapan Selatan,","2,70M",5,4,180,180
"Balikpapan Timur,",800Jt,2,3,104,104
"Balikpapan Selatan,",330Jt,2,1,140,140
"Balikpapan Selatan,",980Jt,3,1,160,100
"Balikpapan Selatan,",Rp 2M,2,1,324,120
"Balikpapan Selatan,",650Jt,2,1,90,45
"Balikpapan Selatan,",800Jt,3,3,162,90
"Balikpapan Selatan,",Rp 5M,4,2,330,200
"Balikpapan Selatan,","2,50M",4,3,340,200
"Balikpapan Selatan,","2,20M",5,4,200,200
"Balikpapan Selatan,","5,80M",6,4,625,500
"Balikpapan Selatan,","2,50M",3,2,135,135
"Balikpapan Selatan,","4,80M",6,5,395,250
"Balikpapan Selatan,","1,85M",3,2,204,142
"Balikpapan Utara,","1,50M",4,3,320,180
"Balikpapan Kota,","5,50M",9,3,668,120
"Balikpapan Selatan,","4,50M",4,5,554,355
"Balikpapan Tengah,","1,40M",5,4,294,250
"Balikpapan Tengah,",800Jt,4,4,300,150
"Balikpapan Selatan,","1,40M",4,4,153,258
"Balikpapan Utara,",450Jt,4,4,96,176
"Balikpapan Selatan,",930Jt,2,1,145,275
"Balikpapan Utara,",Rp 2M,9,9,170,320
"Balikpapan Tengah,",850Jt,3,3,130,280
"Balikpapan Kota,","2,80M",3,1,238,250
"Balikpapan Selatan,","1,65M",3,2,135,226
"Balikpapan Utara,","1,25M",3,2,72,142
"Balikpapan Utara,",950Jt,3,3,78,135
"Balikpapan Selatan,",700Jt,2,1,90,90
"Balikpapan Selatan,","1,30M",2,2,60,60
"Balikpapan Utara,","1,50M",3,2,340,410
"Balikpapan Utara,","9,80M",4,3,700,700
"Balikpapan Utara,","1,35M",4,2,150,150
"Balikpapan Utara,",870Jt,7,7,150,222
"Balikpapan Utara,","2,30M",3,2,294,294
"Balikpapan Selatan,","2,35M",4,4,135,170
"Balikpapan Selatan,","2,35M",3,3,135,170
"Balikpapan Tengah,",850Jt,3,2,213,180
"Balikpapan Selatan,","2,50M",3,2,105,210
"Balikpapan Selatan,",Rp 2M,3,2,204,142
"Balikpapan Selatan,",850Jt,3,1,160,100
"Balikpapan Selatan,","2,50M",6,4,150,234
"Balikpapan Selatan,","1,85M",3,3,140,190
"Balikpapan Selatan,","1,40M",2,2,78,80
"Balikpapan Tengah,",Rp 3M,5,2,427,220
"Balikpapan Selatan,","3,50M",4,2,240,320
"Balikpapan Selatan,","1,65M",4,2,175,250
"Balikpapan Kota,",900Jt,2,1,160,177
"Balikpapan Selatan,","1,55M",4,3,152,150
"Balikpapan Utara,",277Jt,2,1,72,36
"Balikpapan Selatan,","2,60M",4,3,220,120
"Balikpapan Timur,","1,70M",3,3,135,145
"Balikpapan Selatan,","1,10M",5,3,180,140
"Balikpapan Selatan,",690Jt,2,1,90,45
"Balikpapan Timur,",690Jt,2,1,90,45
"Balikpapan Selatan,","2,10M",3,3,128,117
"Balikpapan Barat,",695Jt,2,1,200,65
"Balikpapan Kota,",700Jt,4,2,149,100
"Balikpapan Utara,","1,35M",2,1,271,90
"Balikpapan Selatan,","1,20M",3,2,200,75
"Balikpapan Kota,",380Jt,2,1,122,70
"Balikpapan Selatan,","1,10M",3,2,150,102
"Balikpapan Selatan,",780Jt,5,3,120,140
"Balikpapan Selatan,","2,50M",4,4,315,210
"Balikpapan Selatan,",950Jt,2,1,96,60
"Balikpapan Selatan,",650Jt,3,1,81,81
"Balikpapan Utara,","3,20M",5,4,225,185
"Balikpapan Selatan,","4,20M",3,2,375,160
"Balikpapan Selatan,",950Jt,2,2,78,47
"Balikpapan Selatan,","4,25M",3,3,325,300
"Balikpapan Selatan,",680Jt,2,1,104,52
"Balikpapan Utara,","2,50M",3,2,294,150
"Balikpapan Selatan,","2,60M",4,3,105,100
"Balikpapan Selatan,",Rp 2M,5,5,332,200
"Balikpapan Utara,","2,80M",3,3,228,185
"Balikpapan Selatan,",550Jt,2,1,90,30
"Balikpapan Utara,","1,75M",3,2,180,90
"Balikpapan Selatan,",590Jt,2,2,150,100
"Balikpapan Selatan,",Rp 2M,3,2,160,104
"Balikpapan Selatan,",Rp 2M,3,2,204,140
"Balikpapan Utara,",875Jt,4,2,270,80
"Balikpapan Selatan,",675Jt,2,1,90,60
"Balikpapan Selatan,",380Jt,3,1,90,90
"Balikpapan Selatan,",Rp 5M,5,5,150,187
"Balikpapan Utara,",550Jt,2,1,196,87
"Balikpapan Selatan,",590Jt,2,2,109,80
"Balikpapan Selatan,","2,60M",4,4,165,300
"Balikpapan Selatan,",Rp 5M,4,3,400,380
"Balikpapan Selatan,","6,50M",3,3,673,500
"Balikpapan Selatan,","3,50M",3,3,363,130
"Balikpapan Tengah,",Rp 1M,3,2,319,150
"Balikpapan Selatan,","1,20M",4,4,150,180
"Balikpapan Tengah,","1,30M",6,3,209,339
"Balikpapan Selatan,","1,20M",4,2,120,144
"Balikpapan Selatan,",450Jt,2,1,96,42
"Balikpapan Timur,",880Jt,3,2,90,70
"Balikpapan Selatan,","1,50M",3,3,150,200
"Balikpapan Selatan,","1,50M",3,3,160,160
"Balikpapan Kota,","1,50M",4,2,105,96
"Balikpapan Kota,","4,50M",5,4,400,236
"Balikpapan Utara,","1,60M",3,3,90,120
"Balikpapan Utara,","1,40M",3,2,96,104
"Balikpapan Utara,","1,10M",3,2,100,60
"Balikpapan Utara,","2,10M",4,3,200,220
"Balikpapan Selatan,",700Jt,3,2,150,80
"Balikpapan Utara,","1,60M",7,5,224,72
"Balikpapan Selatan,",500Jt,2,1,84,45
"Balikpapan Kota,",730Jt,2,1,120,70
"Balikpapan Timur,","1,90M",5,5,180,230
"Balikpapan Selatan,",850Jt,3,2,198,198
"Balikpapan Selatan,","2,50M",15,14,200,460
"Balikpapan Selatan,","2,50M",3,3,150,200
"Balikpapan Kota,","1,90M",4,4,287,125
"Balikpapan Timur,",760Jt,2,1,119,89
"Balikpapan Utara,",900Jt,2,2,150,110
"Balikpapan Selatan,",730Jt,3,2,175,75
"Balikpapan Barat,",570Jt,2,1,200,61
"Balikpapan Selatan,",800Jt,3,2,145,120
"Balikpapan Kota,","1,20M",4,4,150,120
"Balikpapan Selatan,",980Jt,2,1,102,100
"Balikpapan Utara,","1,55M",3,2,150,145
"Balikpapan Selatan,","6,50M",4,3,363,270
"Balikpapan Selatan,",850Jt,3,4,72,90
"Balikpapan Selatan,","3,50M",3,2,180,90
"Balikpapan Selatan,","2,50M",2,2,180,90
"Balikpapan Selatan,",Rp 2M,3,2,102,70
"Balikpapan Utara,",600Jt,3,2,105,70
"Balikpapan Selatan,","2,10M",4,3,180,180
"Balikpapan Selatan,","3,80M",4,3,220,250
"Balikpapan Selatan,","1,30M",3,2,165,80
"Balikpapan Utara,",Rp 2M,4,3,117,150
"Balikpapan Utara,","3,50M",3,4,252,117
"Balikpapan Selatan,",Rp 2M,3,2,216,150
"Balikpapan Utara,","4,80M",5,4,396,400
"Balikpapan Selatan,","1,30M",2,1,131,57
"Balikpapan Utara,","1,70M",3,2,154,150
"Balikpapan Utara,","1,70M",3,3,90,114
"Balikpapan Utara,",830Jt,3,2,72,66
"Balikpapan Timur,",930Jt,3,2,120,100
"Balikpapan Kota,",Rp 5M,4,3,585,250
"Balikpapan Utara,",820Jt,3,2,72,66
"Balikpapan Selatan,","8,50M",4,6,190,380
"Balikpapan Selatan,",650Jt,2,2,120,52
"Balikpapan Selatan,",Rp 11M,8,6,358,458
"Balikpapan Utara,","4,50M",5,4,350,450
"Balikpapan Selatan,",680Jt,2,1,72,40
"Balikpapan Selatan,",Rp 7M,5,4,395,200
"Balikpapan Utara,","3,50M",3,4,252,117
"Balikpapan Selatan,",675Jt,3,2,120,100
"Balikpapan Selatan,","2,30M",5,4,300,5
"Balikpapan Selatan,","1,90M",3,4,112,144
"Balikpapan Selatan,","1,20M",2,2,90,72
"Balikpapan Kota,",800Jt,2,2,153,75
"Balikpapan Kota,",850Jt,3,2,331,102
"Balikpapan Selatan,",Rp 3M,3,2,276,150
"Balikpapan Timur,","1,20M",2,2,162,70
"Balikpapan Timur,","1,60M",3,3,98,140
"Balikpapan Selatan,","4,50M",3,3,465,265
"Balikpapan Tengah,",695Jt,2,1,200,65
"Balikpapan Utara,",700Jt,4,2,149,100
"Balikpapan Selatan,","2,70M",4,2,332,120
"Balikpapan Selatan,",850Jt,2,2,60,120
"Balikpapan Selatan,","3,60M",4,4,280,225
"Balikpapan Selatan,","2,15M",3,3,126,88
"Balikpapan Utara,","1,60M",5,4,180,220
"Balikpapan Selatan,","3,80M",4,4,280,225
"Balikpapan Selatan,",Rp 5M,4,2,330,200
"Balikpapan Selatan,",Rp 1M,4,2,127,107
"Balikpapan Selatan,","1,90M",4,2,180,150
"Balikpapan Selatan,","2,50M",3,3,222,160
"Balikpapan Selatan,","3,40M",5,4,190,250
"Balikpapan Selatan,","2,75M",3,2,95,160
"Balikpapan Timur,",600Jt,2,1,60,30
"Balikpapan Utara,",550Jt,3,3,77,90
"Balikpapan Selatan,","4,50M",3,3,490,330
"Balikpapan Selatan,","2,10M",4,4,204,250
"Balikpapan Timur,","1,20M",5,3,285,106
"Balikpapan Selatan,","2,50M",2,2,300,70
"Balikpapan Selatan,",875Jt,4,2,117,85
"Balikpapan Timur,","1,30M",3,3,140,98
"Balikpapan Utara,",950Jt,3,2,163,100
"Balikpapan Utara,","1,40M",5,4,180,220
"Balikpapan Utara,","1,60M",3,3,91,175
"Balikpapan Utara,","1,70M",2,1,112,56
"Balikpapan Selatan,","2,60M",5,3,300,200
"Balikpapan Selatan,",Rp 17M,5,4,660,770
"Balikpapan Selatan,","2,50M",3,2,300,140
"Balikpapan Kota,","1,60M",4,3,248,182
"Balikpapan Timur,","1,60M",6,4,200,183
"Balikpapan Timur,",880Jt,2,2,156,72
"Balikpapan Timur,","1,20M",3,2,105,91
"Balikpapan Timur,","1,50M",3,2,135,85
"Balikpapan Selatan,","2,80M",3,2,126,135
"Balikpapan Selatan,","2,25M",3,3,120,150
"Balikpapan Utara,","3,40M",5,4,190,300
"Balikpapan Kota,","2,20M",4,4,350,250
"Balikpapan Selatan,","2,50M",4,3,340,200
"Balikpapan Kota,",Rp 15M,67,6,660,750
"Balikpapan Selatan,","2,90M",4,2,220,200
"Balikpapan Selatan,","1,60M",3,3,135,200
"Balikpapan Selatan,","2,55M",4,3,165,250
"Balikpapan Utara,","3,50M",10,6,418,600
"Balikpapan Selatan,","3,50M",4,4,246,300
"Balikpapan Utara,",455Jt,2,1,97,39
"Balikpapan Tengah,",895Jt,4,2,325,240
"Balikpapan Selatan,","2,50M",4,3,419,250
"Balikpapan Utara,",385Jt,2,1,80,45
"Balikpapan Utara,",495Jt,2,1,110,42
"Balikpapan Utara,",550Jt,3,1,120,50
"Balikpapan Utara,","1,10M",2,2,450,114
"Balikpapan Utara,",275Jt,2,1,121,36
"Balikpapan Utara,","1,50M",4,3,170,120
"Balikpapan Utara,",430Jt,2,1,87,36
"Balikpapan Utara,",395Jt,3,1,120,50
"Balikpapan Utara,",500Jt,2,1,78,45
"Balikpapan Selatan,",500Jt,2,1,78,45
"Balikpapan Utara,",395Jt,2,1,96,36
"Balikpapan Selatan,","2,50M",11,10,375,260
"Balikpapan Selatan,","459,25Jt",2,1,102,36
"Balikpapan Utara,","383,68Jt",2,1,107,36
"Balikpapan Selatan,",690Jt,3,2,90,75
"Balikpapan Utara,",475Jt,2,1,120,45
"Balikpapan Timur,",385Jt,2,1,84,36
"Balikpapan Utara,",250Jt,2,1,160,36
"Balikpapan Selatan,",650Jt,2,1,119,119
"Balikpapan Selatan,","2,70M",4,3,180,210
"Balikpapan Selatan,","4,20M",3,4,550,550
"Balikpapan Selatan,","1,60M",3,2,216,100
"Balikpapan Tengah,",Rp 2M,3,2,763,280
"Balikpapan Selatan,",950Jt,3,2,150,120
"Balikpapan Kota,","1,50M",3,2,462,235
"Balikpapan Selatan,","1,10M",2,2,90,90
"Balikpapan Selatan,","3,30M",4,3,147,240
"Balikpapan Selatan,","1,15M",2,2,78,47
"Balikpapan Selatan,",950Jt,3,2,127,120
"""

df = pd.read_csv(io.StringIO(csv_content), sep=',', quotechar='"')

# 2. DATA CLEANING
def clean_price(price_str):
    if pd.isna(price_str) or price_str is None:
        return None
    
    price_str = str(price_str).strip().replace('Rp', '').replace(' ', '')
    
    if 'M' in price_str:
        num_part = price_str.replace('M', '').replace(',', '.') 
        try: return float(num_part) * 10**9
        except ValueError: return None
    elif 'Jt' in price_str:
        num_part = price_str.replace('Jt', '').replace(',', '.')
        try: return float(num_part) * 10**6
        except ValueError: return None
    return None

df['Harga (IDR)'] = df['harga'].apply(clean_price)
df['subdistrict'] = df['daerah'].str.replace(',', '').str.strip()

# Konversi kolom numerik ke tipe numerik
df['Kamar Tidur'] = pd.to_numeric(df['kamar_tidur'], errors='coerce')
df['Kamar Mandi'] = pd.to_numeric(df['kamar_mandi'], errors='coerce')
df['Luas Tanah (m²)'] = pd.to_numeric(df['luas_tanah_m²'], errors='coerce')
df['Luas Bangunan (m²)'] = pd.to_numeric(df['luas_bangunan_m²'], errors='coerce')

# Filter baris yang memiliki nilai NaN pada semua kolom utama yang dibutuhkan model
df_filtered = df.dropna(subset=['Harga (IDR)', 'Luas Tanah (m²)', 'Kamar Tidur', 'Kamar Mandi', 'Luas Bangunan (m²)'])
df_filtered = df_filtered.copy() # Salin untuk menghindari SettingWithCopyWarning

# 3. FEATURE ENGINEERING: ONE-HOT ENCODING
df_dummies = pd.get_dummies(df_filtered['subdistrict'], prefix='Daerah', dtype=int)

# 4. FINAL ASSEMBLY
# Menggabungkan semua fitur numerik dan dummy variables
model_df = pd.concat([
    df_filtered[['Luas Tanah (m²)', 'Luas Bangunan (m²)', 'Kamar Tidur', 'Kamar Mandi']],
    df_dummies,
    df_filtered['Harga (IDR)'] # Variabel Target
], axis=1)

# Mengatur urutan kolom agar Harga (Target) berada di paling kanan
cols = [col for col in model_df.columns if col != 'Harga (IDR)'] + ['Harga (IDR)']
model_df = model_df[cols]

print(f"Data final siap model: {len(model_df)} baris.")
print("\n--- Model-Ready Dataset (Input untuk Regresi Linear) ---")
print(model_df.to_string())

```
Penjelasan :
Cell 3: Import Library & Data Input
Import pandas, numpy, io, matplotlib.pyplot. Define data properti dalam format CSV string (6 kolom: daerah, harga, kamar_tidur, kamar_mandi, luas_tanah_m², luas_bangunan_m²). Parse CSV string ke DataFrame menggunakan pd.read_csv() dengan io.StringIO().

```python
4  pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
     model_df.describe()

```
Penjelasan :
Cell 4: Mengatur opsi tampilan pandas agar semua angka floating point ditampilkan dengan format 6 desimal dan pemisah ribuan koma untuk readability yang lebih baik. Kemudian menjalankan method describe() pada model_df yang akan menghasilkan statistik ringkas untuk semua kolom numerik termasuk jumlah data, nilai rata-rata, standar deviasi, nilai minimum, nilai median (50%), nilai maksimum, serta quartile pertama (25%) dan ketiga (75%)

```python
5    # Kolom numerik yang ingin diperiksa outlier-nya
fitur_outlier = ['Luas Tanah (m²)', 'Luas Bangunan (m²)', 'Kamar Tidur', 'Kamar Mandi', 'Harga (IDR)']

# List untuk menyimpan baris outlier dari semua kolom
all_outliers = []

# Loop untuk tiap kolom numerik
for col in fitur_outlier:
    Q1 = model_df[col].quantile(0.25)
    Q3 = model_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Cari baris outlier pada kolom tersebut
    mask = (model_df[col] < lower) | (model_df[col] > upper)
    outlier_rows = model_df[mask].copy()

    # Beri informasi outliernya dari kolom mana
    outlier_rows['outlier_in'] = col

    # Simpan ke list
    all_outliers.append(outlier_rows)

# Gabungkan semua outlier dari semua kolom
outliers_combined = pd.concat(all_outliers)

# Hilangkan baris duplikat berdasarkan index supaya 1 baris tidak muncul dobel
outliers_combined = outliers_combined[~outliers_combined.index.duplicated(keep='first')]

print(f"Jumlah outlier unik (sebelum dihapus): {len(outliers_combined)}")
display(outliers_combined)

# === Buat DataFrame bersih tanpa outlier ===
df_clean = model_df.drop(index=outliers_combined.index).copy()

print(f"\nJumlah data sebelum cleaning: {len(model_df)}")
print(f"Jumlah data setelah cleaning : {len(df_clean)}")
print(f"Jumlah baris dihapus        : {len(outliers_combined)}")

```
Penjelasan :
Cell 5: Mendefinisikan 5 kolom numerik untuk deteksi outlier menggunakan IQR method. Loop setiap kolom untuk hitung Q1, Q3, dan IQR, kemudian tentukan lower bound (Q1 - 1.5IQR) dan upper bound (Q3 + 1.5IQR). Gunakan boolean masking untuk identifikasi dan simpan semua baris outlier ke list, lalu gabungkan dengan pd.concat() dan hilangkan duplikasi berdasarkan index. Terakhir drop baris outlier dari model_df untuk membuat df_clean dan tampilkan perbandingan jumlah data sebelum-sesudah cleaning.

```python
6    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
df_clean.describe()

```
Penjelasan :
Cell 6: Mengatur opsi tampilan pandas agar angka floating point ditampilkan dengan 6 desimal dan pemisah ribuan koma untuk readability lebih baik. Kemudian menjalankan method describe() pada df_clean (data yang sudah dibersihkan dari outlier) untuk menampilkan statistik ringkas meliputi count, mean, std, min, 25%, 50%, 75%, dan max untuk semua kolom numerik. Output ini membantu memverifikasi bahwa outlier sudah berhasil dihapus dengan melihat perubahan nilai statistik, range data, dan memastikan data siap untuk training model.

```python
7    plt.figure(figsize=(8,5))
plt.scatter(df_clean['Kamar Tidur'], df_clean['Harga (IDR)'], marker='+')
plt.title('Korelasi antara Jumlah Kamar Tidur dengan Harga Rumah')
plt.xlabel('Jumlah Kamar Tidur')
plt.ylabel('Harga Rumah (IDR)')
plt.grid(True, alpha=0.3)
plt.show()

```
Penjelasan :
Cell 7: Membuat figure dengan ukuran 8x5 inch untuk visualisasi scatter plot. Membuat scatter plot dengan x-axis berupa jumlah kamar tidur dan y-axis berupa harga rumah, menggunakan marker '+' untuk menandai setiap data point. Menambahkan judul, label untuk x-axis dan y-axis agar jelas apa yang divisualisasikan. Menambahkan grid dengan transparency 0.3 untuk memudahkan pembacaan nilai pada plot. Terakhir menampilkan plot dengan plt.show() untuk melihat hubungan/korelasi antara jumlah kamar tidur dan harga rumah secara visual.

```python
8    # Hitung nilai korelasi
corr_value = df_clean[['Kamar Tidur', 'Harga (IDR)']].corr().iloc[0,1]
print("Korelasi antara Jumlah Kamar Tidur dan Harga Rumah =", corr_value)

plt.figure(figsize=(8,5))
plt.scatter(df_clean['Kamar Mandi'], df_clean['Harga (IDR)'], marker='+')
plt.title('Korelasi antara Jumlah Kamar Mandi dengan Harga Rumah')
plt.xlabel('Jumlah Kamar Mandi')
plt.ylabel('Harga Rumah (IDR)')
plt.grid(True, alpha=0.3)
plt.show()

# Hitung nilai korelasi
corr_value = df_clean[['Kamar Mandi', 'Harga (IDR)']].corr().iloc[0,1]
print("Korelasi antara Jumlah Kamar Mandi dan Harga Rumah =", corr_value)
```
Penjelasan :
Cell 8: Hitung korelasi antara kamar tidur dan harga dengan corr().iloc. Buat scatter plot 8x5 inch dengan title, label, dan grid. Print nilai korelasi. Ulangi untuk kamar mandi dan harga. Scatter plot dan nilai korelasi menunjukkan kekuatan hubungan linear antara fitur dan harga (mendekati 1 = hubungan positif kuat, -1 = negatif kuat, 0 = tidak ada hubungan)

```python
9    plt.figure(figsize=(8,5))
plt.scatter(df_clean['Luas Bangunan (m²)'], df_clean['Harga (IDR)'], marker='+')
plt.title('Korelasi antara Luas Bangunan (m²) dengan Harga Rumah')
plt.xlabel('Luas Bangunan (m²)')
plt.ylabel('Harga Rumah (IDR)')
plt.grid(True, alpha=0.3)
plt.show()

# Hitung nilai korelasi
corr_value = df_clean[['Luas Bangunan (m²)', 'Harga (IDR)']].corr().iloc[0,1]
print("Korelasi antara Luas Bangunan (m²) dan Harga Rumah =", corr_value)
```
Penjelasan :
Cell 9: Buat scatter plot 8x5 inch untuk visualisasi hubungan antara luas bangunan dan harga rumah dengan marker '+'. Tambahkan title, xlabel, ylabel, dan grid untuk readability. Hitung nilai korelasi antara luas bangunan dan harga menggunakan corr().iloc. Print nilai korelasi untuk menunjukkan seberapa kuat hubungan linear antara kedua variabel tersebut.

```python
10   # Filter manual untuk menghapus outlier ekstrem
df_clean = model_df[
    (model_df["Luas Tanah (m²)"] <= 350) &
    (model_df["Luas Bangunan (m²)"] <= 355) &
    (model_df["Harga (IDR)"] <= 4_500_000_000)
].copy()

print(f"Jumlah data sebelum cleaning: {len(model_df)}")
print(f"Jumlah data setelah cleaning : {len(df_clean)}")
print(f"Jumlah baris dihapus        : {len(model_df) - len(df_clean)}")
```
Penjelasan :
Cell 10: Filter manual untuk menghapus outlier ekstrem dengan kondisi luas tanah maksimal 350 m², luas bangunan maksimal 355 m², dan harga maksimal 4.5 miliar rupiah menggunakan boolean indexing. Copy hasil filter ke dataframe baru df_clean. Print perbandingan jumlah data sebelum dan sesudah cleaning untuk melihat berapa banyak baris outlier yang berhasil dihapus.

```python
11   pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
df_clean.describe()
```
Penjelasan :
Cell 11 : Atur opsi tampilan pandas agar angka floating point ditampilkan dengan 6 desimal dan pemisah ribuan koma. Jalankan describe() pada df_clean untuk menampilkan statistik ringkas (count, mean, std, min, 25%, 50%, 75%, max) setiap kolom dan verifikasi bahwa outlier ekstrem sudah berhasil dihapus dengan melihat perubahan range nilai data.

```python
12   plt.figure(figsize=(8,5))
plt.scatter(df_clean['Luas Tanah (m²)'], df_clean['Harga (IDR)'], marker='+')
plt.title('Korelasi antara Luas Tanah (m²) dengan Harga Rumah')
plt.xlabel('Luas Tanah (m²)')
plt.ylabel('Harga Rumah (IDR)')
plt.grid(True, alpha=0.3)
plt.show()

# Hitung nilai korelasi
corr_value = df_clean[['Luas Tanah (m²)', 'Harga (IDR)']].corr().iloc[0,1]
print("Korelasi antara Luas Tanah (m²) dan Harga Rumah =", corr_value)
```
Penjelasan :
Cell 12 : Buat scatter plot 8x5 inch untuk visualisasi hubungan antara luas tanah dan harga rumah dengan marker '+'. Tambahkan title, xlabel, ylabel, dan grid untuk readability. Hitung nilai korelasi antara luas tanah dan harga menggunakan corr().iloc. Print nilai korelasi untuk menunjukkan kekuatan hubungan linear antara luas tanah dan harga rumah

```python
13   plt.figure(figsize=(8,5))
plt.scatter(df_clean['Luas Bangunan (m²)'], df_clean['Harga (IDR)'], marker='+')
plt.title('Korelasi antara Luas Bangunan (m²) dengan Harga Rumah')
plt.xlabel('Luas Bangunan (m²)')
plt.ylabel('Harga Rumah (IDR)')
plt.grid(True, alpha=0.3)
plt.show()

# Hitung nilai korelasi
corr_value = df_clean[['Luas Bangunan (m²)', 'Harga (IDR)']].corr().iloc[0,1]
print("Korelasi antara Luas Bangunan (m²) dan Harga Rumah =", corr_value)

```
Penjelasan:
Cell 13 : Buat scatter plot 8x5 inch untuk visualisasi hubungan antara luas tanah dan harga rumah dengan marker '+'. Tambahkan title, xlabel, ylabel, dan grid untuk readability. Hitung nilai korelasi antara luas bangunan dan harga menggunakan corr().iloc. Print nilai korelasi untuk menunjukkan kekuatan hubungan linear antara luas tanah dan harga rumah

```python
14   pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}") # format angka dengan 2 desimal

# Hapus indeks lama dan buat indeks baru (0, 1, 2, ...)
df_clean_reset = df_clean.reset_index(drop=True)

# Tampilkan seluruh DataFrame bersih
print(f"Jumlah baris setelah cleaning: {len(df_clean_reset)}")
display(df_clean_reset)

# Simpan ke CSV
# Gunakan DataFrame yang sudah di-reset indexnya
df_clean_reset.to_csv("data_final_bersih.csv", index=False)
print("Data final bersih dari outlier telah disimpan ke 'data_final_bersih.csv'.")
```
Penjelasan:
Cell 14 : Atur opsi pandas untuk menampilkan semua baris dan kolom tanpa truncation, serta format angka dengan 2 desimal. Reset index dataframe menggunakan reset_index(drop=True) untuk membuat index baru yang sequential (0, 1, 2, ...). Tampilkan jumlah baris dan seluruh dataframe yang sudah bersih dari outlier. Simpan dataframe ke file CSV bernama "data_final_bersih.csv" tanpa menyertakan index menggunakan to_csv() dengan index=False. Print pesan konfirmasi bahwa data bersih telah berhasil disimpan ke file.

model/model_training.ipynb
```python
1  import sys
    print(sys.executable)
```
Penjelasan:
Cell 1: Baris ini mengimpor modul standar Python bernama sys. Modul sys menyediakan akses ke beberapa variabel dan fungsi yang berinteraksi dengan interpreter Python.

```python
2  from sklearn.model_selection import train_test_split,  GridSearchCV
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import os
```
Penjelasan:
Cell 2: 
- Pandas: Diimpor sebagai tool utama untuk manajemen data, digunakan untuk membersihkan, memfilter, dan menyusun data set ke dalam format DataFrame yang terstruktur.
- NumPy: Diimpor untuk operasi numerik tingkat lanjut, yang sangat penting untuk menerapkan fungsi eksponensial (np.exp) dan logaritma (np.log) selama proses Log Transformation dan Inverse Transformation pada variabel harga.
- train_test_split: Digunakan untuk membagi dataset akhir menjadi set pelatihan dan pengujian, yang merupakan langkah kritis untuk memvalidasi performa model secara objektif dan menghindari overfitting.
- LinearRegression: Ini adalah model utama yang diimpor dan akan digunakan untuk melatih model regresi linier pada data harga.
- R2 Score: Diimpor untuk mengukur daya jelas model—seberapa baik fitur-fitur dapat menjelaskan variasi harga.
- Mean Absolute Error (MAE): Diimpor untuk mengukur rata-rata kesalahan prediksi dalam mata uang (Rupiah). Metrik inilah yang kemudian Anda gunakan untuk menentukan dan menyajikan Rentang Harga Prediksi.

```python
3 pd.set_option("display.max_rows", None)

   # Baca CSV (hasil cleaning / dataset final)
   df = pd.read_csv('data_final_bersih.csv')  # ganti sesuai nama file CSV-mu
   print("Jumlah data:", len(df))
   print("Kolom:")
   print(df.columns.tolist())

   # Tampilkan seluruh data
   df
```
Penjelasan:
Cell 3: Tahap verifikasi dan pemuatan data yang  diawali dengan pengaturan Pandas melalui pd.set_option("display.max_rows", None) untuk memastikan semua baris DataFrame ditampilkan tanpa terpotong selama inspeksi visual. Kemudian, data final yang sudah dibersihkan (data_final_bersih.csv) dimuat ke dalam variabel df menggunakan pd.read_csv() untuk digunakan dalam pemodelan. Proses ini dilanjutkan dengan pencetakan jumlah total baris (len(df)) dan daftar lengkap kolom (df.columns.tolist()), berfungsi sebagai langkah verifikasi struktural terakhir untuk mengonfirmasi bahwa kuantitas data dan urutan fitur telah sesuai dengan yang diharapkan model. Terakhir, df ditampilkan untuk memungkinkan inspeksi visual mendalam terhadap dataset yang akan dilatih.

```python
[5] target = "Harga (IDR)"

     X = df.drop(columns=[target])
     y = df[target]

     X_train, X_test, y_train, y_test = train_test_split(
     X, y,
     test_size=0.2,
     random_state=42
)

     baseline_model = LinearRegression()
     baseline_model.fit(X_train, y_train)

     y_pred = baseline_model.predict(X_test)
     r2_baseline = r2_score(y_test, y_pred)

    print("R² Baseline =", round(r2_baseline, 3))

```
Penjelasan:
Cell 4: Proses dimulai dengan mendefinisikan variabel target "Harga (IDR)" dan memisahkan data menjadi fitur (X)dan target (Y). Data kemudian dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan train_test_split dengan random state yang tetap untuk memastikan reproduktifitas. Sebuah model LinearRegression dasar (baseline) dilatih menggunakan data harga asli yang belum ditransformasi, dan kinerjanya dievaluasi menggunakan R2 score. Hasilnya, R2 Baseline = 0.507, menunjukkan bahwa model linier dasar mampu menjelaskan 50.7% variasi harga, menjadi patokan yang harus dilampaui melalui langkah-langkah optimasi selanjutnya seperti Log Transformation dan Feature Engineering.

```python
[6] y_log = np.log(y)
```
Penjelasan:
Cell 6: fungsi Logaritma Natural (np.log) dari NumPy untuk mengubah skala variabel target y, yaitu Harga (IDR)) dan menyimpannya sebagai y log. Tujuan utamanya adalah untuk mengatasi distribusi harga yang cenderung miring dan non-linear, yang melanggar asumsi dasar Regresi Linier. Dengan transformasi ini, distribusi harga menjadi lebih mendekati normal dan hubungan antara fitur dan target menjadi lebih linear.

```python
[7] X_train, X_test, y_train_log, y_test_log = train_test_split(
     X, y_log,
     test_size=0.2,
     random_state=42
)
```
Penjelasan:
Cell 7: Fungsi train_test_split digunakan untuk membagi fitur X (semua variabel independen) dan target yang sudah ditransformasi y log (ln(Harga)) menjadi set pelatihan (X\_train, y train_log) dan set pengujian (X\_test, y test_log) dengan rasio 80% : 20%. Pembagian ini harus dilakukan setelah transformasi log untuk memastikan bahwa model hanya dilatih pada data logaritmik dan diuji pada data logaritmik yang belum pernah dilihat sebelumnya, sehingga metrik kinerja model yang dihasilkan (generalization score) tidak bias dan tetap dapat diulang (reproduktif) berkat penggunaan random_state = 42.

```python
[8] log_model = LinearRegression()
     log_model.fit(X_train, y_train_log)

     y_pred_log = log_model.predict(X_test)
     r2_log = r2_score(y_test_log, y_pred_log)

     print("R² Setelah Log Transform =", round(r2_log, 3))
```
Penjelasan:
Cell 8:  Melanjutkan proses training dengan memanfaatkan transformasi log yang telah diterapkan, bertujuan untuk meningkatkan kinerja model dibandingkan baseline. Sebuah model LinearRegression baru diinisialisasi (log\_model) dan kemudian dilatih (.fit) menggunakan fitur pelatihan x train dan variabel target yang sudah di-log y train_log. Setelah pelatihan, model tersebut digunakan untuk memprediksi log harga pada set pengujian y pred_log, dan kinerjanya dievaluasi menggunakan R2 score. Hasilnya, R2 setelah Log Transform =0.517 menunjukkan adanya peningkatan akurasi dibandingkan R² Baseline 0.507
```python
[9] df_fe = df.copy()

     # fitur turunan yang masuk akal secara properti
     df_fe["harga_per_m2_tanah"] = df_fe["Harga (IDR)"] / df_fe["Luas Tanah (m²)"]
     df_fe["rasio_lb_lt"] = df_fe["Luas Bangunan (m²)"] / df_fe["Luas Tanah (m²)"]
```
Penjelasan:
Cell 9: Impelementasi Feature Engineering (FE) dengan membuat dua fitur turunan baru yang relevan dengan penilaian properti. Pertama, harga_per_m2_tanah dihitung dengan membagi Harga(IDR) dengan Luas Tanah, yang meskipun merupakan variabel target di dalam fitur, ini adalah strategi umum untuk membantu Regresi Linier memahami nilai relatif. Kedua, rasio_lb_lt (Rasio luas bangunan terhadap luas Tanah dihitung untuk mengukur kepadatan pembangunan atau pemanfaatan lahan. Penambahan dua fitur ini sangat efektif dalam optimasi model Anda, karena model Regresi Linier selanjutnya (yang menggunakan FE ini) menunjukkan peningkatan kinerja yang signifikan, mencapai R2 Score 0.677 (dari 0.517 tanpa FE).

```python
[10] X_fe = df_fe.drop(columns=["Harga (IDR)"])
       y_fe_log = np.log(df_fe["Harga (IDR)"])

       X_train, X_test, y_train, y_test = train_test_split(
       X_fe, y_fe_log,
       test_size=0.2,
        random_state=42
)
```
Penjelasan:
Cell 10: Menyelesaikan persiapan dataset final untuk pelatihan model yang optimal, mengintegrasikan hasil Feature Engineering dengan Log Transformation. Pertama, fitur akhir X fe, yang kini mencakup 12 variabel termasuk rasio FE dan dummy daerah, dipisahkan dari variabel target. Variabel target (Harga (IDR)) sekali lagi diubah skalanya menggunakan Logaritma Natural (np log) untuk mendapatkan y fe_log. Terakhir, fungsi train_test_split digunakan untuk membagi X fe dan y fe_log menjadi set pelatihan dan pengujian (80% : 20%), memastikan produktivitas pembagian data dengan random_state=42. Subset data yang telah ditransformasi dan direkayasa ini siap untuk melatih model regresi linier akhir yang secara historis menghasilkan kinerja R2 tertinggi (0.677).

```python
[11] final_model = LinearRegression()
      final_model.fit(X_train, y_train)

      y_pred = final_model.predict(X_test)
      r2_final = r2_score(y_test, y_pred)

       print("R² Final (Log + Feature Engineering) =", round(r2_final, 3))
```
Penjelasan:
Cell 11: Model Regresi Linier Akhir dilatih dan dievaluasi menggunakan dataset yang telah dioptimasi. Sebuah instance baru dari LinearRegression diinisialisasi sebagai final_model dan dilatih (.fit) menggunakan set pelatihan final (xtrain dan y train), yang sekarang mencakup Log Transformation pada target dan fitur Feature Engineering yang kaya. Model yang telah dilatih kemudian membuat prediksi pada data uji (y pred) dan kinerjanya diukur menggunakan R2 score. Hasilnya, R2 Final = 0.677, menunjukkan peningkatan dramatis dari R² Baseline 0.507 dan model Log-only 0.517, memvalidasi keberhasilan optimasi data yang telah dilakukan.

```python
[12] y_pred_exp = np.exp(y_pred)
       y_test_exp = np.exp(y_test)

      mae_final = mean_absolute_error(y_test_exp, y_pred_exp)
      print("MAE (Rp):", f"{mae_final:,.0f}")
```
Penjelasan:
Cell 12: Proses dimulai dengan menerapkan fungsi eksponensial (np.exp) yang merupakan operasi kebalikan dari Logaritma Natural, pada y pred (prediksi harga log) dan y test (harga aktual log), menghasilkan y pred_exp dan y test_exp (harga dalam Rupiah). Metrik kunci yang kemudian dihitung adalah Mean Absolute Error (MAE), yaitu rata-rata selisih absolut antara harga aktual Rupiah dan harga prediksi Rupiah. MAE sebesar Rp 473.813.412 ini menjadi metrik yang paling relevan bagi pengguna, karena nilai inilah yang akan digunakan di aplikasi Flask (app.py) sebagai margin kesalahan untuk menentukan rentang harga yang disarankan.

```python
[13] from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
       import numpy as np

      # Prediksi pada data uji (skala log)
      y_pred_log = final_model.predict(X_test)

      # R² pada skala log
      r2 = r2_score(y_test, y_pred_log)

      # Kembalikan ke skala harga asli
      y_pred = np.exp(y_pred_log)
      y_true = np.exp(y_test)

      # Error metrics pada skala asli
      mae = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))

      print("=== Evaluasi Model Regresi Linear ===")
      print(f"R² Score          : {r2:.3f}")
      print(f"MAE (Rp)          : {mae:,.0f}")
      print(f"RMSE (Rp)         : {rmse:,.0f}")
```
Penjelasan: 
Cell 13: Mengonfirmasi hasil yang dicapai dari optimasi Feature Engineering dan Log Transformation. Setelah model final memprediksi harga pada skala log y pred_log pada data uji, metrik R2 dihitung pada skala log untuk mengukur kualitas model (R2 = 0.677). Kemudian, prediksi (y pred_log) dan nilai aktual (y test) dikonversi kembali ke skala Rupiah asli menggunakan np.exp(). Pada skala Rupiah inilah metrik kesalahan utama dihitung: Mean Absolute Error (MAE) sebesar Rp473.813.412 dan Root Mean Squared Error (RMSE) sebesar Rp917.210.564. Metrik ini memberikan gambaran langsung mengenai rata-rata kesalahan prediksi dalam Rupiah.

```python
[14] kolom_fitur = X_train.columns.tolist()

```
Penjelasan:
Cell 14: Baris kolom_fitur = X_train.columns.tolist() mengambil daftar nama kolom secara berurutan dari DataFrame fitur pelatihan x train dan menyimpannya sebagai list dalam variabel kolom_fitur. Variabel ini sangat penting untuk fase deployment (app.py), karena daftar ini, yang berjumlah 12 kolom, akan disimpan secara terpisah dalam file PKL dan digunakan kembali oleh aplikasi Flask untuk memastikan bahwa input data baru dari pengguna diurutkan persis sama dengan data yang digunakan saat model dilatih.


```python
[15] import joblib

       joblib.dump(final_model, "model_regresi_linear_harga_rumah.pkl")
       joblib.dump(kolom_fitur, "kolom_fitur_model.pkl")

       print("Model & kolom fitur berhasil disimpan")
```
Penjelasan:
Cell 15: Fungsi joblib.dump() digunakan dua kali: pertama, untuk menyimpan objek model yang telah dilatih (final\_model) ke dalam file model_regresi_linear_harg_rumah.pkl; dan kedua, untuk menyimpan daftar terurut dari 12 nama kolom fitur (kolom\_fitur) ke dalam file kolom_fitur_model.pkl

````html
   // templates/index.html

 1 <!DOCTYPE html>
 2 <html lang="id">
 3 <head>
 4     <meta charset="UTF-8">
 5     <title>Prediksi Harga Rumah Balikpapan</title>
 6     <style>
 7         body {
 8             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
 9             background-color: #FFF9E5; /* kuning pastel */
10         }
11         .container {
12             width: 480px;
13             margin: 50px auto;
14             background: white;
15             padding: 30px;
16             border-radius: 12px;
17             box-shadow: 0 4px 15px rgba(0,0,0,0.15);
18             border-top: 5px solid  #FFF9E5; /* biru navy */
19         }
20         h2 {
21             text-align: center;
22             color: #001F54;
23             margin-bottom: 25px;
24         }
25         label {
26             font-weight: bold;
27             color: #001F54;
28         }
29         input, select, button {
30             width: 100%;
31             padding: 10px;
32             margin: 8px 0 15px;
33             border-radius: 6px;
34             border: 1px solid #ccc;
35             font-size: 14px;
36         }
37         input::placeholder {
38             color: #888;
39         }
40         button {
41             background-color: #001F54; /* biru navy */
42             color: #FFD966; /* kuning pastel */
43             border: none;
44             font-size: 16px;
45             cursor: pointer;
46             transition: 0.3s;
47         }
48         button:hover {
49             background-color: #004080;
50         }
51         .hasil {
52             margin-top: 25px;
53             padding: 20px;
54             background-color: #FFD96633; /* kuning pastel transparan */
55             border-left: 5px solid #001F54; /* biru navy */
56             border-radius: 8px;
57         }
58         .hasil p {
59             margin: 6px 0;
60         }
61         .hasil strong {
62             color: #001F54;
63         }
64         /* Tambahkan style untuk pesan error */
65         .error {
66             color: #B00020;
67             background-color: #FEEEEE;
68             border: 1px solid #B00020;
69             padding: 10px;
70             border-radius: 6px;
71             margin-bottom: 15px;
72         }
73     </style>
74 </head>
75 <body>
76 
77 <div class="container">
78     <h2>Prediksi Harga Rumah Balikpapan</h2>
79 
80     {# Tampilkan Pesan Error #}
81     {% if error %}
82         <div class="error">{{ error }}</div>
83     {% endif %}
84 
85     <form action="/predict" method="post">
86         <label>Kamar Tidur</label>
87         {# Tambahkan value untuk sticky form #}
88         <input type="number" name="kamar_tidur" placeholder="Masukkan jumlah kamar tidur" required
89                value="{{ kamar_tidur if kamar_tidur is not none }}">
90 
91         <label>Kamar Mandi</label>
92         {# Tambahkan value untuk sticky form #}
93         <input type="number" name="kamar_mandi" placeholder="Masukkan jumlah kamar mandi" required
94                value="{{ kamar_mandi if kamar_mandi is not none }}">
95 
96         <label>Luas Tanah (m²)</label>
97         {# Tambahkan value untuk sticky form #}
98         <input type="number" name="luas_tanah" placeholder="Masukkan luas tanah" required
99                value="{{ luas_tanah if luas_tanah is not none }}">
100 
101         <label>Luas Bangunan (m²)</label>
102         {# Tambahkan value untuk sticky form #}
103         <input type="number" name="luas_bangunan" placeholder="Masukkan luas bangunan" required
104                value="{{ luas_bangunan if luas_bangunan is not none }}">
105 
106         <label>Kecamatan</label>
107         <select name="kecamatan" required>
108             <option value="">Pilih Kecamatan</option>
109             
110             {# Ganti opsi hardcoded dengan loop dinamis #}
111             {% for item in kecamatan_list %}
112                 <option value="{{ item }}" {% if item == kecamatan %}selected{% endif %}>{{ item }}</option>
113             {% endfor %}
114             
115             {# Tambahkan Balikpapan Barat meskipun tidak ada di data training #}
116             {% if 'Balikpapan Barat' not in kecamatan_list %}
117                 <option value="Balikpapan Barat" {% if 'Balikpapan Barat' == kecamatan %}selected{% endif %}>
118                     Balikpapan Barat
119                 </option>
120             {% endif %}
121 
122         </select>
123 
124         <button type="submit">Prediksi</button>
125     </form>
126 
127     {% if range_harga %}
128     <div class="hasil">
129         <p><strong>Hasil Prediksi:</strong></p>
130         <p>Jumlah kamar tidur: <strong>{{ kamar_tidur }}</strong></p>
131         <p>Jumlah kamar mandi: <strong>{{ kamar_mandi }}</strong></p>
132         <p>Luas tanah: <strong>{{ luas_tanah }} m²</strong></p>
133         <p>Luas bangunan: <strong>{{ luas_bangunan }} m²</strong></p>
134         <p>Lokasi: <strong>{{ kecamatan }}</strong></p>
135         
136         {# Tampilkan prediksi tengah #}
137         <p>Prediksi Harga Tengah: <strong>{{ prediksi_tengah }}</strong></p>
138         
139         <p><strong>Rentang Estimasi Harga:</strong> {{ range_harga }}</p>
140     </div>
141     {% endif %}
142 </div>
143 
144 </body>
145 </html>

```

Penjelasan:
1. Struktur HTML
Menggunakan <!DOCTYPE html> untuk HTML5, dengan tag <html lang="id"> agar bahasa halaman dikenali sebagai Bahasa Indonesia, serta pemisahan standar <head> dan <body>.

2. Head 
Tag <meta charset="UTF-8"> memastikan karakter Indonesia tampil dengan benar, dan <title> digunakan sebagai judul tab browser “Prediksi Harga Rumah Balikpapan”.

3. CSS (Style Internal)
Blok <style> mengatur tampilan halaman dengan tema kuning pastel dan biru navy, meliputi:
Warna latar belakang halaman (body)
Desain card .container (shadow, radius, dan border)
Styling judul, label, input, select, dan button
Efek hover pada tombol

4. Container Utama (.container)
Elemen utama berbentuk card di tengah halaman yang menampung seluruh komponen: judul aplikasi, pesan error, form input, dan hasil prediksi.

5. Judul Halaman
<h2> menampilkan judul “Prediksi Harga Rumah Balikpapan” dengan warna biru navy untuk menegaskan identitas aplikasi.

6. Blok Pesan Error (Jinja2)
{% if error %} digunakan untuk menampilkan pesan kesalahan dari backend (misalnya input tidak valid), sehingga pengguna mendapat umpan balik langsung tanpa berpindah halaman.

7. Form Prediksi
<form action="/predict" method="post"> berfungsi mengirim data input pengguna ke backend Flask melalui endpoint /predict menggunakan metode POST.

8. Input Fitur Numerik
Field <input type="number"> untuk:
Jumlah kamar tidur
Jumlah kamar mandi
Luas tanah
Luas Bangunan

9. Luas bangunan
Seluruh field bersifat required dan dilengkapi placeholder serta value dari Jinja2 agar form bersifat sticky (data tidak hilang saat error).

10. Dropdown Kecamatan (Dinamis)
<select name="kecamatan"> menampilkan daftar kecamatan menggunakan loop Jinja2 {% for item in kecamatan_list %}, sehingga opsi selalu sinkron dengan data dari backend dan tetap mempertahankan pilihan user.

11. Tombol Submit
<button type="submit">Prediksi</button> digunakan untuk mengirim form, dengan styling biru navy dan teks kuning pastel serta efek hover untuk meningkatkan pengalaman pengguna.

12. Blok Hasil Prediksi (Jinja2)
{% if range_harga %} memastikan hasil hanya ditampilkan setelah proses prediksi berhasil dilakukan oleh backend.

13. Detail Input Pengguna
Menampilkan kembali data yang dimasukkan pengguna (jumlah kamar, luas tanah, luas bangunan, dan lokasi) agar hasil prediksi lebih mudah dipahami.

14. Output Prediksi Harga
Menampilkan:
Prediksi harga tengah ({{ prediksi_tengah }})
Rentang estimasi harga ({{ range_harga }})

```python
// app.py

 1 | import pandas as pd
 2 | import numpy as np 
 3 | import joblib 
 4 | from flask import Flask, render_template, request
 5 | import os
 6 |
 7 | # KONSTANTA DARI PELATIHAN MODEL
 8 | # MAE digunakan untuk rentang harga: Prediksi ± MAE (Rp 473,813,412)
 9 | MAE_FINAL = 473_813_412 
10 |
11 | # Daftar 12 kolom fitur 
12 | FEATURE_COLUMNS = [
13 |     'Luas Tanah (m²)', 
14 |     'Luas Bangunan (m²)', 
15 |     'Kamar Tidur', 
16 |     'Kamar Mandi',
17 |     'Daerah_Balikpapan Barat', 
18 |     'Daerah_Balikpapan Kota', 
19 |     'Daerah_Balikpapan Selatan', 
20 |     'Daerah_Balikpapan Tengah',
21 |     'Daerah_Balikpapan Timur', 
22 |     'Daerah_Balikpapan Utara', 
23 |     'harga_per_m2_tanah', 
24 |     'rasio_lb_lt'         
25 | ]
26 |
27 | # JALUR FILE
28 | MODEL_PATH = "model/model_regresi_linear_harga_rumah.pkl"
29 | COLUMNS_PATH = "model/kolom_fitur_model.pkl"
30 |
31 | # LOAD MODEL
32 | try:
33 |     with open(MODEL_PATH, "rb") as f:
34 |         model = joblib.load(f) 
35 |     print(f"Model berhasil dimuat dari {MODEL_PATH}")
36 | except FileNotFoundError:
37 |     print(f"ERROR: File model tidak ditemukan di {MODEL_PATH}. Harap periksa kembali jalur Anda.")
38 |     exit()
39 | except Exception as e:
40 |     print(f"ERROR saat memuat model dari {MODEL_PATH}: {e}")
41 |     print("Petunjuk: Coba pastikan versi scikit-learn saat training dan deployment sama.")
42 |     exit()
43 |
44 | # LOAD FEATURE COLUMNS
45 | try:
46 |     with open(COLUMNS_PATH, "rb") as f:
47 |         loaded_feature_names = joblib.load(f)
48 |         # Verifikasi panjang harus 12
49 |         if len(loaded_feature_names) == 12: 
50 |             FEATURE_COLUMNS = loaded_feature_names
51 |             print("Kolom fitur berhasil dimuat dan terverifikasi (12 kolom).")
52 |         else:
53 |             print(f"⚠️ Peringatan: Kolom fitur dari file {COLUMNS_PATH} tidak sesuai (bukan 12 kolom). Menggunakan list hardcoded.")
54 |
55 | except FileNotFoundError:
56 |     print(f"ERROR: File kolom fitur tidak ditemukan di {COLUMNS_PATH}. Menggunakan list hardcoded.")
57 |
58 |
59 | app = Flask(__name__)
60 |
61 | kecamatan_list = [
62 |     'Balikpapan Kota',
63 |     'Balikpapan Selatan',
64 |     'Balikpapan Tengah',
65 |     'Balikpapan Timur',
66 |     'Balikpapan Utara',
67 |     'Balikpapan Barat' 
68 | ]
69 |
70 | # FUNGSI UNTUK MENYIAPKAN INPUT FITUR
71 | def create_input_dataframe_final_model(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan):
72 |     
73 |     # 1. Base features & Feature Engineering
74 |     input_data = {
75 |         'Luas Tanah (m²)': luas_tanah,
76 |         'Luas Bangunan (m²)': luas_bangunan,
77 |         'Kamar Tidur': kamar_tidur,
78 |         'Kamar Mandi': kamar_mandi,
79 |         'rasio_lb_lt': luas_bangunan / luas_tanah,
80 |         # 'harga_per_m2_tanah' diisi 0 karena hanya fitur saat training
81 |         'harga_per_m2_tanah': 0.0 
82 |     }
83 |
84 |     for col in FEATURE_COLUMNS:
85 |         if col.startswith('Daerah_'):
86 |             input_data[col] = 0
87 |             
88 |     daerah_col = f"Daerah_{kecamatan}"
89 |     if daerah_col in FEATURE_COLUMNS:
90 |          input_data[daerah_col] = 1
91 |
92 |     # Buat DataFrame dan pastikan urutan kolom sesuai FEATURE_COLUMNS
93 |     fitur = pd.DataFrame([input_data])
94 |     
95 |     return fitur[FEATURE_COLUMNS]
96 |
97 | @app.route('/')
98 | def index():
99 |     return render_template('index.html', kecamatan_list=kecamatan_list)
100 |
101 |
102 | @app.route('/predict', methods=['POST'])
103 | def predict():
104 |     try:
105 |         kamar_tidur = int(request.form['kamar_tidur'])
106 |         kamar_mandi = int(request.form['kamar_mandi'])
107 |         luas_tanah = float(request.form['luas_tanah'])
108 |         luas_bangunan = float(request.form['luas_bangunan'])
109 |         kecamatan = request.form['kecamatan']
110 |         
111 |         # Validasi batas wajar data training
112 |         if not (1 <= kamar_tidur <= 10 and 1 <= kamar_mandi <= 10 and 30 <= luas_tanah <= 2000 and 20 <= luas_bangunan <= 1500):
113 |              raise ValueError("Input di luar batas wajar data training (cth: KT/KM 1-10, LT 30-2000, LB 20-1500).")
114 |
115 |     except ValueError as ve:
116 |         error_msg = str(ve) if str(ve) else "Input harus berupa angka yang valid."
117 |         return render_template(
118 |             'index.html',
119 |             error=error_msg,
120 |             kecamatan_list=kecamatan_list,
121 |             kamar_tidur=kamar_tidur,
122 |             kamar_mandi=kamar_mandi, 
123 |             luas_tanah=luas_tanah,
124 |             luas_bangunan=luas_bangunan, 
125 |             kecamatan=kecamatan
126 |         )
127 |     except Exception as e:
128 |         return render_template('index.html', error=f"Terjadi kesalahan: {e}", kecamatan_list=kecamatan_list)
129 |
130 |     # 1. Susun fitur & Lakukan Feature Engineering
131 |     try:
132 |         fitur = create_input_dataframe_final_model(
133 |             kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan
134 |         )
135 |     except Exception as e:
136 |          return render_template('index.html', error=f"Error saat menyiapkan fitur: {e}", kecamatan_list=kecamatan_list)
137 |
138 |     # 2. Prediksi harga (skala log) & Konversi ke skala asli
139 |     try:
140 |         # Prediksi menghasilkan ln(Harga)
141 |         log_prediksi = model.predict(fitur)[0]
142 |         
143 |         # Kembalikan ke skala harga asli (Rp)
144 |         prediksi = np.exp(log_prediksi)
145 |     except Exception as e:
146 |         return render_template('index.html', error=f"Error saat prediksi. Cek konsistensi fitur model: {e}", kecamatan_list=kecamatan_list)
147 |
148 |     # 3. RANGE HARGA menggunakan MAE
149 |     bawah = max(0, prediksi - MAE_FINAL) 
150 |     atas = prediksi + MAE_FINAL
151 |     
152 |     # Formatting
153 |     prediksi_tengah = f"Rp{prediksi:,.0f}"
154 |     range_harga = f"Rp{bawah:,.0f} – Rp{atas:,.0f}"
155 |
156 |     return render_template(
157 |         'index.html',
158 |         kamar_tidur=kamar_tidur,
159 |         kamar_mandi=kamar_mandi,
160 |         luas_tanah=luas_tanah,
161 |         luas_bangunan=luas_bangunan,
162 |         kecamatan=kecamatan,
163 |         range_harga=range_harga,
164 |         prediksi_tengah=prediksi_tengah,
165 |         kecamatan_list=kecamatan_list 
166 |     )
167 |
168 | if __name__ == '__main__':
169 |     app.run(debug=True)

```

Penjelasan:
1. Import library
Mengimpor pandas untuk membuat dan mengelola DataFrame fitur, numpy untuk operasi numerik (khususnya konversi dari log ke nilai asli), joblib untuk memuat model machine learning, serta Flask (Flask, render_template, request) untuk membangun aplikasi web dan menangani input dari form pengguna.

2. Konstanta hasil pelatihan model (MAE)
Variabel MAE_FINAL menyimpan nilai Mean Absolute Error (MAE) dari hasil evaluasi model saat training. Nilai ini digunakan untuk membentuk rentang estimasi harga (prediksi ± MAE), sehingga output tidak hanya satu angka tunggal.

3. Definisi kolom fitur (FEATURE_COLUMNS)
List FEATURE_COLUMNS berisi 12 nama fitur yang digunakan oleh model saat training, termasuk:
Fitur numerik (luas tanah, luas bangunan, jumlah kamar)
Fitur kategori hasil one-hot encoding kecamatan
Fitur hasil feature engineering (harga_per_m2_tanah, rasio_lb_lt)
Daftar ini memastikan urutan dan nama kolom input saat prediksi identik dengan data training.

4. Penentuan jalur file model dan kolom fitur
Variabel MODEL_PATH dan COLUMNS_PATH menyimpan lokasi file model regresi linear dan file kolom fitur hasil training agar dapat dimuat kembali saat aplikasi dijalankan.

5. Load model machine learning
Model regresi linear dimuat menggunakan joblib.load.
Blok try-except digunakan untuk:
Menangani error jika file tidak ditemukan
Menangani error perbedaan versi library (misalnya scikit-learn)
Menghentikan aplikasi jika model gagal dimuat

6. Load kolom fitur dari file training
File kolom_fitur_model.pkl dimuat untuk memastikan nama dan jumlah kolom fitur sama dengan saat training. Jika jumlah kolom tidak sesuai (bukan 12), sistem akan menampilkan peringatan dan menggunakan daftar fitur hardcoded sebagai cadangan.

7. Inisialisasi aplikasi Flask
app = Flask(__name__) digunakan untuk membuat instance aplikasi Flask yang akan menangani seluruh request HTTP.

8. Daftar kecamatan
kecamatan_list berisi daftar kecamatan yang digunakan untuk:
Mengisi dropdown pada form HTML
Menentukan kolom one-hot encoding saat preprocessing input user

9. Fungsi create_input_dataframe_final_model()
Fungsi ini bertugas menyiapkan data input user agar sesuai dengan format model, meliputi:
Menyusun fitur dasar (luas, kamar)
Membuat fitur turunan rasio_lb_lt
Mengisi harga_per_m2_tanah dengan nilai 0 (karena hanya digunakan saat training)
Melakukan one-hot encoding kecamatan
Mengembalikan DataFrame dengan urutan kolom sesuai FEATURE_COLUMNS

10. Route "/" (index)
Endpoint utama aplikasi yang merender halaman index.html.
Halaman ini menampilkan form input data rumah dan dropdown kecamatan.

11. Route "/predict" (POST)
Endpoint untuk menangani proses prediksi.
Data input diambil dari request.form, lalu dikonversi ke tipe numerik (int dan float).

12. Validasi input pengguna
Sistem memeriksa apakah nilai input masih berada dalam rentang wajar sesuai data training.
Jika tidak valid, aplikasi akan menampilkan pesan error tanpa menjalankan prediksi.

13. Penyusunan fitur & preprocessing
Data input yang valid diproses menggunakan fungsi create_input_dataframe_final_model() untuk menghasilkan DataFrame fitur yang siap diprediksi oleh model.

14. Prediksi harga menggunakan model
Model menghasilkan prediksi dalam bentuk log harga (ln(harga)), kemudian dikonversi kembali ke skala harga asli (Rupiah) menggunakan fungsi np.exp().

15. Perhitungan rentang harga menggunakan MAE
Rentang estimasi harga dihitung sebagai:
Batas bawah = prediksi − MAE
Batas atas = prediksi + MAE
Nilai bawah dibatasi minimal 0 untuk menghindari harga negatif.

16. Formatting hasil prediksi
Nilai prediksi tengah dan rentang harga diformat ke dalam format Rupiah agar mudah dibaca oleh pengguna.

17. Render hasil ke template
Aplikasi mengembalikan halaman index.html dengan membawa:
Detail input user
Nilai prediksi tengah
Rentang estimasi harga
sehingga hasil prediksi langsung ditampilkan di halaman yang sama.

18. Menjalankan server Flask
app.run(debug=True) digunakan untuk menjalankan aplikasi dalam mode debug, sehingga memudahkan proses pengembangan dan pelacakan error.

## Demo



## Summary

Proyek ini menggunakan metodologi yang berfokus pada optimasi Regresi Linier untuk memprediksi harga rumah, sebuah pendekatan yang krusial karena sifat data harga yang cenderung non-linear. Tahap pelatihan dimulai dengan menyiapkan dataset final yang terdiri dari 309 baris data bersih. Untuk meningkatkan kinerja model, kami menerapkan strategi kunci: pertama, variabel target (`Harga (IDR)`) diubah skalanya menggunakan Logaritma Natural (ln) untuk menstabilkan varians dan meningkatkan linearitas. Kedua, penerapan Feature Engineering dengan menciptakan fitur kontekstual seperti rasio_lb_lt. Kombinasi optimasi ini berhasil meningkatkan akurasi model secara signifikan, yang dibuktikan dengan R2 Score Final 0.677. Setelah model dilatih, metrik kinerjanya dihitung kembali ke skala Rupiah asli, menetapkan nilai Mean Absolute Error (MAE) sebesar Rp473.813.412. Model yang telah dilatih (`final\_model`) dan daftar 12 kolom fitur yang telah terurut, lalu disimpan menggunakan `joblib.dump()` untuk menjamin aset model siap untuk tahap deployment.

Proses prediksi dilakukan dengan memasukkan input yang sudah di-feature engineering ke model, menghasilkan output logaritmik, yang kemudian diubah kembali ke Rupiah menggunakan np.exp(). Hasil akhirnya disajikan kepada pengguna sebagai Rentang Harga Prediksi yang dihitung berdasarkan MAE model, yaitu Harga Prediksi sekitar Rp 473.813.412, memberikan estimasi yang realistis dan terukur kepada pengguna.

## References

A. Vermaysha dan Nurmalitasari, "Prediksi Harga Rumah di Kabupaten Karanganyar Menggunakan Metode Regresi Linear," dalam Prosiding Seminar Nasional Teknologi Informasi dan Bisnis (SENATIB), Surakarta, hlm. 6–1, Jul. 2023.

N. Nuris, "Analisis Prediksi Harga Rumah Pada Machine Learning Menggunakan Metode Regresi Linear," Explore: Jurnal Sistem Informasi dan Telematika, vol. 14, no. 2, hlm. 108–112, Jul. 2024.

R. R. Hallan dan I. N. Fajri, "Prediksi Harga Rumah menggunakan Machine Learning Algoritma Regresi Linier," Jurnal Teknologi Dan Sistem Informasi Bisnis, vol. 7, no. 1, hlm. 57-62, Jan. 2025.




