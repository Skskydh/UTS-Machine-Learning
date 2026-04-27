# Klasifikasi Orange dan Grapefruit dengan Machine Learning

Project ini dibuat sebagai implementasi Ujian Tengah Semester mata kuliah Machine Learning.  
Tujuan utama project ini adalah membangun model klasifikasi untuk membedakan buah **orange** dan **grapefruit** berdasarkan fitur numerik pada dataset.

Dataset yang digunakan adalah dataset **Oranges vs. Grapefruit** dari Kaggle. Dataset ini berisi karakteristik buah seperti diameter, berat, dan nilai warna RGB.

Tiga algoritma machine learning yang digunakan untuk perbandingan adalah:

1. Decision Tree
2. Naive Bayes
3. Support Vector Machine

Hasil evaluasi dari ketiga model dibandingkan menggunakan beberapa metrik, yaitu **accuracy**, **precision**, **recall**, **f1-score**, dan **confusion matrix**.


## Dataset

Dataset memiliki enam kolom utama:

| Kolom | Deskripsi |
|---|---|
| `name` | Jenis buah, yaitu orange atau grapefruit |
| `diameter` | Ukuran diameter buah |
| `weight` | Berat buah |
| `red` | Nilai intensitas warna merah |
| `green` | Nilai intensitas warna hijau |
| `blue` | Nilai intensitas warna biru |

Kolom `name` digunakan sebagai target klasifikasi, sedangkan kolom lainnya digunakan sebagai fitur.


## Struktur Project

```text
orange-grapefruit-classification/
├── data/
│   └── citrus.csv
├── outputs/
│   ├── business_understanding.txt
│   ├── data_understanding_5_data_pertama.csv
│   ├── data_understanding_info_dataset.txt
│   ├── data_understanding_statistik_deskriptif.csv
│   ├── data_understanding_missing_values.csv
│   ├── data_understanding_distribusi_kelas.csv
│   ├── data_understanding_ringkasan.txt
│   ├── data_preparation_info.txt
│   ├── distribusi_kelas.png
│   ├── scatter_diameter_weight.png
│   ├── correlation_matrix.png
│   ├── correlation_matrix.csv
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_naive_bayes.png
│   ├── confusion_matrix_support_vector_machine.png
│   ├── classification_report_decision_tree.txt
│   ├── classification_report_naive_bayes.txt
│   ├── classification_report_support_vector_machine.txt
│   ├── perbandingan_performa_model.png
│   ├── model_evaluation_metrics.csv
│   └── model_evaluation_metrics.txt
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```


# Tahapan CRISP-DM

Project ini menggunakan pendekatan **CRISP-DM** agar proses pembuatan model lebih terstruktur.

Tahapan yang digunakan:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation


## 1. Business Understanding

Permasalahan yang diangkat adalah klasifikasi buah berdasarkan karakteristik numerik.  
Model machine learning dibuat untuk memprediksi apakah suatu data buah termasuk kelas **orange** atau **grapefruit**.

Tujuan dari tahap ini adalah:

- Memahami permasalahan klasifikasi.
- Menentukan target prediksi.
- Menentukan algoritma yang akan dibandingkan.
- Menentukan metrik evaluasi model.

Model yang dibandingkan dalam project ini adalah Decision Tree, Naive Bayes, dan Support Vector Machine.


## 2. Data Understanding

Pada tahap ini dilakukan eksplorasi awal untuk memahami isi dataset.

Proses yang dilakukan meliputi:

- Membaca dataset `citrus.csv`.
- Menampilkan lima data pertama.
- Melihat jumlah baris dan kolom.
- Mengecek tipe data setiap kolom.
- Melihat statistik deskriptif.
- Mengecek missing value.
- Mengecek data duplikat.
- Melihat distribusi kelas target.
- Membuat visualisasi awal.
- Membuat correlation matrix.

Hasil eksplorasi data disimpan secara otomatis di folder `outputs`.

Beberapa file hasil Data Understanding:

```text
data_understanding_5_data_pertama.csv
data_understanding_info_dataset.txt
data_understanding_statistik_deskriptif.csv
data_understanding_missing_values.csv
data_understanding_distribusi_kelas.csv
data_understanding_ringkasan.txt
```


## 3. Data Preparation

Tahap Data Preparation dilakukan agar data siap digunakan untuk training model.

Langkah-langkah yang dilakukan:

1. Membersihkan nama kolom agar seragam.
2. Menghapus data duplikat.
3. Menghapus missing value.
4. Mengubah label teks menjadi angka menggunakan `LabelEncoder`.
5. Memisahkan fitur dan target.
6. Membagi data menjadi data training dan data testing.

Encoding target dilakukan pada kolom `name`.

Contoh encoding:

```text
grapefruit -> 0
orange     -> 1
```

Fitur yang digunakan:

```text
diameter, weight, red, green, blue
```

Pembagian dataset:

```text
80% data training
20% data testing
```

Pembagian data menggunakan `stratify=y`, sehingga proporsi kelas pada data training dan testing tetap seimbang.


## 4. Modeling

Pada tahap Modeling, tiga algoritma machine learning dibuat dan dilatih menggunakan data training.

### 4.1 Decision Tree

Decision Tree bekerja dengan membentuk aturan keputusan dalam bentuk struktur pohon.  
Model ini mudah dipahami karena proses klasifikasinya menyerupai percabangan keputusan.

Pada project ini digunakan:

```python
DecisionTreeClassifier(max_depth=6, random_state=42)
```

Parameter `max_depth=6` digunakan untuk membatasi kedalaman pohon agar model tidak terlalu kompleks.


### 4.2 Naive Bayes

Naive Bayes adalah algoritma berbasis probabilitas.  
Model ini menghitung peluang suatu data masuk ke kelas tertentu berdasarkan fitur yang dimiliki.

Pada project ini digunakan:

```python
GaussianNB()
```

Gaussian Naive Bayes dipilih karena fitur pada dataset berbentuk numerik.


### 4.3 Support Vector Machine

Support Vector Machine bekerja dengan mencari batas pemisah terbaik antar kelas.  
Karena SVM sensitif terhadap skala data, model ini menggunakan `StandardScaler`.

Pipeline SVM yang digunakan:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
])
```

Kernel RBF digunakan agar model dapat menangani pola data yang tidak selalu linear.


## 5. Evaluation

Tahap Evaluation digunakan untuk membandingkan performa setiap model.

Metrik evaluasi yang digunakan:

| Metrik | Fungsi |
|---|---|
| Accuracy | Mengukur persentase prediksi yang benar |
| Precision | Mengukur ketepatan prediksi pada suatu kelas |
| Recall | Mengukur kemampuan model menemukan data dari suatu kelas |
| F1-Score | Menggabungkan precision dan recall |
| Confusion Matrix | Melihat detail prediksi benar dan salah |
| Classification Report | Ringkasan precision, recall, f1-score, dan support |

Hasil evaluasi disimpan pada file:

```text
model_evaluation_metrics.csv
model_evaluation_metrics.txt
```

Classification report setiap model disimpan pada file:

```text
classification_report_decision_tree.txt
classification_report_naive_bayes.txt
classification_report_support_vector_machine.txt
```


## Interpretasi Confusion Matrix

Confusion matrix digunakan untuk melihat prediksi benar dan salah dari model.

Dalam project ini, jika **orange** dianggap sebagai kelas positif, maka interpretasinya adalah:

| Istilah | Arti |
|---|---|
| True Positive | Orange diprediksi sebagai orange |
| True Negative | Grapefruit diprediksi sebagai grapefruit |
| False Positive | Grapefruit salah diprediksi sebagai orange |
| False Negative | Orange salah diprediksi sebagai grapefruit |

Nilai pada diagonal utama menunjukkan prediksi yang benar.  
Nilai di luar diagonal utama menunjukkan prediksi yang salah.


## Visualisasi Hasil

Program menghasilkan beberapa visualisasi yang dapat digunakan untuk memahami data dan hasil model.

### Distribusi Kelas

```text
outputs/distribusi_kelas.png
```

Grafik ini menunjukkan jumlah data pada kelas orange dan grapefruit.

### Scatter Plot Diameter dan Weight

```text
outputs/scatter_diameter_weight.png
```

Grafik ini menunjukkan pola sebaran data berdasarkan diameter dan berat buah.

### Correlation Matrix

```text
outputs/correlation_matrix.png
```

Correlation matrix menunjukkan hubungan antar fitur numerik.

Interpretasi korelasi:

| Nilai Korelasi | Arti |
|---|---|
| Mendekati 1 | Hubungan positif kuat |
| Mendekati -1 | Hubungan negatif kuat |
| Mendekati 0 | Hubungan lemah |

### Confusion Matrix

```text
outputs/confusion_matrix_decision_tree.png
outputs/confusion_matrix_naive_bayes.png
outputs/confusion_matrix_support_vector_machine.png
```

Grafik ini menunjukkan performa prediksi masing-masing model.

### Perbandingan Performa Model

```text
outputs/perbandingan_performa_model.png
```

Grafik ini membandingkan Decision Tree, Naive Bayes, dan SVM berdasarkan accuracy, precision, recall, dan f1-score.


## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/Skskydh/UTS-Machine-Learning.git
```

Ganti `USERNAME` dengan username GitHub kamu.

### 2. Masuk ke Folder Project

```bash
cd orange-grapefruit-classification
```

### 3. Install Library

```bash
python -m pip install -r requirements.txt
```

Jika menggunakan Windows dan perintah di atas tidak berjalan:

```bash
py -m pip install -r requirements.txt
```

### 4. Jalankan Program

```bash
python main.py
```

Alternatif untuk Windows:

```bash
py main.py
```

Setelah program selesai, semua hasil akan tersimpan di folder `outputs`.

## Library yang Digunakan

Library utama yang digunakan:

| Library | Fungsi |
|---|---|
| pandas | Membaca dan mengolah dataset |
| matplotlib | Membuat grafik |
| seaborn | Membuat visualisasi data |
| scikit-learn | Membuat model machine learning dan evaluasi |

Isi `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```


## Ringkasan Alur Program

Alur program pada `main.py`:

1. Membaca dataset dari `data/citrus.csv`.
2. Melakukan validasi kolom dataset.
3. Menyimpan penjelasan business understanding.
4. Melakukan eksplorasi data.
5. Membuat visualisasi data.
6. Membersihkan data.
7. Melakukan encoding label.
8. Membagi data menjadi training dan testing.
9. Melatih model Decision Tree, Naive Bayes, dan SVM.
10. Menghitung metrik evaluasi.
11. Membuat confusion matrix.
12. Membuat grafik perbandingan performa model.
13. Menyimpan seluruh hasil ke folder `outputs`.


## Kesimpulan

Project ini berhasil membangun sistem klasifikasi buah orange dan grapefruit menggunakan tiga algoritma machine learning.

Model yang dibandingkan adalah:

1. Decision Tree
2. Naive Bayes
3. Support Vector Machine

Setiap model dievaluasi menggunakan accuracy, precision, recall, f1-score, confusion matrix, dan classification report.
Model terbaik dapat ditentukan berdasarkan nilai evaluasi pada file `model_evaluation_metrics.csv` serta grafik `perbandingan_performa_model.png`.
Dengan adanya visualisasi dan file output otomatis, hasil analisis menjadi lebih mudah dibaca, dipahami, dan dijelaskan kembali dalam laporan atau presentasi.
  
