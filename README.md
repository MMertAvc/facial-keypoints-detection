# Facial Keypoints Detection with CNN

A Convolutional Neural Network that detects 15 facial keypoints (eyes, eyebrows, nose, mouth corners) from 96×96 grayscale face images, with a Streamlit web application for real-time inference.

> **Important:** `training.csv` (227 MB) and the model files (`best_model.h5`, `model.h5`, 74 MB each) are excluded from this repo due to GitHub's 100 MB file limit. Download the dataset from Kaggle and run the notebook to regenerate the models.

---

## English

### About

This project trains a deep learning model to automatically locate 15 key facial landmarks in grayscale face images. Applications include face alignment, emotion recognition, and augmented reality. The model is served through a Streamlit web app that accepts an uploaded face image and returns the image overlaid with detected keypoints (shown as red dots).

### Features

- Detects 15 facial keypoints: left/right eye center, inner/outer corners, eyebrows, nose tip, mouth corners
- CNN trained on 96×96 grayscale face images
- Streamlit web app: upload a face photo → get keypoints visualized
- Two saved models: `model.h5` (initial) and `best_model.h5` (best checkpoint)

### Dataset

**Source:** [Kaggle — Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection)

| File | Size | Status |
|---|---|---|
| `training.csv` | 227 MB | ❌ Excluded (exceeds GitHub limit) |
| `test.csv` | 57 MB | ❌ Excluded |
| `IdLookupTable.csv` | 0.82 MB | ✅ Included |
| `SampleSubmission.csv` | 0.20 MB | ✅ Included |

> Download `training.csv` and `test.csv` from Kaggle before running the notebook.

### Model Architecture / Tech Stack

```
Input (96, 96, 1) — Grayscale
→ Conv2D (64, 3×3, relu) → MaxPooling2D
→ Conv2D (128, 3×3, relu) → MaxPooling2D
→ Conv2D (256, 3×3, relu) → MaxPooling2D
→ Flatten → Dense(500, relu) → Dropout
→ Dense(30)  ← 15 keypoints × (x, y)
```

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Output:** 30 values (15 × x,y coordinate pairs)

**Tech Stack:** Python · TensorFlow/Keras · NumPy · OpenCV · Pillow · Streamlit

### How to Run

**1. Download dataset from Kaggle**
```bash
kaggle competitions download -c facial-keypoints-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model (generates model.h5 / best_model.h5)**
```bash
jupyter notebook "Facial Keypoints Detection.ipynb"
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

Upload a 96×96 grayscale face image to see the detected keypoints.

### Requirements

```
tensorflow
numpy
Pillow
opencv-python
streamlit
```

---

## Türkçe

### Hakkında

Bu proje, gri tonlamalı yüz görüntülerinde 15 önemli yüz noktasını otomatik olarak bulmak için bir derin öğrenme modeli eğitir. Uygulamaları; yüz hizalama, duygu tanıma ve artırılmış gerçekliği kapsar. Model, yüklenen bir yüz görüntüsünü kabul eden ve tespit edilen noktaları (kırmızı noktalar olarak) görüntü üzerine bindirerek gösteren bir Streamlit web uygulaması aracılığıyla sunulur.

> **Önemli:** `training.csv` (227 MB) ve model dosyaları (`best_model.h5`, `model.h5`, her biri 74 MB) GitHub'ın 100 MB dosya limiti nedeniyle bu repodan hariç tutulmuştur. Veri setini Kaggle'dan indirin ve modelleri yeniden oluşturmak için notebook'u çalıştırın.

### Özellikler

- 15 yüz anahtar noktası tespiti: sol/sağ göz merkezi, iç/dış köşeler, kaşlar, burun ucu, ağız köşeleri
- 96×96 gri tonlamalı yüz görüntüleri üzerinde eğitilmiş CNN
- Streamlit web uygulaması: bir yüz fotoğrafı yükle → anahtar noktaları görselleştirilmiş şekilde gör
- İki kaydedilmiş model: `model.h5` (başlangıç) ve `best_model.h5` (en iyi kontrol noktası)

### Veri Seti

**Kaynak:** [Kaggle — Yüz Anahtar Noktaları Tespiti](https://www.kaggle.com/c/facial-keypoints-detection)

| Dosya | Boyut | Durum |
|---|---|---|
| `training.csv` | 227 MB | ❌ Hariç tutuldu (GitHub limitini aşıyor) |
| `test.csv` | 57 MB | ❌ Hariç tutuldu |
| `IdLookupTable.csv` | 0,82 MB | ✅ Dahil |
| `SampleSubmission.csv` | 0,20 MB | ✅ Dahil |

> Notebook'u çalıştırmadan önce `training.csv` ve `test.csv` dosyalarını Kaggle'dan indirin.

### Model Mimarisi / Teknoloji Yığını

```
Giriş (96, 96, 1) — Gri Tonlamalı
→ Conv2D (64, 3×3, relu) → MaxPooling2D
→ Conv2D (128, 3×3, relu) → MaxPooling2D
→ Conv2D (256, 3×3, relu) → MaxPooling2D
→ Flatten → Dense(500, relu) → Dropout
→ Dense(30)  ← 15 anahtar noktası × (x, y)
```

- **Kayıp:** Ortalama Kare Hata (MSE)
- **Optimizör:** Adam
- **Çıkış:** 30 değer (15 × x,y koordinat çifti)

**Teknoloji Yığını:** Python · TensorFlow/Keras · NumPy · OpenCV · Pillow · Streamlit

### Nasıl Çalıştırılır

**1. Kaggle'dan veri setini indirin**
```bash
kaggle competitions download -c facial-keypoints-detection
```

**2. Bağımlılıkları yükleyin**
```bash
pip install -r requirements.txt
```

**3. Modeli eğitin (model.h5 / best_model.h5 oluşturur)**
```bash
jupyter notebook "Facial Keypoints Detection.ipynb"
```

**4. Streamlit uygulamasını başlatın**
```bash
streamlit run app.py
```

Tespit edilen anahtar noktaları görmek için 96×96 gri tonlamalı bir yüz görüntüsü yükleyin.

### Gereksinimler

```
tensorflow
numpy
Pillow
opencv-python
streamlit
```
