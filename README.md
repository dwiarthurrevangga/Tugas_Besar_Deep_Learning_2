# Tugas Besar Deep Learning 2 - Sistem Presensi Otomatis Berbasis Face Recognition

## 📋 Deskripsi Proyek

Proyek ini adalah **sistem presensi otomatis berbasis web** yang memanfaatkan teknologi _Computer Vision_. Sistem ini dirancang untuk mengenali identitas mahasiswa dari foto yang diunggah atau diambil langsung melalui kamera web.

### 🚀 Alur Kerja Sistem (Pipeline)

1.  **Input Citra:** Pengguna mengunggah foto atau menggunakan webcam.
2.  **Deteksi Wajah (MTCNN):** Sistem mendeteksi lokasi wajah (_bounding box_) menggunakan _Multi-task Cascaded Convolutional Networks_ yang tangguh terhadap variasi pose dan _multiple faces_.
3.  **Preprocessing:** Wajah yang terdeteksi di-_crop_ dengan margin tambahan dan di-_resize_ menjadi **224x224 piksel**.
4.  **Pengenalan Wajah (Face Recognition):** Citra wajah yang telah diproses dimasukkan ke model Deep Learning (InceptionResNetV1) untuk memprediksi identitas.
5.  **Output:** Menampilkan _bounding box_ pada wajah, nama mahasiswa, tingkat keyakinan (_confidence_), dan daftar Top-5 kemiripan.

---

## 🧪 Eksperimen Model

Kami membandingkan tiga arsitektur _State-of-the-Art_ untuk menentukan model terbaik pada dataset 70 kelas mahasiswa:

### 1. InceptionResNetV1 (🏆 Best Model)

Model ini dirancang khusus untuk pengenalan wajah dan menggunakan bobot _pretrained_ VGGFace2.

- **Akurasi Validasi:** **87.14%**
- **Pretrained:** VGGFace2
- **Strategi Training:**
  - **Unfreeze:** 2 Blok Terakhir (`mixed_6a`, `repeat_2`, `mixed_7a`, `repeat_3`, `last`, `logits`).
  - **Optimizer:** AdamW (LR Backbone: 1e-4, LR Head: 1e-3, Weight Decay: 1e-3).
  - **Scheduler:** StepLR (step_size=20, gamma=0.5).
  - **Normalization:** _Fixed Image Standardization_ (Standar FaceNet).
  - **Loss:** CrossEntropyLoss dengan Label Smoothing 0.1.

### 2. EfficientNetB0

Model CNN modern yang menyeimbangkan kedalaman, lebar, dan resolusi untuk efisiensi komputasi.

- **Akurasi Validasi:** **74.29%**
- **Pretrained:** ImageNet1K_V1
- **Strategi Training:**
  - **Unfreeze:** 2 Blok Fitur Terakhir (`features.7`, `features.8`) dan _Classifier_.
  - **Optimizer:** AdamW (LR Backbone: 1e-4, LR Head: 1e-3).
  - **Augmentasi:** RandomHorizontalFlip, RandomRotation (15°), ColorJitter.

### 3. DeiT-small (Vision Transformer)

Arsitektur Transformer yang diadaptasi untuk visi komputer dengan strategi pelatihan efisien data.

- **Akurasi Validasi:** **71.43%**
- **Pretrained:** ImageNet
- **Strategi Training:**
  - **Optimizer:** AdamW dengan _Discriminative Learning Rates_ (Layer awal: 1e-5 hingga Head: 5e-4).
  - **Scheduler:** Warmup (5 epochs) + Cosine Annealing.
  - **Loss:** Focal Loss (untuk menangani _class imbalance_).
  - **Augmentasi Berat:** Mixup (alpha=0.2), Random Erasing, Perspective, Affine, Grayscale.

---

## 📊 Hasil Perbandingan

Berikut adalah ringkasan performa ketiga model setelah proses pelatihan pada dataset validasi:

| Model                 | Akurasi Validasi | Macro F1-Score | Weighted F1-Score | Keterangan                                      |
| :-------------------- | :--------------: | :------------: | :---------------: | :---------------------------------------------- |
| **InceptionResNetV1** |    **87.14%**    |   **0.8333**   |    **0.8333**     | **Performa Terbaik** (Domain Wajah)             |
| **EfficientNetB0**    |      74.29%      |     0.6762     |      0.6762       | Efisien namun akurasi moderat                   |
| **DeiT-small**        |      71.43%      |       -        |         -         | _Overfitting_ / Membutuhkan dataset lebih besar |

**Kesimpulan:**
**InceptionResNetV1** dipilih sebagai model utama untuk sistem presensi. Keunggulan ini didapat karena model menggunakan _pretrained weights_ **VGGFace2** yang spesifik untuk fitur wajah manusia, serta strategi _Fixed Image Standardization_ yang sesuai dengan paper FaceNet.

---

## 🛠️ Teknologi & Library

- **Framework:** PyTorch
- **Face Detection:** `facenet-pytorch` (MTCNN & InceptionResnetV1)
- **Transformer:** `timm` (Untuk DeiT)
- **Computer Vision:** `torchvision`, `PIL` (Pillow), `OpenCV`
- **Deployment:** Streamlit
- **Evaluasi:** Scikit-learn

---

## 🚀 Cara Menjalankan Training

1.  **Persiapan Environment:**

    ```bash
    pip install torch torchvision timm facenet-pytorch scikit-learn matplotlib gdown
    ```

2.  **Download Dataset:**
    Gunakan script yang tersedia di notebook untuk mengunduh dataset `train_face224.zip`.

3.  **Jalankan Notebook:**

    - Gunakan `data_preprocessing.ipynb` untuk melihat proses deteksi dan _cropping_ wajah.
    - Jalankan `Inception.ipynb` (atau varian model lainnya) untuk melatih model.

4.  **Load Model (Inference):**

    ```python
    from facenet_pytorch import InceptionResnetV1
    import torch

    # Inisialisasi model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=70)

    # Load bobot terbaik
    checkpoint = torch.load('8714InceptionResnet_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    ```

---

## 🔗 Tautan Penting

- **Aplikasi Web (Demo):** [Face Recognition Presence System](https://facerecognition-blackpanda.streamlit.app/)
- **Poster Proyek:** [Canva Link](https://www.canva.com/design/DAG6O9OSG-Q/fQsuDp6JIuqj2X6EakbbbA/edit?utm_content=DAG6O9OSG-Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- **Paper Referensi:**
  - [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/pdf/1503.03832) (Schroff et al., 2015)
  - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261) (Szegedy et al., 2016)

## 👥 Tim Pengembang

- **Freddy Harahap** (122140018)
- **Dwi Arthur Revangga** (122140144)
