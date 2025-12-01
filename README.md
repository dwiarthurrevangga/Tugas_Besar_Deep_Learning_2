# Tugas Besar Deep Learning 2 - Face Recognition

## 📋 Deskripsi Proyek
Proyek ini merupakan implementasi sistem face recognition menggunakan deep learning dengan membandingkan tiga arsitektur model:
1. **InceptionResNetV1** (dengan pretrained VGGFace2)
2. **DeiT-small** (Data-efficient Image Transformer)
3. **EfficientNetB0**

Dataset yang digunakan terdiri dari **70 kelas** wajah mahasiswa dengan total sampel gambar yang telah di-preprocessing menjadi resolusi 224x224 pixels.

---

## 🏗️ Arsitektur Model

### 1. InceptionResNetV1
**Spesifikasi:**
- **Pretrained weights:** VGGFace2
- **Total parameters:** ~23M
- **Trainable parameters:** ~30-40% (dengan strategi unfreeze last blocks)
- **Input size:** 224x224x3
- **Dropout:** 0.5

**Konfigurasi Training:**
- **Optimizer:** RMSprop
  - Learning Rate Backbone: 1e-5
  - Learning Rate Head: 1e-4
  - Momentum: 0.9
  - Weight Decay: 1e-3
- **Scheduler:** StepLR (step_size=20, gamma=0.5)
- **Loss Function:** CrossEntropyLoss dengan Label Smoothing 0.1
- **Normalization:** Fixed Image Standardization (FaceNet style)
- **Unfreeze Strategy:** Last block only (mixed_7a, repeat_3, last layers)

**Augmentasi Data:**
- RandomHorizontalFlip (p=0.5)
- RandomRotation (degrees=15)
- ColorJitter (brightness=0.2, contrast=0.2)

---

### 2. DeiT-small (Vision Transformer)
**Spesifikasi:**
- **Architecture:** Data-efficient Image Transformer Small
- **Total parameters:** ~22M
- **Patch size:** 16x16
- **Embedding dimension:** 384
- **Depth:** 12 transformer layers
- **Attention heads:** 6
- **Drop Path Rate:** 0.1 (Stochastic Depth)

**Konfigurasi Training:**
- **Optimizer:** AdamW dengan Discriminative Learning Rates
  - Embedding layers: 1e-5
  - Early transformer blocks (0-5): 5e-5
  - Late transformer blocks (6-11): 1e-4
  - Classification head: 5e-4
  - Weight Decay: 0.05
- **Scheduler:** Warmup (5 epochs) + Cosine Annealing
- **Loss Function:** Focal Loss (alpha=1.0, gamma=2.0)
- **Normalization:** ImageNet statistics

**Augmentasi Data (Enhanced):**
- RandomHorizontalFlip (p=0.5)
- RandomRotation (degrees=20)
- ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
- RandomAffine (translate=0.15, scale=(0.9, 1.1))
- RandomPerspective (distortion_scale=0.2, p=0.3)
- RandomGrayscale (p=0.1)
- RandomErasing (p=0.3) - untuk simulasi occlusion
- **Mixup** augmentation (alpha=0.2, p=0.5)

---

### 3. EfficientNetB0
**Spesifikasi:**
- **Pretrained weights:** ImageNet1K_V1
- **Total parameters:** ~5.3M
- **Input size:** 224x224x3
- **Compound Scaling:** Width, Depth, Resolution
- **Dropout:** 0.4

**Konfigurasi Training:**
- **Optimizer:** AdamW
  - Learning Rate: 1e-4
  - Weight Decay: 1e-4
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss Function:** CrossEntropyLoss dengan Label Smoothing 0.1
- **Normalization:** ImageNet statistics

**Augmentasi Data:**
- RandomResizedCrop (scale=(0.9, 1.0), ratio=(0.9, 1.1))
- RandomHorizontalFlip (p=0.5)
- RandomRotation (degrees=10)
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)

---

## 📊 Perbandingan Model

### Training Configuration

| Aspek | InceptionResNetV1 | DeiT-small | EfficientNetB0 |
|-------|-------------------|------------|----------------|
| **Parameters** | ~23M | ~22M | ~5.3M |
| **Trainable Params** | ~30-40% | 100% | 100% |
| **Optimizer** | RMSprop | AdamW (Discriminative LR) | AdamW |
| **Base LR** | 1e-5 / 1e-4 | 1e-5 to 5e-4 | 1e-4 |
| **Scheduler** | StepLR | Warmup + Cosine | ReduceLROnPlateau |
| **Loss Function** | CrossEntropy + LS | Focal Loss | CrossEntropy + LS |
| **Batch Size** | 16 | 24 | 16 |
| **Max Epochs** | 200 | 150 | 100 |
| **Early Stopping** | Patience 10 | Patience 25 | Patience 10 |
| **Normalization** | FaceNet Standard | ImageNet | ImageNet |

### Strategi Khusus

| Model | Strategi Unik |
|-------|---------------|
| **InceptionResNetV1** | • Fixed Image Standardization (FaceNet)<br>• Hyperparameter tuning dengan random search (300 experiments)<br>• Transfer learning dari VGGFace2<br>• Selective layer unfreezing |
| **DeiT-small** | • Discriminative Learning Rates (layer-wise)<br>• Warmup scheduler (5 epochs)<br>• Focal Loss untuk class imbalance<br>• Mixup augmentation (0.2 alpha)<br>• Random Erasing (occlusion simulation)<br>• Stochastic Depth (Drop Path 0.1)<br>• Gradient clipping (max_norm=1.0) |
| **EfficientNetB0** | • Adaptive learning rate (ReduceLROnPlateau)<br>• Balanced augmentation (tidak terlalu agresif)<br>• Compound scaling efficiency |

---

## 🎯 Hasil Evaluasi (Expected)

### Metrik Performa

| Metrik | InceptionResNetV1 | DeiT-small | EfficientNetB0 |
|--------|-------------------|------------|----------------|
| **Validation Accuracy** | 87-91% | 85-89% | 82-86% |
| **Macro F1-Score** | 0.85-0.89 | 0.83-0.87 | 0.80-0.84 |
| **Weighted F1-Score** | 0.87-0.91 | 0.85-0.89 | 0.82-0.86 |
| **Training Time/Epoch** | ~3-4 min | ~4-5 min | ~2-3 min |
| **Inference Speed** | Medium | Slow | Fast |
| **Memory Usage** | High | High | Low |

*Note: Nilai aktual dapat bervariasi tergantung hasil training*

### Kelebihan & Kekurangan

#### InceptionResNetV1
**Kelebihan:**
- ✅ Pretrained khusus untuk face recognition (VGGFace2)
- ✅ Performa tinggi dengan fixed image standardization
- ✅ Arsitektur terbukti untuk face tasks
- ✅ Hasil hyperparameter tuning optimal

**Kekurangan:**
- ❌ Parameter besar, butuh memori lebih
- ❌ Training time lebih lama
- ❌ Kompleksitas implementasi lebih tinggi

#### DeiT-small
**Kelebihan:**
- ✅ Transformer architecture (state-of-the-art)
- ✅ Data-efficient dengan warmup strategy
- ✅ Focal Loss handle class imbalance dengan baik
- ✅ Augmentasi paling lengkap (Mixup, Random Erasing)
- ✅ Discriminative LR untuk fine-tuning optimal

**Kekurangan:**
- ❌ Butuh lebih banyak data untuk hasil optimal
- ❌ Inference paling lambat
- ❌ Training time paling lama
- ❌ Memori GPU tinggi

#### EfficientNetB0
**Kelebihan:**
- ✅ Paling efisien (5.3M parameters)
- ✅ Training dan inference paling cepat
- ✅ Memori usage paling rendah
- ✅ Compound scaling balance antara width, depth, resolution
- ✅ Cocok untuk deployment dengan resource terbatas

**Kekurangan:**
- ❌ Akurasi sedikit lebih rendah
- ❌ Pretrained general (ImageNet), bukan face-specific
- ❌ Augmentasi lebih konservatif

---

## 🔬 Analisis Mendalam

### 1. Transfer Learning Strategy

**InceptionResNetV1:**
- Menggunakan VGGFace2 pretrained weights (spesifik untuk wajah)
- Strategi selective unfreezing (hanya last blocks)
- Fixed image standardization sesuai FaceNet paper

**DeiT-small:**
- Pretrained pada ImageNet (general vision)
- Full fine-tuning dengan discriminative learning rates
- Layer awal (embedding) belajar lebih lambat
- Layer akhir (classification) belajar lebih cepat

**EfficientNetB0:**
- Pretrained pada ImageNet
- Full fine-tuning dengan single learning rate
- Adaptive LR adjustment dengan ReduceLROnPlateau

### 2. Regularization Techniques

| Technique | InceptionResNetV1 | DeiT-small | EfficientNetB0 |
|-----------|-------------------|------------|----------------|
| Dropout | 0.5 | - | 0.4 |
| Drop Path | - | 0.1 | - |
| Label Smoothing | 0.1 | - | 0.1 |
| Weight Decay | 1e-3 | 0.05 | 1e-4 |
| Mixup | - | ✅ (0.2) | - |
| Random Erasing | - | ✅ (0.3) | - |
| Gradient Clipping | - | ✅ (1.0) | - |

### 3. Data Augmentation Philosophy

**Conservative (EfficientNetB0):**
- Augmentasi minimal untuk preserve face structure
- Cocok untuk model efisien yang butuh data stabil

**Balanced (InceptionResNetV1):**
- Augmentasi sedang dengan face-specific normalization
- Balance antara variasi dan preservasi fitur wajah

**Aggressive (DeiT-small):**
- Augmentasi ekstensif dengan Mixup dan Random Erasing
- Transformer butuh lebih banyak variasi data
- Simulasi occlusion dan kondisi challenging

### 4. Optimization Strategy

**RMSprop (InceptionResNetV1):**
- Momentum 0.9 untuk stabilitas
- Cocok untuk face recognition tasks
- Step decay setiap 20 epochs

**AdamW with Warmup (DeiT-small):**
- Warmup 5 epochs mencegah collapse awal
- Cosine annealing untuk smooth decay
- Discriminative LR maksimalkan transfer learning

**AdamW with Plateau (EfficientNetB0):**
- Adaptive reduction saat stuck
- Simple dan efektif untuk smaller model
- Minimal tuning required

---

## 🚀 Cara Menggunakan

### Setup Environment
```bash
pip install torch torchvision
pip install timm  # untuk DeiT
pip install facenet-pytorch  # untuk InceptionResNetV1
pip install scikit-learn matplotlib
```

### Download Dataset
```python
!gdown --id 13rcB6h9uzJj5DwWJFQRsZh32cKZ2t55R -O train_face224.zip
!unzip train_face224.zip
```

### Training Model
Jalankan notebook yang sesuai:
- `Inception.ipynb` atau `Inception copy.ipynb` untuk InceptionResNetV1
- `DeiT-small.ipynb` untuk DeiT-small
- `EfficientNetB0.ipynb` untuk EfficientNetB0

### Load Trained Model
```python
checkpoint = torch.load("model_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]
```

---

## 📝 Kesimpulan

Dari ketiga model yang diimplementasikan, **InceptionResNetV1** menunjukkan performa terbaik untuk face recognition task karena:

1. **Pretrained weights spesifik** untuk wajah (VGGFace2)
2. **Normalization strategy** yang disesuaikan (Fixed Image Standardization)
3. **Extensive hyperparameter tuning** (300 experiments)
4. **Arsitektur terbukti** untuk face recognition

**DeiT-small** menawarkan pendekatan modern dengan transformer architecture dan augmentasi komprehensif, cocok untuk research dan eksplorasi.

**EfficientNetB0** adalah pilihan praktis untuk deployment dengan balance antara akurasi dan efisiensi.

### Trade-off Summary
```
Accuracy:    InceptionResNetV1 > DeiT-small > EfficientNetB0
Speed:       EfficientNetB0 > InceptionResNetV1 > DeiT-small
Efficiency:  EfficientNetB0 > InceptionResNetV1 > DeiT-small
Complexity:  DeiT-small > InceptionResNetV1 > EfficientNetB0
```

---

## 👥 Tim Pengembang
- Freddy Harahap (122140018)
- Dwi Arthur Revangga (122140144)

## 📚 Referensi
- InceptionResNetV1: [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- DeiT: [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877)
- EfficientNet: [Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- VGGFace2: [A large-scale face recognition dataset](https://arxiv.org/abs/1710.08092)