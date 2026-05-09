# Diabetic Retinopathy Classification — Improved Pipeline

Cross-dataset generalisation study for 5-stage diabetic retinopathy (DR) grading.  
Models are trained exclusively on **APTOS 2019** and evaluated zero-shot on **IDRiD** — no fine-tuning, no data leakage.

The core finding: standard accuracy metrics hide catastrophic stage-wise failures under domain shift. This project introduces the **Granularity Gap** — the disproportionate collapse of recall at clinically critical transition stages (Stage 1: Mild, Stage 3: Severe) when the camera changes between datasets.

---

## Results Summary

| Model | APTOS QWK | IDRiD QWK | Stage 1 Recall | Stage 3 Recall |
|---|---|---|---|---|
| Swin-V1 Improved | 0.922 | 0.681 | **96.0%** | **51.6%** |
| Swin-V2 Improved | 0.924 | 0.689 | 76.0% | 39.8% |
| DeiT Improved | 0.903 | **0.745** | 52.0% | 37.6% |
| ResNet-50 Basic | 0.897 | 0.727 | 36.0% | 31.2% |

DeiT achieves the best cross-domain QWK. Swin-V1 achieves the best stage-wise sensitivity.  
Key finding: the same preprocessing pipeline (fundus crop + Focal Loss + Mixup) improves all transformers but *worsens* ResNet-50 — a **preprocessing-architecture interaction effect**.

---

## Project Structure

```
Diabetics-CV/
│
├── main_improved.ipynb          ← main notebook (start here)
├── dataset.py                   ← APTOSDataset and IDRiDDataset classes
│
├── aptos_dataset/               ← YOU MUST CREATE THIS (see Setup below)
│   ├── train.csv
│   └── train_images/
│       ├── 00a3bf413b17.png
│       └── ...
│
├── B. Disease Grading/          ← YOU MUST CREATE THIS (see Setup below)
│   ├── 1. Original Images/
│   │   ├── a. Training Set/
│   │   │   ├── IDRiD_001.jpg
│   │   │   └── ...
│   │   └── b. Testing Set/
│   │       └── ...
│   └── 2. Groundtruths/
│       ├── a. IDRiD_Disease Grading_Training Labels.csv
│       └── b. IDRiD_Disease Grading_Testing Labels.csv
│
├── best_swinv2_improved.pth     ← saved after first training run
├── best_deit_improved.pth       ← saved after first training run
├── best_dino_improved.pth       ← saved after first training run
└── best_swinv1_improved.pth     ← saved after first training run (mainV1.ipynb)
```

> **The `.pth` weight files are not included** (too large for GitHub). The notebook detects whether weights exist and skips training automatically if they do. On first run, training will execute.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/lotanna-okeke/Diabetics-CV.git
cd Diabetics-CV
```

### 2. Install dependencies

Python 3.9+ required.

```bash
pip install torch torchvision timm transformers
pip install opencv-python scikit-learn pandas matplotlib seaborn
pip install pytorch-grad-cam
pip install jupyter
```

On Apple Silicon (M1/M2/M3) the notebook automatically uses the MPS backend — no changes needed.

### 3. Download APTOS 2019 (training data)

Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Download and extract. You only need two things:
- `train_images/` folder (3,661 `.png` retinal fundus images)
- `train.csv` (image IDs and DR grade labels)

Place them in a folder called `aptos_dataset/` in the project root:

```
aptos_dataset/
├── train.csv
└── train_images/
    ├── 00a3bf413b17.png
    └── ...
```

**APTOS class distribution:**

| Stage | Label | Count |
|---|---|---|
| 0 — No DR | 1805 | dominant class |
| 1 — Mild | 370 | |
| 2 — Moderate | 999 | |
| 3 — Severe | 193 | |
| 4 — Proliferative | 294 | |

### 4. Download IDRiD (evaluation data — never used for training)

Go to: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

Download **Task B: Disease Grading**. Extract and place in the project root maintaining the exact folder structure:

```
B. Disease Grading/
├── 1. Original Images/
│   ├── a. Training Set/     ← 413 .jpg images
│   └── b. Testing Set/      ← 103 .jpg images
└── 2. Groundtruths/
    ├── a. IDRiD_Disease Grading_Training Labels.csv
    └── b. IDRiD_Disease Grading_Testing Labels.csv
```

**IDRiD class distribution (training + test combined = 516 images):**

| Stage | Count |
|---|---|
| 0 — No DR | 168 |
| 1 — Mild | 25 |
| 2 — Moderate | 168 |
| 3 — Severe | 93 |
| 4 — Proliferative | 62 |

> IDRiD uses a **different camera** to APTOS — higher contrast, narrower field of view. This is intentional: the domain shift between the two datasets is what this study measures.

---

## Running the Notebook

```bash
jupyter notebook main_improved.ipynb
```

Run all cells top to bottom. The notebook is divided into sections:

| Section | What it does |
|---|---|
| **0 — Imports & Device** | Detects CUDA / MPS / CPU automatically |
| **1 — Helper Functions** | `crop_fundus`, `FocalLoss`, `mixup_batch`, training loop |
| **2 — Data Pipeline** | Loads APTOS (train/val split) and IDRiD (eval only) |
| **3 — Model Training** | Trains Swin-V2, DeiT, DINOv2 with the improved pipeline |
| **4 — Cross-Dataset Comparison** | Stage-wise accuracy table and Granularity Gap bar chart |
| **5 — Analysis** | Dataset distribution stats, confidence analysis, Grad-CAM |

> **To skip re-training:** place the `.pth` weight files in the project root. The training loop detects them and loads weights instead of training.  
> **To force re-training:** delete the relevant `.pth` file and re-run that model's cells.

---

## Key Design Decisions

### Why single-source training?
Models are trained on APTOS only and never see IDRiD during training. This exposes the true domain shift. Most existing work merges datasets, which artificially narrows the gap and hides stage-wise failures.

### Fundus Cropping
Retinal fundus images have black circular borders that vary by camera device. Without cropping, Grad-CAM shows models attending to borders rather than lesions. A contour-detection algorithm (OpenCV `findContours`) crops to the retinal region and square-pads to preserve aspect ratio.

```python
def crop_fundus(image):
    gray = np.array(image.convert('L'))
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image.crop((x, y, x + w, y + h))
```

### Ordinal Encoding
DR stages are ordered (0 < 1 < 2 < 3 < 4). Each label is encoded as a cumulative binary vector:

```
Stage 0 → [0, 0, 0, 0]
Stage 1 → [1, 0, 0, 0]
Stage 2 → [1, 1, 0, 0]
Stage 3 → [1, 1, 1, 0]
Stage 4 → [1, 1, 1, 1]
```

At inference: predicted stage = number of thresholds exceeding 0.5.  
This penalises large ordinal errors more than small ones.

### Focal Loss
APTOS Stage 0 makes up 49% of training images. Focal Loss down-weights easy examples so the model is forced to learn harder minority stages:

```
L_focal = -α (1 - p_t)^γ log(p_t)
```

`α=1.0`, `γ=2.0` applied independently to each of the four ordinal thresholds.

### Mixup Augmentation
Blends pairs of training images and their label vectors:

```
x̃ = λ xᵢ + (1-λ) xⱼ,   λ ~ Beta(0.3, 0.3)
```

This smooths decision boundaries tied to APTOS-specific pixel statistics, improving cross-dataset robustness.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.05 |
| Scheduler | CosineAnnealingLR |
| Epochs | 15 (50 for Swin-V1 in mainV1.ipynb) |
| Batch size | 32 |
| Image size | 224 × 224 |
| Train/val split | 80/20 stratified (seed 78) |

---

## Architectures

| Model | Backbone | Pretrained | Head |
|---|---|---|---|
| **Swin-V1 Tiny** | `microsoft/swin-tiny-patch4-window7-224` | ImageNet | 4-output ordinal |
| **Swin-V2 Tiny** | `microsoft/swinv2-tiny-patch4-window8-256` | ImageNet | 4-output ordinal |
| **DeiT Small** | `deit_small_patch16_224` (timm) | ImageNet | 4-output ordinal |
| **DINOv2 Small** | `facebook/dinov2-small` | Self-supervised | 4-output ordinal |

All transformer models use the full improved pipeline (fundus crop + Focal Loss + Mixup).  
ResNet-50 experiments are in `main.ipynb` (basic) and `mainV1.ipynb` (Swin-V1 extended).

---

## Evaluation Metrics

- **QWK (Quadratic Weighted Kappa)** — primary aggregate metric; penalises large ordinal errors
- **Stage 1 Recall** — sensitivity for Mild DR (only 25 IDRiD images; high variance)
- **Stage 3 Recall** — sensitivity for Severe DR; the critical clinical metric
- **Referable DR Sensitivity** — recall for Stage ≥ 2 (cases requiring clinical referral)
- **Macro F1** — unweighted average F1 across all 5 stages

---

## Citation / Context

This work was produced as part of the Machine Learning Practical (MLP) MSc coursework at the University of Edinburgh (Group G074, 2025–26). The full report is available on request.
