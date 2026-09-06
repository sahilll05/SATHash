# 🛰️ SpectralHashNet

**Self-Supervised Deep Hashing for Multi-Spectral Satellite Image Retrieval**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20GRSL%20Under%20Review-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BigEarthNet--S2-green)](https://bigearth.net/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SpectralHashNet is a self-supervised framework that converts 10-band Sentinel-2 satellite patches into compact **64-bit binary hash codes** — enabling sub-millisecond retrieval from large-scale archives **without any labeled data**.

---

## 🧠 How It Works

```
                ┌──────────────────────────────────┐
                │  Sentinel-2 Patch (10 bands)     │
                │       120 × 120 pixels            │
                └────────────────┬─────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │         SimCLR Augmentation         │
              │  (Spectral jitter + spatial flips)  │
              └──────────┬────────────────┬─────────┘
                         │                │
                   View 1 (z₁)      View 2 (z₂)
                         │                │
              ┌──────────▼────────────────▼─────────┐
              │       4-Stage Residual Encoder       │
              │                                      │
              │  Stage 1: ResBlock + CBAM            │
              │           10 ch → 32 ch (120→60 px)  │
              │  Stage 2: ResBlock + CBAM            │
              │           32 ch → 64 ch  (60→30 px)  │
              │  Stage 3: ResBlock + CBAM            │
              │           64 ch → 128 ch (30→15 px)  │
              │  Stage 4: ResBlock + CBAM            │
              │           128 ch → 256 ch (15→8 px)  │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │           Projector MLP              │
              │    256×8×8 → 256 → 128 (L2-norm)    │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  Composite Loss (Training Only)      │
              │  ① NT-Xent (contrastive)            │
              │  ② Quantization regularizer         │
              │  ③ Bit-independence regularizer     │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │            Hash Layer                │
              │       128 → 64  →  tanh → sign      │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │     64-bit Binary Hash Code          │
              │   e.g.  1011 0011 0100 1101 ...     │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │    FAISS Hamming Search (< 1 ms)     │
              │  Returns Top-K similar patches       │
              └──────────────────────────────────────┘
```

---

## ✨ Key Components

### CBAM — Convolutional Block Attention Module
At every residual stage, CBAM learns **which spectral bands matter** (channel attention) and **where to focus spatially** (spatial attention). This is critical for multi-spectral data where bands like NIR and SWIR carry land-cover information invisible in RGB.

```
Feature Map → Channel Attention → Spatial Attention → Refined Feature Map
```

### Composite Loss
Three losses work together during training to produce high-quality binary codes:

| Loss | What it does |
|---|---|
| **NT-Xent (SimCLR)** | Pulls two augmented views of the same patch together; pushes apart different patches |
| **Quantization loss** | Forces hash logits toward ±1, making `tanh(h) → sign(h)` lossless |
| **Bit-independence loss** | Prevents all bits from being identical; ensures each of the 64 bits carries unique information |

Without the regularizers, the model collapses to outputting the same hash code for everything — a common failure mode in unsupervised hashing. With them, **100% of generated codes are unique**.

---

## 📊 Results on BigEarthNet-S2 (64-bit codes)

| Method | Type | mAP | P@5 | NDCG@10 |
|---|---|---|---|---|
| HashNet | Supervised | 0.580 | 0.640 | 0.595 |
| DHN | Supervised | 0.560 | 0.605 | 0.580 |
| SSDH | Semi-Supervised | 0.608 | 0.660 | 0.625 |
| RS-Hash | Self-Supervised | 0.595 | 0.655 | 0.612 |
| **SpectralHashNet (Ours)** | **Self-Supervised** | **0.726** | **0.786** | **0.783** |

**SpectralHashNet outperforms all baselines including supervised methods — without using a single label during training.**

---

## 🗂️ Repository Structure

```
SATHash/
├── frontend/
│   ├── app.py                        # Streamlit retrieval dashboard
│   ├── generate_test_patches.py      # Generate test patch subsets
│   ├── pick_test_patches.py          # Select query patches for retrieval
│   └── requirements.txt              # Python dependencies
├── models/
│   └── v6/
│       ├── train_improved_v6.py      # Final training script
│       ├── satellite_image_files_v6.json   # File-to-index mapping
│       └── training_history_improved.json  # Loss/metric curves (30 epochs)
├── .gitignore
└── README.md
```

> Model weights (`.pth`), hash matrices (`.npy`), and the dataset are excluded via `.gitignore` due to file size.

---

## ⚙️ Installation

### 1. Clone

```bash
git clone https://github.com/sahilll05/SATHash.git
cd SATHash
```

### 2. Install Dependencies

```bash
pip install -r frontend/requirements.txt
```

### 3. Download the Dataset

Download [BigEarthNet-S2](https://bigearth.net/) and place it at:

```
dataset/big-earth-net/BigEarthNet-S2/
├── train/
├── validation/
└── test/
```

---

## 🚀 Usage

### Train the Model

```bash
python models/v6/train_improved_v6.py
```

Trains for **30 epochs** on **13,683 BigEarthNet-S2 patches** using AMP. The best checkpoint is saved automatically based on mAP.

### Run the Retrieval Dashboard

```bash
cd frontend
streamlit run app.py
```

Upload a `.tif` Sentinel-2 patch → get its 64-bit hash → retrieve the top-K most similar patches from the database in under 1 ms using FAISS Hamming search.

---

## 📄 License

This project is licensed under the MIT License.