# SpectralHashNet: Self-Supervised Spectral-Spatial Contrastive Hashing for Multi-Spectral Satellite Image Retrieval

[![Paper](https://img.shields.io/badge/Paper-IEEE%20GRSL%20(Under%20Review)-blue)](https://github.com/sahilll05/SATHash)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-BigEarthNet--S2-green)](https://bigearth.net/)

> **Sahil Shashikant Powar** (Student Member, IEEE) and **Pranav Santosh Sukale**  
> Department of Artificial Intelligence and Machine Learning,  
> SIES Graduate School of Technology, Navi Mumbai 400706, India  
> 📧 sahilpowarsp46@gmail.com

---

## Abstract

Content-based retrieval from large-scale multi-spectral satellite archives requires compact binary descriptors that faithfully encode spectral-spatial semantics without relying on costly human annotations. We present **SpectralHashNet**, a fully self-supervised deep hashing framework for Sentinel-2 imagery that simultaneously addresses three fundamental challenges: dependence on labeled data, inadequate spectral-spatial feature fusion, and hash collapse in unsupervised settings.

Evaluated on **13,683 patches** from the publicly available BigEarthNet-S2 dataset, SpectralHashNet achieves:

| Metric | Score |
|---|---|
| **Mean Average Precision (mAP)** | **0.726** |
| **Precision@5 (P@5)** | **0.786** |
| **NDCG@10** | **0.783** |
| **Hash Uniqueness** | **100%** (64-bit codes) |
| **Retrieval Speed** | **< 1 ms** per query (CPU, FAISS) |
| **Storage Reduction** | **99.9%** vs raw features |

SpectralHashNet surpasses supervised baselines (HashNet, DHN, SSDH) and the self-supervised RS-Hash by **+22.0% mAP**, without using any labeled data during training.

---

## Architecture

SpectralHashNet processes 10-band Sentinel-2 patches (120×120 px) through the following pipeline:

```
Input: Sentinel-2 Patch (10 bands, 120×120)
         │
   ┌─────▼─────┐
   │  Stage 1  │  ResidualBlock + CBAM  (10 → 32 ch, 120 → 60 px)
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │  Stage 2  │  ResidualBlock + CBAM  (32 → 64 ch,  60 → 30 px)
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │  Stage 3  │  ResidualBlock + CBAM  (64 → 128 ch, 30 → 15 px)
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │  Stage 4  │  ResidualBlock + CBAM  (128 → 256 ch, 15 → 8 px)
   └─────┬─────┘
         │
   ┌─────▼──────────┐
   │  Projector MLP │  256×8×8 → 256 → 128 (L2-normalized embeddings)
   └─────┬──────────┘
         │
   ┌─────▼──────┐
   │ Hash Layer │  128 → 64  (Tanh → Sign → 64-bit binary code)
   └────────────┘
```

**Key Components:**
- **CBAM (Convolutional Block Attention Module):** Channel + Spatial attention applied at every residual stage to focus on spectral-band importance and spatial regions.
- **Group Normalization:** Replaces Batch Normalization for stability at small batch sizes.
- **SimCLR Contrastive Objective (NT-Xent loss):** Self-supervised encoder training with spectral + spatial augmentations (per-band jitter, flips, rotations).
- **Composite Regularization Loss:** Combines quantization loss and bit-independence loss to provably prevent hash collapse and ensure near-maximum-entropy binary code distribution.

---

## Results

### Comparison with Baselines on BigEarthNet-S2 (64-bit codes)

| Method | Type | mAP | P@5 | NDCG@10 |
|---|---|---|---|---|
| HashNet | Supervised | 0.580 | 0.640 | 0.595 |
| DHN | Supervised | 0.560 | 0.605 | 0.580 |
| SSDH | Semi-Sup. | 0.608 | 0.660 | 0.625 |
| RS-Hash | Self-Sup. | 0.595 | 0.655 | 0.612 |
| **SpectralHashNet (Ours)** | **Self-Sup.** | **0.726** | **0.786** | **0.783** |

### Ablation Study: Loss Components (64-bit)

| Configuration | mAP | P@5 | Hash Uniqueness |
|---|---|---|---|
| No regularization | 0.542 | 0.610 | 22.4% |
| + Quantization loss | 0.615 | 0.685 | 45.2% |
| + Independence loss | 0.631 | 0.702 | 58.6% |
| **Full model (30 ep.)** | **0.726** | **0.786** | **100%** |

---

## Dataset

We use [BigEarthNet-S2](https://bigearth.net/), a large-scale benchmark of Sentinel-2 multi-spectral patches over European countries.

| Split | Patches |
|---|---|
| Train | 7,180 |
| Validation | 3,255 |
| Test | 3,248 |
| **Total** | **13,683** |

Each patch contains **10 spectral bands** (B2, B3, B4, B5, B7, B8, B8A, B11, B12) stored as GeoTIFF files with 120×120 pixels at 10 m ground sampling distance.

> **Note:** The dataset is not included in this repository. Download it from [https://bigearth.net/](https://bigearth.net/) and place it at `dataset/big-earth-net/BigEarthNet-S2/`.

---

## Repository Structure

```
SATHash/
├── frontend/
│   ├── app.py                       # Streamlit retrieval dashboard
│   ├── generate_test_patches.py     # Utility: generate test patch subsets
│   ├── pick_test_patches.py         # Utility: select retrieval query patches
│   └── requirements.txt             # Frontend Python dependencies
├── models/
│   └── v6/
│       ├── train_improved_v6.py     # Final training script (camera-ready)
│       ├── satellite_image_files_v6.json  # File-to-hash index mapping
│       └── training_history_improved.json # Loss/metric history (30 epochs)
├── .gitignore
└── README.md
```

> **Note:** Model weights (`spectral_hash_v6.pth`), hash matrices (`*.npy`), and dataset files are excluded from this repository via `.gitignore` due to file size constraints.

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended; RTX 4060 used in experiments)
- BigEarthNet-S2 dataset

### 1. Clone the Repository

```bash
git clone https://github.com/sahilll05/SATHash.git
cd SATHash
```

### 2. Install Dependencies

```bash
pip install -r frontend/requirements.txt
```

### 3. Train the Model

```bash
python models/v6/train_improved_v6.py
```

Training runs for **30 epochs** on 13,683 BigEarthNet-S2 patches with AMP (Automatic Mixed Precision) for fast GPU training. Best model checkpoint is saved automatically.

### 4. Run the Retrieval Dashboard

```bash
cd frontend
streamlit run app.py
```

The Streamlit dashboard lets you upload a `.tif` Sentinel-2 patch, generates its 64-bit hash code, and retrieves the top-K most similar patches from the pre-computed database using FAISS Hamming search.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.