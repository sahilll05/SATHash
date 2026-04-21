# ðŸ›°ï¸  HyperHash: Hyperspectral Satellite Image Hashing Network

**Sub-second content retrieval from multi-band satellite archives** | Built with real Sentinel-2 satellite data

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![FAISS](https://img.shields.io/badge/FAISS-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ðŸŒ  The Problem

Imagine trying to search through **billions of satellite images** where:
- Each image has **multiple spectral bands** (not just RGB)
- Each patch is **2 MB in size**
- You need results in **milliseconds**, not minutes
- You need to understand **material composition**, not just visual appearance

Traditional image retrieval is too slow. RGB-only hashing loses critical spectral information that distinguishes crops from concrete, healthy plants from diseased ones, or minerals from soil.

**We built the first deep hashing system for hyperspectral satellite data.**

---

## âœ¨ What We Built

### The Core Innovation
A **Hybrid Spectral-Spatial Deep Hashing Network** (HybridHashNet) that:

| Feature | Capability |
|---------|-----------|
| **Input** | Multi-spectral bands from satellites |
| **Output** | Compact 64-bit or 128-bit binary hash codes |
| **Compression** | Huge size reduction (MBs â†’ bytes) |
| **Retrieval Speed** | Sub-millisecond per query via FAISS/Hamming Search |
| **Search Scale** | Millions of patches in milliseconds |

### Architecture Stages

```text
INPUT PATCH (Multi-spectral bands, e.g., 120Ã—120 px)
        â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â””‚   STAGE 1: INPUT TENSOR           â”‚
â”‚   Shape: (batch, bands, H, W)    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
        â†“ Splits into branches
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â””Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â””‚ SPATIAL BRANCH   â”‚  â”‚ SPECTRAL BRANCH        â”‚
â”‚ (What it looks   â”‚  â”‚ (What it's made of)    â”‚
â”‚  like spatially) â”‚  â”‚ + Features extraction  â”‚
â”‚ â†’ Dense vector   â”‚  â”‚ â†’ Dense vector         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
        â†“              â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â””‚   STAGE 3: FUSION LAYER           â”‚
â”‚   Concatenate & Transform        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
        â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â””‚   STAGE 4: HASH LAYER             â”‚
â”‚   FC + Tanh + Sign                â”‚
â”‚   Output: Binary Hash Code       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ðŸ—ï¸  Project Structure

```text
SATHash/
â”œâ”€â”€ assets/                      # Application screenshots and demo assets
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ app.py                   # Streamlit dashboard
â”‚   â”œâ”€â”€ generate_test_patches.py # Utilities for generating test sets
â”‚   â”œâ”€â”€ pick_test_patches.py     # Retrieval selection utilities
â”‚   â”œâ”€â”€ requirements.txt         # Frontend dependencies
â”‚   â””â”€â”€ dataset/                 # Evaluation dataset (BigEarthNet)
â”œâ”€â”€ models/
â”‚   â””â”€â”€ v6/
â”‚       â”œâ”€â”€ satellite-model-v6.ipynb    # Model training and architecture definition
â”‚       â”œâ”€â”€ spectral_hash_v6.pth        # Saved PyTorch model weights
â”‚       â”œâ”€â”€ satellite_hash_matrix_v6.npy # Embedded pre-computed database
â”‚       â””â”€â”€ satellite_image_files_v6.json # Hash mapped file indexing
â”œâ”€â”€ .gitignore                   # Git exceptions
â””â”€â”€ README.md                    # This Walkthrough & Information
```

---

## ðŸ’¡ Key Innovations

### How Hamming Distance Works
Two hash codes compared in nanoseconds:

```text
Hash A: 10110011 01001101 11000010 ...
Hash B: 10100011 01011101 11000000 ...
XOR:    00010000 00010000 00000010 ...
        â†“ Count set bits
Hamming Distance = 3 bits
```
Interpretation: 3 out of bounds bits differ â†’ High similarity threshold. 
This is why retrieval is sub-millisecond: CPU XOR + popcount = nanoseconds.

### Why This Matters vs RGB-Only Hashing
Brown soil + Brown concrete â†’ Standard RGB image hash will flag them as similar. Hyperspectral hashing uses SWIR/NIR bands, identifying true material composites, resulting in a completely different and correct separation.

---

## ðŸš€ Walkthrough & Dashboard

The Streamlit frontend lets you upload `.tif` satellite files, parses the spectral data into our Deep Hashing system, converts them into binary structures (64-bit compact code), and does cross-comparison against our offline generated FAISS-ready hash matrix.

### App Features Overview
- **Pick or Upload**: Use an existing subset image or upload a raw Sentinel-2 `.tif`.
- **Query Hash Visualization**: Your image translates directly into a binary string map visually depicted below the image.
- **Microsecond Retrieval**: Computes finding the top $K$ nearest comparisons (`k=5`).
- **Bit Differing Highlight**: Returns specific matching metadata and isolates differing bits (in red) versus the base query.

### Example Retrieval Results

#### Retrieve Query 1: Broad-leaved Forest & Inland Waters
We achieved `1.8 ms` search time capturing semantic maps visually. Differentials (`red = differing bits`) clearly isolate the accuracy based on the Hamming separation.

![Retrieval Result 1](assets/retrieval_output_1.png)

#### Retrieve Query 2: Arable Land Cultivation 
For large farmlands, the spatial characteristics effectively retrieved completely independent geographical nodes with exactly matched label categories `(5/5)` at sub-2 millisecond retrieval constraints.

![Retrieval Result 2](assets/retrieval_output_2.png)

*(Note: Please ensure you drop your two screenshot images into `assets/retrieval_output_1.png` and `assets/retrieval_output_2.png` respectively to visualize!)*

---

## âš™ï¸  Installation & Usage

### 1. Prerequisites
- Python 3.8+
- Pytorch 1.9+

### 2. Setup

Clone the repository and install requirements:

```bash
git clone https://github.com/sahilll06/SATHash.git
cd SATHash
cd frontend

pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

This will deploy the application onto `localhost:8501` featuring the interactive inference engine.

---

## ðŸ“„ License
This project is subject to the MIT License.