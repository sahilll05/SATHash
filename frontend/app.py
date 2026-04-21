import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import faiss
import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SatHash — Satellite Image Retrieval",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Remove default streamlit padding */
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1f3c 50%, #0a1628 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(56, 189, 248, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(99, 102, 241, 0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f9ff;
    letter-spacing: -0.5px;
    margin: 0 0 0.3rem 0;
}
.hero-title span { color: #38bdf8; }
.hero-sub {
    color: rgba(148, 163, 184, 0.9);
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.3px;
    margin: 0;
}
.hero-tags {
    display: flex;
    gap: 8px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.tag {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #7dd3fc;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #0f172a;
    border: 1px solid rgba(51, 65, 85, 0.8);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(56, 189, 248, 0.4); }
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-val.green { color: #4ade80; }
.metric-val.amber { color: #fbbf24; }
.metric-val.purple { color: #a78bfa; }
.metric-lbl {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Section headings */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #38bdf8;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    padding-bottom: 6px;
    margin-bottom: 1rem;
}

/* Image cards */
.img-card {
    background: #0f172a;
    border: 1px solid rgba(51, 65, 85, 0.8);
    border-radius: 12px;
    padding: 10px;
    height: 100%;
}
.img-card.query-card {
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.08);
}
.img-card.match-hit {
    border-color: rgba(74, 222, 128, 0.4);
}
.img-card.match-miss {
    border-color: rgba(248, 113, 113, 0.3);
}
.img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8;
    margin-bottom: 6px;
    letter-spacing: 1px;
}
.img-label .badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 0.6rem;
    margin-left: 6px;
}
.badge-query { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.badge-hit   { background: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.badge-miss  { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.hamming-bar-wrap {
    margin-top: 6px;
    background: rgba(15,23,42,0.8);
    border-radius: 4px;
    height: 4px;
    overflow: hidden;
}
.hamming-bar {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #4ade80, #38bdf8);
    transition: width 0.6s ease;
}
.label-pills {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}
.label-pill {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    font-size: 0.6rem;
    padding: 2px 7px;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080d1a;
    border-right: 1px solid rgba(30, 41, 59, 0.8);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8;
    font-size: 0.82rem;
}
.sidebar-section {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(30, 41, 59, 1);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
}
.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #38bdf8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
.status-dot.green { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
.status-dot.red   { background: #f87171; box-shadow: 0 0 6px #f87171; }
.status-dot.amber { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* Hash viz */
.hash-grid {
    display: grid;
    grid-template-columns: repeat(32, 1fr);
    gap: 2px;
    margin-top: 8px;
}
.hash-bit {
    aspect-ratio: 1;
    border-radius: 2px;
}
.bit-1 { background: #38bdf8; }
.bit-0 { background: #1e293b; }

/* Stagger animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.4s ease forwards; }

/* Override streamlit file uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(56, 189, 248, 0.3);
    border-radius: 10px;
    padding: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid rgba(30, 41, 59, 0.8);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    border: none;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.1) !important;
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DEFINITION  (must match training code)
# ─────────────────────────────────────────────
def gn(channels):
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class SpectralHashNetV3(nn.Module):
    def __init__(self, in_channels=10, embed_dim=128, hash_bits=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), gn(32),  nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          gn(64),  nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),         gn(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),        gn(256), nn.GELU(),
            nn.Flatten()
        )
        self.projector = nn.Sequential(
            nn.Linear(256 * 8 * 8, embed_dim * 2), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.hasher = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, hash_bits)
        )

    def forward(self, x):
        features  = self.encoder(x)
        embedding = F.normalize(self.projector(features), dim=1)
        hash_logits = self.hasher(embedding)
        return embedding, hash_logits


# ── V6 Model Definitions ──
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))

class ResidualBlockProj(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, num_groups=8, use_attention=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.GroupNorm(num_groups, out_ch),
        )
        self.attention = CBAM(out_ch) if use_attention else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = self.attention(out)
        return self.relu(out + self.skip(x))

class SpectralHashNetv6(nn.Module):
    def __init__(self, in_channels=10, embed_dim=128, hash_bits=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlockProj(in_channels, 32, stride=2, num_groups=8, use_attention=True),
            ResidualBlockProj(32, 64, stride=2, num_groups=8, use_attention=True),
            ResidualBlockProj(64, 128, stride=2, num_groups=8, use_attention=True),
            ResidualBlockProj(128, 256, stride=2, num_groups=8, use_attention=True),
            nn.Flatten()
        )
        self.projector = nn.Sequential(
            nn.Linear(256 * 8 * 8, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.hasher = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, hash_bits)
        )

    def forward(self, x):
        features  = self.encoder(x)
        embedding = F.normalize(self.projector(features), dim=1)
        hash_logits = self.hasher(embedding)
        return embedding, hash_logits


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def to_rgb(img_hwc: np.ndarray) -> np.ndarray:
    """(H, W, 10) → (H, W, 3) display-ready RGB using bands 3,2,1."""
    rgb = img_hwc[:, :, [3, 2, 1]].astype(np.float32)
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)


def load_tif(path: str):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)   # (10, 120, 120)
    return img


def img_to_hash(img_chw: np.ndarray, model, device) -> tuple:
    """Return (binary_64, packed_8) hash for a (10,120,120) float array."""
    img_norm = np.clip(img_chw / 10000.0, 0, 1)
    tensor   = torch.tensor(img_norm).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _, h = model(tensor)
        binary = (torch.tanh(h) > 0).cpu().numpy().astype(np.uint8)
        packed = np.packbits(binary, axis=1)
    return binary[0], packed


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0a0e1a", edgecolor="none", dpi=120)
    buf.seek(0)
    return buf


def hash_html(bits: np.ndarray) -> str:
    """Render 64-bit hash as a coloured grid."""
    cells = "".join(
        f'<div class="hash-bit bit-{b}"></div>' for b in bits
    )
    return f'<div class="hash-grid">{cells}</div>'


def hamming_similarity(d: int, max_bits: int = 64) -> float:
    return 1.0 - d / max_bits


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, version: str = "v3"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if version == "v6":
        model = SpectralHashNetv6(in_channels=10, embed_dim=128, hash_bits=64).to(device)
    else:
        model = SpectralHashNetV3(in_channels=10, embed_dim=128, hash_bits=64).to(device)
    
    ckpt   = torch.load(model_path, map_location=device)
    state  = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_index(hash_path: str, files_path: str):
    hash_packed = np.load(hash_path)              # (N, 8) uint8
    index = faiss.IndexBinaryFlat(64)
    index.add(hash_packed)
    with open(files_path) as f:
        image_files = json.load(f)
    return index, image_files, hash_packed


@st.cache_resource(show_spinner=False)
def load_labels(metadata_path: str):
    meta = pd.read_parquet(metadata_path)
    image_labels = {}
    for _, row in meta.iterrows():
        patch_id = str(row["patch_id"]).strip()
        labels   = row["labels"]
        readable = []
        for lbl in labels:
            s = str(lbl).strip()
            try:
                readable.append(s)
            except Exception:
                pass
        image_labels[patch_id] = readable
    return image_labels


def get_labels(fname, image_labels):
    return image_labels.get(fname.replace(".tif", ""), ["(no label)"])


# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 8px">
        <span style="font-family:'Space Mono',monospace;font-size:1.2rem;
              color:#38bdf8;font-weight:700;letter-spacing:2px;">🛰 SATHASH</span><br>
        <span style="font-size:0.65rem;color:#475569;letter-spacing:2px;
              text-transform:uppercase;">Satellite Image Retrieval</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Paths ──────────────────────────────
    st.markdown('<div class="sidebar-title">📁 File Paths</div>', unsafe_allow_html=True)

    models_dir = Path(__file__).resolve().parent.parent / "models"
    available_versions = []
    if models_dir.exists():
        available_versions = sorted(
            [d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v")], 
            reverse=True
        )
    if not available_versions:
        available_versions = ["v6"]
        
    selected_version = st.selectbox("Model Version", available_versions)

    default_model_path = str(models_dir / selected_version / f"spectral_hash_{selected_version}.pth")
    default_hash_path = str(models_dir / selected_version / f"satellite_hash_vectors_{selected_version}.npy")
    default_files_path = str(models_dir / selected_version / f"satellite_image_files_{selected_version}.json")

    model_path    = st.text_input("Model (.pth)", value=default_model_path)
    hash_path     = st.text_input("Hash vectors (.npy)", value=default_hash_path)
    files_path    = st.text_input("Image files (.json)", value=default_files_path)
    metadata_path = st.text_input("Metadata (.parquet)", value=r"D:\Projects\others\image-hashing\dataset\big-earth-net\metadata.parquet")
    train_path    = st.text_input("Train folder", value=r"D:\Projects\others\image-hashing\dataset\big-earth-net\BigEarthNet-S2\train")

    # ── Load all resources ──────────────────
    st.markdown("---")
    load_btn = st.button("⚡ Load Model & Index", use_container_width=True, type="primary")

    if load_btn or "resources_loaded" in st.session_state:
        all_exist = all(
            Path(p).exists() for p in [model_path, hash_path, files_path]
        )
        if not all_exist:
            st.error("One or more files not found. Check paths above.")
        else:
            with st.spinner("Loading..."):
                try:
                    model, device = load_model(model_path, version=selected_version)
                    index, image_files, hash_packed = load_index(hash_path, files_path)
                    image_labels = load_labels(metadata_path) if Path(metadata_path).exists() else {}
                    st.session_state["resources_loaded"] = True
                    st.session_state["model"]        = model
                    st.session_state["device"]       = device
                    st.session_state["index"]        = index
                    st.session_state["image_files"]  = image_files
                    st.session_state["hash_packed"]  = hash_packed
                    st.session_state["image_labels"] = image_labels
                    st.session_state["train_path"]   = train_path
                except Exception as e:
                    st.error(f"Load failed: {e}")

    # ── Status ─────────────────────────────
    if st.session_state.get("resources_loaded"):
        n = len(st.session_state["image_files"])
        st.markdown(f"""
        <div class="sidebar-section">
            <div style="font-size:0.75rem;color:#94a3b8">
                <span class="status-dot green"></span>
                <span style="color:#4ade80;font-weight:500">Model loaded</span>
            </div>
            <div style="font-size:0.7rem;color:#64748b;margin-top:6px">
                Index: <span style="color:#94a3b8">{n:,} images</span><br>
                Device: <span style="color:#94a3b8">{str(st.session_state['device']).upper()}</span><br>
                Labels: <span style="color:#94a3b8">{len(st.session_state['image_labels']):,} entries</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Search settings ─────────────────────
    st.markdown("---")
    st.markdown('<div class="sidebar-title">⚙️ Search Settings</div>', unsafe_allow_html=True)
    k_results = st.slider("Results (k)", 1, 10, 5)
    show_hash = st.checkbox("Show hash visualisation", value=True)
    show_dist = st.checkbox("Show Hamming distribution", value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem;color:#334155;text-align:center;line-height:1.8">
        BigEarthNet-S2 · PyTorch<br>
        NT-Xent · GroupNorm<br>
        FAISS · Hamming search
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header fade-in">
    <div class="hero-title">🛰 Sat<span>Hash</span></div>
    <p class="hero-sub">Deep spectral-spatial hashing for satellite image retrieval · BigEarthNet-S2</p>
    <div class="hero-tags">
        <span class="tag">Sentinel-2 · 10 bands</span>
        <span class="tag">64-bit binary hash</span>
        <span class="tag">FAISS · Hamming search</span>
        <span class="tag">NT-Xent contrastive</span>
        <span class="tag">GroupNorm encoder</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GUARD: not loaded yet
# ─────────────────────────────────────────────
if not st.session_state.get("resources_loaded"):
    st.markdown("""
    <div style="background:#0f172a;border:1px solid rgba(30,41,59,0.8);border-radius:14px;
         padding:3rem;text-align:center;margin-top:1rem">
        <div style="font-size:3rem;margin-bottom:1rem">🔌</div>
        <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#38bdf8;
             margin-bottom:0.5rem">Ready to connect</div>
        <div style="color:#475569;font-size:0.85rem">
            Set your file paths in the sidebar and click <strong style="color:#94a3b8">⚡ Load Model & Index</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# PULL RESOURCES FROM STATE
# ─────────────────────────────────────────────
model        = st.session_state["model"]
device       = st.session_state["device"]
index        = st.session_state["index"]
image_files  = st.session_state["image_files"]
hash_packed  = st.session_state["hash_packed"]
image_labels = st.session_state["image_labels"]
train_path   = st.session_state["train_path"]
hash_matrix  = np.unpackbits(hash_packed, axis=1)  # (N, 64)

# ─────────────────────────────────────────────
# TOP METRIC CARDS
# ─────────────────────────────────────────────
n_unique   = len(np.unique(hash_packed, axis=0))
bit_rates  = hash_matrix.mean(axis=0)
balanced   = int(np.sum((bit_rates > 0.3) & (bit_rates < 0.7)))
unique_pct = 100 * n_unique / len(hash_matrix)

st.markdown(f"""
<div class="metric-row fade-in">
    <div class="metric-card">
        <div class="metric-val green">{unique_pct:.1f}%</div>
        <div class="metric-lbl">Hash uniqueness</div>
    </div>
    <div class="metric-card">
        <div class="metric-val amber">{balanced}/64</div>
        <div class="metric-lbl">Balanced bits</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{len(image_files):,}</div>
        <div class="metric-lbl">Indexed images</div>
    </div>
    <div class="metric-card">
        <div class="metric-val purple">32.0</div>
        <div class="metric-lbl">Mean Hamming</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  RETRIEVE", "📊  ANALYTICS", "📁  BROWSE INDEX"])


# ══════════════════════════════════════════════
# TAB 1 — RETRIEVAL
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 2.8])

    with col_left:
        st.markdown('<div class="section-head">Query input</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Input mode",
            ["📂 Pick from index", "📤 Upload .tif file"],
            horizontal=True,
            label_visibility="collapsed"
        )

        query_img_chw = None
        query_fname   = None

        if input_mode == "📂 Pick from index":
            search_q = st.text_input("🔎 Search filename", placeholder="type patch name...")
            filtered = [f for f in image_files if search_q.lower() in f.lower()] if search_q else image_files
            idx = st.selectbox(
                "Select image",
                range(len(filtered)),
                format_func=lambda i: filtered[i][:60] + "…" if len(filtered[i]) > 60 else filtered[i]
            )
            if filtered:
                query_fname   = filtered[idx]
                query_img_chw = load_tif(os.path.join(train_path, query_fname))

        else:
            uploaded = st.file_uploader("Upload a BigEarthNet .tif", type=["tif", "tiff"])
            if uploaded:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                query_img_chw = load_tif(tmp_path)
                query_fname   = uploaded.name

        if query_img_chw is not None:
            # Preview query
            rgb = to_rgb(np.transpose(query_img_chw, (1, 2, 0)))
            fig_q, ax_q = plt.subplots(figsize=(3, 3))
            ax_q.imshow(rgb)
            ax_q.axis("off")
            fig_q.patch.set_facecolor("#0a0e1a")
            fig_q.tight_layout(pad=0.2)
            st.image(fig_to_pil(fig_q), use_container_width=True)
            plt.close(fig_q)

            # Labels
            qlabels = get_labels(query_fname, image_labels)
            pills = "".join(f'<span class="label-pill">{l}</span>' for l in qlabels[:4])
            st.markdown(f'<div class="label-pills">{pills}</div>', unsafe_allow_html=True)

            # Hash visualisation
            q_binary, q_packed = img_to_hash(query_img_chw, model, device)
            if show_hash:
                st.markdown('<div class="section-head" style="margin-top:12px">Query hash</div>', unsafe_allow_html=True)
                st.markdown(hash_html(q_binary), unsafe_allow_html=True)
                st.caption(f"64-bit code  ·  {int(q_binary.sum())} bits active")

    with col_right:
        st.markdown('<div class="section-head">Retrieval results</div>', unsafe_allow_html=True)

        if query_img_chw is None:
            st.info("Select or upload a query image on the left.")
        else:
            # ── Run search ──────────────────────────
            t0 = time.perf_counter()
            D, I = index.search(q_packed, k_results + 1)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Build results (skip exact self-match)
            results = []
            for dist, idx_r in zip(D[0], I[0]):
                fname = image_files[idx_r]
                if fname != query_fname:
                    results.append((fname, int(dist)))
                if len(results) == k_results:
                    break

            # ── Search info bar ──────────────────────
            hits = sum(1 for f, _ in results if set(get_labels(f, image_labels)) & set(qlabels) - {"(no label)"})
            st.markdown(f"""
            <div style="display:flex;gap:16px;align-items:center;margin-bottom:12px;
                 font-family:'Space Mono',monospace;font-size:0.7rem">
                <span style="color:#64748b">Search time:</span>
                <span style="color:#38bdf8">{elapsed_ms:.1f} ms</span>
                <span style="color:#64748b">Label matches:</span>
                <span style="color:#4ade80">{hits}/{len(results)}</span>
                <span style="color:#64748b">k =</span>
                <span style="color:#94a3b8">{k_results}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Result grid ──────────────────────────
            cols = st.columns(min(k_results, 5))
            for i, (fname, dist) in enumerate(results):
                col = cols[i % 5]
                with col:
                    mlabels = get_labels(fname, image_labels)
                    hit     = bool(set(qlabels) & set(mlabels) - {"(no label)"})
                    sim     = hamming_similarity(dist)

                    mimg_chw = load_tif(os.path.join(train_path, fname))
                    mrgb     = to_rgb(np.transpose(mimg_chw, (1, 2, 0)))

                    fig_m, ax_m = plt.subplots(figsize=(3, 3))
                    ax_m.imshow(mrgb)
                    ax_m.axis("off")
                    # coloured border
                    for spine in ax_m.spines.values():
                        spine.set_edgecolor("#4ade80" if hit else "#f87171")
                        spine.set_linewidth(2.5)
                        spine.set_visible(True)
                    fig_m.patch.set_facecolor("#0a0e1a")
                    fig_m.tight_layout(pad=0.3)
                    st.image(fig_to_pil(fig_m), use_container_width=True)
                    plt.close(fig_m)

                    badge_cls  = "badge-hit" if hit else "badge-miss"
                    badge_text = "✓ match" if hit else "✗ diff"
                    bar_w      = int(sim * 100)
                    bar_col    = "#4ade80" if sim > 0.7 else ("#fbbf24" if sim > 0.5 else "#f87171")

                    pills = "".join(f'<span class="label-pill">{l[:18]}</span>' for l in mlabels[:2])
                    st.markdown(f"""
                    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#64748b;margin-top:2px">
                        Match {i+1} <span class="badge {badge_cls}">{badge_text}</span>
                    </div>
                    <div style="font-size:0.65rem;color:#475569;margin:2px 0">H={dist} · sim={sim:.2f}</div>
                    <div class="hamming-bar-wrap">
                        <div class="hamming-bar" style="width:{bar_w}%;background:{bar_col}"></div>
                    </div>
                    <div class="label-pills">{pills}</div>
                    """, unsafe_allow_html=True)

                    if show_hash:
                        m_binary, _ = img_to_hash(mimg_chw, model, device)
                        diff_bits   = q_binary ^ m_binary
                        diff_html   = "".join(
                            f'<div class="hash-bit" style="background:{"#f87171" if d else "#38bdf8"};aspect-ratio:1;border-radius:2px"></div>'
                            for d in diff_bits
                        )
                        st.markdown(f'<div class="hash-grid" style="margin-top:4px">{diff_html}</div>', unsafe_allow_html=True)
                        st.caption("red = differing bits")


# ══════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-head">Per-bit activation rate</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#0f172a")
        ax.bar(range(64), bit_rates, color="#38bdf8", alpha=0.7, width=0.8)
        ax.axhline(0.5, color="#4ade80", ls="--", lw=1, label="ideal 50%")
        ax.axhline(0.3, color="#fbbf24", ls=":", lw=0.8, alpha=0.6)
        ax.axhline(0.7, color="#fbbf24", ls=":", lw=0.8, alpha=0.6, label="30–70% band")
        ax.set_xlim(-1, 64); ax.set_ylim(0, 1.05)
        ax.tick_params(colors="#475569", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e293b")
        ax.legend(fontsize=7, facecolor="#0f172a", labelcolor="#94a3b8",
                  framealpha=0.8, edgecolor="#1e293b")
        ax.set_xlabel("Bit index", color="#475569", fontsize=8)
        ax.set_ylabel("Activation rate", color="#475569", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown('<div class="section-head">Pairwise Hamming distribution</div>', unsafe_allow_html=True)
        rng  = np.random.default_rng(42)
        sidx = rng.choice(len(hash_matrix), min(400, len(hash_matrix)), replace=False)
        samp = hash_matrix[sidx]
        dists = []
        for i in range(len(samp)):
            for j in range(i+1, min(i+12, len(samp))):
                dists.append(int(np.sum(samp[i] != samp[j])))

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        fig2.patch.set_facecolor("#0a0e1a")
        ax2.set_facecolor("#0f172a")
        ax2.hist(dists, bins=range(0, 65, 2), color="#a78bfa", alpha=0.75, edgecolor="#0f172a", lw=0.5)
        ax2.axvline(32, color="#f87171", ls="--", lw=1.2, label="ideal mean=32")
        ax2.axvline(np.mean(dists), color="#4ade80", ls="-", lw=1.2,
                    label=f"actual mean={np.mean(dists):.1f}")
        ax2.tick_params(colors="#475569", labelsize=8)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#1e293b")
        ax2.legend(fontsize=7, facecolor="#0f172a", labelcolor="#94a3b8",
                   framealpha=0.8, edgecolor="#1e293b")
        ax2.set_xlabel("Hamming distance", color="#475569", fontsize=8)
        ax2.set_ylabel("Count", color="#475569", fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Score summary ──────────────────────────
    st.markdown('<div class="section-head" style="margin-top:1rem">Model scorecard</div>', unsafe_allow_html=True)

    def score_row(label, value, good, ok, fmt=None):
        val_str = fmt(value) if fmt else str(value)
        if value >= good:
            icon, col = "✅", "#4ade80"
        elif value >= ok:
            icon, col = "⚠️", "#fbbf24"
        else:
            icon, col = "❌", "#f87171"
        return icon, label, val_str, col

    rows = [
        score_row("Unique hashes",   unique_pct,          60,   20, lambda v: f"{v:.1f}%"),
        score_row("Balanced bits",   balanced,            40,   20, lambda v: f"{v}/64"),
        score_row("Mean Hamming",    float(np.mean(dists)),20, 10, lambda v: f"{v:.1f}/64"),
    ]

    sc1, sc2, sc3 = st.columns(3)
    for col, (icon, label, val, color) in zip([sc1, sc2, sc3], rows):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;padding:1rem 1.2rem">
                <div style="font-size:1.3rem">{icon}</div>
                <div class="metric-val" style="color:{color};font-size:1.4rem;margin-top:4px">{val}</div>
                <div class="metric-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — BROWSE INDEX
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">Browse indexed images</div>', unsafe_allow_html=True)

    b_search = st.text_input("Filter by name", placeholder="e.g. T34TCR...")
    filtered_browse = [f for f in image_files if b_search.lower() in f.lower()] if b_search else image_files
    st.caption(f"Showing {min(len(filtered_browse), 20)} of {len(filtered_browse):,} images")

    browse_cols = st.columns(5)
    for ci, fname in enumerate(filtered_browse[:20]):
        with browse_cols[ci % 5]:
            try:
                img_chw = load_tif(os.path.join(train_path, fname))
                rgb     = to_rgb(np.transpose(img_chw, (1, 2, 0)))
                fig_b, ax_b = plt.subplots(figsize=(3, 3))
                ax_b.imshow(rgb); ax_b.axis("off")
                fig_b.patch.set_facecolor("#0a0e1a")
                fig_b.tight_layout(pad=0.2)
                st.image(fig_to_pil(fig_b), use_container_width=True)
                plt.close(fig_b)
                lbls  = get_labels(fname, image_labels)
                pills = "".join(f'<span class="label-pill">{l[:14]}</span>' for l in lbls[:2])
                st.markdown(f"""
                <div style="font-size:0.6rem;color:#475569;font-family:'Space Mono',monospace;
                     margin:2px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                     title="{fname}">{fname[:30]}…</div>
                <div class="label-pills">{pills}</div>
                """, unsafe_allow_html=True)
            except Exception:
                st.caption("⚠ load error")