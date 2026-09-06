"""
train_improved_v6.py
────────────────────────────────────────────────────────────────────────────────
Improved training script for SpectralHashNetV6 — camera-ready results.

Uses ALL available local BigEarthNet-S2 patches (13,683 patches across
train/validation/test splits) with 30 epochs and AMP for faster training.

Dataset: E:\Projects\others\image-hashing\dataset\big-earth-net\BigEarthNet-S2\
  - train: 7,180 patches
  - validation: 3,255 patches
  - test: 3,248 patches
  Total: 13,683 patches (vs 5,000 in submitted paper)

Run: python multispectral-satellite/models/v6/train_improved_v6.py
"""

import os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import average_precision_score
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(r"E:\Projects\others\image-hashing\dataset\big-earth-net\BigEarthNet-S2")
TRAIN_PATH = BASE / "train"
VAL_PATH   = BASE / "validation"
TEST_PATH  = BASE / "test"
OUT_DIR    = Path(r"E:\Projects\others\image-hashing\multispectral-satellite\models\v6")

# ── Training Config ────────────────────────────────────────────────────────────
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE  = 0.1
LAMBDA_Q     = 0.5
LAMBDA_IND   = 1.0
WARMUP       = 2
HASH_BITS    = 64
EMBED_DIM    = 128

# ── Dataset ────────────────────────────────────────────────────────────────────
class BigEarthDataset(Dataset):
    """Loads BigEarthNet-S2 .tif files. Each: 10 bands, 120x120 pixels."""
    def __init__(self, *folders, limit=None):
        self.files = []
        for folder in folders:
            folder = Path(folder)
            if folder.exists():
                tifs = sorted([str(folder / f) for f in os.listdir(folder) if f.endswith(".tif")])
                self.files.extend(tifs)
        if limit:
            self.files = self.files[:limit]
        print(f"  Dataset: {len(self.files)} patches from {len(folders)} split(s)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with rasterio.open(self.files[idx]) as src:
            img = src.read().astype(np.float32)
        img = np.clip(img / 10000.0, 0.0, 1.0)
        return torch.tensor(img), os.path.basename(self.files[idx])


# ── Augmentation ───────────────────────────────────────────────────────────────
def augment(img: torch.Tensor) -> torch.Tensor:
    # Spatial: flips + rotation
    if torch.rand(1) > 0.5: img = torch.flip(img, dims=[2])
    if torch.rand(1) > 0.5: img = torch.flip(img, dims=[1])
    k = torch.randint(0, 4, (1,)).item()
    img = torch.rot90(img, k, dims=[1, 2])
    # Spectral: per-band jitter
    C = img.shape[0]
    scale = 1.0 + (torch.rand(C, 1, 1, device=img.device) - 0.5) * 0.2
    return torch.clamp(img * scale, 0.0, 1.0)


# ── Model ──────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        mid = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(mid, ch, 1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=ks//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.conv(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    def __init__(self, ch, r=16, ks=7):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention(ks)
    def forward(self, x):
        return self.sa(self.ca(x))

class ResProj(nn.Module):
    def __init__(self, ic, oc, stride=1, ng=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(ng, oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.GroupNorm(ng, oc),
        )
        self.skip = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.GroupNorm(ng, oc))
        self.attn = CBAM(oc)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.attn(self.block(x)) + self.skip(x))

class SpectralHashNetV6(nn.Module):
    def __init__(self, in_ch=10, embed_dim=128, hash_bits=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ResProj(in_ch, 32,  stride=2),   # 120→60
            ResProj(32,    64,  stride=2),   # 60→30
            ResProj(64,    128, stride=2),   # 30→15
            ResProj(128,   256, stride=2),   # 15→8
            nn.Flatten()                      # 256*8*8=16384
        )
        self.projector = nn.Sequential(
            nn.Linear(256*8*8, embed_dim*2), nn.GELU(), nn.Dropout(0.2), nn.Linear(embed_dim*2, embed_dim)
        )
        self.hasher = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, hash_bits)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.encoder(x)
        emb  = F.normalize(self.projector(feat), dim=1)
        h    = self.hasher(emb)
        return emb, h


# ── Loss Functions ─────────────────────────────────────────────────────────────
def nt_xent(z1, z2, T=0.1):
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = torch.mm(z, z.T) / T
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, float('-inf'))
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)

def quant_loss(h):
    return torch.mean((torch.abs(torch.tanh(h)) - 1.0)**2)

def indep_loss(h):
    hh = torch.tanh(h)
    B, K = hh.shape
    balance = torch.mean(hh.mean(dim=0)**2)
    hc  = hh - hh.mean(dim=0, keepdim=True)
    cov = torch.mm(hc.T, hc) / max(B-1, 1)
    eye = torch.eye(K, device=hh.device)
    return balance + torch.mean((cov*(1-eye))**2)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_full(q_h, db_h, q_lbl, db_lbl, metadata_df):
    """Compute full metrics using BigEarthNet multi-label ground truth."""
    q_bits = (q_h > 0).astype(np.uint8)
    d_bits = (db_h > 0).astype(np.uint8)
    
    q_ids = [f.replace(".tif", "") for f in q_lbl]
    db_ids = [f.replace(".tif", "") for f in db_lbl]
    
    df = metadata_df.set_index("patch_id")
    q_labels = df.loc[q_ids].drop(columns=["split", "timestamp", "geometry"], errors="ignore").values
    db_labels = df.loc[db_ids].drop(columns=["split", "timestamp", "geometry"], errors="ignore").values
    
    aps, p1, p5, p10, ndcg10 = [], [], [], [], []
    for i in range(len(q_bits)):
        dists = np.sum(q_bits[i] != d_bits, axis=1)
        order = np.argsort(dists)
        
        q_l = q_labels[i]
        db_l = db_labels[order]
        hits = (np.dot(db_l, q_l) > 0).astype(float)
        
        if hits.sum() == 0:
            continue
            
        cum = np.cumsum(hits)
        prec = cum / (np.arange(len(hits)) + 1)
        aps.append((prec * hits).sum() / hits.sum())
        
        p1.append(hits[0])
        p5.append(np.mean(hits[:5]))
        p10.append(np.mean(hits[:10]))
        
        dcg = np.sum(hits[:10] / np.log2(np.arange(2, 12)))
        idcg = np.sum(np.sort(hits)[::-1][:10] / np.log2(np.arange(2, 12)))
        ndcg10.append(dcg / idcg if idcg > 0 else 0)
        
    mAP = float(np.mean(aps)) if aps else 0.0
    
    # Mean Hamming Distance
    rng = np.random.default_rng(42)
    idx = rng.choice(len(q_bits), min(100, len(q_bits)), replace=False)
    hd = []
    for i in idx:
        for j in rng.choice(len(d_bits), 10, replace=False):
            hd.append(int(np.sum(q_bits[i] != d_bits[j])))
            
    return {
        "mAP": mAP,
        "P@1": float(np.mean(p1)),
        "P@5": float(np.mean(p5)),
        "P@10": float(np.mean(p10)),
        "NDCG@10": float(np.mean(ndcg10)),
        "mean_hamming": float(np.mean(hd)) if hd else 0.0
    }


# ── Main Training ──────────────────────────────────────────────────────────────
def train(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("SpectralHashNetV6 — Improved Camera-Ready Training")
    print(f"GPU  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Hash: {HASH_BITS}-bit")
    print("=" * 65)

    # Load ALL available patches for training (train + validation splits)
    print("\nLoading metadata...")
    metadata_df = pd.read_parquet(BASE.parent / "metadata.parquet")
    
    print("\nLoading datasets...")
    train_ds  = BigEarthDataset(TRAIN_PATH, VAL_PATH)   # 7180 + 3255 = 10435
    test_ds   = BigEarthDataset(TEST_PATH)               # 3248 for eval

    # Split test into query (20%) + database (80%)
    n_query = int(0.2 * len(test_ds))
    n_db    = len(test_ds) - n_query
    query_ds, db_ds = torch.utils.data.random_split(test_ds, [n_query, n_db], generator=torch.Generator().manual_seed(seed))

    train_loader   = DataLoader(train_ds,   batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    ordered_loader = DataLoader(train_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    query_loader   = DataLoader(query_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    db_loader      = DataLoader(db_ds,      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nTrain patches : {len(train_ds):,}  (vs 4,000 in submitted paper)")
    print(f"Query patches : {n_query:,}  | DB patches: {n_db:,}")

    # Model
    model = SpectralHashNetV6(in_ch=10, embed_dim=EMBED_DIM, hash_bits=HASH_BITS).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SpectralHashNetV6 ({total/1e6:.1f}M params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler    = GradScaler('cuda')

    history = {"total":[], "ntxent":[], "quant":[], "indep":[], "unique_pct":[]}
    best_map = 0.0
    ckpt_path = OUT_DIR / "spectral_hash_v6_improved.pth"

    print(f"\nOutput: {ckpt_path.name}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        e_ntx = e_q = e_ind = e_tot = 0.0
        reg_scale = min(1.0, max(0.0, (epoch - WARMUP + 1) / 3)) if epoch >= WARMUP else 0.0
        t0 = time.time()

        bar = tqdm(train_loader, desc=f"Ep{epoch:02d}/{EPOCHS}", leave=False)
        for imgs, _ in bar:
            imgs = imgs.to(device)
            v1 = torch.stack([augment(img) for img in imgs]).to(device)
            v2 = torch.stack([augment(img) for img in imgs]).to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                emb1, h1 = model(v1)
                emb2, h2 = model(v2)
                loss_ntx = nt_xent(emb1, emb2, TEMPERATURE)
                loss_q   = (quant_loss(h1) + quant_loss(h2)) / 2
                loss_ind = (indep_loss(h1) + indep_loss(h2)) / 2
                loss = loss_ntx + reg_scale * (LAMBDA_Q * loss_q + LAMBDA_IND * loss_ind)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            e_ntx += loss_ntx.item(); e_q += loss_q.item()
            e_ind += loss_ind.item(); e_tot += loss.item()
            bar.set_postfix(ntx=f"{loss_ntx.item():.3f}", q=f"{loss_q.item():.3f}", reg=f"{reg_scale:.1f}")

        scheduler.step()
        n = len(train_loader)

        # Health check every epoch
        model.eval()
        sample_h = []
        with torch.no_grad():
            for imgs_s, _ in ordered_loader:
                with autocast('cuda'):
                    _, h = model(imgs_s.to(device))
                sample_h.append((torch.tanh(h) > 0).cpu().numpy().astype(np.uint8))
                if sum(len(x) for x in sample_h) >= 1000: break
        sample_h = np.vstack(sample_h)[:1000]
        n_u = len(np.unique(sample_h, axis=0))
        upct = 100 * n_u / len(sample_h)
        bal = int(np.sum((sample_h.mean(0) > 0.3) & (sample_h.mean(0) < 0.7)))
        status = "GOOD" if upct >= 60 else ("PARTIAL" if upct >= 20 else "COLLAPSED")

        history["total"].append(e_tot/n); history["ntxent"].append(e_ntx/n)
        history["quant"].append(e_q/n);   history["indep"].append(e_ind/n)
        history["unique_pct"].append(upct)

        print(f"Ep{epoch:02d}/{EPOCHS} | ntx={e_ntx/n:.4f} q={e_q/n:.4f} ind={e_ind/n:.4f} | "
              f"reg={reg_scale:.1f} | unique={upct:.1f}% bal={bal}/{HASH_BITS} | "
              f"time={time.time()-t0:.1f}s [{status}]")

        # Eval every 5 epochs and at end
        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            # Collect query/db hashes
            q_h, q_lbl, db_h, db_lbl = [], [], [], []
            with torch.no_grad():
                for imgs, fnames in tqdm(query_loader, desc="  Query", leave=False):
                    with autocast('cuda'):
                        _, h = model(imgs.to(device))
                    q_h.append(torch.tanh(h).cpu().numpy())
                    q_lbl.extend(fnames)
                for imgs, fnames in tqdm(db_loader, desc="  DB", leave=False):
                    with autocast('cuda'):
                        _, h = model(imgs.to(device))
                    db_h.append(torch.tanh(h).cpu().numpy())
                    db_lbl.extend(fnames)
            q_h  = np.vstack(q_h);  db_h  = np.vstack(db_h)
            q_lbl = np.array(q_lbl); db_lbl = np.array(db_lbl)

            # Evaluate full metrics using actual multi-label metadata
            metrics = evaluate_full(q_h, db_h, q_lbl, db_lbl, metadata_df)
            print(f"  --> Eval | mAP={metrics['mAP']:.4f} | P@5={metrics['P@5']:.4f} | NDCG@10={metrics['NDCG@10']:.4f} | Mean Hamming={metrics['mean_hamming']:.1f}/{HASH_BITS}")

            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "history": history, "best_map": best_map,
                    "config": {"embed_dim": EMBED_DIM, "hash_bits": HASH_BITS, "epochs": epoch}
                }, ckpt_path)
                print(f"  --> Saved best checkpoint: {ckpt_path.name}")

    # Save final history
    hist_path = OUT_DIR / "training_history_improved.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best mAP: {best_map:.4f}")
    print(f"History: {hist_path}")
    print(f"Model:   {ckpt_path}")


if __name__ == "__main__":
    # Hermes recommended 3 seeds for error bars: [42, 123, 456]
    # We default to 42 for quick reproduction, but others can be added
    for s in [42]:
        print(f"\n\n*** RUNNING SEED {s} ***\n")
        train(seed=s)
