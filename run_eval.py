"""Reproduce exact mAP and P@5 from the final checkpoint."""
import torch, numpy as np, os, pandas as pd
import torch.nn as nn, torch.nn.functional as F
import rasterio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
        mid = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(mid, ch, 1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x): return x * self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=ks//2, bias=False); self.sig = nn.Sigmoid()
    def forward(self, x): return x * self.sig(self.conv(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    def __init__(self, ch, r=16, ks=7):
        super().__init__(); self.ca = ChannelAttention(ch, r); self.sa = SpatialAttention(ks)
    def forward(self, x): return self.sa(self.ca(x))

class ResProj(nn.Module):
    def __init__(self, ic, oc, stride=1, ng=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(ng, oc), nn.ReLU(inplace=True), nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.GroupNorm(ng, oc))
        self.skip = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.GroupNorm(ng, oc))
        self.attn = CBAM(oc); self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.attn(self.block(x)) + self.skip(x))

class SpectralHashNetV6(nn.Module):
    def __init__(self, in_ch=10, embed_dim=128, hash_bits=64):
        super().__init__()
        self.encoder = nn.Sequential(ResProj(in_ch, 32, stride=2), ResProj(32, 64, stride=2), ResProj(64, 128, stride=2), ResProj(128, 256, stride=2), nn.Flatten())
        self.projector = nn.Sequential(nn.Linear(256*8*8, embed_dim*2), nn.GELU(), nn.Dropout(0.2), nn.Linear(embed_dim*2, embed_dim))
        self.hasher = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, hash_bits))
    def forward(self, x):
        feat = self.encoder(x); emb = F.normalize(self.projector(feat), dim=1); h = self.hasher(emb); return emb, h

class BigEarthDataset(Dataset):
    def __init__(self, *folders):
        self.files = []
        for folder in folders:
            folder = Path(folder)
            if folder.exists():
                self.files.extend(sorted([str(folder / f) for f in os.listdir(folder) if f.endswith('.tif')]))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        with rasterio.open(self.files[idx]) as src: img = src.read().astype(np.float32)
        return torch.tensor(np.clip(img/10000., 0., 1.)), os.path.basename(self.files[idx])

BASE = Path(r'E:\Projects\others\image-hashing\dataset\big-earth-net\BigEarthNet-S2')
BEST_CKPT = Path(r'E:\Projects\others\image-hashing\multispectral-satellite\models\v6\spectral_hash_v6_best.pth')

print("Loading best checkpoint (epoch 10)...")
ckpt = torch.load(str(BEST_CKPT), map_location='cpu', weights_only=False)
model = SpectralHashNetV6(in_ch=10, embed_dim=128, hash_bits=64)
model.load_state_dict(ckpt['model_state']); model.eval()
print(f"  Stored best_map: {ckpt['best_map']:.6f}  epoch: {ckpt['epoch']}")

test_ds = BigEarthDataset(BASE / 'test')
print(f"  Total test: {len(test_ds)}")
n_query = int(0.2 * len(test_ds)); n_db = len(test_ds) - n_query
print(f"  Query: {n_query}  DB: {n_db}")

query_ds, db_ds = torch.utils.data.random_split(test_ds, [n_query, n_db], generator=torch.Generator().manual_seed(42))

def get_hashes(ds):
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    hs, fns = [], []
    with torch.no_grad():
        for imgs, names in loader:
            _, h = model(imgs)
            hs.append(torch.tanh(h).numpy()); fns.extend(names)
    return np.vstack(hs), np.array(fns)

print("Extracting query hashes...")
q_h, q_lbl = get_hashes(query_ds)
print("Extracting DB hashes...")
db_h, db_lbl = get_hashes(db_ds)

print("Loading metadata...")
metadata_df = pd.read_parquet(BASE.parent / 'metadata.parquet')
df = metadata_df.set_index('patch_id')
q_ids = [f.replace('.tif','') for f in q_lbl]
db_ids = [f.replace('.tif','') for f in db_lbl]
q_labels = np.array([set(l) for l in df.loc[q_ids, 'labels']])
db_labels = np.array([set(l) for l in df.loc[db_ids, 'labels']])

q_bits = (q_h > 0).astype(np.uint8)
d_bits = (db_h > 0).astype(np.uint8)

aps, p5_list, p10_list, ndcg10_list = [], [], [], []
for i in range(len(q_bits)):
    dists = np.sum(q_bits[i] != d_bits, axis=1)
    order = np.argsort(dists)
    hits = np.array([len(q_labels[i] & db_labels[j]) > 0 for j in order], dtype=float)
    if hits.sum() == 0: continue
    cum = np.cumsum(hits); prec = cum / (np.arange(len(hits)) + 1)
    aps.append((prec * hits).sum() / hits.sum())
    p5_list.append(float(np.mean(hits[:5])))
    p10_list.append(float(np.mean(hits[:10])))
    dcg = float(np.sum(hits[:10] / np.log2(np.arange(2, 12))))
    idcg = float(np.sum(np.sort(hits)[::-1][:10] / np.log2(np.arange(2, 12))))
    ndcg10_list.append(dcg / idcg if idcg > 0 else 0.0)

print("\n=== VERIFIED METRICS (best checkpoint, seed=42) ===")
print(f"mAP      = {np.mean(aps):.6f}")
print(f"P@5      = {np.mean(p5_list):.6f}")
print(f"P@10     = {np.mean(p10_list):.6f}")
print(f"NDCG@10  = {np.mean(ndcg10_list):.6f}")
print(f"N queries  = {len(q_bits)}")
print(f"N DB       = {len(d_bits)}")
print(f"Checkpoint = {BEST_CKPT.name}")
