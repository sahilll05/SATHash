"""
pick_test_patches.py
--------------------
Picks 20 real BigEarthNet validation patches covering diverse land-cover types
and copies them to a folder you can drag-and-drop into the Streamlit app.

These patches ARE in metadata.parquet, so the green/red label-match
badges in Streamlit will work correctly — unlike the synthetic Indian patches.

Run this once from any folder. Edit VAL_PATH if needed.
"""

import os, shutil, random, json
import pandas as pd

# ── Paths — edit if yours differ ──────────────────────────────────────────────
VAL_PATH  = r"D:\Projects\others\image-hashing\dataset\big-earth-net\BigEarthNet-S2\validation"
META_PATH = r"D:\Projects\others\image-hashing\dataset\big-earth-net\metadata.parquet"
OUT_DIR   = r"D:\Projects\others\image-hashing\streamlit_test_patches"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# Load metadata to find patches by land-cover label
print("Loading metadata...")
meta = pd.read_parquet(META_PATH)

# Keep only validation split
val_meta = meta[meta["split"] == "validation"].copy()
print(f"Validation patches in metadata: {len(val_meta):,}")

# Build label → list of patch_ids mapping
label_to_patches = {}
for _, row in val_meta.iterrows():
    for lbl in row["labels"]:
        lbl = str(lbl).strip()
        label_to_patches.setdefault(lbl, []).append(str(row["patch_id"]).strip())

# Target labels — pick one patch per land-cover type for diversity
target_labels = [
    "Arable land",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Urban fabric",
    "Industrial or commercial units",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Transitional woodland, shrub",
    "Inland waters",
    "Inland wetlands",
    "Natural grassland",
    "Moors and heathland",
    "Beaches, dunes, sands",
    "Coastal wetlands",
    "Sclerophyllous vegetation",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
]

copied  = []
missing = []

random.seed(42)

print(f"\nPicking one patch per land-cover type...")
print(f"{'Land cover':<55s} {'Patch ID':<50s} {'Status'}")
print("-" * 120)

for label in target_labels:
    candidates = label_to_patches.get(label, [])
    if not candidates:
        print(f"  {'(not in metadata)':<55s} {'—':<50s} skip")
        missing.append(label)
        continue

    random.shuffle(candidates)

    found = False
    for patch_id in candidates[:20]:   # try up to 20 candidates
        fname = patch_id + ".tif"
        src   = os.path.join(VAL_PATH, fname)
        if os.path.exists(src):
            dst = os.path.join(OUT_DIR, fname)
            shutil.copy2(src, dst)
            size = os.path.getsize(dst) / 1024
            print(f"  {label:<55s} {patch_id:<50s} {size:.0f} KB  ✓")
            copied.append((label, patch_id, fname))
            found = True
            break

    if not found:
        print(f"  {label:<55s} {'(file not in val folder)':<50s} skip")
        missing.append(label)

# Also add 5 random validation patches for extra variety
print(f"\nAdding 5 random validation patches for extra variety...")
all_val_files = [f for f in os.listdir(VAL_PATH) if f.endswith(".tif")]
random.shuffle(all_val_files)
extra_count = 0
for fname in all_val_files:
    if fname not in [c[2] for c in copied] and extra_count < 5:
        src = os.path.join(VAL_PATH, fname)
        dst = os.path.join(OUT_DIR, fname)
        shutil.copy2(src, dst)
        patch_id = fname.replace(".tif", "")
        row = val_meta[val_meta["patch_id"] == patch_id]
        lbls = list(row["labels"].iloc[0]) if len(row) > 0 else ["?"]
        print(f"  {fname:<60s} {', '.join(str(l) for l in lbls[:2])}")
        extra_count += 1

# Summary
print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
print(f"  Patches copied : {len(copied) + extra_count}")
print(f"  Labels covered : {len(copied)}")
print(f"  Output folder  : {OUT_DIR}")
print()
print(f"HOW TO USE IN STREAMLIT:")
print(f"  1. Open your app at http://localhost:8501")
print(f"  2. Go to the Retrieve tab")
print(f"  3. Select '📤 Upload .tif file'")
print(f"  4. Drag any .tif from:  {OUT_DIR}")
print(f"  5. Green badge = label match, Red = different land cover")
print()
print(f"EXPECTED BEHAVIOUR:")
print(f"  - Query shows real land-cover labels (e.g. 'Arable land')")
print(f"  - Results show green borders when semantically similar")
print(f"  - Hamming distances H=0-10 are normal for similar patches")
print(f"  - Patches from same country/season should score more greens")

# Save a manifest
manifest = [{"label": l, "patch_id": p, "file": f} for l, p, f in copied]
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"\n  Manifest saved: {OUT_DIR}\\manifest.json")
