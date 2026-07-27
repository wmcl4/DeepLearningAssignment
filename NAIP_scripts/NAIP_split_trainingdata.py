import shutil
import random
from pathlib import Path

# ---- CONFIG ----
source_dir = Path(r"/naip_tiles_art_converted")
output_dir = Path(r"/naip_finetune_data")

ext = ".bmp"
label_suffix = "_label"

# With very few images, an 80/20 split (no separate test set) is reasonable --
# your "test" for this model is really the broader stitched NAIP output you're
# already inspecting qualitatively.
train_ratio = 0.80
val_ratio = 0.20

seed = 13
# ----------------

random.seed(seed)

all_files = list(source_dir.glob(f"*{ext}"))
image_files = [f for f in all_files if not f.stem.endswith(label_suffix)]

pairs = []
for img in image_files:
    mask = source_dir / f"{img.stem}{label_suffix}{ext}"
    if mask.exists():
        pairs.append((img, mask))
    else:
        print(f"Warning: no mask found for {img.name}, skipping.")

print(f"Found {len(pairs)} image/mask pairs.")

random.shuffle(pairs)
n = len(pairs)
n_train = int(n * train_ratio)
n_val = n - n_train  # remainder to avoid rounding loss

train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:]

print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

splits = {
    "train": train_pairs,
    "val": val_pairs,
}

for split_name, split_pairs in splits.items():
    out_dir = output_dir / split_name
    if out_dir.exists():
        shutil.rmtree(out_dir)  # clear stale files from previous runs
    out_dir.mkdir(parents=True, exist_ok=True)

    for img, mask in split_pairs:
        shutil.copy2(img, out_dir / img.name)
        shutil.copy2(mask, out_dir / mask.name)

print("Done! Files copied into train/val folders.")