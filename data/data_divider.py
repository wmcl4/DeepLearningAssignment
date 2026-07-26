import shutil
import random
from pathlib import Path

# ---- CONFIG ----
category_dirs = [
    Path(r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03\PV03_Ground_Cropland"),
    Path(r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03\PV03_Ground_Grassland"),
    Path(r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03\PV03_Ground_SalineAlkali"),
    Path(r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03\PV03_Ground_Shrubwood"),
    Path(r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03\PV03_Ground_WaterSurface"),
    Path(r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03\PV03_Rooftop"),
]

output_dir = Path(r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03")

ext = ".bmp"
label_suffix = "_label"

train_ratio = 0.70
val_ratio = 0.20
test_ratio = 0.10

seed = 13
# ----------------

random.seed(seed)

splits = {"train": [], "val": [], "test": []}

for data_dir in category_dirs:
    category_name = data_dir.name  # e.g. "PV03_Ground_Cropland"

    # Find all "true color" images (i.e. not label masks)
    all_files = list(data_dir.glob(f"*{ext}"))
    image_files = [f for f in all_files if not f.stem.endswith(label_suffix)]

    # Build (image, mask) pairs, verifying the mask exists
    pairs = []
    for img in image_files:
        mask = data_dir / f"{img.stem}{label_suffix}{ext}"
        if mask.exists():
            pairs.append((img, mask))
        else:
            print(f"Warning: no mask found for {img.name}, skipping.")

    print(f"{category_name}: found {len(pairs)} image/mask pairs.")

    # Shuffle and split THIS category on its own, so each split gets a
    # proportional mix of every category rather than an uneven global shuffle
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # remainder to avoid rounding loss

    splits["train"].extend((category_name, img, mask) for img, mask in pairs[:n_train])
    splits["val"].extend((category_name, img, mask) for img, mask in pairs[n_train:n_train + n_val])
    splits["test"].extend((category_name, img, mask) for img, mask in pairs[n_train + n_val:])

    print(f"  -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

print(f"\nTotal - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# Create output folders and copy files (image + mask together, no subfolders)
# Filenames are prefixed with their category to avoid collisions between categories
for split_name, split_items in splits.items():
    out_dir = output_dir / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for category_name, img, mask in split_items:
        new_img_name = f"{category_name}_{img.name}"
        new_mask_name = f"{category_name}_{mask.name}"
        shutil.copy2(img, out_dir / new_img_name)
        shutil.copy2(mask, out_dir / new_mask_name)

print("Done! Files from all categories merged and copied into train/val/test folders.")