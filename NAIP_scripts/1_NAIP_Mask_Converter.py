import os
import glob
import numpy as np
from PIL import Image

# ---- CONFIG ----
SOURCE_DIR = r"/naip_tiles_art"
# Where to write the corrected masks. Set this equal to SOURCE_DIR to overwrite
# in place, or point it somewhere new to keep your original painted files untouched.
OUTPUT_DIR = r"/naip_tiles_art_converted"
THRESHOLD = 128  # pixel values below this = "panel" in your painted (dark-on-light) masks
# ----------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label_files = glob.glob(os.path.join(SOURCE_DIR, "*_label.bmp"))
    print(f"Found {len(label_files)} label file(s) to convert.\n")

    for label_path in label_files:
        mask = np.array(Image.open(label_path).convert("L"))

        # my painted convention: dark pixels = panel, light pixels = background
        # Target convention (what dataset_provider.py expects): black = background,
        # white = panel
        panel_mask = mask < THRESHOLD
        converted = np.where(panel_mask, 255, 0).astype(np.uint8)

        out_name = os.path.basename(label_path)
        out_path = os.path.join(OUTPUT_DIR, out_name)
        Image.fromarray(converted).save(out_path)

        print(f"Converted: {out_name}  (panel coverage: {panel_mask.mean() * 100:.2f}%)")

    # Also copy over the matching (non-label) image files, unchanged, so
    # OUTPUT_DIR ends up as a complete, ready-to-use folder
    image_files = [
        f for f in glob.glob(os.path.join(SOURCE_DIR, "*.bmp"))
        if not f.endswith("_label.bmp")
    ]
    for img_path in image_files:
        # only copy images that actually have a corresponding label
        id_ = os.path.basename(img_path).replace(".bmp", "")
        label_path = os.path.join(SOURCE_DIR, id_ + "_label.bmp")
        if os.path.exists(label_path):
            Image.open(img_path).save(os.path.join(OUTPUT_DIR, os.path.basename(img_path)))

    print(f"\nDone. Converted dataset ready at: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()