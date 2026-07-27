# Splits the data into three groups: Training, Validation and Test

# Load libraries

import os
import shutil
import random

# Load data

root = r"C:\Users\fhlou\Documents\Deep_learning_solar\PV01"              # path to data
base = r"C:\Users\fhlou\Documents\Deep_learning_solar\PV1_split"         # new split path
os.makedirs(os.path.join(base, "train"), exist_ok=True)                  
os.makedirs(os.path.join(base, "val"), exist_ok=True)
os.makedirs(os.path.join(base, "test"), exist_ok=True)

pairs = []

# Find image pairs

for current_path, dirs, files in os.walk(root):
    for f in files:
        if f.endswith(".bmp") and not f.endswith("_label.bmp"):
            img = os.path.join(current_path, f)

            # possible image-file endings
            candidates = [
                f.replace(".bmp", "_label.bmp"),
                f.replace(".bmp", "_LABEL.bmp"),
                f.replace(".bmp", "_mask.bmp"),
                f.replace(".bmp", "_seg.bmp"),
                f.replace(".bmp", "_label.tif"),
                f.replace(".bmp", "_label.png"),
            ]

            lbl = None
            for c in candidates:
                p = os.path.join(current_path, c)
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    lbl = p
                    break

            # accept only real pairs
            if os.path.exists(img) and os.path.getsize(img) > 0 and lbl:
                pairs.append((img, lbl))

print("Valid pairs found:", len(pairs))

# Create Split

random.shuffle(pairs)

total = len(pairs) # total number of data pairs
# Calculate split boundaries
train_end = int(0.7 * total) # 70% of data is for training 
val_end = int(0.9 * total) # 90% of data is for draining + validation

train = pairs[:train_end] # takes the first 70% for training
val = pairs[train_end:val_end] # takes the next 20% for validation
test = pairs[val_end:] # takes the remaining 10% for testing 

# Copy function

def safe_copy(pairs, folder):
    for img, lbl in pairs:
        # Copy only if the file exists and is >0 bytes
        if os.path.exists(img) and os.path.getsize(img) > 0:
            shutil.copy(img, os.path.join(base, folder, os.path.basename(img)))

        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            shutil.copy(lbl, os.path.join(base, folder, os.path.basename(lbl)))

safe_copy(train, "train")
safe_copy(val, "val")
safe_copy(test, "test")

print("Split complete")
