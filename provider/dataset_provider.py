import glob
import os
import random
import torch
import torch.utils.data as td
import skimage.io as io
import numpy as np
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# Dataset provider (DoubleConv-style U-Net dataset + augmentation)

#ensures images are identical size
custom_transform = transforms.Compose([
    transforms.Resize((512, 512))
])


#loads image-mask pairs and confirms validity; applies augmentation
class RooftopPanels(td.Dataset):
    def __init__(self, base_path, dataset_type):
        self.dataset = []
        self.augment = (dataset_type == "train")  # only augment training data

        split_dir = os.path.join(base_path, dataset_type)

        # Grab all .bmp files that are NOT label masks
        all_files = glob.glob(os.path.join(split_dir, "*.bmp"))
        rgb_files = [f for f in all_files if not f.endswith("_label.bmp")]

        skipped = []

        #load image pairs into memory
        for rgb_file in tqdm(rgb_files):
            id_ = os.path.basename(rgb_file).replace('.bmp', '')

            mask_file = os.path.join(split_dir, id_ + "_label.bmp")

            #skip if either file is missing or 0 bytes
            if not os.path.exists(mask_file):
                skipped.append((rgb_file, "missing mask"))
                continue

            if os.path.getsize(rgb_file) == 0:
                skipped.append((rgb_file, "0-byte image"))
                continue

            if os.path.getsize(mask_file) == 0:
                skipped.append((rgb_file, "0-byte mask"))
                continue
            #attempt to open image pairs to confirm validity
            try:
                rbg_read = np.array(Image.open(rgb_file).convert("RGB"))
                mask_read = np.array(Image.open(mask_file).convert("L"))
            except Exception as e:
                skipped.append((rgb_file, f"failed to open: {e}"))
                continue

            #convert to numpy
            rgb_read = np.transpose(rbg_read, (2, 0, 1))
            mask_read = np.expand_dims(mask_read, axis=0)

            #convert to pytorch tensor
            rgb_read = torch.from_numpy(rgb_read.copy()).float()
            mask_read = torch.from_numpy(mask_read.copy()).float()

            #resize image and normalize values
            rgb_read = custom_transform(rgb_read) / 255.0
            mask_read = custom_transform(mask_read)
            mask_read = (mask_read > 0).float()

            self.dataset.append((rgb_read, mask_read))

        #report skipped files
        if skipped:
            print(f"Skipped {len(skipped)} file(s) in '{dataset_type}':")
            for f, reason in skipped[:20]:
                print(f"  {f} -> {reason}")
            if len(skipped) > 20:
                print(f"  ...and {len(skipped) - 20} more")

    def __len__(self):
        return len(self.dataset) #total number of pairs

    def __getitem__(self, index):
        rgb, mask = self.dataset[index]

        if self.augment:
            rgb, mask = self._apply_augmentation(rgb, mask)

        return rgb, mask

    #applying augmentations to improve results
    @staticmethod
    def apply_augmentation(rgb, mask):
        # Random horizontal flip
        if random.random() < 0.5:
            rgb = TF.hflip(rgb)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() < 0.5:
            rgb = TF.vflip(rgb)
            mask = TF.vflip(mask)

        # Random 90-degree rotation (0, 90, 180, or 270)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            rgb = torch.rot90(rgb, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        return rgb, mask

def get_loader(base_path, dataset_type):
    dataset = RooftopPanels(base_path, dataset_type)

    return td.DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
    )