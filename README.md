# Solar Panel Segmentation from Aerial Imagery (U-Net)

Binary semantic segmentation of rooftop and ground-mounted solar photovoltaic (PV) panels from RGB aerial imagery, using a U-Net convolutional neural network implemented from scratch in PyTorch.

---

## 1. Introduction & Motivation

Accurately mapping the location and extent of solar photovoltaic installations is valuable for utilities planning grid integration, policymakers tracking renewable energy adoption, and researchers estimating solar potential across regions. Manually annotating panels across large aerial or satellite image collections doesn't scale to city-, regional-, or national-level coverage.

This project frames the problem as **binary semantic segmentation**: given an RGB aerial image tile, predict a pixel-wise mask indicating which pixels belong to a solar panel versus background (roofs, vegetation, roads, open ground).

---

## 2. Dataset & Network Architecture

### Dataset

**PV01 / PV03 Solar Panel Segmentation Dataset**
*Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery* — https://zenodo.org/record/5171712

- RGB image tiles paired with binary segmentation masks (`_label` suffix)
- Mask values: 255 (white) = solar panel, 0 (black) = background
- Subsets are named by approximate ground sample distance: **PV01 (~0.1 m/pixel)** and **PV03 (~0.3 m/pixel)** — two different spatial resolutions of the same benchmark, not simply two geographic batches
- ~645 image/mask pairs, split 70/20/10 into train/validation/test at the image level with a fixed random seed for reproducibility

### Network Architecture

**U-Net** (Ronneberger et al., 2015 — *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)), implemented from scratch in PyTorch.

- **Encoder:** 5 stages of double 3×3 convolutions (batch norm + ReLU), with 2×2 max pooling between stages — channel depth 64 → 128 → 256 → 512 → 1024
- **Decoder:** 4 stages of 2×2 transposed convolutions + skip connections from the matching encoder stage + double convolutions — channel depth 1024 → 512 → 256 → 128 → 64
- **Output:** 1×1 convolution producing a single-channel raw logit map (no sigmoid applied internally — paired with `BCEWithLogitsLoss`; `torch.sigmoid()` applied manually at inference time)
- **Input size:** 512×512 (resized from native tile resolution)

---

## 3. What We Did

We fed approximately 645 image/mask pairs from the PV01/PV03 dataset into a U-Net trained from scratch, resizing all images and masks to 512×512. Training used `BCEWithLogitsLoss`, Adam (lr = 1e-4), batch size 4, for up to 50 epochs with early stopping (patience = 5 epochs on validation loss). Model checkpoints were saved after every epoch, and the best-performing epoch was selected based on lowest validation loss for final evaluation.

Training and inference both run on GPU (CUDA) where available, with automatic CPU fallback. 

---

## 4. Results

| Metric | Value |
|---|---|
| IoU (test set) | — |
| Dice Coefficient | 0.957 |
| F1 Score | 0.957 |
| Accuracy | 0.975 |
| Precision | 0.982 |
| Recall | 0.934 |

|Metric	       |Count	   |Percent |
|---|---|---|
|True Positive	 |4753280	|27.895883 |
|True Negative	 |11861087	|69.609933 |
|False Positive |89561	   |0.525612 |
|False Negative |335432	   |1.968572 |


**Training / Validation Loss:**

<img width="1140" height="824" alt="Screenshot 2026-07-26 170651" src="https://github.com/user-attachments/assets/8dab3086-2677-4d2a-ae1c-7f297330b052" />

**Qualitative predictions:**
<img width="1418" height="446" alt="Screenshot 2026-07-26 170916" src="https://github.com/user-attachments/assets/f6cfa61c-846e-4e7d-99c0-b7f9eba83fa2" />

<img width="1414" height="444" alt="Screenshot 2026-07-26 171353" src="https://github.com/user-attachments/assets/56f7c775-347f-4fee-b689-af5eef4da5b3" />

---

## Runnable Example

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib scikit-image pillow tqdm

# Expected folder structure:
# data/
# ├── train/
# │   ├── PV01_....bmp
# │   ├── PV01_..._label.bmp
# │   └── ...
# ├── val/
# └── test/

# Train
python training.py

# Evaluate on test set (IoU, Dice, F1, accuracy, precision, recall + visualizations)
python test_solar_panel_unet.py

# Run inference on a new data, full-size aerial image
python predict_full_image.py
```

---






















