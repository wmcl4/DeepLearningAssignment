#Solar Panel Segmentation from Aerial Imagery (U-Net)

## 1. Introduction & Motivation

Binary semantic segmentation of rooftop and ground-mounted solar photovoltaic (PV) panels from RGB aerial imagery, using a U-Net convolutional neural network

Given an aerial image tile, the model predicts a pixel-wise mask indicating which pixels belong to a solar panel versus background (roofs, vegetation, roads, open ground).

---

## Installation

Create a Python virtual environment and install dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-image pillow tqdm
```

If running on Google Colab, GPU support (CUDA) is enabled automatically if the runtime has one attached (**Runtime > Change runtime type > GPU**).

---

## How to start

1. Place your dataset under `data/`, organized into pre-split subfolders:
   ```
   data/
   ├── train/
   │   ├── PV01_....bmp
   │   ├── PV01_..._label.bmp
   │   └── ...
   ├── val/
   └── test/
   ```
2. Update `BASE_PATH` in `train_solar_panel_unet.py` to point to your data folder.
3. Run training:
   ```bash
   python train_solar_panel_unet.py
   ```
4. Once trained, evaluate on the held-out test set (reports IoU, Dice, F1, accuracy, precision, recall, plus visual predictions):
   ```bash
   python test_solar_panel_unet.py
   ```
---

## Data Source

**PV01 / PV03 Solar Panel Segmentation Dataset**
*Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery* — https://zenodo.org/record/5171712

- RGB image tiles paired with binary segmentation masks (`_label` suffix)
- Mask values: 255 (white) = solar panel, 0 (black) = background
- Dataset subsets are named by approximate ground sample distance: **PV01 (~0.1 m/pixel)**, **PV03 (~0.3 m/pixel)

---

## Model

**U-Net** (Ronneberger et al., 2015 — *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)), implemented in PyTorch.

- **Encoder:** 5 stages of double 3×3 convolutions (batch norm + ReLU), with 2×2 max pooling between stages — channel depth 64 → 128 → 256 → 512 → 1024
- **Decoder:** 4 stages of 2×2 transposed convolutions + skip connections from the matching encoder stage + double convolutions — channel depth 1024 → 512 → 256 → 128 → 64
- **Output:** 1×1 convolution producing a single-channel raw logit map (no sigmoid applied internally — paired with `BCEWithLogitsLoss` for numerical stability; apply `torch.sigmoid()` manually at inference time)
- **Input size:** 512×512 (resized from native tile resolution)

---

## Training Setup

| | |
|---|---|
| Loss function | `MSE` |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Max epochs | 50 |
| Early stopping | Patience = 5 epochs (validation loss) |
| Hardware | GPU (CUDA) where available, CPU fallback |

Checkpoints (model weights, optimizer state, loss history) are saved after every epoch, allowing training to be resumed or the best-performing epoch to be selected retrospectively based on lowest validation loss.

---

## Results

*(Insert final values)*

| Metric | Value |
|---|---|
| IoU (test set) | — |
| Dice Coefficient | — |
| F1 Score | — |
| Accuracy | — |
| Precision | — |
| Recall | — |

**Training / Validation Loss:**

`[insert loss curve image here]`

**Qualitative predictions:**

`[insert original / prediction / ground truth comparison images here]`

---

## 🌐 Project Organization

```
├── data/                          <- Dataset (train/val/test subfolders)
├── model/
│   ├── unet_layers.py             <- DoubleConv building block
│   └── unet_model.py              <- Full U-Net architecture
├── provider/
│   └── solar_dataset_provider.py  <- Dataset loading, preprocessing, augmentation
├── train_solar_panel_unet.py      <- Training script
├── test_solar_panel_unet.py       <- Test set evaluation (IoU/Dice/F1 + visualizations)
├── predict_full_image.py          <- Tiled inference on full-size, out-of-distribution images
└── README.md
```
























