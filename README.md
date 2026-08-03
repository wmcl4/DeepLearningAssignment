# Solar Panel Segmentation from Aerial Imagery (U-Net)

Binary semantic segmentation of rooftop and ground-mounted solar photovoltaic (PV) panels from RGB aerial imagery, using a U-Net convolutional neural network implemented in PyTorch.



## 1. Introduction & Motivation

Accurately mapping the location and extent of solar photovoltaic installations is valuable for utilities planning grid integration, policymakers tracking renewable energy adoption, and researchers estimating solar potential across regions. Manually annotating panels across large aerial or satellite image collections doesn't scale to city-, regional-, or national-level coverage.

This project frames the problem as **binary semantic segmentation**: given an RGB aerial image tile, predict a pixel-wise mask indicating which pixels belong to a solar panel versus background (roofs, vegetation, roads, open ground).



## 2. Dataset & Network Architecture

### Dataset

**PV01 / PV03 Solar Panel Segmentation Dataset**
*Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery* — https://zenodo.org/record/5171712
- RGB image tiles paired with binary segmentation masks (`_label` suffix)
- Mask values: 255 (white) = solar panel, 0 (black) = background
- Subsets are named by approximate ground sample distance: **PV01 (~0.1 m/pixel)** and **PV03 (~0.3 m/pixel)**
- 645 image/mask pairs, split 70/20/10 into train/validation/test at the image level with a fixed random seed for reproducibility

**NAIP (National Agriculture Imagery Program)**
Aerial imagery program run by the USDA Farm Service Agency, covering the United States — https://www.fsa.usda.gov/resources/programs/national-agriculture-imagery-program-naip
- Four-band (RGB + near-infrared) aerial orthophotos
- Native spatial resolution of ~0.6–1 m/pixel
- A single NAIP tile covering Oregon, USA (`m_4412042_sw_10_030_20220628.tif`) was used to test the model on real, out-of-distribution imagery


### Network Architecture

**U-Net** (Ronneberger et al., 2015 — *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)), implemented in PyTorch.

- **Encoder:** 5 stages of double 3×3 convolutions (batch norm + ReLU), with 2×2 max pooling between stages — channel depth 64 → 128 → 256 → 512 → 1024
- **Decoder:** 4 stages of 2×2 transposed convolutions + skip connections from the matching encoder stage + double convolutions — channel depth 1024 → 512 → 256 → 128 → 64
- **Output:** 1×1 convolution producing a single-channel raw logit map 




## 3. What We Did

We fed 645 image/mask pairs from the PV01 dataset into a U-Net trained from scratch, resizing all images and masks to 512×512. Training used `BCEWithLogitsLoss`, Adam (lr = 1e-4), batch size 4, for up to 50 epochs with early stopping (patience = 5 epochs on validation loss). Model checkpoints were saved after every epoch, and the best-performing epoch was selected based on lowest validation loss for final evaluation.

Training and inference both run on GPU (CUDA) where available, with automatic CPU fallback. 



## 4. Results

**Training / Validation Loss:**

<img width="1140" height="824" alt="Screenshot 2026-07-26 170651" src="https://github.com/user-attachments/assets/8dab3086-2677-4d2a-ae1c-7f297330b052" />

<br><br>

**Qualitative predictions:**
<img width="1418" height="446" alt="Screenshot 2026-07-26 170916" src="https://github.com/user-attachments/assets/f6cfa61c-846e-4e7d-99c0-b7f9eba83fa2" />

<img width="1414" height="444" alt="Screenshot 2026-07-26 171353" src="https://github.com/user-attachments/assets/56f7c775-347f-4fee-b689-af5eef4da5b3" />

<br><br>

**Segmentation Performance Metrics**
| Metric | Value |
|---|---|
| IoU (test set) | 0.918 |
| Dice Coefficient | 0.957 |
| F1 Score | 0.957 |
| Accuracy | 0.975 |
| Precision | 0.982 |
| Recall | 0.934 |

<br><br>

**Confusion Matrix**
| Metric | Count | Percent |
|---|---|---|
| True Positive | 4753280 | 27.90 |
| True Negative | 11861087 | 69.61 |
| False Positive | 89561 | 0.53 |
| False Negative | 335432 | 1.97 |



## 5. Extension: Cross-Resolution and Real-World Generalization

Beyond the core PV01 benchmark results above, we tested how well the trained model generalizes beyond its training distribution — both to a different resolution tier of the same dataset, and to real-world U.S. aerial imagery it had never seen.

**Method:** The U-Net was retrained on PV03 (0.3m resolution) using all six of its sub-categories combined (five ground-cover types plus rooftop), then evaluated directly against 0.3m NAIP aerial imagery of Arlington, Oregon — chosen to match PV03's resolutionr. A small set of hand-labeled NAIP tiles (~50) was then used to fine-tune the PV03 model on this new domain.

**Results:** Direct transfer of PV03 model to NAIP imagery was weak, consistent with the source dataset's own reported finding that cross-domain transfer without fine-tuning is unreliable. Fine-tuning on the small labeled NAIP set meaningfully improved detection, though there are some limitations. This work is still in progress.

The scripts used to fine-tune the PV03 model on hand-labeled NAIP tiles and run
it against full NAIP scenes are documented separately (`NAIP_README.md`), since
they form their own small pipeline. Summary:

**Quick start (run inference only, using the provided checkpoint):**
Install dependencies, point `4_NAIP_generalizer.py` at the checkpoint and a NAIP
`.tif` file, and run it. Output is a single stitched image with detected
panels highlighted in red. Email me (Will) to request a model epoch if you want to skip training the model yourself.

**Full pipeline (to retrain or extend with new labels), run in order:**
1. `1_NAIP_Mask_Converter.py` — converts hand-painted masks into the
   black-background/white-panel convention the pipeline expects
2. `2_NAIP_split_trainingdata.py` — splits labeled tiles into train/val (80/20,
   fixed seed)
3. `3_NAIP_tuner.py` — fine-tunes from the PV03 checkpoint (low
   learning rate, early stopping)
4. `4_NAIP_generalizer.py` — runs the resulting model against a full scene

Each image requires a matching `_label.bmp` mask of the same dimensions. See
`NAIP_README.md` for full configuration details, requirements, and notes.









