# NAIP Solar Panel Fine-Tuning Pipeline

This folder contains the scripts used to fine-tune a PV03-trained U-Net solar
panel segmentation model on real-world NAIP aerial imagery, and to run that
model against a full NAIP scene.

---

## Quick Start: Just Run Inference (Pretrained Model Provided)

If you have a pretrained checkpoint (`.pt` file) and just want to run
it against a NAIP image, you only need **`4_NAIP_generalizer`**. You do **not**
need to run any of the training/data-prep scripts below.

### Steps

1. Install dependencies:
   ```bash
   pip install torch torchvision rasterio pillow numpy
   ```

2. Place the provided model checkpoint file somewhere on disk, and note its path.

3. Open `4_NAIP_generalizer.py` and update the config block at the top:
   ```python
   NAIP_TIFF_PATH = r"path\to\your\naip_image.tif"       # the full NAIP scene you want to test
   BASE_PATH = r"path\to\folder\containing\model_folder"  # parent folder of MODEL_FOLDER
   MODEL_FOLDER = "name_of_the_folder_the_checkpoint_is_in"
   CHECKPOINT_NAME = "model_epochXX.pt"                    # exact filename of the checkpoint
   ```
   `BASE_PATH` + `MODEL_FOLDER` + `CHECKPOINT_NAME` should combine to the full
   path of the `.pt` file you were given.

4. Run it:
   ```bash
   python 4_NAIP_generalizer.py
   ```

5. Output: a single stitched image (`STITCHED_FILENAME`, saved to `OUTPUT_DIR`)
   showing the full NAIP scene with detected solar panels highlighted in red.

### Notes
- `THRESHOLD` (default 0.4) controls how confident the model must be before a
  pixel counts as "panel." Lower = more detections (and more false positives),
  higher = fewer, more conservative detections.
- Large NAIP scenes can take a few minutes to process, and the stitched output
  image itself may be large (hundreds of MB of pixels). Progress is printed
  every 20 tiles.
- Requires a `main_training_script.py` file in the same directory (or importable
  on your path) containing the matching `UNetModel` class definition, since the
  script imports the architecture from there before loading the checkpoint's
  weights.

---

## Full Pipeline (For Retraining / Further Fine-Tuning)

The scripts below were used to produce the fine-tuned model. Run them **in
this order**:

### 1. `1_NAIP_Mask_Converter`
Converts hand-painted label masks (white background, black panel shapes) into
the black-background/white-panel convention the training pipeline expects.

```python
SOURCE_DIR = r"path\to\your\painted_tiles"
OUTPUT_DIR = r"path\to\converted_output"
```

Run:
```bash
python 1_NAIP_Mask_Converter.py
```

Safe to re-run any time you add new labeled tiles — it overwrites cleanly and
won't create duplicates.

### 2. `2_NAIP_split_trainingdata`
Splits the converted, labeled tiles into `train`/`val` folders (80/20 split,
fixed random seed for reproducibility).

```python
source_dir = Path(r"path\to\converted_output")   # matches step 1's OUTPUT_DIR
output_dir = Path(r"path\to\finetune_data")
```

Run:
```bash
python 2_NAIP_split_trainingdata.py.py
```

Re-run this any time the labeled set changes, so `train`/`val` stay in sync
with your current labels. It clears old split contents before rebuilding, so
stale files from previous runs won't linger.

### 3. `3_NAIP_tuner`
Loads a previously-trained model checkpoint as a starting point and fine-tunes
it on the NAIP train/val split.

```python
NAIP_DATA_PATH = r"path\to\finetune_data"          # matches step 2's output_dir
PV03_BASE_PATH = r"path\to\PV03\model\folder"
PV03_MODEL_FOLDER = "name_of_base_model_folder"
PV03_CHECKPOINT = "model_epochXX.pt"

FINETUNE_LR = 0.00001
FINETUNE_EPOCHS = 500
PATIENCE = 5
RUN_NAME = "naip_finetuned"
```

Run:
```bash
python 3_NAIP_tuner.py
```

This trains with early stopping (stops automatically if validation loss
doesn't improve for `PATIENCE` consecutive epochs) and saves a checkpoint
after every epoch to `NAIP_DATA_PATH/<generated_folder_name>/model_epochN.pt`.
A loss curve plot is shown at the end of training.

To find the best-performing checkpoint rather than assuming the last epoch is
best, load any saved checkpoint and check its stored `val_losses` list for the
epoch with the lowest value.

### 4. `4_NAIP_generalizer`
See the **Quick Start** section above — point its config at the checkpoint
produced by step 3 to visualize the fine-tuned model's predictions.

---

## Requirements

```bash
pip install torch torchvision rasterio pillow numpy tqdm matplotlib
```

A CUDA-capable GPU is strongly recommended for training (step 3); inference
(step 4) will run on CPU but more slowly.

## File Structure Expected by the Pipeline

```
<data_dir>/
├── train/
│   ├── tile_name.bmp
│   ├── tile_name_label.bmp
│   └── ...
└── val/
    ├── tile_name.bmp
    ├── tile_name_label.bmp
    └── ...
```

Each image (`.bmp`) must have a matching `_label.bmp` mask of the same
dimensions, with panel pixels marked white (255) and background black (0).
