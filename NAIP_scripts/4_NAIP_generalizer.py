import os
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from PIL import Image
from main_training_script import UNetModel

# ---- CONFIG ----
NAIP_TIFF_PATH = r"C:\Users\mclou\Downloads\oregon_panels_NAIP\m_4412042_sw_10_030_20220628\m_4412042_sw_10_030_20220628.tif"
BASE_PATH = r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\solar_panel_data\data"
MODEL_FOLDER = "BCEWithLogitsLoss_Adam_0.0001_augmented"
CHECKPOINT_NAME = "model_epoch48.pt"

TILE_SIZE = 512
THRESHOLD = 0.4

OUTPUT_DIR = r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\naip_results"
STITCHED_FILENAME = "untuned_model_.png"
OVERLAY_ALPHA = 0.6  # opacity of the red detection overlay, only where panels are predicted
# ----------------


def load_model(device):
    model = UNetModel(in_channels=3, out_channels=1).to(device)
    state_dict = torch.load(
        os.path.join(BASE_PATH, MODEL_FOLDER, CHECKPOINT_NAME),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(state_dict["net_state_dict"])
    model.eval()
    return model


def get_tiles(width, height, tile_size):
    """
    Yields (row_off, col_off) for a grid of non-overlapping tiles covering
    the full raster. Edge strips smaller than tile_size are skipped, so the
    stitched output is trimmed to a whole number of tiles.
    """
    for row_off in range(0, height - tile_size + 1, tile_size):
        for col_off in range(0, width - tile_size + 1, tile_size):
            yield row_off, col_off


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(device)

    with rasterio.open(NAIP_TIFF_PATH) as src:
        print(f"Raster size: {src.width} x {src.height}, bands: {src.count}")

        n_tiles_x = src.width // TILE_SIZE
        n_tiles_y = src.height // TILE_SIZE
        canvas_w = n_tiles_x * TILE_SIZE
        canvas_h = n_tiles_y * TILE_SIZE

        print(f"Stitched output will be {canvas_w} x {canvas_h} "
              f"({n_tiles_x * n_tiles_y} tiles)")

        # Pre-allocate the full base image and a matching detection mask
        base_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        full_mask = np.zeros((canvas_h, canvas_w), dtype=bool)

        tile_list = list(get_tiles(src.width, src.height, TILE_SIZE))
        total_detected_px = 0

        for i, (row_off, col_off) in enumerate(tile_list):
            window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
            tile = src.read([1, 2, 3], window=window)  # (3, H, W), NAIP is often 4-band

            if tile.max() == 0:
                continue  # skip nodata/blank tiles

            rgb_tensor = torch.from_numpy(tile.copy()).float() / 255.0
            rgb_tensor = rgb_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(rgb_tensor)
                pred = torch.sigmoid(pred)

            pred_np = pred.detach().cpu().squeeze().numpy()
            pred_mask = pred_np > THRESHOLD
            total_detected_px += pred_mask.sum()

            # Place this tile into the full canvas at the correct position
            tile_rgb_hwc = np.transpose(tile, (1, 2, 0))  # (H, W, 3)
            base_rgb[row_off:row_off + TILE_SIZE, col_off:col_off + TILE_SIZE] = tile_rgb_hwc
            full_mask[row_off:row_off + TILE_SIZE, col_off:col_off + TILE_SIZE] = pred_mask

            if (i + 1) % 20 == 0 or (i + 1) == len(tile_list):
                print(f"  Processed {i + 1}/{len(tile_list)} tiles...")

        print(f"\nTotal predicted panel pixels across whole image: {total_detected_px}")

        # Build the final overlay: true transparency where nothing was detected,
        # semi-opaque red only where the model predicted a panel
        base_float = base_rgb.astype(np.float32) / 255.0
        red = np.array([1.0, 0.0, 0.0])

        blended = base_float.copy()
        blended[full_mask] = (
            (1 - OVERLAY_ALPHA) * base_float[full_mask] + OVERLAY_ALPHA * red
        )

        out_img = Image.fromarray((blended * 255).astype(np.uint8))
        out_path = os.path.join(OUTPUT_DIR, STITCHED_FILENAME)
        out_img.save(out_path)

        print(f"Saved stitched overlay to: {out_path}")


if __name__ == '__main__':
    main()