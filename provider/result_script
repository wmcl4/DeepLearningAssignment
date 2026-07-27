import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from provider.dataset_provider import get_loader
from classcode_training import UNetModel

BASE_PATH = r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\data\PV03"

# Change this to match the run_name/folder you want to evaluate
MODEL_FOLDER = "BCEWithLogitsLoss_Adam_0.0001_30cm_augmented"
CHECKPOINT_NAME = "model_epoch45.pt"

THRESHOLD = 0.4   # probability above this = predicted "panel"
SHOW_PLOTS = False  # set True to view prediction/ground-truth images; False to just compute metrics fast


def compute_iou_dice(pred_mask, true_mask, eps=1e-7):
    """
    pred_mask, true_mask: boolean numpy arrays, same shape (H, W)
    Returns (iou, dice) for a single image.
    """
    intersection = (pred_mask & true_mask).sum()
    union = (pred_mask | true_mask).sum()
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_mask.sum() + true_mask.sum() + eps)
    return iou, dice


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_ds = get_loader(
        base_path=BASE_PATH,
        dataset_type="val",
    )

    # --- DEBUG SECTION ---
    sample_data, sample_target = next(iter(test_ds))
    print(f"Data shape: {sample_data.shape}")
    print(f"Target shape: {sample_target.shape}")
    print(f"Target Min: {sample_target.min().item()}, Max: {sample_target.max().item()}")
    print(f"Non-zero target pixels: {(sample_target > 0).sum().item()}")
    # ---------------------

    model = UNetModel(in_channels=3, out_channels=1).to(device)

    state_dict = torch.load(
        os.path.join(BASE_PATH, MODEL_FOLDER, CHECKPOINT_NAME),
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(state_dict["net_state_dict"])
    model.eval()

    loop = tqdm(test_ds)

    all_ious = []
    all_dices = []
    ious_panel_only = []
    dices_panel_only = []

    total_intersection = 0
    total_union = 0
    total_pred_sum = 0
    total_true_sum = 0

    for (data, target) in loop:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            prediction = model(data.float())
            prediction = torch.sigmoid(prediction)

        prediction = prediction.detach().cpu().squeeze(1).numpy()
        target = target.detach().cpu().squeeze(1).numpy()

        for i in range(prediction.shape[0]):
            pred_img = prediction[i]
            true_img = target[i]

            pred_mask = pred_img > THRESHOLD
            true_mask = true_img > 0.5  # target is already ~0/1, but keep this robust

            iou, dice = compute_iou_dice(pred_mask, true_mask)
            all_ious.append(iou)
            all_dices.append(dice)

            # Track dataset-wide (aggregate) counts for a more robust summary
            total_intersection += (pred_mask & true_mask).sum()
            total_union += (pred_mask | true_mask).sum()
            total_pred_sum += pred_mask.sum()
            total_true_sum += true_mask.sum()

            # Also track metrics only on images that actually contain a panel,
            # since empty/empty matches trivially score 1.0 and can inflate the average
            if true_mask.sum() > 0:
                ious_panel_only.append(iou)
                dices_panel_only.append(dice)

            loop.set_postfix_str(f"IoU: {iou:.3f} | Dice: {dice:.3f}")

            if SHOW_PLOTS:
                plt.imshow(pred_img, cmap="viridis", vmin=0, vmax=1)
                plt.colorbar()
                plt.title(f"Prediction (IoU={iou:.3f}, Dice={dice:.3f})")
                plt.show()

                plt.imshow(true_img, cmap="viridis")
                plt.title("Ground Truth")
                plt.show()

    mean_iou = sum(all_ious) / len(all_ious)
    mean_dice = sum(all_dices) / len(all_dices)

    eps = 1e-7
    dataset_iou = (total_intersection + eps) / (total_union + eps)
    dataset_dice = (2 * total_intersection + eps) / (total_pred_sum + total_true_sum + eps)

    n_panel_images = len(ious_panel_only)
    n_empty_images = len(all_ious) - n_panel_images


    print(f"\n=== Test Set Summary ({MODEL_FOLDER}, threshold={THRESHOLD}) ===")
    print(f"Images evaluated: {len(all_ious)}  ({n_panel_images} with panels, {n_empty_images} background-only)")
    print(f"\n[Per-image average, ALL images -- inflated by trivial empty/empty matches]")
    print(f"Mean IoU:  {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")

    if n_panel_images > 0:
        print(f"\n[Per-image average, PANEL-CONTAINING images only -- more meaningful]")
        print(f"Mean IoU:  {sum(ious_panel_only) / n_panel_images:.4f}")
        print(f"Mean Dice: {sum(dices_panel_only) / n_panel_images:.4f}")

    print(f"\n[Dataset-wide aggregate -- pixels pooled across all images, most robust]")
    print(f"IoU:  {dataset_iou:.4f}")
    print(f"Dice: {dataset_dice:.4f}")

# after this you would typically download rgb imagery and crop exactly into the training slices (512x512)
