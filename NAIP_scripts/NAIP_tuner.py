import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from provider.dataset_provider import get_loader
from classcode_training import UNetModel, train, validation

SEED = 13

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---- CONFIG ----
NAIP_DATA_PATH = r"/naip_finetune_data"

# The already-trained PV03 model to start from
PV03_BASE_PATH = r"/data/PV03"
PV03_MODEL_FOLDER = "BCEWithLogitsLoss_Adam_0.0001_30cm_augmented"
PV03_CHECKPOINT = "model_epoch45.pt"

# Fine-tuning settings -- much lower LR, far fewer epochs than training from scratch
FINETUNE_LR = 0.00001  # 10x lower than the original 0.0001
FINETUNE_EPOCHS = 500
PATIENCE = 5

# Distinct run name so this saves separately from the frozen-encoder run,
# letting you compare both directly afterward
RUN_NAME = "naip_finetuned"
# ----------------


if __name__ == '__main__':
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the architecture, then load the PV03 trained weights into it
    model = UNetModel(in_channels=3, out_channels=1).to(device)

    pv03_checkpoint = torch.load(
        os.path.join(PV03_BASE_PATH, PV03_MODEL_FOLDER, PV03_CHECKPOINT),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(pv03_checkpoint["net_state_dict"])
    print(f"Loaded starting weights from: {PV03_MODEL_FOLDER}/{PV03_CHECKPOINT}")

    # NOTE: no encoder freeze here -- every layer is trainable, for
    # direct comparison against the frozen-encoder run
    print("Encoder NOT frozen -- fine-tuning the entire network.")

    loss_fn = nn.BCEWithLogitsLoss()
    optim = opt.Adam(model.parameters(), lr=FINETUNE_LR)

    print(f"Loading NAIP fine-tuning data from: {NAIP_DATA_PATH}")
    train_ds = get_loader(base_path=NAIP_DATA_PATH, dataset_type="train")
    val_ds = get_loader(base_path=NAIP_DATA_PATH, dataset_type="val")

    all_train_losses = []
    all_val_losses = []

    model_path = f"{loss_fn.__class__.__qualname__}_{optim.__class__.__qualname__}_{FINETUNE_LR}_{RUN_NAME}"
    full_save_dir = os.path.join(NAIP_DATA_PATH, model_path)

    if not os.path.exists(full_save_dir):
        os.makedirs(full_save_dir, exist_ok=True)

    patience_counter = 0

    for epoch in range(FINETUNE_EPOCHS):
        tr_loss = train(model=model, loss_fn=loss_fn, optimizer=optim, epoch=epoch, train_ds=train_ds, device=device)
        val_loss = validation(model=model, loss_fn=loss_fn, epoch=epoch, val_ds=val_ds, device=device)

        torch.cuda.empty_cache()

        all_train_losses.append(tr_loss)
        all_val_losses.append(val_loss)

        if epoch > 0:
            if val_loss > all_val_losses[epoch - 1]:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter > PATIENCE:
                print(f"Stopped fine-tuning after {epoch} epochs (patience {patience_counter})")
                break

        torch.save({
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "train_losses": all_train_losses,
            "val_losses": all_val_losses
        }, os.path.join(full_save_dir, f"model_epoch{epoch}.pt"))

    plt.plot(all_train_losses, color='b', label='Train Loss')
    plt.plot(all_val_losses, color='r', label='Val Loss')
    plt.legend()
    plt.title("NAIP Fine-tuning Loss (Unfrozen Encoder)")
    plt.show()