import torch
import matplotlib.pyplot as plt
import os

BASE_PATH = r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\solar_panel_data\data"

checkpoint = torch.load(
    os.path.join(BASE_PATH, "BCEWithLogitsLoss_Adam_0.0001", "model_epoch49.pt"),
    weights_only=False
)

train_losses = checkpoint["train_losses"]
val_losses = checkpoint["val_losses"]

plt.plot(train_losses, color='b', label='Train Loss')
plt.plot(val_losses, color='r', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

import numpy as np
from PIL import Image

mask_path = r"C:\Users\mclou\PycharmProjects\DeepLearningAssignment\solar_panel_data\data\test\PV01_324958_1203801_label.bmp"  # pick a real filename
mask = np.array(Image.open(mask_path).convert("L"))
print(np.unique(mask))
