import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from provider.dataset_provider import get_loader


#set seed for reproducibility
SEED = 13

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

BASE_PATH = r"" #path to root

###########FUNCTIONS#############

#pytorch training loop; processes images, updates model weights, and returns avg loss
def train(model, loss_fn, optimizer, epoch, train_ds, device):
    model.train()

    running_loss = []

    loop = tqdm(train_ds)

    for (data, target) in loop:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(data.float())

        loss = loss_fn(prediction, target)
        loss.backward()

        optimizer.step()

        running_loss.append(loss.item())
        loop.set_postfix_str(f"Epoch {epoch}: loss is: " + str(loss.item()))

    return np.mean(running_loss)

#validation loop
def validation(model, loss_fn, epoch, val_ds, device):
    model.eval()

    running_loss = []

    loop = tqdm(val_ds)

    with torch.no_grad():
        for (data, target) in loop:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            prediction = model(data.float())
            loss = loss_fn(prediction, target)
            running_loss.append(loss.item())
            loop.set_postfix_str(f"Epoch {epoch}: loss is: " + str(loss.item()))

    return np.mean(running_loss)

# Model architecture (DoubleConv + U-Net)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.dc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.dc(x)


class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetModel, self).__init__()

        # Encoder
        self.dc1 = DoubleConv(in_channels, 64)
        self.dc2 = DoubleConv(64, 128)
        self.dc3 = DoubleConv(128, 256)
        self.dc4 = DoubleConv(256, 512)
        self.dc5 = DoubleConv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)

        # Decoder
        self.tc4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.tc3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.tc2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.tc1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dc6 = DoubleConv(1024, 512)
        self.dc7 = DoubleConv(512, 256)
        self.dc8 = DoubleConv(256, 128)
        self.dc9 = DoubleConv(128, 64)

        # out_channels=1 for binary panel/background segmentation, raw logits
        # (no sigmoid here -- BCEWithLogitsLoss applies it internally)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connection = []

        x = self.dc1(x); skip_connection.append(x); x = self.maxpool(x)
        x = self.dc2(x); skip_connection.append(x); x = self.maxpool(x)
        x = self.dc3(x); skip_connection.append(x); x = self.maxpool(x)
        x = self.dc4(x); skip_connection.append(x); x = self.maxpool(x)
        x = self.dc5(x)

        x = self.tc4(x); x = torch.cat((x, skip_connection[3]), dim=1); x = self.dc6(x)
        x = self.tc3(x); x = torch.cat((x, skip_connection[2]), dim=1); x = self.dc7(x)
        x = self.tc2(x); x = torch.cat((x, skip_connection[1]), dim=1); x = self.dc8(x)
        x = self.tc1(x); x = torch.cat((x, skip_connection[0]), dim=1); x = self.dc9(x)

        return self.final_conv(x)


################MODEL RUN#############
if __name__ == '__main__':
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #use GPU if available
    print(f"Using device: {device}")

    model = UNetModel(in_channels=3, out_channels=1).to(device) #out=1 since mask is binary

    #learning rate and early stopping config
    lr = 0.0001
    patience = 5
    patience_counter = 0

    loss_fn = nn.BCEWithLogitsLoss() #loss function type

    optim = opt.Adam(model.parameters(), lr=lr) #optimizer type

    #data loading for training and validation
    print(f"Loading data from: {BASE_PATH}")
    train_ds = get_loader(base_path=BASE_PATH, dataset_type="train")
    val_ds = get_loader(base_path=BASE_PATH, dataset_type="val")

    all_train_losses = []
    all_val_losses = []

    #creating and confirming output directory
    run_name = "30cm_augmented"  # change this label for new model
    model_path = f"{loss_fn.__class__.__qualname__}_{optim.__class__.__qualname__}_{lr}_{run_name}"
    full_save_dir = os.path.join(BASE_PATH, model_path)

    if not os.path.exists(full_save_dir):
        os.makedirs(full_save_dir, exist_ok=True)

    # main training loop
    for epoch in range(50): #range(x) where x=epoch number
        tr_loss = train(model=model, loss_fn=loss_fn, optimizer=optim, epoch=epoch, train_ds=train_ds, device=device)
        val_loss = validation(model=model, loss_fn=loss_fn, epoch=epoch, val_ds=val_ds, device=device)

        all_train_losses.append(tr_loss)
        all_val_losses.append(val_loss)

        #early stopping check
        if epoch > 0:
            if val_loss > all_val_losses[epoch - 1]:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter > patience:
                print(f"We stopped training after {epoch} epochs with a total patience of: {patience_counter}")
                break

        #saving each checkpoint individually
        save_dir = os.path.join(BASE_PATH, model_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "net_state_dict": model.state_dict(),
            "train_losses": all_train_losses,  # or whatever your dict keys are
            "val_losses": all_val_losses
        }, os.path.join(save_dir, f"model_epoch{epoch}.pt"))

    #plot loss curves
    plt.plot(all_train_losses, color='b')
    plt.plot(all_val_losses, color='r')
    plt.show()

