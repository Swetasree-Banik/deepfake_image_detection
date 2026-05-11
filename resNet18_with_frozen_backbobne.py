import os
import copy
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split
)

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay
)

import numpy as np
import timm
from tqdm import tqdm
import kagglehub

from torch.cuda.amp import (
    autocast,
    GradScaler
)

from cifake_dataloader_fft import CIFAKEDataset

# Config

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAVE_DIR = "outputs_fully_frozen_backbone"

os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "best_model.pth"
)

RESOLUTION = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
VAL_SPLIT = 0.2

PATIENCE = 4

torch.manual_seed(42)

# Model

class AdapterConv(nn.Module):

    def __init__(
        self,
        in_channels=5,
        out_channels=3
    ):

        super().__init__()

        self.adapter = nn.Sequential(

            nn.Conv2d(
                in_channels,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                out_channels,
                kernel_size=1
            )
        )

    def forward(self, x):

        return self.adapter(x)


class FrozenResNet18FFT(nn.Module):

    def __init__(self):

        super().__init__()

        # 5-channel -> 3-channel Adaptation
        self.adapter = AdapterConv(
            in_channels=5,
            out_channels=3
        )

        # Pretrained Backbone
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=1
        )

        # Freezing Entire Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):

        x = self.adapter(x)

        return self.backbone(x)

# Loading Dataloader

path = kagglehub.dataset_download(
    "birdy654/cifake-real-and-ai-generated-synthetic-images"
)

full_dataset = CIFAKEDataset(
    dataset_path=path,
    split="train",
    resolution=RESOLUTION,
    augment=True
)

test_dataset = CIFAKEDataset(
    dataset_path=path,
    split="test",
    resolution=RESOLUTION,
    augment=False
)

val_size = int(
    VAL_SPLIT * len(full_dataset)
)

train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Intializing Model

model = FrozenResNet18FFT().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

# Learning Rate Scheduler
optimizer = torch.optim.Adam(
    model.adapter.parameters(),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

scaler = GradScaler()

best_val_acc = 0.0
early_stop_counter = 0

# Training

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    model.train()

    running_loss = 0.0

    for batch in tqdm(train_loader):

        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        labels = (
            batch["label"]
            .unsqueeze(1)
            .float()
            .to(DEVICE)
        )

        optimizer.zero_grad()

        with autocast():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    print(f"Train Loss: {train_loss:.4f}")

    # Validation

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            with autocast():

                logits = model(images).squeeze()

                probs = torch.sigmoid(logits)

            preds = (probs > 0.5).long()

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    val_acc = accuracy_score(
        y_true,
        y_pred
    )

    print(f"Validation Accuracy: {val_acc:.4f}")

    scheduler.step(val_acc)

    # Saving Best Model

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print("Best model saved")

        early_stop_counter = 0

    else:

        early_stop_counter += 1

        print(
            f"No improvement "
            f"({early_stop_counter}/{PATIENCE})"
        )

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered")

        break

# Loading Best Model to Evaluate

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
)

model.eval()

# Testing

y_true = []
y_pred = []
y_score = []

with torch.no_grad():

    for batch in tqdm(
        test_loader,
        desc="Testing"
    ):

        images = batch["image"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        with autocast():

            logits = model(images).squeeze()

            probs = torch.sigmoid(logits)

        preds = (probs > 0.5).long()

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

        y_score.extend(
            probs.cpu().numpy()
        )

# Showing Metrics

auc = roc_auc_score(
    y_true,
    y_score
)

acc = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

print("\nTEST RESULTS")

print(f"Accuracy : {acc*100:.2f}%")
print(f"ROC-AUC  : {auc*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

# Showing Confusion Matrix

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

plt.figure(figsize=(6, 6))

disp.plot(
    cmap="Blues",
    values_format='d'
)

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "confusion_matrix.png"
    )
)

plt.show()

# Showing ROC-CURVE

fpr, tpr, _ = roc_curve(
    y_true,
    y_score
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "roc_curve.png"
    )
)

plt.show()