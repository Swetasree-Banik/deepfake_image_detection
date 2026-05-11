import os
import torch
import torch.nn as nn
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

import timm
from tqdm import tqdm

from torch.cuda.amp import (
    autocast,
    GradScaler
)

from data_loader_fft_lbp import create_dataloaders_fft

# Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "outputs_two_stage_training"
os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "best_model.pth")

RESOLUTION = 224
BATCH_SIZE = 32
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
VAL_SPLIT = 0.2
PATIENCE = 4

torch.manual_seed(42)

# Model

class AdapterConv(nn.Module):
    def __init__(self, in_channels=5, out_channels=3):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.adapter(x)

class TwoStageResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = AdapterConv(in_channels=5, out_channels=3)
        self.backbone = timm.create_model(
            "resnet18", 
            pretrained=True, 
            num_classes=1
        )

    def forward(self, x):
        x = self.adapter(x)
        return self.backbone(x)

# Loading Dataloader

train_loader, val_loader, test_loader = create_dataloaders_fft(
    resolution=RESOLUTION,
    batch_size=BATCH_SIZE,
    val_split=VAL_SPLIT,
    num_workers=2
)

# Initialize

model = TwoStageResNet18().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler()
best_val_acc = 0.0
early_stop_counter = 0

# Stage 1: Train Adapter Only

print("\nStage 1: Adapter Alignment")
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.adapter.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.adapter.parameters(), lr=STAGE1_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

for epoch in range(STAGE1_EPOCHS):
    print(f"\nStage 1 Epoch [{epoch+1}/{STAGE1_EPOCHS}]")
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader):
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].unsqueeze(1).float().to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(images).squeeze()
                probs = torch.sigmoid(logits)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((probs > 0.5).long().cpu().numpy())

    val_acc = accuracy_score(y_true, y_pred)
    print(f"Val Acc: {val_acc:.4f} | Loss: {running_loss/len(train_loader):.4f}")
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Model Updated")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE: break

# Stage-2: Fine-tune Specific Backbone Layers

print("\nStage 2: Backbone Fine-tuning")
model.load_state_dict(torch.load(CHECKPOINT_PATH))

# Resetting 
early_stop_counter = 0 
best_val_acc = 0.0 

# Unfreeze Logic
for param in model.adapter.parameters():
    param.requires_grad = True # Keep adapter learning
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
for param in model.backbone.get_classifier().parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE2_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

for epoch in range(STAGE2_EPOCHS):
    print(f"\nStage 2 Epoch [{epoch+1}/{STAGE2_EPOCHS}]")
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader):
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].unsqueeze(1).float().to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(images).squeeze()
                probs = torch.sigmoid(logits)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((probs > 0.5).long().cpu().numpy())

    val_acc = accuracy_score(y_true, y_pred)
    print(f"Val Acc: {val_acc:.4f} | Loss: {running_loss/len(train_loader):.4f}")
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Model Updated")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE: break

# Evaluation

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
y_true, y_pred, y_score = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Final Testing"):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        with autocast():
            logits = model(images).squeeze()
            probs = torch.sigmoid(logits)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend((probs > 0.5).long().cpu().numpy())
        y_score.extend(probs.cpu().numpy())

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