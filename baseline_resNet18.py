import os
import copy
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

from tqdm import tqdm

import timm

from data_loader import create_dataloaders

# Config

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAVE_DIR = "test_outputs_resnet18_improved"

os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "best_checkpoint.pth"
)

RESOLUTION = 224

BATCH_SIZE = 32

EPOCHS = 20

LR = 1e-4

VAL_SPLIT = 0.2

PATIENCE = 5

# Mixed Precision Training

scaler = torch.cuda.amp.GradScaler()

# Loading Dataset

train_loader, val_loader, test_loader = create_dataloaders(

        resolution=RESOLUTION,

        batch_size=BATCH_SIZE,

        val_split=VAL_SPLIT
    )

# Model

model = timm.create_model(

    "resnet18",

    pretrained=True,

    num_classes=1

).to(DEVICE)

# Loss Function

criterion = nn.BCEWithLogitsLoss()

# Optimizer

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LR
)

# Learning Rate Scheduler

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)

# Training

best_val_acc = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)

early_stop_counter = 0


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()

    running_loss = 0.0

    train_bar = tqdm(

        train_loader,

        desc="Training"
    )

    for batch in train_bar:

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

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        train_bar.set_postfix(
            loss=loss.item()
        )

    avg_loss = running_loss / len(train_loader)

    print(
        f"Training Loss: "
        f"{avg_loss:.4f}"
    )

    # Validation

    model.eval()

    y_true = []

    y_pred = []

    with torch.no_grad():

        for batch in tqdm(

            val_loader,

            desc="Validation"
        ):

            images = batch["image"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            with torch.cuda.amp.autocast():

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

    print(
        f"Validation Accuracy: "
        f"{val_acc:.4f}"
    )

    scheduler.step(val_acc)

    print(
        f"Current LR: "
        f"{optimizer.param_groups[0]['lr']}"
    )

    # Saving Best Model

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(

            best_model_weights,

            CHECKPOINT_PATH
        )

        print(
            f"Best model saved "
            f"(Val Acc = {val_acc:.4f})"
        )

        early_stop_counter = 0

    else:

        early_stop_counter += 1

        print(
            f"No improvement "
            f"({early_stop_counter}/{PATIENCE})"
        )

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered!")

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

        with torch.cuda.amp.autocast():

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

print(f"Accuracy  : {acc * 100:.2f}%")

print(f"ROC-AUC   : {auc * 100:.2f}%")

print(f"Precision : {precision * 100:.2f}%")

print(f"Recall    : {recall * 100:.2f}%")

print(f"F1 Score  : {f1 * 100:.2f}%")

# Showing Confusion Matrix

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Fake", "Real"]
)

plt.figure(figsize=(6, 6))

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix - Test")

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "confusion_matrix_test.png"
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
        "roc_curve_test.png"
    )
)

plt.show()