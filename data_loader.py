import os
from os import listdir
from os.path import isdir, join

import random
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

import torchvision.transforms as T

from PIL import Image

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD
)

import matplotlib.pyplot as plt
import einops
import kagglehub


# =========================================================
# DOWNLOAD DATASET
# =========================================================

path = kagglehub.dataset_download(
    "birdy654/cifake-real-and-ai-generated-synthetic-images"
)

print("Dataset Path:", path)


# =========================================================
# TRANSFORMS
# =========================================================

def get_train_transform(resolution):

    return T.Compose([

        # Resize image
        T.Resize(
            resolution + resolution // 8,
            interpolation=T.InterpolationMode.BILINEAR
        ),

        # Random crop augmentation
        T.RandomCrop(resolution),

        # Horizontal flip augmentation
        T.RandomHorizontalFlip(p=0.5),

        # Rotation augmentation
        T.RandomRotation(10),

        # Color augmentation
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),

        # Convert image to tensor
        T.ToTensor(),

        # Normalize image
        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
    ])


def get_test_transform(resolution):

    return T.Compose([

        T.Resize(
            resolution + resolution // 8,
            interpolation=T.InterpolationMode.BILINEAR
        ),

        T.CenterCrop(resolution),

        T.ToTensor(),

        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
    ])


# =========================================================
# DATASET CLASS
# =========================================================

class CIFAKEDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        split,
        transform=None
    ):

        assert isdir(dataset_path), \
            f"Invalid dataset path: {dataset_path}"

        assert split in {"train", "test"}, \
            f"Split must be 'train' or 'test'"

        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform

        self.items = self.parse_dataset()

    # -----------------------------------------------------
    # READ DATASET
    # -----------------------------------------------------

    def parse_dataset(self):

        def is_image(filename):

            return filename.lower().endswith(
                ("jpg", "jpeg", "png")
            )

        split_path = join(
            self.dataset_path,
            self.split
        )

        real_dir = join(split_path, "REAL")

        fake_dir = join(split_path, "FAKE")

        items = []

        # ---------- REAL ----------

        if os.path.exists(real_dir):

            items += [

                {
                    "image_path": join(real_dir, img),
                    "is_real": True
                }

                for img in listdir(real_dir)

                if is_image(img)
            ]

        # ---------- FAKE ----------

        if os.path.exists(fake_dir):

            items += [

                {
                    "image_path": join(fake_dir, img),
                    "is_real": False
                }

                for img in listdir(fake_dir)

                if is_image(img)
            ]

        return items

    # -----------------------------------------------------
    # DATASET SIZE
    # -----------------------------------------------------

    def __len__(self):

        return len(self.items)

    # -----------------------------------------------------
    # GET SINGLE SAMPLE
    # -----------------------------------------------------

    def __getitem__(self, idx):

        item = self.items[idx]

        # Load image
        image = Image.open(
            item["image_path"]
        ).convert("RGB")

        # Apply transforms
        if self.transform:

            image = self.transform(image)

        # Create label tensor
        label = torch.tensor(
            1 if item["is_real"] else 0,
            dtype=torch.float32
        )

        return {

            "image": image,

            "label": label,

            "image_path": item["image_path"]
        }

    # -----------------------------------------------------
    # VISUALIZE IMAGE
    # -----------------------------------------------------

    def plot_image(self, image_tensor):

        image = einops.rearrange(
            image_tensor,
            "c h w -> h w c"
        )

        # Unnormalize for visualization
        image = image * torch.tensor(
            IMAGENET_DEFAULT_STD
        ) + torch.tensor(
            IMAGENET_DEFAULT_MEAN
        )

        image = image.clamp(0, 1)

        plt.imshow(image)

        plt.axis("off")

        plt.show()

    # -----------------------------------------------------
    # LABEL DISTRIBUTION
    # -----------------------------------------------------

    def plot_labels_distribution(self):

        counts = {

            "Real": 0,

            "Fake": 0
        }

        for item in self.items:

            if item["is_real"]:

                counts["Real"] += 1

            else:

                counts["Fake"] += 1

        labels = list(counts.keys())

        values = list(counts.values())

        plt.figure(figsize=(7, 5))

        plt.bar(
            labels,
            values,
            color=["blue", "orange"]
        )

        plt.title(
            f"{self.split.upper()} Label Distribution"
        )

        plt.xlabel("Class")

        plt.ylabel("Count")

        for i, v in enumerate(values):

            plt.text(
                i,
                v + max(values) * 0.01,
                str(v),
                ha="center"
            )

        plt.tight_layout()

        plt.show()


# =========================================================
# CREATE DATALOADERS
# =========================================================

def create_dataloaders(

    dataset_path=path,

    resolution=224,

    batch_size=32,

    val_split=0.2,

    num_workers=2
):

    # -----------------------------------------------------
    # TRANSFORMS
    # -----------------------------------------------------

    train_transform = get_train_transform(
        resolution
    )

    test_transform = get_test_transform(
        resolution
    )

    # -----------------------------------------------------
    # DATASETS
    # -----------------------------------------------------

    full_dataset = CIFAKEDataset(

        dataset_path=dataset_path,

        split="train",

        transform=train_transform
    )

    test_dataset = CIFAKEDataset(

        dataset_path=dataset_path,

        split="test",

        transform=test_transform
    )

    # -----------------------------------------------------
    # TRAIN / VALID SPLIT
    # -----------------------------------------------------

    val_size = int(
        val_split * len(full_dataset)
    )

    train_size = len(full_dataset) - val_size

    torch.manual_seed(42)

    train_dataset, val_dataset = random_split(

        full_dataset,

        [train_size, val_size]
    )

    # Validation should NOT use augmentation
    val_dataset.dataset.transform = test_transform

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# =========================================================
# TEST DATALOADER
# =========================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = \
        create_dataloaders()

    print("\nTrain batches:", len(train_loader))

    batch = next(iter(train_loader))

    print("\nBatch keys:")

    for k, v in batch.items():

        if isinstance(v, torch.Tensor):

            print(k, v.shape)

        else:

            print(k, type(v))