import os
import cv2
import torch
import random
import numpy as np

from PIL import Image

from skimage import feature

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

import torchvision.transforms as T

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD
)

import kagglehub

# FFT Feature Extraction + Normalization

def compute_fft(image, resolution):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    fft = np.fft.fft2(gray)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    magnitude = np.log1p(magnitude)

    magnitude = cv2.resize(
        magnitude,
        (resolution, resolution)
    )

    magnitude = (

        magnitude - magnitude.mean()

    ) / (

        magnitude.std() + 1e-8
    )

    return magnitude

# LBP Feature Extraction + Normalization

def compute_lbp(image, resolution):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    lbp = feature.local_binary_pattern(

        gray,

        P=8,

        R=1,

        method="uniform"
    )

    lbp = cv2.resize(
        lbp,
        (resolution, resolution)
    )

    lbp = (

        lbp - lbp.mean()

    ) / (

        lbp.std() + 1e-8
    )

    return lbp

# Dataset

class CIFAKEDatasetFFT(Dataset):

    def __init__(

        self,

        dataset_path,

        split,

        resolution=224
    ):

        self.items = []

        self.resolution = resolution

        # Training Transforms

        if split == "train":

            self.transform = T.Compose([

                T.Resize(
                    (resolution, resolution)
                ),

                # Data augmentation
                T.RandomHorizontalFlip(p=0.5),

                T.RandomRotation(10),

                T.ColorJitter(

                    brightness=0.2,

                    contrast=0.2,

                    saturation=0.2,

                    hue=0.1
                ),

                T.RandomResizedCrop(

                    resolution,

                    scale=(0.8, 1.0)
                ),

                T.ToTensor(),

                # RGB normalization
                T.Normalize(

                    mean=IMAGENET_DEFAULT_MEAN,

                    std=IMAGENET_DEFAULT_STD
                )
            ])

        # Valid and Test Transforms

        else:

            self.transform = T.Compose([

                T.Resize(
                    (resolution, resolution)
                ),

                T.ToTensor(),

                T.Normalize(

                    mean=IMAGENET_DEFAULT_MEAN,

                    std=IMAGENET_DEFAULT_STD
                )
            ])

        # Loading Dataset

        root = os.path.join(
            dataset_path,
            split
        )

        label_mapping = {

            "REAL": 1,

            "FAKE": 0
        }

        for label_dir_name in os.listdir(root):

            if label_dir_name in label_mapping:

                label_dir_path = os.path.join(

                    root,

                    label_dir_name
                )

                if os.path.isdir(label_dir_path):

                    label_value = label_mapping[
                        label_dir_name
                    ]

                    for fname in os.listdir(
                        label_dir_path
                    ):

                        if fname.lower().endswith(

                            (
                                ".jpg",
                                ".jpeg",
                                ".png"
                            )
                        ):

                            self.items.append({

                                "path": os.path.join(

                                    label_dir_path,

                                    fname
                                ),

                                "label": label_value
                            })

    # Loading RGB and Computing FFT + LBP

    def __getitem__(self, idx):

        item = self.items[idx]

        img = Image.open(
            item["path"]
        ).convert("RGB")

        img_np = np.array(img)

        fft = compute_fft(

            img_np,

            self.resolution
        )

        lbp = compute_lbp(

            img_np,

            self.resolution
        )

        rgb_tensor = self.transform(img)

        fft_tensor = torch.tensor(
            fft,
            dtype=torch.float32
        ).unsqueeze(0)

        lbp_tensor = torch.tensor(
            lbp,
            dtype=torch.float32
        ).unsqueeze(0)

        # Concanating RGB(3) + FFT + LBP

        image_5ch = torch.cat(

            [
                rgb_tensor,
                fft_tensor,
                lbp_tensor
            ],

            dim=0
        )

        label = torch.tensor(

            item["label"],

            dtype=torch.float32
        )

        return {

            "image": image_5ch,

            "label": label
        }

    def __len__(self):

        return len(self.items)

# Creating Dataloaders

def create_dataloaders_fft(

    resolution=224,

    batch_size=32,

    val_split=0.2,

    num_workers=2
):


    dataset_path = kagglehub.dataset_download(

        "birdy654/cifake-real-and-ai-generated-synthetic-images"
    )

    # Full Train Dataset

    full_dataset = CIFAKEDatasetFFT(

        dataset_path=dataset_path,

        split="train",

        resolution=resolution
    )

    # Test Dataset

    test_dataset = CIFAKEDatasetFFT(

        dataset_path=dataset_path,

        split="test",

        resolution=resolution
    )

    # Spliting Train-Test

    val_size = int(
        val_split * len(full_dataset)
    )

    train_size = len(full_dataset) - val_size

    torch.manual_seed(42)

    train_dataset, val_dataset = random_split(

        full_dataset,

        [train_size, val_size]
    )

    # REMOVing Augmentation From Validation

    val_dataset.dataset.transform = T.Compose([

        T.Resize(
            (resolution, resolution)
        ),

        T.ToTensor(),

        T.Normalize(

            mean=IMAGENET_DEFAULT_MEAN,

            std=IMAGENET_DEFAULT_STD
        )
    ])

    # Dataloader

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

# Test Dataloader

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders_fft()

    sample_batch = next(iter(train_loader))

    print(

        "Image shape:",

        sample_batch["image"].shape
    )

    print(

        "Label shape:",

        sample_batch["label"].shape
    )