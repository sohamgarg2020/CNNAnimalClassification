import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import zipfile
import subprocess







device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")


transform  = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

data_zip = "animal-image-dataset-90-different-animals.zip"
data_dir = "./animals"

if not os.path.exists(data_dir):
    print("Dataset not in directory. Downloading from Kaggle.")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "iamsouravbanerjee/animal-image-dataset-90-different-animals",
        "-p", ".", "--force"
    ])
    print("Unzipping dataset.")
    os.makedirs(data_dir, exist_ok=True)

    with zipfile.ZipFile("animal-image-dataset-90-different-animals.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir)
    print("Dataset downloaded and extracted.")
else:
    print("Dataset already exists, skipping download.")

data_dir = "./animals/animals/animals"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

train_set = Subset(dataset, train_idx)
test_set = Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

classes = dataset.classes
print(f"We have {len(classes)} classes.")