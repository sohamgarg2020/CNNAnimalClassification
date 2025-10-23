import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import zipfile
import subprocess
import shutil
from PIL import Image








if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device}!")

transform  = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")), #makes it so that every image has 3 channels - RGB
    transforms.Resize((224, 224)), #has a standard pixel size 128x128
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness =0.1, contrast = 0.1),
    transforms.ToTensor(),
    transforms.Normalize( #normalizing it so that the values are between -1 to 1
        mean = (0.5, 0.5, 0.5),
        std = (0.5, 0.5, 0.5)
    )
])

data_zip = "animals.zip"
data_dir = "./data_animals"

if not os.path.exists(data_dir):
    print("Dataset not found. Downloading from Kaggle.")

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "antobenedetti/animals",
        "-p", ".", "--force"
    ])

    print("Unzipping dataset...")
    os.makedirs(data_dir, exist_ok=True)

    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print("Dataset downloaded and extracted successfully at:", data_dir)
else:
    print("Dataset already exists, skipping download.")

data_dir = "./data_animals/animals/"

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")

train_dataset = datasets.ImageFolder(root = train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root = test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = train_dataset.classes
num_classes = len(classes)
print(f"We have {num_classes} classes.")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        
        
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(512 * 7 * 7, 256)
        self.relu6 = nn.ReLU()

        self.dropout7 = nn.Dropout(p = 0.25)
        self.fc7 = nn.Linear(256, 128)
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(128, num_classes)
        
    def forward(self, x):

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))


        x = x.reshape(x.size(0), -1)
        
        x = self.dropout6(x)
        x = self.relu6(self.fc6(x))
        x = self.dropout7(x)
        x = self.relu7(self.fc7(x))
        x = self.fc8(x)
        return x
    
    
model = SimpleCNN(num_classes=num_classes).to(device)
loss_func = nn.CrossEntropyLoss()
lr = 0.008
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=0.001)



MODEL_PATH = "animal_cnn.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model has already been trained.")
else:
    print("Training model from scratch now. Get excited!")


    epochs = 50
    for epoch in range(epochs):
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            all_val_loss = []
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                total += labels.size(0)

                predicted = torch.argmax(outputs, dim = 1)
                correct += (predicted == labels).sum().item()
                all_val_loss.append(loss_func(outputs, labels).item())

            mean_val_loss = sum(all_val_loss)/ len(all_val_loss)
            mean_val_acc = 100 * (correct/total)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val-Loss: {mean_val_loss:.4f}, Val-acc: {mean_val_acc:.4f}%")


    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved to animal_cnn.pth")


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

PREDICT_IMAGES = "test_images"

def predict_image(img_path):
    model.eval()
    image = Image.open(os.path.join(PREDICT_IMAGES, img_path))
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim =  1)
        pred_idx = torch.argmax(probabilities, dim = 1).item()

    confidence = probabilities[0][pred_idx].item()*100
    print(f"Predicted class for {img_path}: {classes[pred_idx]} ({confidence:.2f}% confidence)")


for filename in os.listdir(PREDICT_IMAGES):
    predict_image(filename)