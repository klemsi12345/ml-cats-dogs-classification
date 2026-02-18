"""
Train a cat/dog classifier using transfer learning.
Expects images in: data/train/cats/ and data/train/dogs/
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/train")
MODEL_PATH = Path("model.pt")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def create_model(num_classes: int = 2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train():
    if not DATA_DIR.exists():
        print(f"Error: Create {DATA_DIR} with subfolders 'cats' and 'dogs' containing images.")
        print("Example: data/train/cats/*.jpg, data/train/dogs/*.jpg")
        return

    transform = get_transforms()
    dataset = ImageFolder(str(DATA_DIR), transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = create_model(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {DEVICE} with {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={acc:.2f}%")

    model.classes = dataset.classes
    torch.save({"model": model.state_dict(), "classes": dataset.classes}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
