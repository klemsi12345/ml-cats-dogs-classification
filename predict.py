"""
Classify an image as cat or dog.
Usage: python predict.py path/to/image.jpg
"""
import sys
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("model.pt")
IMG_SIZE = 224


def load_model():
    if not MODEL_PATH.exists():
        print("Error: No model found. Run 'python train.py' first.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint.get("classes", ["cats", "dogs"])

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(DEVICE)
    return model, classes


def predict(image_path: str) -> str:
    model, classes = load_model()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(1).item()
        confidence = probs[0][pred_idx].item()

    return classes[pred_idx], confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    label, conf = predict(path)
    print(f"Prediction: {label} ({conf:.1%} confidence)")
