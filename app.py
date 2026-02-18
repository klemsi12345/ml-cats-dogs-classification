"""
Simple web app to classify cat/dog images.
Run: streamlit run app.py
"""
from pathlib import Path

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("model.pt")
IMG_SIZE = 224


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint.get("classes", ["cats", "dogs"])
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(DEVICE)
    return model, classes


def predict(model, classes, image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return {c: p.item() for c, p in zip(classes, probs)}


def main():
    st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")
    st.title("🐾 Cat vs Dog Classifier")
    st.caption("Upload an image to classify it as a cat or a dog")

    model, classes = load_model()
    if model is None:
        st.warning("No model found. Run `python train.py` first, then restart this app.")
        st.info(
            "Quick start:\n"
            "1. `python download_data.py --cats 500 --dogs 500`\n"
            "2. `python train.py`\n"
            "3. `python -m streamlit run app.py`"
        )
        return

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, use_container_width=True)
        with col2:
            probs = predict(model, classes, img)
            pred = max(probs, key=probs.get)
            st.metric("Prediction", pred)
            st.progress(probs[pred])
            st.write("Probabilities:")
            for c, p in probs.items():
                st.write(f"  **{c}**: {p:.1%}")


if __name__ == "__main__":
    main()
