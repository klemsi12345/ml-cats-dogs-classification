# Cat vs Dog Classifier

A simple ML app that classifies images as cats or dogs using transfer learning (ResNet18).

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install -r requirements.txt
```

## Quick Start

1. **Get training images** (choose one):

   - **Option A – Download sample data:**
     ```bash
     python download_data.py
     ```
     Downloads 500 cats and 500 dogs from the public Oxford-IIIT Pet dataset into `data/train/`.

   - **Option B – Use your own images:**
     Place images in:
     ```
     data/train/cats/   ← cat images
     data/train/dogs/   ← dog images
     ```

2. **Train the model:**
   ```bash
   python train.py
   ```
   Saves `model.pt` in the project root.

3. **Classify images:**

   - **CLI:**
     ```bash
     python predict.py path/to/image.jpg
     ```

   - **Web app:**
     ```bash
     python -m streamlit run app.py
     ```

## Project Structure

```
ml-cats-dogs-classification/
├── data/
│   └── train/
│       ├── cats/
│       └── dogs/
├── train.py         # Training script
├── predict.py       # CLI prediction
├── app.py           # Streamlit web app
├── download_data.py # Download sample dataset
├── model.pt         # Trained model (after training)
└── requirements.txt
```

## Tips

- More images → better accuracy. Aim for at least 100–200 per class.
- Training uses GPU if available.
- Increase `EPOCHS` in `train.py` for better results.
