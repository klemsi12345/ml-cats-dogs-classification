# Cat vs Dog Classifier

A simple ML app that classifies images as cats or dogs using transfer learning (ResNet18).
Training data is sourced from [Google Open Images](https://storage.googleapis.com/openimages/web/index.html).

## Setup

```powershell
& "C:\...\python3.12\python.exe" -m venv venv   # use Python 3.12 for CUDA support
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For CUDA/GPU support, install PyTorch with CUDA wheels:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Quick Start

1. **Download training images from Open Images:**

   ```powershell
   python download_data.py --cats 20000 --dogs 20000
   ```

   Downloads up to 20,000 cat and 20,000 dog images from Google Open Images into `data/train/`.
   Adjust counts as needed (e.g. `--cats 500 --dogs 500` for a quick test).

   Or **use your own images** by placing them in:

   ```
   data/train/cats/   <- cat images
   data/train/dogs/   <- dog images
   ```

2. **Train the model:**

   ```powershell
   python train.py
   ```

   Saves `model.pt` in the project root.

3. **Classify images:**

   - **CLI:**

     ```powershell
     python predict.py path/to/image.jpg
     ```

   - **Web app:**

     ```powershell
     python -m streamlit run app.py
     ```

## Project Structure

```
ml-cats-dogs-classification/
├── data/
│   ├── train/
│   │   ├── cats/              # Training cat images
│   │   └── dogs/              # Training dog images
│   └── source/openimages/     # Raw Open Images download cache
├── train.py                   # Training script (ResNet18 transfer learning)
├── predict.py                 # CLI prediction
├── app.py                     # Streamlit web app
├── download_data.py           # Download images from Open Images
├── model.pt                   # Trained model (generated after training)
├── requirements.txt
└── .gitignore
```

## Tips

- More images lead to better accuracy. Start with 500 per class for quick tests, scale to 20,000+ for production.
- Training uses GPU automatically if CUDA is available (requires Python 3.12 + CUDA PyTorch build).
- Increase `EPOCHS` in `train.py` for better results.
- The download script is resumable: rerun it to top up existing images toward the target count.
