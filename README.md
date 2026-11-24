# NeuroScan — Brain Tumor MRI Classifier

Flask web app that classifies brain MRI scans with a VGG16 model using a pure NumPy inference engine (no TensorFlow/GPU required).

## Classes (verified label order)

| Index | Label |
|------:|-------|
| 0 | Other / Non-MRI |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Glioma |
| 4 | Pituitary Tumor |

These were verified against the labeled Testing set. Do not reorder them unless you retrain.

## Requirements (macOS / Apple Silicon)

- Python 3.12+ (tested on 3.14 arm64)
- Packages: `flask`, `numpy`, `h5py`, `pillow`

The included `venv/` folder is from Windows and will **not** work on Mac. Use `.venv/` instead.

## Setup (MacBook M-series)

```bash
cd "/Users/ahmad/Desktop/ML Project"

# Create a native macOS virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
cd "/Users/ahmad/Desktop/ML Project"
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:5001** in your browser, then upload an MRI image (JPEG/PNG/TIFF).

> Note: Port 5000 is often taken on macOS by AirPlay Receiver, so this app uses **5001**.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Flask server (`/` UI, `/predict` API) |
| `numpy_inference.py` | NumPy VGG16 forward pass |
| `BrainTumor_VGG16.h5` | Model weights |
| `templates/`, `static/` | NeuroScan UI |
| `uploads/` | Temporary upload storage |
| `requirements.txt` | Python dependencies |
| `.venv/` | macOS virtualenv (use this) |
| `venv/` | Windows virtualenv (ignore on Mac) |

## Notes

- Inference is CPU-only via vectorized NumPy; first prediction may take a few seconds.
- Input images are resized to **128×128** and normalized to `[0, 1]`.
