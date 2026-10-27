# Sign Language Recognition: MediaPipe Holistic + MLP

This is a landmark-only baseline for six classes: `hello`, `i_love_you`, `ok`, `peace`, `thank_you`, and `yes`. It intentionally does not use MobileNetV3 or raw pixels as model input.

Each 141-value feature vector contains:

- normalised 3D shape coordinates for 21 landmarks from each hand (wrist-relative and scale-normalised);
- left/right hand centres and presence flags;
- face centre and a presence flag; and
- each hand's face-relative `dx`, `dy`, and Euclidean distance.

Missing hands or a missing face are represented by zeros with their corresponding presence flag set to `0`. Spatial values are also zero when either member of the pair is unavailable.

## Setup

Use Python 3.12 and create an isolated environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the baseline

Raw images must be organised as `dataset/raw/<class_name>/image.jpg`, as the supplied dataset already is. Run the stages in order:

```powershell
python src/extract_features.py
python src/preprocess.py --seed 42
python src/train.py --epochs 100 --learning-rate 0.001
python src/evaluate.py
python src/webcam.py --threshold 0.70 --show-landmarks
```

Extraction writes `dataset/processed/features.csv`. Preprocessing creates `train.csv`, `validation.csv`, and `test.csv` in the same directory. The fixed seed and a group-aware split keep a `source_group` entirely in one split. For nested source folders, that folder is the group; for the current flat folders, adjacent numeric image IDs are conservatively grouped in blocks of 15. This prevents nearby frames from crossing splits, although true recording/session IDs would be preferable if they become available.

`train.py` fits `StandardScaler` on training data only, then writes `models/scaler.pkl`, `models/classes.json`, and the validation-best `models/best_model.keras`. `evaluate.py` uses only the untouched test split and writes the classification report, metrics, and confusion matrix under `results/`.

The webcam path calls the same `extract_features_from_rgb` function as training, ensuring feature calculation remains identical. It will not classify a frame when neither hand is detected; those frames display `Unknown / Uncertain` instead. Press `q` to exit it.

## Project layout

```text
dataset/raw/                 labelled source images
dataset/processed/           generated feature and split CSVs
models/                      generated model, scaler, class labels
results/                     generated learning and evaluation plots
src/landmarks.py             shared Holistic feature implementation
src/extract_features.py      image-to-feature extraction
src/preprocess.py            group-aware 70/15/15 split
src/train.py                 MLP training
src/evaluate.py              untouched-test evaluation
src/webcam.py                real-time inference
```
