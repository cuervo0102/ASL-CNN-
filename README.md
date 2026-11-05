# ASL-CNN: Real-Time ASL Alphabet Recognition

Real-time American Sign Language (ASL) alphabet recognition using **MediaPipe hand landmarks** and a **1D-CNN** built with TensorFlow/Keras.

Instead of classifying raw images, this project extracts 21 3D hand landmarks per frame and classifies those. The model stays lightweight and fast, and it is much less sensitive to background, lighting, and skin tone than an image-based CNN.


## Pipeline

```
Image / Webcam frame
        │
        ▼
MediaPipe Hands  →  21 landmarks × (x, y, z) = 63 features
        │
        ▼
Normalization (wrist-centered, scaled to [-1, 1])
        │
        ▼
1D-CNN classifier  →  Predicted letter + confidence
```

## Project Structure

```
ASL-CNN-/
├── extract.py      # Landmark extraction from the image dataset
├── train.py        # Data augmentation, model training, evaluation
├── webcame.py      # Real-time webcam inference
└── .gitignore
```

Generated at runtime:

```
processed_data/     # X_landmarks.npy, y_labels.npy, label_map.json
models/             # best_model.keras, final_model.keras, training_history.png
```

## Installation

```bash
git clone https://github.com/cuervo0102/ASL-CNN-.git
cd ASL-CNN-
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install tensorflow mediapipe opencv-python numpy scikit-learn matplotlib tqdm
```

## Dataset

Download the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle and extract it so you have an `asl_alphabet_train/` folder with one subfolder per letter.

## Usage

### 1. Extract landmarks

Edit `DATASET_PATH` at the bottom of `extract.py` to point to your local `asl_alphabet_train` folder, then run:

```bash
python extract.py
```

- Runs MediaPipe Hands on each image (up to 300 images per class).
- Skips the `del`, `nothing`, and `space` classes.
- Saves the landmark arrays and the label map to `processed_data/`.

### 2. Train the model

```bash
python train.py
```

- Stratified split: 15% test, then the remaining 85% split 50/50 into train and validation.
- Landmark-level augmentation (training set only).
- Saves the best checkpoint, the final model, and training curves to `models/`.
- Prints test accuracy and an overfitting check (train/validation gap).

### 3. Run real-time recognition

```bash
python webcame.py
```

Show an ASL letter to your webcam. The window displays the predicted letter, a confidence bar, and FPS. Press **q** to quit.

## Model Details

**Input:** `(21, 3)`, the 21 hand landmarks with 3 coordinates each.

| Block | Layers |
|-------|--------|
| Conv block 1 | Conv1D(64, k=3) → BatchNorm → Dropout(0.3) |
| Conv block 2 | Conv1D(128, k=3) → BatchNorm → Dropout(0.3) |
| Conv block 3 | Conv1D(256, k=3) → BatchNorm → GlobalMaxPooling1D |
| Dense head | Dense(256) → BatchNorm → Dropout(0.5) → Dense(128) → BatchNorm → Dropout(0.4) |
| Output | Dense(num_classes, softmax) |

- Dense layers use L2 regularization (0.001).
- Optimizer: Adam (lr = 1e-3), loss: sparse categorical cross-entropy.
- Callbacks: EarlyStopping (patience 20), ReduceLROnPlateau, ModelCheckpoint.

### Preprocessing

Landmarks are translated so the wrist is the origin, then divided by the maximum absolute value. The features become independent of hand position in the frame and of distance from the camera. The same normalization is applied at training and inference time.

### Data augmentation

Each training sample generates 3 extra variants with:

- random rotation in the image plane (±30°)
- random scaling (0.85 – 1.15)
- Gaussian noise (σ = 0.015)
- random horizontal flip (50% chance)



## Limitations

- Static signs only. Letters that involve motion are not captured by single-frame landmarks.
- One hand at a time (`max_num_hands=1`).
- Performance depends on MediaPipe detecting the hand correctly (good lighting, hand fully in frame).

## Possible Improvements

- Word and sentence building from the recognized letters.
- Sequence model (LSTM / Transformer) over several frames to support dynamic signs.
- Two-hand support and a larger vocabulary of signs.
- Export to TFLite or ONNX for mobile and web deployment.

## Tech Stack

Python · TensorFlow / Keras · MediaPipe · OpenCV · NumPy · scikit-learn · Matplotlib

## Author

**Siham** ([@cuervo0102](https://github.com/cuervo0102))
