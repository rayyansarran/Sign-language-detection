# Real-Time Sign Language Detection

This project allows you to train a custom sign language detection model using your own hand gestures.

## Prerequisites

- Python 3.x
- Webcam

## Installation

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

## Usage

### 1. Collect Data

Run the data collection script to record your signs. You will need to do this for each sign you want to detect (e.g., 'A', 'B', 'Hello').

```bash
python data_collection.py
```

*   Enter the name of the sign when prompted.
*   Press 'Q' to start recording (it will take a burst of 100 frames).
*   Move your hand slightly while recording to capture variations.
*   Repeat for as many signs as you want.

### 2. Train Model

After collecting data, train the classifier.

```bash
python train_model.py
```

This will create a `model.p` file.

### 3. Run Detection

Start the real-time detector.

```bash
python inference.py
```

*   Press 'Q' to quit.

## Troubleshooting

-   **Accuracy**: If accuracy is low, try collecting more data with different hand angles and distances found in `data_collection.py`.
-   **Lighting**: Ensure good lighting for MediaPipe to detect hands correctly.
