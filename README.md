# Emotion Detection Project

A deep learning-based facial emotion detection system that recognizes 7 different emotions from facial images and can play corresponding music tracks.

## Features

- **Real-time Emotion Detection**: Detects emotions from facial images using a trained CNN model
- **7 Emotion Classes**: Recognizes Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised
- **Pre-trained Model**: Includes a pre-trained `model.h5` file for immediate use
- **Music Integration**: Plays music tracks corresponding to detected emotions
- **Face Detection**: Uses Haar Cascade classifiers for accurate face detection
- **Model Training**: Includes tools to train custom models on your own datasets

## Project Structure

```
Emotion-detection/
├── emotions.py              # Main emotion detection model and training script
├── dataset_prepare.py       # Dataset preparation script (converts FER2013 CSV to images)
├── musictest.py            # Test script for music playback
├── model.h5                # Pre-trained emotion detection model
├── haarcascade_frontalface_default.xml  # Face detection cascade classifier
├── requirements.txt        # Project dependencies
├── data/
│   ├── train/             # Training dataset (organized by emotion)
│   │   ├── angry/
│   │   ├── disgusted/
│   │   ├── fearful/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprised/
│   └── test/              # Test dataset (same structure as train/)
├── emotions/               # Folder for storing emotion-related files
└── music/                 # Folder containing music tracks for each emotion
    └── Neutral.mp3        # (and similar for other emotions)
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Steps

1. Clone or download the project:
```bash
cd Emotion-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the pre-trained model:
   - The `model.h5` file should be in the project root directory

4. Add music files:
   - Place emotion-specific MP3 files in the `music/` folder
   - File naming convention: `{Emotion}.mp3` (e.g., `Happy.mp3`, `Sad.mp3`)

## Usage

### Display Emotion Detection (Real-time Webcam)

Run the emotion detection with display mode to see real-time emotion detection from your webcam:

```bash
python emotions.py --mode display
```

### Train a Custom Model

Train a new emotion detection model on your dataset:

```bash
python emotions.py --mode train
```

### Test Music Playback

Test the music playback functionality:

```bash
python musictest.py
```

### Prepare Dataset

If you have the FER2013 dataset, convert it to the proper folder structure:

```bash
python dataset_prepare.py
```

**Note**: Requires a `fer2013.csv` file in the project directory.

## Dependencies

- **numpy** (1.17.4): Numerical computing library
- **opencv-python** (4.1.2.30): Computer vision library for image processing
- **tensorflow** (2.1.3): Deep learning framework
- **Pillow** (8.1): Image processing library
- **scikit-image**: Image processing utilities
- **playsound**: Audio playback library
- **pandas**: Data manipulation (for dataset preparation)

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 48×48 pixel grayscale images
- **Output Classes**: 7 emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised)
- **Pre-trained on**: FER2013 (Facial Expression Recognition) dataset

## Emotion Classes

| Emotion | Description |
|---------|-------------|
| Angry | Facial expression indicating anger |
| Disgusted | Expression of disgust |
| Fearful | Expression showing fear |
| Happy | Smiling or joyful expression |
| Neutral | Calm, expressionless face |
| Sad | Expression of sadness |
| Surprised | Expression showing surprise |

## Key Functions

### emotions.py
- `plot_model_history()`: Visualizes model accuracy and loss curves
- Model training pipeline with data augmentation
- Real-time emotion prediction from webcam feed

### dataset_prepare.py
- Converts FER2013 CSV dataset into organized image folders
- Separates data into train/test splits by emotion category

### musictest.py
- Simple audio playback test for emotion-related music

## Results

After training or using the pre-trained model:
- Accuracy and loss plots are saved as `plot.png`
- Real-time predictions display bounding boxes around detected faces
- Emotional predictions are shown with confidence levels

## Troubleshooting

- **Webcam not working**: Ensure your camera is properly connected and no other application is using it
- **Music not playing**: Check that audio files exist in the `music/` folder with correct naming
- **Model not found**: Ensure `model.h5` is in the project root directory
- **Face detection issues**: Ensure `haarcascade_frontalface_default.xml` is present

## Future Enhancements

- Support for multiple face detection in a single image
- Real-time model inference optimization
- GUI interface for easier interaction
- Support for video file input (not just webcam)
- Emotion-based recommendation system

## License

This project uses the FER2013 dataset and OpenCV's Haar Cascade classifiers.

## Author

Emotion Detection Project
