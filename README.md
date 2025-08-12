
# Hand Sign Language Classification

This project captures hand sign language gestures via webcam, processes the images, and classifies the sign using a trained deep learning model.

## Project Overview

- Uses OpenCV to capture live video from the webcam.
- Uses `cvzone` library for hand detection and classification.
- Preprocesses the detected hand images to a fixed size with a custom background.
- Classifies hand signs from A-Z and a "DELETE" sign using a Keras model.
- Displays live predictions on the video feed.

## Dataset

The dataset used for training and testing the model was created with the provided `datasetProcess.py` script. You can make your own dataset using `datasetProcess.py. You can find the dataset I made here:

[Dataset Link](https://drive.google.com/drive/folders/1R8HeH1vjDePMzr5K6PHw0i-voMm3efAg?usp=drive_link)

## Requirements

- Python 3.x
- OpenCV
- Numpy
- cvzone
- TensorFlow / Keras (for the classification model)
- mediapipe

Install dependencies with:

```
pip install opencv-python numpy cvzone tensorflow mediapipe
```

## Usage

1. The webcam window will open. Perform hand signs in front of the camera.
2. The recognized sign will be displayed on the video.
3. Press `Esc` key to exit.

## How it works

- The hand detector locates the hand bounding box.
- The hand image is cropped with an offset and resized to 400x400 pixels on a purple background.
- The processed image is passed to the classifier.
- The predicted class label is shown on screen in real-time.
