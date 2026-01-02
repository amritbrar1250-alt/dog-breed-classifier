# Dog Breed Classifier

A deployed machine learning web application that predicts dog breeds from images using deep learning.  
Built with TensorFlow and Streamlit, leveraging transfer learning on EfficientNet.

## Live Demo
https://amritbrar1250-alt-dog-breed-classifier.streamlit.app

## Overview
This project classifies dog breeds from images across **120 different breeds**.  
Users can upload their own photos or select example images to receive Top-K predictions along with confidence scores.

The model is trained using transfer learning and optimized for real-time inference in a web interface.

## Features
- Image upload and example gallery
- Top-K breed predictions with confidence scores
- Low-confidence detection with user guidance
- Correct handling of phone image orientation (EXIF metadata)
- Downloadable prediction results (JSON)
- Clean, responsive Streamlit UI

## Model & Dataset
- **Architecture:** EfficientNetB0 (transfer learning)
- **Framework:** TensorFlow / Keras
- **Dataset:** Stanford Dogs Dataset
  - 120 dog breeds
- **Test Accuracy:** **84.56%**

## How It Works
1. User uploads an image or selects an example
2. Image is preprocessed and normalized
3. Model performs inference
4. Top-K breed predictions are displayed with probabilities
5. Low-confidence results trigger user guidance

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Author
_Amrit Singh_
