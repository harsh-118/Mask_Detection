# Mask Detection System using CNN and OpenCV
This project aims to detect whether a person is wearing a mask or not using Convolutional Neural Networks (CNNs) and OpenCV. The system utilizes a trained CNN model to predict mask presence on live video streams captured from a webcam.

## Introduction
The COVID-19 pandemic has emphasized the importance of wearing masks to prevent the spread of the virus. This project provides a solution for automated mask detection, which can be deployed in various environments such as airports, hospitals, and public places to ensure compliance with mask-wearing regulations.

## Features
* Real-time mask detection using webcam feed.
* Efficient face detection with OpenCV's Haar Cascade Classifier.
* CNN model trained on a dataset of masked and unmasked faces.
* User-friendly interface for easy deployment and usage.

# Dependencies
To run this project, you need the following dependencies:

**1.Python 3.x:** The code is written in Python 3.\
**2.TensorFlow:** TensorFlow is an open-source machine learning framework. You can install it via pip:
```
pip install tensorflow
```

**3.OpenCV:** OpenCV is a library used for computer vision tasks. It's used here for video capture and face detection. You can install it via pip:
```
pip install opencv-python
```

**4.Pillow:** Pillow is a Python Imaging Library that adds support for opening, manipulating, and saving many different image file formats. Install it via pip:
```
pip install Pillow
```


## Installation
To run the mask detection system locally, follow these steps:

**1.Clone the Repository:**
```
git clone https://github.com/yourusername/mask-detection.git
```
**2.Navigate to Project Directory:**
```
cd mask-detection
```
**3.Install Dependencies:**
```
pip install -r requirements.txt
```
### Usage
**1.Run the Mask Detection System:**
```
python mask_detection.py
```
**2.Allow Webcam Access:** Grant permission for the program to access your webcam.\
**3.Real-time Detection:** The system will display a live video stream with real-time mask detection results.

# Objective: Develop a system to detect whether a person is wearing a mask in real-time using a webcam feed.
## Approach:
### 1.Data Collection: 
Collect images of faces with and without masks to train and validate the CNN.
### 2.Data Preprocessing:
Use image data generators (ImageDataGenerator) to prepare and augment training and testing datasets.
### 3.Model Building:
Build a CNN model (cnn) that takes input images (150x150 pixels) and predicts mask presence (binary classification).
### 4.Model Training:
Train the CNN model using the prepared datasets (train_dataset and test_dataset).
### 5.Real-time Mask Detection:
Utilize the trained model (Harsh.h5) along with OpenCV (cv2) to perform real-time face detection and mask prediction from webcam frames.


# Detailed Explanation of Each Part:
### 1.Data Preparation (ImageDataGenerator):
train_datagen and test_datagen: Image data generators with augmentation settings (zoom_range, rotation_range, shear_range, rescale) for training and testing images.
train_dataset and test_dataset: Flow generators that fetch images from specified directories (train and test) and prepare batches for model training.
### 2.CNN Model Building (Sequential):
Construct a sequential CNN model (cnn) using tf.keras.models.Sequential().
Add convolutional (Conv2D) and pooling (MaxPool2D) layers to extract features from input images.
Flatten (Flatten) the feature maps and add dense (Dense) layers for classification (units=1 for binary classification).
Use appropriate activation functions (relu for hidden layers and sigmoid for binary classification).
### 3.Model Compilation (compile) and Training (fit):
Compile the CNN model (cnn) with optimizer (adam), loss function (binary_crossentropy), and evaluation metric (accuracy).
Train the compiled model (cnn) using training dataset (train_dataset) and validate on the testing dataset (test_dataset) for a specified number of epochs (epochs=50).
### 4.Real-time Mask Detection using Webcam:
Load the trained CNN model (Harsh.h5) and initialize a webcam capture (cv2.VideoCapture(0)).
Continuously read frames (cap.read()) from the webcam, detect faces using haarcasccade.detectMultiScale, and predict mask presence using the loaded model (model.predict(face)).
Draw rectangles and labels on detected faces based on the prediction (ans > 0.5).
### 5.End of Execution:
Exit the webcam loop (while cap.isOpened()) when the user presses 'q' (if cv2.waitKey(1) == 113:).
Release the video capture (cap.release()) and close OpenCV windows (cv2.destroyAllWindows()).


# With Mask
![With_mask](https://github.com/harsh-118/Mask_Detection/assets/166591573/1d49ba23-34f9-4c2c-abca-2ca6a4be46ae)


# Without Mask
![Without_mask](https://github.com/harsh-118/Mask_Detection/assets/166591573/fdc17f48-16e0-4489-8f33-453c8b4e3c04)

