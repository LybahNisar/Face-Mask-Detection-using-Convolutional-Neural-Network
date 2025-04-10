# Face Mask Detection using CNN 🥼🖼️

## Overview

This project implements a **Convolutional Neural Network (CNN)** to detect whether individuals in images are wearing face masks. The model is trained on a dataset sourced from Kaggle, containing images of individuals both wearing and not wearing masks.

## Table of Contents

- [Installation](#installation) 📦
- [Dataset](#dataset) 📊
- [Usage](#usage) ⚙️
- [Model Architecture](#model-architecture) 🏗️
- [Model Training](#model-training) 🏋️‍♀️
- [Results](#results) 📈
- [Sample Predictions](#sample-predictions) ✅❌
- [Contributing](#contributing) 🤝
- [License](#license) 📜

## Installation

To set up the environment, ensure you have **Python 3.x** installed. You can use the following commands to install the necessary dependencies:

```bash
!pip install kaggle numpy tensorflow matplotlib
## Dataset

The dataset used for training is the **Face Mask Dataset** available on Kaggle. It consists of images categorized into two classes:

- **With Mask** 😷
- **Without Mask** 🚫😷

The dataset contains a total of **7553 images**:

- **3725 images** of individuals wearing masks
- **3828 images** of individuals not wearing masks
## Usage

### 1. Downloading and Extracting the Dataset

Use the Kaggle API to download the dataset and extract it:

```bash
kaggle datasets download -d databumb/face-mask-detection
### 2. Preprocessing

Resize images to **128x128 pixels** and convert them into numpy arrays suitable for the model. This ensures that the data is consistent and ready for model input.

```python
# Example for image preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_image(image_path):
    # Resize the image to 128x128 pixels
    image = load_img(image_path, target_size=(128, 128))
    
    # Convert the image to a numpy array
    image = img_to_array(image)
    
    # Add an extra dimension to the image (for batch size)
    image = np.expand_dims(image, axis=0)
    
    # Scale pixel values to [0, 1]
    image = image / 255.0
    
    return image
## Model Architecture

The CNN architecture consists of the following layers:

- **Convolutional layers** with **ReLU** activation
- **MaxPooling layers**
- **Flattening layer** to convert the 2D matrix into a 1D vector
- **Dense layers** with **Dropout** for regularization
- **Output layer** with **Sigmoid** activation for binary classification

The model is compiled with the **Adam optimizer** and **binary cross-entropy loss**:
