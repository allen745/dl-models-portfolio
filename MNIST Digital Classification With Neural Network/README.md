# MNIST Handwritten Digit Classification Using Neural Network

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Machine Learning](https://img.shields.io/badge/Project-Deep%20Learning-red)


## 📌 Project Overview

This project is a **Deep Learning based Handwritten Digit Recognition System** using a Neural Network.

The model is trained on the **MNIST handwritten digit dataset** and learns to classify digits from **0 to 9**.

The complete workflow:

```
Dataset
   ↓
Image Processing
   ↓
Data Normalization
   ↓
Neural Network Training
   ↓
Model Evaluation
   ↓
Custom Image Prediction
```


## 🎯 Objective

The goal of this project is to build a neural network model that can recognize handwritten digits from images.

Example:

Input:

```
Handwritten Image
```

Output:

```
Predicted Digit: 7
```


## 📂 Dataset

Dataset used:

**MNIST Dataset**

The dataset contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Classes: 0-9 digits


Sample:

```
Image Shape: (28,28)

Labels:
0 1 2 3 4 5 6 7 8 9
```


## 🛠️ Technologies Used

### Programming Language

- Python


### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- TensorFlow
- Keras


## 🧠 Neural Network Architecture


Model:

```
Input Image
(28 x 28)

        ↓

Flatten Layer

        ↓

Dense Layer
50 Neurons
ReLU Activation

        ↓

Dense Layer
50 Neurons
ReLU Activation

        ↓

Output Layer
10 Classes
Softmax Activation
```


## ⚙️ Project Workflow


### 1. Import Libraries

Required Python libraries are imported.


### 2. Load Dataset

MNIST dataset is loaded using Keras.


```python
(x_train,y_train),(x_test,y_test)=mnist.load_data()
```


### 3. Data Preprocessing

Pixel values are normalized:

Before:

```
0 - 255
```

After:

```
0 - 1
```


### 4. Model Training

The neural network is trained using:

Optimizer:

```
Adam
```

Loss:

```
Sparse Categorical Crossentropy
```


Training:

```
Epochs = 10
```


### 5. Model Evaluation

The model performance is measured using accuracy.


Example:

```
Test Accuracy: ~97%
```


### 6. Custom Image Prediction

The system can predict your own handwritten digit image.

Process:

```
Upload Image

        ↓

Convert to Grayscale

        ↓

Resize to 28x28

        ↓

Normalize Pixels

        ↓

Neural Network Prediction

        ↓

Output Digit
```


## 📊 Model Performance


| Metric | Result |
|-|-|
| Training Accuracy | ~98% |
| Testing Accuracy | ~97% |
| Classes | 10 Digits |


## 📈 Confusion Matrix

The confusion matrix is used to analyze prediction performance between actual and predicted labels.



## 📁 Project Structure


```
MNIST-Digit-Classification/

│

├── MNIST Digital Classification With Neural Network.py

│

├── data/

│   └── MNIST_digit.png

│

├── README.md

│

└── requirements.txt

```


## 🚀 How To Run Project


### 1. Clone Repository

```
git clone <repository-link>
```


### 2. Install Requirements

```
pip install -r requirements.txt
```


### 3. Run Python File

```
python "MNIST Digital Classification With Neural Network.py"
```



## 📦 Requirements

```
numpy
matplotlib
seaborn
opencv-python
tensorflow
keras
```


## 🔮 Future Improvements

- Convert Neural Network into CNN model
- Deploy using Streamlit
- Create Web Application
- Improve accuracy to 99%+
- Save trained model


## 👨‍💻 Author

**Allen Christian**

Deep Learning Project  
MNIST Handwritten Digit Recognition


## ⭐ Conclusion

This project demonstrates how Deep Learning and Neural Networks can be used for image classification tasks.

The model successfully recognizes handwritten digits using image processing and neural network prediction.
