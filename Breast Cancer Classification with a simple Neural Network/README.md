# 🧠 Breast Cancer Classification using Deep Learning

> A Deep Learning-based Breast Cancer Classification system built using TensorFlow and Keras Neural Networks on the Wisconsin Breast Cancer Dataset.

---

## 👨‍💻 Author

**Allen Stivanson Christian**

AI & ML Engineer | Full Stack Developer

🔗 GitHub: https://github.com/allen745

---

## 📌 Project Overview

This project uses an Artificial Neural Network (ANN) to classify breast tumors as:

- Malignant (Cancerous)
- Benign (Non-Cancerous)

The model is trained on the Wisconsin Breast Cancer Dataset available through Scikit-Learn.

The project demonstrates:

- Data preprocessing
- Feature scaling
- Deep Learning model building
- Model training and validation
- Performance evaluation
- Real-time prediction

---

## 🎯 Problem Statement

Breast cancer is one of the most common cancers worldwide.

Early diagnosis can significantly increase treatment success rates.

This project uses Deep Learning techniques to predict whether a tumor is:

✅ Benign

❌ Malignant

based on medical diagnostic measurements.

---

## 📂 Dataset Information

Dataset Source:

```python
from sklearn.datasets import load_breast_cancer
```

Dataset Statistics:

| Feature | Value |
|----------|--------|
| Samples | 569 |
| Features | 30 |
| Classes | 2 |
| Malignant | 212 |
| Benign | 357 |

---

## 🧰 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow
- Keras

---

## 🏗️ Deep Learning Architecture

```text
Input Layer (30 Features)
          ↓
Dense Layer (20 Neurons, ReLU)
          ↓
Output Layer (2 Neurons, Sigmoid)
```

---

## 🔄 Project Workflow

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Scaling
       ↓
Train-Test Split
       ↓
Build ANN Model
       ↓
Train Neural Network
       ↓
Validation
       ↓
Performance Evaluation
       ↓
Prediction System
```

---

## 📊 Model Training

### Optimizer

```python
Adam
```

### Loss Function

```python
Sparse Categorical Crossentropy
```

### Epochs

```python
10
```

### Validation Split

```python
10%
```

---

## 📈 Training Accuracy Graph

Add image:

```markdown
![Training Accuracy](accuracy_graph.png)
```

---

## 📉 Training Loss Graph

Add image:

```markdown
![Training Loss](loss_graph.png)
```

---

## 🚀 Model Performance

| Metric | Score |
|----------|--------|
| Test Accuracy | ~95% to 98% |
| Classes | Malignant / Benign |

---

## 🔍 Prediction Example

Input:

```python
input_data = (
13.54,14.36,87.46,566.3,
0.09779,0.08129,0.06664,
0.04781,0.1885,0.05766,
0.2699,0.7886,2.058,
23.56,0.008462,0.0146,
0.02387,0.01315,0.0198,
0.0023,15.11,19.26,
99.7,711.2,0.144,
0.1773,0.239,0.1288,
0.2977,0.07259
)
```

Output:

```text
The tumor is Benign
```

---

## 📁 Project Structure

```text
Breast-Cancer-Classification-DL/
│
├── Breast Cancer Classification DL.py
├── README.md
├── requirements.txt
│
├── accuracy_graph.png
├── loss_graph.png
│
└── Dataset (Scikit-Learn Built-in)
```

---

## ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/allen745/Breast-Cancer-Classification-DL.git
```

Move into folder

```bash
cd Breast-Cancer-Classification-DL
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run project

```bash
python "Breast Cancer Classification DL.py"
```

---

## 📚 Deep Learning Concepts Covered

- Artificial Neural Networks (ANN)
- Dense Layers
- Activation Functions
- ReLU
- Sigmoid
- Adam Optimizer
- Feature Scaling
- Forward Propagation
- Classification
- TensorFlow
- Keras

---

## 🎓 Learning Outcomes

Through this project I learned:

- Building Neural Networks using Keras
- Data Standardization
- Binary Classification using ANN
- Model Evaluation
- TensorFlow Workflow
- Medical Dataset Analysis

---

## ⭐ Future Improvements

- Add Dropout Layers
- Hyperparameter Tuning
- Early Stopping
- TensorBoard Integration
- Streamlit Deployment
- Confusion Matrix Visualization

---

## 🤝 Connect With Me

GitHub:

https://github.com/allen745

---

## ⭐ Support

If you found this project useful, please consider giving it a star ⭐

---

<p align="center">
<b>Built with ❤️ by Allen Stivanson Christian</b>
</p>
