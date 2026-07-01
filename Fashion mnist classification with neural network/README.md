# Fashion MNIST Classification with a Simple Neural Network

A simple Dense (fully-connected) Neural Network trained to classify clothing images from the **Fashion-MNIST** dataset. This project follows on from `Breast Cancer Classification` and `MNIST Digit Classification`, reusing the same simple-NN architecture on a visually harder dataset.

## 📌 Overview

- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale images (28x28), 10 clothing categories
- **Model:** Simple Sequential Neural Network (Dense layers only, no convolutions)
- **Task:** Multi-class image classification
- **Framework:** TensorFlow / Keras

## 🗂️ Classes

| Label | Class |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 🏗️ Model Architecture

```
Input (784,)  →  flattened 28x28 image
Dense (128, relu)
Dense (64, relu)
Dense (10, softmax)  →  10 class probabilities
```

**Total params:** 109,386 (427.29 KB) — all trainable

| Layer | Output Shape | Param # |
|-------|-------------|---------|
| Dense | (None, 128) | 100,480 |
| Dense | (None, 64) | 8,256 |
| Dense | (None, 10) | 650 |

## ⚙️ Training Config

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 15
- **Batch size:** 32
- **Validation split:** 10%

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **88.23%** |
| **Test Loss** | 0.3658 |
| Final Train Accuracy | 92.19% |
| Final Val Accuracy | 88.92% |

Training accuracy climbed steadily from **82.1% → 92.2%** over 15 epochs, while validation accuracy plateaued around **~88–89%**, with the gap between train and val widening in later epochs — a sign of mild overfitting typical of a plain Dense network with no regularization or convolutional structure.

### Most Frequently Misclassified Classes

| Class | Mistakes |
|-------|----------|
| Pullover | 299 |
| Shirt | 225 |
| Coat | 182 |
| T-shirt/top | 177 |
| Dress | 101 |

Unsurprising — **Pullover, Shirt, and Coat** are visually similar upper-body garments, and a flattened Dense network has no way to learn spatial/edge features that would help tell them apart. This is the expected limitation of this architecture on this dataset.

## 🖼️ Outputs

Running the script generates:

- `sample_images.png` — grid of sample training images with labels
- `training_history.png` — accuracy & loss curves over training
- `predictions.png` — sample correct (green) vs incorrect (red) predictions

## ▶️ How to Run

```bash
python "Fashion mnist classification with neural network.py"
```

Requires:
```bash
pip install tensorflow matplotlib numpy
```

The dataset downloads automatically via `keras.datasets.fashion_mnist` on first run.

## 🔑 Key Takeaway

A simple Dense NN gets solid but imperfect results (~88%) on Fashion-MNIST — noticeably lower than the ~97–98% typically achievable on digit-based MNIST with the same architecture. This gap is the motivation for the next step in this learning path: a **Convolutional Neural Network (CNN)**, which preserves spatial structure in images and should meaningfully improve accuracy — especially on the confused Pullover/Shirt/Coat cluster.

## 📁 Related Projects
- `Breast Cancer Classification with a simple Neural Network.py`
- `MNIST Digital Classification With Neural Network.py`
- `Fashion mnist classification with neural network.py` *(this project)*

## 👨‍💻 Author

**Allen Christian**

Deep Learning Project  
Fashion mnist classification with neural network
