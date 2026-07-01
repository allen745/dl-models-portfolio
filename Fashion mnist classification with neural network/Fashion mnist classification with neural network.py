"""
Fashion MNIST Classification with a Simple Neural Network
------------------------------------------------------------
Dataset: Fashion-MNIST (70,000 grayscale images, 28x28, 10 clothing categories)
Goal   : Classify clothing images using a simple Dense Neural Network.

This is a follow-up to the MNIST Digit Classification project. Same image
shape (28x28 grayscale), same simple NN architecture -- but Fashion-MNIST
images are visually harder to separate than digits, so accuracy will be
noticeably lower. That gap is intentional: it sets up the case for CNNs
in a future project.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------------------------
# 1. Load the Data
# -------------------------------------------------
# Fashion-MNIST ships with Keras, just like MNIST did.
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Class names (Fashion-MNIST labels are 0-9, mapped to these categories)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape    : {x_test.shape}")

# -------------------------------------------------
# 2. Explore a Few Samples
# -------------------------------------------------
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.suptitle("Sample Training Images")
plt.tight_layout()
plt.savefig("sample_images.png")
plt.close()

# -------------------------------------------------
# 3. Preprocess the Data
# -------------------------------------------------
# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-length vectors for the Dense NN
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(f"Flattened training shape: {x_train_flat.shape}")

# -------------------------------------------------
# 4. Build the Simple Neural Network
# -------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 output classes
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------
# 5. Train the Model
# -------------------------------------------------
history = model.fit(
    x_train_flat, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -------------------------------------------------
# 6. Evaluate on Test Data
# -------------------------------------------------
test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss    : {test_loss:.4f}")

# -------------------------------------------------
# 7. Plot Training History
# -------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.close()

# -------------------------------------------------
# 8. Visualize Predictions (Correct vs Incorrect)
# -------------------------------------------------
predictions = model.predict(x_test_flat)
predicted_labels = np.argmax(predictions, axis=1)

# Find some correct and incorrect predictions
correct_idx = np.where(predicted_labels == y_test)[0]
incorrect_idx = np.where(predicted_labels != y_test)[0]

plt.figure(figsize=(10, 10))
plt.suptitle("Sample Predictions (Green = Correct, Red = Wrong)")
sample_idx = np.concatenate([correct_idx[:6], incorrect_idx[:6]])

for i, idx in enumerate(sample_idx):
    plt.subplot(4, 3, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    true_label = class_names[y_test[idx]]
    pred_label = class_names[predicted_labels[idx]]
    color = "green" if predicted_labels[idx] == y_test[idx] else "red"
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("predictions.png")
plt.close()

# -------------------------------------------------
# 9. Which Classes Are Hardest? (Confusion Insight)
# -------------------------------------------------
from collections import Counter

wrong_true_labels = y_test[incorrect_idx]
hardest_classes = Counter(wrong_true_labels).most_common()

print("\nMost frequently misclassified classes:")
for class_idx, count in hardest_classes[:5]:
    print(f"  {class_names[class_idx]:15s} -> {count} mistakes")

print("\nDone! Check sample_images.png, training_history.png, and predictions.png")
