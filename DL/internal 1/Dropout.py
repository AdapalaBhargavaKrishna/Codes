import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model_no_dropout = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_no_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_no = model_no_dropout.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
train_acc = history_no.history['accuracy'][-1] *100
val_acc = history_no.history['val_accuracy'][-1] * 100

if train_acc - val_acc >= 5:
    print("\nOverfitting is there in the model")
else:
    print("\nNo Overfitting in the model")

plt.figure()
plt.plot(history_no.history['accuracy'], label="Train Acc (No Dropout)")
plt.plot(history_no.history['val_accuracy'], label="Val Acc (No Dropout)")
plt.title("Accuracy without Dropout")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history_no.history['loss'], label="Train Loss (No Dropout)")
plt.plot(history_no.history['val_loss'], label="Val Loss (No Dropout)")
plt.title("Loss without Dropout")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model_dropout = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_do = model_dropout.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
train_acc = history_do.history['accuracy'][-1] *100
val_acc = history_do.history['val_accuracy'][-1] * 100

plt.figure()
plt.plot(history_do.history['accuracy'], label="Train Acc (Dropout)")
plt.plot(history_do.history['val_accuracy'], label="Val Acc (Dropout)")
plt.title("Accuracy with Dropout")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history_do.history['loss'], label="Train Loss (Dropout)")
plt.plot(history_do.history['val_loss'], label="Val Loss (Dropout)")
plt.title("Loss with Dropout")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""
====================================================================================================
EXPERIMENT 9: Dropout Regularization to Prevent Overfitting
====================================================================================================

DATASET: Make Circles (Modified)
--------------------------------
- 1000 samples (more data than previous examples)
- 2 features (x1, x2)
- 2 classes (0 and 1)
- noise = 0.2, factor = 0.5 (creates more complex pattern)
- Purpose: Complex enough to cause overfitting

ARCHITECTURE FOR MODEL 1 (Without Dropout):
-------------------------------------------
Input Layer: 2 neurons (Feature 1, Feature 2)
Hidden Layer 1: 64 neurons with ReLU activation
Hidden Layer 2: 32 neurons with ReLU activation
Output Layer: 1 neuron with Sigmoid activation

PARAMETERS COUNT (Model 1 - No Dropout):
---------------------------------------
Layer 1: (2 × 64) + 64 = 128 + 64 = 192 parameters
Layer 2: (64 × 32) + 32 = 2,048 + 32 = 2,080 parameters
Layer 3: (32 × 1) + 1 = 32 + 1 = 33 parameters
TOTAL: 192 + 2,080 + 33 = 2,305 trainable parameters!

ARCHITECTURE FOR MODEL 2 (With Dropout):
---------------------------------------
Input Layer: 2 neurons
Hidden Layer 1: 64 neurons with ReLU + Dropout(0.5)
Hidden Layer 2: 32 neurons with ReLU + Dropout(0.3)
Output Layer: 1 neuron with Sigmoid

PARAMETERS COUNT (Model 2 - With Dropout):
-----------------------------------------
SAME architecture! Dropout doesn't add parameters, it modifies training:
Layer 1: (2 × 64) + 64 = 128 + 64 = 192 parameters
Layer 2: (64 × 32) + 32 = 2,048 + 32 = 2,080 parameters
Layer 3: (32 × 1) + 1 = 32 + 1 = 33 parameters
TOTAL: 192 + 2,080 + 33 = 2,305 trainable parameters! (Same count)
"""