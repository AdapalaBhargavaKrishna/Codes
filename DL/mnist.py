import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

plt.imshow(X_train[0])
X_train = X_train/255
X_test = X_test/255

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=25,validation_split=0.2)

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
accuracy_score(y_test,y_pred)

plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Valid Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# 1. load image
img = cv2.imread("/content/digit_9.png")

# 2. grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. invert (MNIST style)
img = cv2.bitwise_not(img)

# 4. resize to 28x28
img = cv2.resize(img, (28, 28))

# 5. normalize
img = img / 255.0

# 6. reshape for model
img = img.reshape(1, 28, 28)

plt.imshow(img[0])
y_prob = model.predict(img)
print(y_prob)
argmax = y_prob.argmax()
print(argmax)

"""
====================================================================================================
EXPERIMENT 4: Handwritten Digit Classification using MLP with MNIST Dataset
====================================================================================================

DATASET: MNIST (Modified National Institute of Standards and Technology)
------------------------------------------------------------------------
- 70,000 images of handwritten digits (0-9)
- Training: 60,000 images
- Testing: 10,000 images
- Each image: 28×28 pixels (grayscale)
- Pixel values: 0-255 (0=black, 255=white)

ARCHITECTURE DETAILS:
====================
Input Layer: 784 neurons (28×28 flattened)
Hidden Layer 1: 128 neurons with ReLU activation
Hidden Layer 2: 32 neurons with ReLU activation
Output Layer: 10 neurons with Softmax activation

PARAMETERS COUNT:
================
LAYER 1 (Flatten): NO parameters! Just reshapes data
- Input: 28×28=784 values
- Output: 784 flattened values

LAYER 2 (Dense 128):
- Weights: 784 × 128 = 100,352 weights
- Biases: 128 biases
- Total: 100,480 parameters

LAYER 3 (Dense 32):
- Weights: 128 × 32 = 4,096 weights
- Biases: 32 biases
- Total: 4,128 parameters

LAYER 4 (Dense 10):
- Weights: 32 × 10 = 320 weights
- Biases: 10 biases
- Total: 330 parameters

TOTAL PARAMETERS: 100,480 + 4,128 + 330 = 104,938 trainable parameters!

WEIGHTS AND BIASES VISUALIZATION:
===============================

Input (784)    Hidden1 (128)    Hidden2 (32)    Output (10)
p1 ──w11──▶ n1 ──w11──▶ n1 ──w11──▶ d0
p2 ──w12──▶ n2 ──w12──▶ n2 ──w12──▶ d1
p3 ──w13──▶ n3 ──w13──▶ n3 ──w13──▶ d2
...         ...            ...            ...
p784─w1_128─▶ n128─w128_32─▶ n32─w32_10─▶ d9

BIAS:        b1_1..b1_128   b2_1..b2_32    b3_1..b3_10

DATA FLOW:
=========
Step 1: 28×28 image → Flatten → 784 pixel values
Step 2: 784 → 128 (ReLU) → Learn low-level features (edges, curves)
Step 3: 128 → 32 (ReLU) → Learn high-level patterns (digit parts)
Step 4: 32 → 10 (Softmax) → Probability distribution over digits

WHAT EACH LAYER LEARNS:
======================
Layer 1 (128 neurons):
- Detects basic patterns: edges, lines, curves
- Each neuron specializes in different feature
- Example: Neuron 45 activates when seeing vertical lines

Layer 2 (32 neurons):
- Combines basic features into digit parts
- Recognizes patterns like "circle" (for 0,8,9)
- Recognizes patterns like "vertical line" (for 1,4,7)

Output Layer (10 neurons):
- Each neuron represents one digit (0-9)
- Softmax converts scores to probabilities
- Highest probability = predicted digit

VISUALIZATION OF NEURON ACTIVATIONS:
===================================
Input Image:      Layer 1 Activations:    Layer 2 Activations:
████              ██  ██  ██              ██  ██
██  ██     →      ████ ████ ████     →     ████ ████
██  ██            ██  ██  ██                ██  ██
████              ██  ██  ██                ██  ██

Digit 8           Detects circles            Detects "8" pattern

TRAINING PROCESS:
================
Epochs 1-5: 
- Loss drops rapidly
- Accuracy increases from ~10% to ~90%
- Network learns basic patterns

Epochs 5-15:
- Fine-tuning features
- Accuracy reaches ~95-97%
- Validation loss may start increasing (overfitting)

Epochs 15-25:
- Training accuracy continues to increase
- Validation accuracy plateaus
- Possible overfitting (gap between train and val)

PREDICTION PROCESS:
==================
1. Image → Flatten → Normalize
2. Forward pass through network
3. Output probabilities: [0.01, 0.02, 0.85, 0.01, ...]
4. argmax → predicted digit (2 in this example)

CUSTOM IMAGE PREDICTION:
=======================
Steps for predicting on custom image:
1. Load image (any size)
2. Convert to grayscale
3. Invert (MNIST has white digit on black background)
4. Resize to 28×28
5. Normalize to 0-1
6. Reshape for model (1,28,28)
7. Predict!
"""