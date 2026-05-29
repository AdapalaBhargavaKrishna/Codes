import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([0,1,1,0])

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], marker='o', label='Class 0' if i==0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], marker='x', label='Class 1' if i==1 else "")
plt.title("XOR Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid()
plt.show()

model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=500, verbose=0)
pred = model.predict(X)

print("\nPredictions:")
for i in range(len(X)):
    print(X[i], "->", round(pred[i][0]))

plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

"""
====================================================================================================
EXPERIMENT 3: XOR Function using Multilayer Perceptron (MLP)
====================================================================================================

WHY XOR NEEDS MLP:
-----------------
XOR Problem: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
This is NOT linearly separable - NO single line can separate the classes!
That's why we need HIDDEN LAYERS!

ARCHITECTURE DETAILS:
====================
Input Layer: 2 neurons (x1, x2)
Hidden Layer: 4 neurons with ReLU activation
Output Layer: 1 neuron with Sigmoid activation

PARAMETERS COUNT:
================
LAYER 1 (Hidden): Dense(4, input_shape=(2,))
- Weights: 2 inputs × 4 neurons = 8 weights
- Biases: 4 neurons = 4 biases
- Total Layer 1: 8 + 4 = 12 parameters

LAYER 2 (Output): Dense(1)
- Weights: 4 inputs × 1 neuron = 4 weights
- Biases: 1 neuron = 1 bias
- Total Layer 2: 4 + 1 = 5 parameters

TOTAL PARAMETERS: 12 + 5 = 17 trainable parameters!

WEIGHTS AND BIASES VISUALIZATION:
===============================

Input      Hidden Layer (4 neurons)      Output
Layer          with ReLU                  Layer
                   
        ┌─▶ n1 ──┐
        │   w11  │
x1 ──┬──┼─▶ n2 ──┼──▶ y
     │  │   w12  │
     │  └─▶ n3 ──┘
     │      w13  
     └─┬──▶ n4 ──┘
        w14

x2 ──┬──▶ n1
     │   w21
     ├──▶ n2
     │   w22
     ├──▶ n3
     │   w23
     └──▶ n4
         w24

WEIGHT MATRICES:
===============
W1 (Input to Hidden): Shape [2, 4]
    [w11, w12, w13, w14]  # weights from x1 to each hidden neuron
    [w21, w22, w23, w24]  # weights from x2 to each hidden neuron

W2 (Hidden to Output): Shape [4, 1]
    [w31]  # weight from neuron1 to output
    [w32]  # weight from neuron2 to output
    [w33]  # weight from neuron3 to output
    [w34]  # weight from neuron4 to output

BIAS VECTORS:
=============
b1: [b11, b12, b13, b14]  # biases for 4 hidden neurons
b2: [b21]                  # bias for output neuron

HOW XOR IS SOLVED:
=================
The hidden layer learns to transform the non-linear XOR into a linearly separable problem!

Step 1: Hidden layer creates new representations:
   Input (x1,x2) → Hidden layer outputs (h1,h2,h3,h4)
   (0,0) → [~0, ~0, ~0, ~0]
   (0,1) → [~1, ~0, ~1, ~0]
   (1,0) → [~0, ~1, ~1, ~0]
   (1,1) → [~0, ~0, ~0, ~1]

Step 2: Output layer combines these:
   (0,0): low activation → 0
   (0,1): high activation → 1
   (1,0): high activation → 1
   (1,1): low activation → 0

VISUALIZATION OF DECISION BOUNDARY:
==================================
Without Hidden Layer (Perceptron):    With Hidden Layer (MLP):
   x2                                 x2
    ↑                                   ↑
  1 | x     o      Can't draw         1 |    /¯\_
    |              single line!          |   /    \_
  0 | o     x                          0 |  /      \¯o
    +------→ x1                          +----------→ x1
    0      1                             0         1
    
    NOT POSSIBLE!                         WORKS!

WHAT HAPPENS DURING TRAINING:
============================
Epochs 1-50: Loss decreases rapidly
Epochs 50-200: Loss continues decreasing
Epochs 200-500: Fine-tuning to reach near-zero loss

The network learns weights that create the XOR pattern:
- First hidden neuron detects (0,1) pattern
- Second hidden neuron detects (1,0) pattern
- Third hidden neuron helps with separation
- Fourth neuron acts as inhibitor for (1,1)
- Output combines these to produce correct XOR

Architecture Diagram:

          Hidden Layer
        [h1] [h2] [h3] [h4]

x1 -----> O    O    O    O
            \  |   / 
             \ |  /
x2 -----> O----O----O----O

                |
                v

            Output Layer
                 [y]

"""