import numpy as np

def step(x):
    return 1 if x >= 0 else 0

def perceptron(X, y, lr = 0.1, epochs=10):
    w = np.zeros(2)
    b = 0

    for _ in range(epochs):
        for i in range(len(X)):
            z = np.dot(X[i], w) + b

            y_pred = step(z)

            error = y[i] - y_pred

            w += lr * error * X[i] 

            b += lr * error
    return w, b

def test(X, w, b):
    for x in X:
        print(x, step(np.dot(x,w) + b))

X = np.array([[0,0],[0,1],[1,0],[1,1]])

AND = np.array([0,0,0,1])
OR = np.array([0,1,1,1])
NOR = np.array([1,0,0,0])
XOR = np.array([0,1,1,0])


print("AND Gate")
w,b = perceptron(X,AND)
test(X,w,b)

print("\nOR Gate")
w,b = perceptron(X,OR)
test(X,w,b)

print("\nNOR Gate")
w,b = perceptron(X,NOR)
test(X,w,b)

print("\nXOR Gate")
w,b = perceptron(X,XOR)
test(X,w,b)

"""
====================================================================================================
EXPERIMENT IDENTIFICATION: This is Experiment 2 - Perceptron Learning Algorithm for Boolean functions
====================================================================================================

ARCHITECTURE DETAILS:
--------------------
Input Layer: 2 neurons (x1, x2)
Hidden Layer: None (single layer)
Output Layer: 1 neuron with step activation

WEIGHTS AND BIASES:
------------------
Initially: w = [0, 0], b = 0 (all zeros)
After training: They are LEARNED, not manually set!

LEARNING PROCESS:
----------------
For each epoch (10 times):
    For each training example (0,0), (0,1), (1,0), (1,1):
        1. Calculate: z = (x1*w1 + x2*w2) + b
        2. Predict: y_pred = step(z)
        3. Calculate error = actual - predicted
        4. Update weights: w += lr * error * x
        5. Update bias: b += lr * error

LEARNING RATE (lr = 0.1):
------------------------
Controls how much weights change per update
- Too high: May overshoot solution
- Too low: Learns slowly

WHAT HAPPENS FOR EACH GATE:
--------------------------

1. AND GATE:
   Target: [0,0,0,1]
   After learning: w ≈ [0.2, 0.2], b ≈ -0.3
   Why: Both inputs need to be 1 to fire

2. OR GATE:
   Target: [0,1,1,1]
   After learning: w ≈ [0.2, 0.2], b ≈ -0.1
   Why: Either input being 1 can fire

3. NOR GATE:
   Target: [1,0,0,0]
   After learning: w ≈ [-0.2, -0.2], b ≈ 0.1
   Why: Needs negative weights to implement NOT OR

4. XOR GATE:
   Target: [0,1,1,0]
   After learning: Will FAIL!
   Why: XOR is NOT linearly separable

THE XOR PROBLEM (Important for exam):
------------------------------------
XOR truth table:
0,0 → 0
0,1 → 1
1,0 → 1
1,1 → 0

Can you draw a single straight line to separate:
- Output 0: (0,0) and (1,1)
- Output 1: (0,1) and (1,0)

IMPOSSIBLE! That's why perceptron FAILS for XOR
This is why we need MLP (Experiment 3)!

PARAMETERS SUMMARY:
------------------
Each gate: 2 weights + 1 bias = 3 parameters (learned)

ARCHITECTURE VISUALIZATION:
--------------------------
Input Layer (2)    Output Layer (1)
   [x1] -----\
              \
   [x2] -------[Neuron] --> Output (0 or 1)
              /
         [Bias]
      (learned)

KEY POINTS FOR EXAM:
-------------------
1. This demonstrates SUPERVISED LEARNING (inputs with labels)
2. Perceptron Convergence Theorem: Algorithm will find solution IF data is linearly separable
3. XOR proves limitation of single-layer perceptrons
4. Learning happens through ERROR-CORRECTION rule
5. Weights and biases are INITIALIZED to zero, then UPDATED during training

WHAT THE CODE OUTPUTS:
---------------------
- AND, OR, NOR: Will work correctly after learning
- XOR: Will give wrong answers (probably 50% accuracy)
      Example output might be: [0,0,0,0] or [0,1,1,1] - never correct!
"""