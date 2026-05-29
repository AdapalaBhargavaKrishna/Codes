import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return 1 if x >= 0 else 0

def perceptron(X, y, lr=0.1, epochs=10):

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
    outputs = []

    for x in X:
        y = step(np.dot(x, w) + b)
        outputs.append(y)
        print(x, "->", y)

    return outputs

def plot_graph(X, outputs, w, b, title):
    for i, point in enumerate(X):
        if outputs[i] == 0:
            plt.scatter(point[0], point[1], marker='o', s=200, color='blue', label='Class 0' if i == 0 else "")
        else:
            plt.scatter(point[0], point[1], marker='x', s=200, color='red', label='Class 1' if i == 1 else "")

    x = np.linspace(-0.2, 1.2, 100)
    y = -(w[0] * x + b) / w[1]
    plt.plot(x, y, label='Decision Boundary')
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

AND  = np.array([0,0,0,1])
OR   = np.array([0,1,1,1])
NAND = np.array([1,1,1,0])
NOR  = np.array([1,0,0,0])

print("AND Gate")
w_and, b_and = perceptron(X, AND)
and_outputs = test(X, w_and, b_and)
print("Weights:", w_and)
print("Bias:", b_and)

plot_graph(X, and_outputs, w_and, b_and,
           "AND Gate Linear Separability")

print("\nOR Gate")
w_or, b_or = perceptron(X, OR)
or_outputs = test(X, w_or, b_or)

print("Weights:", w_or)
print("Bias:", b_or)

plot_graph(X, or_outputs, w_or, b_or,
           "OR Gate Linear Separability")

print("\nNAND Gate")
w_nand, b_nand = perceptron(X, NAND)
nand_outputs = test(X, w_nand, b_nand)

print("Weights:", w_nand)
print("Bias:", b_nand)

plot_graph(X, nand_outputs, w_nand, b_nand,
           "NAND Gate Linear Separability")

print("\nNOR Gate")
w_nor, b_nor = perceptron(X, NOR)
nor_outputs = test(X, w_nor, b_nor)

print("Weights:", w_nor)
print("Bias:", b_nor)

plot_graph(X, nor_outputs, w_nor, b_nor,
           "NOR Gate Linear Separability")


"""
========================================================
PERCEPTRON ARCHITECTURE
========================================================

Input Layer (2)      Output Layer (1)

   [x1] -----\
               \
   [x2] -------[Neuron] --> Output
               /
          [Bias]

========================================================
WEIGHTED SUM FUNCTION
========================================================

z = (x1*w1 + x2*w2) + b

========================================================
ACTIVATION FUNCTION
========================================================

step(z) = 1 if z >= 0
          0 otherwise

========================================================
TRAINABLE PARAMETERS
========================================================

Weights = 2
Bias    = 1

Total Trainable Parameters = 3

========================================================
LEARNING RULE
========================================================

w = w + lr * error * x
b = b + lr * error

========================================================
"""