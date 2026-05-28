import numpy as np
import matplotlib.pyplot as plt

def mp_neuron(x, threshold):
    z = np.sum(x)
    return 1 if z >= threshold else 0

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

print("AND Gate using MP Neuron")
and_outputs = []
for x in inputs:
    y = mp_neuron(x, threshold=2)
    and_outputs.append(y)
    print(x, "->", y)

print("\nOR Gate using MP Neuron")
or_outputs = []
for x in inputs:
    y = mp_neuron(x, threshold=1)
    or_outputs.append(y)
    print(x, "->", y)

# GRAPH FOR AND GATE
plt.figure(figsize=(6,6))

for i, point in enumerate(inputs):
    if and_outputs[i] == 0:
        plt.scatter(point[0], point[1], marker='o', s=200, label='Class 0' if i == 0 else "")
    else:
        plt.scatter(point[0], point[1], marker='x', s=200, label='Class 1')

# Decision Boundary: x1 + x2 = 1.5
x = np.linspace(-0.2, 1.2, 100)
y = 1.5 - x
plt.plot(x, y, label='Decision Boundary')

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('AND Gate Linear Separability')
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(6,6))

for i, point in enumerate(inputs):
    if or_outputs[i] == 0:
        plt.scatter(point[0], point[1], marker='o', s=200, label='Class 0')
    else:
        plt.scatter(point[0], point[1], marker='.', s=200, label='Class 1' if i == 1 else "")

# Decision Boundary: x1 + x2 = 0.5
x = np.linspace(-0.2, 1.2, 100)
y = 0.5 - x
plt.plot(x, y, label='Decision Boundary')

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('OR Gate Linear Separability')
plt.grid(True)
plt.legend()
plt.show()

'''
AND/OR/NOR Architecture:
Input Layer (2)    Output Layer (1)
   [x1] -----\
              \
   [x2] -------[Neuron] --> Output (0 or 1)
              /
         [Bias]
'''