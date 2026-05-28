import numpy as np
import matplotlib.pyplot as plt

def mp_neuron(x, threshold):
    z = np.sum(x)
    return 1 if z < threshold else 0

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# ---------------- NAND GATE ----------------

print("NAND Gate using MP Neuron")

nand_outputs = []

for x in inputs:
    y = mp_neuron(x, threshold=2)
    nand_outputs.append(y)
    print(x, "->", y)

# GRAPH FOR NAND GATE
plt.figure(figsize=(6,6))

for i, point in enumerate(inputs):

    if nand_outputs[i] == 0:
        plt.scatter(point[0], point[1],
                    marker='x',
                    s=200,
                    color='red',
                    label='Class 0' if i == 3 else "")

    else:
        plt.scatter(point[0], point[1],
                    marker='o',
                    s=200,
                    color='blue',
                    label='Class 1' if i == 0 else "")

# Decision Boundary: x1 + x2 = 1.5
x = np.linspace(-0.2, 1.2, 100)
y = 1.5 - x

plt.plot(x, y, label='Decision Boundary')

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)

plt.xlabel('x1')
plt.ylabel('x2')

plt.title('NAND Gate Linear Separability')

plt.grid(True)
plt.legend()
plt.show()


# ---------------- NOR GATE ----------------

print("\nNOR Gate using MP Neuron")

nor_outputs = []

for x in inputs:
    y = mp_neuron(x, threshold=1)
    nor_outputs.append(y)
    print(x, "->", y)

# GRAPH FOR NOR GATE
plt.figure(figsize=(6,6))

for i, point in enumerate(inputs):

    if nor_outputs[i] == 0:
        plt.scatter(point[0], point[1],
                    marker='x',
                    s=200,
                    color='red',
                    label='Class 0' if i == 1 else "")

    else:
        plt.scatter(point[0], point[1],
                    marker='o',
                    s=200,
                    color='blue',
                    label='Class 1')

# Decision Boundary: x1 + x2 = 0.5
x = np.linspace(-0.2, 1.2, 100)
y = 0.5 - x

plt.plot(x, y, label='Decision Boundary')

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)

plt.xlabel('x1')
plt.ylabel('x2')

plt.title('NOR Gate Linear Separability')

plt.grid(True)
plt.legend()
plt.show()


'''
NAND/NOR Architecture:
Input Layer (2)    Output Layer (1)

   [x1] -----\
               \
   [x2] -------[Neuron] --> Output (0 or 1)
               /
          [Bias]
'''