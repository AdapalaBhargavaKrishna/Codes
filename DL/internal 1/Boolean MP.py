import numpy as np

def step(x):
    return 1 if x >= 0 else 0

def neuron(x, w, b):
    z = np.dot(x,w) + b
    return step(z)

inputs = np.array([[0,0] , [0,1] , [1,0] , [1,1]])

print('AND Gate')
for x in inputs:
    print(x, neuron(x, [1,1], -1.5))
    
print('OR Gate')
for x in inputs:
    print(x, neuron(x, [1,1], -0.5))

print('NOR Gate')
for x in inputs:
    print(x, neuron(x, [-1,-1], 0.5))

print('NOT Gate')
for x in [0,1]:
    print(x , step(-1*x + 0.5))

'''
AND/OR/NOR Architecture:
Input Layer (2)    Output Layer (1)
   [x1] -----\
              \
   [x2] -------[Neuron] --> Output (0 or 1)
              /
         [Bias]

NOT Architecture:
Input Layer (1)    Output Layer (1)
   [x] -----------[Neuron] --> Output (0 or 1)
                /
           [Bias]
'''

def mp_neuron(x, threshold):
    z = np.sum(x)
    return 1 if z >= threshold else 0

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])

print('AND Gate using MP Neuron')
for x in inputs:
    print(x, mp_neuron(x, threshold=2))

print('\nOR Gate using MP Neuron')
for x in inputs:
    print(x, mp_neuron(x, threshold=1))