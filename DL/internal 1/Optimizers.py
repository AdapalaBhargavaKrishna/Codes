from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten
from tensorflow.keras.optimizers import SGD , Adam
import matplotlib.pyplot as plt
import time

# Load MNIST dataset - 60,000 training images and 10,000 test images
# Each image is 28x28 pixels (grayscale)
(X_train , y_train) , (X_test , y_test) = mnist.load_data()

# Normalize pixel values from range 0-255 to 0-1
# This helps neural networks train better (smaller numbers = stable gradients)
X_train , X_test = X_train / 255.0 , X_test / 255.0

# Function to create the MLP architecture
def create_model():
    return Sequential([
        # Flatten converts 2D image (28x28) into 1D array (784 pixels)
        # input_shape specifies the size of input images
        Flatten(input_shape=(28,28)),
        
        # Hidden Layer: 128 neurons with ReLU activation
        # ReLU = Rectified Linear Unit (f(x) = max(0,x))
        # kernel_initializer='glorot_uniform' (Xavier init) sets initial weights properly
        Dense(128, activation='relu', kernel_initializer='glorot_uniform'),
        
        # Output Layer: 10 neurons (one for each digit 0-9)
        # Softmax converts outputs to probabilities (all sum to 1)
        Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
    ])

def run_experiment(name, optimizer, batch_size):
    model = create_model()

    # Good for integer labels (0,1,2...)
    model.compile(optimizer=optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    start = time.time()
    history = model.fit(X_train , y_train,epochs=5, batch_size=batch_size, verbose=0)
    end = time.time()

    train_acc = history.history['accuracy'][-1]
    train_loss = history.history['loss'][-1]

    # Print results for this optimizer
    print(f'{name}')
    print(f'Time: {end - start:.2f}s')
    print(f'Accuracy: {train_acc*100:.2f}%')
    print(f'Loss: {train_loss:.4f}')

    return history.history['loss'], name

results = []

# Experiment 1: Batch Gradient Descent
# Updates weights once after seeing ALL images
results.append(run_experiment("Batch GD",
                             SGD(learning_rate=0.01),
                             batch_size=len(X_train)))  # 60,000

# Experiment 2: Mini-batch GD with batch size 500
# Updates weights after every 500 images
results.append(run_experiment("Mini-batch GD (500)",
                             SGD(learning_rate=0.01),
                             batch_size=500))

# Experiment 3: Mini-batch GD with batch size 50
# Updates weights after every 50 images
results.append(run_experiment("Mini-batch GD (50)",
                             SGD(learning_rate=0.01),
                             batch_size=50))

# Experiment 4: Momentum-based GD
# momentum=0.9 helps escape local minima and speeds up convergence
# Like a ball rolling downhill - maintains velocity
results.append(run_experiment("Momentum GD",
                             SGD(learning_rate=0.01, momentum=0.9),
                             batch_size=50))

# Experiment 5: Adam Optimizer (Adaptive Moment Estimation)
# Combines momentum + adaptive learning rates
# Most popular optimizer - works well out of the box
results.append(run_experiment("Adam",
                             Adam(learning_rate=0.01),
                             batch_size=50))

# Experiment 6: Stochastic Gradient Descent
# batch_size = 1 (updates weights after EVERY SINGLE image)
# Very noisy but can escape local minima
results.append(run_experiment("Stochastic GD",
                             SGD(learning_rate=0.01),
                             batch_size=1))

for loss, name in results:
    plt.plot(loss, label=name)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Comparison Across Optimizers")
plt.legend()
plt.grid()
plt.show()

"""
====================================================================================================
EXPERIMENT IDENTIFICATION: This is Experiment 5 - Comparing different optimizers
====================================================================================================

ARCHITECTURE DETAILS (Important for exam):

INPUT LAYER:
-----------
- Number of neurons: 784 (28 pixels × 28 pixels)
- What it represents: Each neuron corresponds to one pixel in the image
- Values: Normalized pixel intensities from 0 (black) to 1 (white)

HIDDEN LAYER:
------------
- Number of neurons: 128
- Activation function: ReLU (Rectified Linear Unit) - f(x) = max(0, x)
- Purpose: Learn complex patterns and features from the input pixels
- Weight initialization: Glorot Uniform (Xavier) - helps with vanishing gradient

OUTPUT LAYER:
------------
- Number of neurons: 10 (one for each digit: 0,1,2,3,4,5,6,7,8,9)
- Activation function: Softmax - converts raw scores to probabilities
- Purpose: Predict which digit the image represents
- The neuron with highest probability is the predicted digit

WEIGHTS AND BIASES COUNT:
------------------------
- Input to Hidden layer: 784 × 128 = 100,352 weights
- Hidden layer biases: 128 biases
- Hidden to Output layer: 128 × 10 = 1,280 weights
- Output layer biases: 10 biases
- TOTAL TRAINABLE PARAMETERS: 100,352 + 128 + 1,280 + 10 = 101,770

WHAT THIS CODE DEMONSTRATES:
----------------------------
This experiment compares 6 different optimization algorithms:

1. BATCH GRADIENT DESCENT: Uses ALL 60,000 images to compute gradient
   - Most stable but slowest per epoch
   - Updates weights once per epoch

2. MINI-BATCH GD (500): Uses 500 images per update
   - Balance between stability and speed
   - 120 updates per epoch (60,000/500)

3. MINI-BATCH GD (50): Uses 50 images per update
   - More updates = faster convergence
   - 1,200 updates per epoch

4. MOMENTUM GD: Adds "velocity" to parameter updates
   - Helps escape local minima
   - Faster convergence on flat surfaces

5. ADAM: Adaptive learning rate + momentum
   - Most sophisticated optimizer
   - Adapts learning rate per parameter

6. STOCHASTIC GD: Uses 1 image per update
   - Most frequent updates (60,000 per epoch)
   - Very noisy but can find better minima

WHY THIS ARCHITECTURE WORKS FOR MNIST:
--------------------------------------
- Input 784: Matches the 28×28 pixel images
- Hidden 128: Enough capacity to learn digit patterns without overfitting
- Output 10: Matches the 10 possible digits (0-9)
- ReLU: Helps with vanishing gradient problem
- Softmax: Gives probability distribution over digits

KEY POINTS FOR EXAM:
-------------------
1. This is a MULTILAYER PERCEPTRON (MLP) with 1 hidden layer
2. It's a MULTI-CLASS CLASSIFICATION problem (10 classes)
3. The SAME architecture is used with DIFFERENT optimizers
4. Optimizers differ in HOW they update weights, not the network structure
5. Batch size determines how many samples are seen before weight update
"""