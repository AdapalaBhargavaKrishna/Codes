import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Conv2D(6, (5,5), activation='tanh', input_shape=(28,28,1)),
    layers.AveragePooling2D((2,2)),
    layers.Conv2D(16, (5,5), activation='tanh'),
    layers.AveragePooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(120, activation='tanh'),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Train Acc:", history.history['accuracy'][-1])
print("Val Acc:", history.history['val_accuracy'][-1])
print("Test Acc:", test_acc)

plt.imshow(x_train[0].reshape(28,28), cmap='gray')
plt.title("Sample Image")
plt.show()

pred = model.predict(x_train[0].reshape(1,28,28,1))
print("Predicted Label:", np.argmax(pred))

"""
================================================================================
LENET-5 ARCHITECTURE ANALYSIS ON MNIST DATASET
================================================================================

1. ARCHITECTURE
================================================================================
Input Layer: 28×28×1 (Grayscale image)
↓
Conv2D: 6 filters, 5×5 kernel, tanh activation → Output: 24×24×6
↓
AveragePooling2D: 2×2 pool → Output: 12×12×6
↓
Conv2D: 16 filters, 5×5 kernel, tanh activation → Output: 8×8×16
↓
AveragePooling2D: 2×2 pool → Output: 4×4×16
↓
Flatten → Output: 256 (4×4×16)
↓
Dense: 120 neurons, tanh activation
↓
Dense: 84 neurons, tanh activation
↓
Dense: 10 neurons, softmax activation (Output)

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|-----------
Conv2D (6×5×5×1)         | 6×5×5×1 = 150        | 6       | 156
Conv2D (16×5×5×6)        | 16×5×5×6 = 2,400     | 16      | 2,416
Dense (256→120)          | 256×120 = 30,720     | 120     | 30,840
Dense (120→84)           | 120×84 = 10,080      | 84      | 10,164
Dense (84→10)            | 84×10 = 840          | 10      | 850
TOTAL                    | 44,190               | 236     | 44,426

3. LOSS CALCULATION FORMULA
================================================================================
Categorical Cross-Entropy:

For batch: Loss = - Σ Σ y_ic × log(p_ic)
                  i=1..N c=1..10

For single sample: Loss = -log(p_i[true_class])

where:
- N = number of samples in batch
- M = 10 (number of classes)
- y_ic = ground truth (1 if sample i belongs to class c, else 0)
- p_ic = predicted probability for sample i belonging to class c

4. WEIGHT UPDATE FORMULA
================================================================================
Adam Optimizer:

θ(t+1) = θ(t) - (η / (√v̂(t) + ε)) × m̂(t)

where:
- θ(t) = weights/biases at time step t
- η = learning rate
- m̂(t) = bias-corrected first moment (mean of gradients)
- v̂(t) = bias-corrected second moment (uncentered variance)
- ε = 10⁻⁷ (small constant for numerical stability)

For standard Gradient Descent:
θ(t+1) = θ(t) - η × ∇L(θ(t))

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
Total Training Samples: 60,000 (original MNIST training set)
Training Samples: 54,000 (90% of 60,000, validation_split=0.1)
Validation Samples: 6,000 (10% of 60,000)
Testing Samples: 10,000 (original MNIST test set)

6. LABELED OR UNLABELED DATA?
================================================================================
Labeled Data ✓

Each image has a corresponding digit label (0-9) used for supervised learning.
The labels are one-hot encoded using keras.utils.to_categorical().
"""