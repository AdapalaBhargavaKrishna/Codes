import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from tensorflow.keras.optimizers import Adam

X , y = make_circles(n_samples = 100, noise = 0.1, random_state = 1)

X_train , X_test , y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2)

model = Sequential()
model.add(Dense(128, input_dim = 2, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.01), metrics = ['accuracy'])
history = model.fit(X_train , y_train, validation_data = (X_test , y_test), epochs = 3500, verbose = 1)

train_acc = history.history['accuracy'][-1] * 100
val_acc = history.history['val_accuracy'][-1] * 100

if train_acc - val_acc >= 5:
    print("\nOverfitting is there in the model")
else:
    print("\nNo Overfitting in the model")

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plot_decision_regions(X_test , y_test.ravel(), clf = model, legend = 2)
plt.show()

model = Sequential()
model.add(Dense(256, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
callbacks = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    verbose = 1,
    mode = 'auto',
    restore_best_weights = True
)

history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=3500, callbacks=callbacks)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plot_decision_regions(X_test, y_test.ravel(), clf = model, legend=2)
plt.show()

"""
====================================================================================================
EXPERIMENT IDENTIFICATION: This is Experiment 8 - Early Stopping
====================================================================================================

DATASET: Make Circles (Concentric circles)
----------------------------------------
- 100 samples
- 2 features (x1, x2) - Circular pattern
- 2 classes (0 and 1) - Inner and outer circle
- noise = 0.1 - Slight overlap between circles
- Purpose: Creates non-linear decision boundary perfect for demonstrating overfitting

ARCHITECTURE FOR MODEL 1 (Without Early Stopping):
--------------------------------------------------
Input Layer: 2 neurons (Feature 1, Feature 2)
Hidden Layer: 128 neurons with ReLU activation
Output Layer: 1 neuron with Sigmoid activation (binary classification)

PARAMETERS COUNT (Model 1):
--------------------------
Layer 1: (2 inputs × 128 neurons) + 128 biases = 256 + 128 = 384 parameters
Layer 2: (128 × 1) + 1 bias = 128 + 1 = 129 parameters
TOTAL: 384 + 129 = 513 trainable parameters!

ARCHITECTURE FOR MODEL 2 (With Early Stopping):
-----------------------------------------------
Input Layer: 2 neurons
Hidden Layer: 256 neurons with ReLU activation (bigger model = more prone to overfitting!)
Output Layer: 1 neuron with Sigmoid activation

PARAMETERS COUNT (Model 2):
--------------------------
Layer 1: (2 × 256) + 256 = 512 + 256 = 768 parameters
Layer 2: (256 × 1) + 1 = 256 + 1 = 257 parameters
TOTAL: 768 + 257 = 1,025 trainable parameters!
"""

"""
====================================================================================================
MODEL 1 - WEIGHTS AND BIASES (Without Early Stopping)
====================================================================================================

LAYER 1: Dense(128, input_dim=2, activation='relu')
---------------------------------------------------
WEIGHTS MATRIX (W1): Shape [2, 128]
   [w11, w12, w13, ..., w1_128]  # Weights from x1 to each neuron
   [w21, w22, w23, ..., w2_128]  # Weights from x2 to each neuron
   Total weights = 2 × 128 = 256

BIAS VECTOR (b1): Shape [128]
   [b11, b12, b13, ..., b1_128]  # One bias per neuron
   Total biases = 128

LAYER 2: Dense(1, activation='sigmoid')
--------------------------------------
WEIGHTS MATRIX (W2): Shape [128, 1]
   [w11]  # Weight from neuron 1 to output
   [w21]  # Weight from neuron 2 to output
   [w31]  # Weight from neuron 3 to output
   ...
   [w128_1]  # Weight from neuron 128 to output
   Total weights = 128 × 1 = 128

BIAS VECTOR (b2): Shape [1]
   [b21]  # Single bias for output neuron
   Total biases = 1

SUMMARY TABLE - MODEL 1:
-----------------------
Layer      | Weights Shape | Weights Count | Biases Count | Total
-----------+---------------+---------------+--------------+-------
Layer 1    | [2, 128]      | 256           | 128          | 384
Layer 2    | [128, 1]      | 128           | 1            | 129
-----------+---------------+---------------+--------------+-------
TOTAL      |               | 384           | 129          | 513

====================================================================================================
MODEL 2 - WEIGHTS AND BIASES (With Early Stopping)
====================================================================================================

LAYER 1: Dense(256, input_dim=2, activation='relu')
---------------------------------------------------
WEIGHTS MATRIX (W1): Shape [2, 256]
   [w11, w12, w13, ..., w1_256]  # Weights from x1 to each neuron
   [w21, w22, w23, ..., w2_256]  # Weights from x2 to each neuron
   Total weights = 2 × 256 = 512

BIAS VECTOR (b1): Shape [256]
   [b11, b12, b13, ..., b1_256]  # One bias per neuron
   Total biases = 256

LAYER 2: Dense(1, activation='sigmoid')
--------------------------------------
WEIGHTS MATRIX (W2): Shape [256, 1]
   [w11]  # Weight from neuron 1 to output
   [w21]  # Weight from neuron 2 to output
   [w31]  # Weight from neuron 3 to output
   ...
   [w256_1]  # Weight from neuron 256 to output
   Total weights = 256 × 1 = 256

BIAS VECTOR (b2): Shape [1]
   [b21]  # Single bias for output neuron
   Total biases = 1

SUMMARY TABLE - MODEL 2:
-----------------------
Layer      | Weights Shape | Weights Count | Biases Count | Total
-----------+---------------+---------------+--------------+-------
Layer 1    | [2, 256]      | 512           | 256          | 768
Layer 2    | [256, 1]      | 256           | 1            | 257
-----------+---------------+---------------+--------------+-------
TOTAL      |               | 768           | 257          | 1,025
"""
