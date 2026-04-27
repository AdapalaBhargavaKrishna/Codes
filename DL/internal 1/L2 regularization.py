import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

X , y = make_moons(100 , noise = 0.25, random_state=42)

plt.scatter(X[y==0,0], X[y == 0,1], label='Class 0')
plt.scatter(X[y==1,0], X[y == 1,1], label='Class 1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter Plot with Legend")
plt.legend()
plt.show()

model1 = Sequential()
model1.add(Dense(128, input_dim = 2, activation='relu'))
model1.add(Dense(128, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.summary()

model1.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
history1 = model1.fit(X , y , epochs = 200 , validation_split = 0.2, verbose = 1)

train_acc = history1.history['accuracy'][-1] * 100
val_acc = history1.history['val_accuracy'][-1] * 100

if train_acc - val_acc >= 5:
    print("\nOverfitting is there in the model")    
else:
    print("\nNo Overfitting in the model")

plot_decision_regions(X, y.astype('int'), clf=model1, legend=2, 
                      X_highlight=None)
plt.xlim(-2,3)
plt.ylim(-1.5,2)
plt.show()


plt.plot(history1.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()


model2 = Sequential()
model2.add(Dense(128, input_dim = 2, activation = 'relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001)))
model2.add(Dense(128, activation = 'relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001)))
model2.add(Dense(1, activation='sigmoid'))
model2.summary()

model2.compile(loss='binary_crossentropy', optimizer = Adam(learning_rate = 0.01), metrics = ['accuracy'])
history2 = model2.fit(X, y, epochs = 200 , validation_split = 0.2 , verbose = 1)

train_acc = history2.history['accuracy'][-1] *100
val_acc = history2.history['val_accuracy'][-1] * 100

print(f"Last Epoch Training Accuracy: {train_acc:.2f}%")
print(f"Last Epoch Validation Accuracy: {val_acc:.2f}%")

if train_acc - val_acc >= 5:
    print("\nOverfitting is there in the model")
else:
    print("\nNo Overfitting in the model")

plot_decision_regions(X, y.astype('int'), clf=model2, legend=2)
plt.xlim(-2,3)
plt.ylim(-1.5,2)
plt.show()

plt.plot(history2.history['accuracy'], label='Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

"""
====================================================================================================
MODEL 2: WITH L2 REGULARIZATION - WEIGHTS AND BIASES DETAILS
====================================================================================================

SAME ARCHITECTURE as Model 1, but with L2 penalty on weights!

LAYER 1: Dense(128, input_dim=2, activation='relu', kernel_regularizer=l2(0.001))
--------------------------------------------------------------------------------
WEIGHTS MATRIX (W1): Shape [2, 128] - 256 weights
   [w11, w12, w13, ..., w1_128]
   [w21, w22, w23, ..., w2_128]
   
BIAS VECTOR (b1): Shape [128] - 128 biases
   [b11, b12, b13, ..., b1_128]

LAYER 2: Dense(128, activation='relu', kernel_regularizer=l2(0.001))
-------------------------------------------------------------------
WEIGHTS MATRIX (W2): Shape [128, 128] - 16,384 weights
   [w11, w12, ..., w1_128]
   [w21, w22, ..., w2_128]
   ...
   [w128_1, w128_2, ..., w128_128]
   
BIAS VECTOR (b2): Shape [128] - 128 biases
   [b21, b22, b23, ..., b2_128]

LAYER 3: Dense(1, activation='sigmoid')  # No regularization on output layer usually
----------------------------------------
WEIGHTS MATRIX (W3): Shape [128, 1] - 128 weights
   [w31]
   [w32]
   ...
   [w3_128]
   
BIAS VECTOR (b3): Shape [1] - 1 bias
   [b31]

SUMMARY TABLE - MODEL 2:
-----------------------
Layer      | Weights Count | Biases Count | Regularization | Total Parameters
-----------+---------------+--------------+----------------+-----------------
Layer 1    | 256           | 128          | L2(0.001)      | 384
Layer 2    | 16,384        | 128          | L2(0.001)      | 16,512
Layer 3    | 128           | 1            | None           | 129
-----------+---------------+--------------+----------------+-----------------
TOTAL      | 16,768        | 257          | L2 on 16,640   | 17,025
                                           weights
"""

"""
====================================================================================================
EXPERIMENT IDENTIFICATION: This is Experiment 7 - L2 Regularization
                       Also touches on concepts from Experiment 9 (Overfitting prevention)
====================================================================================================

DATASET: Make Moons (Non-linear dataset)
---------------------------------------
- 100 samples
- 2 features (x1, x2)
- 2 classes (0 and 1)
- noise = 0.25 (adds some overlap between classes)
- Purpose: Creates non-linear decision boundary to demonstrate overfitting

ARCHITECTURE FOR MODEL 1 (Without Regularization):
--------------------------------------------------
Input Layer: 2 neurons (Feature 1, Feature 2)
Hidden Layer 1: 128 neurons with ReLU activation
Hidden Layer 2: 128 neurons with ReLU activation
Output Layer: 1 neuron with Sigmoid activation (binary classification)

PARAMETERS COUNT (Model 1):
--------------------------
Layer 1: (2 inputs × 128 neurons) + 128 biases = 256 + 128 = 384 parameters
Layer 2: (128 × 128) + 128 biases = 16,384 + 128 = 16,512 parameters
Layer 3: (128 × 1) + 1 bias = 128 + 1 = 129 parameters
TOTAL: 384 + 16,512 + 129 = 17,025 trainable parameters!

ARCHITECTURE FOR MODEL 2 (With L2 Regularization):
--------------------------------------------------
Same architecture but with kernel_regularizer=l2(0.001)
Adds penalty to large weights during training

WHAT IS L2 REGULARIZATION?
-------------------------
L2 regularization adds a penalty term to the loss function:
Loss = Original Loss + (λ/2) * Σ(w²)

Where:
- λ (lambda) = 0.001 (regularization strength)
- Σ(w²) = sum of squared weights
- This prevents weights from becoming too large
- Larger λ = stronger regularization

VISUALIZATION COMPONENTS:
------------------------
1. Scatter plot: Shows the moon-shaped data
2. Decision regions: Shows how model separates classes
3. Accuracy plots: Shows training vs validation accuracy over epochs

KEY OBSERVATIONS:
----------------
Model 1 (No Regularization):
- Training accuracy: Very high (near 100%)
- Validation accuracy: Lower (gap indicates overfitting)
- Decision boundary: Very complex, wiggly (overfits to noise)

Model 2 (With L2 Regularization):
- Training accuracy: Slightly lower
- Validation accuracy: Higher (better generalization)
- Decision boundary: Smoother, simpler (ignores noise)

OVERFITTING DETECTION:
---------------------
Code checks: if train_acc - val_acc >= 5%
If gap ≥ 5% → Overfitting present
If gap < 5% → No significant overfitting

LEARNING PROCESS:
----------------
- Optimizer: Adam (learning_rate=0.01)
- Loss: Binary Crossentropy (for binary classification)
- Epochs: 200
- Validation split: 20% of data used for validation

WHAT THE CODE DEMONSTRATES:
--------------------------
1. Without regularization: Model memorizes training data (overfits)
2. With L2 regularization: Model generalizes better to unseen data
3. Regularization acts as a penalty on complex models
4. Trade-off: Slightly lower training accuracy for better validation accuracy

ARCHITECTURE VISUALIZATION:
--------------------------
Input (2) → Hidden1 (128) → Hidden2 (128) → Output (1)
   ↑            ↑               ↑              ↑
Features    ReLU + L2       ReLU + L2      Sigmoid
           (Model 2 only)  (Model 2 only)

EXPECTED RESULTS:
----------------
Without L2:
- Training accuracy: ~98-100%
- Validation accuracy: ~85-90%
- Gap: 10-15% (OVERFITTING!)

With L2:
- Training accuracy: ~92-95%
- Validation accuracy: ~90-93%
- Gap: 2-5% (GOOD GENERALIZATION!)

KEY CONCEPTS FOR EXAM:
--------------------
1. OVERFITTING: Model learns noise instead of pattern
2. REGULARIZATION: Technique to prevent overfitting
3. L2 REGULARIZATION: Adds weight penalty to loss function
4. VALIDATION ACCURACY: True measure of model performance
5. GENERALIZATION: Model's ability to perform on new data

WHY THIS ARCHITECTURE:
---------------------
- 2 inputs: Matches the 2 features in moon dataset
- 128 neurons: Enough capacity to learn complex patterns
- 2 hidden layers: Can learn non-linear decision boundaries
- L2 penalty: Prevents weights from exploding
"""