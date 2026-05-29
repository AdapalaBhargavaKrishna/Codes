import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae','accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
loss, mae, acc= model.evaluate(X_test, y_test)

predictions = model.predict(X_test[:5])
print("Predicted Prices:", predictions)

# Graph
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("House Price Prediction Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""
====================================================================================================
EXPERIMENT 6: House Price Prediction using MLP (Regression Problem)
====================================================================================================

DATASET: California Housing Dataset
----------------------------------
- 20,640 samples
- 8 features (input dimensions)
- Target: Median house value (in $100,000s)

FEATURES (Inputs):
-----------------
1. MedInc     - Median income in block group
2. HouseAge   - Median house age in block group
3. AveRooms   - Average number of rooms per household
4. AveBedrms  - Average number of bedrooms per household
5. Population - Block group population
6. AveOccup   - Average number of household members
7. Latitude   - Block group latitude
8. Longitude  - Block group longitude

TARGET (Output):
---------------
MedHouseVal - Median house value ($100,000s)

ARCHITECTURE DETAILS:
====================
Input Layer: 8 neurons (one for each feature)
Hidden Layer 1: 64 neurons with ReLU activation
Hidden Layer 2: 32 neurons with ReLU activation
Output Layer: 1 neuron with LINEAR activation (for regression!)

PARAMETERS COUNT:
================
LAYER 1: Dense(64, input_shape=(8,))
- Weights: 8 inputs × 64 neurons = 512 weights
- Biases: 64 neurons = 64 biases
- Total Layer 1: 512 + 64 = 576 parameters

LAYER 2: Dense(32)
- Weights: 64 inputs × 32 neurons = 2,048 weights
- Biases: 32 neurons = 32 biases
- Total Layer 2: 2,048 + 32 = 2,080 parameters

LAYER 3: Dense(1)
- Weights: 32 inputs × 1 neuron = 32 weights
- Biases: 1 neuron = 1 bias
- Total Layer 3: 32 + 1 = 33 parameters

TOTAL PARAMETERS: 576 + 2,080 + 33 = 2,689 trainable parameters!

WEIGHTS AND BIASES VISUALIZATION:
===============================

Input (8)    Hidden1 (64)    Hidden2 (32)    Output (1)
   x1 ──w11──▶ n1 ──w11──▶ n1 ──w11──▶ 
   x2 ──w12──▶ n2 ──w12──▶ n2 ──w12──▶ 
   x3 ──w13──▶ n3 ──w13──▶ n3 ──w13──▶ 
   x4 ──w14──▶ n4 ──w14──▶ n4 ──w14──▶ y
   x5 ──w15──▶ n5 ──w15──▶ n5 ──w15──▶ 
   x6 ──w16──▶ n6 ──w16──▶ n6 ──w16──▶ 
   x7 ──w17──▶ n7 ──w17──▶ n7 ──w17──▶ 
   x8 ──w18──▶ n8 ──w18──▶ n8 ──w18──▶ 
            ...        ...        ...
            n64        n32

BIAS:     b1_1..b1_64   b2_1..b2_32    b3_1

KEY DIFFERENCES FROM CLASSIFICATION:
===================================
1. OUTPUT ACTIVATION: Linear (not sigmoid/softmax)
   - Can predict any real number (house price)
   
2. LOSS FUNCTION: MSE (Mean Squared Error)
   - Measures squared difference between predicted and actual prices
   - Penalizes large errors more heavily

3. METRICS: MAE (Mean Absolute Error)
   - Average absolute difference in $100,000s
   - More interpretable: "Our predictions are off by $X on average"

4. ACCURACY METRIC: Not meaningful for regression!
   - accuracy metric shown is actually inappropriate here
   - Should use R² score or MAE instead

WHAT THE MODEL LEARNS:
=====================
The network learns complex relationships between features:
- Higher income → Higher house price (strong positive correlation)
- Location (latitude/longitude) → Price variations by area
- Room counts → Size of house affects price
- Population density → Urban vs rural pricing

FEATURE IMPORTANCE (Typical):
===========================
1. MedIncome     : Most important (highest weights)
2. Latitude/Long : Location matters
3. AveRooms      : House size
4. HouseAge      : Newer houses often more expensive
5. Population    : Less important
"""