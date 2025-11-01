# ===============================================
# PART 1: Import Required Libraries
# ===============================================

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from keras.layers import Dropout
import numpy as np

# ===============================================
# PART 2: Load Dataset
# ===============================================

dataset = np.loadtxt("/content/pima-indians-diabetes.csv", delimiter=',')

# Split into input features (X) and output label (y)
X = dataset[:, 0:8]   # 8 features
y = dataset[:, 8]     # target label (0 or 1)

#Rows and Cols in the dataset
print(dataset.shape)

# ===============================================
# PART 3: Preprocess Data (Feature Scaling)
# ===============================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================================
# PART 4: Train-Test Split
# ===============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print("Training Data:",X_train.shape)
print("Testing Data:",X_test.shape)

# ===============================================
# PART 5: Build and Compile Model
# ===============================================

model = Sequential([
    Dense(32, input_dim=8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#Model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ===============================================
# PART 6: Train and Evaluate Model
# ===============================================

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)


score = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {score[1]*100:.2f}%")

# ===============================================
# PART 7: Visualize Performance and Predict
# ===============================================

# Plot accuracy curve
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.title('Epochs vs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss curve
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Predict for new sample
new_data = np.array([[3.0, 14.0, 7.0, 35.0, 0.0, 33.6, 0.627, 50.0]])
new_data_scaled = scaler.transform(new_data)
pred = model.predict(new_data_scaled)

print("\nPredicted Probability:", pred[0][0])
print("Prediction:", "Diabetic (1)" if pred > 0.5 else "Non-Diabetic (0)")

