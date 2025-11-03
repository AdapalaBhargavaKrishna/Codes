# This program builds and trains an Artificial Neural Network (ANN) using the Keras library to classify whether a person is diabetic or not based on medical data from the Pima Indians Diabetes Dataset.

# The dataset contains 8 input features such as pregnancies, glucose level, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age — and one output label indicating whether the person is diabetic (1) or not (0).
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv("/content/pima-indians-diabetes-classification.csv")
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

model = Sequential([
    Dense(32, input_dim=8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Loss: binary_crossentropy → best for binary classification.
# Optimizer: adam → adaptive learning optimizer.
# Metric: Accuracy (to evaluate performance).

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
score = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {score[1]*100:.2f}%")

plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.title('Epochs vs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

new_data = np.array([[3.0, 14.0, 7.0, 35.0, 0.0, 33.6, 0.627, 50.0]])
new_data_scaled = scaler.transform(new_data)
pred = model.predict(new_data_scaled)

print("\nPredicted Probability:", pred[0][0])
print("Prediction:", "Diabetic (1)" if pred > 0.5 else "Non-Diabetic (0)")
