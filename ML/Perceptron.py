import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

# Perceptron training
def perceptron_train(X, y, learning_rate=0.1, epochs=10):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    print(f"Initial Weights:\n Weights: {weights}, Bias: {bias}")

    for epoch in range(epochs):
        for i in range(n_samples):
            linear_output = np.dot(X[i], weights) + bias
            y_pred = step_function(linear_output)

            # Weight update rule
            update = learning_rate * (y[i] - y_pred)
            weights += update * X[i]
            bias += update

        print(f"Epoch {epoch+1} completed:\n Weights: {weights}, Bias: {bias}")

    return weights, bias

# Prediction
def perceptron_predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return step_function(linear_output)

# -------------------------
# Example Dataset
# Features: [CGPA, Communication Skill (0-10)]
# Label: 1 = Placed, 0 = Not Placed
# -------------------------
X = np.array([
    [8.5, 7],
    [7.0, 6],
    [6.0, 5],
    [9.0, 9],
    [5.0, 4],
    [8.0, 8]
])

y = np.array([1, 0, 0, 1, 0, 1])  # placement labels

# Train perceptron
weights, bias = perceptron_train(X, y, learning_rate=0.1, epochs=10)

# Test new student
new_student = np.array([[7.5, 8]])
prediction = perceptron_predict(new_student, weights, bias)
print("\nPrediction for new student:", "Placed" if prediction[0] == 1 else "Not Placed")
