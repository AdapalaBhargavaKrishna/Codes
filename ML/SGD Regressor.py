import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset (California housing)
data = fetch_california_housing()
X = data.data[:, [0]]  # pick one feature (MedInc = median income)
y = data.target        # target = house price

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create SGDRegressor model
model = SGDRegressor(
    max_iter=1000, tol=1e-3, eta0=0.01, learning_rate="constant", random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_[0])
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 7. Plot the results (scatter + regression line)
plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("SGD Regression on California Housing Dataset")
plt.legend()
plt.show()