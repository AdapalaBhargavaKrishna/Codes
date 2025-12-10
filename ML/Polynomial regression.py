import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Step 1: Generate quadratic data with noise
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

plt.plot(X, y, 'b.')
plt.xlabel("X"); plt.ylabel("y"); plt.title("Generated Quadratic Data")
plt.show()

# Step 2: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Step 3: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Linear Regression R^2:", r2_score(y_test, y_pred))

plt.plot(X_train, lr.predict(X_train), 'r', label='Linear Fit')
plt.plot(X, y, 'b.', label='Data Points')
plt.xlabel("X"); plt.ylabel("y"); plt.title("Linear Regression Fit"); plt.legend()
plt.show()

# Step 4: Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_poly, y_train)
y_pred = lr.predict(X_test_poly)
print("Polynomial Regression R^2:", r2_score(y_test, y_pred))

# Step 5: Visualize Polynomial Regression
X_new = np.linspace(-3, 3, 200).reshape(200, 1)
y_new = lr.predict(poly.transform(X_new))

plt.plot(X_new, y_new, 'r-', linewidth=2, label="Polynomial Fit")
plt.plot(X_train, y_train, 'b.', label='Training Points')
plt.plot(X_test, y_test, 'g.', label='Testing Points')
plt.xlabel("X"); plt.ylabel("y"); plt.title("Polynomial Regression Fit (Degree 2)")
plt.legend(); plt.show()