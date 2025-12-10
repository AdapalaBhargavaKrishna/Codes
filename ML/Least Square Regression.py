import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv('placement_cgpa.csv')

# 2. Scatter plot of CGPA vs Package
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in LPA)')
plt.show()

# 3. Split into features (X) and target (y)
X = df.iloc[:, 0:1]   # CGPA column
y = df.iloc[:, -1]    # Package column

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 5. Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict package for the first test sample
print("cgpa:", X_test.iloc[0].values)
print("Predicted package:", lr.predict(X_test.iloc[0].values.reshape(1, 1)))

# Predict for all test values
y_pred = lr.predict(X_test)

# 6. Plot regression line with training data
plt.scatter(df['cgpa'], df['package'])
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in LPA)')
plt.show()

# 7. Model parameters
m = lr.coef_
print("Slope (m):", m)

b = lr.intercept_
print("Intercept (b):", b)

# 8. Error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # corrected to square root
r2 = r2_score(y_test, y_pred)

# Display the error values
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')

# 9. Predictions using equation y = mx + b
print("Package (manual calc) for CGPA=8.58:", m * 8.58 + b)
print("Package (model prediction) for CGPA=8.58:", lr.predict([[8.58]]))

print("Package (manual calc) for CGPA=9.5:", m * 9.5 + b)
print("Package (model prediction) for CGPA=9.5:", lr.predict([[9.5]]))