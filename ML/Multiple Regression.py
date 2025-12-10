import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dataset (Features & Target)
X = np.array([
    [1500, 3, 10],
    [1800, 4, 15],
    [2400, 3, 20],
    [3000, 5, 8],
    [3500, 4, 12],
    [4000, 5, 5]
])

# Target: House prices in $1000s
Y = np.array([300, 400, 350, 500, 450, 550])

# 2. Train Linear Regression Model
model = LinearRegression()  
model.fit(X, Y)

# 3. Make Predictions
Y_pred = model.predict(X)

# 4. Evaluate Model
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Y_pred)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print(f"\nMean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# 5. Predict new data 
new_data = np.array([2000, 3, 10]).reshape(1, -1)
new_prediction = model.predict(new_data)
print(f"\nPrediction for new house (2000 sq ft, 3 bedrooms, 10 years old): "
      f"${new_prediction[0] * 1000:.2f}")

# 6. Compare Actual vs Predicted
print("\nActual Prices vs Predicted Prices:")
for i in range(len(Y)):
    print(f"Actual Price: ${Y[i] * 1000:.2f} \t Predicted Price: ${Y_pred[i] * 1000:.2f}")

# 7. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(Y, Y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual House Prices ($1000s)')
plt.ylabel('Predicted House Prices ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()