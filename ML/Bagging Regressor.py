from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 2: Generate synthetic regression data ---
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_regressor = DecisionTreeRegressor(max_depth=5)
bagging_regressor = BaggingRegressor(
    estimator=base_regressor,
    n_estimators=50,       # Number of base regressors
    max_samples=0.8,       # Each tree gets 80% of data
    bootstrap=True,        # Sampling with replacement
    random_state=42
)
bagging_regressor.fit(X_train, y_train)
y_pred = bagging_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))

# --- Step 8: Test with a new custom sample ---
new_sample = [[0.5, -1.2, 3.3, 0.8, 1.0, -0.5, 2.2, -0.9, 1.1, 0.4]]

predicted_value = bagging_regressor.predict(new_sample)[0]
print("\nPredicted value for new sample:", round(predicted_value, 2))

# --- Step 9: Check predictions of few individual base models ---
print("\n--- Individual Base Learner Predictions (first 5 models) ---")
for i, model in enumerate(bagging_regressor.estimators_[:5]):
    pred = model.predict(new_sample)[0]
    print(f"Model {i+1} predicts: {round(pred, 2)}")

print("\nFinal (average) prediction:", round(predicted_value, 2))
