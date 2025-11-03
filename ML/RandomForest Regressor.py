# Random Forest Regressor is an ensemble learning algorithm used for regression tasks (predicting continuous values).
# It builds multiple Decision Trees, each trained on a random subset of the data and features.
# Each tree predicts a numerical value.
# The average of all tree predictions is taken as the final output.
# Reduces overfitting
# Improves prediction accuracy and stability

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=1000, n_features=10, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100,random_state=42)
print("No of decision trees in the forest:",regressor.n_estimators)
print("Criterion for decision tree:\n",regressor.criterion)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 3))

# --- Step 6: Test the model on a new sample ---
new_sample = np.array([[0.5, -1.2, 0.3, 2.1, -0.9, 0.8, 1.0, -0.5, 0.2, -1.0]])
predicted_value = regressor.predict(new_sample)
print("\nNew Sample:", new_sample)
print("Predicted Target Value:", predicted_value[0])

# --- Step 7 (Optional): Display predictions of all trees ---
tree_predictions = [tree.predict(new_sample)[0] for tree in regressor.estimators_]
print("\nPredictions from each Decision Tree:")
print(tree_predictions[:10], "...")  # Show first 10 predictions for brevity

# Average of tree predictions = Final Random Forest prediction
print("\nAverage of all tree predictions:", np.mean(tree_predictions))

# --- Step 9 (Optional): Plot 5 Decision Trees visually ---
for i in range(5):
    plot_tree(
        regressor.estimators_[i],
        feature_names=[f"X{i}" for i in range(X.shape[1])],
        filled=True,
        fontsize=8,
        max_depth=5
    )
    plt.title(f"Decision Tree #{i+1}")
    plt.show()
    