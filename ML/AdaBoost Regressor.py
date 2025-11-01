from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 2: Generate a synthetic regression dataset ---
X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_learner = DecisionTreeRegressor(max_depth=3)

model = AdaBoostRegressor(
    estimator=base_learner,   # base model (weak learner)
    n_estimators=50,          # number of boosting rounds
    learning_rate=0.8,        # step size to control weight updates
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))

print("\nWeights of estimators (alpha values):")
print(model.estimator_weights_)

print("\nErrors of each weak learner:")
print(model.estimator_errors_)