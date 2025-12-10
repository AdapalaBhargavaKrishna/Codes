import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_train_pred_lr = lin_reg.predict(X_train)
y_test_pred_lr = lin_reg.predict(X_test)

print("Linear Regression:")
print("Train R²:", r2_score(y_train, y_train_pred_lr))
print("Test R²:", r2_score(y_test, y_test_pred_lr))

ridge = Ridge(alpha=10)   # regularization strength
ridge.fit(X_train, y_train)

y_train_pred_ridge = ridge.predict(X_train)
y_test_pred_ridge = ridge.predict(X_test)

print("\nRidge Regression:")
print("Train R²:", r2_score(y_train, y_train_pred_ridge))
print("Test R²:", r2_score(y_test, y_test_pred_ridge))

alphas = np.logspace(-3, 3, 50)  # test different alpha values
  #np.logspace(start, stop, num) generate num values b/w (10 power start) and (10 power stop)
train_scores = []
test_scores = []
#print(alphas)
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    train_scores.append(r2_score(y_train, ridge.predict(X_train)))
    test_scores.append(r2_score(y_test, ridge.predict(X_test)))

plt.semilogx(alphas, train_scores, label="Train R²")
#It’s like plt.plot(), but the x-axis is on a logarithmic scale, while the y-axis stays linear.
plt.semilogx(alphas, test_scores, label="Test R²")
plt.xlabel("Alpha (λ)")
plt.ylabel("R² Score")
plt.legend()
plt.title("Ridge Regression: Effect of Regularization on Overfitting")
plt.show()