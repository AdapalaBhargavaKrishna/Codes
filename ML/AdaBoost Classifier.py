from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=100, 
    learning_rate=0.3, 
    random_state=42
).fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("First 10 Predictions:", y_pred[:10])

# Weak learner performance
for i, est in enumerate(model.estimators_[:5]):  # show first 5 learners
    acc = accuracy_score(y_test, est.predict(X_test))
    print(f"Weak Learner {i+1}: Acc = {acc:.2f}")

# Staged accuracy
print("\nEnsemble Progress:")
for i, preds in enumerate(model.staged_predict(X_test)):
    print(f"After {i+1} learners: {accuracy_score(y_test, preds):.2f}")

# Test new sample
sample = np.array([[0.5, -1.2, 0.3, 2.1, -0.9, 0.8, 1.0, -0.4, 0.2, 1.3]])
print("\nNew Sample Prediction:", model.predict(sample)[0])
print("Class Probabilities:", np.round(model.predict_proba(sample)[0], 3))

# Weights & Errors
print("\nLearner Weights:", np.round(model.estimator_weights_[:10], 3))
print("Learner Errors:", np.round(model.estimator_errors_[:10], 3))
