from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10,
                           n_informative=5, n_redundant=0,
                           random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("No.of samples in the training dataset:",X_train.shape)
print("No.of samples in the testing dataset:",X_test.shape)


# Base learner: Decision stump
base_learner = DecisionTreeClassifier(max_depth=2)

# AdaBoost model
model = AdaBoostClassifier(estimator=base_learner,
                           n_estimators=100,
                           learning_rate=0.3,
                           random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Step 4: Display each weak learner’s prediction ---
print("\n--- Predictions of Each Weak Learner ---")
for i, learner in enumerate(model.estimators_):
    preds = learner.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Weak Learner {i+1}: Accuracy = {acc:.2f}")
    print(f"Predictions: {preds[:10]}")  # show first 10 predictions
    print("-" * 60)

# --- Step 5: Display final AdaBoost prediction ---
final_preds = model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print("\nFinal AdaBoost Accuracy:", round(final_acc, 2))
print("Final AdaBoost Predictions (first 10):", final_preds[:10])

print("\n--- Cumulative Predictions after each boosting round ---")
for i, preds in enumerate(model.staged_predict(X_test)):
    acc = accuracy_score(y_test, preds)
    print(f"After {i+1} weak learners: Ensemble Accuracy = {acc:.2f}")

# --- Step 4: Display performance after each weak learner ---
print("\n=== Performance of Each Weak Learner on Training Data ===\n")

for i, learner in enumerate(model.estimators_):
    preds = learner.predict(X_train)
    correct_idx = np.where(preds == y_train)[0]
    misclassified_idx = np.where(preds != y_train)[0]

    num_correct = len(correct_idx)
    num_misclassified = len(misclassified_idx)

    print(f"Weak Learner {i+1}")
    print(f"Correctly Classified: {num_correct}")
    print(f"Misclassified: {num_misclassified}")

# --- Step 6: Test on a new sample (10 features) ---
new_sample = np.array([[0.5, -1.2, 0.3, 2.1, -0.9, 0.8, 1.0, -0.4, 0.2, 1.3]])  # example input

predicted_class = model.predict(new_sample)
predicted_prob = model.predict_proba(new_sample)

print("\n=== New Sample Test ===")
print("Input Sample (10 features):", new_sample)
print("Predicted Class:", predicted_class[0])
print("Class Probabilities [Class 0, Class 1]:", np.round(predicted_prob[0], 3))

# --- Step 6: Display weights ---

print("\nLearner Weights (αₜ):")
print(model.estimator_weights_)

print("\nLearner Errors (errₜ):")
print(model.estimator_errors_)