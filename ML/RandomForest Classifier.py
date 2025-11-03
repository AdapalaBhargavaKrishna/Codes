# Random Forest is an ensemble learning algorithm that builds multiple Decision Trees and combines their results to improve accuracy and stability.
# Creates many decision trees (using random samples of data & features).
# Each tree predicts a class.
# The majority vote among all trees is taken as the final prediction.
# ✅ This reduces overfitting (common in single decision trees).
# ✅ Works well for both classification and regression.

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000,n_features=10,n_informative=5,n_redundant=0,random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Model predicted test data class Labels:\n",y_pred)
print("Actual test data class Labels:\n",y_test)

print("Number of trees in the forest:",clf.n_estimators)
print("max no. of samples in the subset:",clf.max_samples)
print("Criterion used to construct decision:",clf.criterion)

accuracy=accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % accuracy)

# --- Step 6: Test the model on a new sample ---
new_sample = np.array([[0.5, -1.2, 0.3, 2.1, -0.9, 0.8, 1.0, -0.5, 0.2, -1.0]])
predicted_class = clf.predict(new_sample)
predicted_prob = clf.predict_proba(new_sample)

print("\nNew Sample:", new_sample)
print("Predicted Class:", predicted_class[0])
print("Predicted Probabilities:", predicted_prob[0])

# --- Display predictions from each individual decision tree ---
tree_predictions = [estimator.predict(new_sample)[0] for estimator in clf.estimators_]
# count how many trees voted for each class
print("\nVote counts from trees:", Counter(tree_predictions))

# --- Step 5: Visualize individual decision trees ---
for i in range(5):
    tree.plot_tree(
        clf.estimators_[i],
        filled=True,
        feature_names=[f"Feature {j}" for j in range(X.shape[1])],
        class_names=['Class 0', 'Class 1'],
        rounded=True,
        proportion=True,
        fontsize=8
    )
    plt.title(f"Decision Tree {i+1} in the Random Forest")
    plt.show()
    