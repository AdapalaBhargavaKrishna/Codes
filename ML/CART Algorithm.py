import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
class_names = load_iris().target_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# CART Decision Tree (Gini Index)
clf = DecisionTreeClassifier(
    criterion="gini",   # CART uses Gini
    max_depth=5,
    random_state=42
)
clf.fit(X_train, y_train)

plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

# Predictions on custom samples
print("Class of the Flower:", clf.predict([[3, 15, 4, 1.5]]))
print("Class of the Flower:", clf.predict([[5.1, 3.5, 1.4, 0.2]]))
print("Class of the Flower:", clf.predict([[5.9, 3.5, 5.1, 1.8]]))

# Prediction with class name
pred = clf.predict([[3, 15, 4, 1.5]])[0]
print("Class of the Flower:", pred, "-", class_names[pred])

# Confusion Matrix
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy of CART Decision Tree:", acc)