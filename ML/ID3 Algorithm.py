import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix,accuracy_score
# Load dataset
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
class_names = load_iris().target_names

# Split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)

# Train Decision Tree (using entropy like ID3)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(12,8))
plot_tree(clf,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,          # Color nodes by class
          rounded=True,         # Rounded corners
          fontsize=10)
plt.show()

print("Class of the Flower:",clf.predict([[3,15,4,1.5]]))
print("Class of the Flower:",clf.predict([[5.1,3.5,1.4,0.2]]))
print("Class of the Flower:",clf.predict([[5.9,3.5,5.1,1.8]]))

# Prediction
pred = clf.predict([[3, 15, 4, 1.5]])[0]

# Print class number and name
print("Class of the Flower:", pred, "-", class_names[pred])

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree:", acc)