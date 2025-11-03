# 1. **Choose K** → number of nearest neighbors to consider.
# 2. **Calculate distance** → find how close each training point is (usually Euclidean).
# 3. **Pick K nearest points** → select the K closest data points.
# 4. **For classification:** take the **majority vote** among K neighbors.
# 5. **For regression:** take the **average** of their values.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

iris = load_iris()
X = iris.data      # Features
y = iris.target    # Labels (setosa, versicolor, virginica)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3,metric='manhattan')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

unique, counts = np.unique(y_test, return_counts=True)

print("Number of flowers in each category in Test Data:")
for cls, count in zip(unique, counts):
    print(f"{iris.target_names[cls]}: {count}")

cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.show()

new_sample = np.array([[5.5, 3.2, 1.5, 0.2]])

new_sample_scaled = scaler.transform(new_sample)
predicted_class = knn.predict(new_sample_scaled)[0]

print("New Sample:", new_sample)
print("Predicted Category:", iris.target_names[predicted_class],"(",predicted_class,")")
