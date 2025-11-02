from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.show()

# --- Simple 2D Visualization ---
X_2D, y_2D = X[:, :2], y
scaler_2D = StandardScaler()
X_2D_scaled = scaler_2D.fit_transform(X_2D)
svm.fit(X_2D_scaled, y_2D)

x_min, x_max = X_2D_scaled[:, 0].min()-1, X_2D_scaled[:, 0].max()+1
y_min, y_max = X_2D_scaled[:, 1].min()-1, X_2D_scaled[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_2D_scaled[:, 0], X_2D_scaled[:, 1], c=y_2D, cmap='viridis', s=40, edgecolor='k')
plt.title("SVM Decision Boundaries (2 Scaled Features)")
plt.xlabel("Scaled Sepal Length")
plt.ylabel("Scaled Sepal Width")
plt.show()