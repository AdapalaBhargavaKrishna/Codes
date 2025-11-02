import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode

iris = load_iris()
X = iris.data
y_true = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db = DBSCAN(eps=0.6, min_samples=5)
y_pred = db.fit_predict(X_scaled)
print("Cluster labels found by DBSCAN:", np.unique(y_pred))

# DBSCAN may label some points as -1 (noise)
mask = y_pred != -1  # exclude noise
y_pred_valid = y_pred[mask]
y_true_valid = y_true[mask]
labels = np.zeros_like(y_pred_valid)

for i in np.unique(y_pred_valid):
    mask2 = (y_pred_valid == i)
    labels[mask2] = mode(y_true_valid[mask2], keepdims=True).mode[0]

acc = accuracy_score(y_true_valid, labels)
print(f"\nDBSCAN Clustering Accuracy (excluding noise): {acc:.4f}")
cm = confusion_matrix(y_true_valid, labels)
print("\nConfusion Matrix:", cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(cmap='Blues')
plt.title("Confusion Matrix - DBSCAN Clustering")
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', s=50)
plt.title("DBSCAN Clustering Results")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.show()