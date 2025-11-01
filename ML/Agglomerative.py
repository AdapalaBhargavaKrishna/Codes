import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode

iris = load_iris()
X = iris.data
y_true = iris.target

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_pred = agg.fit_predict(X)


labels = np.zeros_like(y_pred)
for i in range(3):
    mask = (y_pred == i)
    labels[mask] = mode(y_true[mask], keepdims=True).mode[0]


acc = accuracy_score(y_true, labels)
print(f"\nAgglomerative Clustering Accuracy: {acc:.4f}")


cm = confusion_matrix(y_true, labels)
print("\nConfusion Matrix:", cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(cmap='Blues')
plt.title("Confusion Matrix - Agglomerative Clustering")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title("Agglomerative Clustering Results")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()