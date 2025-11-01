import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode

iris = load_iris()
X = iris.data   
y_true = iris.target

plt.scatter(X[:,0], X[:,1], s=50)
plt.title("Iris Data - Feature Plot")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++',random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.labels_

print("Model predicted Cluster Labels:\n",y_kmeans)

labels = np.zeros_like(y_kmeans)
for i in range(3):
    mask = (y_kmeans == i)
    labels[mask] = mode(y_true[mask])[0]

print("Mapped Predicted labels to true labels :\n", labels)
print("Original labels:\n", y_true)

cm = confusion_matrix(y_true, labels)
accuracy = accuracy_score(y_true, labels)

print("Confusion Matrix:\n", cm)
print("\nAccuracy:", accuracy)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - KMeans")
plt.show()

for cluster_id in range(3):
    indices = np.where(labels == cluster_id)[0]  # indices of samples in this cluster
    flower_types = y_true[indices]               # actual flower labels
    unique, counts = np.unique(flower_types, return_counts=True)

    dominant_index = unique[np.argmax(counts)]
    dominant_flower = iris.target_names[dominant_index]

    print(f"\nCluster {cluster_id} → {dominant_flower}")
    for u, c in zip(unique, counts):
        print(f"  {iris.target_names[u]}: {c} samples")

plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis', s=50)
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()