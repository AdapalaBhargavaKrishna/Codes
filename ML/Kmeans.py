import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode
#from sklearn.metrics import silhouette_score


# --- Load dataset ---
iris = load_iris()
X = iris.data
y_true = iris.target  # True labels

#-----Displaying Dataset ----
#df = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(df.head(150))
print(y_true) # Class 0 - setosa , Class 1 - versicolor, Class 2 - Virginica

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], s=50)
plt.show()

# --- Fit KMeans ---
kmeans = KMeans(n_clusters=3, init='k-means++',random_state=42)
kmeans.fit(X)

y_kmeans = kmeans.labels_
print("Model predicted Cluster Labels:\n",y_kmeans)
print("orginal Flower labels:\n",iris.target)

#Quality of clustering
#score = silhouette_score(X, y_kmeans)
#print("Silhouette Score:", score)

# --- Map cluster labels to true labels ---
# For each cluster, assign the most common true label
labels = np.zeros_like(y_kmeans)
for i in range(3):
    mask = (y_kmeans == i)
    labels[mask] = mode(y_true[mask])[0]

print("Mapped Predicted labels to true labels :\n", labels)
print("Original labels:\n", y_true)

# --- Confusion matrix and accuracy ---
cm = confusion_matrix(y_true, labels)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_true, labels)
print("\nAccuracy:", accuracy)
print("\n")

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - KMeans")
plt.show()

# --- Display which flower types are in each cluster ---
for cluster_id in range(3):
    indices = np.where(labels == cluster_id)[0]  # indices of samples in this cluster
    flower_types = y_true[indices]               # actual flower labels
    unique, counts = np.unique(flower_types, return_counts=True)

   # Determine the dominant (most frequent) flower type in this cluster
    dominant_index = unique[np.argmax(counts)]
    cluster_name = iris.target_names[dominant_index]

    print(f"\nCluster {cluster_id} → {cluster_name}")
    for u, c in zip(unique, counts):
        print(f"  {iris.target_names[u]}: {c} samples")

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis', s=50)
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()