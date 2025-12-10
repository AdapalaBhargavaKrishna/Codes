import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load dataset (Iris dataset as an example)
iris = load_iris()
X = iris.data
y = iris.target

print("Original shape:", X.shape)   # (150 samples, 4 features)

# Step 3: Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA (reduce to 2 dimensions for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced shape:", X_pca.shape)   # (150 samples, 2 features)
print(X_pca[:,0])

#print principal components
print("\nprint principal components(eigenvectors)&weightage of the features\n",pca.components_)

# Eigenvalues are proportional to explained variance
print("\nEigenvalues:", pca.explained_variance_)

#How much information (variance) from the original data is preserved in each principal component.”
print("\n orginal information preserved in the PCAs:",pca.explained_variance_ratio_)

# Step 5: Plot results
print("\n")
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k", s=80)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset (2D Projection)")
plt.colorbar(label="Class")
plt.show()