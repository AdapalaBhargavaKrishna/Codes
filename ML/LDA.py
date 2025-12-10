import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 1. Load dataset (Iris for example)
iris = load_iris()
X = iris.data      # Features (4D)
y = iris.target    # Labels (3 classes)

# 2. Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply LDA
lda = LDA(n_components=2) #no.of linear discriminators(new axes)
X_lda = lda.fit_transform(X_scaled, y)
print("\nshape of  LDA:",X_lda.shape)
print("\n")
print(X_lda)

# 4. Plot the LDA projection
plt.figure(figsize=(8,6))
for label, marker, color in zip(range(3), ('o', 's', '^'), ('red', 'green', 'blue')):
    plt.scatter(
        X_lda[y == label, 0],
        X_lda[y == label, 1],
        marker=marker,
        color=color,
        alpha=0.7,
        label=iris.target_names[label]
    )
plt.xlabel("Linear Discriminant 1 (LD1)")
plt.ylabel("Linear Discriminant 2 (LD2)")
plt.title("Dimensionality Reduction with LDA (Iris Dataset)")
plt.legend()
plt.grid(True)
plt.show()