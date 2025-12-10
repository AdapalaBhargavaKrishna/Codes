from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# 1. Load Iris dataset
iris = load_iris()
X = iris.data           # Features
y = iris.target         # Target
feature_names = iris.feature_names

X_df = pd.DataFrame(X, columns=feature_names)
print("Original Features (first 5 rows):")
print(X_df.head())

# 2. Calculate Mutual Information (MI) scores
mi = mutual_info_classif(X, y)
print("\nMutual Information scores for each feature:")
for name, score in zip(feature_names, mi):
    print(f"{name}: {score:.4f}")

# 3. Select top k features using MI
k = 2  # number of top features to select
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_reduced = selector.fit_transform(X, y)

# Get indices and names of selected features
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]
print("\nSelected feature indices:", selected_indices)
print("Selected feature names :", selected_features)

# Identify removed features
removed_cols = [feature_names[i] for i in range(len(feature_names)) if i not in selected_indices]
print("Removed feature names:", removed_cols)

# 4. Reduce original DataFrame
X_df.drop(columns=removed_cols, axis=1, inplace=True)
print("\nReduced DataFrame (after MI-based feature selection):")
print(X_df.head())