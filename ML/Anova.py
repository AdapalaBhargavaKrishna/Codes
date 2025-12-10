import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# 1. Load a sample dataset (Iris)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. Optional: Scale the features (recommended for ANOVA)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# 3. Apply ANOVA F-test for feature selection
k = 2
anova_selector = SelectKBest(score_func=f_classif, k=k)
X_reduced = anova_selector.fit_transform(X_scaled, y)

# 4. Get ANOVA F-scores
scores = anova_selector.scores_
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'ANOVA F-Score': scores
}).sort_values(by='ANOVA F-Score', ascending=False)

# 5. Display results
print("ANOVA F-Scores for Each Feature:")
print(feature_scores)

print("\nReduced Dataset (Top", k, "features):")
selected_columns = X.columns[anova_selector.get_support()]
X_reduced_df = pd.DataFrame(X_reduced, columns=selected_columns)
print(X_reduced_df.head())