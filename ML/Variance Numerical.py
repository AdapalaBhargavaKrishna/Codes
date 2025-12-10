import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# 1. Create sample dataset
X = np.array([
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0, 1, 1, 3],
    [0, 1, 0, 3],
    [0, 1, 3, 3]
])

column_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
df = pd.DataFrame(X, columns=column_names)
print("Original DataFrame:")
print(df)

# 2. Calculate variance of each feature
variances = df.var()
print("\nVariance of each feature:")
print(variances)

# 3. Apply Variance Threshold
sel = VarianceThreshold(threshold=0)
X_new = sel.fit_transform(df)

# Get a mask of selected features
selected_mask = sel.get_support()

# Get names of selected columns
selected_columns = df.columns[selected_mask]

# Create a DataFrame with only the selected features
df_selected = pd.DataFrame(X_new, columns=selected_columns)
print("\nSelected features (after VarianceThreshold):")
print(df_selected)