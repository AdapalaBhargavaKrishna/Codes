import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder

# 1. Create sample dataset
data = {
    'Color': ['Red', 'Red', 'Blue', 'Blue', 'Green'],
    'Size': ['S', 'S', 'M', 'L', 'L'],
    'Shape': ['Circle', 'Circle', 'Circle', 'Circle', 'Circle']
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# 2. One-Hot Encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop=None)
X_encoded = encoder.fit_transform(df)

# Get the names of the new one-hot encoded features
encoded_feature_names = encoder.get_feature_names_out(df.columns)

# Create a new DataFrame with one-hot encoded features
df_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)
print("\nOne-Hot Encoded DataFrame:\n", df_encoded)

# 3. Apply Variance Threshold
sel = VarianceThreshold(threshold=0.0)
X_reduced = sel.fit_transform(df_encoded)

# Get the names of columns that were kept
selected_columns = df_encoded.columns[sel.get_support()]

# Create a new reduced DataFrame
df_reduced = pd.DataFrame(X_reduced, columns=selected_columns)
print("\nReduced DataFrame (after VarianceThreshold):\n", df_reduced)