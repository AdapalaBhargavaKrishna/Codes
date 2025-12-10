from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pandas as pd    
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# 2. Train-test split
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3. Correlation analysis on training data

plt.figure(figsize=(12,10))
cor = X_train.corr()    
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.title("Feature Correlation Heatmap")
plt.show()

# Find features with correlation > 0.8
corr_features = {col for i , col in enumerate(cor.columns)
                 for j in range(i) if abs(cor.iloc[i,j] > 0.8)}

# 5. Function to return correlation pairs above threshold
def correlation_pairs(dataset, threshold):
    corr_matrix = dataset.corr()

    return [
        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
        for i in range(len(corr_matrix.columns))
        for j in range(i)
        if abs(corr_matrix.iloc[i, j]) > threshold
    ]

# Find correlated feature pairs with correlation > 0.8
corr_pairs = correlation_pairs(X_train, 0.8)
print(corr_pairs)

# 6. Drop correlated features from training and test sets
X_train = X_train.drop(corr_features, axis=1)
X_test = X_test.drop(corr_features, axis=1)