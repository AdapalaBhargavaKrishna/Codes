import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Sample Data
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'Study_Hours_Category': ['Low', 'High', 'Medium', 'High', 'Medium'],
    'Tuition_Class': ['No', 'Yes', 'Yes', 'No', 'Yes'],
    'Parent_Education': ['High', 'Low', 'Medium', 'Medium', 'Low'],
    'Result': ['Fail', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)
print(df)

le = LabelEncoder()
X = df.drop('Result', axis=1).apply(le.fit_transform)
y = le.fit_transform(df['Result'])  # Pass=1, Fail=0
print("\nEncoded Data:")
print(X)

# Calculate MI scores
mi_scores = mutual_info_classif(X, y, discrete_features=True)

# Create a Series with MI scores and sort
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\nMutual Information Scores:")
print(mi_results)

# Get top 2 feature names
top2_features = mi_results.head(2).index.tolist()

print("\nTop 2 Features based on MI:")
print(top2_features)

# Drop other columns from original DataFrame
reduced_df = df[top2_features + ['Result']]

# Display reduced DataFrame
print("\nReduced DataFrame with Top 2 MI Features:")
print(reduced_df)