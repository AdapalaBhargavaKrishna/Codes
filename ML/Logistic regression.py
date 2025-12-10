import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("pima-indians-diabetes-classification.csv")

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Split the Data into Features and Target
X = df.drop('class', axis=1)  # Features
y = df['class']               # Target

# Step 3: Apply Standard Scaling to the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Step 5: Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(" The confusion matrix is:", cm)

# Accuracy Score   
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

num_non_diabetic_predicted = (y_test == 0).sum()
print("Number of non diabetic persons in test data (actual):", num_non_diabetic_predicted)

num_diabetic_predicted = (y_test == 1).sum()
print("Number of diabetic persons in test data (actual):", num_diabetic_predicted)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()