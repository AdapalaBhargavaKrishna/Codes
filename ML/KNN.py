import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load iris dataset
iris = load_iris()
X = iris.data      # Features
y = iris.target    # Labels (setosa, versicolor, virginica)

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

# Split into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(X_train.shape)
print(X_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create KNN classifier with k=5

knn = KNeighborsClassifier(n_neighbors=3,metric='manhattan')

# Train model
knn.fit(X_train, y_train)

import numpy as np

# Count number of flowers in each category of test set
unique, counts = np.unique(y_test, return_counts=True)

print("Number of flowers in each category in Test Data:")
for cls, count in zip(unique, counts):
    print(f"{iris.target_names[cls]}: {count}")

# Predict test set results
y_pred = knn.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("Test samples:",X_test.shape)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# Plot confusion matrix
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.show()

# Example: New flower sample [sepal length, sepal width, petal length, petal width]
new_sample = np.array([[5.5, 3.2, 1.5, 0.2]])

# Scale it using the same scaler (important!)
new_sample_scaled = scaler.transform(new_sample)

# Predict class
predicted_class = knn.predict(new_sample_scaled)[0]

# Display result
print("New Sample:", new_sample)
print("Predicted Category:", iris.target_names[predicted_class],"(",predicted_class,")")
