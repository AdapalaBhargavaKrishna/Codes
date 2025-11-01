from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

base_learner = DecisionTreeClassifier(max_depth=3)

bagging_model = BaggingClassifier(
    estimator=base_learner,  # Base learner
    n_estimators=50,         # Number of base learners (trees)
    max_samples=0.8,         # Each tree gets 80% of samples
    bootstrap=True,          # Sampling with replacement
    random_state=42
)

bagging_model.fit(X_train, y_train)

y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Bagging Classifier Accuracy:", round(accuracy, 2))

# --- Step 8: Test with a new sample ---
# Example: new flower measurements [sepal_length, sepal_width, petal_length, petal_width]
new_sample = [[5.1, 3.5, 1.4, 0.2]]
predicted_class = bagging_model.predict(new_sample)[0]

class_name = data.target_names[predicted_class]
print("Predicted class for new sample:", class_name)

print("\n--- Individual Base Learner Predictions ---")
for i, estimator in enumerate(bagging_model.estimators_):
    pred = estimator.predict(new_sample)[0]
    class_name = data.target_names[pred]
    print(f"Model {i+1} predicts: {class_name}")

final_pred = bagging_model.predict(new_sample)[0]
final_class = data.target_names[final_pred]

print("\nFinal (majority-voted) prediction:", final_class) 