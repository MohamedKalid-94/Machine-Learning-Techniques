"""
Bootstrap Aggregation (Bagging):
--------------------------------
Bagging is an ensemble machine learning technique that improves the stability and accuracy of models.
It works by training multiple base estimators (like decision trees) on different random subsets
of the training dataset (with replacement) and then averaging their predictions (for regression)
or using majority vote (for classification).

This reduces variance and helps prevent overfitting, especially with high-variance models like decision trees.
"""

# Import necessary libraries
from sklearn.ensemble import BaggingClassifier  # For bagging ensemble
from sklearn.tree import DecisionTreeClassifier  # Base estimator
from sklearn.datasets import load_iris  # Sample dataset
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.metrics import accuracy_score  # To evaluate performance

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Create a base estimator (a decision tree in this case)
base_estimator = DecisionTreeClassifier()

# Create the BaggingClassifier using the base estimator
# Note: Use `estimator=` instead of deprecated `base_estimator=`
bagging_model = BaggingClassifier(estimator=base_estimator,
                                  n_estimators=10,      # Number of models in ensemble
                                  max_samples=0.8,      # Each model sees 80% of data (sampled with replacement)
                                  bootstrap=True,       # Enable sampling with replacement
                                  random_state=42)

# Train the bagging model
bagging_model.fit(X_train, y_train)

# Predict on the test set
y_pred = bagging_model.predict(X_test)

# Calculate accuracy of the ensemble model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Bagging (Bootstrap Aggregation) Accuracy:", accuracy)
