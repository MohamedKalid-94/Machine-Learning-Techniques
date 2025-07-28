# Import required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import classification_report

# Load dataset
iris = datasets.load_iris()
X = iris.data      # Feature matrix
y = iris.target    # Target vector

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model (SVM Classifier)
model = SVC()

# Define the parameter grid to search over
param_grid = {
    'C': [0.1, 1, 10],               # Regularization parameter
    'kernel': ['linear', 'rbf'],     # Kernel type
    'gamma': [0.001, 0.01, 0.1]      # Kernel coefficient for 'rbf'
}

# Set up Grid Search with cross-validation (cv=5)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, scoring='accuracy')

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print best parameters found by Grid Search
print("Best Parameters:", grid_search.best_params_)

# Print best accuracy score achieved during Grid Search
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
y_pred = grid_search.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
