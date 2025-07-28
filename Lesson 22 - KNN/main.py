# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load dataset
# -----------------------------
iris = load_iris()
X = iris.data         # Features
y = iris.target       # Labels

# -----------------------------
# Split into train and test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Create KNN model (K=3)
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=3)

# -----------------------------
# Train the model
# -----------------------------
knn.fit(X_train, y_train)

# -----------------------------
# Predict on test data
# -----------------------------
y_pred = knn.predict(X_test)

# -----------------------------
# Evaluate the model
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
