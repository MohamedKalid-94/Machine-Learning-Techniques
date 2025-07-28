# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
import numpy as np

# -----------------------------
# Load sample dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -----------------------------
# Choose a model (Logistic Regression)
# -----------------------------
model = LogisticRegression(max_iter=200)

# -----------------------------
# Set up K-Fold cross-validation (5 folds)
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# Apply cross-validation
# -----------------------------
scores = cross_val_score(model, X, y, cv=kf)

# -----------------------------
# Output the results
# -----------------------------
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Standard Deviation:", np.std(scores))
