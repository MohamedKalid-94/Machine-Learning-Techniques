# ------------------- Import Necessary Libraries -------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ------------------- Definition -------------------
# Confusion Matrix:
# A confusion matrix is a performance measurement tool for classification problems.
# It shows how many predictions were:
# - True Positives (TP): Correctly predicted positives
# - True Negatives (TN): Correctly predicted negatives
# - False Positives (FP): Incorrectly predicted as positive
# - False Negatives (FN): Incorrectly predicted as negative
# For multi-class classification, it extends to show actual vs predicted counts across all classes.

# ------------------- Load Dataset -------------------
iris = load_iris()
X = iris.data
y = iris.target

# ------------------- Split Dataset -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------- Train Decision Tree Classifier -------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------- Make Predictions -------------------
y_pred = model.predict(X_test)

# ------------------- Generate Confusion Matrix -------------------
cm = confusion_matrix(y_test, y_pred)

# ------------------- Display Confusion Matrix -------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)  # Use a blue color map
plt.title("Confusion Matrix - Decision Tree (Iris Dataset)")
plt.grid(False)
plt.show()

# ------------------- Accuracy -------------------
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", accuracy)
