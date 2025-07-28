# -------------------------------
# Step 1: Import required libraries
# -------------------------------
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# -------------------------------
# Step 2: Generate binary classification data
# -------------------------------
# Generate a sample dataset (2 classes)
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=2, n_redundant=10,
                           n_classes=2, random_state=42)

# -------------------------------
# Step 3: Split data into train and test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# -------------------------------
# Step 4: Train a classification model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Get predicted probabilities
# -------------------------------
# We need probability scores for ROC curve
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# -------------------------------
# Step 6: Calculate FPR, TPR for ROC
# -------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# -------------------------------
# Step 7: Calculate AUC
# -------------------------------
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.2f}")

# -------------------------------
# Step 8: Plot ROC curve
# -------------------------------
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid()
plt.show()
