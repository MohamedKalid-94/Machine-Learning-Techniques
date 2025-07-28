# ----------------- Import Necessary Libraries -----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------- Sample Dataset (for binary classification) -----------------
# Let's create a simple dataset: student exam scores and admission (yes/no)
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed_Exam':   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]   # Target variable (binary)
}
df = pd.DataFrame(data)

# ----------------- Split Features and Target -----------------
X = df[['Hours_Studied']]       # Feature matrix (must be 2D)
y = df['Passed_Exam']           # Target vector

# ----------------- Split into Train and Test Data -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------- Create and Train Logistic Regression Model -----------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------- Make Predictions -----------------
y_pred = model.predict(X_test)

# ----------------- Evaluate the Model -----------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# ----------------- Plotting the Logistic Regression Curve -----------------
# Generate a range of values to plot sigmoid curve
X_range = np.linspace(0, 11, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label="Actual")
plt.plot(X_range, y_prob, color='blue', label="Logistic Curve")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Hours Studied vs. Passed Exam")
plt.legend()
plt.grid(True)
plt.show()
