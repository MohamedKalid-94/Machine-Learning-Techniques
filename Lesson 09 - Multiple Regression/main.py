# ------------------- Import Libraries -------------------
from sklearn.linear_model import LinearRegression
import numpy as np

# ------------------- Definition -------------------
# Multiple Linear Regression:
# Models the relationship between two or more independent variables and one dependent variable.

# ------------------- Sample Data -------------------
# Features: [hours studied, number of mock tests]
X = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 3]
])
y = np.array([35, 45, 55, 65, 75])  # Exam scores

# ------------------- Create and Train Model -------------------
model = LinearRegression()
model.fit(X, y)

# ------------------- Predict -------------------
y_pred = model.predict(X)

# ------------------- Output Coefficients -------------------
print("📘 Multiple Linear Regression")
print("Coefficients (weights):", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted Scores:", y_pred)
