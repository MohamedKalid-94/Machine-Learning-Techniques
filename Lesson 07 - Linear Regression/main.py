# ------------------- Import Libraries -------------------
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Definition -------------------
# Linear Regression:
# Models the relationship between a single independent variable (X) and a dependent variable (y)
# using a straight line (y = mx + c).

# ------------------- Sample Data -------------------
# X = hours studied, y = exam scores
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshape for sklearn
y = np.array([30, 40, 50, 60, 70])

# ------------------- Create and Train Model -------------------
model = LinearRegression()
model.fit(X, y)

# ------------------- Predict -------------------
y_pred = model.predict(X)

# ------------------- Plot -------------------
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Linear Fit')
plt.title("📈 Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()
