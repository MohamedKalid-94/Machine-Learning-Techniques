# ------------------- Import Libraries -------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ------------------- Definition -------------------
# Polynomial Regression:
# Extends linear regression by fitting a polynomial curve (e.g., y = a + bx + cx² + ...).

# ------------------- Sample Data -------------------
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 6, 15, 28, 45])  # Non-linear relationship

# ------------------- Convert X to Polynomial Features -------------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# ------------------- Create and Train Model -------------------
model = LinearRegression()
model.fit(X_poly, y)

# ------------------- Predict -------------------
y_pred = model.predict(X_poly)

# ------------------- Plot -------------------
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='green', label='Polynomial Fit (deg=2)')
plt.title("📈 Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
