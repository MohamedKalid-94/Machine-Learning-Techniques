# ------------------- Train-Test Split -------------------
# Goal: Train a machine learning model on part of the data (train set)
#       and test it on the remaining unseen data (test set)

# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1️⃣ Create sample dataset
# X is the input (features), y is the output (target)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # feature
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # target (y = 2x)

# 2️⃣ Split the data into training and testing sets
# test_size=0.3 means 30% for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("✅ Training Data (X_train):\n", X_train)
print("✅ Testing Data (X_test):\n", X_test)

# 3️⃣ Train a linear regression model using the training data
model = LinearRegression()
model.fit(X_train, y_train)

# 4️⃣ Predict using the test data
y_pred = model.predict(X_test)

# 5️⃣ Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("\n📊 Predicted Values:", y_pred)
print("✅ Mean Squared Error:", mse)
