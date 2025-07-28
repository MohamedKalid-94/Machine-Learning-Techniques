# ------------------- Feature Scaling -------------------
# Definition:
# Feature scaling helps normalize input data so that all features have equal weight in machine learning models.
# Two common types:
# 1. StandardScaler: Scales data to have mean = 0 and standard deviation = 1.
# 2. MinMaxScaler: Scales data to a fixed range (usually 0 to 1).

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample dataset with different scales
# Each row is an observation; each column is a feature
data = np.array([
    [100, 1.5],
    [200, 2.5],
    [300, 3.5],
    [400, 4.5],
    [500, 5.5]
])

print("🔹 Original Data:\n", data)

# 1️⃣ Standard Scaling (Z-score Normalization)
standard_scaler = StandardScaler()
data_standard_scaled = standard_scaler.fit_transform(data)

print("\n📘 Standard Scaled Data (mean=0, std=1):\n", data_standard_scaled)

# 2️⃣ Min-Max Scaling (0 to 1 range)
minmax_scaler = MinMaxScaler()
data_minmax_scaled = minmax_scaler.fit_transform(data)

print("\n📘 Min-Max Scaled Data (range 0 to 1):\n", data_minmax_scaled)
