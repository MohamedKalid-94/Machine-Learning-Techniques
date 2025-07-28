# Import necessary libraries
import matplotlib.pyplot as plt

# ------------------- Definition -------------------

# Data Distribution:
# In statistics, data distribution describes how values in a dataset are spread out.
# It helps us understand the shape, center, and spread (e.g., normal distribution, skewed, uniform).

# ------------------- Input Data -------------------

# Sample dataset – modify as needed
data = [55, 60, 65, 70, 75, 75, 80, 80, 85, 90, 90, 90, 95, 95, 100, 100, 100, 100]

# ------------------- Histogram Plot -------------------

# A histogram shows the frequency distribution of a dataset
plt.figure(figsize=(10, 6))
plt.hist(data, bins=7, color='skyblue', edgecolor='black')

# ------------------- Adding Labels -------------------

plt.title("📊 Data Distribution Histogram", fontsize=14)
plt.xlabel("Data Values")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# ------------------- Show Plot -------------------

plt.show()
