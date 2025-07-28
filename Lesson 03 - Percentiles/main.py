# Importing the statistics and numpy modules
import numpy as np

# ------------------- Definition -------------------

# Percentile:
# A percentile is a measure used in statistics to indicate the value below which
# a given percentage of observations fall.
# For example, the 75th percentile is the value below which 75% of the data lies.

# ------------------- Input Data -------------------

# Sample dataset – feel free to modify it
data = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# ------------------- User-defined Percentiles -------------------

# List of percentiles to calculate
percentile_values = [25, 50, 75, 90]  # 25th, 50th (median), 75th, 90th percentiles

# ------------------- Calculation -------------------

print("Percentile Calculation")
print("--------------------------")
print(f"Data: {data}")

# Loop through each percentile and calculate using numpy
for p in percentile_values:
    result = np.percentile(data, p)
    print(f"{p}th Percentile: {result}")
