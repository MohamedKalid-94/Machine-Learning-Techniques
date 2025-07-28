# Importing the statistics module from Python's standard library
import statistics

# ------------------- Definition -------------------

# Standard Deviation:
# It measures the amount of variation or dispersion of a set of values.
# A low standard deviation means the values tend to be close to the mean.
# A high standard deviation indicates the values are spread out over a wider range.

# ------------------- Input Data -------------------

# Sample dataset – you can change the values for testing
data = [10, 12, 23, 23, 16, 23, 21, 16]

# ------------------- Calculation -------------------

# Calculate the Standard Deviation using statistics module
std_dev = statistics.stdev(data)

# ------------------- Output Result -------------------

print("📊 Standard Deviation Calculation")
print("----------------------------------")
print(f"Data            : {data}")
print(f"Standard Deviation: {std_dev:.2f}")
