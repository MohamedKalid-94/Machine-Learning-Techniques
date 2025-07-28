# Importing necessary libraries
import statistics

# ------------------- Definitions -------------------

# Mean: The average of a dataset; calculated by summing all values and dividing by the number of values.
# Median: The middle value in a sorted dataset; if even number of values, it's the average of the two middle ones.
# Mode: The most frequently occurring value in the dataset.

# ------------------- Input Data -------------------

# You can modify this list to test with different numbers
data = [1,2,3,4,5,6,7,8,9,10]

# ------------------- Calculations -------------------

# Calculate Mean
mean_value = statistics.mean(data)

# Calculate Median
median_value = statistics.median(data)

# Calculate Mode
mode_value = statistics.mode(data)

# ------------------- Output Results -------------------

print("Statistical Summary")
print("-------------------------")
print(f"Mean   : {mean_value}")
print(f"Median : {median_value}")
print(f"Mode   : {mode_value}")
