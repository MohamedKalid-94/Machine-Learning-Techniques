# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------- Definition -------------------

# Normal Distribution:
# Also called the Gaussian distribution, it is a symmetric, bell-shaped distribution
# characterized by its mean (μ) and standard deviation (σ).
# Most data points cluster around the mean, and the probability decreases as you move away from the mean.

# ------------------- Generate Normally Distributed Data -------------------

# Mean (μ) and standard deviation (σ)
mu = 50
sigma = 10

# Generate 1000 data points following a normal distribution
data = np.random.normal(mu, sigma, 1000)

# ------------------- Plot Histogram -------------------

plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

# ------------------- Plot Normal Distribution Curve -------------------

# Plot the theoretical normal distribution curve over the histogram
plt.plot(bins, norm.pdf(bins, mu, sigma), 'r--', linewidth=2, label='Normal Distribution Curve')

# ------------------- Add Labels and Show Plot -------------------

plt.title("📈 Normal Distribution (μ = 50, σ = 10)", fontsize=14)
plt.xlabel("Data Values")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
