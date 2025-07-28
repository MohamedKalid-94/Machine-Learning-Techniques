# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# -----------------------------
# Generate synthetic data
# -----------------------------
# make_blobs creates clustered data for testing
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Plot the raw data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Original Data Distribution")
plt.show()

# -----------------------------
# Apply K-Means clustering
# -----------------------------
# n_clusters: number of clusters you want
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Predict the clusters for each data point
y_kmeans = kmeans.predict(X)

# -----------------------------
# Visualize the clustered data
# -----------------------------
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')  # color by cluster
centers = kmeans.cluster_centers_  # get cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

plt.title("K-Means Clustered Data")
plt.legend()
plt.show()
