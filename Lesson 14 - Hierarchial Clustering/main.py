# ----------------- Import Libraries -----------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# ----------------- Generate Sample Data -----------------
# Creating synthetic data with 3 clusters
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# ----------------- Plot Dendrogram -----------------
# Perform hierarchical/agglomerative clustering
# 'ward' method minimizes variance within clusters
linked = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# ----------------- Fit Agglomerative Clustering -----------------
# Creating model with 3 clusters
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
labels = model.fit_predict(X)

# ----------------- Visualize Clusters -----------------
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("Clusters after Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
