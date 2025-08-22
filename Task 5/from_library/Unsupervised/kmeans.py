import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Inertia 
print("Inertia (lower is better):", kmeans.inertia_)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap="viridis", edgecolor="k")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", s=200, marker="X", label="Centroids")
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
