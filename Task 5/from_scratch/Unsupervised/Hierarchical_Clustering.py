import numpy as np

class HierarchicalClusteringScratch:
    # Agglomerative Hierarchical Clustering

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        clusters = [[i] for i in range(n_samples)]

        def distance(c1, c2):
            pts1, pts2 = X[c1], X[c2]
            return np.mean([np.linalg.norm(p1 - p2) for p1 in pts1 for p2 in pts2])

        while len(clusters) > self.n_clusters:
            min_dist = float("inf")
            to_merge = (0, 0)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = distance(clusters[i], clusters[j])
                    if d < min_dist:
                        min_dist = d
                        to_merge = (i, j)

            i, j = to_merge
            clusters[i].extend(clusters[j])
            del clusters[j]

        labels = np.zeros(n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        self.labels_ = labels
        return self

    def predict(self, X):
        return self.labels_
