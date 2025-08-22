import numpy as np
from collections import Counter
from decision_tree import DecisionTreeScratch

class RandomForestScratch:
    # Random Forest Classifier

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)

            if self.max_features is None:
                self.max_features = X.shape[1]
            feature_idxs = np.random.choice(X.shape[1], self.max_features, replace=False)

            tree.fit(X_sample[:, feature_idxs], y_sample)
            tree.feature_indices = feature_idxs
            self.trees.append(tree)
        return self

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X[:, tree.feature_indices]) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._most_common_label(preds) for preds in tree_preds])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
