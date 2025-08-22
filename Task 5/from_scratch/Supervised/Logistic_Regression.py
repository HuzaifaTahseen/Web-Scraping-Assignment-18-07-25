import numpy as np

class LogisticRegressionGD:
    # Logistic Regression trained with batch Gradient Descent.


    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            z = X @ self.w
            h = self._sigmoid(z)
            grad = (1 / X.shape[0]) * X.T @ (h - y)
            self.w -= self.lr * grad
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(X @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
