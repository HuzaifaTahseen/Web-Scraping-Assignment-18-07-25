import numpy as np

class LinearRegressionGD:
    # Linear Regression trained with batch Gradient Descent.
 
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        self.w = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            y_pred = X @ self.w
            grad = (2 / X.shape[0]) * X.T @ (y_pred - y)
            self.w -= self.lr * grad
        return self

    def predict(self, X):
        X = np.asarray(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ self.w

class LinearRegressionNormal:
    # Linear Regression using the closed-form normal equation.

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        self.w = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        return self

    def predict(self, X):
        X = np.asarray(X)
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        return X_ @ self.w
