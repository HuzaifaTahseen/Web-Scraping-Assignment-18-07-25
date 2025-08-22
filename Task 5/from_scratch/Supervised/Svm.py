import numpy as np

class SVMfromScratch:
    # Support Vector Machine using sub-gradient descent

    def __init__(self, lr=0.001, lambda_param=0.01, n_iter=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    self.w -= self.lr * dw
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                    self.w -= self.lr * dw
                    self.b -= self.lr * db
        return self

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
