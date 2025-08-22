import numpy as np

class GaussianMixtureScratch:
    # Gaussian Mixture Model (GMM) using the EM algorithm.

    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize means randomly
        rng = np.random.default_rng()
        self.means_ = X[rng.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X, rowvar=False)] * self.n_components)
        self.weights_ = np.ones(self.n_components) / self.n_components

        log_likelihood_old = 0

        for _ in range(self.max_iter):
            # E-step: responsibilities
            resp = self._estimate_responsibilities(X)

            # M-step: update params
            Nk = resp.sum(axis=0)
            self.weights_ = Nk / n_samples
            self.means_ = np.dot(resp.T, X) / Nk[:, np.newaxis]

            self.covariances_ = []
            for k in range(self.n_components):
                diff = X - self.means_[k]
                cov = np.dot(resp[:, k] * diff.T, diff) / Nk[k]
                self.covariances_.append(cov)
            self.covariances_ = np.array(self.covariances_)

            # Check convergence
            log_likelihood_new = np.sum(np.log(resp.sum(axis=1)))
            if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

        return self

    def _estimate_responsibilities(self, X):
        likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihood[:, k] = self.weights_[k] * self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
        resp = likelihood / likelihood.sum(axis=1, keepdims=True)
        return resp

    def _multivariate_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
        diff = X - mean
        return norm_const * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))

    def predict(self, X):
        resp = self._estimate_responsibilities(X)
        return np.argmax(resp, axis=1)
