import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


def draw_ellipse(ax, mean, cov, color="k", alpha=0.3, n_std=2.0):

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Angle of ellipse 
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # Width and height
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor=color, alpha=alpha, linewidth=2)
    ax.add_patch(ell)


def cov_to_matrix(cov, cov_type, i, n_features):
    """Return a full 2x2 covariance matrix for component i depending on cov_type."""
    if cov_type == "full":
        return cov[i]
    if cov_type == "tied":
        return cov
    if cov_type == "diag":
        return np.diag(cov[i])
    if cov_type == "spherical":
        return np.eye(n_features) * cov[i]
    raise ValueError(f"Unknown covariance_type: {cov_type}")


def main():
    # Create synthetic data 
    X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=1.05, random_state=42)

    n_components = 3
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    # Diagnostics
    print("Converged:", gmm.converged_)
    print("N iterations:", gmm.n_iter_)
    print("AIC:", gmm.aic(X))
    print("BIC:", gmm.bic(X))

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap="viridis", edgecolor="k", alpha=0.6)
    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c="red", s=100, marker="x", label="Component Means")

    # Draw an ellipse per component
    n_features = X.shape[1]
    for i, mean in enumerate(gmm.means_):
        cov = cov_to_matrix(gmm.covariances_, gmm.covariance_type, i, n_features)
        draw_ellipse(ax, mean, cov, color="red", alpha=0.15, n_std=2.0)

    ax.set_title(f"GMM clustering (n_components={n_components})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.show()

    print("Example responsibilities (first 6 samples):\n", probs[:6].round(3))


if __name__ == "__main__":
    main()
