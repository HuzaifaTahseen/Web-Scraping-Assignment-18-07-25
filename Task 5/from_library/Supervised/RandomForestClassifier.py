import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Generate synthetic dataset
X, y = make_classification(
    n_samples=300, n_features=2, n_classes=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Accuracy
print("Accuracy:", accuracy_score(y, y_pred))

# Visualization - Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(("lightblue", "lightgreen")))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=ListedColormap(("blue", "green")))
plt.title("Random Forest Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
