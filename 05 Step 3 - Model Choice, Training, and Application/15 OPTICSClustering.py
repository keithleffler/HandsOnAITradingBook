import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
    
# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
    
# Standardize features (feature normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
    
# Apply OPTICS clustering
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
optics.fit(X_scaled)
    
# Extract labels and core sample indices
labels = optics.labels_
core_samples = np.zeros_like(labels, dtype=bool)
    
# Print numerical performance metrics
# Filter out noise points (–1) for metrics calculation
mask = labels != –1
print("Silhouette Score:", silhouette_score(X_scaled[mask], labels[mask]))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled[mask], labels[mask]))
    
# Plot the clustering results
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
for k, col in zip(unique_labels, colors):
    if k == –1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    
    class_member_mask = (labels == k)
    xy = X_scaled[class_member_mask & core_samples]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    
    xy = X_scaled[class_member_mask & ~core_samples]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
plt.title('OPTICS Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()