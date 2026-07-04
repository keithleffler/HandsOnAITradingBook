import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
    
# Generate random sample data with correlations
np.random.seed(42)
size = 100
feature1 = np.random.randn(size)
feature2 = feature1 + np.random.randn(size) * 0.1   # Highly correlated with feature1
feature3 = np.random.randn(size)
target = feature1 * 0.5 + np.random.randn(size) * 0.1   # Highly correlated with feature1
    
data = {
    'Feature1': feature1,
    'Feature2': feature2,
    'Feature3': feature3,
    'Target': target
}
df = pd.DataFrame(data)
X = df.drop('Target', axis=1)
    
# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
    
# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_standardized)
    
# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    
# Display the explained variance ratio
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    
# Plot the principal components
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis')
plt.show()