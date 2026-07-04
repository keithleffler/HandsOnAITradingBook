import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
    
# Generate random sample data with correlations
np.random.seed(42)
size = 100
feature1 = np.random.randn(size)
feature2 = feature1 + np.random.randn(size) * 0.1   # Highly correlated with feature1
feature3 = np.random.randn(size)
target = feature1 * 0.5 + np.random.randn(size) * 0.1   # Highly correlated with targetfeature1
    
data = {
    'Feature1': feature1,
    'Feature2': feature2,
    'Feature3': feature3,
    'Target': target
}
df = pd.DataFrame(data)
X = df.drop('Target', axis=1)
y = df['Target']
    
# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_reduced = X.drop(columns=to_drop)
    
# Apply Recursive Feature Elimination
model = RandomForestRegressor()
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(X_reduced, y)
    
# Get selected features
selected_features = X_reduced.columns[rfe.support_]
print(f'Selected Features: {selected_features}')
    
# Plot selected features
importance_df = pd.DataFrame({'Feature': X_reduced.columns, 'Importance': rfe.support_.astype(int)})
importance_df.plot(kind='bar', x='Feature', y='Importance')
plt.title('Auto-Identification of Features')
    
# Adjust the bottom margin to add more space for x-axis labels
plt.subplots_adjust(bottom=0.2)
    
plt.show()