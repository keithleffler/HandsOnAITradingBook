import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
    
# Generate random sample data with correlations
np.random.seed(42)
size = 100
feature1 = np.random.randn(size)
feature2 = feature1 + np.random.randn(size) * 0.1   # Highly correlated with feature1
feature3 = np.random.randn(size)
target = feature1 * 0.5 + np.random.randn(size) * 0.1   # Highly correlated with target
    
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
    
# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X_reduced, y)
    
# Get feature importance
importances = model.feature_importances_
feature_names = X_reduced.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
# Display feature importance
print(importance_df)
    
# Plot feature importance
importance_df.sort_values(by='Importance', ascending=False).plot(kind='bar', x='Feature', y='Importance')
plt.title('Feature Importance')
plt.show()