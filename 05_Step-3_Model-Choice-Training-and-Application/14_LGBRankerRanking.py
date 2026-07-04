import numpy as np
from lightgbm import LGBMRanker
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
    
# Step 1: Generate Synthetic Stock Data
np.random.seed(42)
n_stocks = 100
n_groups = 10
    
# Features: historical returns, volatility, and momentum
historical_returns = np.random.rand(n_stocks, 1)
volatility = np.random.rand(n_stocks, 1)
momentum = np.random.rand(n_stocks, 1)
    
# Create a feature matrix
X = np.hstack((historical_returns, volatility, momentum))
    
# Generate synthetic target ranking based on a combination of features
true_rank = historical_returns * 0.5 + volatility * 0.3 + momentum * 0.2
y = np.argsort(np.argsort(true_rank.flatten()))  # Rank stocks
    
# Ensure the labels are within the expected range for LightGBM
y = np.digitize(y, bins=np.linspace(0, n_stocks, 32)) - 1
    
# Assume each group consists of n_stocks // n_groups stocks for ranking
groups = np.repeat(np.arange(n_groups), n_stocks // n_groups)
    
# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=groups)
    
# Define group sizes
train_group_sizes = np.bincount(group_train)
test_group_sizes = np.bincount(group_test)
    
# Remove zero entries in group sizes
train_group_sizes = train_group_sizes[train_group_sizes != 0]
test_group_sizes = test_group_sizes[test_group_sizes != 0]
    
# Step 3: Initialize and Train LGBMRanker
model = LGBMRanker(
    objective="lambdarank",
    metric= "ndc"",
    boosting_type ""gb"t",
    learning_rate=0.05,
    num_leaves=31,
    ndcg_eval_at=[1, 3, 5]
)
    
# Fit the model
model.fit(X_train, y_train, group=train_group_sizes, eval_set=[(X_test, y_test)], eval_group=[test_group_sizes])
    
# Step 4: Predict and Evaluate
y_pred = model.predict(X_test)
    
# Filter out groups with fewer than 2 documents
unique_groups = np.unique(group_test)
valid_groups = [g for g in unique_groups if np.sum(group_test == g) > 1]
    
# Calculate NDCG score for each valid group and average them
ndcg_scores = [ndcg_score([y_test[group_test == g]], [y_pred[group_test == g]], k=5) for g in valid_groups]
mean_ndcg = np.mean(ndcg_scores)
print(f'Mean NDCG Score @5: {mean_ndcg}')
    
# Visualization of results
    
# Scatter Plot of True vs. Predicted Rankings for Multiple Groups
plt.figure(figsize=(12, 8))
for group in valid_groups:
    plt.scatter(y_test[group_test == group], y_pred[group_test == group], alpha=0.6, label=f'Group {group}')
    
plt.xlabel('True Rankings')
plt.ylabel('Predicted Rankings')
plt.title('True vs. Predicted Rankings for Multiple Groups')
plt.legend()
plt.show()