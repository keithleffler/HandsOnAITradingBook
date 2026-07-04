import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
    
# Generate Sample Data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature: Random values between 0 and 10
y = 2.5 * np.sin(X).ravel() + np.random.randn(100) * 0.5  # Target: Non-linear relationship with noise
    
# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
# Use Decision Tree Regression
model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)
    
# Plot the sample chart with the training and test data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
X_plot = np.linspace(0, 10, 100).reshape(–1, 1)
y_plot = model.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Fitted model')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()
    
# Display the generated trees
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True)
plt.title('Decision Tree Structure')
plt.show()
    
# Retrieve model fit statistics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
    
# Interpretation of the results
# Decision tree regression results include the structure of the tree, showing splits and leaf values.
# Lower MSE and higher R^2 indicate a good fit. Pruning can help improve model generalization.