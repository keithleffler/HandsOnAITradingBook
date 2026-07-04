import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
    
# Generate Sample Data
np.random.seed(0)
X = np.random.rand(100, 10)  # Features: Random values with 10 features
true_coeffs = np.array([22.5, –1.5, 0, 100, 3, 0, 45, 0, 1, 0])
y = X @ true_coeffs + np.random.randn(100) * 2  # Target: Linear combination with noise
    
# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
# Use Lasso Regression
model = Lasso(alpha=0.01)  # Adjusted alpha for better fit
model.fit(X_train_scaled, y_train)
    
# Plot the sample chart with the training and test data and the fitted model
plt.figure(figsize=(10, 6))
y_pred_train = model.predict(X_train_scaled)
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Training data')
y_pred_test = model.predict(X_test_scaled)
plt.scatter(y_test, y_pred_test, color='green', alpha=0.5, label='Test data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal fit')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Lasso Regression: Predicted vs. Actual')
plt.legend()
plt.show()
    
# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.plot(model.coef_, marker='o', linestyle='none', label='Lasso coefficients')
plt.xlabel('Feature index')
plt.ylabel('Coefficient value')
plt.title('Lasso Regression Coefficients')
plt.legend()
plt.show()
    
# Retrieve model fit statistics
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
    
# Interpretation of the results
# Lasso regression coefficients help identify important features. Lower MSE and higher R^2 indicate a good fit.
# The regularization parameter (alpha) can be tweaked to adjust the amount of shrinkage.