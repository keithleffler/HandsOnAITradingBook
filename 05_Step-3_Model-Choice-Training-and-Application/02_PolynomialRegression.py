import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
    
# Generate Sample Data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature: Random values between 0 and 10
y = 2.5 * X**2 + np.random.randn(100, 1) * 10  # Target: Quadratic relationship with noise
    
# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
# Use Polynomial Features and Linear Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
    
model = LinearRegression()
model.fit(X_train_poly, y_train)
    
# Plot the sample chart with the training and test data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot = model.predict(poly.transform(X_plot))
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Fitted curve')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
    
# Retrieve model fit statistics
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
    
# Interpretation of the results
# Lower MSE and higher R^2 indicate a good fit. Watch for overfitting by examining the performance on test data.