import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pywt
    
# Generate Sample Data
np.random.seed(0)
n = 200
X = np.linspace(0, 20, n).reshape(–1, 1)
y = 2.5 * np.sin(X).ravel() + np.random.randn(n) * 0.5  # Target: Non-linear relationship with noise
    
# Apply Wavelet Transform
coeffs = pywt.wavedec(y, 'db1', level=2)
y_wavelet = pywt.waverec(coeffs, 'db1')
    
# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y_wavelet, test_size=0.2, random_state=0)
    
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
# Use SVM Regression
model = SVR(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]
}
    
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
    
# Fit the model
grid_search.fit(X_train_scaled, y_train)
    
# Retrieve the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
    
print(f'Best parameters: {best_params}')
print(f'Best R² score: {best_score}')
    
# Use the best model
best_model = grid_search.best_estimator_
    
# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
    
print(f'Test Mean Squared Error: {mse}')
print(f'Test R² Score: {r2}')