import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, LSTM
import tensorflow as tf
    
# Enable logging
tf.get_logger().setLevel('INFO')
    
# Generate Sample Data
np.random.seed(0)
n = 1000
X = np.linspace(0, 1000, n)
y = np.sin(X/100)  # Target: Long sinusoidal wave
    
# Plot the sine wave
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='sin(x)')
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()
    
# Reshape data for CNN
X = X.reshape(–1, 1)
y = y.reshape(–1, 1)
    
# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
# Reshape for CNN input
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, 1))
    
# Use CNN-LSTM for Regression
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, 1)))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(Dropout(0.5))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
# Fit the model with logging
history = model.fit(X_train_scaled, y_train, epochs=300, batch_size=32, verbose=1, validation_split=0.2)
    
# Plot the sample chart with the training and test data and the fitted model
y_pred = model.predict(X_test_scaled)
    
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.scatter(X_test, y_pred, color='red', label='Fitted model')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('CNN-LSTM Neural Network Regression')
plt.legend()
plt.show()
    
# Retrieve model fit statistics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
    
# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
    
# Interpretation of the results
# CNN-LSTM regression results provide insights into the model's ability to capture complex patterns.
# Lower MSE and higher R^2 indicate a good fit. Adjust the network architecture and parameters to optimize performance.