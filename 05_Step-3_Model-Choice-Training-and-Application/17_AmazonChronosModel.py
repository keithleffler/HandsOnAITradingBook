import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
    
# Load the pre-trained Chronos model
model = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)
    
# Generate a representative time series (e.g., sine wave with noise)
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', periods=200, freq='D')
series = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.5, 200)
data = {'date': dates, 'close': series}
df = pd.DataFrame(data)
    
# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[['close']])
    
# Convert normalized data to a PyTorch tensor
tensor_data = torch.tensor(normalized_data)
    
# Forecast future prices for the next 30 days
prediction_length = 30
forecast = model.predict(tensor_data, prediction_length)
    
# Convert forecast back to original scale
forecast = scaler.inverse_transform(forecast[0].numpy())
    
# Combine original and forecasted data for visualization
forecast_dates = pd.date_range(start=dates[–1] + pd.Timedelta(days=1), periods=prediction_length, freq='D')
    
forecast_values = np.median(forecast, axis=0)
    
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast_values})
combined_df = pd.concat([df[['date', 'close']], forecast_df], axis=0)
    
# Plot the original and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='Original')
plt.plot(forecast_df['date'], forecast_df['forecast'], label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Original vs Forecasted Closing Prices')
plt.legend()
plt.show()
    
# Calculate and print performance metrics
# For simplicity, compare last 30 days of original data with forecast
actual = df['close'].values[-prediction_length:]
predicted = forecast.flatten()[:len(actual)]
    
mse = mean_squared_error(actual, predicted)
mae = mean_absolute_error(actual, predicted)
    
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")