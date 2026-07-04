import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
from hurst import compute_Hc
    
# Generate synthetic cointegrated time series data
np.random.seed(42)
n = 1000  # Number of data points
time = np.arange(n)
    
# Generate a common stochastic trend
trend = np.cumsum(np.random.randn(n))
    
# Create two cointegrated series with added noise
asset1 = trend + 0.3 * np.random.randn(n)
asset2 = trend + 0.1 * np.random.randn(n)
    
# Create a DataFrame
data = pd.DataFrame({'asset1': asset1, 'asset2': asset2})
    
# Function to perform ADF test
def adf_test(series, name):
    result = adfuller(series)
    print(f'ADF Statistic for {name}: {result[0]}')
    print(f'p-value for {name}: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    print('\n')
    
# Test each series for stationarity
print("ADF Test for asset1:")
adf_test(data['asset1'], 'asset1')
print("ADF Test for asset2:")
adf_test(data['asset2'], 'asset2')
    
# Perform the Engle-Granger cointegration test
score, pvalue, _ = coint(data['asset1'], data['asset2'])
    
print(f'Engle-Granger Cointegration Test score: {score}')
print(f'Engle-Granger Cointegration Test p-value: {pvalue}\n')
    
# Calculate the spread
data['spread'] = data['asset1'] - data['asset2']
    
# Step 4: Test the spread for stationarity
print("ADF Test for spread:")
adf_test(data['spread'], 'spread')
    
# Remove zero and negative values in the spread for Hurst calculation
spread_non_zero = data['spread'][data['spread'] > 0]
if len(spread_non_zero) < len(data['spread']):
    print("Warning: Non-positive values in spread were removed for Hurst calculation.")
    
# Ensure spread_non_zero is not empty and does not contain invalid values
if len(spread_non_zero) > 0 and (spread_non_zero <= 0).sum() == 0:
    # Calculate the Hurst exponent of the spread
    H, c, data_hurst = compute_Hc(spread_non_zero, kind='price', simplified=True)
    print(f'Hurst Exponent for spread: {H}')
    
    # Visualize the series and their spread
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data['asset1'], label='Asset 1')
    plt.plot(data['asset2'], label='Asset 2')
    plt.legend()
    plt.title('Cointegrated Time Series')
    
    plt.subplot(2, 1, 2)
    plt.plot(data['spread'], label='Spread (Asset 1 - Asset 2)')
    plt.legend()
    plt.title('Spread (Should be Stationary)')
    plt.tight_layout()
    plt.show()
else:
    print("Error: Spread contains invalid values or is empty after removing non-positive values.")