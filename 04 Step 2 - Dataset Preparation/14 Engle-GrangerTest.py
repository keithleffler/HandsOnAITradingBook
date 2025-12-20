import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
    
# Generate synthetic time series data
np.random.seed(42)
n = 100
time = np.arange(n)
# Simulate two non-stationary series (random walks)
asset1 = np.cumsum(np.random.randn(n)) + 41
asset2 = asset1 + np.random.randn(n)
    
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
    
# Step 1: Test each series for stationarity
print("ADF Test for asset1:")
adf_test(data['asset1'], 'asset1')
print("ADF Test for asset2:")
adf_test(data['asset2'], 'asset2')
    
# Step 2: Perform the Engle-Granger cointegration test
score, pvalue, _ = coint(data['asset1'], data['asset2'])
    
print(f'Engle-Granger Cointegration Test score: {score}')
print(f'Engle-Granger Cointegration Test p-value: {pvalue}\n')
    
# Step 3: Visualize the series and their spread
data['spread'] = data['asset1'] - data['asset2']
    
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(data['asset1'], label='Asset 1')
plt.plot(data['asset2'], label='Asset 2')
plt.legend()
plt.title('Non-Stationary Time Series')
    
plt.subplot(2, 1, 2)
plt.plot(data['spread'], label='Spread (Asset 1 - Asset 2)')
plt.legend()
plt.title('Spread (Should be Stationary)')
plt.tight_layout()
plt.show()
    
# Step 4: Test the spread for stationarity
print("ADF Test for spread:")
adf_test(data['spread'], 'spread')