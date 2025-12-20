import pandas as pd
import numpy as np
    
# Set random seed for reproducibility
np.random.seed(42)
    
# Generate synthetic financial data
n_samples = 1000
data = {
    'Date': pd.date_range(start='1/1/2023', periods=n_samples, freq='D'),
    'Open': np.random.uniform(100, 500, size=n_samples),
    'High': np.random.uniform(100, 500, size=n_samples),
    'Low': np.random.uniform(100, 500, size=n_samples),
    'Close': np.random.uniform(100, 500, size=n_samples),
    'Volume': np.random.randint(1000, 100000, size=n_samples)
}
    
# Create a DataFrame
df = pd.DataFrame(data)
    
# Calculate moving averages
df['MA5'] = df['Close'].rolling(window=5).mean()      # 5-day moving average
df['MA10'] = df['Close'].rolling(window=10).mean()    # 10-day moving average