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
    
# Convert Volume to float
data['Volume'] = data['Volume'].astype(float)
    
# Introduce some missing values for demonstration
data['Volume'][np.random.choice(n_samples, size=50, replace=False)] = np.nan
data['Close'][np.random.choice(n_samples, size=20, replace=False)] = np.nan
    
# Create a DataFrame
df = pd.DataFrame(data)
    
# Check for missing values
missing_data = df.isnull().sum()
print(missing_data)