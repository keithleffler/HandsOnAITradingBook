import pandas as pd
import numpy as np
    
# Sample financial data
data = {'Price': [100, 95, 96, 101, 103, 98, 99, 500, 103, 110]}
df = pd.DataFrame(data)
    
# Calculate IQR
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
    
# Capping outliers using IQR method
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Price_capped'] = np.where(df['Price'] > upper_bound, upper_bound,
np.where(df['Price'] < lower_bound, lower_bound, df['Price']))
print("Data after capping outliers:")
print(df)