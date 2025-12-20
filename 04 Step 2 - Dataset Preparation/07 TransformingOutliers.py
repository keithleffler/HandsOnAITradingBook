import pandas as pd
import numpy as np
    
# Sample financial data
data = {'Price': [100, 95, 96, 101, 103, 98, 99, 500, 103, 110]}
df = pd.DataFrame(data)
    
# Log transformation to reduce the impact of outliers
df['Price_log'] = np.log(df['Price'])
print("Data after log transformation:")
print(df)