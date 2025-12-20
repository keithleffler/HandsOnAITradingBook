import pandas as pd
import numpy as np
    
# Sample financial data
data = {'Price': [100, 95, 96, 101, 103, 98, 99, 500, 103, 110]}
df = pd.DataFrame(data)
    
# Calculate Z-scores
df['Z_score'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
# Identify outliers
outliers = df[np.abs(df['Z_score']) > 2]
print("Outliers using Z-score method:")
print(outliers)