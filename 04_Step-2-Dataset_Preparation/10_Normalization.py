import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
    
# Sample data
data = {
    'Feature1': [.4, .2, .1, .9, .6],
    'Feature2': [90, 101, 95, 94, 102],
    'Feature3': [9000, 10100, 9500, 9400, 10200]
}
df = pd.DataFrame(data)
    
# Initialize the MinMaxScaler
scaler = MinMaxScaler()
    
# Fit and transform the data
normalized_data = scaler.fit_transform(df)
    
# Convert the normalized data back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
    
# Display the normalized DataFrame
print("Normalized DataFrame:")
print(normalized_df)