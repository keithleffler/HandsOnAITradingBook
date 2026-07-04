import pandas as pd
from sklearn.preprocessing import StandardScaler
    
# Sample data
data = {
    'Feature1': [.4, .2, .1, .9, .6],
    'Feature2': [90, 101, 95, 94, 102],
    'Feature3': [9000, 10100, 9500, 9400, 10200]
}
df = pd.DataFrame(data)
    
# Initialize the StandardScaler
scaler = StandardScaler()
    
# Fit and transform the data
standardized_data = scaler.fit_transform(df)
    
# Convert the standardized data back to a DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=df.columns)
    
# Display the standardized DataFrame
print("Standardized DataFrame:")
print(standardized_df)