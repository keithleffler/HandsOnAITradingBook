import pandas as pd
    
# Sample financial data
data = {'Price': [100, 95, 96, 101, 103, 98, 99, 500, 103, 110]}
df = pd.DataFrame(data)
    
# Calculate IQR
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
    
# Identify outliers
outliers = df[(df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR))]
print("Outliers using IQR method:")
print(outliers)