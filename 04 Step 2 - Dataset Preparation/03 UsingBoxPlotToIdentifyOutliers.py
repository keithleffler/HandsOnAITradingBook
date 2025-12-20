import pandas as pd
import matplotlib.pyplot as plt
# Sample financial data
data = {'Price': [100, 95, 96, 101, 103, 98, 99, 500, 103, 110]}
df = pd.DataFrame(data)
    
# Box plot
plt.figure(figsize=(10, 6))
plt.boxplot(df['Price'])
plt.title('Box Plot for Price')
plt.ylabel('Price')
plt.show()