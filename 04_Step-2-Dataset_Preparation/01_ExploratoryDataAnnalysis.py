import pandas as pd
import numpy as np
import sweetviz as sv
    
np.random.seed(42)
    
# Generate sample data
data = {
    'Age': np.random.randint(8, 90, size=100),
    'Income': np.random.randint(10000, 500000, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'City': np.random.choice(['New York', 'Singapore', 'Paris', 'Rome', 'Tokyo'], size=100)
}
    
# Create a DataFrame
df = pd.DataFrame(data)
    
# Generate the general report
general_report = sv.analyze(df)
general_report.show_html('sweetviz_report.html')
    
# Generate report by gender
gender_report = sv.compare_intra(df, df['Gender'] == 'Male', ['Male', 'Female'])
gender_report.show_html('sweetviz_gender_comparison.html')