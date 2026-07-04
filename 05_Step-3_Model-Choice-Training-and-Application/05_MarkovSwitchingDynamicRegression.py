import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    
# Generate Sample Data with Different Volatilities
np.random.seed(0)
n = 200
X = np.linspace(0, 20, n)
regime_1 = 2.5 * X[:n//2] + np.random.randn(n//2) * 5  # Higher volatility in regime 1
regime_2 = –1.5 * X[n//2:] + 30 + np.random.randn(n//2) * 1  # Lower volatility in regime 2
y = np.concatenate([regime_1, regime_2])
    
# Create a DataFrame
data = pd.DataFrame({'X': X, 'y': y})
    
# Fit Markov Switching Dynamic Regression
model = MarkovRegression(data['y'], k_regimes=2, trend='c', switching_variance=True)
result = model.fit()
    
# Plot the observed data and fitted regimes
plt.figure(figsize=(12, 8))
    
# Plot observed data
plt.plot(data['X'], data['y'], label='Observed', color='blue')
    
# Extract smoothed probabilities
smoothed_probs = result.smoothed_marginal_probabilities
    
# Plot the observed data and regime probabilities
for t in range(len(smoothed_probs)):
    if smoothed_probs.iloc[t, 0] > 0.5:
        plt.plot(data['X'].iloc[t], data['y'].iloc[t], 'ro', alpha=0.5)
    else:
        plt.plot(data['X'].iloc[t], data['y'].iloc[t], 'go', alpha=0.5)
    
# Highlight regime changes
regime_changes = np.argmax(smoothed_probs.values, axis=1)
for i in range(1, len(regime_changes)):
    if regime_changes[i] != regime_changes[i - 1]:
        plt.axvline(x=data['X'].iloc[i], color='gray', linestyle='--', linewidth=1)
    
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Markov Switching Dynamic Regression with Two Regimes')
plt.legend(['Observed', 'Regime 1', 'Regime 2'])
plt.show()
    
# Plot the regime probabilities
plt.figure(figsize=(12, 6))
plt.plot(data['X'], smoothed_probs[0], label='Regime 1 Probability', color='red')
plt.plot(data['X'], smoothed_probs[1], label='Regime 2 Probability', color='green')
plt.xlabel('Feature (X)')
plt.ylabel('Probability')
plt.title('Smoothed Regime Probabilities')
plt.legend()
plt.show()
    
# Retrieve model fit statistics
print(result.summary())
    
# Interpretation of the results
# The summary provides detailed statistics on the fitted model, including transition probabilities and regime-specific parameters.
# These statistics help understand the regime changes and their impact on the dependent variable.