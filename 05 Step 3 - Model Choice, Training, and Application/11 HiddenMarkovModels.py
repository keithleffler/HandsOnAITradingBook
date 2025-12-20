import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
    
# Generate Sample Data
np.random.seed(0)
n = 200
X = np.linspace(0, 20, n)
state_1 = 2.5 * np.sin(X[:n//2]) + np.random.randn(n//2) * 0.5
state_2 = –1.5 * np.sin(X[n//2:]) + 2 + np.random.randn(n//2) * 0.5
y = np.concatenate([state_1, state_2])
    
# Fit Hidden Markov Model
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
model.fit(y.reshape(–1, 1))
    
# Predict hidden states
hidden_states = model.predict(y.reshape(–1, 1))
    
# Plot the sample chart with the hidden states
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Observed', color='blue')
plt.plot(X[hidden_states == 0], y[hidden_states == 0], 'ro', label='State 1')
plt.plot(X[hidden_states == 1], y[hidden_states == 1], 'go', label='State 2')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Hidden Markov Model')
plt.legend()
plt.show()
    
# Retrieve model fit statistics
print('Transition Matrix')
print(model.transmat_)
print('Means and Variances of Each State')
print(model.means_)
print(model.covars_)
    
# Interpretation of the results
# The transition matrix shows the probabilities of moving from one state to another.
# The means and variances provide insights into the characteristics of each hidden state.