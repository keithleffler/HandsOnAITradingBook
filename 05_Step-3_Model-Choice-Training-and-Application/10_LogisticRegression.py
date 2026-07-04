import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    
# Generate Sample Data
np.random.seed(0)
X = np.random.rand(100, 10)  # Features: Random values with 10 features
true_coeffs = np.array([2.5, –1.5, 0, 0, 3, 0, 0, 0, 1, 0])
logits = X @ true_coeffs + np.random.randn(100) * 2
y = (logits > np.median(logits)).astype(int)  # Target: Binary outcome based on logits
    
# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
# Use Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
    
# Plot the sample chart with the training and test data and the fitted model
# For binary classification, we plot the ROC curve
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
    
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
    
# Retrieve model fit statistics
print("Classification Report")
print(classification_report(y_test, model.predict(X_test_scaled)))
print("Confusion Matrix")
print(confusion_matrix(y_test, model.predict(X_test_scaled)))
    
# Interpretation of the results
# The classification report and confusion matrix provide detailed metrics on the model's performance.
# A higher AUC indicates better model performance. Adjust regularization parameters to control overfitting.