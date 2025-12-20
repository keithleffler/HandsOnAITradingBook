import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
    
# Load dataset
data = load_wine()
X, y = data.data, data.target
    
# Initialize the model
model = RandomForestClassifier(random_state=42)
    
# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())
    
# Perform a typical train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Train the model
model.fit(X_train, y_train)
    
# Test the model
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Train-test split accuracy:", test_accuracy)
    
# Compare the results
print("\nComparison of Results:")
print(f"Mean cross-validation score: {cv_scores.mean():.4f}")
print(f"Train-test split accuracy: {test_accuracy:.4f}")