import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
    
# Generate sensible sample data using make_classification
n_samples = 1000
n_features = 5
n_classes = 3
    
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features * 0.6), n_redundant=n_features - int(n_features * 0.6), n_classes=n_classes, random_state=0)
    
# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Visualize the distribution of the classes in the dataset
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.title("Distribution of Classes in the Dataset")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
    
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'num_leaves': [30, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_freq': [1, 5, 10]
}
    
# Create a LightGBM model
lgb_estimator = lgb.LGBMClassifier(boosting_type='rf', objective='multiclass', num_class=n_classes, metric='multi_logloss')
    
# Perform Grid Search
grid_search = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
    
# Get the best parameters and accuracy from the grid search
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)
    
# Train the model with the best parameters
model = lgb.LGBMClassifier(boosting_type='rf', objective='multiclass', num_class=n_classes, **best_params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after tuning:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
    
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(n_classes), yticklabels=range(n_classes))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
    
# Plot feature importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10, importance_type="gain")
plt.title("Feature Importance")
plt.show()