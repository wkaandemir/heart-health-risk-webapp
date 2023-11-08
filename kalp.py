import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

# 1. Data Preprocessing
heart = pd.read_excel("dataset_1.xlsx")

target = heart['target']
features = heart.drop(['target'], axis=1)

# Perform data preprocessing steps as needed (e.g., handling missing values, scaling, encoding)

# 2. Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# 3. Train and evaluate model
def fit_eval_model(model, train_features, y_train, test_features, y_test):
    model.fit(train_features, y_train)
    y_pred = model.predict(test_features)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }

# 4. Initialize the logistic regression model with tuned hyperparameters
lr = LogisticRegression(random_state=1, max_iter=1000)  # You can tune hyperparameters here

# 5. Fit and evaluate the logistic regression model
results = fit_eval_model(lr, X_train, y_train, X_test, y_test)

# 6. Print classifier results
print("Logistic Regression Results:")
print(f"Accuracy: {results['accuracy']:.2f}")
print("Confusion Matrix:")
print(results['confusion_matrix'])
print("Classification Report:")
print(results['classification_report'])

# 7. Visualize feature importance
feature_importance = lr.coef_[0]
sorted_features = sorted(zip(features.columns, feature_importance), key=lambda x: abs(x[1]), reverse=True)

for feature, coef in sorted_features:
    print(f"Feature: {feature}, Coefficient: {coef:.4f}")

# 8. Save the logistic regression model as a serialized object using pickle
with open('kalp.pkl', 'wb') as file:
    pickle.dump(results['model'], file)
