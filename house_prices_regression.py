# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
df_train = pd.read_csv("/content/train.csv")
df_test = pd.read_csv("/content/test.csv")

!pip install scikit-learn

from sklearn.preprocessing import LabelEncoder
import pandas as pd

unseen_categories = set(df_test[categorical_cols].values.ravel()) - set(df_train[categorical_cols].values.ravel())
print(f"Unseen categories: {unseen_categories}")

print(unseen_categories)

print(df_test.columns)

print(df_test.isnull().sum())

# Feature engineering (replace with more advanced techniques as needed)
# Handle categorical features
categorical_cols = [col for col in df_train.columns if df_train[col].dtype == 'object']
le = LabelEncoder()
le.fit(df_train[categorical_cols].values.ravel())  # Fit on all categories (including unseen)

def encode_label(col):
  # Encode with le, unseen labels get -1
  return le.transform(col)

df_train[categorical_cols] = df_train[categorical_cols].apply(encode_label)
df_test[categorical_cols] = df_test[categorical_cols].apply(encode_label)

# Handle missing values (replace with more sophisticated methods)
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_train.mean(), inplace=True)

# Separate features and target variable
X_train = df_train.drop('SalePrice', axis=1)
y_train = df_train['SalePrice']

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Random Forest Regression
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)  # Colon added after function definition
y_pred_rf = model_rf.predict(X_val)  # Colon added after function definition
rf_mse = mean_squared_error(y_val, y_pred_rf)
print(f"Random Forest MSE: {rf_mse}")

# Separate features and target variable (drop "Id" consistently)
X_train = df_train.drop(['Id', 'SalePrice'], axis=1)
y_train = df_train['SalePrice']

# Train-Test Split (already done, but ensuring consistency)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model_rf.fit(X_train, y_train)

# Scatter plot of actual vs predicted values
plt.scatter(y_val, y_pred_rf)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Random Forest Regression - Actual vs Predicted")
plt.show()

# Prediction on Test Data (assuming model_rf is already trained)
X_test = df_test.drop('Id', axis=1)  # Drop ID as it's not a feature
y_pred_test_rf = model_rf.predict(X_test)

# Create the submission DataFrame
submission_df = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': y_pred_test_rf})

# Save predictions for submission
submission_df.to_csv('submission.csv', index=False)
