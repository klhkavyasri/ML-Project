import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.inspection import permutation_importance
df = pd.read_csv("data/DT_Placement.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print(df)
# Convert Yes/No to 1/0
mapping = {"No": 0, "Yes": 1}

df["Backlogs"] = df["Backlogs"].map(mapping)
df["Placed"] = df["Placed"].map(mapping)

# Define features and target
X = df[["CGPA", "Internships", "Backlogs", "Aptitude"]]
y = df["Placed"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

# Prediction on validation data
dt_pred = dt_model.predict(X_val)

# Calculate accuracy
dt_accuracy = accuracy_score(y_val, dt_pred)

print("\nDecision Tree Results")
print("--------------------")
print("Accuracy:", dt_accuracy)
# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    oob_score=True
)

rf_model.fit(X_train, y_train)

# Prediction on validation data
rf_pred = rf_model.predict(X_val)

# Calculate accuracy
rf_accuracy = accuracy_score(y_val, rf_pred)

print("\nRandom Forest Results")
print("--------------------")
print("Accuracy:", rf_accuracy)
print("OOB Score:", rf_model.oob_score_)
print("OOB Error:", 1 - rf_model.oob_score_)
# Bagging with Decision Trees
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)

bagging_model.fit(X_train, y_train)

bagging_pred = bagging_model.predict(X_val)

bagging_accuracy = accuracy_score(y_val, bagging_pred)

print("\nBagging Results")
print("--------------------")
print("Accuracy:", bagging_accuracy)
# Random Forest with Feature Subsampling
feature_subsampling_model = RandomForestClassifier(
    n_estimators=100,
    max_features=2,
    random_state=42,
    oob_score=True
)

feature_subsampling_model.fit(X_train, y_train)

feature_pred = feature_subsampling_model.predict(X_val)

feature_accuracy = accuracy_score(y_val, feature_pred)

print("\nFeature Subsampling Results")
print("--------------------")
print("Accuracy:", feature_accuracy)
print("OOB Score:", feature_subsampling_model.oob_score_)