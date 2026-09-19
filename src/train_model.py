import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data/placement_predict_50k Dataset_final.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

# Target column
target = "PlacementStatus"

# Remove columns that should not be used for prediction
X = data.drop(columns=[target, "StudentID", "Salary Package"])

y = data[target]

# Convert text columns into numbers
label_encoders = {}

for column in X.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column].astype(str))
    label_encoders[column] = encoder

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/placement_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Model saved successfully!")