import pandas as pd

df = pd.read_csv("data/water_potability.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
# Handle missing values
df['ph'] = df['ph'].fillna(df['ph'].median())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

print("Missing values after preprocessing:")
print(df.isnull().sum())

# Scaling the features
from sklearn.preprocessing import StandardScaler

X = df.drop('Potability', axis=1)
y = df['Potability']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling completed!")
print(X_scaled[:5])
