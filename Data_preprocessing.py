import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("Dataset/Titanic_dataset.csv")

# Drop unnecessary columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)  # Fill missing Age with median
df["Fare"].fillna(df["Fare"].median(), inplace=True)  # Fill missing Fare with median
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # Fill missing Embarked with mode

# Convert categorical features
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Ensure no missing values remain
print("Missing values after preprocessing:")
print(df.isnull().sum())  # Should print all zeros

# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data
with open("preprocessed_data.pkl", "wb") as file:
    pickle.dump((X_train, X_test, y_train, y_test, scaler), file)

print("✅ Data Preprocessing Completed. No Missing Values!")
