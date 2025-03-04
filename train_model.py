import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data
with open("preprocessed_data.pkl", "rb") as file:
    X_train, X_test, y_train, y_test, scaler = pickle.load(file)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Save the trained model
with open("titanic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Training Completed. Saved as 'titanic_model.pkl'.")
