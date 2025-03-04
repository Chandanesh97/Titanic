import pickle
import numpy as np

# Load the trained model and scaler
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("preprocessed_data.pkl", "rb") as file:
    _, _, _, _, scaler = pickle.load(file)

# Example passenger data: Pclass=3, Sex=Male, Age=22, Fare=7.25, SibSp=0, Parch=0, Embarked_S=1, Embarked_Q=0
sample_data = np.array([[3, 0, 22, 7.25, 0, 0, 1, 0]])  
sample_data = scaler.transform(sample_data)
prediction = model.predict(sample_data)

print("Prediction (0 = Did not survive, 1 = Survived):", prediction[0])