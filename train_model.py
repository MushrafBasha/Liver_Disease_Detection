import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("indian_liver_patient.csv")  # Make sure dataset is here

# Convert gender
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Fill missing values
data = data.fillna(data.mean())

X = data.drop('Dataset', axis=1)
y = data['Dataset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")
