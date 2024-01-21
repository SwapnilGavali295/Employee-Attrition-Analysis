# Load required libraries
from sklearn.externals import joblib
import pandas as pd

# Load the PyCaret model
model = joblib.load(r'path\to\your\model\best-model.pkl')

# Function to make predictions
def predict(row):
    data = pd.DataFrame([row])
    prediction = predict_model(model, data)
    return prediction['Label'][0]

# Apply the function to your dataset
dataset['Predictions'] = dataset.apply(predict, axis=1)
