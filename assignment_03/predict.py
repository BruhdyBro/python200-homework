import pandas as pd
import joblib
from pathlib import Path

import json

parent = Path(__file__).resolve().parent
outputs_folder = parent / "outputs"
models_folder = parent / "models"


logi_model = joblib.load(models_folder / "weather_classifier.pkl")

with open(models_folder / "weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)


print("Model Metadata:")
print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test Accuracy: {metadata['test_accuracy']}")
print(f"F1 Score: {metadata['test_f1']}")
print()

new_days = pd.DataFrame(
    [
        {"temperature_2m_max": 20, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 15},   # Confidently Good day
        {"temperature_2m_max": 35, "temperature_2m_min": 0, "precipitation_sum": 9.0, "wind_speed_10m_max": 26},    # Confidently Bad day
        {"temperature_2m_max": 24, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 13},   # Borderline Good day
        {"temperature_2m_max": 25, "temperature_2m_min": 12, "precipitation_sum": 2.0, "wind_speed_10m_max": 15},   # Borderline Good day
    ]
)[metadata["features"]]

preds = logi_model.predict(new_days)
probs = logi_model.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Day {i + 1}:")
    print(new_days.iloc[i])
    print(f"Result: {label}  (confidence: {prob:.2f})")
    print()





# -=-=- Task 8 -=-=-
"""
1. On one of the borderline days, it gave a probability of 0.69, which was uncertain but decided to predict a good day.
The other borderline day gave a probability of 0.47, which was uncertain but decided to predict a bad day. These are 
uncertain because of how close they are to the 0.50 tipping point. A day that predicts 0.52 would say that it is a good
day to run, however I would personally change the threshold to have a higher number like 0.65

2. If someone ran predict.py before train_classifier.py, then they would recieve a FileNotFoundError because the model
and metadata had not been created or downloaded. To make it better, I could add a try/except statement to catch the error 
and tell the user to run train_classifier.py before running predict.py

3. One thing that would need to change is where the model is located from, as it currently loads from a specified local
file path. Furthermore, it should automatically download the latest data to predict future days.
""" 