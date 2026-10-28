import requests
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report,
)
import joblib
from pathlib import Path

import json
import sys
import sklearn


parent = Path(__file__).resolve().parent
outputs_folder = parent / "outputs"
models_folder = parent / "models"





# -=-=- Task 1 -=-=-
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 28.54,
    "longitude": -81.38,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)


print("Showing DF info")
df.info()
print()
print("Showing DF shape")
print(df.shape)
print()
print("Showing first 5 rows")
print(df.head(5))
print()

# I chose to do Orlando, Florida as the city again because I love theme parks and roller coasters.





# -=-=- Task 2 -=-=-
#helper function
def is_good_for_running(row):
    if row['temperature_2m_max'] < 7 or row['temperature_2m_max'] > 26:
        return 0
    if row['temperature_2m_min'] < 0:
        return 0
    if row['precipitation_sum'] >= 3:
        return 0
    if row['wind_speed_10m_max'] >= 30:
        return 0
    return 1

print("Checking if good for running...")
print()

df['good_for_running'] = df.apply(is_good_for_running, axis=1)

print("Showing DF info")
df.info()
print()
print("Showing DF shape")
print(df.shape)
print()
print("Showing first 5 rows")
print(df.head(5))
print()


print(f"Printing number of Good Days: {(df['good_for_running'] == 1).sum()}")
print(f"Printing number of Bad Days: {(df['good_for_running'] == 0).sum()}")
print(f"Total amount of days: {((df['good_for_running'] == 0).sum() + (df['good_for_running'] == 1).sum())}")
print()

#print(f"rain amount: {df['precipitation_sum'].sum() / 365}")

# There's about a 1:3 ratio of good days to bad days. This is because Orlando can be very rainy
# and hot all year round. The average rain is just over 3.6, meaning it rains often and hard.





# -=-=- Task 3 -=-=-
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn_values = [1, 3, 5, 7, 9, 11, 15, 21]
for value in knn_values:

    # build pipeline
    curr_knn_pipeline = Pipeline([
        ("scalar", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=value))
    ])

    # Get current Cross Validation Score
    curr_cv_score = cross_val_score(curr_knn_pipeline, X_train, y_train, cv=5)

    print("CV Score for k =", value)
    print(curr_cv_score)
    print(f"Mean: {curr_cv_score.mean():.3f}")
    print(f"Std: {curr_cv_score.std():.3f}")
    print()

# K = 11 chosen, as it's the point where the mean starts to dip back down
knn_pipeline = Pipeline([
    ("scalar", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=11))
])
knn_pipeline.fit(X_train, y_train)
knn_y_pred = knn_pipeline.predict(X_test)
print(f"Scaled Accuracy: {accuracy_score(y_test, knn_y_pred):.3f}")
print(classification_report(y_test, knn_y_pred))
print()





# -=-=- Task 4 -=-=-
logi_model = Pipeline([
    ("scalar", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

logi_model.fit(X_train, y_train)

logi_y_pred = logi_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, logi_y_pred):.3f}")
print(f"Precision: {precision_score(y_test, logi_y_pred):.3f}")
print(f"Recall: {recall_score(y_test, logi_y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, logi_y_pred):.3f}")
print()

logreg = logi_model.named_steps["logreg"]
for name, coef in zip(numeric, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")
print()

# All coefficients are negative, meaning that their individual unit increases lower the
# possibility for a good day of running. This is probably due to the high amounts of rain
# and overall high temperatures in Florida. Interestingly enough, increasing the minimum
# temperature also lowers the probability even though we defined having a min temp > 0 as
# a good day. This is due to the fact that a higher low temperature usually means a higher
# high temperature, and therefore is much more restrictive, especially in already hot places.





# -=-=- Task 5 -=-=-
"""
1. The Logistic Regression model performed slightly better by about 0.013 or ~1.3% on the
accuracy.


2. For a public running app, I would prioritize Precision as to me, the most important part
would be having a reliable app. One of the worst feelings is being rained on while you're 
trying to do stuff. Spending your day on a leisurely jog just to have rain start pouring
would ruin user experiences with the app and cause them to alienate. Precision focuses on
checking how often it is truly correct versus false positives. Focusing on a high precision
would mean fewer false positives and more truly correct days. I don't believe a false negative
(i.e predict bad but truly good) would ruin a user's experience as much as a false positive.

3. I would use the Logistic Regression model as it is slightly more accurate, and less load
bearing necessary to predict. For example, the KNN model would have to store the points and
data for each date and how they cluster, while the Logistic Regression only has to store the
coefficients and scalars. The KNN slows down with larger sets so trying to predict over larger
dates would cause it to be slower, even though the realistic set of use wouldn't go that far.
"""





# -=-=- Task 6 -=-=-
joblib.dump(logi_model, models_folder / "weather_classifier.pkl")
print("Model saved")

FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "features": FEATURES,
    "label": "good_for_running",
    "city": "Orlando, FL",
    "test_accuracy": 0.986,
    "test_f1": 0.974,
    "label_rule": {
        "temperature_2m_max": "7 to 26 C",
        "temperature_2m_min": ">= 0 C",
        "precipitation_sum": "< 3 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

with open(models_folder / "weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved")