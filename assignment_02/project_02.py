import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests

from pathlib import Path

# ---  TASK 1 ---


parent = Path(__file__).resolve().parent
outputs_folder = parent / "outputs"
# Commented so I don't have issues or have to wait for API Call
# Data originates from API Call so this simulates that

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

# I chose to do Orlando, Florida as the city because it is the Theme Park Capital of the World.



# ---  TASK 2 ---

# Check for missing values 
print("Describing Data")
print(df.describe())
print()

print("Checking for Missing/Null Values")
print(df.isna().sum())
print()

print("Showing Correlations of each Numeric Column to \'temperature_2m_max\'")
print(df.corr(numeric_only=True)['temperature_2m_max'].sort_values())
print()
# The feature that has the strongest correlation to the daily high is
# the daily low (temperature_2m_min) with a 0.888 positive correlation.

print("Creating a scatterplot of Daily High vs. Daily Low")
plt.scatter(df['temperature_2m_min'], df['temperature_2m_max'])
plt.title("Daily High vs. Daily Low in Orlando, Florida for 2025")
plt.ylabel("Day's High Temperature in Degrees Celsius")
plt.xlabel("Day's Low Temperature in Degrees Celsius")
plt.savefig(outputs_folder / "high_vs_low_project.png")
plt.close()
# This graph shows each day's high and low as a point.
# What is noticable is that there is a positive correlation between the two.
# As the Daily High rises in temperature, the Daily Low also rises in temp.



# --- TASK 3 ---

print("Creating is_summer column...")
df['is_summer'] = df['date'].dt.month.apply(lambda x: 1 if (x == 6 or x == 7 or x == 8) else 0)
print()

print(f"Checking if is_summer column has a sum of 92: {df['is_summer'].sum()}")
print(f"Checking if is_summer column has 273 \"0\"s: {(df['is_summer'] == 0).sum()}")
print()
# June has 30 days, July has 31, and August has 31. 30+31+31 = 92 summer days. 
# By extension, 365-92 = 273 non-summer days. 2025 is not a leap year.

# I expect is_summer to help predict the daily high as the summer months
# are usually hotter in temperature and the hottest days of the year



# --- TASK 4 ---

simple_model = LinearRegression()

X_simple = df[["temperature_2m_min"]]
y_simple = df["temperature_2m_max"]

x_train_simple, x_test_simple, y_train_simple, y_test_simple  = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

simple_model.fit(x_train_simple, y_train_simple)

y_pred_simple = simple_model.predict(x_test_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test_simple, y_pred_simple))
r2_train_simple  = simple_model.score(x_train_simple , y_train_simple )
r2_test_simple  = simple_model.score(x_test_simple , y_test_simple )

print("Simple Slope:", simple_model.coef_[0])
print("Simple RMSE:", rmse_simple)
print("Simple Train R^2:", r2_train_simple)
print("Simple Test R^2:", r2_test_simple)
print()
# The RMSE gives an average of how far off the prediction will be. The RMSE of 2.7
# explains that, on average, the temperature will be off by 2.7 degrees Celsius.



# --- TASK 5 ---
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X_full = df[feature_cols].values
y_full = df["temperature_2m_max"].values

x_train_full, x_test_full, y_train_full, y_test_full  = train_test_split(X_full , y_full , test_size=0.2, random_state=42)

full_model = LinearRegression()
full_model.fit(x_train_full, y_train_full) # Train

y_pred_full = full_model.predict(x_test_full) # Predict
rmse_full = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
r2_train_full = full_model.score(x_train_full, y_train_full)
r2_test_full = full_model.score(x_test_full, y_test_full)

for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

print()
print("Full RMSE:", rmse_full)
print("Full Train R^2:", r2_train_full)
print("Full Test R^2:", r2_test_full)
print()

# The features slightly improved on the RMSE, as it lowers it so there is slightly 
# less error on average. Originally it is 2.70 but now is 2.67 after full features.
# However, the train r^2 is higher than the test r^2 which means the prediction fit
# the training data better than the test data by about 0.13. 


# Each coefficient slightly adds or removes from the prediction, based on positive or negative.
# Min Temp: Higher min temp leads to higher predicted high temp
# Rain: Higher rain sum means lower predicted high temp
# Wind Speed: Higher wind speed means lower predicted high temp
# is_summer: If is a summer day, predicts a lower daily high 



# --- TASK 6 ---
plt.scatter(y_pred_full, y_test_full)
plt.plot([y_test_full.min(), y_test_full.max()], [y_test_full.min(), y_test_full.max()], color="green", linewidth=2, label="Perfect Prediction")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.legend()
plt.savefig(outputs_folder / "predicted_vs_actual_high.png")

# The model struggles to predict on the  low end of temperatures before having more accuracy
# towards the higher temperatures.

"""
1.
The size of the total dataset is 365, for the 365 days in the year. However, it is split 80/20 so
the training set is 292 days and the testing set is 73 days.

2.
The RMSE shows an average error of ~2.67 degrees Celsius on each day. The R^2 of 0.70 on the testing
data explains that the model is 70% better aT predicting the high than using the average high temp.

3.
The feature that holds the largest effect is the is_summer, with -0.977. This means that when is_summer
increases by one, there is an approximate decrease of 0.977 degrees celsius. However, this is slightly
misleading because is_summer is a flag so it can only take effect once. Instead, the stronger one is
the daily minimum temperature with +0.860. This instead says that when the minimum increases by one,
there is an approximate increase of 0.860 degrees celsius. The 2 main differences between these 2 is 
the signs between them, and that the minimum temperature can increase more than just 1 (non-binary).

4.
What suprised me the most is that the is_summer flag tended to decrease the average temperature. This
is mainly suprising because you'd expect summer to be the hottest days. However, temperature and rain
do cause lower temperatures, and Florida's rain and Hurricane season tend to align with the summer
months so while it is usually hotter, these effects also pull the temperature down, causing the flag
to also bring the temperature down.
"""