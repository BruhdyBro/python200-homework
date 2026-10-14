import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path

# --- Scikit-learn API ---

# Q1
temp_min = np.array([1,4,8,12,16,20]).reshape(-1,1)
temp_max = np.array([8,10,15,18,23,27])

model = LinearRegression()
model.fit(temp_min, temp_max)

print("Slope:", model.coef_[0])
print("Intercept,", model.intercept_)
print("Low of 6:", model.predict([[6]])[0])
print("Low of 18:", model.predict([[18]])[0])
print()

# Q2
x = np.array([10,20,30,40,50])
print("x before reshape:", x.shape) # before reshape
x = x.reshape(-1,1)
print("x after reshape", x.shape) # after reshape
print()

# Q:    in your own words, why does scikit-learn need X to be 2D.
#
# A:    Scikit-learn needs X to be a 2D array because the first array explains the testing data
#       while the second array explains how many features it has. In this case, X only has 1 feature
#       so it is reshaped to have 1 to define how many features it has.


# --- Linear Regression ---

# Q1
rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")

parent = Path(__file__).resolve().parent
outputs_folder = parent / "outputs"
plt.savefig(outputs_folder / "high_vs_low.png")

# Q:    Add a comment describing what you see. Are there two visible groups? What does that suggest about the is_rainy variable?
#
# A:    There are 2 lines: one for days that are rainy, and one for days that are not rainy. The days that are rainy have a lower
#       temperature than the non-rainy days. This suggest that the is_rainy variable does have an influence on the temperatures.

# Q2

x = temp_min.reshape(-1,1)
y = temp_max

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# Q3
#   training model
new_model = LinearRegression()
new_model.fit(x_train,y_train)

#   making predictions
y_pred = new_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
r2 = new_model.score(x_test, y_test)

print("Slope:", new_model.coef_[0])
print("Intercept:", new_model.intercept_)
print("RMSE:", rmse)
print("R^2:", r2)
print()

# Q:    Add a comment interpreting the slope in plain English. What does it mean for the daily high?
#
# A:    The prediction for the daily high rises by about 0.94 for each degree from the daily low
#       since it is a positive slope.


# Q4

x_full = np.column_stack([temp_min, is_rainy])
x_train, x_test, y_train, y_test  = train_test_split(x_full, y, test_size=0.2, random_state=42)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# Q4

#   training model
model_full = LinearRegression()
model_full.fit(x_train,y_train)

#   making predictions
y_pred = model_full.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
r2 = model_full.score(x_test, y_test)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])
print("R^2:", r2)
print()

# Q:    Add a comment interpreting the is_rainy coefficient. What does it represent in practical terms?
#
# A:    The rainy coefficient means that it'll be rougly 3 degrees cooler on rainy days since it is
#       a negative coefficient.


# Q5
plt.scatter(y_pred, y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="green", linewidth=2, label="Perfect Prediction")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.savefig(outputs_folder / "predicted_vs_actual_warmup.png")

# Q:    What does it mean when a point falls above the diagonal? What about below?
#
# A:    A point above the diagonal means a temperature greater than predicted while data
#       below the line represents temperatures lower than predicted.