from pathlib import Path
from weather_model import Prediction, WeatherClassifier

# -=-=- Task 5 -=-=-

def main():

    assignment_04_dir = Path(__file__).resolve().parent
    models_folder = assignment_04_dir / "models"
    weather_model = models_folder / "weather_classifier.pkl"

    my_model = WeatherClassifier(weather_model)

    days = [
            {"temperature_2m_max": 20, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 15},   # Confidently Good day
            {"temperature_2m_max": 35, "temperature_2m_min": 0, "precipitation_sum": 9.0, "wind_speed_10m_max": 26},    # Confidently Bad day
            {"temperature_2m_max": 24, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 13},   # Borderline Good day
            {"temperature_2m_max": 25, "temperature_2m_min": 12, "precipitation_sum": 2.0, "wind_speed_10m_max": 15},   # Borderline Good day
        ]

    returned_list = my_model.predict(days)

    for i, result in enumerate(returned_list):

        print(f"Day {i+1}:")
        print(f"temperature_2m_max: {days[i]["temperature_2m_max"]}")
        print(f"temperature_2m_min: {days[i]["temperature_2m_min"]}")
        print(f"precipitation_sum: {days[i]["precipitation_sum"]}")
        print(f"wind_speed_10m_max: {days[i]["wind_speed_10m_max"]}")
        print(f"result: {result.label} (probability: {result.probability:.3f})")
        print()

    # Omitting the guard would run this program every time this program is imported
    # in other scripts as a helper.
    
if __name__ == "__main__":
    main()

"""
1. This would prevent wrong predictions because of incorrect factors, like a switched high temperature of 0 and a precipitation sum of 25. These factors
have varying influences on the model so it could lead to incorrect results.

2. Keeping the original model separate makes the tests independent of the original model and can prevent any errors in the testing due to changing the
original model.

3. The component is already set up to handle multiple weather records as input and checks which features are in which position. One thing that would 
have to be added is a way to handle missing data.
"""