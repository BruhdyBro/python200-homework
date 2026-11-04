import json
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

# -=-=- Task 1 -=-=-
@dataclass
class Prediction:
    """
    Prediction values of a running day.

    Attributes:
        label: Either "good" or "skip".
        probability: The probability, from 0 to 1, that the day is good for running.

    """

    label: str
    probability: float





# -=-=- Task 2 -=-=-
class WeatherClassifier:
    """
    Weather predicter model. Stores the model and makes predictions.
    """

    def __init__(self, model_path: str | Path):
        """
        Stores a trained model from a specified path

        Attributes:
            model_path: The specific file path to the model.

        Raises:
            FileNotFoundError: If the file is not found or file path is incorrect
        """
        try:
            self.model = joblib.load(model_path)

        except FileNotFoundError:
            raise FileNotFoundError("No file exists at that path.")



    def predict(self, days: list[dict]) -> list[Prediction]:
        """
        Makes a prediction based off a list of daily weather observations

        Attributes:
            days: a list of dicts in the SPECIFIC format (temperature_2m_max, temperature_2m_min, precipitation_sum, wind_speed_10m_max).

        Returns:
            proba_list: A list of Prediction objects, one for each day in the input list.

        Raises:
            ValueError: If a specific feature not provided properly in the specified order.
        """

        if not days:
            return []

        features = {"temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"}

        for day in days:
            curr_features = set(day.keys())
            list_features = list(day.keys())

            missing = features - curr_features

            if missing:
                raise ValueError(f"Missing a feature. Missing: {missing}")

            if not (list_features[0] == "temperature_2m_max"):
                raise ValueError("Missing feature: temperature_2m_max")
            
            if not (list_features[1] == "temperature_2m_min"):
                raise ValueError("Missing feature: temperature_2m_min")

            if not (list_features[2] == "precipitation_sum"):
                raise ValueError("Missing feature: precipitation_sum")
                        
            if not (list_features[3] == "wind_speed_10m_max"):
                raise ValueError("Missing feature: wind_speed_10m_max")

        
        X = pd.DataFrame(
            [
                {"temperature_2m_max": day["temperature_2m_max"], 
                "temperature_2m_min": day["temperature_2m_min"], 
                "precipitation_sum": day["precipitation_sum"], 
                "wind_speed_10m_max": day["wind_speed_10m_max"]} for day in days
            ]
        )

        predicts = self.model.predict(X)
        probabs = self.model.predict_proba(X)[:, 1]

        proba_list = []

        for prediction, probability in zip(predicts, probabs):

            if prediction == 1:
                label_text = "good"

            else:
                label_text = "skip"

            proba_list.append(Prediction(label_text, probability))
                

        return proba_list