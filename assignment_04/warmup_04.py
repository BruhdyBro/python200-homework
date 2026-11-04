import json
from dataclasses import dataclass
from pathlib import Path
import pytest

parent = Path(__file__).resolve().parent
models_folder = parent / "models"

# -=-=- Warmup Question 1 -=-=-
class ModelInfo:
    def __init__(self, path):
        
        with open(path) as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        return len(self.data["features"])

print("Creating ModelInfo object")
my_model = ModelInfo(models_folder / "weather_classifier_metadata.json")
print("Number of features:", my_model.feature_count())
print()





# -=-=- Warmup Question 2 -=-=-
@dataclass
class DayForecast:
    """Forecast for a single day
    
    Attributes:
        date: The date in which the weather was observed.
        high_c: High temperature of the day.
        will_rain: Whether or not it is forcasted to rain that day.
    """
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:

        if (self.will_rain):
            return f"{self.date}: high {self.high_c}C, rain expected"

        # only reached if will rain = false, i.e rain not expected
        return f"{self.date}: high {self.high_c}C, rain not expected"


day1 = DayForecast("2026-11-03", 23.7, False)
day2 = DayForecast("2023-06-15", 24.1, True)

# The dataclass automatically wrote the __init__ and __repr__ methods.

print("Summary of Day 1:", day1.summary())
print("Summary of Day 2:", day2.summary())
print()





# -=-=- Warmup Question 3 -=-=-
def parse_temperature(text: str) -> float:

    try:
        return float(text)

    except ValueError:
        raise ValueError("temperature must be a number.")

def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)

def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")

# Without using match, it wouldn't fail to catch a ValueError, but would rather fail
# to catch the specific ValueError. Using match allows us to find a specific text in
# the error message, so if somehow a different ValueError was raised, it wouldn't
# catch that one and would pass because it looks for any ValueError.