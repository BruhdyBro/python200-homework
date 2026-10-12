from weatherkit import HourlyBlock, WeatherResponse
import pytest
from pydantic import ValidationError

import json
from pathlib import Path

parent = Path(__file__).resolve().parent.parent
weather_raw_file = parent / "weather_raw.json" 
try:
    with open(weather_raw_file, "r", encoding="utf-8") as file:
        data = json.load(file)

except FileNotFoundError:
    print(f"Couldn't find weather_raw.json at {weather_raw_file}")

# Using a relative path is unreliable because it assumes the current working directory. If you were to run the tests
# from a different directory, it wouldn't be able to locate the JSON file because it's relative to the directory, 
# which is not the same one as the test files. By resolving the file path, getting the parent, and finding the JSON
# file like that, you build a path relative to the test files, instead of the working directory


# Test 1
def test_valid_response():
    processed_data = WeatherResponse.model_validate(data)

    assert len(processed_data.hourly.time) == 168
    assert len(processed_data.hourly.temperature_2m) == 168
    assert len(processed_data.hourly.precipitation) == 168


# Test 2
def test_invalid_latitude():

    with pytest.raises(ValidationError):
        test_weather = {"latitude":200.0, 
                        "longitude":55.0, 
                        "timezone":"EST", 
                        "elevation":12.0, 
                        "hourly":HourlyBlock(
                            time=["2026-04-08T01:00"],
                            temperature_2m=[14.6],
                            precipitation=[0.0])}
        
        large_latitude = WeatherResponse.model_validate(test_weather)


# Test 3
def test_mismatch_lengths():

    with pytest.raises(ValidationError):
        HourlyBlock(time=["2026-04-08T01:00","2026-04-08T02:00","2026-04-08T03:00","2026-04-08T04:00"],
                    temperature_2m=[14.6, 14.8, 14.9],
                    precipitation=[0.0, 2.1, 1.3, 1.8])


# Test 4
def test_null_temp():
    
    with pytest.raises(ValidationError):
            HourlyBlock(time=["2026-04-08T01:00","2026-04-08T02:00","2026-04-08T03:00","2026-04-08T04:00"],
                        temperature_2m=[14.6, 14.8, None, 14.9],
                        precipitation=[0.0, 2.1, 1.3, 1.8])