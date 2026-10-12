from weatherkit import WeatherResponse, HourlyBlock, HourlyReading, to_readings


import json
from pathlib import Path

parent = Path(__file__).resolve().parent.parent
weather_raw_file = parent / "weather_raw.json" 
try:
    with open(weather_raw_file, "r", encoding="utf-8") as file:
        data = json.load(file)

        processed_data = WeatherResponse.model_validate(data)

except FileNotFoundError:
    print(f"Couldn't find weather_raw.json at {weather_raw_file}")



# Test 1
def test_to_readings():

    readings = to_readings(processed_data)
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-14T23:00"


# Test 2
def test_value_matches():

    test_weather = {"latitude":20.0, 
                    "longitude":55.0, 
                    "timezone":"EST", 
                    "elevation":12.0, 
                    "hourly":HourlyBlock(
                        time=["2026-04-08T01:00","2026-04-08T02:00","2026-04-08T03:00","2026-04-08T04:00","2026-04-09T01:00","2026-04-09T02:00","2026-04-09T03:00","2026-04-09T04:00"],
                        temperature_2m=[14.6, 14.8, 14.9, 15.1, 13.6, 13.8, 14.9, 14.1],
                        precipitation=[0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3])}
            
    weather_data = WeatherResponse.model_validate(test_weather)
    
    readings = to_readings(weather_data)
    for index, reading in enumerate(readings):
        assert reading.timestamp == weather_data.hourly.time[index]
        assert reading.temperature_c == weather_data.hourly.temperature_2m[index]
        assert reading.precipitation_mm == weather_data.hourly.precipitation[index]


# Test 3
def test_equal_readings():
    
    hourly1 = HourlyReading("2026-04-14T23:00", 24.8, 0.00)
    hourly2 = HourlyReading("2026-04-14T23:00", 24.8, 0.00)

    assert hourly1 == hourly2