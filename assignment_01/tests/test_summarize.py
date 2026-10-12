from weatherkit import WeatherResponse, to_readings, DailyAggregator, DailySummary
import pytest
from pydantic import ValidationError

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


@pytest.fixture
def test_summary_list() -> list[DailySummary]:
    """Returns a list of 2 DailySummary objects for testing"""

    test_weather = {"latitude":20.0, 
                    "longitude":55.0, 
                    "timezone":"EST", 
                    "elevation":12.0, 
                    "hourly":{
                        "time":["2026-04-08T01:00","2026-04-08T02:00","2026-04-08T03:00","2026-04-08T04:00","2026-04-09T01:00","2026-04-09T02:00","2026-04-09T03:00","2026-04-09T04:00"],
                        "temperature_2m":[14.6, 14.8, 14.9, 15.1, 13.6, 13.8, 14.9, 14.1],
                        "precipitation":[0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3]}}
                    
    weather_data = WeatherResponse.model_validate(test_weather)
    readings = to_readings(weather_data)

    aggregator = DailyAggregator(min_hours=4)

    return aggregator.summarize(readings)


# Test 1
def test_date_grouping(test_summary_list):


    assert len(test_summary_list) == 2


# Test 2
def test_temp_min_max(test_summary_list):

    assert test_summary_list[0].temp_max == 15.1
    assert test_summary_list[0].temp_min == 14.6
    assert test_summary_list[1].temp_max == 14.9
    assert test_summary_list[1].temp_min == 13.6


# Test 3
def test_precipitation_sum(test_summary_list):

    assert test_summary_list[0].precipitation_sum == pytest.approx(0.6)
    assert test_summary_list[1].precipitation_sum == pytest.approx(4.6)


# Test 4
def test_min_hours():

    # Different from earlier test data as it features one extra hour of data on 2026-04-09T05:00.
    test_weather = {"latitude":20.0, 
                    "longitude":55.0, 
                    "timezone":"EST", 
                    "elevation":12.0, 
                    "hourly":{
                        "time":["2026-04-08T01:00","2026-04-08T02:00","2026-04-08T03:00","2026-04-08T04:00","2026-04-09T01:00","2026-04-09T02:00","2026-04-09T03:00","2026-04-09T04:00", "2026-04-09T05:00"],
                        "temperature_2m":[14.6, 14.8, 14.9, 15.1, 13.6, 13.8, 14.9, 14.1, 14.4],
                        "precipitation":[0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 1.2]}}
                    
    weather_data = WeatherResponse.model_validate(test_weather)
    readings = to_readings(weather_data)

    aggregator = DailyAggregator(min_hours=5)

    summary_list = aggregator.summarize(readings)
    missed_days = aggregator.incomplete_days(readings)

    assert len(summary_list) == 1
    assert len(missed_days) == 1



# Test 5
def test_low_min_hours_on_full_data():


                    
    weather_data = WeatherResponse.model_validate(processed_data)
    readings = to_readings(weather_data)

    aggregator = DailyAggregator(min_hours=5)

    summary_list = aggregator.summarize(readings)

    assert len(summary_list) == 7


# Test 6 - 10
# Deliberately breaking the temp_range function by changing the subtraction to addition subsequentely failed
# all these parametrized tests. Each test showed numbers larger than the max, indicating not only a larger number
# than was input, but that an addition actually occured. The results were as follows:
# T6: 30.1 | T7: 36.1 | T8: 14.8 | T9: 26.7 | T10: 30.1
# Manual addition of the max and min values (1st and 2nd after the string date) confirms these values
@pytest.mark.parametrize(
    "values, expected",
    [
        (DailySummary("2026-04-08", 18.2, 11.9, 0.0, 24), 6.3),
        (DailySummary("2026-04-09", 21.7, 14.4, 6.2, 24), 7.3),
        (DailySummary("2026-04-10", 11.6, 3.2, 10.8, 24), 8.4),
        (DailySummary("2026-04-11", 14.9, 11.8, 2.5, 10), 3.1),
        (DailySummary("2026-04-12", 18.2, 11.9, 0.0, 5), 6.3),
    ]
)
def test_temp_range(values, expected):

    assert values.temp_range() == pytest.approx(expected)