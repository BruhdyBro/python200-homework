from weatherkit import WeatherResponse, to_readings, DailyAggregator, DailySummary
from pydantic import ValidationError
import json
from pathlib import Path

def main():
# Find the file
    parent = Path(__file__).resolve().parent
    weather_raw_file = parent / "weather_raw.json" 
    try:
        with open(weather_raw_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        try:
            processed_data = WeatherResponse.model_validate(data)

        except ValidationError as e:
            print("Couldn't validate model:", e)

        # Process it into a WeatherResponse model
        hourly_readings = to_readings(processed_data)

        # Convert to HourlyReading objects
        aggregator = DailyAggregator()

        # Aggregate to DailySummary objects
        daily_sums = aggregator.summarize(hourly_readings)
        missed_days = aggregator.incomplete_days(hourly_readings)

        # Print daily summaries with one per row
        for summary in daily_sums:
            print(
                f"Date: {summary.date} | Max Temperature: {summary.temp_max} | Min Temperature: {summary.temp_min} | Total Precipitation: {summary.precipitation_sum} | Hours Observed: {summary.hours_observed}")

        if missed_days:
            print()
            print(f"WARNING: the following days were skipped as they were under the minimum hour limit of {aggregator.min_hours}")

            for day in missed_days:
                print(day)

    except FileNotFoundError:
        print(f"Couldn't find weather_raw.json at {weather_raw_file}")

if __name__ == "__main__":
    main()

# Omitting the __main__ guard would make this script run the program it is imported and called in. When this program is called directly,
# it is assigned the name as "__main__". However, when imported as a modue, it's name becomes the module's name instead. Therefore, the
# guard ensure that the script only runs when this program is the one being called.



#                                                               --- Task 7 ---
# Q1:   Your WeatherResponse rejects the whole file if a single temperature is null. Is that the right behavior for a weather pipeline? 
#       Describe one situation where you would want it, and one where you would rather tolerate the gap. What would you change in the 
#       schema to tolerate it?
#
# A:    It is not the right behavior, as the entire dataset should not be invalidated by a singular Null value. If you had an entire
#       year's worth of data, a singular hour shouldn't ruin the entire year. However, there are specific situations where it would 
#       be useful. If you only had 5 hours worth of data for only one day, it could be ill-represented data or could lead to more
#       issues further down in the weather pipeline. To tolerate it, changing the data type from list[float] to "list[float | None]" 
#       in the schema would be able to handle Null values, however the rest of the modules would need to handle Null as well.
#
#
# Q2:   DailyAggregator.min_hours defaults to 24. What goes wrong if a pipeline runs at noon and the day is only half over? How does 
#       incomplete_days() help?
#
# A:    If the pipeline ran at noon, then the day would be marked as incomplete as it only shows 12 hours (at max) were observed and
#       would not have a DailySummary object created for that day's data. The incomplete_days() function helps by showing which days
#       were under the min_hours and would display each date that was not included, which shows the day when the pipeline ran at noon 
#
#
# Q3:   You wrote weatherkit as a package rather than one file. Name one concrete thing that becomes easier in Week 10, when a pipeline 
#       needs to import this code.
#
# A:    Weatherkit is written as a package rather than one large file because it improves maintainability and allows it to become modular.
#       This means that each section can be handled independently of the other, and doesn't require loading the entire file each time to
#       test, but rather just each small section.