from dataclasses import dataclass
from .records import HourlyReading

@dataclass
class DailySummary:

    """A summary of the weather events on a given day

    Attributes:
        date: The specific day that was observed
        temp_max: The highest temperature in degrees Celsius observed that day
        temp_min: The lowest temperature in degrees Celsius observed that day
        precipitation_sum: The total amount of precipitation observed in milimeters
        hours_observed: The amount of hours that data was recorded for on that day
    
    """

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Returns the range for the temperature (max - min)."""
        return self.temp_max - self.temp_min


class DailyAggregator:

    """Aggregates weather readings by each day."""

    def __init__(self, min_hours: int = 24):
        self.min_hours = min_hours

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Groups readings by calendar date and creates a DailySummary for each of them"""

        # dict of (str: list[HourlyReading]) to capture each day and its hourly readings
        daily_data = {} 

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in daily_data:
                daily_data[date] = []

            daily_data[date].append(reading)

        summaries = []

        for date, day_readings in sorted(daily_data.items()):
            
            hours = len(day_readings) 

            if hours < self.min_hours:
                continue

            temps = []
            precipitation = []
            for reading in day_readings:
                temps.append(reading.temperature_c)
                precipitation.append(reading.precipitation_mm)

            summaries.append(DailySummary(date, max(temps), min(temps), sum(precipitation), hours))

        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str] | None:
        """Returns dates with fewer than min_hours observations or None if all days were complete"""
        daily_counts = {}

        for reading in readings:
            date = reading.timestamp[:10]
            daily_counts[date] = daily_counts.get(date, 0) + 1

        days = []
        for date, count in daily_counts.items():
            if count < self.min_hours:
                days.append(date)

        if len(days) == 0:
            return None

        return days