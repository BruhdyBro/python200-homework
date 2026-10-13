from dataclasses import dataclass
from .schemas import WeatherResponse

@dataclass
class HourlyReading:
    """An hourly reading of data from the weather
    
    Attributes:
        timestamp: The time at which the data was taken (formatted YYYY-MM-DDTHH:MM)
        temperature_c: The temperature reading in °C
        precipitation_mm: The amount of precipitation in milimeters"""
        
    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert an Open-Meteo API response into a list of hourly readings
        
    Args:
        response: The Open-Meteo API Response
            
    Returns:
        A list of readings each hour in HourlyReading format"""

    readings = response.hourly
    listHourly = []

    for i in range(len(readings.time)):
        listHourly.append(HourlyReading(readings.time[i], readings.temperature_2m[i], readings.precipitation[i]))

    return listHourly


# Q:    Why is HourlyReading a dataclass rather than a Pydantic model, when WeatherResponse is a Pydantic model? 
#       Be specific. A good answer identifies where the boundary is.
#
# A:    WeatherResponse is a Pydantic model as it is the boundary where the oustide data meets the program and shapes 
#       into the data written for the program.HourlyReading is a dataclass rather than a pydantic model because it is 
#       where data has already been validated.