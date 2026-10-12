from pydantic import BaseModel, Field, model_validator

class HourlyBlock(BaseModel):
    """The columnar Arrays from the Open-Meteo API
    
    Attributes:
        time: list of times in which the weather was observed.
        temperature_2m: The temperature recorded.
        precipitation: The amount of rain recorded.
    """

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def bad_length(self):
        len_a = len(self.time)
        len_b = len(self.temperature_2m)
        len_c = len(self.precipitation)

        if len_a != len_b or len_b != len_c:
            raise ValueError(
                f"length of input lists are not the same. time: {len_a}, temperature_2m: {len_b}, precipitation: {len_c}"
            )
        return self



class WeatherResponse(BaseModel):
    """The top level response processor from the Open-Meteo API
    
    Attributes:
        latitude: The latitudinal degree where the weather was observed.
        longitude: The longitudinal degree where the weather was observed.
        timezone: The timezone in which the weather was observed.
        elevation: The elevation level where the weather was observed.
        hourly: List of times and data where the weather observed.
    """

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock