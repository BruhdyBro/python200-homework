# --- Classes ---

# Q1

class Thermometer:
    def __init__(self, location, readings = []):
        self.location = location
        self.readings = readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if len(self.readings) < 1:
            return None
        else:
            total = 0
            for reading in self.readings:
                total += reading
            return (total / len(self.readings))

    def hottest(self):
        if len(self.readings) < 1:
            return None
        hottest = self.readings[0]
        for reading in self.readings:
            if reading > hottest:
                hottest = reading
        return hottest

    # Q2 Function
    def __repr__(self):
        return f"{type(self).__name__}(location='{self.location}', n_readings={len(self.readings)}, average={self.average()})"


print("--- Testing Thermometer Class ---")
therm1 = Thermometer("Orlando", [80, 85, 105, 90])

print(therm1.average())
print(therm1.hottest())
print()

# Q: why does average() need to handle the empty case? What would happen without that check?
# A: When calculating the average of a set, it divides by the total by amount of numbers in the set.
#    If you try to calculate the average of a an empty set, you would be dividing by 0, which causes
#    a ZeroDivisionError.


# Q2
print("--- Testing Thermometer Repr ---")
print(therm1)
print()


# Q3
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        breached = []
        for reading in thermometer.readings:
            if reading > self.threshold:
                breached.append(reading)

        return breached
    
print("--- Testing Temperature Alert ---")

Alert1 = TemperatureAlert(75.5)
Alert2 = TemperatureAlert(87.5)

print(Alert1.breaches(therm1))
print(Alert2.breaches(therm1))
print()

# Q: why is the threshold stored on TemperatureAlert rather than passed as an argument to breaches()? 
#    What advantage does that give you if you have twenty thermometers to check?
# A: Using the threshold as a stored value means that an extra argument doesn't need to be passed and 
#    evaluated when searching for breaches. It means the same threshold can be easily repeated and
#    used to check all 20 thermometers without having to manually input the same threshold.



# --- Dataclasses ---

from dataclasses import dataclass, FrozenInstanceError, field

# Q1

@dataclass(frozen=True) # Frozen is Q2
class Station:
    """Name and location for station
    
    Attributes:
        station_id: Specific ID number for the station.
        name: Name of the station.
        latitude: Latitude index of the station on a map.
        longitude: longitude index of the station on a map.
        elevation: Elevation level of the station in feet.
    """
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float

print("--- Testing Station Dataclass ---")

station_a  = Station(4, "Newport News", 107.2, 75.9, 4.5)
station_b = Station(4, "Newport News", 107.2, 75.9, 4.5)

print("station_a == station_b :", station_a == station_b)
print()
# Prints True because the dataclass automatically writes an __init__, __repr__, and __eq__ method for
# the class. This means it can properly test if 2 station objects are equal to each other. If the 
# print would be written without the dataclass decorator, there would be no __eq__ method set up, and
# the print statement would return false because it compares the location in memory, not object values.


# Q2
print("--- Testing Station Frozen and Length of Station Set ---")
try:
    station_a.name = "Orlando"

except FrozenInstanceError as e:
    print("Couldn't change the object:", e)
    print()


station_set = {Station(1, "Orlando", 82.4, 69.5, 1.3), Station(1, "Orlando", 82.4, 69.5, 1.3), Station(2, "Jacksonville", 82.7, 70.0, 1.6)}

print("3 stations when creating set, with 2 identical. Length of station set:", len(station_set))
print()
# Frozen dataclasses become hashable, which means it can be put into sets to remove duplicate entries.


# Q3
@dataclass
class StationBatch:
    """A set of stations clustered by region
    Attributes:
        region: Name of the region where the stations are in
        stations: A list of Station objects that are in the same region
    """
    region: str 
    stations: list[Station] = field(default_factory=list)


    def add(self, station: Station) -> None:
        """Add a station to the batch"""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Find the station with the highest elevation, 
        or return None if there are no stations."""
        if len(self.stations) < 1:
            return None
        highest = self.stations[0]
        for station in self.stations:
            if station.elevation > highest.elevation:
                highest = station
        return highest
        
#ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
# Creating an empty set as [] would define them the same list as a default value, and appending to
# one list would append to all of them. Instead, the fix (using default_factory) allows for the 
# dataclass to understand what type of data it is and call the function to make the default for 
# that class



# --- Pydantic ---

from pydantic import BaseModel, Field, ValidationError, model_validator

# Q1 

class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def bad_sensor(self):
        if self.temperature_c < -40 and self.humidity == 0.0:
            raise ValueError(
                f"humidity of 0.0 and {self.temperature_c} (below -40) indicates a failed sensor rather than real weather"
            )
        return self

print("--- Testing Reading Initial Creation ---")

read1 = Reading(
    station_id="reading1", 
    timestamp="2026-10-05", 
    temperature_c=22.7, 
    humidity=65.9
)

print(read1)
print()


# Q2

print("--- Testing Reading Invalid Creation ---")

try:
    read2 = Reading(
        station_id="reading2", 
        temperature_c=22.7, 
        humidity=65.9
    )
except ValidationError as e:
    print("Could not create the Reading:", e)
    print()

try:
    read3 = Reading(
        station_id="reading3", 
        timestamp="2026-10-05", 
        temperature_c=150.0, 
        humidity=65.9
    )
except ValidationError as e:
    print("Could not create the Reading:", e)
    print()

try:
    read4 = Reading(
        station_id="reading4", 
        timestamp="2026-10-05", 
        temperature_c=22.7, 
        humidity="Very Humid"
    )
except ValidationError as e:
    print("Could not create the Reading:", e)
    print()


read5 = Reading(
    station_id="reading5",
    timestamp="2026-10-05",
    temperature_c="21.5",
    humidity=40
)

print("Valid Reading:")
print(read5)
print("Type of temperature_c:", type(read5.temperature_c))
print("Type of humidity:", type(read5.humidity))
print()

# Q: why does Pydantic accept "21.5" but reject "very humid"? State the rule in your own words.
# A: Pydantic accepts "21.5" because it is a string that can be converted to a float, so it 
#    automatically does so and inputs it as the value. It cannot, however, convert the text
#    "Very Humid" into a float because it is not a number, just words. This is called Sensible
#    Coercion and only occurs "when the conversion is unambiguous and safe".


# Q3

print("--- Testing Invalid Creation with Multiple Errors ---")

try:
    read6 = Reading(
        station_id="reading6",
        temperature_c="number",
        humidity=150.0
    )

except ValidationError as e:
    for error in e.errors():
        print("loc:", error["loc"])
        print("msg:", error["msg"])
print()

# Q: how many errors were reported, and why is reporting all of them at once more useful than stopping at the first?
# A: 3 errors were reported. It is more useful to report all of them so you can see all the errors at the same time
#    rather than fixing one and then running the test again, especially when under things like time constraints.  

print("--- Testing Model Validator ---")

# Q4

try:
    read7 = Reading(
            station_id="reading7",
            timestamp="2026-10-05",
            temperature_c=50.0,
            humidity=0.0
        )

    print(read7)
    print()

except ValidationError as e:
    print("Bad sensor reading:", e)
    print()

try:
    read8 = Reading(
        station_id="reading8",
        timestamp="2026-10-05",
        temperature_c=-50.0,
        humidity=0.0
    )

    print(read8)
    print()

except ValidationError as e:
    print("Bad sensor reading:", e)
    print()

# A bad sensor couldnt be detected with Field constraints alone because in this case, field constraints can only
# check individual fields, while this rule requires to check a relationship between 2 different fields.


# --- Pytest ---

# Q1

import pytest

def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts degress Celsius into degrees Fahrenheit"""
    return (celsius*9)/5 + 32

def test_0c():
    temp = celsius_to_fahrenheit(0)
    print(temp)
    assert temp == 32.0

def test_100c():
    temp = celsius_to_fahrenheit(100)
    print(temp)
    assert temp == 212

def test_37c():
    temp = celsius_to_fahrenheit(37)
    assert temp == pytest.approx(98.6)
    # Approx is used because there is a small difference in the far decimals which has  the test fail since 
    # it is not exact


# Q2
def mean(values: list[float]) -> float:
    """Given a list of values, returns the mean of the list"""
    length = len(values)
    if length < 1:
        raise ValueError("values cannot be empty")
    return sum(values) / length


def test_mean_of_empty_raises():
    
    with pytest.raises(ValueError, match="empty"):
        mean([])
        # pytest.raises(ValueError) would pass if any ValueError occurs, but the match="empty" makes sure it 
        # is from an empty list rather than any ValueError.


# Q3
@pytest.mark.parametrize(
    "values, expected",
    [
       ([5], 5),
       ([1,2,3], 2),
       ([-10, -35, -45], -30),
       ([0, 0], 0),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == expected

#warmup_01.py::test_0c PASSED                                                                                                                                                           [ 12%]
#warmup_01.py::test_100c PASSED                                                                                                                                                         [ 25%]
#warmup_01.py::test_37c PASSED                                                                                                                                                          [ 37%]
#warmup_01.py::test_mean_of_empty_raises PASSED  
#warmup_01.py::test_mean_values[values0-5] PASSED                                                                                                                                       [ 62%]
#warmup_01.py::test_mean_values[values1-2] PASSED                                                                                                                                       [ 75%]
#warmup_01.py::test_mean_values[values2--30] PASSED                                                                                                                                     [ 87%]
#warmup_01.py::test_mean_values[values3-0] PASSED 

#============== 8 passed in 0.09s ==============

# Q: why is one parametrized test with four cases better than four nearly identical test functions?
# A: Parametrizing allows you to combine similar tests into one set rather than having numerous similar tests


# Q4

"""
warmup_01.py::test_0c PASSED                                                                                                                                                           [ 12%]
warmup_01.py::test_100c FAILED                                                                                                                                                         [ 25%]
warmup_01.py::test_37c FAILED                                                                                                                                                          [ 37%]
warmup_01.py::test_mean_of_empty_raises PASSED                                                                                                                                         [ 50%]
warmup_01.py::test_mean_values[values0-5] PASSED                                                                                                                                       [ 62%]
warmup_01.py::test_mean_values[values1-2] PASSED                                                                                                                                       [ 75%]
warmup_01.py::test_mean_values[values2--30] PASSED                                                                                                                                     [ 87%]
warmup_01.py::test_mean_values[values3-0] PASSED                                                                                                                                       [100%]

============== FAILURES ==============
______________ test_100c ______________

    def test_100c():
        temp = celsius_to_fahrenheit(100)
        print(temp)
>       assert temp == 212
E       assert 257.0 == 212

warmup_01.py:340: AssertionError
-------------- Captured stdout call --------------
257.0
______________ test_37c ______________

    def test_37c():
        temp = celsius_to_fahrenheit(37)
>       assert temp == pytest.approx(98.6)
E       assert 115.25 == 98.6 ± 9.9e-05
E         
E         comparison failed
E         Obtained: 115.25
E         Expected: 98.6 ± 9.9e-05

warmup_01.py:344: AssertionError
============== short test summary info ==============
FAILED warmup_01.py::test_100c - assert 257.0 == 212
FAILED warmup_01.py::test_37c - assert 115.25 == 98.6 ± 9.9e-05
============== 2 failed, 6 passed in 0.15s ==============
"""

# Q: what specific values did pytest show you in the failure report, and why is that more useful than a bare 
#    "assertion failed"?
# A: The pytest showed what the number was that it was testing against the assert, i.e the converted number.
#    This helps describe why it failed because you can see it being higher/lower than expected and narrow
#    down which sections cause problems.