import pytest
import pandas as pd
import joblib
from weather_model import Prediction, WeatherClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -=-=- Task 4 -=-=-

@pytest.fixture
def model_path(tmp_path):
    """Train a tiny model on obvious days and save it to a temp file."""
    good = [{"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10}] * 15
    bad = [{"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40}] * 15
    FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
    X = pd.DataFrame(good + bad)[FEATURES]
    y = [1] * 15 + [0] * 15

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)

    path = tmp_path / "weather_classifier.pkl"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def clf(model_path):
    """A WeatherClassifier loaded from the saved tiny model."""
    return WeatherClassifier(model_path)


@pytest.mark.parametrize("day, label, result", [
    ([{"temperature_2m_max": 20, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 15}], "good", 0.972),
    ([{"temperature_2m_max": 5, "temperature_2m_min": -15, "precipitation_sum": 19.0, "wind_speed_10m_max": 36},], "skip", 0.007)
])
def test_single_day(clf, day, label, result):

    returned_list = clf.predict(day)

    assert returned_list[0].label == label
    assert round(returned_list[0].probability, 3) == pytest.approx(result)


def test_probability_between_0_1(clf):

    day = [{"temperature_2m_max": 24, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 13}]

    returned_list = clf.predict(day)

    assert returned_list[0].probability >= 0
    assert returned_list[0].probability <= 1


def test_3_days(clf):

    days = [
        {"temperature_2m_max": 24, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 13},
        {"temperature_2m_max": 20, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 15},
        {"temperature_2m_max": 35, "temperature_2m_min": 0, "precipitation_sum": 9.0, "wind_speed_10m_max": 26}
        ]

    returned_list = clf.predict(days)

    assert len(returned_list) == 3

    assert returned_list[0].label == "good"
    assert round(returned_list[0].probability, 3) == pytest.approx(0.991)

    assert returned_list[1].label == "good"
    assert round(returned_list[1].probability, 3) == pytest.approx(0.972) 

    assert returned_list[2].label == "good"
    assert round(returned_list[2].probability, 3) == pytest.approx(0.887)


def test_empty_list(clf):

    day = []
    
    returned_list = clf.predict(day)

    assert len(returned_list) == 0
    assert returned_list is not None


def test_feature_order(clf):

    day = [{"temperature_2m_max": 24, "temperature_2m_min": 15, "wind_speed_10m_max": 13, "precipitation_sum": 0.0}]

    returned_list = clf.predict(day)
    
    assert returned_list[0].label == "good"
    assert round(returned_list[0].probability, 3) == pytest.approx(0.991)


def test_missing_feature(clf):

    with pytest.raises(ValueError, match="wind"):

        day = [{"temperature_2m_max": 24, "temperature_2m_min": 15, "precipitation_sum": 0.0}]

        returned_list = clf.predict(day)


def test_missing_file():

    with pytest.raises(FileNotFoundError, match="No file exists at that path."):

        day = [{"temperature_2m_max": 24, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 13}]

        classifier = WeatherClassifier("weather_model.pkl")
        