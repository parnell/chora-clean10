from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
import pytest
from river import base, dummy, metrics, optim, preprocessing, reco, stats
from river.evaluate import progressive_val_score

from chora.models.online_models import BiasedMFModel


class TestModel(base.Regressor):
    def predict_one(self, x, user=None, item=None):
        # Extract feature1 and feature2 based on the type of x
        if isinstance(x, dict):  # If x is a dictionary
            feature1 = x.get("feature1", 0)
            feature2 = x.get("feature2", 0)
        elif isinstance(x, (list, pd.Series)):  # If x is a list or pandas Series
            feature1 = x[0]  # Adjust the index based on your data structure
            feature2 = x[1]  # Adjust the index based on your data structure
        elif isinstance(x, np.ndarray):  # If x is a NumPy array
            feature1 = x[0]  # Assuming feature1 is at index 0
            feature2 = x[1]  # Assuming feature2 is at index 1
        else:
            raise ValueError("Unsupported input type for x")

        return feature1 * 0.5 + feature2 * 0.5  # Example prediction logic

    def learn_one(self, x, y):
        pass  # Implement learning logic if necessary


@pytest.fixture
def online_model():
    om = BiasedMFModel()
    om.model = TestModel()  # Use the simple TestModel for predictable predictions
    return om


@pytest.fixture
def sample_data():
    # Create a sample DataFrame for songs
    return pd.DataFrame(
        {
            "id": ["song1", "song2", "song3"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [0.4, 0.5, 0.6],
        }
    )


@pytest.fixture
def user_ratings():
    # Create a sample DataFrame for user ratings
    return pd.DataFrame(
        {"user_id": ["user1", "user2"], "song_id": ["song1", "song2"], "rating": [1, 1]}
    )


def test_recommend(online_model, sample_data, user_ratings):
    n = 2
    excluded_set = {"song3"}  # Exclude one song

    recommendations: List[Tuple[str, float]] = online_model.recommend(
        data=sample_data, user_ratings=user_ratings, n=n, excluded_set=excluded_set
    )

    # Assert that the recommendations contain only the expected song IDs
    assert len(recommendations) == n
    assert all(song_id in {"song1", "song2"} for song_id, _ in recommendations)

    # Check predicted ratings based on the TestModel's logic
    expected_scores = {
        "song1": 0.1 * 0.5 + 0.4 * 0.5,  # (feature1 * 0.5 + feature2 * 0.5)
        "song2": 0.2 * 0.5 + 0.5 * 0.5,
    }

    for song_id, score in recommendations:
        assert score == expected_scores[song_id]  # Compare against expected score


if __name__ == "__main__":
    pytest.main(["-v", __file__])
