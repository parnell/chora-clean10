from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chora.models.clustering import ClusteringModel


@pytest.fixture
def sample_data() -> pd.DataFrame:
    # Create a sample DataFrame for testing
    data: dict[str, list[Any]] = {
        "id": [1, 2, 3],
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "genres": [["rock"], ["pop"], ["jazz"]],
    }
    return pd.DataFrame(data)


@pytest.fixture
def model(sample_data: pd.DataFrame) -> ClusteringModel:
    # Create a simple KMeans pipeline for the model
    pipeline: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=2, random_state=ClusteringModel.RANDOM_STATE)),
        ]
    )
    return ClusteringModel(pipeline=pipeline, feature_columns=["feature1", "feature2"])


def test_fit_valid_data(model: ClusteringModel, sample_data: pd.DataFrame) -> None:
    model.fit(sample_data[["feature1", "feature2"]])
    assert model.pipeline.named_steps["kmeans"].n_clusters == 2
    assert model.pipeline.named_steps["kmeans"].inertia_ is not None


def test_fit_invalid_data(model: ClusteringModel) -> None:
    with pytest.raises(ValueError):
        df = pd.DataFrame()
        model.fit(df)

    with pytest.raises(ValueError, match="Input dataframe X is empty."):
        model.fit(pd.DataFrame())


def test_predict_before_fit(model: ClusteringModel, sample_data: pd.DataFrame) -> None:
    with pytest.raises(AttributeError):
        model.predict(sample_data[["feature1", "feature2"]])


def test_predict_valid_data(model: ClusteringModel, sample_data: pd.DataFrame) -> None:
    model.fit(sample_data[["feature1", "feature2"]])
    predictions: np.ndarray = model.predict(sample_data[["feature1", "feature2"]])
    assert len(predictions) == len(sample_data)


def test_recommend(model: ClusteringModel, sample_data: pd.DataFrame) -> None:
    user_ratings: pd.DataFrame = pd.DataFrame({"song_id": [1, 2], "rating": [5, 3]})
    model.fit(sample_data[["feature1", "feature2"]])

    recommendations: list[tuple[str, float]] = model.recommend(
        data=sample_data, user_ratings=user_ratings, n=2, excluded_set=set()
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 2  # Should return at most n recommendations


def test_recommend_with_two_ratings(model: ClusteringModel, sample_data: pd.DataFrame) -> None:
    # User provides ratings for two songs
    user_ratings: pd.DataFrame = pd.DataFrame({"song_id": [1, 2], "rating": [5, 3]})

    # Fit the model with the sample data
    model.fit(sample_data[["feature1", "feature2"]])

    # Call the recommend method to get recommendations
    recommendations: list[tuple[str, float]] = model.recommend(
        data=sample_data, user_ratings=user_ratings, n=1, excluded_set=set()
    )

    # Check that the recommendations are a list and not empty
    assert isinstance(recommendations, list)
    assert len(recommendations) == 1  # Should return exactly one recommendation

    # Check that the recommended song is not in the user's rated songs
    rated_songs = set(user_ratings["song_id"])
    assert recommendations[0] not in rated_songs


if __name__ == "__main__":
    pytest.main([__file__])
