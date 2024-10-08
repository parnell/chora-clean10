import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from chora.models.abstract_model import AbstractModel

logging.basicConfig(level=logging.DEBUG)


@dataclass
class ClusteringModel(AbstractModel):
    RANDOM_STATE: ClassVar[int] = 42
    DEFAULT_N_CLUSTERS: ClassVar[int] = 20

    pipeline: Pipeline
    feature_columns: list[str]
    data_df: Optional[pd.DataFrame] = None
    n_clusters: int = DEFAULT_N_CLUSTERS
    clusters: Optional[np.ndarray] = field(default=None)
    genre_column: str = "genres"
    similarity_threshold: float = 0.0

    def save(self, filepath: str) -> None:
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClusteringModel":
        return joblib.load(filepath)

    def fit(self, X: pd.DataFrame, **kwargs: Any) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input dataframe X is empty.")

        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        X_numeric = X[self.feature_columns]
        self.pipeline.fit(X_numeric)
        self.__dict__["clusters"] = self.pipeline.predict(X_numeric)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        try:
            check_is_fitted(self.pipeline.named_steps["kmeans"], "cluster_centers_")
        except NotFittedError:
            raise AttributeError(
                "Model has not been trained yet. Call 'fit' with appropriate data first."
            )

        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        X_numeric = X[self.feature_columns]
        return self.pipeline.predict(X_numeric)  # type: ignore

    def get_weighted_vector(self, data: pd.DataFrame, user_ratings: pd.DataFrame) -> np.ndarray:
        merged_data = pd.merge(
            user_ratings[["song_id", "rating"]], data, left_on="song_id", right_on="id", how="inner"
        )

        if merged_data.empty:
            raise ValueError("No ratings found matching the songs.")

        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(merged_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in merged data: {missing_features}")

        # Select and order features to match the fitted model
        features_for_transform = merged_data[self.feature_columns]

        scaler = self.pipeline.named_steps["scaler"]
        normalized_features = scaler.transform(features_for_transform)

        weights = merged_data["rating"].values
        weighted_features = normalized_features * weights[:, np.newaxis]
        weighted_vector = np.average(weighted_features, axis=0, weights=np.abs(weights))  # type: ignore

        return weighted_vector

    def recommend(
        self,
        n: int,
        data: Optional[pd.DataFrame],
        user_id: Optional[str] = None,
        excluded_set: Optional[Set[str]] = None,
        user_ratings: Optional[pd.DataFrame] = None,
        genre_weight: Optional[float] = 0.3,
        sample_size: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        assert user_ratings is not None
        if data is None:
            data = self.data_df
        assert data is not None
        try:
            # Pre-filter data
            if excluded_set:
                data = data[~data["id"].isin(excluded_set)]

            # Prepare user vector
            song_vector = self.get_weighted_vector(data, user_ratings)
            logging.debug(f"Weighted song vector: {song_vector}")

            # Prepare genre scores
            genre_counts: Counter[str] = Counter()
            positive_ratings = user_ratings[user_ratings["rating"] > 0]
            for genres in data.loc[data["id"].isin(positive_ratings["song_id"]), self.genre_column]:
                genre_counts.update(genres)

            # Check if genre_counts is empty
            if not genre_counts:
                genre_similarities = np.zeros(len(data))  # No genres, set similarities to zero
                logging.debug("No relevant genres found, setting genre similarities to zero.")
            else:
                total_genre_count = sum(genre_counts.values())
                genre_scores = {
                    genre: count / total_genre_count for genre, count in genre_counts.items()
                }

                # Calculate genre similarities
                genre_similarities = np.array(
                    [
                        sum(genre_scores.get(genre, 0) for genre in genres)
                        for genres in data[self.genre_column]
                    ]
                )
                if total_genre_count > 0:
                    genre_similarities /= total_genre_count
                else:
                    genre_similarities.fill(0)  # Avoid NaNs if total genre count is 0
                logging.debug(f"Genre similarities: {genre_similarities}")

            # Normalize features
            features = data[self.feature_columns].values
            scaler = self.pipeline.named_steps["scaler"]
            normalized_features = scaler.transform(features)
            logging.debug(f"Normalized features: {normalized_features}")

            # Calculate cosine similarities
            # Before calculating cosine similarities
            # Ensure both vectors are valid
            cosine_similarities = 1 - cdist([song_vector], normalized_features, metric="cosine")[0]
            cosine_similarities = np.nan_to_num(cosine_similarities, nan=0.0)  # Replace NaN with 0
            logging.debug(f"Cosine similarities: {cosine_similarities}")

            # Combine scores
            if not genre_weight:
                combined_scores = cosine_similarities
            else:
                combined_scores = (
                    1 - genre_weight
                ) * cosine_similarities + genre_weight * genre_similarities
            logging.debug(f"Combined scores: {combined_scores}")

            # Get top recommendations
            top_indices = np.argsort(combined_scores)[-n:][::-1]
            recommendations = list(zip(data.iloc[top_indices]["id"], combined_scores[top_indices]))

            return [
                (song_id, sim)
                for song_id, sim in recommendations
                if sim >= self.similarity_threshold
            ]

        except Exception as e:
            logging.error(f"An error occurred in the recommend method: {str(e)}")
            raise
