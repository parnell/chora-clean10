from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from chora.models.abstract_model import AbstractModel


@dataclass
class RecommenderService:
    models: dict[str, AbstractModel]
    likes_df: pd.DataFrame
    data_df: pd.DataFrame
    tdata_df: pd.DataFrame  # transformed data

    def __post_init__(self):
        for model in self.models.values():
            model.fit(self.data_df)

    def recommend(
        self,
        n: int,
        user_id: Optional[str] = None,
        excluded_set: Optional[set[str]] = None,
        user_ratings: Optional[pd.DataFrame] = None,
        genre_weight: Optional[float] = None,
        sample_size: Optional[int] = 10000,
        model_weighting: Optional[List[float]] = None,
        **kwargs
    ) -> pd.DataFrame:
        # Ensure we have weights for each model
        if model_weighting is None:
            model_weighting = [1.0] * len(self.models)
        elif len(model_weighting) != len(self.models):
            raise ValueError("model_weighting must have the same length as the number of models")
        if user_ratings is None:
            user_ratings = pd.merge(self.likes_df, self.data_df, left_on="song_id", right_on="id")

        # Normalize weights
        normalized_model_weighting: list[float] = np.array(model_weighting) / sum(model_weighting)  # type: ignore

        # Get recommendations from each model
        all_recommendations: List[Tuple[str, float]] = []
        for (_, model), weight in zip(self.models.items(), normalized_model_weighting):
            model_recommendations = model.recommend(
                data=self.data_df,
                ddata=self.tdata_df,
                n=n,
                user_id=user_id,
                excluded_set=excluded_set,
                user_ratings=user_ratings,
                genre_weight=genre_weight,
                **kwargs
            )
            all_recommendations.extend(
                (item, score * weight) for item, score in model_recommendations
            )

        # Combine recommendations
        combined_recommendations = {}
        for item, score in all_recommendations:
            if excluded_set and item in excluded_set:
                continue
            if item not in combined_recommendations:
                combined_recommendations[item] = 0.0
            combined_recommendations[item] += score

        # Sort by score and return top n recommendations
        sorted_recommendations = sorted(
            combined_recommendations.items(), key=lambda x: x[1], reverse=True
        )

        rs = sorted_recommendations[:n]
        ## join them back with the original data so we can return the song names
        rec_df = pd.DataFrame(rs, columns=["user_id", "score"])
        ## convert the song ids to the song names
        rec_df = pd.merge(
            rec_df,
            self.data_df[["name", "artists", "genres", "id"]],
            left_on="user_id",
            right_on="id",
            how="left",
        )
        rec_df.drop("id", axis=1, inplace=True)
        return rec_df
