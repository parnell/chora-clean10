import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
from river import base, dummy, metrics, optim, preprocessing, reco, stats
from river.evaluate import progressive_val_score

from chora.models.abstract_model import AbstractModel
from chora.types.song_data import RatedSongSchema, SongDataSchema


@dataclass
class OnlineModel:
    model: base.Regressor = field(init=False)

    def evaluate_and_train(
        self, data_generator: Iterable[Tuple[Dict[str, Any], float]]
    ) -> base.Regressor:
        metric = metrics.MAE() + metrics.RMSE()
        progressive_val_score(
            data_generator,
            self.model,
            metric,
            print_every=25000,
            show_time=True,
            show_memory=True,
        )
        return self.model


class NaiveModel(OnlineModel):
    def __init__(self) -> None:
        self.model = dummy.StatisticRegressor(stats.Mean())


@dataclass
class BiasedMFModel(AbstractModel):
    data_df: Optional[pd.DataFrame] = None
    n_factors: int = 10
    bias_optimizer: Any = field(default_factory=lambda: optim.SGD(0.05))
    latent_optimizer: Any = field(default_factory=lambda: optim.SGD(0.1))
    weight_initializer: Any = field(
        default_factory=lambda: optim.initializers.Normal(mu=0.0, sigma=0.1, seed=73)
    )
    latent_initializer: Any = field(
        default_factory=lambda: optim.initializers.Normal(mu=0.0, sigma=0.1, seed=73)
    )
    l2_bias: float = 0.005
    l2_latent: float = 0.005
    learn_count: int = 10
    metric: Any = field(init=False)
    model: Any = field(init=False)

    def __post_init__(self):
        self.metric = metrics.MAE() + metrics.RMSE()
        reg = reco.BiasedMF(
            n_factors=self.n_factors,
            bias_optimizer=self.bias_optimizer,
            latent_optimizer=self.latent_optimizer,
            weight_initializer=self.weight_initializer,
            latent_initializer=self.latent_initializer,
            l2_bias=self.l2_bias,
            l2_latent=self.l2_latent,
        )

        self.model = preprocessing.PredClipper(
            regressor=cast(base.Regressor, reg), y_min=-1, y_max=1
        )

    def train(self, data: pd.DataFrame) -> None:
        print(data.columns)
        gen = RatedSongSchema.iterate(data, with_y=True, with_user_item=True)
        progressive_val_score(
            gen,
            self.model,
            self.metric,
            print_every=25000,
            show_time=True,
            show_memory=True,
        )

    def learn_one(
        self, user_id: str, song: SongDataSchema, rating: float, prev_rating: float
    ) -> None:

        X = cast(pd.DataFrame, song).to_dict()

        # Learn multiple times to reinforce the learning
        for _ in range(self.learn_count):
            self.model.learn_one(  # type: ignore
                x=X,
                y=rating,  # type: ignore
                user=user_id,  # type: ignore
                item=song.id,  # type: ignore
            )
        predicted_rating = self.model.predict_one(X, user=user_id, item=song.id)  # type: ignore
        self.metric.update(y_true=rating, y_pred=predicted_rating)

    @staticmethod
    def diverse_recommend(biased_model, user_id, data, n, excluded_set, exploration_rate=0.2):
        # Get more recommendations than needed
        recommendations = biased_model.recommend(
            user_id=user_id,
            data=data,
            n=int(n * 1.5),  # Get 50% more recommendations
            excluded_set=excluded_set,
        )

        # Split into exploit and explore sets
        exploit_count = int(n * (1 - exploration_rate))
        explore_count = n - exploit_count

        exploit_set = recommendations[:exploit_count]
        explore_set = random.sample(recommendations[exploit_count:], explore_count)

        return exploit_set + explore_set

    def recommend(
        self,
        n: int,
        data: Optional[pd.DataFrame],
        user_id: Optional[str] = None,
        excluded_set: Optional[Set[str]] = None,
        user_ratings: Optional[pd.DataFrame] = None,
        genre_weight: Optional[float] = None,
        sample_size: Optional[int] = 10000,
        **kwargs
    ) -> List[Tuple[str, float]]:
        excluded_set = excluded_set or set()
        if data is None:
            data = self.data_df
        assert data is not None

        # Convert data to numpy arrays for faster computation
        song_ids = data["id"].values
        X = data.drop("id", axis=1).to_numpy()

        # Filter out excluded songs using boolean indexing
        mask = ~np.isin(song_ids, list(excluded_set))  # type: ignore
        song_ids = song_ids[mask]
        X = X[mask]

        # Optional: Sampling to reduce the number of predictions
        if sample_size is not None and len(song_ids) > sample_size:
            sample_indices = np.random.choice(len(song_ids), sample_size, replace=False)
            song_ids = song_ids[sample_indices]
            X = X[sample_indices]

        # Predict ratings using a list comprehension
        predicted_ratings = np.array(
            [
                self.model.predict_one(x, user=user_id, item=song_id)  # type: ignore
                for x, song_id in zip(X, song_ids)
            ]
        )

        # Use argpartition for efficient top-k selection
        top_n_indices = np.argpartition(predicted_ratings, -n)[-n:]
        top_n_indices = top_n_indices[np.argsort(-predicted_ratings[top_n_indices])]

        return list(zip(song_ids[top_n_indices], predicted_ratings[top_n_indices]))

    def fit(self, X: pd.DataFrame, **kwargs: Any) -> None:
        pass
