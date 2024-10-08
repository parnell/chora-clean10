from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

VectorsType = pd.DataFrame | list[list[Any]] | np.ndarray | list[Any]


class AbstractModel(ABC):
    @abstractmethod
    def recommend(
        self,
        n: int,
        data: Optional[pd.DataFrame],
        user_id: Optional[str] = None,
        excluded_set: Optional[set[str]] = None,
        user_ratings: Optional[pd.DataFrame] = None,
        genre_weight: Optional[float] = None,
        sample_size: Optional[int] = 10000,
        **kwargs
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, **kwargs: Any) -> None: ...