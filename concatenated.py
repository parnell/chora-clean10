### chora/recommender.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from chora.ingestion.data_loader import DataLoader
from chora.models.abstract_model import AbstractModel
from chora.types.music_model import SongData


@dataclass
class RecommenderService:
    models: List[AbstractModel]
    data_loader: DataLoader

    def __post_init__(self):
        self.data_loader.load_data()
        self._precompute_data()

    def _precompute_data(self):
        # Prepare a DataFrame with all songs and their features
        all_songs = []
        for song_id, song_info in self.data_loader.song_map.items():
            song_data = {
                "song_id": song_id,
                "genres": ",".join(song_info.genres),
                **{
                    feature: getattr(song_info, feature)
                    for feature in SongData.get_audio_features()
                },
            }
            all_songs.append(song_data)

        all_songs_df = pd.DataFrame(all_songs)

        # Precompute data for each model
        for model in self.models:
            model.precompute(all_songs_df)

    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        exclude: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        exclude_rated: bool = False,
    ) -> pd.DataFrame:
        user_ratings = self.data_loader.get_user_ratings(user_id)
        if user_ratings.empty:
            raise ValueError(f"No ratings found for user ID: {user_id}")

        excluded_set = set(exclude or [])
        if exclude_rated:
            excluded_set.update(user_ratings["song_id"])

        if weights is None:
            weights = [1.0] * len(self.models)
        elif len(weights) != len(self.models):
            raise ValueError("The number of weights must match the number of models.")

        all_recommendations: List[tuple[str, float]] = []

        for model, weight in zip(self.models, weights):
            model_recommendations = model.recommend(
                user_ratings,
                n=top_n * 2,
                excluded_set=excluded_set,
            )
            all_recommendations.extend(
                (song_id, score * weight) for song_id, score in model_recommendations
            )

        # Aggregate scores for each song
        song_scores: Dict[str, float] = {}
        for song_id, score in all_recommendations:
            song_scores[song_id] = song_scores.get(song_id, 0) + score

        # Sort songs based on aggregated scores in descending order
        sorted_songs = sorted(song_scores.items(), key=lambda item: item[1], reverse=True)

        # Select the top_n song IDs
        recommended_song_ids = [song_id for song_id, _ in sorted_songs[:top_n]]

        # Retrieve SongData for recommended songs
        recommended_songs = [
            self.data_loader.get_song_info(song_id) for song_id in recommended_song_ids
        ]

        # Convert the list of SongData objects to a DataFrame, excluding any None entries
        recommended_df = SongData.to_pandas_df(
            [song for song in recommended_songs if song is not None]
        )

        return recommended_df

    def add_model(self, model: AbstractModel) -> None:
        """
        Add a new model to the recommender service.

        Args:
            model (AbstractModel): The model to add.
        """
        self.models.append(model)

    def train(self, data: pd.DataFrame | str) -> None:
        """
        Train all the models provided to the service.
        """
        for model in self.models:
            model.train(data)

    def update_model_with_rating(self, user_id: str, song_id: str, rating: float) -> None:
        rating = max(1, min(5, rating))
        song = self.data_loader.get_song_info(song_id)
        for model in self.models:
            model.learn_one(user_id, song, rating)
        # song_info = self.merged_df[self.merged_df["id"] == song_id].iloc[0]
        # x = {feature: float(song_info[feature]) for feature in self.audio_features}
        # x["genres"] = DataLoader._process_genres(song_info["genres"])
        # x["user"] = user_id
        # x["item"] = song_id
        # self.model.learn_one(x, rating)

### chora/ingestion/data_loader.py

# data_loader.py

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from chora import DATA_DIR
from chora.types.music_model import SongData

RECOMMENDATIONS_FILE = f"{DATA_DIR}/generated/recommendations.csv"
MUSIC_FILE = f"{DATA_DIR}/merged_music_data.csv"


@dataclass
class DataLoader:
    recommendations_file: str = RECOMMENDATIONS_FILE
    music_file: str = MUSIC_FILE
    merged_df: Optional[pd.DataFrame] = None
    recommendations_df: Optional[pd.DataFrame] = None
    songs: Optional[List[SongData]] = None  # Cached SongData instances
    song_map: Optional[Dict[str, SongData]] = None  # Cached mapping
    mlb_genres: MultiLabelBinarizer = field(default_factory=lambda: MultiLabelBinarizer(sparse_output=False))
    
    # New configuration options
    drop_na: bool = False
    fillna_with_empty_lists: List[str] = field(default_factory=list)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.merged_df is None or self.recommendations_df is None:
            self.merged_df = pd.read_csv(self.music_file)
            self.recommendations_df = pd.read_csv(self.recommendations_file)
            self.preprocess_data()
            # Convert merged_df to SongData instances once and cache
            self.songs = SongData.from_pandas_df(self.merged_df)
            # Create and cache the song_map
            self.song_map = {song.id: song for song in self.songs}
            # Fit the MultiLabelBinarizer on all genres
            all_genres = [song.genres for song in self.songs if song.genres]
            self.mlb_genres.fit(all_genres)
        return self.merged_df, self.recommendations_df

    def preprocess_data(self) -> None:
        if self.merged_df is not None:
            # Handle NaN values for list-type columns
            for column in self.fillna_with_empty_lists:
                if column in self.merged_df.columns:
                    self.merged_df[column] = self.merged_df[column].apply(
                        lambda x: x if isinstance(x, list) else [] if pd.isna(x) else x
                    )
                else:
                    print(f"Warning: Column '{column}' not found in merged_df for fillna_with_empty_lists.")

            # Optionally drop rows with any NaN values
            if self.drop_na:
                self.merged_df.dropna(inplace=True)
                print("Dropped rows with NaN values from merged_df.")

        if self.recommendations_df is not None:
            # Handle NaN values for list-type columns in recommendations_df if needed
            for column in self.fillna_with_empty_lists:
                if column in self.recommendations_df.columns:
                    self.recommendations_df[column] = self.recommendations_df[column].apply(
                        lambda x: x if isinstance(x, list) else [] if pd.isna(x) else x
                    )
                else:
                    print(f"Warning: Column '{column}' not found in recommendations_df for fillna_with_empty_lists.")

            # Optionally drop rows with any NaN values
            if self.drop_na:
                self.recommendations_df.dropna(inplace=True)
                print("Dropped rows with NaN values from recommendations_df.")

            # Map ratings as before
            self.recommendations_df["rating"] = self.recommendations_df["rating"].map(
                {-1: 1, 0: 3, 1: 5}
            )

    def get_user_ratings(self, user_id: str) -> pd.DataFrame:
        if self.recommendations_df is not None:
            return self.recommendations_df[self.recommendations_df["user_id"] == user_id]
        return pd.DataFrame()

    def get_song_info(self, song_id: str) -> Optional[SongData]:
        """
        Retrieve SongData instance for a given song_id.
        """
        if self.song_map is not None:
            return self.song_map.get(song_id)
        return None

    def get_audio_features(self) -> List[str]:
        """
        Dynamically retrieve audio feature fields from SongData based on category.
        """
        return SongData.get_audio_features()

    def create_feature_matrix(self) -> pd.DataFrame:
        """
        Create a DataFrame containing all feature vectors for modeling.
        This includes audio features and encoded genres.
        """
        feature_dicts = []
        for song in self.songs:
            feature_vector = self.create_feature_vector(song)
            feature_dicts.append(feature_vector)
        feature_df = pd.DataFrame(feature_dicts)
        return feature_df

    def create_feature_vector(self, song: SongData) -> Dict[str, Any]:
        """
        Create a feature dictionary from a SongData instance based on audio features and encoded genres.
        """
        audio_features = self.get_audio_features()
        feature_vector: Dict[str, Any] = {}
        for feature in audio_features:
            value = getattr(song, feature, None)
            if value is None:
                feature_vector[feature] = 0.0
            elif isinstance(value, (int, float)):
                feature_vector[feature] = float(value)
            else:
                try:
                    feature_vector[feature] = float(value)
                except (ValueError, TypeError):
                    print(f"Warning: Unexpected type for feature '{feature}': {value}")
                    feature_vector[feature] = 0.0

        # Encode genres
        if song.genres:
            # Cast the transformed data to np.ndarray to assist the type checker
            transformed = cast(np.ndarray, self.mlb_genres.transform([song.genres]))
            encoded_genres = transformed[0].tolist()
            for i, genre in enumerate(self.mlb_genres.classes_):
                feature_vector[f"genre_{genre}"] = float(encoded_genres[i])
        else:
            # If no genres, initialize all genre features to 0
            for genre in self.mlb_genres.classes_:
                feature_vector[f"genre_{genre}"] = 0.0

        return feature_vector

    # def get_features_and_targets(self, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Retrieve the feature matrix and target vector for modeling.

    #     Args:
    #         target_column (str): The name of the target column in the recommendations_df.

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: Feature matrix X and target vector y.
    #     """
    #     # Align recommendations with features
    #     # Assuming 'song_id' in recommendations_df corresponds to 'id' in merged_df
    #     merged_with_recommendations = self.recommendations_df.merge(
    #         self.merged_df, left_on="song_id", right_on="id", how="left"
    #     )
    #     # Drop rows with missing song data
    #     merged_with_recommendations = merged_with_recommendations.dropna(subset=["id"])
    #     # Create feature matrix
    #     X = self.create_feature_matrix_from_recommendations(merged_with_recommendations)
    #     # Create target vector
    #     y = merged_with_recommendations[target_column].values
    #     return X, y

    # def create_feature_matrix_from_recommendations(
    #     self, recommendations_df: pd.DataFrame
    # ) -> np.ndarray:
    #     """
    #     Create a feature matrix based on recommendations DataFrame.

    #     Args:
    #         recommendations_df (pd.DataFrame): DataFrame containing recommendations.

    #     Returns:
    #         np.ndarray: Feature matrix.
    #     """
    #     feature_dicts = []
    #     for _, row in recommendations_df.iterrows():
    #         song_id = row["song_id"]
    #         song = self.get_song_info(song_id)
    #         if song:
    #             feature_vector = self.create_feature_vector(song)
    #             # Include additional metadata if needed
    #             feature_vector.update(
    #                 {
    #                     "user": str(row["user_id"]),
    #                     "item": str(song.id),
    #                 }
    #             )
    #             feature_dicts.append(feature_vector)
    #         else:
    #             print(f"Warning: Song ID {song_id} not found in merged data.")
    #     feature_df = pd.DataFrame(feature_dicts)
    #     # Optionally, drop non-feature columns like 'user' and 'item' if not needed
    #     feature_columns = [
    #         col
    #         for col in feature_df.columns
    #         if not col.startswith("user") and not col.startswith("item")
    #     ]
    #     return feature_df[feature_columns].values

    def save_transformed_data(self, feature_matrix: pd.DataFrame, filepath: str) -> None:
        """
        Save the transformed feature matrix to a CSV file.

        Args:
            feature_matrix (pd.DataFrame): The transformed feature matrix.
            filepath (str): The path to save the CSV file.
        """
        feature_matrix.to_csv(filepath, index=False)
        print(f"Transformed data saved to {filepath}")

    def load_transformed_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the transformed feature matrix from a CSV file.

        Args:
            filepath (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded feature matrix.
        """
        feature_matrix = pd.read_csv(filepath)
        print(f"Transformed data loaded from {filepath}")
        return feature_matrix

### chora/types/music_model.py

import ast
from datetime import date
from typing import Any, ClassVar, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class SongData(BaseModel):
    # --------------------- Model Fields ---------------------
    id: str = Field(..., description="Unique identifier for the song")
    name: str = Field(..., description="Name of the song")
    artists: List[str] = Field(..., description="List of artists")
    genres: List[str] = Field(default_factory=list, description="List of genres")
    valence: float = Field(..., description="Valence of the song")
    year: int = Field(..., description="Year of release")
    acousticness: float = Field(..., description="Acousticness of the song")
    danceability: float = Field(..., description="Danceability of the song")
    duration_ms: int = Field(..., description="Duration of the song in milliseconds")
    energy: float = Field(..., description="Energy of the song")
    explicit: bool = Field(..., description="Explicit content flag")
    instrumentalness: float = Field(..., description="Instrumentalness of the song")
    key: int = Field(..., description="Musical key of the song")
    liveness: float = Field(..., description="Liveness of the song")
    loudness: float = Field(..., description="Loudness of the song")
    mode: bool = Field(..., description="Mode of the song")
    popularity: int = Field(..., description="Popularity score of the song")
    release_date: Optional[date] = Field(None, description="Release date of the song")
    speechiness: float = Field(..., description="Speechiness of the song")
    tempo: float = Field(..., description="Tempo of the song")
    count: Optional[int] = Field(None, description="Additional count metric")

    # ------------------- Category Definitions -------------------
    # Define audio feature fields as a class-level variable
    audio_feature_fields: ClassVar[List[str]] = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ]

    # --------------------- Validators ---------------------
    @field_validator("artists", mode="before")
    @classmethod
    def parse_artists(cls, v):
        if isinstance(v, str):
            return [artist.strip() for artist in v.split(",")]
        return v

    @field_validator("genres", mode="before")
    @classmethod
    def parse_genres(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v or v in {"[]", '""'}:
                return []
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return [genre.strip() for genre in v.split(",")]
        elif isinstance(v, list):
            return v
        return []

    @field_validator("explicit", "mode", mode="before")
    @classmethod
    def parse_boolean_fields(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.isdigit():
                return bool(int(v))
        return bool(v)

    @field_validator("release_date", mode="before")
    @classmethod
    def parse_release_date(cls, v):
        if isinstance(v, str):
            v = v.strip()
            try:
                return date.fromisoformat(v)
            except ValueError:
                try:
                    return date(int(v), 1, 1)
                except ValueError:
                    return None
        return None

    @field_validator("count", mode="before")
    @classmethod
    def parse_count(cls, v):
        if v in {None, "", "null"}:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    @model_validator(mode="before")
    def check_year(cls, values):
        if values.get("year") is None:
            raise ValueError("Year must be provided")
        return values

    # ------------------- Helper Functions -------------------
    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame) -> List["SongData"]:
        """
        Create a list of SongData instances from a pandas DataFrame.
        Utilizes vectorized operations and batch processing for efficiency.
        """
        # Dynamically retrieve field names from the model
        song_columns = list(cls.model_fields.keys())

        # Ensure all necessary columns are present
        missing_cols = set(song_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns for SongData: {missing_cols}")

        # Replace NaN with None for proper handling in Pydantic
        df = df[song_columns].where(pd.notnull(df[song_columns]), None)

        # Convert DataFrame to dictionaries
        song_dicts = df.to_dict(orient="records")

        # Batch create SongData instances
        songs = []
        for song_dict in song_dicts:
            try:
                song = cls(**{str(k): v for k, v in song_dict.items()})
                songs.append(song)
            except Exception as e:
                song_id = song_dict.get("id", "N/A")
                print(f"Error parsing song ID {song_id}: {e}")
        return songs

    @staticmethod
    def to_pandas_df(songs: List["SongData"]) -> pd.DataFrame:
        """
        Convert a list of SongData instances to a pandas DataFrame.
        """
        # Serialize each SongData instance to a dictionary
        songs_dicts = [song.model_dump() for song in songs]

        # Convert date objects to ISO format strings for compatibility
        for song_dict in songs_dicts:
            if isinstance(song_dict.get("release_date"), date):
                song_dict["release_date"] = song_dict["release_date"].isoformat()

        # Create DataFrame
        df = pd.DataFrame(songs_dicts)

        # Ensure the DataFrame has the correct column order based on the model
        columns_order = list(SongData.model_fields.keys())
        df = df[columns_order]
        return df

    @staticmethod
    def from_numpy_ndarray(ndarray: np.ndarray, columns: List[str]) -> List["SongData"]:
        """
        Create a list of SongData instances from a NumPy ndarray.
        """
        df = pd.DataFrame(ndarray, columns=columns)
        return SongData.from_pandas_df(df)

    @staticmethod
    def to_numpy_ndarray(songs: List["SongData"]) -> np.ndarray:
        """
        Convert a list of SongData instances to a NumPy ndarray.
        """
        df = SongData.to_pandas_df(songs)
        return df.to_numpy()

    @classmethod
    def get_audio_features(cls) -> List[str]:
        return cls.audio_feature_fields


### chora/models/clustering.py

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from chora.models.abstract_model import AbstractModel, VectorsType


@dataclass
class ClusteringModel(AbstractModel):
    """
    ClusteringModel utilizes K-Means clustering to group similar data points based on provided features.
    It also provides optional t-SNE embeddings for visualization purposes.
    """

    # Class constants
    RANDOM_STATE: ClassVar[int] = 42
    DEFAULT_N_CLUSTERS: ClassVar[int] = 20

    # Instance variables
    n_clusters: int = DEFAULT_N_CLUSTERS
    pipeline: Pipeline = field(init=False)
    tsne_pipeline: Optional[Pipeline] = field(default=None, init=False)
    clusters: Optional[np.ndarray] = field(default=None, init=False)
    tsne_embeddings: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Initializes the clustering pipeline with appropriate preprocessors and KMeans.
        """
        # Initialize placeholders; actual preprocessing will be determined during training
        self.pipeline = Pipeline([])  # Will be set in train method

    def _initialize_pipeline(self, X: pd.DataFrame) -> None:
        """
        Initializes the pipeline based on the data's feature types.

        Args:
            X (pd.DataFrame): The input dataframe containing various features.
        """
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        transformers = []

        if numeric_features:
            transformers.append(("num", StandardScaler(), numeric_features))

        if categorical_features:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            )

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # Define the pipeline
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("kmeans", KMeans(n_clusters=self.n_clusters, random_state=self.RANDOM_STATE)),
            ]
        )

    def train(self, X: VectorsType, **kwargs: Any) -> None:
        """
        Trains the K-Means clustering model on the provided feature matrix.

        Args:
            X (VectorsType): Preprocessed feature matrix. Can be a pandas DataFrame,
                             list of lists, or a NumPy ndarray.
            **kwargs (Any): Additional keyword arguments (not used).
        """
        # Ensure X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("Input dataframe X is empty.")

        # Initialize the pipeline based on the data's feature types
        self._initialize_pipeline(X)

        # Fit the pipeline to the data
        self.pipeline.fit(X)

        # Predict cluster assignments for the training data
        self.clusters = self.pipeline.predict(X)

        # Optionally perform t-SNE for visualization
        # For t-SNE, we need to transform the data first
        transformed_X = self.pipeline.named_steps["preprocessor"].transform(X)

        self.tsne_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("tsne", TSNE(n_components=2, random_state=self.RANDOM_STATE, init="random")),
            ]
        )
        self.tsne_embeddings = self.tsne_pipeline.fit_transform(transformed_X)

    def predict(self, X: VectorsType) -> np.ndarray:
        """
        Predicts cluster assignments for new data points.

        Args:
            X (VectorsType): Preprocessed feature matrix for new data.
                             Must match the feature dimensions used during training.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        try:
            # This will raise NotFittedError if the pipeline is not fitted
            check_is_fitted(self.pipeline.named_steps["kmeans"], "cluster_centers_")
        except NotFittedError:
            raise AttributeError(
                "Model has not been trained yet. Call 'train' with appropriate data first."
            )

        return self.pipeline.predict(X)

    def get_cluster_assignments(self) -> Optional[np.ndarray]:
        """
        Retrieves the cluster assignments for the training data.

        Returns:
            Optional[np.ndarray]: Cluster labels for the training data, or None if not available.
        """
        return self.clusters

    def get_tsne_embeddings(self) -> Optional[np.ndarray]:
        """
        Retrieves the t-SNE embeddings for the training data.

        Returns:
            Optional[np.ndarray]: 2D t-SNE embeddings, or None if not available.
        """
        return self.tsne_embeddings

    def __getstate__(self) -> Dict[str, Any]:
        """
        Customizes the state for pickling. Excludes the t-SNE pipeline to avoid serialization issues.

        Returns:
            Dict[str, Any]: The state dictionary without the t-SNE pipeline.
        """
        state = self.__dict__.copy()
        state.pop("tsne_pipeline", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores the state from the unpickled state dictionary. Reinitializes the t-SNE pipeline.

        Args:
            state (Dict[str, Any]): The unpickled state dictionary.
        """
        self.__dict__.update(state)
        self.tsne_pipeline = None

    def save(self, filepath: str) -> None:
        import joblib

        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClusteringModel":
        import joblib

        return joblib.load(filepath)
