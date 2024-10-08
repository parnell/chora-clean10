from datetime import date, datetime
from typing import Any, ClassVar

import pandera as pa
from pandera.typing import Series

from chora.types.data_schema import DataSchema


# Define the schema
class SongDataSchema(DataSchema):
    id: Series[str] = pa.Field(description="Unique identifier for the song")
    name: Series[str] = pa.Field(description="Name of the song")
    artists: Series[str] = pa.Field(description="List of artists")
    genres: Series[Any] = pa.Field(description="List of genres", metadata={"list": "str"})
    valence: Series[float] = pa.Field(ge=0, le=1, description="Valence of the song")
    year: Series[int] = pa.Field(ge=1900, le=date.today().year, description="Year of release")
    acousticness: Series[float] = pa.Field(ge=0, le=1, description="Acousticness of the song")
    danceability: Series[float] = pa.Field(ge=0, le=1, description="Danceability of the song")
    duration_ms: Series[int] = pa.Field(ge=0, description="Duration of the song in milliseconds")
    energy: Series[float] = pa.Field(ge=0, le=1, description="Energy of the song")
    explicit: Series[bool] = pa.Field(description="Explicit content flag")
    instrumentalness: Series[float] = pa.Field(
        ge=0, le=1, description="Instrumentalness of the song"
    )
    key: Series[int] = pa.Field(ge=0, le=11, description="Musical key of the song")
    liveness: Series[float] = pa.Field(ge=0, le=1, description="Liveness of the song")
    loudness: Series[float] = pa.Field(description="Loudness of the song")
    mode: Series[bool] = pa.Field(description="Mode of the song")
    popularity: Series[int] = pa.Field(ge=0, le=100, description="Popularity score of the song")
    release_date: Series[datetime] = pa.Field(nullable=True, description="Release date of the song") # type: ignore
    speechiness: Series[float] = pa.Field(ge=0, le=1, description="Speechiness of the song")
    tempo: Series[float] = pa.Field(ge=0, description="Tempo of the song")

    _relevant_features: ClassVar[Any] = [
        "valence",
        "year",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "popularity",
        "speechiness",
        "tempo",
    ]
    _transformed_features: ClassVar[Any] = [
        "genres"
    ]
    

class RatedSongSchema(SongDataSchema):
    rating: float = pa.Field(ge=-1, le=1, description="Rating given to the song")
    user_id: str = pa.Field(description="User identifier")
    timestamp: datetime = pa.Field(description="Timestamp of the rating")

    _target: ClassVar[str] = "rating"
    _user_item: ClassVar[Any] = ["user_id", "id"]
