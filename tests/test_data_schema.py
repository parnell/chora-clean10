from datetime import datetime
from typing import Any

import pandas as pd
import pandera as pa
import pytest
from pandera.errors import SchemaError
from pandera.typing import Series

from chora.types.data_schema import DataSchema


class TestSchema(DataSchema):
    id: Series[str] = pa.Field()
    name: Series[str] = pa.Field()
    genres: Series[Any] = pa.Field(metadata={"list": "str"})
    release_date: Series[datetime] = pa.Field()  # type: ignore
    count: Series[float] = pa.Field()
    is_popular: Series[bool] = pa.Field()


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "name": ["Song A", "Song B", "Song C"],
            "genres": ["['pop', 'rock']", "['jazz']", "[]"],
            "release_date": ["2021-01-01", "2022", "2023-03-15"],
            "count": ["10", "20.5", "30"],
            "is_popular": [True, False, True],
        }
    )


def test_preprocess_dataframe(sample_data):
    processed_df = TestSchema.preprocess_data(sample_data)

    assert isinstance(processed_df["genres"][0], list)
    assert processed_df["genres"][0] == ["pop", "rock"]
    assert processed_df["genres"][2] == []

    assert isinstance(processed_df["release_date"][0], datetime)
    assert processed_df["release_date"][1] == datetime(2022, 1, 1)

    assert processed_df["count"].dtype == float
    assert processed_df["is_popular"].dtype == bool


def test_validate_df(sample_data):
    validated_df = TestSchema.validate(sample_data)
    assert isinstance(validated_df, pd.DataFrame)
    assert list(validated_df.columns) == [
        "id",
        "name",
        "genres",
        "release_date",
        "count",
        "is_popular",
    ]
    assert type(validated_df) == pd.DataFrame


def test_validate_df_with_invalid_data(sample_data):
    invalid_data = sample_data.copy()
    invalid_data.loc[0, "count"] = "invalid"

    with pytest.raises(SchemaError):
        TestSchema.validate(invalid_data)


def test_transform_for_ml(sample_data):
    validated_df = TestSchema.validate(sample_data)
    transformed_df = TestSchema.transform_data(validated_df)

    assert "genres_pop" in transformed_df.columns
    assert "genres_rock" in transformed_df.columns
    assert "genres_jazz" in transformed_df.columns

    assert transformed_df["genres_pop"][0] == 1
    assert transformed_df["genres_rock"][0] == 1
    assert transformed_df["genres_jazz"][1] == 1


if __name__ == "__main__":
    pytest.main([__file__])
