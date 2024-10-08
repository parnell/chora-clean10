import ast
from datetime import date, datetime
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series
from pandera.typing.common import DataFrameBase
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from chora import DATA_DIR

T = TypeVar("T", bound=pd.DataFrame)


class ListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str):
        self.column = column
        self.mlb = MultiLabelBinarizer(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Self:
        # Fit the MultiLabelBinarizer to the specified column
        # Handle empty lists and convert to list if necessary
        series = X[self.column].apply(lambda x: x if isinstance(x, list) else [])
        self.mlb.fit(series)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Apply MultiLabelBinarizer to the specified column
        # Handle empty lists and convert to list if necessary
        series = X[self.column].apply(lambda x: x if isinstance(x, list) else [])
        matrix = self.mlb.transform(series)
        df = pd.DataFrame(matrix, columns=self.mlb.classes_, index=X.index)  # type: ignore
        return df

    def get_feature_names_out(self, input_features=None):
        # Return the feature names for the transformed columns
        return [f"{cls}" for cls in self.mlb.classes_]


# Function to convert date strings to datetime
def _custom_date_parser(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        # If conversion fails, assume it's a year and return January 1st of that year
        return pd.to_datetime(date_str + "-01-01")


class DataSchema(pa.DataFrameModel):
    """Parent class to handle common type conversions and preprocessing."""

    _relevant_features: ClassVar[Any] = None
    _target: ClassVar[Any] = None
    _user_item: ClassVar[Any] = None

    @classmethod
    def __pre_checks__(cls):
        # Remove _relevant_features from the schema fields
        cls.__fields__.pop("_relevant_features", None)

    @classmethod
    def preprocess_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess DataFrame by handling common type conversions based on type hints and metadata.
        """
        # Get type hints from the schema
        type_hints = get_type_hints(cls)

        # Convert Pandera schema to access the field metadata
        schema = cls.to_schema()

        # Iterate through type hints and field metadata to apply conversions dynamically
        for column, column_type in type_hints.items():
            if column in df.columns:
                # Handle DateTime conversion
                if column_type == Series[datetime]:  # type: ignore
                    df[column] = df[column].apply(_custom_date_parser)
                # Handle list conversion
                elif hasattr(schema.columns.get(column), "metadata"):
                    # Get field metadata from the schema
                    field_metadata = schema.columns[column].metadata

                    # Check if field_metadata is not None
                    if field_metadata and field_metadata.get("list") == "str":
                        df[column] = df[column].apply(
                            lambda x: (
                                ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
                            )
                        )

                    # Handle boolean conversion
                    if column_type == Series[bool]:
                        df[column] = df[column].astype(bool)

                    # Handle float conversions for numeric fields like "count"
                    if column_type == Series[float] and column == "count":
                        df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    @classmethod
    def validate_df(
        cls: Type[Self],
        check_obj: object,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        obj = cls.validate(  # type: ignore
            check_obj=check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )
        return cast(pd.DataFrame, obj)

    @classmethod
    def validate(  # type: ignore
        cls: Type[Self],
        check_obj: object,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> T:  # type: ignore
        """
        Validate DataFrame after applying preprocessing steps.
        """
        df = cast(pd.DataFrame, check_obj)
        df = cls.preprocess_data(df)
        return super().validate(  # type: ignore
            df,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @classmethod
    def transform_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        transformers = []
        passthrough_columns = []

        type_hints = get_type_hints(cls)
        schema = cls.to_schema()

        for column, column_type in type_hints.items():
            if column in schema.columns:
                field_metadata = schema.columns[column].metadata
                if field_metadata and field_metadata.get("list") == "str":
                    if column not in df.columns:
                        print(
                            f"Warning: Column '{column}' not found in DataFrame. Available columns: {df.columns.tolist()}"
                        )
                        continue
                    transformers.append((column, ListTransformer(column), [column]))
                else:
                    passthrough_columns.append(column)

        if not transformers:
            print("Warning: No list columns found for transformation.")
            return df

        column_transformer = ColumnTransformer(
            transformers=[("passthrough", "passthrough", passthrough_columns)] + transformers,
            remainder="drop",
        )

        try:
            transformed_df = column_transformer.fit_transform(df)
            feature_names = cls._get_feature_names(column_transformer)
            return pd.DataFrame(transformed_df, columns=feature_names, index=df.index)  # type: ignore
        except Exception as e:
            print(f"Error during transformation: {str(e)}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame info:")
            df.info()
            raise

    @staticmethod
    def _get_feature_names(column_transformer):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers_:
            if name == "passthrough":
                feature_names.extend(columns)
            elif isinstance(transformer, ListTransformer):
                # Use only the column name as prefix, without repeating it
                feature_names.extend(
                    [f"{name}_{cls}" for cls in transformer.get_feature_names_out()]
                )
            else:
                feature_names.extend(transformer.get_feature_names_out(columns))
        return feature_names

    @classmethod
    def iterate(
        cls,
        df: pd.DataFrame,
        with_y: bool = False,
        with_user_item: bool = False,
        with_row: bool = False,
    ) -> Any:
        # Ensure relevant features exist in the DataFrame
        for feature in cls._relevant_features:
            if feature not in df.columns:
                raise ValueError(f"Missing expected column: {feature}")

        # Get type hints for the relevant features
        type_hints = get_type_hints(cls)

        # Iterator[Dict[str, Any]]
        for _, row in df.iterrows():

            x = {}
            for feature in cls._relevant_features:
                # Get the expected type for the feature
                expected_type = type_hints.get(feature)

                # Handle conversion based on expected type
                if expected_type == Series[float]:  # type: ignore
                    x[feature] = float(row[feature]) if pd.notna(row[feature]) else None  # type: ignore
                elif expected_type == Series[int]:  # type: ignore
                    x[feature] = int(row[feature]) if pd.notna(row[feature]) else None  # type: ignore
                elif expected_type == Series[bool]:  # type: ignore
                    x[feature] = bool(row[feature]) if pd.notna(row[feature]) else None
                else:
                    # For other types, assign directly
                    x[feature] = row[feature]
            ret: list[Any] = [x]
            if with_y:
                if not cls._target or cls._target not in df.columns:
                    raise ValueError("Missing target column in DataFrame.")
                y = row[cls._target]
                ret.append(y)
            if with_user_item:
                ui = {feature: row[feature] for feature in cls._user_item}
                ret.append(ui)
            if with_row:
                ret.append(row)
            yield tuple(ret)
