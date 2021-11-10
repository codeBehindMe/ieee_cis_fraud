"""Data interfacing classes"""
import os
from typing import Dict, Final, List
import pandas as pd
from pandas.core.arrays.sparse import dtype
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


class DataSet:

    SAMPLE_SUBMISSION: Final[str] = "sample_submission.csv"

    TEST_IDENTITY: Final[str] = "test_identity.csv"
    TEST_TRANSACTION: Final[str] = "test_transaction.csv"

    TRAIN_IDENTITY: Final[str] = "train_identity.csv"
    TRAIN_TRANSACTION: Final[str] = "train_transaction.csv"

    def __init__(self, path_to_data: str) -> None:
        self.path_to_data = path_to_data

    def _construct_path(self, file) -> str:
        """
        Constructs relative path
        """
        return os.path.join(self.path_to_data, file)

    def get_train_transaction(self) -> pd.DataFrame:
        """
        Returns the train transaction dataframe
        """
        return pd.read_csv(self._construct_path(DataSet.TRAIN_TRANSACTION))

    def get_train_identity(self) -> pd.DataFrame:
        """
        Returns teh train identity dataframe
        """
        return pd.read_csv(self._construct_path(DataSet.TRAIN_IDENTITY))


class Statistics:
    @staticmethod
    def isFraud_percentage(d: pd.DataFrame):
        counts = d.groupby("isFraud").size().to_dict()
        return (counts[1] / (counts[0] + counts[1])) * 100

    @staticmethod
    def isFraud_percentage_plot(**datasets: pd.DataFrame):
        ratios: Dict = {}
        for name, dataset in datasets.items():
            ratios[name] = Statistics.isFraud_percentage(dataset)

        fig = px.bar(
            pd.DataFrame.from_dict(ratios, orient="index"), title="Fraud percentage"
        )
        fig.layout.showlegend = False
        return fig


class FeatureFilters:
    @staticmethod
    def remove_cols_with_nulls(
        df: pd.DataFrame, null_perc: float = 0.0
    ) -> pd.DataFrame:
        """
        Removes columns whose data contains nulls above a certain threshold.
        """

        if null_perc < 0 or null_perc > 1:
            raise ValueError(
                f"null perc needs to be between 0 and 1, recieved {null_perc}"
            )

        df_na_perc: pd.Series = df.isna().sum(axis=0) / df.shape[0]

        keep_columns = df_na_perc[(df_na_perc <= null_perc)].index.tolist()

        return df[keep_columns]


class FeatureEncoders:
    @staticmethod
    def encode_string_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes string columns in dataframe.
        """

        _df = df.copy(deep=True)

        type_cast_cols = []
        for c, t in _df.dtypes.to_dict().items():
            if t == "object":
                _df[c].update(pd.Series(LabelEncoder().fit_transform(_df[c])))
                type_cast_cols.append(c)

        _df = _df.astype({x: "int" for x in type_cast_cols})

        return _df
