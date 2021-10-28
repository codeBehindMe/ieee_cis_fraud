"""Data interfacing classes"""
import os
from typing import Final, List
import pandas as pd


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
    def set_is_fraud_ratio(*sets: List[pd.DataFrame]):
        for i, s in enumerate(sets):
            pass
            