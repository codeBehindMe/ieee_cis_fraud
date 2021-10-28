"""Contains utilities of various kinds"""
from typing import Optional
import xgboost as xgb
import pandas as pd


class XGBUtils:
    @staticmethod
    def create_d_matrix(
        d: pd.DataFrame, label_column: Optional[str] = None
    ) -> xgb.DMatrix:
        """
        Creates a xgboost device memory data matrix object from a pandas dataframe
        """
        if label_column is not None:
            return xgb.DMatrix(d.drop([label_column], axis=1), label=d[label_column])
        return xgb.DMatrix(d)
