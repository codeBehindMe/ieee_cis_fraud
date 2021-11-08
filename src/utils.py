"""Contains utilities of various kinds"""
from typing import Optional
import xgboost as xgb
import pandas as pd
import neptune.new as neptune
from contextlib import contextmanager
from src.constants import NEPTUNE_PROJECT


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


class NeptuneUtils:
    @staticmethod
    def upload_plotly(run, name, fig):
        """
        Uploads a plotly figure to neptune
        """

        return run[name].upload(neptune.types.File.as_html(fig))

    @staticmethod
    @contextmanager
    def run(token_path: str = "."):
        """
        Creates a a neptune run context
        """

        with open(token_path, "r") as f:
            neptune_token = f.readline()

        run = neptune.init(project=NEPTUNE_PROJECT, api_token=neptune_token)
        yield run
        run.stop()
