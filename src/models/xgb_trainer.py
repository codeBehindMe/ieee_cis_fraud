"""Trains xgb models"""
from neptune_xgboost.impl import NeptuneCallback
import pandas as pd
from src.constants import SEED
from typing import Any, Dict, Final, List, Optional
from src.data import Statistics
from src.utils import NeptuneUtils, XGBUtils
import xgboost as xgb
from xgboost.core import Booster
import numpy as np
from src.evaluation import evaluate_and_launch_to_neptune
from src.evaluation import curve_evals


class XGBTrainer:
    def __init__(
        self,
        max_depth: int = 2,
        eta: float = 1,
        n_thread: int = 8,
        eval_metric: List[str] = ["auc", "error", "logloss", "aucpr"],
        seed=SEED,
        num_round: int = 10,
    ) -> None:
        self.max_depth = max_depth
        self.eta = eta
        self.n_thread = n_thread
        self.eval_metric = eval_metric
        self.seed = seed
        self.num_round = num_round

        self.objective: Final[str] = "binary:logistic"
        self.label_column: Final[str] = "isFraud"
        self.decision_threshold: Final[float] = 0.5

        self.booster: Booster = None

    def _make_params_dict(self) -> Dict[str, Any]:
        params = {}
        params["max_depth"] = self.max_depth
        params["eta"] = self.eta
        params["objective"] = self.objective
        params["nthread"] = self.n_thread
        return params

    def train_evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        params = self._make_params_dict()

        d_train = XGBUtils.create_d_matrix(train_df, self.label_column)
        d_test = XGBUtils.create_d_matrix(test_df, self.label_column)

        eval_list = [(d_train, "train"), (d_test, "eval")]

        with NeptuneUtils.run() as run:
            run["data/train/columns"] = train_df.columns.values.tolist()
            run["data/test/columns"] = test_df.columns.values.tolist()
            run["data/train/row_count"] = train_df.shape[0]
            run["data/test/row_count"] = test_df.shape[0]
            run["data/train/isFraudPerc"] = Statistics.isFraud_percentage(train_df)
            run["data/test/isFraudPerc"] = Statistics.isFraud_percentage(test_df)

            run["model/flavour"] = "xgbtree"
            run["model/params"] = params
            run["model/num_round"] = self.num_round

            xgb_callback = NeptuneCallback(run, log_tree=[0, 1, 2, 3, 4])

            self.booster: Booster = xgb.train(
                params=params,
                dtrain=d_train,
                num_boost_round=self.num_round,
                callbacks=[xgb_callback],
                evals=eval_list,
            )

            train_probas = self.booster.predict(d_train)
            train_preds = np.where(
                self.booster.predict(d_train) > self.decision_threshold, 1, 0
            )
            evaluate_and_launch_to_neptune(
                run, train_df[self.label_column], train_preds, train_probas, "train"
            )

            test_probas = self.booster.predict(d_test)
            test_preds = np.where(
                self.booster.predict(d_test) > self.decision_threshold, 1, 0
            )
            evaluate_and_launch_to_neptune(
                run, test_df[self.label_column], test_preds, test_probas, "test"
            )

            # We return test_roc_auc since it's used in the kaggle competition as the score
            # thus we could use it as the optimisation objective in the future for hp tuning.
            test_roc_auc = curve_evals(test_df[self.label_column], test_probas)[
                "roc_auc"
            ]
            return test_roc_auc
