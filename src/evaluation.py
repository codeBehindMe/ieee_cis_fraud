from typing import Dict
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
)
import plotly.express as px
from src.utils import NeptuneUtils


def scalar_evals(y_true, y_pred):
    """
    Evaluate scalar metrics
    """

    pack: Dict = {}

    pack["precision"] = precision_score(y_true, y_pred)
    pack["recall"] = recall_score(y_true, y_pred)
    pack["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    pack["f1"] = f1_score(y_true, y_pred)

    return pack


def curve_evals(y_true, probas):
    """
    Evaluate and plot curve metrics
    """

    pack: Dict = {}

    p, r, t = precision_recall_curve(y_true, probas)
    prc_df = pd.DataFrame()
    prc_df["precision"] = p
    prc_df["recall"] = r

    prc_auc = auc(r, p)
    pack["pr_curve"] = px.scatter(
        prc_df,
        x="recall",
        y="precision",
        title=f"Precision Recall Curveu: AUC:{prc_auc}",
    )
    pack["prc_auc"] = prc_auc

    fpr, tpr, t = roc_curve(y_true, probas)
    roc_df = pd.DataFrame()
    roc_df["fpr"] = fpr
    roc_df["tpr"] = tpr
    roc_df["threshold"] = t

    roc_auc = auc(fpr, tpr)
    pack["roc_curve"] = px.scatter(
        roc_df,
        x="fpr",
        y="tpr",
        color="threshold",
        title=f"Receiver operating characteristic. AUC:{roc_auc}",
        range_color=[0, 1],
    )
    pack["roc_auc"] = roc_auc

    return pack


def plot_confusion_matrix(cfm: np.array):
    """
    Plot confusion matrix
    """

    return ff.create_annotated_heatmap(
        cfm, x=["Predicted Not Fraud", "Predicted Fraud"], y = ["True Not Fraud", "True Fraud"]
    )

def evaluate_and_launch_to_neptune(run, y_true, y_pred, probas, set_name):
    """
    Evaluate a set and upload to neptune
    """
    scalar_results = scalar_evals(y_true, y_pred)
    plot_results = curve_evals(y_true, probas)

    for k, v in scalar_results.items():

        if k == "confusion_matrix":
            NeptuneUtils.upload_plotly(run, f"{set_name}/{k}", plot_confusion_matrix(v))
        else:
            run[f"{set_name}/{k}"] = v

    for k, v in plot_results.items():

        if k in ("prc_auc", "roc_auc"):
            run[f"{set_name}/{k}"] = v
        else:
            NeptuneUtils.upload_plotly(run, f"{set_name}/{k}", v)
