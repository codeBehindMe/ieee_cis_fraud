#%%
from xgboost.training import train
from src.data import DataSet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.constants import SEED
from src.data import Statistics
from src.evaluation import scalar_evals, curve_evals, evaluate_and_launch_to_neptune
from src.utils import XGBUtils, NeptuneUtils
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback
from src.data import FeatureFilters
from src.data import FeatureEncoders

# %%
data_set = DataSet("data")

# %%
train_data = data_set.get_train_transaction()

# %%
Statistics.isFraud_percentage_plot(train=train_data)
# %%
model_columns = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "isFraud",
]
# %%
# Filter columns  based on number of NAs. Let's pick everything that has NA ratio less than 15%
train_subset = FeatureFilters.remove_cols_with_nulls(train_data,0.15)

train_subset = train_subset.dropna(axis=0, how="any").reset_index(drop=True)

#%%
train_subset = FeatureEncoders.encode_string_columns(train_subset)
# %%
train_df, test_df = train_test_split(
    train_subset, random_state=SEED, stratify=train_subset["isFraud"]
)
# %%
is_fraud_plot = Statistics.isFraud_percentage_plot(train=train_df, test=test_df)
is_fraud_plot

# %%
d_train = XGBUtils.create_d_matrix(train_df, "isFraud")
d_test = XGBUtils.create_d_matrix(test_df, "isFraud")

# %%
params = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
params["nthread"] = 8
params["eval_metric"] = ["auc", "error", "logloss", "aucpr"] + [
    f"error@{x:0.1f}" for x in np.arange(0.1, 0.9, 0.1)
]
params["seed"] = SEED

# %%
eval_list = [(d_train, "train"), (d_test, "eval")]

#%%
num_round = 10

#%%
with NeptuneUtils.run() as run:
    run["model_columns"] = model_columns
    run["params"] = params
    run["num_round"] = 10

    xgb_callback = NeptuneCallback(run, log_tree=[0, 1, 2, 3, 4])
    bst = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=num_round,
        callbacks=[xgb_callback],
        evals=eval_list,
    )
    train_probas = bst.predict(d_train)
    train_preds = np.where(bst.predict(d_train) > 0.5, 1, 0)
    evaluate_and_launch_to_neptune(
        run, train_df["isFraud"], train_preds, train_probas, "train"
    )

    test_probas = bst.predict(d_test)
    test_preds = np.where(bst.predict(d_test) > 0.5, 1, 0)
    evaluate_and_launch_to_neptune(
        run, test_df["isFraud"], test_preds, test_probas, "test"
    )

# %%
train_metrics = scalar_evals(
    train_df["isFraud"], np.where(bst.predict(d_train) > 0.5, 1, 0)
)
# %%
eval_metrics = scalar_evals(
    test_df["isFraud"], np.where(bst.predict(d_test) > 0.5, 1, 0)
)
eval_metrics
# %%
