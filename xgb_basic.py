#%%
from xgboost.training import train
from src.data import DataSet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.constants import SEED
from src.data import Statistics
from src.evaluation import scalar_evals, curve_evals
from src.utils import XGBUtils
import xgboost as xgb

# %%
data_set = DataSet("data")

# %%
train_data = data_set.get_train_transaction()

# %%
Statistics.isFraud_percentage_plot(train=train_data)
# %%
model_columns = ["TransactionAmt", "ProductCD", "card1", "card4", "isFraud"]
train_subset = train_data[model_columns]
# %%
train_subset["ProductCD"] = LabelEncoder().fit_transform(train_subset["ProductCD"])

#%%
train_subset["card4"] = LabelEncoder().fit_transform(train_subset["card4"])
# %%
train_df, test_df = train_test_split(
    train_subset[model_columns], random_state=SEED, stratify=train_data["isFraud"]
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
bst = xgb.train(params=params, dtrain=d_train, num_boost_round=num_round)
# %%
train_metrics = scalar_evals(
    train_df["isFraud"], np.where(bst.predict(d_train) > 0.5, 1, 0)
)
train_metrics
# %%
eval_metrics = scalar_evals(
    test_df["isFraud"], np.where(bst.predict(d_test) > 0.5, 1, 0)
)
eval_metrics
# %%
