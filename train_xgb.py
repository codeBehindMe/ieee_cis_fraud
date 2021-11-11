#%%
from src.constants import SEED
from src.models.xgb_trainer import XGBTrainer
import pandas as pd
from src.data import DataSet, FeatureEncoders, FeatureFilters
from sklearn.model_selection import train_test_split

#%%
data_set = DataSet("data")
# %%
train_data = data_set.get_train_transaction()
# %%
train_subset = FeatureFilters.remove_cols_with_nulls(train_data, 0.3)

# %%
train_subset = train_subset.dropna(axis=0, how="any").reset_index(drop=True)

#%%
train_subset = FeatureEncoders.encode_string_columns(train_subset)
# %%
train_df, test_df = train_test_split(
    train_subset, random_state=SEED, stratify=train_subset["isFraud"]
)
# %%
xgb_trainer = XGBTrainer()
xgb_trainer.train_evaluate(train_df, test_df)