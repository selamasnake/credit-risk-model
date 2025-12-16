import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pytest
import pandas as pd
import numpy as np
from data_processing import FeatureEngineer, WoETransformer

# ----------------------
# Fixtures
# ----------------------
@pytest.fixture
def transaction_df():
    return pd.DataFrame({
        "TransactionId": [1,2,3,4,5,6],
        "CustomerId": [101,101,102,102,102,103],
        "Amount": [100, 150, 200, 300, 250, 400],
        "TransactionStartTime": [
            "2025-12-01 10:00:00",
            "2025-12-01 12:00:00",
            "2025-12-02 09:30:00",
            "2025-12-02 15:45:00",
            "2025-12-03 11:20:00",
            "2025-12-03 08:00:00"
        ],
        "CurrencyCode": ["USD","USD","EUR","EUR","EUR","GBP"],
        "CountryCode": ["US","US","DE","DE","DE","UK"],
        "ProductCategory": ["A","B","A","B","C","A"],
        "ChannelId": ["Online","Offline","Online","Offline","Online","Online"],
        "PricingStrategy": ["Standard","Premium","Standard","Premium","Standard","Standard"]
    })

@pytest.fixture
def binary_target_df():
    return pd.DataFrame({
        "feature1": [10, 20, 30, 40],
        "feature2": [5, 3, 2, 7],
        "is_high_risk": [0, 1, 0, 1]
    })

# ----------------------
# FeatureEngineer Tests
# ----------------------
def test_extract_datetime_features(transaction_df):
    fe = FeatureEngineer()
    df = fe.extract_datetime_features(transaction_df)
    assert all(col in df.columns for col in ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"])
    assert df["transaction_hour"].iloc[0] == 10
    assert df["transaction_day"].iloc[0] == 1

def test_aggregate_customer_features(transaction_df):
    fe = FeatureEngineer()
    agg_df = fe.aggregate_customer_features(transaction_df)
    # Check shape: 3 unique customers
    assert agg_df.shape[0] == 3
    # Check columns
    for col in ["total_transaction_amount", "avg_transaction_amount", "transaction_count", "std_transaction_amount"]:
        assert col in agg_df.columns
    # Check that single-transaction customer has std=0
    customer_103_std = agg_df.loc[agg_df["CustomerId"]==103, "std_transaction_amount"].iloc[0]
    assert customer_103_std == 0

def test_build_preprocessing_pipeline(transaction_df):
    fe = FeatureEngineer()
    preprocessor = fe.build_preprocessing_pipeline()
    X = preprocessor.fit_transform(transaction_df.assign(
        total_transaction_amount=[100,100,200,200,200,400],
        avg_transaction_amount=[100,100,250,250,250,400],
        transaction_count=[2,2,3,3,3,1],
        std_transaction_amount=[35.36,35.36,50,50,50,0],
        transaction_hour=[10,12,9,15,11,8],
        transaction_day=[1,1,2,2,3,3],
        transaction_month=[12,12,12,12,12,12],
        transaction_year=[2025]*6
    ))
    assert X.shape[0] == transaction_df.shape[0]

# ----------------------
# WoETransformer Tests
# ----------------------

# def test_woe_transformer_fit_transform(binary_target_df):
#     woe = WoETransformer(target_col="is_high_risk", features=["feature1", "feature2"])
#     transformed = woe.fit_transform(binary_target_df)
#     assert isinstance(transformed, pd.DataFrame)
#     assert transformed.shape[0] == binary_target_df.shape[0]

# def test_woe_transformer_transform(binary_target_df):
#     woe = WoETransformer(target_col="is_high_risk", features=["feature1", "feature2"])
#     _ = woe.fit_transform(binary_target_df)
#     transformed2 = woe.transform(binary_target_df)
#     assert isinstance(transformed2, pd.DataFrame)
#     assert transformed2.shape[0] == binary_target_df.shape[0]

# def test_woe_transformer_raises_on_nonbinary():
#     df = pd.DataFrame({"x":[1,2,3]})
#     woe = WoETransformer(target_col="x", features=["x"])
#     with pytest.raises(ValueError):
#         woe.fit_transform(df)