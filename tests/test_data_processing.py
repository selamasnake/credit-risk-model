import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pytest
import pandas as pd
import numpy as np
import os
from tempfile import NamedTemporaryFile
from data_processing import DataLoader, EDAProcessor

# ----------------------
# Fixtures
# ----------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Amount": [100, 200, 300, 400, 10000],
        "Value": [10, 20, 30, 40, 500],
        "Category": ["A", "A", "B", "B", "B"]
    })

# ----------------------
# DataLoader Tests
# ----------------------
def test_load_and_save_data(sample_df):
    with NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        sample_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    # Load data
    loader = DataLoader()
    df_loaded = loader.load_data(tmp_path)
    pd.testing.assert_frame_equal(df_loaded, sample_df)

    # Save data
    loader.df = sample_df
    output_path = tmp_path + "_copy.csv"
    loader.save_data(output_path)
    df_saved = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(df_saved, sample_df)

    os.remove(tmp_path)
    os.remove(output_path)

def test_load_data_file_not_found():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_data("non_existent_file.csv")

def test_load_data_no_path():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.load_data()

# ----------------------
# EDAProcessor Tests
# ----------------------
def test_missing_values(sample_df):
    eda = EDAProcessor(sample_df)
    missing = eda.missing_values()
    assert all(missing["missing_count"] == 0)
    assert all(missing["missing_pct"] == 0)

def test_summary_statistics(sample_df):
    eda = EDAProcessor(sample_df)
    stats = eda.summary_statistics(["Amount", "Value"])
    assert "mean" in stats.columns
    assert stats.loc["Amount", "mean"] == sample_df["Amount"].mean()
    assert stats.loc["Value", "skew"] == pytest.approx(sample_df["Value"].skew())

def test_outlier_metrics(sample_df):
    eda = EDAProcessor(sample_df)
    outliers = eda.outlier_metrics(["Amount"])
    assert "outlier_count" in outliers.columns
    assert outliers.loc["Amount", "outlier_count"] == 1  # 10000 is an outlier

def test_categorical_metrics(sample_df):
    eda = EDAProcessor(sample_df)
    metrics = eda.categorical_metrics(["Category"])
    assert metrics.loc["Category", "num_categories"] == 2
    assert 0 <= metrics.loc["Category", "entropy"] <= np.log2(2) + 0.01  # entropy is non-negative

def test_correlation_matrix(sample_df):
    eda = EDAProcessor(sample_df)
    corr = eda.correlation_matrix(["Amount", "Value"])
    assert corr.shape == (2, 2)
    assert corr.loc["Amount", "Amount"] == 1
