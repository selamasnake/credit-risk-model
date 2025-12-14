import os
import pandas as pd
from scipy.stats import entropy


class DataLoader:
    """Load and save CSV datasets."""

    def __init__(self, path: str | None = None):
        """Initialize with optional default file path."""
        self.path = path
        self.df = None

    def load_data(self, path: str | None = None) -> pd.DataFrame:
        """Load CSV into DataFrame. Returns the DataFrame."""
        file_path = path or self.path
        if not file_path:
            raise ValueError("No file path specified.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.df = pd.read_csv(file_path, low_memory=False)
        return self.df

    def save_data(self, output_path: str) -> None:
        """Save current DataFrame to CSV."""
        if self.df is None:
            raise ValueError("No data loaded.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)


class EDAProcessor:
    """Compute EDA metrics for numerical and categorical features."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data

    def data_overview(self) -> dict:
        """Returns dataset information and dtypes."""

        return self.data.info()

    def missing_values(self) -> pd.DataFrame:
        """Checks for missing counts and percentages per column."""

        missing = self.data.isna().sum()
        return pd.DataFrame({
            "missing_count": missing,
            "missing_pct": missing / len(self.data)
        })

    def summary_statistics(self, numerical_cols: list[str]) -> pd.DataFrame:
        """Calculates summary stats + skewness/kurtosis for numerical features."""

        desc = self.data[numerical_cols].describe().T
        desc["skew"] = self.data[numerical_cols].skew()
        desc["kurtosis"] = self.data[numerical_cols].kurtosis()
        return desc


    def outlier_metrics(self, numerical_cols: list[str]) -> pd.DataFrame:
        """Calculate IQR-based outlier bounds and counts."""

        results = {}
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.data[col] < lower) | (self.data[col] > upper)).sum()
            results[col] = {
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": outliers,
                "outlier_pct": outliers / len(self.data)
            }
        return pd.DataFrame(results).T

    def categorical_metrics(self, categorical_cols: list[str]) -> pd.DataFrame:
        """Returns number of categories, top category share, and entropy."""

        summaries = {}
        for col in categorical_cols:
            counts = self.data[col].value_counts()
            probs = counts / counts.sum()
            summaries[col] = {
                "num_categories": counts.shape[0],
                "top_category_pct": probs.iloc[0],
                "entropy": entropy(probs)
            }
        return pd.DataFrame(summaries).T

    def correlation_matrix(self, numerical_cols: list[str], method: str = "pearson") -> pd.DataFrame:
        """Returns correlation matrix for numerical features."""

        return self.data[numerical_cols].corr(method=method)
